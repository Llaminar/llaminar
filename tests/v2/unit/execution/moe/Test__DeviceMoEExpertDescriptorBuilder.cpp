/**
 * @file Test__DeviceMoEExpertDescriptorBuilder.cpp
 * @brief Device-free regressions for typed routed-expert descriptor export.
 *
 * These tests lock down the graph-construction boundary used by CPU, CUDA,
 * and ROCm model loading. They intentionally use inert GEMM engines: only the
 * immutable prepared-weight descriptor contract is under test, so no device,
 * allocation, model file, or execution fallback is involved.
 */

#include <gtest/gtest.h>

#include "execution/moe/DeviceMoEExpertDescriptorBuilder.h"
#include "execution/moe/DeviceMoETransferSlotDirectory.h"
#include "execution/moe/MoEOverlayFixedDownProjectionBank.h"
#include "loaders/ExpertGemmRegistry.h"
#include "../../../utils/QuantizedVerifierFormats.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace llaminar2::test
{
    namespace
    {
        /** @brief Minimal prepared GEMM exposing exactly one descriptor family. */
        class DescriptorGemm final : public ITensorGemm
        {
        public:
            /** @brief Construct a contiguous floating-point descriptor source. */
            explicit DescriptorGemm(
                ContiguousFloatingPointWeightDescriptor floating) noexcept
                : floating_(floating), exports_floating_(true)
            {
            }

            /** @brief Construct a NativeVNNI descriptor source. */
            explicit DescriptorGemm(DeviceNativeVNNIMatrixDesc native) noexcept
                : native_(native), source_{native.source_codebook_id,
                    native.source_is_superblock != 0, native.source_identity_present != 0}, exports_native_(true)
            {
            }

            /** @brief Model engines may publish source provenance independently of their raw view. */
            DescriptorGemm(DeviceNativeVNNIMatrixDesc native, NativeVnniSourceIdentity source) noexcept
                : native_(native), source_(source), exports_native_(true)
            {
            }

            /** @return True because this inert fixture is backend-independent. */
            bool supports_device(int) const override { return true; }

            /** @return False; arithmetic is outside this descriptor unit test. */
            bool multiply_tensor(
                const TensorBase *,
                TensorBase *,
                int,
                int,
                int,
                bool,
                float,
                float,
                const TensorBase *,
                const IMPIContext *,
                int,
                DeviceWorkspaceManager *,
                int) override
            {
                return false;
            }

            /** @brief Export the fixture's NativeVNNI view when configured. */
            bool exportNativeVNNIMatrixDesc(
                DeviceNativeVNNIMatrixDesc &output) override
            {
                output = exports_native_
                             ? native_
                             : DeviceNativeVNNIMatrixDesc{};
                return exports_native_;
            }

            /** @brief Export the fixture's floating view when configured. */
            bool exportContiguousFloatingPointWeights(
                ContiguousFloatingPointWeightDescriptor &output) const override
            {
                output = exports_floating_
                             ? floating_
                             : ContiguousFloatingPointWeightDescriptor{};
                return exports_floating_;
            }

            /** @return The exact separately prepared source identity, when available. */
            bool exportNativeVNNISourceIdentity(NativeVnniSourceIdentity &output) const override
            {
                output = source_;
                return source_.present;
            }

        private:
            DeviceNativeVNNIMatrixDesc native_{};
            NativeVnniSourceIdentity source_{};
            ContiguousFloatingPointWeightDescriptor floating_{};
            bool exports_native_ = false;
            bool exports_floating_ = false;
        };

        /** @return Exact contiguous descriptor for one row-major fixture. */
        ContiguousFloatingPointWeightDescriptor floatingDescriptor(
            const void *data,
            TensorType type,
            int n,
            int k)
        {
            const std::size_t element_bytes =
                type == TensorType::FP32 ? sizeof(float) : sizeof(std::uint16_t);
            return {
                .data = data,
                .type = type,
                .n = n,
                .k = k,
                .bytes = static_cast<std::size_t>(n) *
                         static_cast<std::size_t>(k) * element_bytes,
            };
        }

        /** @return Minimally valid NativeVNNI fixture descriptor. */
        DeviceNativeVNNIMatrixDesc nativeDescriptor(
            const std::uint8_t *payload,
            const std::uint16_t *scales,
            int n,
            int k)
        {
            return {
                .payload = payload,
                .scales = scales,
                .n = n,
                .k = k,
                .blocks_per_row = static_cast<std::uint32_t>(k / 32),
                .codebook_id = 2,
            };
        }

        /** @brief Retain exact registry identities while the caller tests descriptor validity. */
        std::shared_ptr<const MoEOverlayFixedDownProjectionBank> fixedDownBank(
            DeviceId device, const MoEExpertProjectionOwnership &ownership,
            const std::vector<std::shared_ptr<ITensorGemm>> &engines)
        {
            ExpertGemmRegistry registry;
            const MoEExpertOwnerParticipant participant{
                .participant_id = 7, .domain_name = "compute", .domain_participant_index = ownership.participant(),
                .device = device, .world_rank = 2, .world_rank_known = true};
            for (size_t expert = 0; expert < engines.size(); ++expert)
                registry.registerEngineForParticipant("compute", device, 2, ownership.participant(), 0,
                    static_cast<int>(expert), ExpertGemmRegistry::WeightRole::DOWN,
                    engines[expert].get(), engines[expert], ownership);
            return MoEOverlayFixedDownProjectionBank::resolve(registry, participant, 0, ownership);
        }
    } // namespace

    /** @test Routing and publication agree on the exact declared movable family. */
    TEST(DeviceMoEExpertDescriptorBuilder, ProjectionReadinessRequiresExactPayload)
    {
        std::uint8_t payload{};
        std::uint16_t scale{};
        const auto check = [](DeviceMoEExpertDescriptor complete)
        {
            const auto whole = DeviceMoEProjectionSet::CompleteExpert;
            const auto pair = DeviceMoEProjectionSet::GateUp;
            EXPECT_TRUE(deviceMoEExpertProjectionReady(complete, whole));
            EXPECT_FALSE(deviceMoEExpertProjectionReady(complete, pair));
            auto split = complete;
            split.projection_set = pair;
            // A fixed down pointer may never leak into a movable pair.
            EXPECT_FALSE(deviceMoEExpertProjectionReady(split, pair));
            split.down = {};
            split.floating_down = {};
            EXPECT_TRUE(deviceMoEExpertProjectionReady(split, pair));
            EXPECT_FALSE(deviceMoEExpertProjectionReady(split, whole));
            auto invalid = split;
            invalid.up = {};
            invalid.floating_up = {};
            EXPECT_FALSE(deviceMoEExpertProjectionReady(invalid, pair));
            invalid = split;
            invalid.logical_expert_id = -1;
            EXPECT_FALSE(deviceMoEExpertProjectionReady(invalid, pair));
            invalid = split;
            invalid.projection_set = static_cast<DeviceMoEProjectionSet>(99);
            EXPECT_FALSE(deviceMoEExpertProjectionReady(invalid, invalid.projection_set));
            invalid = split;
            invalid.weight_format = static_cast<DeviceMoEWeightFormat>(99);
            EXPECT_FALSE(deviceMoEExpertProjectionReady(invalid, pair));
            complete.down = {};
            complete.floating_down = {};
            EXPECT_FALSE(deviceMoEExpertProjectionReady(complete, whole));
        };
        for (const auto &format : quantizedVerifierFormats())
        {
            SCOPED_TRACE(format.label);
            DeviceMoEExpertDescriptor expert;
            expert.logical_expert_id = 0;
            expert.gate = expert.up = expert.down = nativeDescriptor(&payload, &scale, 64, 64);
            expert.gate.source_codebook_id = expert.up.source_codebook_id =
                expert.down.source_codebook_id = format.source_codebook_id;
            check(expert);
        }
        for (const auto format : {DeviceMoEWeightFormat::FP16,
                 DeviceMoEWeightFormat::BF16, DeviceMoEWeightFormat::FP32})
        {
            DeviceMoEExpertDescriptor expert;
            expert.logical_expert_id = 0;
            expert.weight_format = format;
            expert.floating_gate = expert.floating_up = expert.floating_down =
                DeviceMoEFloatingMatrixDesc{.data = &payload, .n = 64, .k = 64};
            check(expert);
        }
    }

    TEST(DeviceMoEExpertDescriptorBuilder, ExportsEveryFloatingPrecision)
    {
        constexpr int d_model = 32;
        constexpr int intermediate = 64;
        std::array<std::uint16_t, intermediate * d_model> gate16{};
        std::array<std::uint16_t, intermediate * d_model> up16{};
        std::array<std::uint16_t, d_model * intermediate> down16{};
        std::array<float, intermediate * d_model> gate32{};
        std::array<float, intermediate * d_model> up32{};
        std::array<float, d_model * intermediate> down32{};

        struct Case
        {
            TensorType tensor_type;
            DeviceMoEWeightFormat expected_format;
            const void *gate;
            const void *up;
            const void *down;
        };
        const std::array cases{
            Case{TensorType::FP16,
                 DeviceMoEWeightFormat::FP16,
                 gate16.data(), up16.data(), down16.data()},
            Case{TensorType::BF16,
                 DeviceMoEWeightFormat::BF16,
                 gate16.data(), up16.data(), down16.data()},
            Case{TensorType::FP32,
                 DeviceMoEWeightFormat::FP32,
                 gate32.data(), up32.data(), down32.data()},
        };

        for (const Case &test_case : cases)
        {
            DescriptorGemm gate(floatingDescriptor(
                test_case.gate,
                test_case.tensor_type,
                intermediate,
                d_model));
            DescriptorGemm up(floatingDescriptor(
                test_case.up,
                test_case.tensor_type,
                intermediate,
                d_model));
            DescriptorGemm down(floatingDescriptor(
                test_case.down,
                test_case.tensor_type,
                d_model,
                intermediate));
            DeviceMoEExpertDescriptor descriptor{
                .logical_expert_id = 7,
                .owner_participant = 2,
                .local_slot = 5,
                .flags = toMoEExpertFlags(DeviceMoEExpertFlags::Valid),
            };

            ASSERT_TRUE(exportDeviceMoEExpertWeightDescriptors(
                &gate,
                &up,
                &down,
                d_model,
                intermediate,
                descriptor));
            EXPECT_EQ(descriptor.weight_format, test_case.expected_format);
            EXPECT_EQ(descriptor.floating_gate.data, test_case.gate);
            EXPECT_EQ(descriptor.floating_up.data, test_case.up);
            EXPECT_EQ(descriptor.floating_down.data, test_case.down);
            EXPECT_TRUE(descriptor.weightsReady());
            EXPECT_FALSE(descriptor.gate.valid());
            EXPECT_EQ(descriptor.logical_expert_id, 7);
            EXPECT_EQ(descriptor.owner_participant, 2);
            EXPECT_EQ(descriptor.local_slot, 5);
        }
    }

    TEST(DeviceMoEExpertDescriptorBuilder, ExportsCompleteNativeFamily)
    {
        constexpr int d_model = 32;
        constexpr int intermediate = 64;
        std::array<std::uint8_t, 32> payload{};
        std::array<std::uint16_t, 8> scales{};
        DescriptorGemm gate(nativeDescriptor(
            payload.data(), scales.data(), intermediate, d_model));
        DescriptorGemm up(nativeDescriptor(
            payload.data(), scales.data(), intermediate, d_model));
        DescriptorGemm down(nativeDescriptor(
            payload.data(), scales.data(), d_model, intermediate));
        DeviceMoEExpertDescriptor descriptor{};

        ASSERT_TRUE(exportDeviceMoEExpertWeightDescriptors(
            &gate,
            &up,
            &down,
            d_model,
            intermediate,
            descriptor));
        EXPECT_EQ(
            descriptor.weight_format,
            DeviceMoEWeightFormat::NativeVNNI);
        EXPECT_TRUE(descriptor.gate.valid());
        EXPECT_TRUE(descriptor.up.valid());
        EXPECT_TRUE(descriptor.down.valid());
        EXPECT_FALSE(descriptor.floating_gate.valid());
        EXPECT_TRUE(descriptor.weightsReady());
    }

    TEST(DeviceMoEExpertDescriptorBuilder, RejectsMixedPrecisionAndGeometry)
    {
        constexpr int d_model = 32;
        constexpr int intermediate = 64;
        std::array<std::uint16_t, intermediate * d_model> gate_data{};
        std::array<std::uint16_t, intermediate * d_model> up_data{};
        std::array<std::uint16_t, d_model * intermediate> down_data{};
        DescriptorGemm gate(floatingDescriptor(
            gate_data.data(), TensorType::FP16, intermediate, d_model));
        DescriptorGemm up(floatingDescriptor(
            up_data.data(), TensorType::BF16, intermediate, d_model));
        DescriptorGemm down(floatingDescriptor(
            down_data.data(), TensorType::FP16, d_model, intermediate));
        DeviceMoEExpertDescriptor descriptor{};

        EXPECT_FALSE(exportDeviceMoEExpertWeightDescriptors(
            &gate,
            &up,
            &down,
            d_model,
            intermediate,
            descriptor));

        DescriptorGemm uniform_up(floatingDescriptor(
            up_data.data(), TensorType::FP16, intermediate, d_model));
        DescriptorGemm wrong_down(floatingDescriptor(
            down_data.data(), TensorType::FP16, intermediate, d_model));
        EXPECT_FALSE(exportDeviceMoEExpertWeightDescriptors(
            &gate,
            &uniform_up,
            &wrong_down,
            d_model,
            intermediate,
            descriptor));
    }

    TEST(DeviceMoEExpertDescriptorBuilder, RejectsInvalidByteExtent)
    {
        constexpr int d_model = 32;
        constexpr int intermediate = 64;
        std::array<std::uint16_t, intermediate * d_model> gate_data{};
        std::array<std::uint16_t, intermediate * d_model> up_data{};
        std::array<std::uint16_t, d_model * intermediate> down_data{};
        auto invalid_gate = floatingDescriptor(
            gate_data.data(), TensorType::BF16, intermediate, d_model);
        --invalid_gate.bytes;
        DescriptorGemm gate(invalid_gate);
        DescriptorGemm up(floatingDescriptor(
            up_data.data(), TensorType::BF16, intermediate, d_model));
        DescriptorGemm down(floatingDescriptor(
            down_data.data(), TensorType::BF16, d_model, intermediate));
        DeviceMoEExpertDescriptor descriptor{};

        EXPECT_FALSE(exportDeviceMoEExpertWeightDescriptors(
            &gate,
            &up,
            &down,
            d_model,
            intermediate,
            descriptor));
    }
    /** @brief A movable pair must never masquerade as a complete expert. */
    TEST(DeviceMoEExpertDescriptorBuilder, GateUpExportRequiresExplicitOwnershipAndPreservesPlacement)
    {
        std::array<std::uint8_t, 32> payload{};
        std::array<std::uint16_t, 8> scales{};
        DescriptorGemm gate(nativeDescriptor(payload.data(), scales.data(), 64, 256));
        DescriptorGemm up(nativeDescriptor(payload.data(), scales.data(), 64, 256));
        const auto ownership = MoEExpertProjectionOwnership::gateUpOwnedDownColumns({8, 256, 64}, 1, 2);
        DeviceMoEExpertDescriptor descriptor{.logical_expert_id = 3, .owner_participant = 1, .local_slot = 7};
        ASSERT_TRUE(exportDeviceMoEGateUpWeightDescriptors(&gate, &up, ownership, descriptor));
        EXPECT_EQ(sizeof(descriptor), 240u);
        EXPECT_EQ(descriptor.projection_set, DeviceMoEProjectionSet::GateUp);
        EXPECT_TRUE(descriptor.movableWeightsReady());
        EXPECT_FALSE(descriptor.weightsReady());
        EXPECT_EQ(descriptor.down.payload, nullptr);
        EXPECT_EQ(descriptor.logical_expert_id, 3);
        EXPECT_EQ(descriptor.local_slot, 7);
        const auto before = descriptor;
        EXPECT_FALSE(exportDeviceMoEExpertWeightDescriptors(&gate, &up, nullptr, 256, 64, descriptor));
        EXPECT_FALSE(exportDeviceMoEGateUpWeightDescriptors(&gate, &up,
            MoEExpertProjectionOwnership::completeExperts({8, 256, 64}, 1, 2), descriptor));
        DescriptorGemm wrong_up(nativeDescriptor(payload.data(), scales.data(), 64, 128));
        EXPECT_FALSE(exportDeviceMoEGateUpWeightDescriptors(&gate, &wrong_up, ownership, descriptor));
        EXPECT_EQ(std::memcmp(&before, &descriptor, sizeof(descriptor)), 0);
        descriptor.down = nativeDescriptor(payload.data(), scales.data(), 128, 64);
        EXPECT_FALSE(descriptor.movableWeightsReady()) << "Fixed down storage cannot enter a movable bank";
        descriptor = before;
        descriptor.projection_set = static_cast<DeviceMoEProjectionSet>(2);
        EXPECT_FALSE(descriptor.movableWeightsReady());
    }

    /** @brief All floating formats share pair publication and exact transfer admission. */
    TEST(DeviceMoEExpertDescriptorBuilder, FloatingGateUpContractRejectsWholeExpertSlots)
    {
        std::array<float, 64 * 256> storage{};
        const auto ownership = MoEExpertProjectionOwnership::gateUpOwnedDownColumns({8, 256, 64}, 0, 2);
        for (const auto type : {TensorType::FP16, TensorType::BF16, TensorType::FP32})
        {
            DescriptorGemm gate(floatingDescriptor(storage.data(), type, 64, 256));
            DescriptorGemm up(floatingDescriptor(storage.data(), type, 64, 256));
            DeviceMoEExpertDirectoryEntry source;
            ASSERT_TRUE(exportDeviceMoEGateUpWeightDescriptors(&gate, &up, ownership, source.descriptor));
            EXPECT_TRUE(source.descriptor.movableWeightsReady());
            EXPECT_FALSE(source.descriptor.weightsReady());
            auto destination = source;
            destination.descriptor.floating_allocation_format = DeviceMoEWeightFormat::FP32;
            EXPECT_TRUE(deviceMoEDirectoryCopyReady(source));
            EXPECT_TRUE(deviceMoEDirectoryFitsTransferCapacity(source, destination));
            destination.descriptor.projection_set = DeviceMoEProjectionSet::CompleteExpert;
            destination.descriptor.floating_down = {storage.data(), 256, 64};
            EXPECT_FALSE(deviceMoEDirectoryFitsTransferCapacity(source, destination));
            EXPECT_FALSE(deviceMoEDirectoryFitsTransferCapacity(destination, source));
            destination = source;
            destination.descriptor.floating_down = {storage.data(), 128, 64};
            EXPECT_FALSE(deviceMoEDirectoryCopyReady(destination));
        }
    }

    /** @brief Arrival export authenticates the same payload contract for every packed/floating format. */
    TEST(DeviceMoEExpertDescriptorBuilder, PreparedPayloadExportPreservesExactFamily)
    {
        using Payload = MoEOverlayPreparedExpertPayload;
        std::array<float, 64 * 256> data{};
        std::array<std::uint16_t, 512> scales{};
        const auto verify = [](const Payload &whole) {
            DeviceMoEExpertDescriptor full{};
            ASSERT_TRUE(exportDeviceMoEPreparedPayload(whole, full));
            EXPECT_TRUE(full.weightsReady());
            DeviceMoEExpertDescriptor pair{.logical_expert_id = 7, .local_slot = 3};
            ASSERT_TRUE(exportDeviceMoEPreparedPayload(Payload::gateUp(whole.gate(), whole.up()), pair));
            EXPECT_TRUE(pair.movableWeightsReady());
            EXPECT_FALSE(pair.weightsReady());
            EXPECT_EQ(pair.logical_expert_id, 7);
            EXPECT_EQ(pair.local_slot, 3);
            EXPECT_EQ(pair.projection_set, DeviceMoEProjectionSet::GateUp);
            EXPECT_EQ(pair.down.payload, nullptr);
            EXPECT_EQ(pair.floating_down.data, nullptr);
            const auto retained = pair;
            EXPECT_FALSE(exportDeviceMoEPreparedPayload({}, pair));
            EXPECT_EQ(std::memcmp(&retained, &pair, sizeof(pair)), 0);
            // Lifetime readiness is necessary but not a substitute for geometry.
            EXPECT_FALSE(exportDeviceMoEPreparedPayload(Payload::gateUp(whole.gate(), whole.down()), pair));
            EXPECT_EQ(std::memcmp(&retained, &pair, sizeof(pair)), 0);
        };
        for (const auto &format : quantizedMoEVerifierFormats())
        {
            SCOPED_TRACE(format.label);
            auto gate = nativeDescriptor(reinterpret_cast<const uint8_t *>(data.data()), scales.data(), 64, 256);
            auto down = nativeDescriptor(reinterpret_cast<const uint8_t *>(data.data()), scales.data(), 256, 64);
            gate.codebook_id = down.codebook_id = format.device_execution_codebook_id;
            ASSERT_NO_FATAL_FAILURE(verify(Payload{std::make_shared<DescriptorGemm>(gate),
                std::make_shared<DescriptorGemm>(gate), std::make_shared<DescriptorGemm>(down)}));
        }
        for (const auto type : {TensorType::FP16, TensorType::BF16, TensorType::FP32})
        {
            SCOPED_TRACE(static_cast<int>(type));
            const auto gate = floatingDescriptor(data.data(), type, 64, 256);
            const auto down = floatingDescriptor(data.data(), type, 256, 64);
            ASSERT_NO_FATAL_FAILURE(verify(Payload{std::make_shared<DescriptorGemm>(gate),
                std::make_shared<DescriptorGemm>(gate), std::make_shared<DescriptorGemm>(down)}));
        }
    }

    /** @brief Every quantized format retains complete K and original arithmetic under N slicing. */
    TEST(DeviceMoEExpertDescriptorBuilder, FixedDownTablePreservesAllNativeSourceIdentities)
    {
        using Bank = MoEOverlayFixedDownProjectionBank;
        std::array<uint8_t, 32> data{};
        std::array<uint16_t, 32> scales{};
        for (const auto device : {DeviceId::cuda(3), DeviceId::rocm(3)})
        for (const int degree : {1, 2, 3, 4, 8})
        for (const auto &format : quantizedVerifierFormats())
        {
            SCOPED_TRACE(::testing::Message() << device.to_string() << '/' << degree << '/' << format.label);
            const auto ownership = MoEExpertProjectionOwnership::gateUpOwnedDownColumns({2, 768, 256}, degree - 1, degree);
            auto native = nativeDescriptor(data.data(), scales.data(), 768 / degree, 256);
            native.codebook_id = format.device_execution_codebook_id;
            const NativeVnniSourceIdentity source{format.source_codebook_id, format.source_is_superblock, true};
            auto first = std::make_shared<DescriptorGemm>(native, source);
            auto second = std::make_shared<DescriptorGemm>(native, source);
            auto bank = fixedDownBank(device, ownership, {first, second});
            const auto table = bank->exportDescriptorTable();
            const auto *entries = std::get_if<Bank::NativeDescriptorTable>(&table);
            ASSERT_NE(entries, nullptr);
            ASSERT_EQ(entries->experts.size(), 2u);
            for (const auto &entry : entries->experts)
            {
                EXPECT_EQ(entry.payload, data.data());
                EXPECT_EQ(entry.scales, scales.data());
                EXPECT_EQ(entry.n, 768 / degree);
                EXPECT_EQ(entry.k, 256);
                EXPECT_EQ(entry.blocks_per_row, 8u);
                EXPECT_EQ(entry.codebook_id, format.device_execution_codebook_id);
                EXPECT_EQ(entry.source_codebook_id, source.codebook_id);
                EXPECT_EQ(entry.source_is_superblock, source.is_superblock);
                EXPECT_EQ(entry.source_identity_present, 1u);
            }
            // Enriching the exported value must not mutate the engine's view.
            DeviceNativeVNNIMatrixDesc retained{};
            ASSERT_TRUE(first->exportNativeVNNIMatrixDesc(retained));
            EXPECT_EQ(retained.source_identity_present, 0u);
        }
    }

    /** @brief Floating tables keep their exact precision and reject mixed/bad prepared views. */
    TEST(DeviceMoEExpertDescriptorBuilder, FixedDownTableValidatesFloatingPrecisionAndPhysicalGeometry)
    {
        using Bank = MoEOverlayFixedDownProjectionBank;
        std::array<float, 32> data{};
        const auto ownership = MoEExpertProjectionOwnership::gateUpOwnedDownColumns({2, 768, 256}, 2, 3);
        for (const auto device : {DeviceId::cuda(3), DeviceId::rocm(3)})
        for (const auto type : {TensorType::FP16, TensorType::BF16, TensorType::FP32})
        {
            const auto descriptor = floatingDescriptor(data.data(), type, 256, 256);
            auto engine = std::make_shared<DescriptorGemm>(descriptor);
            const auto table = fixedDownBank(device, ownership, {engine, engine})->exportDescriptorTable();
            const auto *entries = std::get_if<Bank::FloatingDescriptorTable>(&table);
            ASSERT_NE(entries, nullptr);
            ASSERT_EQ(entries->experts.size(), 2u);
            EXPECT_EQ(entries->format, type == TensorType::FP32 ? DeviceMoEWeightFormat::FP32 :
                type == TensorType::BF16 ? DeviceMoEWeightFormat::BF16 : DeviceMoEWeightFormat::FP16);
            for (const auto &entry : entries->experts)
            {
                EXPECT_EQ(entry.data, data.data());
                EXPECT_EQ(entry.n, 256);
                EXPECT_EQ(entry.k, 256);
            }
            for (int defect = 0; defect < 5; ++defect)
            {
                SCOPED_TRACE(::testing::Message() << static_cast<int>(type) << '/' << defect);
                auto wrong = descriptor;
                if (defect == 0) wrong.n = 768; // Full matrix is not this participant's slice.
                if (defect == 1) wrong.k = 128; // A K shard changes the serial arithmetic.
                if (defect == 2) wrong.bytes -= 1;
                if (defect == 3) wrong.data = nullptr;
                if (defect == 4) wrong = floatingDescriptor(data.data(),
                    type == TensorType::FP32 ? TensorType::FP16 : TensorType::FP32, 256, 256);
                auto bad = std::make_shared<DescriptorGemm>(wrong);
                EXPECT_THROW((void)fixedDownBank(device, ownership, {engine, bad})->exportDescriptorTable(), std::runtime_error);
            }
        }
    }

    /** @brief Registry tags cannot excuse a malformed native engine or conflicting provenance. */
    TEST(DeviceMoEExpertDescriptorBuilder, FixedDownTableRejectsInvalidNativeAndMixedFamilies)
    {
        std::array<uint8_t, 32> data{};
        std::array<uint16_t, 32> scales{};
        const auto ownership = MoEExpertProjectionOwnership::gateUpOwnedDownColumns({2, 768, 256}, 1, 2);
        auto native = nativeDescriptor(data.data(), scales.data(), 384, 256);
        native.codebook_id = 19;
        const NativeVnniSourceIdentity source{19, false, true};
        auto good = std::make_shared<DescriptorGemm>(native, source);
        for (const auto device : {DeviceId::cuda(3), DeviceId::rocm(3)})
        {
            for (int defect = 0; defect < 9; ++defect)
            {
                SCOPED_TRACE(defect);
                auto wrong = native;
                auto provenance = source;
                if (defect == 0) wrong.n *= 2;
                if (defect == 1) wrong.k /= 2;
                if (defect == 2) wrong.blocks_per_row -= 1;
                if (defect == 3) wrong.payload = nullptr;
                if (defect == 4) provenance.present = false;
                if (defect == 5) provenance.codebook_id = 255;
                if (defect == 6) { wrong.source_identity_present = 1; wrong.source_codebook_id = 0; }
                if (defect == 7) wrong.codebook_id = 0;
                if (defect == 8) provenance.is_superblock = true;
                auto bad = std::make_shared<DescriptorGemm>(wrong, provenance);
                EXPECT_THROW((void)fixedDownBank(device, ownership, {good, bad})->exportDescriptorTable(), std::runtime_error);
            }
            auto floating = std::make_shared<DescriptorGemm>(floatingDescriptor(data.data(), TensorType::FP16, 384, 256));
            EXPECT_THROW((void)fixedDownBank(device, ownership, {good, floating})->exportDescriptorTable(), std::runtime_error);
            EXPECT_THROW((void)fixedDownBank(device, ownership, {floating, good})->exportDescriptorTable(), std::runtime_error);
        }
    }

    /** @brief The canonical directory BOM and wire profile omit fixed down storage. */
    TEST(DeviceMoEExpertDescriptorBuilder, GateUpDirectoryProfilePricesOnlyMovableBytes)
    {
        using Directory = DeviceMoETransferSlotDirectory;
        const auto capacity = Directory::planBufferedCapacity(1, 1, 2);
        for (const auto type : {TensorType::FP16, TensorType::BF16, TensorType::FP32})
        {
            const auto format = ExpertWeightFormat::floating(type);
            std::vector<Directory::ProjectionSpec> pair{
                {.label = "gate", .N = 64, .K = 256, .format = format},
                {.label = "up", .N = 64, .K = 256, .format = format}};
            auto triple = pair;
            triple.push_back({.label = "down", .N = 256, .K = 64, .format = format});
            const auto projected = Directory::profileForLayerFormats({pair}, DeviceMoEProjectionSet::GateUp);
            const auto complete = Directory::profileForLayerFormats({triple});
            EXPECT_EQ(projected.max_wire_payload_bytes, 2u * 64u * 256u * format.floatingElementBytes());
            EXPECT_EQ(projected.max_wire_payload_bytes * 3u, complete.max_wire_payload_bytes * 2u);
            const auto pair_bom = Directory::allocationBOM(capacity, projected);
            const auto triple_bom = Directory::allocationBOM(capacity, complete);
            EXPECT_EQ(pair_bom.descriptor_bytes, triple_bom.descriptor_bytes);
            EXPECT_EQ(pair_bom.payload_bytes * 3u, triple_bom.payload_bytes * 2u);
            EXPECT_LT(pair_bom.total_bytes, triple_bom.total_bytes);
            EXPECT_THROW(Directory::profileForLayerFormats({pair}), std::invalid_argument);
            EXPECT_THROW(Directory::profileForLayerFormats({triple}, DeviceMoEProjectionSet::GateUp), std::invalid_argument);
            auto forged = projected;
            forged.projection_set = DeviceMoEProjectionSet::CompleteExpert;
            EXPECT_THROW(Directory::allocationBOM(capacity, forged), std::invalid_argument);
            pair[0].label = "up";
            EXPECT_THROW(Directory::profileForLayerFormats({pair}, DeviceMoEProjectionSet::GateUp), std::invalid_argument);
        }
    }
} // namespace llaminar2::test
