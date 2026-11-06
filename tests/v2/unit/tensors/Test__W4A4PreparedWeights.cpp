/**
 * @file Test__W4A4PreparedWeights.cpp
 * @brief Native W4A4 final-write admission and model-owned Store publication contracts.
 *
 * The real streamed writer fills one PMA-owned final allocation in original
 * source coordinates. It may publish only after complete source validation.
 * Rejected conversions, partial coverage and failed reads cannot produce a
 * prepared reference. These are lifecycle contracts, not numerical replay.
 */
#include "kernels/cpu/gemm/CPUPreparedWeightWriter.h"
#include "loaders/PreparedWeightStore.h"
#include "tensors/W4A4Tensor.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstring>

namespace llaminar2::w4a4
{
    namespace
    {
        constexpr std::size_t kRows = 3, kColumns = 512;
        constexpr auto kOwner = PhysicalMemoryOwner::PrimaryModelWeights;
        constexpr ModelContextId kModel{177};
        const DeviceId kCPU{DeviceType::CPU, 1};
        using Blocks = std::array<SourceBlock, kRows * kColumns / kBlockElements>;

        /** @return Original source rows selected from a larger named matrix. */
        CPUPreparedWeightSource specification(TensorType type, CPUWeightConsumer consumer)
        {
            return {.original = {{kColumns, kRows + 2}, type, kBlockElements, kSourceBlockBytes},
                .range = {.first_row = 1, .rows = kRows, .columns = kColumns,
                    .native_row_bytes = 2 * kSourceBlockBytes}, .consumer = consumer};
        }

        /** @return Complete valid bytes with distinguishable per-row units. */
        Blocks source(TensorType type)
        {
            Blocks blocks{};
            for (std::size_t i = 0; i < blocks.size(); ++i)
            {
                blocks[i].unit = float(i / 2 + 1) * 0.125f;
                blocks[i].descriptor = 0x60004321;
                blocks[i].rotation_length = type == TensorType::W4A4 ? kColumns : 128;
                std::fill_n(blocks[i].codes, 128, std::uint8_t(17 * i));
            }
            return blocks;
        }

        /** @return The sole physical-memory authority for one exact final native allocation. */
        std::shared_ptr<PhysicalMemoryAuthority> authority()
        {
            const auto bytes = CPUStorageRoot<uint8_t>(sizeof(Blocks), kOwner, kCPU).physicalBytes();
            PhysicalMemoryPlanBuilder builder;
            builder.add(PhysicalMemoryResource{0, kCPU, bytes, bytes}, kOwner, bytes);
            return std::make_shared<PhysicalMemoryAuthority>(
                std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(builder.build()), 0);
        }

        /** @return An unmaterialized binding in this actual Store/model/allocator domain. */
        WeightBinding binding()
        {
            WeightBinding value;
            value.binding_id = 1;
            value.identity = makeSourceWeightIdentity("blk.0.ffn_gate.weight", kModel, 1);
            value.residency.home_device = kCPU;
            value.residency.resident_device = kCPU;
            value.residency.raw_host_data_available = false;
            value.immutable = true;
            return value;
        }

        /** @return The real final-weight claim; no fixture-local live ledger exists. */
        std::size_t live(const std::shared_ptr<PhysicalMemoryAuthority> &memory)
        {
            return memory->claimedBytes(kCPU, kOwner, PhysicalMemoryMaterializationKind::NewAllocation);
        }
    }

    TEST(W4A4PreparedWeightLifecycle, StreamedOriginalCoordinatesPublishOneValidatedFinalOwner)
    {
        for (auto type : {TensorType::W4A4, TensorType::W4A4H})
            for (auto consumer : {CPUWeightConsumer::ResidentTensor, CPUWeightConsumer::PreparedGemm})
            {
                auto memory = authority();
                PreparedWeightStore store(kModel);
                store.installPhysicalMemoryAuthority(memory);
                const auto spec = specification(type, consumer);
                auto transaction = store.beginCPUPreparation(binding(), spec, kOwner, CPUStoragePlacement::local());
                EXPECT_FALSE(transaction->stagingRequest());
                const auto physical = live(memory);
                EXPECT_GT(physical, 0u);
                EXPECT_FALSE(store.preparedRefForBinding(1, kCPU));
                const auto blocks = source(type);
                for (std::size_t i = 0; i < blocks.size(); ++i)
                {
                    const NativeWeightSourceWindow window{
                        .first_row = spec.range.first_row + i / 2, .rows = 1,
                        .first_column = (i % 2) * kBlockElements, .columns = kBlockElements,
                        .native_row_bytes = kSourceBlockBytes};
                    transaction->readDirect(window, [&](std::span<std::byte> destination) {
                        ASSERT_EQ(destination.size(), sizeof(SourceBlock));
                        std::memcpy(destination.data(), &blocks[i], sizeof(SourceBlock));
                    });
                    EXPECT_EQ(live(memory), physical);
                    EXPECT_FALSE(store.preparedRefForBinding(1, kCPU));
                }
                auto committed = store.finishCPUPreparation(std::move(transaction));
                ASSERT_TRUE(committed.prepared);
                ASSERT_NE(committed.tensor, nullptr);
                EXPECT_EQ(committed.tensor->native_type(), type);
                EXPECT_EQ(committed.tensor->home_device(), kCPU);
                EXPECT_EQ(committed.tensor->shape(), (std::vector<std::size_t>{kRows, kColumns}));
                EXPECT_EQ(std::memcmp(committed.tensor->raw_data(), blocks.data(), sizeof(blocks)), 0);
                EXPECT_TRUE(committed.residency.raw_host_data_available);
                EXPECT_EQ(live(memory), physical);
                std::shared_ptr<ITensorGemm> engine;
                if (consumer == CPUWeightConsumer::PreparedGemm)
                {
                    engine = store.gemmKernelLifetime(*committed.prepared);
                    ASSERT_NE(engine, nullptr);
                    EXPECT_EQ(engine->packedWeightBytes(), sizeof(blocks));
                    EXPECT_TRUE(engine->canReleaseSourceWeightTensor());
                    EXPECT_TRUE(committed.tensor->hasPreparedDeviceState());
                }
                else
                    EXPECT_FALSE(committed.tensor->hasPreparedDeviceState());
                store.clear();
                committed = {};
                if (engine) EXPECT_EQ(live(memory), physical);
                engine.reset();
                EXPECT_EQ(live(memory), 0u);
            }
    }

    TEST(W4A4PreparedWeightContract, RejectedPoliciesAndIncompleteOrMalformedSourcesNeverPublish)
    {
        for (auto type : {TensorType::W4A4, TensorType::W4A4H})
        {
            auto memory = authority();
            PreparedWeightStore store(kModel);
            store.installPhysicalMemoryAuthority(memory);
            const auto spec = specification(type, CPUWeightConsumer::PreparedGemm);
            for (auto precision : {WeightPrecision::CONVERT_TO_FP32, WeightPrecision::CONVERT_TO_FP16,
                                   WeightPrecision::CONVERT_TO_BF16, WeightPrecision::CONVERT_TO_INT8})
            {
                auto converted = spec;
                converted.destination_precision = precision;
                EXPECT_THROW(store.beginCPUPreparation(binding(), converted, kOwner, CPUStoragePlacement::local()), std::invalid_argument);
                EXPECT_EQ(live(memory), 0u);
            }
            auto fragment = spec;
            fragment.range.columns = kBlockElements;
            fragment.range.native_row_bytes = kSourceBlockBytes;
            EXPECT_THROW(store.beginCPUPreparation(binding(), fragment, kOwner, CPUStoragePlacement::local()), std::invalid_argument);
            EXPECT_EQ(live(memory), 0u);
            EXPECT_THROW(quoteCPUWeightPreparation(type, kRows, kColumns + 1, WeightPrecision::NATIVE,
                CPUWeightConsumer::PreparedGemm), std::invalid_argument);
            auto incomplete = store.beginCPUPreparation(binding(), spec, kOwner, CPUStoragePlacement::local());
            EXPECT_THROW(store.finishCPUPreparation(std::move(incomplete)), std::logic_error);
            EXPECT_EQ(live(memory), 0u);
            EXPECT_FALSE(store.preparedRefForBinding(1, kCPU));

            auto malformed = store.beginCPUPreparation(binding(), spec, kOwner, CPUStoragePlacement::local());
            auto blocks = source(type);
            blocks.back().unit = -1;
            malformed->readDirect(spec.range, [&](std::span<std::byte> destination) {
                std::memcpy(destination.data(), blocks.data(), destination.size());
            });
            EXPECT_THROW(store.finishCPUPreparation(std::move(malformed)), std::invalid_argument);
            EXPECT_EQ(live(memory), 0u);
            EXPECT_FALSE(store.preparedRefForBinding(1, kCPU));

            auto failed = store.beginCPUPreparation(binding(), spec, kOwner, CPUStoragePlacement::local());
            EXPECT_THROW(failed->readDirect(spec.range, [](std::span<std::byte>) {
                throw std::runtime_error("Recorded read failure");
            }), std::runtime_error);
            EXPECT_THROW(store.finishCPUPreparation(std::move(failed)), std::logic_error);
            EXPECT_EQ(live(memory), 0u);
            EXPECT_FALSE(store.preparedRefForBinding(1, kCPU));
        }
    }
} // namespace llaminar2::w4a4
