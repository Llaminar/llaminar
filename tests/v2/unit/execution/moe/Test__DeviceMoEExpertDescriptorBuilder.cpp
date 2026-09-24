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

#include <array>
#include <cstddef>
#include <cstdint>

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
                : native_(native), exports_native_(true)
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

        private:
            DeviceNativeVNNIMatrixDesc native_{};
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
    } // namespace

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
} // namespace llaminar2::test
