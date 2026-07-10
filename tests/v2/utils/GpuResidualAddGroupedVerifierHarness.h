/**
 * @file GpuResidualAddGroupedVerifierHarness.h
 * @brief Shared all-format grouped residual-add byte-equivalence GPU sweep.
 *
 * Residual addition has no cross-row arithmetic, so the economical MTP route is
 * one flat device launch over the complete M-row span. This harness proves that
 * CUDA and ROCm preserve every native FP32, BF16, and FP16 output byte relative
 * to production M=1 decode. It also verifies device-only pointer ownership,
 * explicit stream binding, and one-launch perfstats publication.
 */

#pragma once

#include <gtest/gtest.h>

#include "backends/DeviceId.h"
#include "tensors/SIMDHelpers.h"
#include "tensors/Tensors.h"
#include "utils/DebugEnv.h"
#include "utils/PerfStatsCollector.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace llaminar2::test::gpu_residual_add_verifier
{
    /** @brief Enable route counters and restore the caller environment on exit. */
    class ScopedPerfStats
    {
    public:
        ScopedPerfStats()
        {
            if (const char *old = std::getenv("LLAMINAR_PERF_STATS_SUMMARY"))
            {
                had_old_value_ = true;
                old_value_ = old;
            }
            setenv("LLAMINAR_PERF_STATS_SUMMARY", "1", 1);
            mutableDebugEnv().reload();
            PerfStatsCollector::reset();
        }

        ~ScopedPerfStats()
        {
            if (had_old_value_)
                setenv("LLAMINAR_PERF_STATS_SUMMARY", old_value_.c_str(), 1);
            else
                unsetenv("LLAMINAR_PERF_STATS_SUMMARY");
            mutableDebugEnv().reload();
            PerfStatsCollector::reset();
        }

        ScopedPerfStats(const ScopedPerfStats &) = delete;
        ScopedPerfStats &operator=(const ScopedPerfStats &) = delete;

    private:
        bool had_old_value_ = false;
        std::string old_value_;
    };

    /**
     * @brief Generate deterministic finite inputs with distinct row patterns.
     *
     * @param count Number of FP32 source values.
     * @param seed Xorshift seed unique to an operand and width.
     * @return Bounded source values suitable for every native format.
     */
    inline std::vector<float> makeValues(size_t count, uint32_t seed)
    {
        std::vector<float> values(count);
        uint32_t state = seed;
        for (size_t index = 0; index < count; ++index)
        {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            const int centered = static_cast<int>(state % 4093u) - 2046;
            values[index] = static_cast<float>(centered) / 1024.0f;
        }
        return values;
    }

    /**
     * @brief Convert FP32 source values into one production native tensor.
     *
     * @tparam Precision FP32, BF16, or FP16 activation precision.
     * @param shape Logical tensor shape.
     * @param source FP32 values to convert exactly once on the host.
     * @return Type-erased native tensor.
     */
    template <ActivationPrecision Precision>
    std::unique_ptr<TensorBase> makeNativeTensor(
        const std::vector<size_t> &shape,
        const float *source)
    {
        if constexpr (Precision == ActivationPrecision::FP32)
        {
            auto tensor = std::make_unique<FP32Tensor>(shape);
            std::copy_n(source, tensor->numel(), tensor->mutable_data());
            return tensor;
        }
        else if constexpr (Precision == ActivationPrecision::BF16)
        {
            auto tensor = std::make_unique<BF16Tensor>(shape);
            simd::convert_fp32_to_bf16(
                source, tensor->mutable_typed_data(), tensor->numel());
            return tensor;
        }
        else
        {
            static_assert(Precision == ActivationPrecision::FP16);
            auto tensor = std::make_unique<FP16Tensor>(shape);
            simd::convert_fp32_to_fp16(
                source, tensor->mutable_typed_data(), tensor->numel());
            return tensor;
        }
    }

    /** @brief Create a deterministic zero-filled native output tensor. */
    template <ActivationPrecision Precision>
    std::unique_ptr<TensorBase> makeZeroNativeTensor(
        const std::vector<size_t> &shape)
    {
        size_t count = 1;
        for (size_t extent : shape)
            count *= extent;
        const std::vector<float> zeros(count, 0.0f);
        return makeNativeTensor<Precision>(shape, zeros.data());
    }

    /** @brief Return a required perfstats tag or a visible missing marker. */
    inline std::string tag(const PerfStatRecord &record, const char *name)
    {
        const auto it = record.tags.find(name);
        return it == record.tags.end()
                   ? std::string("<missing:") + name + '>'
                   : it->second;
    }

    /** @brief Require native byte equality and report the first mismatch. */
    inline void expectNativeBytesEqual(
        const std::vector<uint8_t> &grouped,
        const std::vector<uint8_t> &serial,
        const std::string &label)
    {
        ASSERT_EQ(grouped.size(), serial.size()) << label;
        if (grouped == serial)
            return;

        for (size_t index = 0; index < grouped.size(); ++index)
        {
            if (grouped[index] != serial[index])
            {
                ADD_FAILURE() << label << " first native mismatch at byte " << index
                              << " grouped=" << static_cast<unsigned>(grouped[index])
                              << " serial=" << static_cast<unsigned>(serial[index]);
                return;
            }
        }
    }

    /**
     * @brief Prove one native specialization rejects an unbound default stream.
     *
     * Allocation uses a valid stream; only the production kernel is left
     * unbound. A rejected call must emit no grouped-route telemetry.
     */
    template <ActivationPrecision Precision, typename Kernel, typename Runtime>
    void verifyExplicitStreamContract(
        Runtime &runtime,
        DeviceId device,
        void *allocation_stream,
        const char *counter_name,
        const char *format_label)
    {
        constexpr int rows = 2;
        constexpr int cols = 128;
        const auto input_values = makeValues(
            static_cast<size_t>(rows) * cols, 0xA11CE001u);
        const auto residual_values = makeValues(
            static_cast<size_t>(rows) * cols, 0xA11CE002u);
        auto input = makeNativeTensor<Precision>(
            {static_cast<size_t>(rows), static_cast<size_t>(cols)},
            input_values.data());
        auto residual = makeNativeTensor<Precision>(
            {static_cast<size_t>(rows), static_cast<size_t>(cols)},
            residual_values.data());
        auto output = makeZeroNativeTensor<Precision>(
            {static_cast<size_t>(rows), static_cast<size_t>(cols)});

        ASSERT_TRUE(input->ensureOnDevice(device, allocation_stream));
        ASSERT_TRUE(residual->ensureOnDevice(device, allocation_stream));
        ASSERT_TRUE(output->ensureOnDevice(device, allocation_stream));
        runtime.synchronize(allocation_stream);

        Kernel unbound_kernel(0);
        PerfStatsCollector::reset();
        EXPECT_FALSE(unbound_kernel.apply_tensor(
            input.get(), residual.get(), output.get(),
            static_cast<size_t>(rows) * cols,
            nullptr, device.toKernelDeviceIndex()))
            << format_label << " residual add accepted a default stream";

        const std::string metric = std::string("kernel.") + counter_name;
        EXPECT_TRUE(PerfStatsCollector::snapshot({metric}).empty())
            << format_label << " failed launch published grouped telemetry";
    }

    /**
     * @brief Sweep one native format across M=2..4 and two production widths.
     *
     * Serial witnesses consume byte copies of the already-converted grouped
     * operands, ensuring host conversion cannot hide a native-format mismatch.
     * Every call enters the same tensor-aware API used by ResidualAddStage.
     */
    template <ActivationPrecision Precision, typename Kernel, typename Runtime>
    void runNativeFormat(
        Runtime &runtime,
        DeviceId device,
        void *stream,
        const char *backend_label,
        const char *counter_name,
        const char *format_label)
    {
        constexpr std::array<int, 2> column_counts = {128, 4096};

        for (int cols : column_counts)
        {
            const auto input_values = makeValues(
                static_cast<size_t>(4) * cols,
                0x1A2B3000u ^ static_cast<uint32_t>(cols));
            const auto residual_values = makeValues(
                static_cast<size_t>(4) * cols,
                0x4C5D6000u ^ static_cast<uint32_t>(cols));

            for (int verifier_rows : {2, 3, 4})
            {
                SCOPED_TRACE(
                    std::string(backend_label) + " format=" + format_label +
                    " M=" + std::to_string(verifier_rows) +
                    " cols=" + std::to_string(cols));

                Kernel kernel(0);
                kernel.setGPUStream(stream);
                const std::vector<size_t> grouped_shape = {
                    static_cast<size_t>(verifier_rows),
                    static_cast<size_t>(cols),
                };
                auto grouped_input = makeNativeTensor<Precision>(
                    grouped_shape, input_values.data());
                auto grouped_residual = makeNativeTensor<Precision>(
                    grouped_shape, residual_values.data());
                auto grouped_output = makeZeroNativeTensor<Precision>(grouped_shape);

                std::vector<uint8_t> native_input(grouped_input->size_bytes());
                std::vector<uint8_t> native_residual(grouped_residual->size_bytes());
                std::memcpy(
                    native_input.data(), grouped_input->raw_data(), native_input.size());
                std::memcpy(
                    native_residual.data(), grouped_residual->raw_data(),
                    native_residual.size());
                const size_t row_bytes = native_input.size() /
                                         static_cast<size_t>(verifier_rows);
                std::vector<uint8_t> serial_output(grouped_output->size_bytes(), 0u);

                for (int row = 0; row < verifier_rows; ++row)
                {
                    const std::vector<size_t> row_shape = {
                        1u, static_cast<size_t>(cols)};
                    auto row_input = makeNativeTensor<Precision>(
                        row_shape,
                        input_values.data() + static_cast<size_t>(row) * cols);
                    auto row_residual = makeNativeTensor<Precision>(
                        row_shape,
                        residual_values.data() + static_cast<size_t>(row) * cols);
                    auto row_output = makeZeroNativeTensor<Precision>(row_shape);
                    std::memcpy(
                        row_input->raw_mutable_data(),
                        native_input.data() + static_cast<size_t>(row) * row_bytes,
                        row_bytes);
                    std::memcpy(
                        row_residual->raw_mutable_data(),
                        native_residual.data() + static_cast<size_t>(row) * row_bytes,
                        row_bytes);

                    ASSERT_TRUE(row_input->ensureOnDevice(device, stream));
                    ASSERT_TRUE(row_residual->ensureOnDevice(device, stream));
                    ASSERT_TRUE(row_output->ensureOnDevice(device, stream));
                    ASSERT_TRUE(kernel.apply_tensor(
                        row_input.get(), row_residual.get(), row_output.get(),
                        static_cast<size_t>(cols), nullptr,
                        device.toKernelDeviceIndex()));
                    runtime.copyDeviceToHost(
                        serial_output.data() + static_cast<size_t>(row) * row_bytes,
                        row_output->gpu_data_ptr(), row_bytes, stream);
                    runtime.synchronize(stream);
                }

                ASSERT_TRUE(grouped_input->ensureOnDevice(device, stream));
                ASSERT_TRUE(grouped_residual->ensureOnDevice(device, stream));
                ASSERT_TRUE(grouped_output->ensureOnDevice(device, stream));
                runtime.synchronize(stream);

                const size_t active_elements =
                    static_cast<size_t>(verifier_rows) * cols;
                PerfStatsCollector::reset();
                ASSERT_TRUE(kernel.apply_tensor(
                    grouped_input.get(), grouped_residual.get(), grouped_output.get(),
                    active_elements, nullptr, device.toKernelDeviceIndex()));

                std::vector<uint8_t> grouped_bytes(grouped_output->size_bytes());
                runtime.copyDeviceToHost(
                    grouped_bytes.data(), grouped_output->gpu_data_ptr(),
                    grouped_bytes.size(), stream);
                runtime.synchronize(stream);
                expectNativeBytesEqual(
                    grouped_bytes, serial_output,
                    std::string(backend_label) + " " + format_label +
                        " grouped residual add");

                const std::string metric = std::string("kernel.") + counter_name;
                const auto records = PerfStatsCollector::snapshot({metric});
                ASSERT_EQ(records.size(), 1u)
                    << PerfStatsCollector::summaryString({metric}, 20);
                const auto &record = records.front();
                EXPECT_EQ(record.count, 1u);
                EXPECT_EQ(tag(record, "tensor_format"), format_label);
                EXPECT_EQ(
                    tag(record, "verifier_rows"),
                    std::to_string(verifier_rows));
                EXPECT_EQ(tag(record, "cols"), std::to_string(cols));
                EXPECT_EQ(
                    tag(record, "active_elements"),
                    std::to_string(active_elements));
                EXPECT_EQ(tag(record, "capture_mode"), "direct");
                EXPECT_EQ(
                    tag(record, "invocation_policy"),
                    "single_flat_launch");
            }
        }
    }

    /** @brief Execute all native formats and the explicit-stream contract. */
    template <template <ActivationPrecision> class KernelTemplate, typename Runtime>
    void runAllFormats(
        Runtime &runtime,
        DeviceId device,
        void *stream,
        const char *backend_label,
        const char *counter_name)
    {
        ScopedPerfStats perfstats;
        verifyExplicitStreamContract<ActivationPrecision::FP32,
                                     KernelTemplate<ActivationPrecision::FP32>>(
            runtime, device, stream, counter_name, "FP32");
        verifyExplicitStreamContract<ActivationPrecision::BF16,
                                     KernelTemplate<ActivationPrecision::BF16>>(
            runtime, device, stream, counter_name, "BF16");
        verifyExplicitStreamContract<ActivationPrecision::FP16,
                                     KernelTemplate<ActivationPrecision::FP16>>(
            runtime, device, stream, counter_name, "FP16");

        runNativeFormat<ActivationPrecision::FP32,
                        KernelTemplate<ActivationPrecision::FP32>>(
            runtime, device, stream, backend_label, counter_name, "FP32");
        runNativeFormat<ActivationPrecision::BF16,
                        KernelTemplate<ActivationPrecision::BF16>>(
            runtime, device, stream, backend_label, counter_name, "BF16");
        runNativeFormat<ActivationPrecision::FP16,
                        KernelTemplate<ActivationPrecision::FP16>>(
            runtime, device, stream, backend_label, counter_name, "FP16");
    }
} // namespace llaminar2::test::gpu_residual_add_verifier
