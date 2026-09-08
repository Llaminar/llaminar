/**
 * @file GpuFusedResidualNormGroupedVerifierHarness.h
 * @brief Shared CUDA/ROCm grouped fused-residual RMSNorm proof matrix.
 *
 * This harness enters FusedResidualNormStage rather than calling backend launch
 * wrappers directly. It therefore covers stage type dispatch, strict device
 * pointer ownership, explicit-stream enforcement, in-place residual publication,
 * normalized-output publication, and grouped route telemetry. CUDA and ROCm test
 * binaries supply only their native copy/synchronization mechanics.
 */

#pragma once

#include <gtest/gtest.h>

#include "execution/compute_stages/stages/FusedResidualNormStage.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "tensors/SIMDHelpers.h"
#include "tensors/Tensors.h"
#include "utils/DebugEnv.h"
#include "utils/PerfStatsCollector.h"
#include "utils/VerifierRowTestInventory.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace llaminar2::test::gpu_fused_residual_norm_verifier
{
    /** @brief Enable perfstats for the matrix and restore the caller on exit. */
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

    /** @brief Generate deterministic finite activations with row-distinct values. */
    inline std::vector<float> makeValues(size_t count, uint32_t seed)
    {
        std::vector<float> values(count);
        uint32_t state = seed;
        for (size_t index = 0; index < count; ++index)
        {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            const int centered = static_cast<int>(state % 3079u) - 1539;
            values[index] = static_cast<float>(centered) / 1024.0f;
        }
        return values;
    }

    /** @brief Generate non-uniform positive FP32 RMSNorm scale weights. */
    inline std::vector<float> makeGamma(size_t cols)
    {
        std::vector<float> gamma(cols);
        for (size_t col = 0; col < cols; ++col)
            gamma[col] = 0.625f + static_cast<float>((col * 53u) % 383u) / 512.0f;
        return gamma;
    }

    /**
     * @brief Materialize an activation tensor in its true native representation.
     *
     * @tparam Precision FP32, BF16, or FP16 production activation precision.
     * @param shape Logical tensor shape.
     * @param source FP32 values converted once before serial/grouped execution.
     * @return Type-erased tensor retaining the requested native bytes.
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

    /** @brief Create a native output tensor initialized to deterministic zero. */
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

    /** @brief Read one required telemetry tag with an explicit missing marker. */
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

    /** @brief Build one production FusedResidualNormStage with fixed bindings. */
    inline FusedResidualNormStage makeStage(
        const TensorBase *input,
        TensorBase *residual,
        const TensorBase *gamma,
        TensorBase *norm_output,
        DeviceId device,
        int rows,
        int cols)
    {
        FusedResidualNormStage::Params params{};
        params.device_id = device;
        params.input = input;
        params.residual = residual;
        params.gamma = gamma;
        params.norm_output = norm_output;
        params.eps = 1e-6f;
        params.seq_len = rows;
        params.hidden_dim = cols;
        return FusedResidualNormStage(std::move(params));
    }

    /**
     * @brief Prove the production stage rejects an unbound GPU stream.
     *
     * Allocation uses the suite's valid stream so all tensors own real device
     * buffers. The stage itself is deliberately left unbound. Failure must occur
     * before launch and must not publish grouped route telemetry.
     */
    template <ActivationPrecision Precision, typename Runtime>
    void verifyExplicitStreamContract(
        Runtime &runtime,
        IDeviceContext *context,
        DeviceId device,
        void *allocation_stream,
        const char *counter_name,
        const char *format_label)
    {
        constexpr int rows = 2;
        constexpr int cols = 128;
        const auto input_values = makeValues(
            static_cast<size_t>(rows) * cols, 0xF1010000u);
        const auto residual_values = makeValues(
            static_cast<size_t>(rows) * cols, 0xF1020000u);
        const auto gamma_values = makeGamma(cols);

        auto input = makeNativeTensor<Precision>(
            {static_cast<size_t>(rows), static_cast<size_t>(cols)},
            input_values.data());
        auto residual = makeNativeTensor<Precision>(
            {static_cast<size_t>(rows), static_cast<size_t>(cols)},
            residual_values.data());
        auto norm_output = makeZeroNativeTensor<Precision>(
            {static_cast<size_t>(rows), static_cast<size_t>(cols)});
        auto gamma = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(cols)});
        std::copy(gamma_values.begin(), gamma_values.end(), gamma->mutable_data());

        ASSERT_TRUE(input->ensureOnDevice(device, allocation_stream));
        ASSERT_TRUE(residual->ensureOnDevice(device, allocation_stream));
        ASSERT_TRUE(norm_output->ensureOnDevice(device, allocation_stream));
        ASSERT_TRUE(gamma->ensureOnDevice(device, allocation_stream));
        runtime.synchronize(allocation_stream);

        auto stage = makeStage(
            input.get(), residual.get(), gamma.get(), norm_output.get(),
            device, rows, cols);
        PerfStatsCollector::reset();
        EXPECT_THROW(
            {
                (void)stage.execute(context);
            },
            std::logic_error)
            << format_label
            << " production stage did not fail hard for an unbound GPU stream";

        const std::string metric = std::string("kernel.") + counter_name;
        EXPECT_TRUE(PerfStatsCollector::snapshot({metric}).empty())
            << format_label << " failed launch published grouped telemetry";
    }

    /**
     * @brief Sweep one native GPU format over every verifier depth and width.
     *
     * Runtime provides only asynchronous D2H copies and synchronization. Every
     * operation under proof enters FusedResidualNormStage::execute on the same
     * explicit non-default stream. Serial witnesses receive exact slices of the
     * grouped native input bytes, and both output tensors are observed directly
     * from device memory without adopting host-visible production storage.
     */
    template <ActivationPrecision Precision, typename Runtime>
    void runNativeFormat(
        Runtime &runtime,
        IDeviceContext *context,
        DeviceId device,
        void *stream,
        const char *backend_label,
        const char *counter_name,
        const char *format_label)
    {
        // Keep the narrow reduction and established 4096-wide lane, while also
        // exercising the exact hidden widths used by Qwen3.5-122B (3072) and
        // Qwen3.6/Qwen2.5-32B (5120).  Neither production width is represented
        // by a nearby power of two: their distinct per-lane traversal counts
        // have exposed model-only grouped-verifier drift in the past.
        constexpr std::array<int, 4> column_counts = {128, 3072, 4096, 5120};
        constexpr int max_rows = kGroupedVerifierRuntimeRows.back();

        for (int cols : column_counts)
        {
            const auto input_values = makeValues(
                static_cast<size_t>(max_rows) * cols,
                0xF2010000u ^ static_cast<uint32_t>(cols));
            const auto residual_values = makeValues(
                static_cast<size_t>(max_rows) * cols,
                0xF2020000u ^ static_cast<uint32_t>(cols));
            const auto gamma_values = makeGamma(static_cast<size_t>(cols));

            for (int verifier_rows : kGroupedVerifierRuntimeRows)
            {
                SCOPED_TRACE(
                    std::string(backend_label) + " format=" + format_label +
                    " M=" + std::to_string(verifier_rows) +
                    " cols=" + std::to_string(cols));

                auto gamma = std::make_unique<FP32Tensor>(
                    std::vector<size_t>{static_cast<size_t>(cols)});
                std::copy(
                    gamma_values.begin(), gamma_values.end(), gamma->mutable_data());
                ASSERT_TRUE(gamma->ensureOnDevice(device, stream));

                const std::vector<size_t> grouped_shape = {
                    static_cast<size_t>(verifier_rows),
                    static_cast<size_t>(cols),
                };
                auto grouped_input = makeNativeTensor<Precision>(
                    grouped_shape, input_values.data());
                auto grouped_residual = makeNativeTensor<Precision>(
                    grouped_shape, residual_values.data());
                auto grouped_norm = makeZeroNativeTensor<Precision>(grouped_shape);

                std::vector<uint8_t> native_input(grouped_input->size_bytes());
                std::vector<uint8_t> native_residual(grouped_residual->size_bytes());
                std::memcpy(
                    native_input.data(), grouped_input->raw_data(), native_input.size());
                std::memcpy(
                    native_residual.data(), grouped_residual->raw_data(),
                    native_residual.size());

                const size_t row_bytes = native_input.size() /
                                         static_cast<size_t>(verifier_rows);
                std::vector<uint8_t> serial_residual(grouped_residual->size_bytes(), 0u);
                std::vector<uint8_t> serial_norm(grouped_norm->size_bytes(), 0u);

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
                    auto row_norm = makeZeroNativeTensor<Precision>(row_shape);
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
                    ASSERT_TRUE(row_norm->ensureOnDevice(device, stream));

                    auto serial_stage = makeStage(
                        row_input.get(), row_residual.get(), gamma.get(),
                        row_norm.get(), device, 1, cols);
                    serial_stage.setGPUStream(stream);
                    ASSERT_TRUE(serial_stage.execute(context));
                    runtime.copyDeviceToHost(
                        serial_residual.data() + static_cast<size_t>(row) * row_bytes,
                        row_residual->gpu_data_ptr(), row_bytes, stream);
                    runtime.copyDeviceToHost(
                        serial_norm.data() + static_cast<size_t>(row) * row_bytes,
                        row_norm->gpu_data_ptr(), row_bytes, stream);
                    runtime.synchronize(stream);
                }

                ASSERT_TRUE(grouped_input->ensureOnDevice(device, stream));
                ASSERT_TRUE(grouped_residual->ensureOnDevice(device, stream));
                ASSERT_TRUE(grouped_norm->ensureOnDevice(device, stream));
                runtime.synchronize(stream);

                PerfStatsCollector::reset();
                auto grouped_stage = makeStage(
                    grouped_input.get(), grouped_residual.get(), gamma.get(),
                    grouped_norm.get(), device, verifier_rows, cols);
                grouped_stage.setGPUStream(stream);
                ASSERT_TRUE(grouped_stage.execute(context));

                std::vector<uint8_t> grouped_residual_bytes(
                    grouped_residual->size_bytes());
                std::vector<uint8_t> grouped_norm_bytes(grouped_norm->size_bytes());
                runtime.copyDeviceToHost(
                    grouped_residual_bytes.data(), grouped_residual->gpu_data_ptr(),
                    grouped_residual_bytes.size(), stream);
                runtime.copyDeviceToHost(
                    grouped_norm_bytes.data(), grouped_norm->gpu_data_ptr(),
                    grouped_norm_bytes.size(), stream);
                runtime.synchronize(stream);

                expectNativeBytesEqual(
                    grouped_residual_bytes, serial_residual,
                    std::string(backend_label) + " " + format_label +
                        " grouped residual publication");
                expectNativeBytesEqual(
                    grouped_norm_bytes, serial_norm,
                    std::string(backend_label) + " " + format_label +
                        " grouped normalized publication");

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
                EXPECT_EQ(tag(record, "implementation"), "native_fused_kernel");
                EXPECT_EQ(tag(record, "capture_mode"), "direct");
                EXPECT_EQ(tag(record, "row_mapping"), "independent_rows");
                EXPECT_EQ(tag(record, "invocation_policy"), "single_grouped_launch");
            }
        }
    }

    /**
     * @brief Execute the complete symmetric GPU fused-residual RMSNorm matrix.
     */
    template <typename Runtime>
    void runAllFormats(
        Runtime &runtime,
        IDeviceContext *context,
        DeviceId device,
        void *stream,
        const char *backend_label,
        const char *counter_name)
    {
        ScopedPerfStats perfstats;
        verifyExplicitStreamContract<ActivationPrecision::FP32>(
            runtime, context, device, stream, counter_name, "FP32");
        verifyExplicitStreamContract<ActivationPrecision::BF16>(
            runtime, context, device, stream, counter_name, "BF16");
        verifyExplicitStreamContract<ActivationPrecision::FP16>(
            runtime, context, device, stream, counter_name, "FP16");
        runNativeFormat<ActivationPrecision::FP32>(
            runtime, context, device, stream,
            backend_label, counter_name, "FP32");
        runNativeFormat<ActivationPrecision::BF16>(
            runtime, context, device, stream,
            backend_label, counter_name, "BF16");
        runNativeFormat<ActivationPrecision::FP16>(
            runtime, context, device, stream,
            backend_label, counter_name, "FP16");
    }
} // namespace llaminar2::test::gpu_fused_residual_norm_verifier
