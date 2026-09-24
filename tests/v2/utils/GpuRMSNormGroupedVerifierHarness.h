/**
 * @file GpuRMSNormGroupedVerifierHarness.h
 * @brief Shared all-format grouped RMSNorm decode-equivalence integration sweep.
 *
 * CUDA and ROCm both provide native FP32, BF16, and FP16 RMSNorm kernels. MTP
 * executes a runtime-sized verifier group in one production launch, whereas
 * ordinary decode executes one row. This harness proves that grouping does not
 * alter any native output byte and that the production tensor-aware API emits
 * exactly one economical grouped-route observation.
 *
 * Two row widths are intentional. A 128-value row represents per-head Q/K
 * normalization and selects the narrow reduction launch. A 4096-value row
 * represents model hidden-state normalization and selects the wide reduction
 * launch. Together they guard both launch policies used by Qwen-family graphs.
 */

#pragma once

#include <gtest/gtest.h>

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
#include <string>
#include <vector>

namespace llaminar2::test::gpu_rmsnorm_verifier
{
    /**
     * @brief Enable perfstats for one suite and restore the caller environment.
     *
     * Route counters are part of the correctness contract: byte equality alone
     * could be produced by a hidden serial-row implementation. The scoped owner
     * makes counter collection deterministic without leaking test configuration
     * into later integration binaries.
     */
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
     * @brief Generate deterministic finite values that expose row-stride bugs.
     *
     * The xorshift sequence is independent of the C library random generator,
     * making byte failures reproducible on every machine and backend.
     *
     * @param count Number of FP32 source values to create.
     * @param seed Non-zero sequence seed.
     * @return Host FP32 values in a bounded range suitable for RMSNorm.
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
     * @brief Generate deterministic positive FP32 RMSNorm scale weights.
     *
     * @param cols Number of values in one scale vector.
     * @return One non-uniform scale per normalized column.
     */
    inline std::vector<float> makeGamma(size_t cols)
    {
        std::vector<float> gamma(cols);
        for (size_t col = 0; col < cols; ++col)
            gamma[col] = 0.75f + static_cast<float>((col * 37u) % 257u) / 512.0f;
        return gamma;
    }

    /**
     * @brief Materialize an activation tensor in its real native representation.
     *
     * @tparam Precision FP32, BF16, or FP16 production activation precision.
     * @param shape Tensor shape in logical values.
     * @param source FP32 source values converted once on the host.
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

    /**
     * @brief Create a native output tensor initialized to deterministic zeros.
     *
     * @tparam Precision Native output precision.
     * @param shape Output tensor shape.
     * @return Native tensor ready for device allocation.
     */
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

    /**
     * @brief Read a required perfstats tag and expose absent tags in failures.
     *
     * @param record Counter record under inspection.
     * @param name Required tag key.
     * @return Tag value or a descriptive missing-tag marker.
     */
    inline std::string tag(const PerfStatRecord &record, const char *name)
    {
        const auto it = record.tags.find(name);
        return it == record.tags.end()
                   ? std::string("<missing:") + name + '>'
                   : it->second;
    }

    /**
     * @brief Require native byte equality and identify the first differing byte.
     *
     * @param grouped Bytes produced by one runtime-M production launch.
     * @param serial Bytes produced by M independent production M=1 launches.
     * @param label Matrix-cell description printed on failure.
     */
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
     * @brief Prove that the production tensor API rejects a default GPU stream.
     *
     * Tensor allocation uses the suite's valid non-default stream, then a fresh
     * kernel is intentionally left unbound. The failed call must not enqueue
     * work or publish grouped-route telemetry. This is a regression guard for
     * accidental reintroduction of CUDA/HIP legacy-stream execution.
     *
     * @tparam Precision Native activation/output precision.
     * @tparam Kernel Concrete backend RMSNorm specialization.
     * @tparam Runtime Backend synchronization policy.
     * @param runtime Backend runtime policy object.
     * @param device Device placement for tensor allocation.
     * @param allocation_stream Valid stream used only to prepare tensors.
     * @param counter_name Backend grouped-route counter name.
     * @param format_label Native tensor-format label for diagnostics.
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
        constexpr float epsilon = 1e-6f;
        const auto source = makeValues(
            static_cast<size_t>(rows) * cols, 0x57EA0000u);
        const auto gamma_values = makeGamma(cols);

        auto input = makeNativeTensor<Precision>(
            {static_cast<size_t>(rows), static_cast<size_t>(cols)},
            source.data());
        auto output = makeZeroNativeTensor<Precision>(
            {static_cast<size_t>(rows), static_cast<size_t>(cols)});
        auto gamma = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(cols)});
        std::copy(gamma_values.begin(), gamma_values.end(), gamma->mutable_data());

        ASSERT_TRUE(input->ensureOnDevice(device, allocation_stream));
        ASSERT_TRUE(output->ensureOnDevice(device, allocation_stream));
        ASSERT_TRUE(gamma->ensureOnDevice(device, allocation_stream));
        runtime.synchronize(allocation_stream);

        Kernel unbound_kernel(0);
        PerfStatsCollector::reset();
        EXPECT_FALSE(unbound_kernel.apply_tensor(
            input.get(), gamma.get(), output.get(),
            rows, cols, epsilon, nullptr, device.toKernelDeviceIndex()))
            << format_label << " production RMSNorm accepted a default stream";

        const std::string metric = std::string("kernel.") + counter_name;
        EXPECT_TRUE(PerfStatsCollector::snapshot({metric}).empty())
            << format_label << " failed launch published grouped telemetry";
    }

    /**
     * @brief Sweep one native format over every verifier depth and row width.
     *
     * Runtime supplies only backend copy and synchronization mechanics. Kernel
     * is a concrete CUDA or ROCm RMSNorm specialization and every invocation
     * enters `apply_tensor`, the same production API used by RMSNormStage.
     * Serial rows are copied from the exact native grouped input bytes so host
     * conversion cannot hide a source-format discrepancy.
     *
     * @tparam Precision Native activation/output precision.
     * @tparam Kernel Concrete backend RMSNorm specialization.
     * @tparam Runtime Backend D2H and stream synchronization policy.
     * @param runtime Backend runtime policy object.
     * @param device Device placement for all tensor allocations.
     * @param stream Mandatory non-default backend stream.
     * @param backend_label Human-readable backend label for diagnostics.
     * @param counter_name Backend grouped-route counter name.
     * @param format_label Native tensor-format label expected in telemetry.
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
        constexpr float epsilon = 1e-6f;
        // The 5120-wide lane is a production Qwen hidden-state geometry, not a
        // synthetic approximation.  Its five values per 1024-thread lane make
        // it a materially different reduction tree from the 4096-wide case.
        constexpr std::array<int, 3> column_counts = {128, 4096, 5120};
        constexpr int max_rows = kGroupedVerifierRuntimeRows.back();

        for (int cols : column_counts)
        {
            const auto source = makeValues(
                static_cast<size_t>(max_rows) * cols,
                0x6D545000u ^ static_cast<uint32_t>(cols));
            const auto gamma_values = makeGamma(static_cast<size_t>(cols));

            for (int verifier_rows : kGroupedVerifierRuntimeRows)
            {
                SCOPED_TRACE(
                    std::string(backend_label) + " format=" + format_label +
                    " M=" + std::to_string(verifier_rows) +
                    " cols=" + std::to_string(cols));

                Kernel kernel(0);
                kernel.setGPUStream(stream);

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
                    grouped_shape, source.data());
                auto grouped_output = makeZeroNativeTensor<Precision>(grouped_shape);

                // Preserve the exact post-conversion bytes before device
                // allocation, then feed each row unchanged to the serial oracle.
                std::vector<uint8_t> native_input(grouped_input->size_bytes());
                std::memcpy(
                    native_input.data(), grouped_input->raw_data(), native_input.size());
                const size_t row_bytes = native_input.size() /
                                         static_cast<size_t>(verifier_rows);
                std::vector<uint8_t> serial_output(grouped_output->size_bytes(), 0u);

                for (int row = 0; row < verifier_rows; ++row)
                {
                    const std::vector<size_t> row_shape = {
                        1u, static_cast<size_t>(cols)};
                    auto row_input = makeNativeTensor<Precision>(
                        row_shape,
                        source.data() + static_cast<size_t>(row) * cols);
                    auto row_output = makeZeroNativeTensor<Precision>(row_shape);
                    std::memcpy(
                        row_input->raw_mutable_data(),
                        native_input.data() + static_cast<size_t>(row) * row_bytes,
                        row_bytes);

                    ASSERT_TRUE(row_input->ensureOnDevice(device, stream));
                    ASSERT_TRUE(row_output->ensureOnDevice(device, stream));
                    ASSERT_TRUE(kernel.apply_tensor(
                        row_input.get(), gamma.get(), row_output.get(),
                        1, cols, epsilon, nullptr,
                        device.toKernelDeviceIndex()));
                    runtime.copyDeviceToHost(
                        serial_output.data() + static_cast<size_t>(row) * row_bytes,
                        row_output->gpu_data_ptr(), row_bytes, stream);
                    runtime.synchronize(stream);
                }

                ASSERT_TRUE(grouped_input->ensureOnDevice(device, stream));
                ASSERT_TRUE(grouped_output->ensureOnDevice(device, stream));
                runtime.synchronize(stream);

                // Reset immediately before the grouped production call. M=1
                // witnesses are intentionally uncounted, while this call must
                // produce one and only one route record.
                PerfStatsCollector::reset();
                ASSERT_TRUE(kernel.apply_tensor(
                    grouped_input.get(), gamma.get(), grouped_output.get(),
                    verifier_rows, cols, epsilon, nullptr,
                    device.toKernelDeviceIndex()));

                std::vector<uint8_t> grouped_bytes(grouped_output->size_bytes());
                runtime.copyDeviceToHost(
                    grouped_bytes.data(), grouped_output->gpu_data_ptr(),
                    grouped_bytes.size(), stream);
                runtime.synchronize(stream);
                expectNativeBytesEqual(
                    grouped_bytes, serial_output,
                    std::string(backend_label) + " " + format_label +
                        " RMSNorm grouped verifier output");

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
                EXPECT_EQ(tag(record, "capture_mode"), "direct");
                EXPECT_EQ(tag(record, "row_mapping"), "one_block_per_row");
                EXPECT_EQ(
                    tag(record, "invocation_policy"),
                    "single_grouped_launch");
            }
        }
    }

    /**
     * @brief Execute the complete symmetric GPU RMSNorm format matrix.
     *
     * @tparam KernelTemplate CUDA or ROCm RMSNorm class template.
     * @tparam Runtime Backend runtime policy.
     * @param runtime Backend copy/synchronization policy.
     * @param device Device under test.
     * @param stream Mandatory non-default stream owned by the test binary.
     * @param backend_label Human-readable backend label.
     * @param counter_name Backend grouped-route metric name.
     */
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
} // namespace llaminar2::test::gpu_rmsnorm_verifier
