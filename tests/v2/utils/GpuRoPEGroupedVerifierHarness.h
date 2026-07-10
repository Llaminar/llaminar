/**
 * @file GpuRoPEGroupedVerifierHarness.h
 * @brief Shared native-byte sweep for production CUDA and ROCm grouped RoPE.
 *
 * MTP verifier rows may use either contiguous positions represented by a
 * device-resident scalar offset or explicit device position rows for batched
 * requests. Both GPU backends implement FP32, BF16, and FP16 fused Q/K kernels
 * for those routes. This harness compares one M=2..4 production launch against
 * the same native rows processed by the backend's production M=1 contiguous
 * path and requires byte equality plus an exact route-counter observation.
 */

#pragma once

#include <gtest/gtest.h>

#include "execution/local_execution/device/DeviceWorkspaceManager.h"
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

namespace llaminar2::test::gpu_rope_verifier
{
    /** @brief Device-position owners exercised by production verifier graphs. */
    enum class PositionRoute
    {
        ContiguousDeviceScalar,
        ExplicitDeviceRows,
    };

    /** @brief Enable route telemetry without leaking environment state. */
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

    /** @brief Build deterministic values with enough variation to expose strides. */
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

    /** @brief Materialize one supported native activation tensor on the host. */
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

    /** @brief Read a required perfstats tag or return a visible marker. */
    inline std::string tag(const PerfStatRecord &record, const char *name)
    {
        const auto it = record.tags.find(name);
        return it == record.tags.end() ? std::string("<missing:") + name + '>' : it->second;
    }

    /** @brief Report the first native byte where grouped and serial differ. */
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
     * @brief Run one native format through every depth and position owner.
     *
     * Runtime supplies only backend copy/synchronization mechanics. Kernel is a
     * concrete CUDARoPEKernelT or ROCmRoPEKernelT specialization, so every call
     * enters the same tensor-aware API used by RoPEStage in production.
     */
    template <ActivationPrecision Precision, typename Kernel, typename Runtime>
    void runNativeFormat(
        Runtime &runtime,
        DeviceId device,
        void *stream,
        const char *backend_label,
        const char *counter_name,
        const char *format_label,
        bool include_partial_rope)
    {
        constexpr int n_heads = 4;
        constexpr int n_kv_heads = 2;
        constexpr int head_dim = 128;
        constexpr int partial_rotary_dim = 64;
        constexpr float rope_theta = 1000000.0f;
        constexpr std::array<int, 4> explicit_positions = {101, 101, 107, 109};
        constexpr std::array<PositionRoute, 2> position_routes = {
            PositionRoute::ContiguousDeviceScalar,
            PositionRoute::ExplicitDeviceRows,
        };

        const size_t q_cols = static_cast<size_t>(n_heads) * head_dim;
        const size_t k_cols = static_cast<size_t>(n_kv_heads) * head_dim;
        const auto q_values = makeValues(4 * q_cols, 0xC001D00Du);
        const auto k_values = makeValues(4 * k_cols, 0x51A7E123u);
        const std::array<int, 2> rotary_dims = {
            head_dim,
            include_partial_rope ? partial_rotary_dim : head_dim,
        };
        const int rotary_case_count = include_partial_rope ? 2 : 1;

        for (int rotary_case = 0; rotary_case < rotary_case_count; ++rotary_case)
        {
            const int effective_rotary_dim = rotary_dims[static_cast<size_t>(rotary_case)];
            const int rotary_argument = effective_rotary_dim == head_dim
                                            ? 0
                                            : effective_rotary_dim;
            for (int verifier_rows : {2, 3, 4})
            {
                for (PositionRoute position_route : position_routes)
                {
                    const char *position_route_label =
                        position_route == PositionRoute::ContiguousDeviceScalar
                            ? "contiguous_device_scalar"
                            : "explicit_device_rows";
                    SCOPED_TRACE(
                        std::string(backend_label) + " format=" + format_label +
                        " M=" + std::to_string(verifier_rows) +
                        " rotary=" + std::to_string(effective_rotary_dim) +
                        " route=" + position_route_label);

                    Kernel kernel(0);
                    kernel.setGPUStream(stream);
                    const auto requirements = kernel.getWorkspaceRequirements(4);
                    DeviceWorkspaceManager workspace(
                        device, requirements.total_bytes_with_alignment() + 4096);
                    ASSERT_TRUE(workspace.allocate(requirements));
                    kernel.bindWorkspace(&workspace);

                    const std::vector<size_t> q_shape = {
                        static_cast<size_t>(verifier_rows), q_cols};
                    const std::vector<size_t> k_shape = {
                        static_cast<size_t>(verifier_rows), k_cols};
                    auto q_source = makeNativeTensor<Precision>(q_shape, q_values.data());
                    auto k_source = makeNativeTensor<Precision>(k_shape, k_values.data());
                    std::vector<uint8_t> serial_q(q_source->size_bytes());
                    std::vector<uint8_t> serial_k(k_source->size_bytes());
                    std::memcpy(serial_q.data(), q_source->raw_data(), serial_q.size());
                    std::memcpy(serial_k.data(), k_source->raw_data(), serial_k.size());
                    const size_t q_row_bytes = serial_q.size() / verifier_rows;
                    const size_t k_row_bytes = serial_k.size() / verifier_rows;

                    // The serial witness always enters the real M=1 contiguous
                    // device-scalar kernel, exactly as graph-captured decode does.
                    for (int row = 0; row < verifier_rows; ++row)
                    {
                        auto q_row = makeNativeTensor<Precision>(
                            {1, q_cols}, q_values.data());
                        auto k_row = makeNativeTensor<Precision>(
                            {1, k_cols}, k_values.data());
                        std::memcpy(
                            q_row->raw_mutable_data(),
                            serial_q.data() + static_cast<size_t>(row) * q_row_bytes,
                            q_row_bytes);
                        std::memcpy(
                            k_row->raw_mutable_data(),
                            serial_k.data() + static_cast<size_t>(row) * k_row_bytes,
                            k_row_bytes);
                        ASSERT_TRUE(q_row->ensureOnDevice(device, stream));
                        ASSERT_TRUE(k_row->ensureOnDevice(device, stream));
                        const int position =
                            position_route == PositionRoute::ContiguousDeviceScalar
                                ? 101 + row
                                : explicit_positions[static_cast<size_t>(row)];
                        kernel.setDynamicPosOffset(position);
                        ASSERT_TRUE(kernel.apply_tensor(
                            q_row.get(), k_row.get(), nullptr,
                            1, n_heads, n_kv_heads, head_dim, rope_theta,
                            nullptr, device.toKernelDeviceIndex(), position,
                            rotary_argument));
                        runtime.copyDeviceToHost(
                            serial_q.data() + static_cast<size_t>(row) * q_row_bytes,
                            q_row->gpu_data_ptr(), q_row_bytes, stream);
                        runtime.copyDeviceToHost(
                            serial_k.data() + static_cast<size_t>(row) * k_row_bytes,
                            k_row->gpu_data_ptr(), k_row_bytes, stream);
                        runtime.synchronize(stream);
                    }

                    auto q_grouped = makeNativeTensor<Precision>(q_shape, q_values.data());
                    auto k_grouped = makeNativeTensor<Precision>(k_shape, k_values.data());
                    ASSERT_TRUE(q_grouped->ensureOnDevice(device, stream));
                    ASSERT_TRUE(k_grouped->ensureOnDevice(device, stream));
                    runtime.synchronize(stream);

                    const int *grouped_positions = nullptr;
                    if (position_route == PositionRoute::ContiguousDeviceScalar)
                    {
                        kernel.setDynamicPosOffset(101);
                    }
                    else
                    {
                        kernel.setDynamicPositionIds(
                            explicit_positions.data(), verifier_rows);
                        grouped_positions = explicit_positions.data();
                    }
                    runtime.synchronize(stream);
                    PerfStatsCollector::reset();
                    ASSERT_TRUE(kernel.apply_tensor(
                        q_grouped.get(), k_grouped.get(), grouped_positions,
                        verifier_rows, n_heads, n_kv_heads, head_dim, rope_theta,
                        nullptr, device.toKernelDeviceIndex(), 101,
                        rotary_argument));

                    std::vector<uint8_t> grouped_q(q_grouped->size_bytes());
                    std::vector<uint8_t> grouped_k(k_grouped->size_bytes());
                    runtime.copyDeviceToHost(
                        grouped_q.data(), q_grouped->gpu_data_ptr(), grouped_q.size(), stream);
                    runtime.copyDeviceToHost(
                        grouped_k.data(), k_grouped->gpu_data_ptr(), grouped_k.size(), stream);
                    runtime.synchronize(stream);
                    expectNativeBytesEqual(
                        grouped_q, serial_q, std::string(format_label) + " grouped Q");
                    expectNativeBytesEqual(
                        grouped_k, serial_k, std::string(format_label) + " grouped K");

                    const auto records = PerfStatsCollector::snapshot(
                        {std::string("kernel.") + counter_name});
                    ASSERT_EQ(records.size(), 1u)
                        << PerfStatsCollector::summaryString(
                               {std::string("kernel.") + counter_name}, 20);
                    const auto &record = records.front();
                    EXPECT_EQ(record.count, 1u);
                    EXPECT_EQ(tag(record, "tensor_format"), format_label);
                    EXPECT_EQ(tag(record, "verifier_rows"), std::to_string(verifier_rows));
                    EXPECT_EQ(tag(record, "rotary_dim"), std::to_string(effective_rotary_dim));
                    EXPECT_EQ(tag(record, "position_route"), position_route_label);
                    EXPECT_EQ(tag(record, "capture_mode"), "direct");
                    EXPECT_EQ(tag(record, "invocation_policy"), "single_grouped_launch");
                }
            }
        }
    }

    /** @brief Execute the complete symmetric native-format matrix. */
    template <template <ActivationPrecision> class KernelTemplate, typename Runtime>
    void runAllFormats(
        Runtime &runtime,
        DeviceId device,
        void *stream,
        const char *backend_label,
        const char *counter_name)
    {
        ScopedPerfStats perfstats;
        runNativeFormat<ActivationPrecision::FP32,
                        KernelTemplate<ActivationPrecision::FP32>>(
            runtime, device, stream, backend_label, counter_name, "FP32", true);
        runNativeFormat<ActivationPrecision::BF16,
                        KernelTemplate<ActivationPrecision::BF16>>(
            runtime, device, stream, backend_label, counter_name, "BF16", false);
        runNativeFormat<ActivationPrecision::FP16,
                        KernelTemplate<ActivationPrecision::FP16>>(
            runtime, device, stream, backend_label, counter_name, "FP16", false);
    }
} // namespace llaminar2::test::gpu_rope_verifier
