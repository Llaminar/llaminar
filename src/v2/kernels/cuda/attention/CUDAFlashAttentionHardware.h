/**
 * @file CUDAFlashAttentionHardware.h
 * @brief Checked physical bindings for CUDA attention preparation and launch.
 *
 * This projection borrows admitted immutable hardware facts. It owns no device
 * cache, context, allocation or loaded-function state. Physical scheduling keeps
 * the existing canonical arithmetic partitions; graph replay never re-enters
 * these host helpers. The same bindings serve native callers and device-free
 * replays, including visible ordinals beyond the former eight-device caches.
 */
#pragma once

#include "backends/cuda/CUDAHardwareMetadata.h"
#include "kernels/attention/AttentionWorkspaceContract.h"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace llaminar2::cuda::fa2_policy
{
    /** @brief Immutable projection consumed by the existing compiled FA2 launchers. */
    struct CUDAFA2HardwareBinding
    {
        int sm_major = 0; ///< Native compute-capability major, not a program qualification.
        int sm_minor = 0; ///< Native compute-capability minor.
        int sm_count = 0; ///< Exact visible participant's multiprocessor count.
        int max_smem_optin = 0; ///< Checked native opt-in per-block shared-memory limit.
    };

    /**
     * @brief Bind the installed warp-32 attention family to one admitted record.
     * @param hardware Exact record selected by the caller's authenticated ordinal.
     * @return Physical inputs only; loaded function resources retain their own lifetime.
     * @throws std::invalid_argument On missing limits or an incompatible warp contract.
     */
    [[nodiscard]] inline CUDAFA2HardwareBinding bindCUDAFA2Hardware(
        const CUDADeviceHardware &hardware)
    {
        if (hardware.sm_major <= 0 || hardware.sm_minor < 0 || hardware.sm_count <= 0 ||
            hardware.native_warp_width != 32 || hardware.maximum_threads_per_block < 256 ||
            hardware.shared_memory_optin_per_block == 0 ||
            hardware.shared_memory_optin_per_block >
                static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw std::invalid_argument("CUDA attention requires admitted warp-32 physical limits");
        return {.sm_major = hardware.sm_major, .sm_minor = hardware.sm_minor,
                .sm_count = hardware.sm_count,
                .max_smem_optin = static_cast<int>(hardware.shared_memory_optin_per_block)};
    }

    /**
     * @brief Preserve the fixed-capacity decode split rule with exact physical ownership.
     * @param hardware Immutable facts for this graph's actual device.
     * @param kv_capacity Positive admitted KV row capacity, never the changing live count.
     * @param query_heads Positive participant-local query-head count.
     * @return Existing bounded split envelope used by scalar and grouped decode.
     * @throws std::invalid_argument On invalid geometry or hardware facts.
     *
     * Four blocks per SM, the 16-key minimum and the 32-split workspace bound
     * retain the existing launch policy. Wide integer arithmetic prevents an
     * invalid overflow from manufacturing a different envelope at larger counts.
     */
    [[nodiscard]] inline int selectCUDAFlashDecodeSplitEnvelope(
        const CUDADeviceHardware &hardware, int kv_capacity, int query_heads)
    {
        const auto physical = bindCUDAFA2Hardware(hardware);
        if (kv_capacity <= 0 || query_heads <= 0)
            throw std::invalid_argument("CUDA decode split planning requires positive capacity and heads");
        if (kv_capacity == 1)
            return 1;
        constexpr std::int64_t resident_blocks_per_sm = 4;
        constexpr int minimum_keys_per_split = 16;
        const auto desired = resident_blocks_per_sm * physical.sm_count / query_heads;
        const auto capacity_bound = std::max(kv_capacity / minimum_keys_per_split, 1);
        return static_cast<int>(std::clamp<std::int64_t>(
            desired, 1, std::min(capacity_bound, attention_workspace::kMaximumDecodeSplits)));
    }
}
