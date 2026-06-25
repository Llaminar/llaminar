/**
 * @file GPUVramPreflight.h
 * @brief Shared diagnostics for GPU weight residency preflight failures.
 */

#pragma once

#include "utils/DebugEnv.h"

#include <algorithm>
#include <cstddef>
#include <string>

namespace llaminar2
{
    inline size_t gpuPipelineVramSafetyMarginBytes(size_t total_vram_bytes)
    {
        const auto &env = debugEnv();
        const auto margin_fraction =
            std::max(0.0L, static_cast<long double>(env.gpu_vram_preflight_margin_pct)) / 100.0L;
        const size_t percent_margin =
            static_cast<size_t>(static_cast<long double>(total_vram_bytes) * margin_fraction);
        const size_t min_margin =
            static_cast<size_t>(std::max(0, env.gpu_vram_preflight_min_margin_mib)) *
            static_cast<size_t>(1024) * static_cast<size_t>(1024);
        return std::max<size_t>(min_margin, percent_margin);
    }

    inline size_t gpuDirectRebalanceVramSafetyMarginBytes()
    {
        /*
         * Runtime GPU-direct expert arrivals allocate an exact packed-weight
         * pool and no upload staging. Keep a small allocator cushion without
         * requiring the large initial-load reserve on already-resident models.
         */
        return static_cast<size_t>(16) * static_cast<size_t>(1024) * static_cast<size_t>(1024);
    }

    inline std::string gpuPipelineVramPreflightMitigations(
        bool weight_streaming_enabled,
        bool includes_resident_moe_experts)
    {
        if (weight_streaming_enabled)
        {
            if (includes_resident_moe_experts)
            {
                return "LLAMINAR_WEIGHT_STREAMING=1 is already enabled, but resident MoE expert streaming is not active for this GPU pipeline path; use a smaller model, reduce context/KV cache pressure, or reduce resident experts.";
            }

            return "LLAMINAR_WEIGHT_STREAMING=1 is already enabled, but this GPU pipeline path still requires the planned resident weights to fit; use a smaller model, reduce context/KV cache pressure, or reduce resident experts.";
        }

        if (includes_resident_moe_experts)
        {
            return "use a smaller model, reduce context/KV cache pressure, or reduce resident experts.";
        }

        return "set LLAMINAR_WEIGHT_STREAMING=1, use a smaller model, reduce context/KV cache pressure, or reduce resident experts.";
    }
} // namespace llaminar2
