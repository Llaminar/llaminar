/**
 * @file GPUVramPreflight.h
 * @brief Shared diagnostics for GPU weight residency preflight failures.
 */

#pragma once

#include <string>

namespace llaminar2
{
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
