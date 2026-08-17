/**
 * @file GPUVramPreflight.h
 * @brief Single accounting authority for GPU weight-load memory admission.
 *
 * Automatic placement and the concrete upload pipeline must reserve the same
 * persistent weights, transient upload ring, and allocator safety margin.  The
 * typed policy and bill of materials below are intentionally pure: early model
 * planning and late device allocation both call the same arithmetic rather
 * than reconstructing it from related environment settings.
 */

#pragma once

#include "GPUHostLoadPreflight.h"
#include "utils/DebugEnv.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

namespace llaminar2
{
    /** @brief Immutable policy inputs for one GPU's initial weight upload. */
    struct GPUWeightLoadMemoryPolicy
    {
        /** Number of independently resident device upload slots. */
        int staging_stream_count = 1;
        /** Total per-device staging cap; zero deliberately means unlimited. */
        size_t staging_budget_bytes = 0;
        /** Percentage of total VRAM retained for allocator/library overhead. */
        double safety_margin_percent = 0.0;
        /** Absolute lower bound for the safety reserve. */
        size_t minimum_safety_margin_bytes = 0;
        /** Explicit reserve used by runtime-arrival infrastructure. */
        std::optional<size_t> safety_margin_override_bytes;
    };

    /**
     * @brief Complete initial-load memory bill for one GPU.
     *
     * `required_bytes` includes every byte that competes for the observed free
     * VRAM: persistent prepared weights, all upload slots, and the safety
     * reserve.  Keeping the components alongside the total makes diagnostics
     * auditable without allowing callers to perform a second fit equation.
     */
    struct GPUWeightLoadMemoryBOM
    {
        size_t planned_weight_bytes = 0;
        size_t maximum_source_bytes = 0;
        size_t staging_slot_bytes = 0;
        size_t staging_bytes = 0;
        /** Persistent weights plus staging, before the independent reserve. */
        size_t load_bytes = 0;
        size_t safety_margin_bytes = 0;
        /** Complete admission requirement: load bytes plus safety reserve. */
        size_t required_bytes = 0;
        size_t free_vram_bytes = 0;
        size_t total_vram_bytes = 0;
        int staging_stream_count = 0;

        /** @return Whether the backend supplied a meaningful free-memory reading. */
        [[nodiscard]] bool hasMemoryObservation() const noexcept
        {
            return free_vram_bytes > 0;
        }

        /**
         * @return Whether the complete bill fits, or true when memory is unknown.
         *
         * Load infrastructure historically permits backends that cannot report
         * free VRAM. Capacity admission itself supplies a positive hardware
         * inventory and therefore never relies on the unknown-memory branch.
         */
        [[nodiscard]] bool fits() const noexcept
        {
            return !hasMemoryObservation() || required_bytes <= free_vram_bytes;
        }

        /** @return Free bytes left after the safety reserve, saturating at zero. */
        [[nodiscard]] size_t availableAfterSafetyReserve() const noexcept
        {
            return free_vram_bytes > safety_margin_bytes
                       ? free_vram_bytes - safety_margin_bytes
                       : 0;
        }
    };

    /**
     * @brief Resolve the process-wide initial-load settings into a typed policy.
     * @param staging_budget_override Optional byte budget owned by a specialized
     *        caller; absent uses the configured per-GPU upload budget.
     * @param safety_margin_override Optional exact safety reserve.
     * @return Policy consumed identically by planning and allocation.
     */
    [[nodiscard]] inline GPUWeightLoadMemoryPolicy
    configuredGPUWeightLoadMemoryPolicy(
        std::optional<size_t> staging_budget_override = std::nullopt,
        std::optional<size_t> safety_margin_override = std::nullopt)
    {
        constexpr size_t kBytesPerMiB = 1024ULL * 1024ULL;
        const auto &env = debugEnv();
        const auto configured_budget =
            gpuPerDeviceLoadStagingBudgetBytes(
                env.rocm.repack_budget_mb);
        return {
            .staging_stream_count =
                std::clamp(env.rocm.repack_streams, 1, 8),
            .staging_budget_bytes = staging_budget_override.value_or(
                configured_budget.value_or(0)),
            .safety_margin_percent =
                std::max(0.0, env.gpu_vram_preflight_margin_pct),
            .minimum_safety_margin_bytes =
                static_cast<size_t>(
                    std::max(0, env.gpu_vram_preflight_min_margin_mib)) *
                kBytesPerMiB,
            .safety_margin_override_bytes = safety_margin_override,
        };
    }

    /**
     * @brief Compute the authoritative persistent/staging/reserve memory bill.
     * @param planned_weight_bytes Persistent prepared-weight pool bytes.
     * @param maximum_source_bytes Largest raw source transaction loaded through
     *        any ring slot. A bounded policy caps each slot independently.
     * @param free_vram_bytes Current or admission-time available VRAM.
     * @param total_vram_bytes Physical VRAM used to derive percentage reserve.
     * @param policy Typed upload and safety policy.
     * @return Auditable bill whose `fits()` method owns the admission equation.
     * @throws std::invalid_argument for an impossible stream policy.
     * @throws std::overflow_error when any byte calculation exceeds `size_t`.
     */
    [[nodiscard]] inline GPUWeightLoadMemoryBOM gpuWeightLoadMemoryBOM(
        size_t planned_weight_bytes,
        size_t maximum_source_bytes,
        size_t free_vram_bytes,
        size_t total_vram_bytes,
        const GPUWeightLoadMemoryPolicy &policy)
    {
        const auto checkedAdd = [](size_t left, size_t right, const char *what)
        {
            if (right > std::numeric_limits<size_t>::max() - left)
            {
                throw std::overflow_error(
                    std::string("GPU weight-load ") + what +
                    " overflows size_t");
            }
            return left + right;
        };
        const auto checkedMultiply = [](
            size_t left, size_t right, const char *what)
        {
            if (left != 0 &&
                right > std::numeric_limits<size_t>::max() / left)
            {
                throw std::overflow_error(
                    std::string("GPU weight-load ") + what +
                    " overflows size_t");
            }
            return left * right;
        };

        if (maximum_source_bytes > 0 && policy.staging_stream_count <= 0)
        {
            throw std::invalid_argument(
                "GPU weight-load staging requires at least one stream");
        }

        const size_t stream_count = static_cast<size_t>(
            std::max(0, policy.staging_stream_count));
        size_t staging_slot_bytes = 0;
        if (maximum_source_bytes > 0)
        {
            staging_slot_bytes = policy.staging_budget_bytes > 0
                                     ? std::min(
                                           maximum_source_bytes,
                                           std::max<size_t>(
                                               1,
                                               policy.staging_budget_bytes /
                                                   stream_count))
                                     : maximum_source_bytes;
        }
        const size_t staging_bytes = checkedMultiply(
            staging_slot_bytes, stream_count, "staging ring");

        size_t safety_margin_bytes = 0;
        if (policy.safety_margin_override_bytes.has_value())
        {
            safety_margin_bytes = *policy.safety_margin_override_bytes;
        }
        else
        {
            const long double percentage =
                static_cast<long double>(
                    std::max(0.0, policy.safety_margin_percent)) /
                100.0L;
            const long double percentage_bytes =
                static_cast<long double>(total_vram_bytes) * percentage;
            if (percentage_bytes >
                static_cast<long double>(
                    std::numeric_limits<size_t>::max()))
            {
                throw std::overflow_error(
                    "GPU weight-load percentage safety reserve overflows size_t");
            }
            safety_margin_bytes = std::max(
                policy.minimum_safety_margin_bytes,
                static_cast<size_t>(percentage_bytes));
        }

        const size_t load_bytes = checkedAdd(
            planned_weight_bytes, staging_bytes, "weight plus staging bill");
        const size_t required_bytes = checkedAdd(
            load_bytes, safety_margin_bytes, "complete bill");
        return {
            .planned_weight_bytes = planned_weight_bytes,
            .maximum_source_bytes = maximum_source_bytes,
            .staging_slot_bytes = staging_slot_bytes,
            .staging_bytes = staging_bytes,
            .load_bytes = load_bytes,
            .safety_margin_bytes = safety_margin_bytes,
            .required_bytes = required_bytes,
            .free_vram_bytes = free_vram_bytes,
            .total_vram_bytes = total_vram_bytes,
            .staging_stream_count = policy.staging_stream_count,
        };
    }

    /** @return Configured initial-load safety reserve for legacy diagnostics. */
    [[nodiscard]] inline size_t gpuPipelineVramSafetyMarginBytes(
        size_t total_vram_bytes)
    {
        return gpuWeightLoadMemoryBOM(
                   /*planned_weight_bytes=*/0,
                   /*maximum_source_bytes=*/0,
                   /*free_vram_bytes=*/0,
                   total_vram_bytes,
                   configuredGPUWeightLoadMemoryPolicy())
            .safety_margin_bytes;
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
