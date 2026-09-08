/**
 * @file GPUVramPreflight.h
 * @brief Single accounting authority for GPU weight-load memory admission.
 *
 * Automatic placement and the concrete upload pipeline must reserve the same
 * persistent weights and transient upload ring. The typed policy and bill of
 * materials below are intentionally pure: early model planning and late device
 * allocation both call the same arithmetic rather than reconstructing it from
 * related environment settings. Unnamed safety reserves are deliberately not
 * representable; every admitted byte must have a concrete allocation owner.
 */

#pragma once

#include "GPUHostLoadPreflight.h"
#include "planning/PhysicalMemoryBOM.h"
#include "utils/DebugEnv.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

namespace llaminar2
{
    /** Device-allocation alignment shared with @ref WeightVRAMPool. */
    inline constexpr std::size_t kGPUWeightLoadAllocationAlignment = 256u;

    /** @brief Immutable policy inputs for one GPU's initial weight upload. */
    struct GPUWeightLoadMemoryPolicy
    {
        /** Number of independently resident device upload slots. */
        int staging_stream_count = 1;
        /** Total per-device staging cap; zero deliberately means unlimited. */
        size_t staging_budget_bytes = 0;
    };

    /** @brief Exact bounded upload-ring geometry before physical admission. */
    struct GPUWeightLoadMemoryGeometry
    {
        size_t maximum_source_bytes = 0;
        /** Logical payload capacity used by row-chunk scheduling. */
        size_t staging_slot_bytes = 0;
        /** Physical distance between slots after allocator alignment. */
        size_t staging_slot_stride_bytes = 0;
        /** Complete device allocation for every simultaneously resident slot. */
        size_t staging_bytes = 0;
        /** Exact pinned-host allocation before any allocator page rounding. */
        size_t host_staging_bytes = 0;
        int staging_stream_count = 0;
    };

    /**
     * @brief Round an allocation without permitting address-space overflow.
     * @param bytes Unaligned byte count.
     * @return Byte count aligned to the GPU weight-pool contract.
     */
    [[nodiscard]] inline size_t alignGPUWeightLoadAllocation(size_t bytes)
    {
        if (bytes == 0)
            return 0;
        constexpr size_t alignment = kGPUWeightLoadAllocationAlignment;
        if (bytes > std::numeric_limits<size_t>::max() - (alignment - 1u))
        {
            throw std::overflow_error(
                "GPU weight-load allocation alignment overflows size_t");
        }
        return (bytes + alignment - 1u) & ~(alignment - 1u);
    }

    /**
     * @brief Resolve the process-wide initial-load settings into a typed policy.
     * @param staging_budget_override Optional byte budget owned by a specialized
     *        caller; absent uses the configured per-GPU upload budget.
     * @return Policy consumed identically by planning and allocation.
     */
    [[nodiscard]] inline GPUWeightLoadMemoryPolicy
    configuredGPUWeightLoadMemoryPolicy(
        std::optional<size_t> staging_budget_override = std::nullopt)
    {
        const auto &env = debugEnv();
        const auto configured_budget =
            gpuPerDeviceLoadStagingBudgetBytes(
                env.rocm.repack_budget_mb);
        return {
            .staging_stream_count =
                std::clamp(env.rocm.repack_streams, 1, 8),
            .staging_budget_bytes = staging_budget_override.value_or(
                configured_budget.value_or(0)),
        };
    }

    /**
     * @brief Resolve the exact logical and physical upload-ring geometry.
     * @param maximum_source_bytes Largest raw source transaction loaded through
     *        any ring slot. A bounded policy caps each slot independently.
     * @param policy Typed upload-slot policy.
     * @return Geometry consumed unchanged by planning and WeightVRAMPool.
     * @throws std::invalid_argument for an impossible stream policy.
     * @throws std::overflow_error when any byte calculation exceeds `size_t`.
     */
    [[nodiscard]] inline GPUWeightLoadMemoryGeometry
    resolveGPUWeightLoadMemoryGeometry(
        size_t maximum_source_bytes,
        const GPUWeightLoadMemoryPolicy &policy)
    {
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
            if (policy.staging_budget_bytes > 0)
            {
                const size_t unaligned_per_stream =
                    policy.staging_budget_bytes / stream_count;
                const size_t physical_per_stream =
                    unaligned_per_stream &
                    ~(kGPUWeightLoadAllocationAlignment - 1u);
                if (physical_per_stream == 0)
                {
                    throw std::invalid_argument(
                        "GPU weight-load staging budget cannot hold one aligned slot per stream");
                }
                staging_slot_bytes = std::min(
                    maximum_source_bytes, physical_per_stream);
            }
            else
            {
                staging_slot_bytes = maximum_source_bytes;
            }
        }
        const size_t staging_slot_stride_bytes =
            alignGPUWeightLoadAllocation(staging_slot_bytes);
        const size_t staging_bytes = checkedMultiply(
            staging_slot_stride_bytes, stream_count, "staging ring");
        const size_t host_staging_bytes = checkedMultiply(
            staging_slot_bytes, stream_count, "pinned host ring");
        return {
            .maximum_source_bytes = maximum_source_bytes,
            .staging_slot_bytes = staging_slot_bytes,
            .staging_slot_stride_bytes = staging_slot_stride_bytes,
            .staging_bytes = staging_bytes,
            .host_staging_bytes = host_staging_bytes,
            .staging_stream_count = policy.staging_stream_count,
        };
    }

    /**
     * @brief Build the canonical physical GPU bill for initial weight loading.
     * @param resource Exact rank/device allocator observation.
     * @param planned_weight_bytes Unaligned terminal offset of the persistent
     *        weight pool.
     * @param geometry Upload geometry returned by
     *        @ref resolveGPUWeightLoadMemoryGeometry.
     * @return Universal typed BOM used directly for fit and certification.
     */
    [[nodiscard]] inline PhysicalMemoryBOM gpuWeightLoadMemoryBOM(
        PhysicalMemoryResource resource,
        size_t planned_weight_bytes,
        const GPUWeightLoadMemoryGeometry &geometry)
    {
        if (!resource.valid() || !resource.device.is_gpu())
        {
            throw std::invalid_argument(
                "GPU weight-load BOM requires a valid CUDA or ROCm physical resource");
        }
        PhysicalMemoryBOMBuilder builder(std::move(resource));
        builder
            .add(
                PhysicalMemoryOwner::PrimaryModelWeights,
                alignGPUWeightLoadAllocation(planned_weight_bytes))
            .add(
                PhysicalMemoryOwner::WeightLoadStaging,
                geometry.staging_bytes);
        return builder.build();
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
