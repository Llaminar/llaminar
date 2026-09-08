/**
 * @file GPUHostLoadPreflight.h
 * @brief Host-memory sizing rules for bounded GPU model loading.
 *
 * GPU model loading has two materially different host-memory regimes. Mapped
 * GGUF tensors can be read directly into the bounded pinned upload ring, while
 * CPU targets and explicit non-mmap loads materialize ordinary host storage.
 * Keeping the arithmetic in this small pure helper makes the preflight contract
 * independently testable and prevents UI or runner configuration from becoming
 * a second source of truth.
 */

#pragma once

#include "planning/PhysicalMemoryBOM.h"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <utility>

namespace llaminar2
{
    /**
     * @brief Convert the DebugEnv startup-staging setting into one GPU's budget.
     *
     * The environment value is intentionally per device: every GPU owns an
     * independent pinned upload ring and matching device scratch region.  This
     * helper accepts no device-count argument, which prevents callers from
     * accidentally dividing the configured cap across concurrently loading
     * devices.  Zero and negative values retain the explicit unlimited mode.
     *
     * @param configured_mebibytes Per-GPU `LLAMINAR_GPU_LOAD_STAGING_MB` value.
     * @return Per-GPU byte cap, or `std::nullopt` for unlimited staging.
     */
    [[nodiscard]] inline std::optional<size_t> gpuPerDeviceLoadStagingBudgetBytes(
        int configured_mebibytes) noexcept
    {
        if (configured_mebibytes <= 0)
            return std::nullopt;

        constexpr size_t kBytesPerMiB = 1024ULL * 1024ULL;
        return static_cast<size_t>(configured_mebibytes) * kBytesPerMiB;
    }

    /**
     * @brief Build the canonical CPU-RAM bill for one model-load operation.
     *
     * GPU targets own a bounded transient upload working set. CPU targets keep
     * their eagerly materialized tensors as the primary model-weight owner.
     * Both are charged to the same rank-local CPU allocator used by serving
     * prefix tiers and ExpertOverlay participant banks.
     *
     * @param host_resource Exact host allocator observation.
     * @param eager_weight_bytes Sum of selected GGUF source payloads.
     * @param target_is_gpu Whether the terminal prepared weights live on GPU.
     * @param uses_mmap Whether GGUF payloads remain file-backed.
     * @param pinned_ring_bytes Exact resolved pinned upload-ring allocation.
     * @return Universal typed BOM; callers certify it before materialization.
     */
    [[nodiscard]] inline PhysicalMemoryBOM hostWeightLoadMemoryBOM(
        PhysicalMemoryResource host_resource,
        size_t eager_weight_bytes,
        bool target_is_gpu,
        bool uses_mmap,
        size_t pinned_ring_bytes)
    {
        if (!host_resource.valid() || !host_resource.device.is_cpu())
        {
            throw std::invalid_argument(
                "Host weight-load BOM requires a valid CPU physical resource");
        }
        PhysicalMemoryBOMBuilder builder(std::move(host_resource));
        if (target_is_gpu)
        {
            if (!uses_mmap)
            {
                builder.add(
                    PhysicalMemoryOwner::ModelSourcePayload,
                    eager_weight_bytes);
            }
            builder.add(
                PhysicalMemoryOwner::WeightLoadStaging,
                pinned_ring_bytes);
        }
        else
        {
            builder.add(
                PhysicalMemoryOwner::PrimaryModelWeights,
                eager_weight_bytes);
        }
        return builder.build();
    }
} // namespace llaminar2
