/**
 * @file RankMemoryPlanInputs.h
 * @brief Shared ordinary-execution BOM inputs for runtime admission and planning.
 *
 * The rank compiler owns placement and sharding; the physical authority owns
 * byte admission. This boundary only assembles their typed inputs, including
 * every retained MTP, prefix, snapshot and graph family. Frontends cannot omit
 * those families by maintaining another lightweight device-plan constructor.
 */
#pragma once

#include "planning/MemoryPlanner.h"
#include "execution/mpi_orchestration/RankExecutionPlan.h"
#include "execution/mpi_orchestration/DeviceInventory.h"
#include "loaders/GPUVramPreflight.h"

namespace llaminar2
{
    /** @brief Immutable observations and execution policies for one rank BOM. */
    struct RankMemoryPlanInputRequest
    {
        const ModelMemoryProfile &model; ///< Parsed GGUF metadata, no loaded weights.
        const RankExecutionPlan &plan; ///< Exact installed rank/shard/layer geometry.
        const RankInventory &inventory; ///< Observation for this same communicator rank.
        /** Fresh upload-ring geometry; absent only for certified retained preparation. */
        std::optional<GPUWeightLoadMemoryGeometry> weight_load_geometry;
        /** Explicit diagnostics only; ordinary production has no snapshot owner. */
        std::optional<GraphSnapshotMemoryCapacity> snapshot_capacity;
        /** Present means captured serving; an empty declared ladder is invalid. */
        std::optional<std::vector<int>> captured_prefill_buckets;
        /** Explicit physical limits, shared with ExpertOverlay and runtime admission. */
        std::optional<std::size_t> max_gpu_memory_bytes;
        std::optional<std::size_t> max_cpu_memory_bytes;
    };

    /**
     * @brief Assemble the complete ordinary rank memory plan without allocation.
     * @param request Exact model, execution plan, hardware and retained-family policy.
     * @return Per-device BOM inputs submitted unchanged to MemoryPlanner/PMA.
     * @throws std::invalid_argument for missing resources or inconsistent geometry.
     *
     * ExpertOverlay has its own topology-wide BOM input resolver and must not
     * be priced again here. Retained-preparation credits remain authenticated by
     * the runner's existing WeightManager/workspace lifetime proof before PMA
     * admits these inputs; a planning frontend cannot claim a retained credit.
     */
    std::vector<DevicePlanConfig> buildRankMemoryPlanInputs(
        const RankMemoryPlanInputRequest &request);
}
