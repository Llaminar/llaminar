/**
 * @file RankMemoryPlanInputs.h
 * @brief Shared ordinary-execution BOM inputs for runtime admission and planning.
 *
 * The rank compiler owns placement and sharding; the physical authority owns
 * byte admission. This boundary only assembles their typed inputs, including
 * every retained MTP, prefix, snapshot and graph family. Frontends cannot omit
 * those families by maintaining another lightweight device-plan constructor.
 * Compute membership retains CPU NUMA identity; only storage ownership projects
 * those workers onto their shared host physical resource.
 */
#pragma once

#include "planning/MemoryPlanner.h"
#include "execution/mtp/DeviceGenerationExecutionPolicy.h"
#include "execution/mpi_orchestration/RankExecutionPlan.h"
#include "execution/mpi_orchestration/DeviceInventory.h"
#include "loaders/GPUVramPreflight.h"

namespace llaminar2
{
    /**
     * @brief Immutable transfer ownership of one authored local pipeline stage.
     *
     * The authored domain selects the leader before physical storage projection.
     * CPU NUMA workers retain independent compute shards and one host allocator;
     * dense replication may label every weight view shard zero without multiplying
     * channel owners. Ordinary and ExpertOverlay planning use this projection.
     */
    class PipelineStageTransferMemory final
    {
    public:
        /**
         * @brief Resolve native or captured transport from the complete parent.
         * @param parent Authored rank pipeline with resolved device membership.
         * @param stage Index in its ordered stage inventory.
         * @return Transfer policy bound to this exact rank, interval and members.
         * @throws std::invalid_argument for incomplete or inconsistent topology.
         */
        [[nodiscard]] static PipelineStageTransferMemory forStage(const RankExecutionPlan &parent, size_t stage);
        /** @return Whether this physical endpoint owns mapped channel storage. */
        [[nodiscard]] bool requiresHostMemory(DeviceId device) const noexcept;
        /**
         * @brief Attach transport only to its actual physical endpoint.
         * @param config One matching continuation graph's memory input.
         * @throws std::invalid_argument for foreign geometry, duplicate binding,
         *         or missing host authority required by a mapped channel owner.
         */
        void bind(DevicePlanConfig &config) const;
    private:
        /** @brief Only forStage() may construct an authenticated stage projection. */
        PipelineStageTransferMemory() = default;
        int world_rank_ = -1;
        int first_layer_ = -1;
        int last_layer_ = -1;
        std::vector<DeviceId> participants_;
        std::optional<CollectiveBackendType> native_backend_;
        std::vector<PipelineBoundarySide> captured_boundaries_;
    };

    /**
     * @brief Authenticate explicit ngram distribution against the compiled rank declaration.
     * @param profile Original model metadata; no payload or backend query.
     * @param plan Exact rank TP/PP declaration in communicator order.
     * @return None for replicated policy; the checked full-main local TP provenance otherwise.
     * @throws std::invalid_argument For unsupported storage, source or topology before preparation.
     */
    [[nodiscard]] ImmutableRowTableTPTopology immutableRowTableTPTopology(
        const ModelMemoryProfile &profile,const RankExecutionPlan &plan);

    /**
     * @brief Project actual rank TP/PP/domain membership into one immutable boundary.
     * @param plan Authored rank and complete local pipeline/domain participants.
     * @return Native domain for one backend, hosted boundary for mixed members.
     *
     * Inventory-only devices are excluded. Remote GlobalPP MTP has no complete
     * captured generation implementation and is not introduced by this projection.
     */
    [[nodiscard]] DeviceGenerationExecutionDomain declaredRankDeviceGenerationDomain(
        const RankExecutionPlan &plan);

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
