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
 *
 * Local and global pipeline stages share the original rank-owned prepared
 * source and PMA handoff. Global topology remains immutable metadata; remote
 * execution, activation transport and model-context reuse are not impersonated.
 */
#pragma once

#include "planning/MemoryPlanner.h"
#include "execution/mtp/DeviceGenerationExecutionPolicy.h"
#include "execution/mpi_orchestration/RankExecutionPlan.h"
#include "execution/mpi_orchestration/DeviceInventory.h"
#include "loaders/GPUVramPreflight.h"

namespace llaminar2
{
    struct GlobalPPTopology;
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
        /**
         * @brief Bind one actual local action from the canonical global pipeline.
         * @param topology Complete compiled topology with observed rank membership.
         * @param world_rank Actual owning rank; no remote GPU is projected locally.
         * @param stage_id Original declared stage ID, not a local vector position.
         * @return Exact local body and original same-node GPU boundary ownership,
         * including per-member completed live-byte native fanout metadata.
         * @throws std::invalid_argument For absent, foreign, non-CPU multi-rank TP or duplicate physical members.
         */
        [[nodiscard]] static PipelineStageTransferMemory forGlobalStage(
            const GlobalPPTopology &topology, int world_rank, int stage_id);
        /** @return Whether this physical endpoint owns mapped channel storage. */
        [[nodiscard]] bool requiresHostMemory(DeviceId device) const noexcept;
        /**
         * @brief Attach transport only to its actual physical endpoint.
         * @param config One matching continuation graph's memory input.
         * @param hosted_row_capacity Exact retained invocation rows; zero uses this graph's authored activation envelope.
         * @throws std::invalid_argument for foreign geometry, duplicate binding,
         *         or missing host authority required by a mapped channel owner.
         */
        void bind(DevicePlanConfig &config, int hosted_row_capacity = 0) const;
    private:
        /** @brief Only forStage() may construct an authenticated stage projection. */
        PipelineStageTransferMemory() = default;
        int world_rank_ = -1;
        int first_layer_ = -1;
        int last_layer_ = -1;
        std::vector<DeviceId> participants_;
        std::optional<CollectiveBackendType> native_backend_;
        std::vector<std::size_t> native_byte_endpoints_; ///< Exact per-member incidence of original native domain edges.
        std::vector<PipelineBoundarySide> captured_boundaries_;
        bool hosted_activation_bank_owner_ = false; ///< First actual local stage quotes the one original parent CPU bank.
        bool native_activation_fanout_ = false; ///< Actual receiving TP domain, quoted once on each member.
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

    /**
     * @brief Quote the original common CPU reader/consumer transaction for one complete device plan.
     * @param model Original native metadata inventory.
     * @param config Final source/shard/component ownership, after PP/TP projection.
     * @param policy Same original startup cap consumed by the common reader.
     * @param precision Same explicit destination intent consumed by WeightManager.
     * @param placement Original CPU source placement, not inferred from allocator ordinal.
     * @return Maximum serialized input/scratch root, absent for direct final writes only.
     */
    std::optional<NativeWeightLoadStagingGeometry> quoteCPUWeightSourceStaging(
        const ModelMemoryProfile& model,const DevicePlanConfig& config,
        GPUWeightLoadMemoryPolicy policy,WeightPrecision precision,CPUStoragePlacement placement);

    /** @brief Immutable observations and execution policies for one rank BOM. */
    struct RankMemoryPlanInputRequest
    {
        const ModelMemoryProfile &model; ///< Parsed GGUF metadata, no loaded weights.
        const RankExecutionPlan &plan; ///< Exact installed rank/shard/layer geometry.
        const RankInventory &inventory; ///< Observation for this same communicator rank.
        /** Fresh upload-ring geometry; absent only for certified retained preparation. */
        std::optional<GPUWeightLoadMemoryGeometry> weight_load_geometry;
        /** Original fresh common CPU source policy; absent only for certified retained preparation. */
        std::optional<GPUWeightLoadMemoryPolicy> native_weight_source_policy;
        /** Same destination representation selected by the original ModelContext/WeightManager. */
        WeightPrecision weight_precision = WeightPrecision::NATIVE;
        /** Original loader/factory CPU placement; required for a fresh CPU source transaction. */
        std::optional<CPUStoragePlacement> native_source_placement;
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
