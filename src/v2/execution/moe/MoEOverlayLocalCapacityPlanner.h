/**
 * @file MoEOverlayLocalCapacityPlanner.h
 * @brief Rank-local fixed-memory BOM builder for ExpertOverlay admission.
 *
 * Live routed experts are deliberately excluded here because the global
 * capacity resolver decides their quotas.  This planner prices everything
 * already fixed by one rank's production execution plan—dense/shared/global
 * weights, KV and recurrent state, graph activations, workspaces, and the
 * host-memory authority needed by background transport—against the same typed
 * `(world rank, device)` resources later consumed by capacity admission.
 */

#pragma once

#include "MoEOverlayActivationChannelPlan.h"
#include "MoEOverlayCapacityAdmission.h"
#include "MoEExpertOverlayExecutionPlan.h"
#include "execution/mpi_orchestration/DeviceInventory.h"
#include "execution/mpi_orchestration/RankExecutionPlan.h"
#include "loaders/GPUVramPreflight.h"
#include "planning/CapturedGraphMemoryEstimator.h"
#include "planning/MemoryPlan.h"
#include "planning/ModelMemoryProfile.h"

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace llaminar2
{
    /**
     * @brief Model-specific GPU upload contract priced during auto-capacity.
     *
     * Policy alone is insufficient because a bounded ring may shrink when all
     * source tensors are smaller than one configured slot. Bundling the largest
     * source transaction with the policy prevents a partially specified load
     * reservation from reaching the planner.
     */
    struct MoEOverlayGPUWeightLoadCapacityInput
    {
        GPUWeightLoadMemoryPolicy policy;
        size_t maximum_source_bytes = 0;
    };

    /**
     * @brief Resolve the physical native graph inventory for ExpertOverlay.
     * @param model_layer_count Number of main-model routed layers.
     * @param authority_execution Frozen host/device authority topology.
     * @param model_graph_identity_count Retained prefill/decode/bridge graphs.
     * @param auxiliary_native_executable_count Retained activation and MTP
     *        helper/controller graphs.
     * @return Valid inventory consumed by device-memory admission.
     * @throws std::invalid_argument for unresolved authority or bad geometry.
     * @throws std::overflow_error when the segment count exceeds size_t.
     *
     * Host-resident overlays contain an intentional CPU ticket boundary at
     * every routed layer. Their continuation graph therefore owns one captured
     * unit before each boundary plus one terminal unit. All-GPU device-owned
     * timelines remain one native executable per complete graph identity.
     */
    [[nodiscard]] CapturedGraphExecutableInventory
    resolveMoEOverlayCapturedGraphExecutableInventory(
        int model_layer_count,
        MoEOverlayAuthorityExecutionKind authority_execution,
        std::size_t model_graph_identity_count,
        std::size_t auxiliary_native_executable_count);

    /** @brief Complete immutable input for one rank's zero-routed-expert BOM. */
    struct MoEOverlayLocalCapacityPlannerInput
    {
        const ModelMemoryProfile *model_profile = nullptr;
        const RankExecutionPlan *rank_plan = nullptr;
        const MoERoutedExpertPlacementPlan *overlay_plan = nullptr;
        const RankInventory *rank_inventory = nullptr;
        /** Complete physical-node authority for distributed activation lanes. */
        const ClusterInventory *cluster_inventory = nullptr;
        /** Exclusive graph/command lifecycle assigned by overlay planning. */
        OverlayRankExecutionKind rank_execution_kind =
            OverlayRankExecutionKind::RelayOnly;
        /** Include CPU authority even when no tier participant computes on CPU. */
        bool require_host_memory_authority = false;
        /** Optional hard physical limits; absent means inventory capacity. */
        std::optional<std::size_t> max_gpu_memory_bytes;
        std::optional<std::size_t> max_cpu_memory_bytes;
        /** Candidate graph rows; zero selects the configured maximum shape. */
        int resident_graph_rows = 0;
        /** Exact max rows embedded by every retained activation transaction. */
        int activation_channel_row_capacity = 0;
        /** Main plus routed MTP graph families sharing each channel. */
        std::size_t activation_graph_family_count = 0;
        /** Exact physical forward/segment/helper inventory retained per GPU. */
        CapturedGraphExecutableInventory captured_graph_inventory;
        /** Required whenever this rank's physical resources include a GPU. */
        std::optional<MoEOverlayGPUWeightLoadCapacityInput>
            gpu_weight_load;
    };

    /** @brief Rank-local fixed BOM plus the exact graph-row shape it priced. */
    struct MoEOverlayLocalCapacityPlannerResult
    {
        int resident_graph_rows = 0;
        MemoryPlan fixed_memory_plan;
        /** Shared pure topology/BOM also consumed by transport preflight. */
        MoEOverlayActivationChannelPlan activation_channel_plan;
        std::vector<MoEOverlayBoundPhysicalMemoryBudget> physical_budgets;
    };

    /** @brief One continuation graph shard physically built by this rank. */
    struct MoEOverlayContinuationShard
    {
        DeviceId device = DeviceId::invalid();
        int shard_index = 0;
        int total_shards = 1;
        int first_layer = 0;
        int last_layer = -1;
    };

    /**
     * @brief Replace discovery-time GPU memory with a runtime-stable observation.
     *
     * CUDA and ROCm discovery may run before the production device context,
     * BLAS handles, streams, and collective resources exist. Expert capacity
     * must therefore use a second observation after those fixed runtime owners
     * have been created. This helper is the single typed mutation point for
     * that observation and deliberately treats both GPU backends identically.
     *
     * @param inventory Rank-local hardware inventory to update.
     * @param device Exact CUDA or ROCm device whose production context is live.
     * @param total_bytes Physical device memory reported by the live backend.
     * @param free_bytes Free memory after fixed runtime initialization.
     * @throws std::invalid_argument for a non-GPU device, invalid byte counts,
     *         or a device absent from the rank inventory.
     */
    void installMoEOverlayRuntimeGPUCapacityObservation(
        RankInventory &inventory,
        DeviceId device,
        std::size_t total_bytes,
        std::size_t free_bytes);

    /**
     * @brief Build exact fixed setup charges without assigning a live expert.
     *
     * The result is deterministic and performs no allocation. Every endpoint
     * device physically owned by this rank receives one resource authority.
     * Continuation devices use the real TP/PP/shard configuration; auxiliary
     * devices use the routed-participant execution role. Multiple logical
     * participants sharing a device remain one physical budget.
     */
    class MoEOverlayLocalCapacityPlanner final
    {
    public:
        /**
         * @brief Resolve exact continuation devices and tensor-shard identity.
         *
         * Both early capacity admission and final graph-memory validation use
         * this method. Keeping the translation in one authority prevents an
         * ExpertOverlay LocalTP continuation from being priced as one dense
         * device plus several expert-only devices.
         *
         * @param rank_plan Fully resolved production rank plan.
         * @param execution_kind Exclusive overlay rank lifecycle. Authority and
         *        peer kinds own continuation shards; follower/relay kinds do not.
         * @return Participant-local continuation shard descriptors.
         * @throws std::invalid_argument for malformed LocalPP boundaries.
         */
        [[nodiscard]] static std::vector<MoEOverlayContinuationShard>
        continuationShards(
            const RankExecutionPlan &rank_plan,
            OverlayRankExecutionKind execution_kind);

        /**
         * @brief Create a stable diagnostic identity for one physical resource.
         * @param world_rank Owning MPI rank.
         * @param device Rank-local CPU/CUDA/ROCm identity.
         * @return Opaque resource id used only for joins and diagnostics.
         */
        [[nodiscard]] static std::string physicalResourceId(
            int world_rank,
            DeviceId device);

        /**
         * @brief Price one rank's fixed production memory contract.
         * @param input Model, rank plan, bound topology, inventory, and limits.
         * @return Fixed MemoryPlanner BOM grouped by physical resource.
         * @throws std::invalid_argument for incomplete/mismatched topology.
         * @throws std::overflow_error when grouped byte arithmetic overflows.
         */
        [[nodiscard]] static MoEOverlayLocalCapacityPlannerResult plan(
            const MoEOverlayLocalCapacityPlannerInput &input);
    };
} // namespace llaminar2
