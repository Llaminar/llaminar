/**
 * @file MoEProjectionPipeline.h
 * @brief Declarative lowering of an owner-local gate/up, distributed-down MoE transaction.
 *
 * One participant owns one local DAG. Both exchanges are explicit graph nodes;
 * compute stages never call another device or communicator. On proven no-P2P
 * domains the intermediate exchange uses counted TransferEngine channels;
 * enabled native P2P and the column publication retain NCCL/RCCL. The
 * existing runtime placement table supplies gate/up ownership and the immutable
 * fixed-down bank supplies every local output slice. Both phases reuse the
 * router's graph-local kernel/workspace, in a strictly ordered lifetime.
 */
#pragma once

#include "execution/moe/MoEOverlayFixedDownProjectionBank.h"
#include "execution/moe/MoEOverlayPreparedExpertPayload.h"
#include "execution/moe/MoEGroupedPlanDemand.h"
#include "execution/moe/DeviceMoEOverlayServiceTelemetry.h"
#include "memory/BufferId.h"
#include "TPAllreduceStage.h"

#include <memory>
#include <string>
#include <vector>

namespace llaminar2
{
    class ComputeGraph;
    class DeviceMoERuntimeTable;
    class ILocalTPContext;
    class DeviceCountedAllGather;
    class ITensor;
    struct MoERoutedPipelineKernelOwner;
    struct ActivationBuffers;

    /** @brief One arena identity and its participant-local tensor; neither owns storage. */
    struct MoEProjectionTensorBinding
    {
        ITensor *tensor = nullptr;
        BufferId id = BufferId::HIDDEN_STATE;
    };

    /**
     * @brief Complete frozen inputs to a native-domain projection graph.
     *
     * The caller admits/materializes the buffer BOM before lowering and retains
     * runtime/communicator lifetimes through graph retirement. The builder
     * retains prepared engine lifetimes, never a copied live owner map. This
     * boundary intentionally accepts only a native homogeneous domain whose
     * runtime participant coordinates equal its communicator coordinates.
     * Cross-tier routing requires its own explicit domain mapping, not ordinal
     * guessing or an implicit host transport.
     */
    struct MoEProjectionPipelineParams
    {
        std::shared_ptr<const MoEOverlayFixedDownProjectionBank> fixed_down;
        std::vector<MoEOverlayPreparedExpertPayload> initial_gate_up;
        std::shared_ptr<MoERoutedPipelineKernelOwner> kernel_owner;
        DeviceMoERuntimeTable *runtime = nullptr;
        ILocalTPContext *tp = nullptr;
        /// Frozen no-P2P domain fabric; absent only for the declared native transport.
        std::shared_ptr<DeviceCountedAllGather> counted_exchange;
        int rows = 0;
        std::optional<DeviceRowRange> live_rows; ///< Exact captured query prefix; ownership-sparse packets use their own counts.
        int top_k = 0;
        MoEGroupedPlanDemand demand = MoEGroupedPlanDemand::None;
        MoEOverlayServicePhaseHint service_phase = MoEOverlayServicePhaseHint::Auto;
        std::string prefix; ///< Unique model/layer graph identity.
        std::string router_node; ///< Existing router is the sole input producer.

        MoEProjectionTensorBinding hidden;
        MoEProjectionTensorBinding routing_indices;
        MoEProjectionTensorBinding routing_weights;
        MoEProjectionTensorBinding local_packet;
        MoEProjectionTensorBinding gathered_packets;
        MoEProjectionTensorBinding local_route_columns; ///< Independent full-K dots, in original router order.
        MoEProjectionTensorBinding local_columns;
        MoEProjectionTensorBinding gathered_columns;
        MoEProjectionTensorBinding output;

        /**
         * @brief Bind common activation roles and extension buffers from one admitted arena.
         * @param buffers Graph-local activation view, including any MTP role remaps.
         *
         * Common roles such as NORMALIZED have named fields; get() addresses
         * extension buffers only. Keeping this distinction in the reusable
         * builder prevents model declarations from accidentally binding null
         * common inputs or ignoring sidecar buffer identities.
         */
        void bindArena(const ActivationBuffers &buffers);
    };

    /**
     * @brief Append a complete six-node projection transaction without device work.
     * @param graph Participant-local model graph with the declared router already present.
     * @param params Prepared ownership plus exact arena bindings and native membership.
     * @return The row-major output publication node, for downstream graph dependencies.
     * @throws std::invalid_argument For missing, mismatched or insufficient bindings.
     * @throws std::logic_error For a foreign fixed bank or unsupported participant mapping.
     *
     * The sequence is gate/up+pack → explicit exchange → regroup/import/down dots →
     * ordered local route fold → native column gather → row-major assembly.
     * Independent down dots can run concurrently; only their short top-k fold
     * is serial. Consumer regrouping reads the
     * original router weights, never locally filtered runtime weights. Native
     * graph edges protect workspace reuse; no second group/activation bank or
     * host completion handshake is introduced.
     */
    std::string appendMoEProjectionPipeline(ComputeGraph &graph, MoEProjectionPipelineParams params);

    /**
     * @brief Overlap an unchanged full-row shared sum with routed down math.
     * @param graph Participant-local graph with both declared producers.
     * @param projection_prefix Prefix supplied to appendMoEProjectionPipeline.
     * @param shared_sum_node Existing native TPAllreduceStage and its one shared producer.
     * @throws std::invalid_argument For a foreign pipeline, sum, or unsafe event window.
     *
     * Serial decode and grouped verification retain their original allreduce,
     * gate and add arithmetic. Only its scheduling moves. The intermediate
     * exchange must finish before this sum is submitted; the final column
     * gather must follow the sum's join on every participant. Local down math
     * occupies that interval without touching shared partials or sum scratch.
     */
    void overlapMoEProjectionSharedAllreduce(ComputeGraph &graph,
        const std::string &projection_prefix, const std::string &shared_sum_node);

    /** @brief Shared-branch declaration for final publication by column owners. */
    struct MoEProjectionSharedColumns
    {
        TPAllreduceStage::Params reduction; ///< Original native sum policy; no sidebands/canonical rank banks.
        std::string producer; ///< Existing shared-FFN partial producer.
        ITensor *gate = nullptr; ///< Prepared full-width FP32 shared gate vector.
        const int *active_rows = nullptr; ///< Optional graph-owned effective row count.
        MoEProjectionTensorBinding combined_output; ///< Final full row, normally ATTN_PROJ.
    };

    /**
     * @brief Complete a projection DAG by summing shared output only to its column owners.
     * @param graph Existing six-node projection pipeline and shared-FFN producer.
     * @param projection_prefix Exact prefix previously passed to appendMoEProjectionPipeline.
     * @param shared Original shared reduction/gate policy and final output bank.
     * @return Final full-row assembly identity for downstream consumers.
     * @throws std::invalid_argument For mismatched geometry, aliases, sidebands or non-native arithmetic.
     *
     * The existing gathered-column bank temporarily packs shared partials.
     * Reduce-scatter overlaps routed down work using the standard paired event
     * lifecycle. After the ordered route fold, gate/combine writes into dead
     * route-addend storage and the original column gather publishes combined
     * columns. No new physical storage, extra gather or hidden collective exists.
     */
    std::string finalizeMoEProjectionSharedColumns(ComputeGraph &graph,
        const std::string &projection_prefix, MoEProjectionSharedColumns shared);
} // namespace llaminar2
