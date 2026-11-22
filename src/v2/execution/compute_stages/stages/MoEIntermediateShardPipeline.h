/**
 * @file MoEIntermediateShardPipeline.h
 * @brief Participant-local routed FFN lowering over original source-native F shards.
 *
 * This declaration owns neither placement nor storage. The acquired device
 * runtime bank selects current experts; the closed family authenticates their
 * original F interval. Local gate/up/down and router-slot folding produce an H
 * partial. A caller must combine its local shared branch and declare one explicit
 * FFN sum before publishing a complete output. No intermediate crosses a device
 * boundary, and this partial operation supplies no complete-FFN timing evidence.
 */
#pragma once
#include "execution/moe/GPUExpertIntermediateTiling.h"
#include "execution/config/RoutedExpertPolicy.h"
#include "execution/moe/MoEGroupedPlanDemand.h"
#include "execution/moe/DeviceMoEOverlayServiceTelemetry.h"
#include "kernels/common/DeviceRowRange.h"
#include "kernels/common/NativeTileCompletion.h"
#include "memory/BufferId.h"
#include "TPAllreduceStage.h"
#include <memory>
#include <optional>
#include <string>

namespace llaminar2
{
    class ComputeGraph;
    class DeviceMoERuntimeTable;
    class FrozenModelWeightSet;
    class TensorParallelConfig;
    class ITensor;
    class MoEOverlayFragmentFamily;
    class PlanningGPUExpertMeasurement;
    struct MoEOverlayParticipantLayerBank;
    struct MoERoutedPipelineKernelOwner;

    /** @brief Arena identity and borrowed tensor; admission and lifetime belong to the caller. */
    struct MoEIntermediateShardTensorBinding
    {
        ITensor *tensor = nullptr;
        BufferId id = BufferId::HIDDEN_STATE;
    };

    /** @brief Immutable row execution shape, independent of the number of currently live rows. */
    enum class MoEIntermediateShardRowMode { Decode, Grouped };

    /** @brief Frozen participant-local inputs; mutable routing and demand remain device-owned. */
    struct MoEIntermediateShardPipelineParams
    {
        std::shared_ptr<const MoEOverlayFragmentFamily> family;
        DeviceMoERuntimeTable *runtime = nullptr;
        std::shared_ptr<MoERoutedPipelineKernelOwner> kernel_owner;
        int rows = 0;
        int top_k = 0;
        MoEIntermediateShardRowMode row_mode = MoEIntermediateShardRowMode::Grouped;
        LocalGPUExpertKernelPolicy verifier_kernel_policy = LocalGPUExpertKernelPolicy::Generic;
        MoEGateUpPublicationPolicy gate_up_publication=MoEGateUpPublicationPolicy::Staged; ///< Explicit ordinary native M1 tile publication; other paths remain Staged.
        MoEDownPublicationPolicy down_publication=MoEDownPublicationPolicy::Staged; ///< Explicit ordinary native M1 down/route completion; other paths remain Staged.
        RoutedExpertRowExecutionPolicy row_policy = RoutedExpertRowExecutionPolicy::IntermediateShardLocal;
        /// Only grouped-plan publication owns this side effect; decode routing already counts demand.
        MoEGroupedPlanDemand demand = MoEGroupedPlanDemand::None;
        MoEOverlayServicePhaseHint phase = MoEOverlayServicePhaseHint::Auto; ///< Required immutable semantic graph role.
        /// Same count borrowed by the router; invalid router slots exclude its inactive suffix.
        std::optional<DeviceRowRange> live_rows;
        std::string prefix;
        std::string router_node;
        MoEIntermediateShardTensorBinding hidden;
        MoEIntermediateShardTensorBinding routing_indices;
        MoEIntermediateShardTensorBinding routing_weights;
        MoEIntermediateShardTensorBinding route_partials; ///< FP32 [rows,top_k,H], not a wire payload.
        MoEIntermediateShardTensorBinding output; ///< FP32 [rows,H] for a local fold; absent for route slots.
    };

    /**
     * @brief Closed original prepared-service route provenance, unavailable to serving builders.
     *
     * Only the canonical prepared GPU sampler can construct this after checking
     * its original source/bank epoch and cold calibration owner. It binds one
     * fixed expert packet to the private source2 runtime and exact phase/rows;
     * it is not a flag that disables normal graph-router authentication.
     */
    class MoEIntermediateShardPreparedServiceRoute final
    {
        friend class PlanningGPUExpertMeasurement;
    public:
        /** @return Whether these exact private inputs retain the original certified sample. */
        bool authenticates(const MoEIntermediateShardPipelineParams& params,
            const MoEOverlayParticipantLayerBank& source) const;
    private:
        /** @brief Issued only by the original sampler after complete source/owner validation. */
        MoEIntermediateShardPreparedServiceRoute(std::shared_ptr<const MoEOverlayFragmentFamily> family,
            DeviceMoERuntimeTable* runtime,int expert,uint64_t epoch,int rows,MoEOverlayServicePhaseHint phase,
            const ITensor* indices,const ITensor* weights)
            :family_(std::move(family)),runtime_(runtime),expert_(expert),epoch_(epoch),rows_(rows),
             phase_(phase),indices_(indices),weights_(weights){}
        std::shared_ptr<const MoEOverlayFragmentFamily> family_;
        DeviceMoERuntimeTable* runtime_;
        int expert_;uint64_t epoch_;int rows_;
        MoEOverlayServicePhaseHint phase_;
        const ITensor* indices_;const ITensor* weights_;
    };

    /**
     * @brief Typed local-partial publication, deliberately distinct from a complete FFN result.
     *
     * The next builder must retain the original family/reduction order, apply
     * the shared gate locally, and sum exactly the device-owned live H rows.
     * This handle does not authorize a whole-expert consumer or serving mode.
     */
    struct MoEIntermediateShardLocalPartial
    {
        std::string producer;
        MoEIntermediateShardTensorBinding output;
        std::shared_ptr<const MoEOverlayFragmentFamily> family;
        std::optional<DeviceRowRange> live_rows;
        int rows;
        int model_columns;
        MoEOverlayServicePhaseHint phase = MoEOverlayServicePhaseHint::Auto;
    };

    /**
     * @brief Append real local fragment kernels and an ordered router-slot fold without device work.
     * @param graph Participant-local graph containing the original router producer.
     * @param params Complete frozen execution policy and pre-admitted arena tensors.
     * @param source Original authenticated layer borrowed only to export setup descriptor metadata.
     * @return Typed full-H local partial; no collective or shared branch is hidden in these stages.
     * @throws std::invalid_argument For missing, aliased, undersized or incompatible bindings.
     * @throws std::logic_error For a foreign source/runtime family or node identity.
     *
     * No initial bank reader, prepared engine or source-layer pointer is retained.
     * Descriptor tables carry the fragment runtime-source arm and never choose
     * current residency. Request reset preserves topology and prepared metadata;
     * explicit graph invalidation retires the shared kernel metadata first.
     */
    MoEIntermediateShardLocalPartial appendMoEIntermediateShardPipeline(
        ComputeGraph &graph, MoEIntermediateShardPipelineParams params,
        const MoEOverlayParticipantLayerBank &source);

    /** @brief Raw original route-slot partials for the tier-aware reducer's single router multiply. */
    struct MoEIntermediateShardRouteSlots
    {
        std::string producer;
        MoEIntermediateShardTensorBinding partials;
        std::shared_ptr<const MoEOverlayFragmentFamily> family;
        std::optional<DeviceRowRange> live_rows;
        int rows;
        int top_k;
        int model_columns;
    };

    /**
     * @brief Append only the fragment producer when a later mixed-tier owner folds route slots.
     * @param graph Participant-local graph containing its original router.
     * @param params Frozen device runtime, original family and original-route workspace. Output must be absent.
     * @param source Cold prepared original source used to seal native descriptor metadata.
     * @return Exact root-reduction input, with no participant-local route fold.
     */
    MoEIntermediateShardRouteSlots appendMoEIntermediateShardRouteSlots(
        ComputeGraph &graph, MoEIntermediateShardPipelineParams params,
        const MoEOverlayParticipantLayerBank &source);

    /**
     * @brief Keep independent shared FFN math after the CPU packet and inside its original producer unit.
     * @param graph Participant-local graph with the original dispatch and ticket contract.
     * @param shared Original shared FFN node.
     * @param packet Original mapped CPU packet producer, present only on its owner.
     * @param cutpoint Original end of the producer unit, present for an explicit CPU ticket.
     * @throws std::invalid_argument For a foreign participant, stage, or capture boundary.
     *
     * The packet may start CPU service while the retained GPU producer still runs.
     * A follower places its shared math in the matching unit without owning a packet.
     * The following captured ingress and the existing collectives own their original dependencies.
     */
    void orderMappedDecodeSharedFFN(ComputeGraph &graph, const std::string &shared,
        const std::optional<std::string> &packet, const std::optional<std::string> &cutpoint);

    /**
     * @brief Reuse the production compute/fold stages for one original cold prepared sample.
     * @param graph Private calibration graph; fixed route tensors are external inputs.
     * @param params Exact source2 family, phase, rows, arena and private runtime.
     * @param source Same original prepared bank borrowed by the closed sample issuer.
     * @param route Closed original sampler-issued route provenance, never serving policy.
     * @return The same local-partial handle; completion still requires an explicit sum.
     * @throws std::invalid_argument Before graph mutation for foreign source/epoch/packet/policy.
     * @note Demand is None; no router arithmetic or serving state is consumed.
     */
    MoEIntermediateShardLocalPartial appendPreparedMoEIntermediateShardPipeline(
        ComputeGraph& graph,MoEIntermediateShardPipelineParams params,
        const MoEOverlayParticipantLayerBank& source,const MoEIntermediateShardPreparedServiceRoute& route);

    /** @brief Frozen local execution composition, independent of the final native collective. */
    enum class MoEIntermediateShardFFNExecution { Staged, ResidentCandidate };

    /** @brief Existing shared producer and the one explicitly ordered final FFN collective. */
    struct MoEIntermediateShardFFNFinalization
    {
        MoEIntermediateShardFFNExecution execution = MoEIntermediateShardFFNExecution::Staged;
        std::string shared_ffn_node; ///< Real SharedExpertFFNStage with authenticated original local-F bindings.
        const FrozenModelWeightSet *shared_weights = nullptr; ///< Original participant-local frozen source, borrowed only during declaration.
        const TensorParallelConfig *shared_partition = nullptr; ///< Original dense/shared TP assignment; absent only for an original SingleDevice full source.
        MoEIntermediateShardTensorBinding local_combined; ///< Distinct admitted FP32 H bank, reduced in place.
        TPAllreduceStage::Params final_sum; ///< Original transport/sidebands; must declare CanonicalRankOrder and FP32.
        std::string prefix; ///< Unique graph identity for the local gate-add and final sum.
    };

    /** @brief Complete FFN publication only after the declared native ordered sum. */
    struct MoEIntermediateShardFFNOutput
    {
        std::string producer;
        MoEIntermediateShardTensorBinding output;
        std::shared_ptr<const MoEOverlayFragmentFamily> family;
        std::optional<DeviceRowRange> live_rows;
        int rows;
        int model_columns;
        MoEOverlayServicePhaseHint phase;
    };

    /**
     * @brief Append local shared gating/addition and exactly one full-H ordered FFN sum.
     * @param graph Same participant-local graph containing both genuine local producers.
     * @param routed Local-partial handle returned by this graph's original fragment lowering.
     * @param finalization Original shared prepared-source declaration, admitted bank and native sum policy.
     * @return Complete FFN output whose producer is the explicit final sum, never the local gate-add.
     * @throws std::invalid_argument For foreign phase/count/source/arena, unsupported sum policy or request-row sideband shape.
     * @throws std::logic_error For a foreign original family, communicator order or duplicate finalization.
     *
     * The prepared store authenticates local shared gate/up N cuts and the same
     * down K cut. Shared F may differ from routed F. This builder retains no
     * host placement map, performs no device work and allocates no tensor. The
     * original sum sidebands pass through unchanged; independent request-row
     * layouts are rejected explicitly. Counted communication reuses the exact
     * original live-prefix authority and native participant order. Existing
     * kernel-grid economy, full model KL and migration gates remain separate.
     */
    MoEIntermediateShardFFNOutput finalizeMoEIntermediateShardFFN(ComputeGraph &graph,
        const MoEIntermediateShardLocalPartial &routed, MoEIntermediateShardFFNFinalization finalization);

    /**
     * @brief Fuse the original router into its existing native-layout resident FFN node.
     * @param graph Original participant-local graph before preparation.
     * @param resident_node Existing resident FFN sink retaining all original public outputs.
     */
    void lowerNativeLayoutRouterFFN(ComputeGraph& graph,const std::string& resident_node);

    /**
     * @brief Close original previous-write/HC and native-router FFN into one local entry.
     * @param graph Original cold participant-local graph.
     * @param resident_node Existing native-router resident FFN sink, before preparation.
     * @param previous_write_node Exact original preceding residual-write node.
     * @param hc_read_node Exact original residual-read terminal producing the FFN input.
     * @details Original final-read absence of injection is retained. A middle read's
     * independent injection and all other original outside readers are explicit
     * region exports; the final collective remains outside. M1 ordinary only.
     */
    void lowerFullNativeHCFFN(ComputeGraph& graph,const std::string& resident_node,
        const std::string& previous_write_node,const std::string& hc_read_node);

    /**
     * @brief Select the retained M1 HC/FFN from the original composed model dependencies.
     * @param graph Cold original attention/FFN graph on an exact ROCm participant.
     * @param previous_write_node Original attention terminal.
     * @details The paired FP32 injection remains an independent reader of the
     * retained normalized branches and the FFN sum remains explicit.
     */
    void lowerRetainedM1HCFFN(ComputeGraph& graph,const std::string& previous_write_node);

    /**
     * @brief Lower the original QFN M1 main-inference decode and retain its model census identity.
     * @param graph Original composed participant-local model graph.
     * @param previous_write_node Original attention write terminal.
     */
    void lowerRetainedM1HCFFNMainInference(ComputeGraph& graph,const std::string& previous_write_node);
}
