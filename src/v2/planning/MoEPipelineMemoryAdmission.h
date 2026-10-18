/**
 * @file MoEPipelineMemoryAdmission.h
 * @brief One metadata-to-PMA transaction for every local MoE pipeline stage.
 *
 * The topology compiler owns stage boundaries; the GGUF manifest owns routed
 * source geometry. This adapter composes their existing fixed and expert BOM
 * builders before publishing any result. It owns no allocator, live byte ledger,
 * GPU context, weight payload or mutable inference state.
 */
#pragma once

#include "planning/MoEOverlayMemoryPlanInputs.h"
#include "planning/ResolvedRankOrchestration.h"

namespace llaminar2
{
    class ModelLoader;
    class PlanningModelMetadata;

    /** @brief Frozen setup policy shared by discovery and live runner admission. */
    struct MoEPipelineMemoryPolicy
    {
        MoEOverlayPrefillMemoryPolicy prefill;
        MoEOverlayGPUWeightLoadCapacityInput gpu_weight_load;
        GraphSnapshotMemoryCapacity snapshot_capacity;
        std::size_t model_graph_topology_variant_count = 1;
    };

    /**
     * @brief One stage's immutable projection of the complete admission transaction.
     *
     * The normalized topology preserves automatic quota intent. Resolved quotas
     * are evidence in capacity(), and may be frozen for live setup only after
     * admission against the runtime's current physical observation.
     */
    class AdmittedMoEPipelineStage final
    {
    public:
        /** @return Exact scope, participants and requested expert policy. */
        const ResolvedMoEPipelineStage &topology() const noexcept { return topology_; }
        /** @return Child plan with the admitted graph and prefill row capacities. */
        const RankExecutionPlan &rankPlan() const noexcept { return rank_plan_; }
        /** @return Main and terminal-only predictor source identities. */
        const MoEOverlayInferenceGraphFamilyIdentity &graphFamily() const noexcept { return family_; }
        /** @return Exact migration storage and controller geometry. */
        const MoEOverlayCapacityAdmissionPolicy &storagePolicy() const noexcept { return storage_policy_; }
        /** @return Metadata-only projection directory retaining global layer indices. */
        const std::vector<MoEOverlayLayerWeightManifest> &weightManifest() const noexcept { return manifest_; }
        /** @return Fixed participant inputs; routed quotas belong to capacity(). */
        const std::vector<DevicePlanConfig> &devicePlans() const noexcept { return devices_; }
        /** @return Stage quotas retaining the same aggregate certificate as every sibling. */
        const std::shared_ptr<const MoEOverlayResolvedCapacityPlan> &capacity() const noexcept { return capacity_; }

    private:
        friend class AdmittedMoEPipelineMemory;
        /** @brief Only complete aggregate admission may publish this stage. */
        explicit AdmittedMoEPipelineStage(const ResolvedMoEPipelineStage &topology)
            : topology_(topology), rank_plan_(topology.rankPlan()) {}
        ResolvedMoEPipelineStage topology_;
        RankExecutionPlan rank_plan_;
        MoEOverlayInferenceGraphFamilyIdentity family_;
        MoEOverlayCapacityAdmissionPolicy storage_policy_;
        std::vector<MoEOverlayLayerWeightManifest> manifest_;
        std::vector<DevicePlanConfig> devices_;
        std::shared_ptr<const MoEOverlayResolvedCapacityPlan> capacity_;
    };

    /** @brief Complete local pipeline admission, never a list of independent admissions. */
    class AdmittedMoEPipelineMemory final
    {
    public:
        /**
         * @brief Admit all fixed owners and routed populations against one observation.
         * @param topology Exact compiler result retaining the authored local pipeline.
         * @param model Complete immutable model descriptor, including predictor metadata.
         * @param loader The same model's metadata loader; no payload bytes are read.
         * @param inventory Discovery or refreshed live observation, in topology rank space.
         * @param policy Explicit captured-prefill, upload and snapshot owner geometry.
         * @return Every stage and one shared physical certificate after complete admission.
         * @throws PhysicalMemoryCapacityExhausted when no declared graph shape fits.
         * @throws std::invalid_argument for invalid source, topology or capture policy.
         * @throws std::logic_error if the existing builders publish inconsistent owners.
         *
         * Only typed capacity exhaustion selects a smaller declared row bucket.
         * Malformed geometry never becomes a capacity rejection or changes topology.
         */
        static AdmittedMoEPipelineMemory admit(const ResolvedRankOrchestration &topology,
            const PlanningModelMetadata &model, const ModelLoader &loader,
            const ClusterInventory &inventory, const MoEPipelineMemoryPolicy &policy);

        /** @return Parent PP geometry with the jointly admitted row envelope. */
        const RankExecutionPlan &rankPlan() const noexcept { return rank_plan_; }
        /** @return Immutable ordered stages; no borrowed setup references survive. */
        const std::vector<AdmittedMoEPipelineStage> &stages() const noexcept { return stages_; }
        /** @return Sole aggregate admission behind every stage's quota evidence. */
        const std::shared_ptr<const PhysicalMemoryPlanAdmissionCertificate> &physicalAdmission() const noexcept
        { return admission_; }

    private:
        /** @brief Only admit() can return a fully sealed pipeline transaction. */
        explicit AdmittedMoEPipelineMemory(RankExecutionPlan parent) : rank_plan_(std::move(parent)) {}
        RankExecutionPlan rank_plan_;
        std::vector<AdmittedMoEPipelineStage> stages_;
        std::shared_ptr<const PhysicalMemoryPlanAdmissionCertificate> admission_;
    };
}
