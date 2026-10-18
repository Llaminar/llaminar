/**
 * @file OrchestrationCandidateAdmission.h
 * @brief Model-aware, topology-wide physical admission for automatic candidates.
 *
 * A proposal is not a plan until every selected rank is compiled and its whole
 * CPU/GPU BOM fits. Ordinary and ExpertOverlay candidates use the same builders
 * as runtime admission. This boundary owns no allocator, live ledger, inference
 * state, hardware discovery or performance ranking.
 */
#pragma once
#include "planning/AutomaticOrchestrationCandidates.h"
#include "planning/PlanningModelMetadata.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "planning/MemoryPlanner.h"
#include "planning/MoEPipelineMemoryAdmission.h"
#include "execution/mpi_orchestration/RankExecutionPlan.h"
#include "execution/moe/MoEOverlayCapacityResolver.h"
#include "loaders/GPUVramPreflight.h"

namespace llaminar2
{
    /** @brief Borrowed expert evidence for one exact global layer of an admitted candidate. */
    struct AdmittedMoELayerView
    {
        const MoERoutedExpertPlacementPlan &placement;
        const MoEOverlayResolvedCapacityPlan &capacity;
    };

    /** @brief Complete setup capture/upload policy, frozen once for a candidate search. */
    struct OrchestrationCandidateMemoryPolicy
    {
        std::vector<int> prefill_bucket_rows;
        int minimum_prefill_sequence_rows = 0;
        int maximum_cached_prefill_buckets = 0;
        GPUWeightLoadMemoryPolicy weight_load;

        /** @return The production startup policy; no workload or memory estimate is invented. */
        static OrchestrationCandidateMemoryPolicy fromStartup();
    };

    /**
     * @brief A compiled candidate with one immutable, complete physical admission.
     *
     * Admission is against the supplied observation, not a reservation on future
     * hardware. Applying a saved selection still requires fresh live admission.
     * Performance selection may compare only successfully admitted values; this
     * object does not claim one candidate is faster than another.
     */
    class AdmittedOrchestrationCandidate final
    {
    public:
        /**
         * @brief Compile every selected rank and admit the complete production BOM.
         * @param candidate Declarative proposal, consumed without altering its source request.
         * @param model One retained metadata-only GGUF owner for the entire search.
         * @param policy Exact captured-prefill and initial-upload policies.
         * @return Complete configuration, per-rank geometry and shared physical certificate.
         * @throws PhysicalMemoryCapacityExhausted when no allowed graph shape fits.
         * @throws std::invalid_argument for malformed topology/model/capture policy.
         * @throws std::logic_error for unsupported or inconsistent production lowering.
         * @throws std::overflow_error for arithmetic outside representable capacity.
         */
        static AdmittedOrchestrationCandidate admit(
            AutomaticOrchestrationCandidate candidate,
            const PlanningModelSource &model,
            const OrchestrationCandidateMemoryPolicy &policy);

        /** @return Original strategy family, independently of its lowered graph form. */
        OrchestrationStrategy strategy() const noexcept { return candidate_.strategy; }
        /** @return Complete explicit placement and admitted ExpertOverlay quotas, if applicable. */
        const OrchestrationConfig &config() const noexcept { return candidate_.config; }
        /** @return Exact discovery-to-execution mapping, never inferred from rank count. */
        const ExecutionRankMembership &membership() const noexcept { return candidate_.membership; }
        /** @return Compiled per-rank geometry with the admitted graph row capacities. */
        const std::vector<RankExecutionPlan> &rankPlans() const noexcept { return ranks_; }
        /**
         * @return Exact admitted participant inputs, including TP assignments and PP ranges.
         *
         * These are compiler facts, not a new physical ledger. Overlay inputs
         * describe the fixed zero-routed-expert BOM; expertLayer() names the
         * final routed residency authority. Costs must not infer live
         * expert counts or invocation traffic from these allocation inputs.
         */
        const std::vector<DevicePlanConfig> &devicePlans() const noexcept { return devices_; }
        /** @return The canonical immutable aggregate behind every resource diagnostic. */
        const PhysicalMemoryPlanAdmissionCertificate &physicalAdmission() const noexcept { return *admission_; }
        /** @return Quota evidence for a model-wide overlay; pipelines use stage-local expertLayer() views. */
        const std::shared_ptr<const MoEOverlayResolvedCapacityPlan> &overlayCapacity() const noexcept { return overlay_; }
        /** @return Complete local PP admission when experts have separate stage authorities. */
        const std::shared_ptr<const AdmittedMoEPipelineMemory> &pipelineMemory() const noexcept { return pipeline_; }
        /**
         * @brief Select the admitted expert authority for a global model layer.
         * @param model_layer Main or retained predictor layer, in model coordinates.
         * @return Borrowed evidence, or no value for an ordinary candidate.
         * @throws std::invalid_argument if an expert candidate does not own that layer.
         *
         * The returned references remain valid only while this candidate lives.
         * Cost consumers must use this scope instead of a parent-wide expert plan.
         */
        std::optional<AdmittedMoELayerView> expertLayer(int model_layer) const;

    private:
        /** @brief Only successful complete admission may create a selectable candidate. */
        AdmittedOrchestrationCandidate(AutomaticOrchestrationCandidate candidate,
            std::vector<RankExecutionPlan> ranks,
            std::vector<DevicePlanConfig> devices,
            std::shared_ptr<const PhysicalMemoryPlanAdmissionCertificate> admission,
            std::shared_ptr<const MoEOverlayResolvedCapacityPlan> overlay,
            std::shared_ptr<const AdmittedMoEPipelineMemory> pipeline = {});
        AutomaticOrchestrationCandidate candidate_;
        std::vector<RankExecutionPlan> ranks_;
        std::vector<DevicePlanConfig> devices_;
        std::shared_ptr<const PhysicalMemoryPlanAdmissionCertificate> admission_;
        std::shared_ptr<const MoEOverlayResolvedCapacityPlan> overlay_;
        std::shared_ptr<const AdmittedMoEPipelineMemory> pipeline_;
    };
}
