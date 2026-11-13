/**
 * @file MoEPipelinePreparedPlan.h
 * @brief Model-bound immutable expert placements for one admitted rank-owned pipeline stage set.
 *
 * Prepared weights retain the aggregate admission and exact initial placement
 * of every stage. Runtime histograms, registries, graphs and controllers are
 * rebuilt per runner and never enter this model-owned reuse value.
 *
 * Local and global pipeline stages share the original rank-owned prepared
 * source and PMA handoff. Global topology remains immutable metadata; remote
 * execution, activation transport and model-context reuse are not impersonated.
 */
#pragma once
#include "planning/MoEPipelineMemoryAdmission.h"
#include "MoERoutedExpertPlacementPlanner.h"
#include <memory>
#include <vector>

namespace llaminar2
{
    class ModelContext;
    class ModelLoader;
    class IMPIContext;
    class MoEOverlayPipelineStageBinding;

    /** @brief Exact source-resolved metadata and frozen initial placements of one stage. */
    struct MoEPipelinePreparedStage
    {
        MoERoutedExpertModelMetadata metadata;
        std::shared_ptr<const MoERoutedExpertPlacementPlan> placement;
    };

    /** @brief Sealed model-owned pipeline authority; no mutable inference state. */
    class MoEPipelinePreparedPlan final
    {
    public:
        /**
         * @brief Freeze all stage placements against their original joint admission.
         * @param admission Complete transaction before prepared payload allocation.
         * @param model Exact live metadata source used later by graph preparation.
         * @return Immutable complete set, never partially published stage placements.
         * @throws std::invalid_argument for a missing or mismatched model/admission.
         */
        static std::shared_ptr<const MoEPipelinePreparedPlan> create(
            std::shared_ptr<const AdmittedMoEPipelineMemory> admission, const std::shared_ptr<ModelContext> &model);

        /**
         * @brief Construct fresh per-runner owners from the retained model placement.
         * @param loader The original model's metadata loader; no payload is read.
         * @param memory The same PMA retained by the prepared weights.
         * @param context Exact local execution communicator, never a synthetic peer.
         * @param current Requested stage policies, checked against original admission before owner construction.
         * @return Complete ordered bindings retained until every child graph retires.
         * @throws std::invalid_argument for a changed certificate, stage or rank.
         */
        std::vector<std::shared_ptr<const MoEOverlayPipelineStageBinding>> createRuntimeBindings(
            const ModelLoader &loader, std::shared_ptr<PhysicalMemoryAuthority> memory,
            std::shared_ptr<IMPIContext> context, const ResolvedRankOrchestration &current) const;

        /**
         * @brief Prepare original CPU stage activation channels on every parent MPI rank.
         * @param loader Same actual model metadata source, never reloaded or substituted.
         * @param context Original parent MPI owner, including ranks owning different pipeline stages.
         * @param inventory Same complete rank/NUMA publication used by admission.
         * @param runtime Parent's admitted common prefill schedule and retained row policy.
         * @param bindings Actual local stage owners, never representatives for remote ranks.
         * @return Local immutable bindings with their exact original CPU rank-batch registries.
         * @throws std::exception Collectively before a later stage if any original source/endpoint fails setup.
         */
        std::vector<std::shared_ptr<const MoEOverlayPipelineStageBinding>> prepareOriginalCPUStageTransports(
            const ModelLoader& loader,std::shared_ptr<IMPIContext> context,const ClusterInventory& inventory,
            const RuntimeConfig& runtime,
            std::span<const std::shared_ptr<const MoEOverlayPipelineStageBinding>> bindings) const;

        /** @return One aggregate admission retained by every prepared stage. */
        const std::shared_ptr<const AdmittedMoEPipelineMemory> &admission() const noexcept { return admission_; }
        /** @return Initial model-owned expert placements, independent of live runtime epochs. */
        const std::vector<MoEPipelinePreparedStage> &stages() const noexcept { return stages_; }

    private:
        /** @brief Only create() publishes a completely bound model plan. */
        explicit MoEPipelinePreparedPlan(std::shared_ptr<const AdmittedMoEPipelineMemory> admission,
            const std::shared_ptr<ModelContext> &model)
            : admission_(std::move(admission)), model_(model) {}
        std::shared_ptr<const AdmittedMoEPipelineMemory> admission_;
        // The reuse contract owns model lifetime. A passive plan cannot retain
        // prepared GPU allocations beyond the model's explicit retirement.
        std::weak_ptr<ModelContext> model_;
        std::vector<MoEPipelinePreparedStage> stages_;
    };
}
