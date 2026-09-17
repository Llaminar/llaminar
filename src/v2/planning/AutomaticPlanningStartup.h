/**
 * @file AutomaticPlanningStartup.h
 * @brief Shared discovery-to-selection transaction for plan and default-auto serve.
 *
 * Every discovery rank participates in bounded evidence collection before any
 * rank is excluded or renumbered. Only root opens the GGUF and searches admitted
 * candidates. The strict existing apply codec publishes the final selection;
 * no frontend owns a private search, workload, metadata or measurement policy.
 */
#pragma once

#include "planning/AutomaticOrchestrationPlanner.h"
#include <memory>
#include <optional>

namespace llaminar2
{
    class IMPIContext;

    /**
     * @brief Borrowed immutable inputs during the one startup evidence transaction.
     *
     * Source ownership stays with startup until selection/publication finishes.
     * Evidence returned from preparation must own its observations, not borrow
     * this context or native sample storage. Followers cannot open a model file.
     */
    class AutomaticPlanningPreparation final
    {
    public:
        /** @return Root's published complete automatic request, not a follower's CLI copy. */
        const OrchestrationConfig &request() const noexcept { return request_; }
        /** @return Canonical physical hosts/ranks, before execution membership selection. */
        const ClusterInventory &inventory() const noexcept { return inventory_; }
        /** @return Exact discovery communicator; absence explicitly means local work. */
        const std::shared_ptr<IMPIContext> &mpi() const noexcept { return mpi_; }
        /** @return Identical metadata on every discovery participant. */
        const PlanningModelMetadata &metadata() const noexcept { return metadata_; }
        /** @return Shared objective, resolved once and checked before model access. */
        const OrchestrationPlanningWorkload &workload() const noexcept { return workload_; }
        /** @return True only for the sole source owner and selecting rank. */
        bool isRoot() const noexcept { return source_ != nullptr; }
        /** @return Root-only retained source; follower access is a precise lifecycle error. */
        const PlanningModelSource &rootSource() const;
    private:
        friend class AutomaticPlanningStartup;
        /** @brief Startup alone may bind these borrowed, already authenticated inputs. */
        AutomaticPlanningPreparation(const OrchestrationConfig &request,
            const ClusterInventory &inventory, const std::shared_ptr<IMPIContext> &mpi,
            const PlanningModelMetadata &metadata, OrchestrationPlanningWorkload workload,
            const PlanningModelSource *source)
            : request_(request), inventory_(inventory), mpi_(mpi), metadata_(metadata),
              workload_(workload), source_(source) {}
        const OrchestrationConfig &request_;
        const ClusterInventory &inventory_;
        const std::shared_ptr<IMPIContext> &mpi_;
        const PlanningModelMetadata &metadata_;
        OrchestrationPlanningWorkload workload_;
        const PlanningModelSource *source_;
    };

    /** @brief Completed apply publication everywhere, with root-only search evidence. */
    struct AutomaticPlanningStartupResult final
    {
        OrchestrationConfig applied;
        std::optional<SelectedAutomaticOrchestration> root_selection;
    };

    /** @brief Single startup authority shared by both public automatic frontends. */
    class AutomaticPlanningStartup final
    {
    public:
        /**
         * @brief Collect immutable evidence on all ranks; return an evaluator only on root.
         *
         * Distributed preparation must use failure-atomic planning publication/
         * sampling APIs for each exchange. Put fallible local work in their local
         * callbacks, never before an unmatched collective. After the last exchange,
         * local evaluator construction is covered by startup's completion consensus.
         * No callback may split ranks or construct an inference runner.
         */
        using Prepare = std::function<std::optional<AutomaticOrchestrationPlanner::Evaluate>(
            const AutomaticPlanningPreparation &)>;

        /** @return Common startup preparation policy; never chosen privately by a frontend. */
        static Prepare preparation();

        /**
         * @brief Publish intent/metadata, prepare evidence, select once, then publish apply.
         * @param request Root's parsed intent; follower paths need not be locally readable.
         * @param inventory Canonical inventory from this discovery communicator.
         * @param mpi Discovery context, before execution rank admission; null is process-local.
         * @param prepare Common evidence policy, injectable without bypassing production admission.
         * @return Complete apply config everywhere; winning BOM/cost only on root.
         * @throws std::runtime_error for publication, evidence or selection failure on any rank.
         */
        static AutomaticPlanningStartupResult run(const OrchestrationConfig &request,
            const ClusterInventory &inventory, const std::shared_ptr<IMPIContext> &mpi,
            const Prepare &prepare = preparation());
    };
}
