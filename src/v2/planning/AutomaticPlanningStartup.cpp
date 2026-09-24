/**
 * @file AutomaticPlanningStartup.cpp
 * @brief One all-rank evidence phase followed by root-only cost selection.
 *
 * Publication authenticates the root request before reading the model. The
 * retained root source supplies both bounded measurement slices and admission
 * metadata; followers receive metadata and sample payloads through the existing
 * failure-atomic transport. Evidence preparation is outside the root-only
 * selection callback, so remote workers can contribute before membership narrows.
 */
#include "planning/AutomaticPlanningStartup.h"
#include "planning/PlanningPublication.h"
#include "planning/PlanningRequestCostModel.h"
#include "config/OrchestrationConfigDocument.h"
#include "interfaces/IMPIContext.h"
#include <exception>
#include <stdexcept>

namespace llaminar2
{
    const PlanningModelSource &AutomaticPlanningPreparation::rootSource() const
    {
        if (!source_) throw std::logic_error("Only discovery root owns the automatic planning model source");
        return *source_;
    }

    AutomaticPlanningStartup::Prepare AutomaticPlanningStartup::preparation()
    {
        // Every discovery rank contributes before membership narrows. Both
        // frontends consume this one policy; only root retains the cost closure.
        return PlanningRequestCostModel::prepare;
    }

    AutomaticPlanningStartupResult AutomaticPlanningStartup::run(const OrchestrationConfig &request,
        const ClusterInventory &inventory, const std::shared_ptr<IMPIContext> &mpi, const Prepare &prepare)
    {
        OrchestrationConfig shared;
        std::optional<OrchestrationPlanningWorkload> workload;
        exchangePlanningArtifact(mpi, PlanningArtifact::AutomaticRequest, [&] {
            auto resolved = request;
            resolved.mpi_procs = inventory.world_size;
            const auto document = serializeOrchestrationConfig(resolved);
            return std::vector<uint8_t>(document.begin(), document.end());
        }, [&](std::span<const uint8_t> bytes) {
            if (!prepare) throw std::invalid_argument("Automatic startup requires an evidence preparation policy");
            const int count = mpi ? mpi->world_size() : 1;
            if (inventory.world_size != count || inventory.ranks.size() != static_cast<size_t>(count))
                throw std::invalid_argument("Automatic inventory does not cover the discovery communicator");
            shared = deserializeOrchestrationConfig(std::string(bytes.begin(), bytes.end()));
            if (!std::holds_alternative<AutomaticOrchestrationRequest>(resolveOrchestrationIntent(shared)) ||
                shared.model_path.empty() || shared.mpi_procs != count || shared.max_seq_len < 2)
                throw std::invalid_argument("Automatic startup requires a model, discovery-sized auto intent and context >= 2");
            workload = shared.automatic_planning.workload.value_or(
                OrchestrationPlanningWorkload::defaultsForContext(shared.max_seq_len));
            workload->requireFitsContext(shared.max_seq_len);
        });

        // The root-only producer creates the sole source owner. It survives all
        // samples and the complete search, avoiding model-directory rereads and
        // avoiding any requirement for peers to mount the launcher's model path.
        std::optional<PlanningModelSource> source;
        const auto metadata = exchangePlanningModelMetadata(mpi, [&] {
            source.emplace(shared.model_path);
            return source->metadata();
        });
        const AutomaticPlanningPreparation context(shared, inventory, mpi, metadata,
            *workload, source ? &*source : nullptr);
        std::optional<AutomaticOrchestrationPlanner::Evaluate> evaluate;
        std::exception_ptr preparation_error;
        try { evaluate = prepare(context); }
        catch (...) { preparation_error = std::current_exception(); }
        // Collectors already agree each distributed failure internally. This
        // final local edge also covers constructing the cost closure after the
        // last gather, and rejects an evaluator accidentally returned by a peer.
        acceptPlanningCostPreparation(mpi, [&] {
            if (preparation_error) std::rethrow_exception(preparation_error);
            if (context.isRoot() != evaluate.has_value() || (evaluate && !*evaluate))
                throw std::invalid_argument("Automatic cost evidence must produce one nonempty root-only evaluator");
        });

        std::optional<SelectedAutomaticOrchestration> selected;
        auto applied = exchangeSelectedOrchestration(mpi, [&] {
            selected.emplace(AutomaticOrchestrationPlanner::select(shared, *source, inventory,
                OrchestrationCandidateMemoryPolicy::fromStartup(), *workload, *evaluate));
            return selected->candidate().config();
        });
        return {std::move(applied), std::move(selected)};
    }
}
