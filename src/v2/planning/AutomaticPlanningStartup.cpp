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
#include "execution/mtp/MTPWeightManifest.h"
#include "loaders/ModelLoader.h"
#include "config/OrchestrationConfigDocument.h"
#include "config/OrchestrationStartupPolicy.h"
#include "interfaces/IMPIContext.h"
#include "utils/Logger.h"
#include <chrono>
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
        const ClusterInventory &inventory, const std::shared_ptr<IMPIContext> &mpi, const Prepare &prepare,
        const PrepareHostServices &prepare_host_services,const PrepareNativeContexts &prepare_native_contexts)
    {
        // Wall clocks include setup, publication and retirement, not just native
        // timed samples. CLI discovery/bootstrap precedes this API and is reported
        // separately by the launcher; this total must never be called CLI wall time.
        const auto startup_begin = std::chrono::steady_clock::now();
        auto phase_begin = startup_begin;
        const auto phase = [&](const char *name) {
            const auto now = std::chrono::steady_clock::now();
            LOG_INFO("[AutomaticPlanningWall] discovery_rank=" << (mpi ? mpi->rank() : 0)
                << " phase=" << name << " seconds="
                << std::chrono::duration<double>(now-phase_begin).count());
            phase_begin = now;
        };
        OrchestrationConfig shared;
        std::optional<OrchestrationPlanningWorkload> workload;
        exchangePlanningArtifact(mpi, PlanningArtifact::AutomaticRequest, [&] {
            auto resolved = request;
            resolved.mpi_procs = inventory.world_size;
            const auto document = serializeOrchestrationConfig(resolved);
            return std::vector<uint8_t>(document.begin(), document.end());
        }, [&](std::span<const uint8_t> bytes) {
            if (!prepare || !prepare_host_services)
                throw std::invalid_argument("Automatic startup requires complete cost and CPU service evidence policies");
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

        phase("request_publication");

        // The root-only producer creates the sole source owner. It survives all
        // samples and the complete search, avoiding model-directory rereads and
        // avoiding any requirement for peers to mount the launcher's model path.
        std::optional<PlanningModelSource> source;
        const auto metadata = exchangePlanningModelMetadata(mpi, [&] {
            source.emplace(shared.model_path, shared.shared_predictor);
            // Root authenticates learned weights before any hardware samples,
            // BOM search or rank selection. This fallible work stays inside the
            // metadata publication transaction so every discovery rank rejects
            // the same request rather than leaving followers in a collective.
            if (retainsMTPGraphCapacity(shared.mtp))
                (void)requireMTPWeightManifest(source->loader(),
                    source->loader().architecture(),
                    static_cast<int>(source->loader().blockCount()));
            return source->metadata();
        });
        resolveMTPStartupPolicy(shared, metadata.memoryProfile());
        phase("model_read_and_metadata");
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

        phase("hardware_compute_memory_communication_evidence");

        // Every original process reports its own class before membership narrows.
        // As with cost preparation, the collector owns its internal collective
        // failure edges; the final vote covers only local result construction.
        PlanningHostServiceBudgets host_services;
        std::exception_ptr service_error;
        try { host_services = prepare_host_services(context); }
        catch (...) { service_error = std::current_exception(); }
        acceptPlanningCostPreparation(mpi, [&] {
            if (service_error) std::rethrow_exception(service_error);
            if (context.isRoot()) {
                if (host_services.size() != inventory.ranks.size())
                    throw std::invalid_argument("Automatic CPU service evidence omitted an original discovery rank");
                for (int rank = 0; rank < inventory.world_size; ++rank)
                    if (!host_services.contains(rank))
                        throw std::invalid_argument("Automatic CPU service evidence changed discovery membership");
            } else if (!host_services.empty()) {
                throw std::invalid_argument("Automatic CPU service evidence published root-only metadata on a follower");
            }
        });

        auto native_budgets=prepare_native_contexts(context);
        phase("host_and_native_budget_metadata");
        std::optional<SelectedAutomaticOrchestration> selected;
        auto applied = exchangeSelectedOrchestration(mpi, [&] {
            auto memory = OrchestrationCandidateMemoryPolicy::fromStartup();
            memory.retained_host_services = std::move(host_services);
            memory.native_context_budgets=std::move(native_budgets);
            memory.require_native_context_budgets=true;
            selected.emplace(AutomaticOrchestrationPlanner::select(shared, *source, inventory,
                memory, *workload, *evaluate));
            return selected->candidate().config();
        });
        phase("candidate_admission_selection_publication");
        LOG_INFO("[AutomaticPlanningWall] discovery_rank=" << (mpi ? mpi->rank() : 0)
            << " phase=startup_api_total seconds="
            << std::chrono::duration<double>(std::chrono::steady_clock::now()-startup_begin).count());
        return {std::move(applied), std::move(selected)};
    }
}
