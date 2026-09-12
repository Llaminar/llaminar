/**
 * @file NamedDomainGraphBuilder.cpp
 * @brief Lower named domains without a second initialization lifecycle.
 *
 * All model loading, capacity admission, snapshot policy, request state, and
 * readiness transitions belong to OrchestrationRunner. This construction-only
 * boundary replaces adoption of an already-built graph through a unit-test
 * constructor, which previously bypassed memory and serving-graph admission.
 */
#include "NamedDomainGraphBuilder.h"
#include "DomainCommunicatorRegistry.h"
#include "GlobalOrchestrator.h"
#include "StageRunnerFactory.h"
#include "../global_pp/GlobalPPRankPlanBuilder.h"
#include "../mpi_orchestration/ExecutionPlanBuilder.h"
#include "../mtp/MTPWeightManifest.h"
#include "../../loaders/ModelContext.h"
#include "../../utils/Logger.h"
#include "../../utils/MPIContext.h"

#include <algorithm>
#include <set>
#include <stdexcept>

namespace llaminar2
{
    bool requiresNamedDomainGlobalGraph(const OrchestrationConfig &config)
    {
        if (!config.usesNamedDomains() || config.pp_stage_definitions.empty())
            return false;

        // Unused domain declarations must not change the model's execution
        // path. Inspect only domains referenced by the complete stage sequence.
        std::set<std::string> stage_domains;
        for (const auto &stage : config.pp_stage_definitions)
            stage_domains.insert(stage.domain_name);

        std::set<int> owners;
        for (const auto &domain : config.domain_definitions)
        {
            if (!stage_domains.contains(domain.name))
                continue;
            if (domain.scope == TPScope::NODE_LOCAL ||
                domain.scope == TPScope::GLOBAL)
                return true;
            if (domain.owner_rank)
                owners.insert(*domain.owner_rank);
            owners.insert(domain.explicit_ranks.begin(), domain.explicit_ranks.end());
            std::set<std::string> hosts;
            for (const auto &device : domain.devices)
                hosts.insert(device.hostname);
            if (hosts.size() > 1)
                return true;
        }
        return owners.size() > 1;
    }

    std::unique_ptr<IInferenceRunner> buildNamedDomainGlobalGraph(
        const OrchestrationConfig &config,
        const RankExecutionPlan &plan,
        const ClusterInventory &inventory,
        IExecutionPlanBuilder &builder,
        const std::shared_ptr<ModelContext> &model_ctx,
        const std::shared_ptr<IMPIContext> &mpi_ctx)
    {
        if (!model_ctx || !mpi_ctx)
            throw std::invalid_argument("Named-domain graph requires admitted model and MPI authorities");
        auto *topology_builder = dynamic_cast<ExecutionPlanBuilder *>(&builder);
        if (!topology_builder)
            throw std::invalid_argument("Named-domain graph requires the canonical topology builder");

        // Use authenticated metadata from the loaded model, not a second GGUF
        // read or guessed geometry after a metadata error.
        ModelConfig model;
        model.n_layers = mainLayerCountExcludingMTP(
            model_ctx->concreteLoader(), model_ctx->architecture(), model_ctx->totalBlockCount());
        model.n_heads = model_ctx->headCount();
        model.n_kv_heads = model_ctx->headCountKV();
        model.hidden_size = model_ctx->embeddingLength();
        model.vocab_size = model_ctx->vocabSize();
        auto topology = topology_builder->buildGlobalPPTopology(config, model, inventory);
        const auto errors = topology.validate();
        if (!errors.empty())
        {
            std::string message = "Invalid admitted named-domain topology:";
            for (const auto &error : errors)
                message += "\n  - " + error;
            throw std::runtime_error(message);
        }
        if (config.show_topology && mpi_ctx->rank() == 0)
            LOG_DEBUG("Multi-domain topology:\n"
                      << renderMultiDomainTopologyInfo(topology, mpi_ctx->world_size()));

        // Every world rank enters communicator construction in the same stage
        // order. Entries retain shared domain contexts beyond this local registry.
        DomainCommunicatorRegistry domains;
        if (std::any_of(topology.stages.begin(), topology.stages.end(),
                        [](const auto &stage) { return stage.is_global_tp; }))
            domains.initialize(topology, mpi_ctx->communicator(), mpi_ctx->rank());

        const auto rank_plan = GlobalPPRankPlanBuilder::build(topology, mpi_ctx->rank());
        StageBuildContext context;
        context.model_ctx = model_ctx;
        context.mpi_ctx = mpi_ctx;
        context.runner_config = InferenceRunnerConfig::fromPlan(plan);
        context.domain_registry = &domains;

        GlobalOrchestrator::Config global;
        global.topology = topology;
        global.rank = mpi_ctx->rank();
        global.world_size = mpi_ctx->world_size();
        global.mpi_ctx = mpi_ctx.get();
        global.vocab_size = model.vocab_size;
        global.d_model = model.hidden_size;
        global.architecture_name = model_ctx->architecture();
        for (const auto &step : rank_plan.steps)
        {
            if (step.type != GlobalPPRankPlan::Step::Type::EXECUTE_STAGE ||
                step.stage_action.role != RankStageAction::Role::EXECUTE)
                continue;
            const auto &action = step.stage_action;
            const auto spec = std::find_if(topology.stages.begin(), topology.stages.end(),
                [&](const auto &stage) { return stage.stage_id == action.stage_id; });
            if (spec == topology.stages.end())
                throw std::runtime_error("Named-domain plan references an absent stage");
            global.stage_runners.push_back(StageRunnerFactory::create(*spec, action, context));
        }
        return std::make_unique<GlobalOrchestrator>(std::move(global));
    }
}
