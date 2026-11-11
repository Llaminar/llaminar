/**
 * @file NamedDomainGraphBuilder.cpp
 * @brief Lower named domains without a second initialization lifecycle.
 *
 * All model loading, capacity admission, snapshot policy, request state, and
 * readiness transitions belong to OrchestrationRunner. This construction-only
 * boundary replaces adoption of an already-built graph through a unit-test
 * constructor, which previously bypassed memory and serving-graph admission.
 *
 * Local and global pipeline stages share the original rank-owned prepared
 * source and PMA handoff. Global topology remains immutable metadata; remote
 * execution, activation transport and model-context reuse are not impersonated.
 */
#include "NamedDomainGraphBuilder.h"
#include "DomainCommunicatorRegistry.h"
#include "OriginalCPUPipelineConstruction.h"
#include "OriginalPipelineStageDomain.h"
#include "GlobalOrchestrator.h"
#include "StageRunnerFactory.h"
#include "../moe/MoEOverlayPipelineStageBinding.h"
#include "../global_pp/GlobalPPRankPlanBuilder.h"
#include "../mpi_orchestration/ExecutionPlanBuilder.h"
#include "../mtp/MTPWeightManifest.h"
#include "../../loaders/ModelContext.h"
#include "../../loaders/PreparedWeightStore.h"
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
        const std::shared_ptr<IMPIContext> &mpi_ctx,
        std::span<const std::shared_ptr<const MoEOverlayPipelineStageBinding>> stage_bindings)
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
        const auto cpu_programs=stage_bindings.empty()?std::vector<OriginalCPUPipelineProgram>{}:
            originalCPUPipelinePrograms(config,topology,inventory);
        const auto admitted_weights=model_ctx->concreteWeightManager();
        if(!admitted_weights || (!cpu_programs.empty() && !admitted_weights->physicalMemoryAuthority()))
            throw std::logic_error("CPU pipeline native setup requires its original admitted model PMA");
        return withOriginalCPUPipelineConstruction(domains,*mpi_ctx,[&]() -> std::unique_ptr<IInferenceRunner> {
            if (std::any_of(topology.stages.begin(), topology.stages.end(),
                            [](const auto &stage) { return stage.is_global_tp; }))
                domains.initialize(topology,mpi_ctx->communicator(),mpi_ctx->rank(),cpu_programs,
                    admitted_weights->physicalMemoryAuthority());

            const auto rank_plan = GlobalPPRankPlanBuilder::build(topology, mpi_ctx->rank());
            StageBuildContext context;
            context.model_ctx = model_ctx;
            context.mpi_ctx = mpi_ctx;
            context.runtime = plan.runtime;
            const auto weights = model_ctx->concreteWeightManager();
            if (!weights)
                throw std::logic_error("Named-domain graph requires the model-owned weight manager");
            context.prepared_weight_store = weights->preparedWeightStoreIfInitialized();
            if (!context.prepared_weight_store)
                context.prepared_weight_store = std::make_shared<PreparedWeightStore>();
            context.domain_registry = &domains;
            if (!stage_bindings.empty())
            {
                if (stage_bindings.size() != rank_plan.executeStages().size() ||
                    std::any_of(stage_bindings.begin(), stage_bindings.end(), [](const auto &value) { return !value; }))
                    throw std::invalid_argument("Named-domain expert construction requires every exact local stage binding");
                for (const auto *action : rank_plan.executeStages())
                {
                    const auto found = std::find_if(stage_bindings.begin(), stage_bindings.end(),
                        [&](const auto &binding) { return binding->scope().first_layer == action->first_layer; });
                    if (found == stage_bindings.end())
                        throw std::invalid_argument("Named-domain expert binding omitted an original local action");
                    const auto &binding = **found;
                    const FactoryPPStageConfig scope{.first_layer = action->first_layer,
                        .last_layer = action->last_layer + 1, .has_embedding = action->has_embedding,
                        .has_lm_head = action->has_lm_head, .generation_domain = binding.scope().generation_domain};
                    const auto spec=std::find_if(topology.stages.begin(),topology.stages.end(),
                        [&](const auto& stage){return stage.stage_id==action->stage_id;});
                    if(spec==topology.stages.end()) throw std::logic_error("Missing original expert pipeline stage");
                    const auto original=OriginalPipelineStageDomain::borrow(*spec,*action);
                    binding.requireDestination(scope, original.participants, original.weights, original.backend,
                        context.runtime.moe_rebalance.mode);
                }
                context.moe_stage_bindings.assign(stage_bindings.begin(), stage_bindings.end());
            }

            GlobalOrchestrator::Config global;
            global.topology = topology;
            global.rank = mpi_ctx->rank();
            global.world_size = mpi_ctx->world_size();
            global.mpi_ctx = mpi_ctx.get();
            global.original_mpi_ctx = mpi_ctx;
            global.original_cpu_domains=domains.originalCPUOwners();
            global.physical_memory=admitted_weights->physicalMemoryAuthority();
            global.hosted_activation_rows=plan.runtime.resident_graph_rows>0
                ?plan.runtime.resident_graph_rows:plan.runtime.max_seq_len;
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
        });
    }
}
