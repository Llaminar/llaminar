/**
 * @file StageRunnerFactory.cpp
 * @brief Stage-scoped construction using canonical policy and resource owners.
 *
 * Pipeline stages retain distinct layer ranges, not competing copies of model
 * weights or collective contexts. The runtime config is projected by the same
 * factories used for whole-model execution so composition cannot silently
 * reset MTP, prefix caching, graph capacity or expert maintenance policy.
 *
 * @author David Sanftenberg
 * @date May 2026
 *
 * Local and global pipeline stages share the original rank-owned prepared
 * source and PMA handoff. Global topology remains immutable metadata; remote
 * execution, activation transport and model-context reuse are not impersonated.
 */

#include "StageRunnerFactory.h"
#include "OriginalPipelineStageDomain.h"
#include "collective/GlobalTPContext.h"
#include "utils/MPIContext.h"
#include "execution/moe/MoEOverlayParticipantResidency.h"
#include "execution/moe/MoEExpertOverlayPreparationPlan.h"
#include "GlobalStageTPValidation.h"
#include "../../utils/Logger.h"
#include "../local_execution/orchestrators/RankOrchestrator.h"
#include "../../collective/TPContextFactory.h"
#include "../../loaders/PreparedWeightStore.h"
#include "../moe/MoEOverlayPipelineStageBinding.h"

#include <stdexcept>
#include <algorithm>

namespace llaminar2
{
    namespace
    {
        /** @brief Retain the model's prepared namespace with this stage's layer scope. */
        std::shared_ptr<StageWeightContext> makeStageWeightContext(
            const RankStageAction &action,
            const FactoryPPStageConfig &pp_cfg,
            const std::shared_ptr<PreparedWeightStore> &prepared_store)
        {
            auto context = std::make_shared<StageWeightContext>();
            context->stage_id = action.stage_id;
            context->domain_name = action.domain_name;
            context->pp_stage_config = pp_cfg;
            if (!prepared_store)
                throw std::invalid_argument("Pipeline stage requires the model-owned prepared weight store");
            context->prepared_store = prepared_store;
            return context;
        }
    }

    // =========================================================================
    // Public entry point
    // =========================================================================

    StageRunnerEntry StageRunnerFactory::create(
        const GlobalPPStageSpec &stage,
        const RankStageAction &action,
        const StageBuildContext &ctx)
    {
        if (action.role != RankStageAction::Role::EXECUTE)
        {
            throw std::invalid_argument(
                "StageRunnerFactory::create called with IDLE action for stage " +
                std::to_string(action.stage_id));
        }

        if (!ctx.model_ctx)
        {
            throw std::runtime_error(
                "StageRunnerFactory::create - model_ctx is null (stage " +
                std::to_string(action.stage_id) + ")");
        }

        if (ctx.test_runner_overrides.test_graph_allocation_admission &&
            std::dynamic_pointer_cast<ModelContext>(ctx.model_ctx))
            throw std::invalid_argument("Concrete stage model cannot accept test graph allocation admission");

        FactoryPPStageConfig pp_cfg = makePPStageConfig(action);
        if (!ctx.moe_stage_bindings.empty())
        {
            if (std::any_of(ctx.moe_stage_bindings.begin(), ctx.moe_stage_bindings.end(),
                    [](const auto &value) { return !value; }))
                throw std::invalid_argument("Global stage factory cannot accept a missing expert owner");
            const auto binding = std::find_if(ctx.moe_stage_bindings.begin(), ctx.moe_stage_bindings.end(),
                [&](const auto &value) { return value && value->scope().first_layer == action.first_layer; });
            if (binding == ctx.moe_stage_bindings.end())
                throw std::invalid_argument("Global stage omitted its admitted expert runtime");
            pp_cfg.generation_domain = (*binding)->scope().generation_domain;
            const auto original=OriginalPipelineStageDomain::borrow(stage,action);
            (*binding)->requireDestination(pp_cfg, original.participants, original.weights, original.backend,
                ctx.runtime.moe_rebalance.mode);
        }

        if (action.is_global_tp)
        {
            return buildGlobalTP(stage, action, ctx, pp_cfg);
        }

        // Single-device or local TP
        const bool is_local_tp = (action.inner_mode == InnerParallelism::LOCAL_TP) ||
                                  (action.devices.size() > 1);
        if (is_local_tp)
        {
            return buildLocalTP(stage, action, ctx, pp_cfg);
        }

        return buildSingleDevice(stage, action, ctx, pp_cfg);
    }

    // =========================================================================
    // makePPStageConfig
    // =========================================================================

    FactoryPPStageConfig StageRunnerFactory::makePPStageConfig(const RankStageAction &action)
    {
        FactoryPPStageConfig cfg;
        cfg.first_layer = action.first_layer;
        // GlobalPP last_layer is inclusive; FactoryPPStageConfig last_layer is exclusive.
        cfg.last_layer = action.last_layer + 1;
        cfg.has_embedding = action.has_embedding;
        cfg.has_lm_head = action.has_lm_head;
        return cfg;
    }

    // =========================================================================
    // buildSingleDevice
    // =========================================================================

    StageRunnerEntry StageRunnerFactory::buildSingleDevice(
        const GlobalPPStageSpec &stage,
        const RankStageAction &action,
        const StageBuildContext &ctx,
        const FactoryPPStageConfig &pp_cfg)
    {
        // Determine device: use first action device if available, else CPU
        DeviceId device = DeviceId::cpu();
        if (!action.devices.empty())
        {
            device = action.devices[0].toLocalDeviceId();
        }
        else if (!action.device.hostname.empty() || action.device.device_type != DeviceType::CPU)
        {
            device = action.device.toLocalDeviceId();
        }

        InferenceRunnerConfig runner_cfg = InferenceRunnerConfig::fromRuntime(ctx.runtime);
        runner_cfg.pp_stage_config = pp_cfg;
        auto weight_context = makeStageWeightContext(action, pp_cfg, ctx.prepared_weight_store);
        runner_cfg.prepared_weight_store = weight_context->prepared_store;
        for (const auto &binding : ctx.moe_stage_bindings)
            if (binding->scope().first_layer == pp_cfg.first_layer)
                binding->installRuntime(runner_cfg, action.devices, action.tp_weights, action.backend);

        std::unique_ptr<IInferenceRunner> runner;
        if (auto concrete_model_ctx = std::dynamic_pointer_cast<ModelContext>(ctx.model_ctx))
        {
            runner = createPPStageRunner(concrete_model_ctx, device, pp_cfg, runner_cfg);
        }
        else
        {
            runner = createTestableInferenceRunner(ctx.model_ctx, device, runner_cfg, ctx.test_runner_overrides);
        }
        if (!runner)
        {
            throw std::runtime_error(
                "StageRunnerFactory: runner creation failed for single-device stage " +
                std::to_string(action.stage_id));
        }

        LOG_DEBUG("[StageRunnerFactory] Built single-device runner for stage "
                 << action.stage_id << " on device " << device.to_string()
                 << " layers [" << action.first_layer << ", " << action.last_layer << "]");

        StageRunnerEntry entry;
        entry.original_mpi_ctx = ctx.mpi_ctx;
        entry.stage_id = action.stage_id;
        entry.domain_name = action.domain_name;
        entry.action = action;
        entry.runner = std::move(runner);
        entry.pp_stage_config = pp_cfg;
        entry.weight_context = std::move(weight_context);
        for (const auto &binding : ctx.moe_stage_bindings)
            if (binding->scope().first_layer == pp_cfg.first_layer)
                entry.moe_runtime = binding;
        return entry;
    }

    // =========================================================================
    // buildLocalTP
    // =========================================================================

    StageRunnerEntry StageRunnerFactory::buildLocalTP(
        const GlobalPPStageSpec &stage,
        const RankStageAction &action,
        const StageBuildContext &ctx,
        const FactoryPPStageConfig &pp_cfg)
    {
        if (action.devices.empty())
        {
            throw std::runtime_error(
                "StageRunnerFactory: local TP stage " + std::to_string(action.stage_id) +
                " has no devices");
        }

        // Ownership moves into RankOrchestrator. Stage diagnostics query that
        // live owner instead of allocating another collective for bookkeeping.
        auto runner_ctx = TPContextFactory::createLocal(
            action.devices,
            action.tp_weights,
            action.backend);

        if (!runner_ctx)
        {
            throw std::runtime_error(
                "StageRunnerFactory: TPContextFactory::createLocal failed for stage " +
                std::to_string(action.stage_id));
        }

        requireOriginalGlobalStageTPBirth(action, runner_ctx->devices(), runner_ctx->backend());

        // Build RankOrchestrator::Config for local TP with nested PP stage config
        auto weight_context = makeStageWeightContext(action, pp_cfg, ctx.prepared_weight_store);
        auto ro_config = RankOrchestrator::Config::fromRuntime(ctx.runtime);
        ro_config.mode = RankOrchestrator::ParallelismMode::TP;
        ro_config.devices = action.devices;
        ro_config.weights = action.tp_weights;
        ro_config.backend = action.backend;
        ro_config.nested_pp_stage_config = pp_cfg;
        ro_config.prepared_weight_store = weight_context->prepared_store;
        ro_config.test_runner_overrides = ctx.test_runner_overrides;
        for (const auto &binding : ctx.moe_stage_bindings)
            if (binding->scope().first_layer == pp_cfg.first_layer)
                ro_config.installPipelineStageRuntime(*binding);

        auto rank_runner = createRankOrchestrator(
            ctx.model_ctx,
            std::move(runner_ctx),
            ro_config);

        if (!rank_runner)
        {
            throw std::runtime_error(
                "StageRunnerFactory: createRankOrchestrator failed for local TP stage " +
                std::to_string(action.stage_id));
        }

        LOG_DEBUG("[StageRunnerFactory] Built local TP runner for stage "
                 << action.stage_id << " with " << action.devices.size() << " device(s)"
                 << " layers [" << action.first_layer << ", " << action.last_layer << "]");

        StageRunnerEntry entry;
        entry.original_mpi_ctx = ctx.mpi_ctx;
        entry.stage_id = action.stage_id;
        entry.domain_name = action.domain_name;
        entry.action = action;
        entry.runner = std::move(rank_runner);
        entry.pp_stage_config = pp_cfg;
        entry.weight_context = std::move(weight_context);
        for (const auto &binding : ctx.moe_stage_bindings)
            if (binding->scope().first_layer == pp_cfg.first_layer)
                entry.moe_runtime = binding;
        return entry;
    }

    // =========================================================================
    // buildGlobalTP
    // =========================================================================

    StageRunnerEntry StageRunnerFactory::buildGlobalTP(
        const GlobalPPStageSpec &stage,
        const RankStageAction &action,
        const StageBuildContext &ctx,
        const FactoryPPStageConfig &pp_cfg)
    {
        if (!ctx.domain_registry)
        {
            throw std::runtime_error(
                "StageRunnerFactory: domain_registry is null for global TP stage " +
                std::to_string(action.stage_id) +
                ". Initialize DomainCommunicatorRegistry before building global TP runners.");
        }

        auto global_tp_ctx = ctx.domain_registry->globalTPContextForStage(action.stage_id);
        if (!global_tp_ctx)
        {
            throw std::runtime_error(
                "StageRunnerFactory: no GlobalTPContext for stage " +
                std::to_string(action.stage_id) +
                " in DomainCommunicatorRegistry. "
                "This rank may not be a participant, or registry was not initialized.");
        }

        // Determine device for this rank in the global TP domain
        DeviceId device = DeviceId::cpu();
        if (action.device.device_type != DeviceType::CPU ||
            action.device.device_ordinal != 0 ||
            action.device.numa_node >= 0)
        {
            device = action.device.toLocalDeviceId();
        }

        const int tp_device_index = global_tp_ctx->myIndex();
        const int tp_domain_size = global_tp_ctx->degree();

        RankStageAction entry_action = action;
        entry_action.tp_rank_in_domain = tp_device_index;
        entry_action.tp_domain_size = tp_domain_size;

        InferenceRunnerConfig runner_cfg = InferenceRunnerConfig::fromRuntime(ctx.runtime);
        runner_cfg.pp_stage_config = pp_cfg;
        runner_cfg.tp_ctx = global_tp_ctx.get();
        runner_cfg.tp_device_index = tp_device_index;
        auto weight_context = makeStageWeightContext(entry_action, pp_cfg, ctx.prepared_weight_store);
        runner_cfg.prepared_weight_store = weight_context->prepared_store;

        std::shared_ptr<IMPIContext> graph_mpi_ctx = ctx.mpi_ctx;
        // Preserve the same admitted expert owners on the cross-rank factory
        // branch. A global CPU domain is not an empty local-TP declaration.
        for (const auto &binding : ctx.moe_stage_bindings)
            if (binding->scope().first_layer == pp_cfg.first_layer) {
                const auto original=OriginalPipelineStageDomain::borrow(stage,action);
                if(global_tp_ctx->worldRanks()!=stage.participating_ranks ||
                    global_tp_ctx->myIndex()!=action.tp_rank_in_domain ||
                    global_tp_ctx->degree()!=action.tp_domain_size || global_tp_ctx->localDevice()!=action.device)
                    throw std::invalid_argument("CPU pipeline factory changed its original global TP context");
                binding->installRuntime(runner_cfg,original.participants,original.weights,original.backend);
                // Dense graph collectives and cold weight progress use the
                // actual stage split. The binding retains the original parent
                // separately for sparse request and rank-batch protocols.
                graph_mpi_ctx = MPIContextFactory::borrowOriginalCPUStage(ctx.mpi_ctx, global_tp_ctx);
                if(binding->owners().residency->projectionPreparation() &&
                    binding->owners().residency->projectionPreparation()->cpuJointOwnerMap()) {
                    const auto original_cpu=std::dynamic_pointer_cast<GlobalTPContext>(global_tp_ctx);
                    if(!original_cpu || !original_cpu->authenticatesOriginalCPUExpertBorrow(global_tp_ctx))
                        throw std::invalid_argument("CPU pipeline fragment graphs require their original admitted expert domain");
                    runner_cfg.moe_expert_overlay_cpu_joint_context=global_tp_ctx;
                    // This original process owns exactly one CPU child for this
                    // stage. It publishes its own prepared bank; remote ranks
                    // own distinct banks and must never wait on this local gate.
                    runner_cfg.moe_overlay_cpu_preparation=
                        MoEOverlayCpuPreparationParticipation::publisher(
                            std::make_shared<MoEOverlayCpuPreparationGate>());
                }
            }

        std::unique_ptr<IInferenceRunner> runner;
        if (auto concrete_model_ctx = std::dynamic_pointer_cast<ModelContext>(ctx.model_ctx))
        {
            runner = createInferenceRunner(concrete_model_ctx, std::move(graph_mpi_ctx), device, runner_cfg);
        }
        else
        {
            runner = createTestableInferenceRunner(ctx.model_ctx, device, runner_cfg, ctx.test_runner_overrides);
        }
        if (!runner)
        {
            throw std::runtime_error(
                "StageRunnerFactory: runner creation failed for global TP stage " +
                std::to_string(action.stage_id));
        }

        LOG_DEBUG("[StageRunnerFactory] Built global TP runner for stage "
                 << action.stage_id << " tp_rank=" << tp_device_index
                 << "/" << tp_domain_size
                 << " device=" << device.to_string()
                 << " layers [" << action.first_layer << ", " << action.last_layer << "]");

        StageRunnerEntry entry;
        entry.original_mpi_ctx = ctx.mpi_ctx;
        entry.stage_id = action.stage_id;
        entry.domain_name = action.domain_name;
        entry.action = entry_action;
        entry.runner = std::move(runner);
        entry.global_tp_ctx = std::move(global_tp_ctx);
        entry.pp_stage_config = pp_cfg;
        entry.weight_context = std::move(weight_context);
        for (const auto &binding : ctx.moe_stage_bindings)
            if (binding->scope().first_layer == pp_cfg.first_layer) entry.moe_runtime=binding;
        return entry;
    }

} // namespace llaminar2
