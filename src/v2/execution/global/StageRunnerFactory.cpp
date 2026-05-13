/**
 * @file StageRunnerFactory.cpp
 * @brief Implementation of StageRunnerFactory
 *
 * @author David Sanftenberg
 * @date May 2026
 */

#include "StageRunnerFactory.h"
#include "../../utils/Logger.h"
#include "../local_execution/orchestrators/RankOrchestrator.h"
#include "../../collective/TPContextFactory.h"
#include "../../loaders/PreparedWeightStore.h"

#include <stdexcept>

namespace llaminar2
{
    namespace
    {
        std::shared_ptr<StageWeightContext> makeStageWeightContext(
            const RankStageAction &action,
            const FactoryPPStageConfig &pp_cfg)
        {
            auto context = std::make_shared<StageWeightContext>();
            context->stage_id = action.stage_id;
            context->domain_name = action.domain_name;
            context->pp_stage_config = pp_cfg;
            context->prepared_store = std::make_shared<PreparedWeightStore>();
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

        const FactoryPPStageConfig pp_cfg = makePPStageConfig(action);

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
        const GlobalPPStageSpec & /*stage*/,
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

        InferenceRunnerConfig runner_cfg = ctx.runner_config;
        runner_cfg.pp_stage_config = pp_cfg;
        auto weight_context = makeStageWeightContext(action, pp_cfg);
        runner_cfg.prepared_weight_store = weight_context->prepared_store;

        std::unique_ptr<IInferenceRunner> runner;
        if (auto concrete_model_ctx = std::dynamic_pointer_cast<ModelContext>(ctx.model_ctx))
        {
            runner = createPPStageRunner(concrete_model_ctx, device, pp_cfg, runner_cfg);
        }
        else
        {
            runner = createTestableInferenceRunner(ctx.model_ctx, device, runner_cfg);
        }
        if (!runner)
        {
            throw std::runtime_error(
                "StageRunnerFactory: runner creation failed for single-device stage " +
                std::to_string(action.stage_id));
        }

        LOG_INFO("[StageRunnerFactory] Built single-device runner for stage "
                 << action.stage_id << " on device " << device.to_string()
                 << " layers [" << action.first_layer << ", " << action.last_layer << "]");

        StageRunnerEntry entry;
        entry.stage_id = action.stage_id;
        entry.domain_name = action.domain_name;
        entry.action = action;
        entry.runner = std::move(runner);
        entry.pp_stage_config = pp_cfg;
        entry.weight_context = std::move(weight_context);
        return entry;
    }

    // =========================================================================
    // buildLocalTP
    // =========================================================================

    StageRunnerEntry StageRunnerFactory::buildLocalTP(
        const GlobalPPStageSpec & /*stage*/,
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

        // Create LocalTPContext for entry lifetime tracking.
        // createRankOrchestrator takes unique_ptr ownership so we create a second
        // independent context with identical parameters for the runner itself.
        auto entry_ctx = TPContextFactory::createLocal(
            action.devices,
            action.tp_weights,
            action.backend);

        if (!entry_ctx)
        {
            throw std::runtime_error(
                "StageRunnerFactory: TPContextFactory::createLocal failed for stage " +
                std::to_string(action.stage_id));
        }

        // Second context owned by RankOrchestrator
        auto runner_ctx = TPContextFactory::createLocal(
            action.devices,
            action.tp_weights,
            action.backend);

        if (!runner_ctx)
        {
            throw std::runtime_error(
                "StageRunnerFactory: second TPContextFactory::createLocal failed for stage " +
                std::to_string(action.stage_id));
        }

        // Build RankOrchestrator::Config for local TP with nested PP stage config
        auto weight_context = makeStageWeightContext(action, pp_cfg);
        RankOrchestrator::Config ro_config;
        ro_config.mode = RankOrchestrator::ParallelismMode::TP;
        ro_config.devices = action.devices;
        ro_config.weights = action.tp_weights;
        ro_config.backend = action.backend;
        ro_config.max_seq_len = static_cast<size_t>(ctx.runner_config.max_seq_len);
        ro_config.batch_size = ctx.runner_config.batch_size;
        ro_config.activation_precision = ctx.runner_config.activation_precision;
        ro_config.kv_cache_precision = ctx.runner_config.kv_cache_precision;
        ro_config.use_mapped_memory = ctx.runner_config.use_mapped_memory;
        ro_config.nested_pp_stage_config = pp_cfg;
        ro_config.prepared_weight_store = weight_context->prepared_store;

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

        LOG_INFO("[StageRunnerFactory] Built local TP runner for stage "
                 << action.stage_id << " with " << action.devices.size() << " device(s)"
                 << " layers [" << action.first_layer << ", " << action.last_layer << "]");

        StageRunnerEntry entry;
        entry.stage_id = action.stage_id;
        entry.domain_name = action.domain_name;
        entry.action = action;
        entry.runner = std::move(rank_runner);
        entry.local_tp_ctx = std::move(entry_ctx);
        entry.pp_stage_config = pp_cfg;
        entry.weight_context = std::move(weight_context);
        return entry;
    }

    // =========================================================================
    // buildGlobalTP
    // =========================================================================

    StageRunnerEntry StageRunnerFactory::buildGlobalTP(
        const GlobalPPStageSpec & /*stage*/,
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

        InferenceRunnerConfig runner_cfg = ctx.runner_config;
        runner_cfg.pp_stage_config = pp_cfg;
        runner_cfg.tp_ctx = global_tp_ctx.get();
        runner_cfg.tp_device_index = tp_device_index;
        auto weight_context = makeStageWeightContext(entry_action, pp_cfg);
        runner_cfg.prepared_weight_store = weight_context->prepared_store;

        std::unique_ptr<IInferenceRunner> runner;
        if (auto concrete_model_ctx = std::dynamic_pointer_cast<ModelContext>(ctx.model_ctx))
        {
            runner = createInferenceRunner(concrete_model_ctx, ctx.mpi_ctx, device, runner_cfg);
        }
        else
        {
            runner = createTestableInferenceRunner(ctx.model_ctx, device, runner_cfg);
        }
        if (!runner)
        {
            throw std::runtime_error(
                "StageRunnerFactory: runner creation failed for global TP stage " +
                std::to_string(action.stage_id));
        }

        LOG_INFO("[StageRunnerFactory] Built global TP runner for stage "
                 << action.stage_id << " tp_rank=" << tp_device_index
                 << "/" << tp_domain_size
                 << " device=" << device.to_string()
                 << " layers [" << action.first_layer << ", " << action.last_layer << "]");

        StageRunnerEntry entry;
        entry.stage_id = action.stage_id;
        entry.domain_name = action.domain_name;
        entry.action = entry_action;
        entry.runner = std::move(runner);
        entry.global_tp_ctx = std::move(global_tp_ctx);
        entry.pp_stage_config = pp_cfg;
        entry.weight_context = std::move(weight_context);
        return entry;
    }

} // namespace llaminar2
