/**
 * @file InferenceRunnerFactory.cpp
 * @brief Factory implementation for creating IInferenceRunner instances
 * @author David Sanftenberg
 * @date December 2025
 */

#include "InferenceRunnerFactory.h"
#include "EagerWeightValidator.h"
#include "../../backends/DeviceId.h"
#include "../../backends/BackendManager.h"
#include "../local_execution/collective/CollectiveContext.h"
#include "../mpi_orchestration/DeviceInventory.h"
#include "../mpi_orchestration/RankExecutionPlan.h"
#include "../local_execution/graph/GraphBuilderRegistry.h"
#include "../../models/IGraphConfigBuilder.h"
#include "../local_execution/graph/SchemaFactoryRegistry.h" // Model-agnostic sharding config
#include "../local_execution/orchestrators/DeviceGraphOrchestrator.h"
#include "../../loaders/ModelContext.h"
#include "../../loaders/ModelLoader.h"
#include "../../loaders/PreparedWeightStore.h"
#include "../../loaders/WeightManager.h"
#include "../../loaders/WeightStreamerFactory.h"
#include "../../collective/ILocalTPContext.h"
#include "../../collective/ITPContext.h"
#include "../../collective/TPContextFactory.h"
#include "../../collective/GlobalTPContext.h"
#include "../local_execution/orchestrators/IRankOrchestrator.h"
#include "../local_execution/orchestrators/RankOrchestrator.h"
#include "../../config/PipelineConfig.h"
#include "../../config/TensorParallelConfig.h"
#include "../../utils/DebugEnv.h"
#include "../../utils/Logger.h"
#include "../../utils/WeightLoadingProfiler.h"
#include "../../kernels/cpu/turboquant/TurboQuantContext.h"
#include "../../kernels/cpu/rotation/ActivationRotation.h"
#include "../../execution/moe/MoERebalanceController.h"
#include "../../execution/moe/DecodeExpertHistogram.h"
#include "../../execution/moe/MoEExpertOverlayExecutionPlan.h"
#include "../../execution/moe/MoEExpertOverlayRuntimePlan.h"
#include "../../execution/moe/MoEExpertParallelPlanner.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <future>
#include <functional>
#include <sstream>
#include <thread>
#include <unordered_set>

namespace llaminar2
{

    // Forward declarations of factory helpers
    static std::unique_ptr<IInferenceRunner> createDeviceGraphOrchestratorImpl(
        std::shared_ptr<ModelContext> model_ctx,
        std::shared_ptr<IMPIContext> mpi_ctx,
        DeviceId device,
        const InferenceRunnerConfig &config,
        const std::string &architecture);

    static bool configureOrchestratorWeightsImpl(
        DeviceGraphOrchestrator *orchestrator,
        std::shared_ptr<ModelContext> model_ctx,
        DeviceId device,
        const InferenceRunnerConfig &config,
        const GraphConfig &graph_config);

    static std::shared_ptr<PreparedWeightStore> installPreparedWeightStoreForPlan(
        WeightManager &weight_mgr,
        const InferenceRunnerConfig &config,
        const WeightPlan &weight_plan,
        const char *context)
    {
        const ModelContextId model_id = weight_plan.strategy().model_id;
        auto store = config.prepared_weight_store
                         ? config.prepared_weight_store
                         : std::make_shared<PreparedWeightStore>(model_id);

        if (!store->bindModelIdIfUnset(model_id))
        {
            LOG_ERROR(context << " prepared store model id mismatch: store="
                              << store->modelId().value
                              << " plan=" << model_id.value);
            return nullptr;
        }

        weight_mgr.setPreparedWeightStore(store);
        return store;
    }

    namespace
    {
        void installTPExecutorCancellation(
            GraphConfig &graph_config,
            ITPContext *tp_ctx,
            const std::string &log_prefix)
        {
            if (!tp_ctx || tp_ctx->degree() <= 1)
                return;

            if (!graph_config.executor_config.cancellation_requested)
            {
                graph_config.executor_config.cancellation_requested = [tp_ctx]()
                {
                    return tp_ctx->isAbortRequested();
                };
            }

            if (!graph_config.executor_config.stage_failure_callback)
            {
                graph_config.executor_config.stage_failure_callback = [tp_ctx, log_prefix](const std::string &stage_name, const std::string &reason)
                {
                    LOG_WARN(log_prefix << " stage failure triggers TP abort"
                                        << " stage='" << stage_name << "' reason='" << reason << "'");
                    tp_ctx->requestAbort();
                };
            }
        }

        MoEExpertModelMetadata moEExpertMetadataForModel(IModelContext &model_ctx)
        {
            auto loader = model_ctx.loader();
            const std::string &arch = model_ctx.architecture();

            MoEExpertModelMetadata metadata;
            metadata.num_layers = std::max(model_ctx.totalBlockCount(), model_ctx.blockCount());
            metadata.num_experts = loader ? loader->getInt(arch + ".expert_count", 0) : 0;
            metadata.d_model = model_ctx.embeddingLength();
            metadata.routed_intermediate_size = loader ? loader->getInt(arch + ".expert_feed_forward_length", 0) : 0;
            if (metadata.routed_intermediate_size <= 0)
                metadata.routed_intermediate_size = model_ctx.feedForwardLength();
            metadata.shared_intermediate_size = metadata.routed_intermediate_size;
            metadata.has_shared_expert = loader && loader->getInt(arch + ".expert_shared_count", 0) > 0;
            return metadata;
        }

        int stableContinuationDomainId(const std::string &domain_name)
        {
            const size_t hashed = std::hash<std::string>{}(domain_name);
            return static_cast<int>(hashed & 0x3fffffffU);
        }

        std::vector<std::string> denseMoEDomainNames(const MoEExpertParallelPlan &plan)
        {
            std::vector<std::string> names;
            std::unordered_set<std::string> seen;
            auto add = [&](const std::string &name)
            {
                if (!name.empty() && seen.insert(name).second)
                    names.push_back(name);
            };

            add(plan.continuation_domain);
            add(plan.effectiveBaseModelDomain());
            add(plan.shared_expert_domain);
            for (const auto &domain : plan.dense_domains)
                add(domain.name);
            return names;
        }

        const ExecutionDomainDefinition *findDenseMoEDomain(
            const MoEExpertParallelPlan &plan,
            const std::string &name)
        {
            auto it = std::find_if(plan.dense_domains.begin(), plan.dense_domains.end(),
                                   [&](const auto &domain)
                                   {
                                       return domain.name == name;
                                   });
            return it == plan.dense_domains.end() ? nullptr : &*it;
        }

        int participantIndexForDenseDomain(
            const ExecutionDomainDefinition &domain,
            const std::shared_ptr<IMPIContext> &runner_mpi_ctx)
        {
            const int rank = runner_mpi_ctx ? runner_mpi_ctx->rank() : 0;
            for (size_t index = 0; index < domain.ranks.size(); ++index)
            {
                if (domain.ranks[index] == rank)
                    return static_cast<int>(index);
            }

            if (domain.owner_rank.has_value() && *domain.owner_rank == rank)
                return 0;

            return 0;
        }

        TPDomainParticipation denseDomainParticipation(
            const ExecutionDomainDefinition &domain,
            const std::shared_ptr<IMPIContext> &runner_mpi_ctx)
        {
            TPDomainParticipation participation;
            participation.domain_id = stableContinuationDomainId(domain.name);
            participation.domain_name = domain.name;
            participation.devices = domain.participants;
            participation.weights = domain.weights;
            participation.backend = domain.backend;
            participation.my_index_in_domain = participantIndexForDenseDomain(domain, runner_mpi_ctx);

            if ((domain.scope == ExecutionDomainScope::NODE_LOCAL ||
                 domain.scope == ExecutionDomainScope::GLOBAL) &&
                participation.backend == CollectiveBackendType::AUTO)
            {
                participation.backend = CollectiveBackendType::UPI;
            }

            return participation;
        }

        int overlayRankFor(
            const std::shared_ptr<IMPIContext> &overlay_mpi_ctx,
            const std::shared_ptr<IMPIContext> &runner_mpi_ctx)
        {
            if (overlay_mpi_ctx)
                return overlay_mpi_ctx->rank();
            if (runner_mpi_ctx)
                return runner_mpi_ctx->rank();
            return 0;
        }

        bool populateMoEContinuationDomainTPContextsForGraph(
            GraphConfig &graph_config,
            DomainTPContextMap &owned_domain_tp_contexts,
            const std::shared_ptr<IMPIContext> &runner_mpi_ctx,
            const std::string &log_prefix,
            const InferenceRunnerConfig *runner_config)
        {
            const auto &plan = graph_config.moe.expert_parallel_plan;
            if (!plan || !plan->isTieredOverlay())
                return true;

            ITPContext *injected_tp_context = runner_config ? runner_config->tp_ctx : nullptr;
            MPI_Comm base_comm = runner_mpi_ctx ? runner_mpi_ctx->communicator() : MPI_COMM_WORLD;

            for (const auto &domain_name : denseMoEDomainNames(*plan))
            {
                const auto *domain = findDenseMoEDomain(*plan, domain_name);
                if (!domain || domain->participants.size() <= 1)
                    continue;

                if (graph_config.domain_tp_contexts.find(domain_name) != graph_config.domain_tp_contexts.end())
                    continue;

                if (injected_tp_context && domain_name == plan->continuation_domain &&
                    injected_tp_context->degree() == static_cast<int>(domain->participants.size()))
                {
                    graph_config.domain_tp_contexts[domain_name] = injected_tp_context;
                    LOG_INFO(log_prefix << " using injected "
                                        << tpScopeToString(injected_tp_context->scope())
                                        << " TP context for MoE continuation dense domain '"
                                        << domain_name << "'");
                    continue;
                }

                if (owned_domain_tp_contexts.find(domain_name) == owned_domain_tp_contexts.end())
                {
                    auto participation = denseDomainParticipation(*domain, runner_mpi_ctx);
                    auto ctx = TPContextFactory::createFromDomain(participation, base_comm);
                    if (!ctx)
                    {
                        LOG_WARN(log_prefix << " failed to create TP context for MoE continuation dense domain '"
                                            << domain_name << "'");
                        return false;
                    }
                    owned_domain_tp_contexts.emplace(domain_name, std::shared_ptr<ITPContext>(std::move(ctx)));
                }

                graph_config.domain_tp_contexts[domain_name] = owned_domain_tp_contexts.at(domain_name).get();
            }

            return true;
        }

        bool applyMoEExpertOverlayConfigToGraph(
            IModelContext &model_ctx,
            const InferenceRunnerConfig &config,
            const std::shared_ptr<IMPIContext> &runner_mpi_ctx,
            GraphConfig &graph_config,
            DomainTPContextMap &owned_domain_tp_contexts,
            const std::string &log_prefix)
        {
            auto plan = resolveMoEExpertParallelPlanForModel(model_ctx, config);
            if (!plan)
                return true;

            graph_config.moe.expert_parallel_plan = plan;
            graph_config.moe.overlay_mpi_ctx = config.moe_expert_overlay_mpi_ctx
                                                   ? config.moe_expert_overlay_mpi_ctx
                                                   : runner_mpi_ctx;

            if (plan->isTieredOverlay())
            {
                graph_config.moe.expert_overlay_runtime_plan.reset();
                graph_config.moe.expert_overlay_execution_plan.reset();
                LOG_INFO(log_prefix << " using graph-native MoE overlay lowering for tiered expert overlay");

                if (!populateMoEContinuationDomainTPContextsForGraph(
                        graph_config,
                        owned_domain_tp_contexts,
                        runner_mpi_ctx,
                        log_prefix,
                        &config))
                {
                    return false;
                }
            }

            return true;
        }
    } // namespace

    bool applyMoEExpertOverlayConfigToGraphForTesting(
        IModelContext &model_ctx,
        const InferenceRunnerConfig &config,
        const std::shared_ptr<IMPIContext> &runner_mpi_ctx,
        GraphConfig &graph_config,
        DomainTPContextMap &owned_domain_tp_contexts,
        const std::string &log_prefix)
    {
        return applyMoEExpertOverlayConfigToGraph(
            model_ctx,
            config,
            runner_mpi_ctx,
            graph_config,
            owned_domain_tp_contexts,
            log_prefix);
    }

    std::shared_ptr<MoEExpertParallelPlan> resolveMoEExpertParallelPlanForModel(
        IModelContext &model_ctx,
        const InferenceRunnerConfig &config)
    {
        auto plan = config.moe_expert_parallel_plan;
        if (!plan)
            return nullptr;
        if (!plan->enabled || !plan->isTieredOverlay() || !plan->placements.empty())
            return plan;

        auto planner_result = MoEExpertParallelPlanner::plan(
            *plan,
            moEExpertMetadataForModel(model_ctx));
        return std::make_shared<MoEExpertParallelPlan>(std::move(planner_result.planned_plan));
    }

    static PreparedWeightKind expectedGemmPreparedKind(DeviceId device)
    {
        if (device.is_cuda())
            return PreparedWeightKind::CudaInt8PackedGemm;
        if (device.is_rocm())
            return PreparedWeightKind::RocmInt8PackedGemm;
        return PreparedWeightKind::CpuPackedGemm;
    }

    static bool isMoEExpertTensorName(const std::string &name)
    {
        return name.find("_exps.weight") != std::string::npos;
    }

    static PreparedWeightKind expectedPreparedKindForWeight(
        const WeightManager &weight_mgr,
        const std::string &name,
        DeviceId device)
    {
        if (isMoEExpertTensorName(name))
            return PreparedWeightKind::None;
        try
        {
            if (weight_mgr.isGemmWeight(name))
                return expectedGemmPreparedKind(device);
        }
        catch (const std::exception &e)
        {
            LOG_WARN("[InferenceRunner] Could not classify prepared kind for " << name << ": " << e.what());
        }
        return PreparedWeightKind::None;
    }

    static WeightManagerConfig makeWeightManagerConfigForGraph(
        const std::string &architecture,
        const GraphConfig &graph_config,
        bool include_tp_config)
    {
        const WeightShardingMode expert_mode =
            graph_config.moe.expert_mode == MoEExpertMode::Replicated
                ? WeightShardingMode::Replicate
                : WeightShardingMode::ExpertParallel;

        WeightManagerConfig wm_config;
        wm_config.sharding = SchemaFactoryRegistry::getWeightShardingConfig(architecture);
        if (architecture == "qwen35moe")
        {
            for (auto &pattern : wm_config.sharding.patterns)
            {
                if (pattern.pattern == "ffn_gate_exps.weight" ||
                    pattern.pattern == "ffn_up_exps.weight" ||
                    pattern.pattern == "ffn_down_exps.weight")
                {
                    pattern.mode = expert_mode;
                    pattern.description = graph_config.moe.expert_mode == MoEExpertMode::Replicated
                                              ? "MoE routed expert weights - replicated"
                                              : "MoE routed expert weights - expert-id parallel";
                }
            }
        }
        if (graph_config.n_heads > 0 && graph_config.head_dim > 0)
        {
            wm_config.dimensions.n_heads = graph_config.n_heads;
            wm_config.dimensions.n_kv_heads = graph_config.n_kv_heads;
            wm_config.dimensions.head_dim = graph_config.head_dim;
        }
        if (graph_config.gdn.enabled())
        {
            wm_config.dimensions.gdn_n_k_heads = graph_config.gdn.group_count;
            wm_config.dimensions.gdn_n_v_heads = graph_config.gdn.time_step_rank;
            wm_config.dimensions.gdn_d_state = graph_config.gdn.state_size;
        }
        if (include_tp_config)
        {
            wm_config.tp_config = graph_config.tp_config;
        }
        return wm_config;
    }

    static void configureWeightManagerForGraph(
        WeightManager &weight_mgr,
        const std::string &architecture,
        const GraphConfig &graph_config,
        bool include_tp_config)
    {
        weight_mgr.configure(makeWeightManagerConfigForGraph(
            architecture, graph_config, include_tp_config));
    }

    static WeightRequirement makeSingleDeviceRequirement(
        const WeightManager &weight_mgr,
        const std::string &canonical_name,
        DeviceId device,
        bool required = true,
        WeightRole role = WeightRole::Other,
        std::string source_name = {},
        WeightDerivationKind derivation = WeightDerivationKind::Source,
        std::optional<DeviceId> lookup_device = DeviceId::cpu(),
        int tp_rank_or_device_index = 0,
        int tp_domain = -1,
        int pp_stage = -1,
        WeightSliceSpec slice = {})
    {
        WeightRequirement requirement;
        requirement.canonical_name = canonical_name;
        requirement.source_name = std::move(source_name);
        requirement.required = required;
        requirement.role = role;
        requirement.derivation = derivation;
        requirement.tp_domain = tp_domain;
        requirement.tp_rank_or_device_index = tp_rank_or_device_index;
        requirement.pp_stage = pp_stage;
        requirement.target_device = device;
        requirement.lookup_device = lookup_device;
        requirement.host_policy = device.is_cpu()
                                      ? WeightHostPolicy::RequiredForCPUExecution
                                      : WeightHostPolicy::RequiredUntilGraphMaterialized;
        requirement.expected_prepared_kind = role == WeightRole::Embedding
                                                 ? PreparedWeightKind::None
                                                 : expectedPreparedKindForWeight(weight_mgr, canonical_name, device);
        requirement.slice = slice;
        return requirement;
    }

    static WeightSliceSpec vocabSliceSpecForAssignment(
        const TensorParallelConfig *tp_config,
        DeviceId device,
        int tp_rank_override = -1)
    {
        WeightSliceSpec slice;
        if (!tp_config)
            return slice;
        try
        {
            const auto &assignment = tp_rank_override >= 0
                                         ? tp_config->forRank(tp_rank_override)
                                         : tp_config->forDevice(device);
            slice.source_rows = static_cast<size_t>(tp_config->totalVocab());
            slice.row_start = static_cast<size_t>(assignment.vocab_start);
            slice.row_count = static_cast<size_t>(assignment.vocab_count);
        }
        catch (const std::out_of_range &)
        {
        }
        return slice;
    }

    static WeightPlan buildSingleDeviceWeightPlan(
        const WeightManager &weight_mgr,
        const ModelContext &model_ctx,
        const WeightValidationResult &validation,
        DeviceId device,
        const TensorParallelConfig *tp_config = nullptr,
        const FactoryPPStageConfig *pp_config = nullptr,
        int tp_rank_override = -1)
    {
        InferenceStrategy strategy;
        const bool has_tp = tp_config && tp_config->worldSize() > 1;
        const bool has_pp = pp_config != nullptr;
        strategy.mode = has_pp && has_tp ? WeightInferenceMode::HybridPPTP
                        : has_pp         ? WeightInferenceMode::LocalPP
                        : has_tp         ? WeightInferenceMode::LocalTP
                                         : WeightInferenceMode::SingleDevice;
        strategy.model_id = ModelContextId{reinterpret_cast<uint64_t>(&model_ctx)};
        strategy.pp_stages = has_pp ? 2 : 1;
        strategy.tp_degree = tp_config ? tp_config->worldSize() : 1;
        if (tp_config)
        {
            for (const auto &assignment : tp_config->assignments())
                strategy.devices.push_back(assignment.device);
        }
        else
        {
            strategy.devices = {device};
        }

        WeightPlan plan(strategy);

        int tp_rank_or_device_index = 0;
        int tp_domain = -1;
        std::optional<DeviceId> lookup_device = DeviceId::cpu();
        if (tp_config)
        {
            const auto &assignment = tp_rank_override >= 0
                                         ? tp_config->forRank(tp_rank_override)
                                         : tp_config->forDevice(device);
            tp_rank_or_device_index = assignment.local_rank;
            tp_domain = 0;
            lookup_device = device;
        }

        const WeightSliceSpec vocab_slice = vocabSliceSpecForAssignment(tp_config, device, tp_rank_override);
        const int pp_stage = pp_config ? pp_config->first_layer : -1;

        if (!pp_config || pp_config->has_embedding)
        {
            plan.add(makeSingleDeviceRequirement(
                weight_mgr,
                "token_embd.weight",
                device,
                true,
                WeightRole::Embedding,
                {},
                WeightDerivationKind::Source,
                lookup_device,
                tp_rank_or_device_index,
                tp_domain,
                pp_stage,
                vocab_slice));
        }

        if (!pp_config || pp_config->has_lm_head)
        {
            plan.add(makeSingleDeviceRequirement(
                weight_mgr,
                "output_norm.weight",
                device,
                true,
                WeightRole::OutputNorm,
                {},
                WeightDerivationKind::Source,
                lookup_device,
                tp_rank_or_device_index,
                tp_domain,
                pp_stage));

            if (model_ctx.hasTensor("output.weight"))
            {
                plan.add(makeSingleDeviceRequirement(
                    weight_mgr,
                    "output.weight",
                    device,
                    true,
                    WeightRole::LMHead,
                    {},
                    WeightDerivationKind::Source,
                    lookup_device,
                    tp_rank_or_device_index,
                    tp_domain,
                    pp_stage,
                    vocab_slice));
            }
            else
            {
                const std::string tied_lm_head_source = tp_config ? "output.weight" : "token_embd.weight";
                plan.add(makeSingleDeviceRequirement(
                    weight_mgr,
                    "output.weight",
                    device,
                    true,
                    WeightRole::LMHead,
                    tied_lm_head_source,
                    WeightDerivationKind::TiedAlias,
                    lookup_device,
                    tp_rank_or_device_index,
                    tp_domain,
                    pp_stage,
                    vocab_slice));
            }
        }

        for (const auto &[weight_name, is_optional] : validation.weights_to_load)
        {
            plan.add(makeSingleDeviceRequirement(
                weight_mgr,
                weight_name,
                device,
                !is_optional,
                WeightRole::Other,
                {},
                WeightDerivationKind::Source,
                lookup_device,
                tp_rank_or_device_index,
                tp_domain,
                pp_stage));
        }

        return plan;
    }

    // =========================================================================
    // Helper: Build ClusterInventory for GPU Collective Context
    // =========================================================================

    /**
     * @brief Build a ClusterInventory from available GPU backends
     *
     * Detects local CUDA and ROCm GPUs via backend APIs and builds a
     * single-rank ClusterInventory suitable for CollectiveContextFactory.
     *
     * For multi-node MPI, each rank builds its local inventory, and
     * the full cluster inventory would be built via MPI_Allgather.
     * For now, we support single-node with multiple GPUs.
     *
     * @param mpi_ctx MPI context (for rank info)
     * @return ClusterInventory with detected devices
     */
    static ClusterInventory buildLocalClusterInventory(
        const std::shared_ptr<IMPIContext> &mpi_ctx)
    {
        ClusterInventory inventory;
        RankInventory rank_inv;

        rank_inv.rank = mpi_ctx ? mpi_ctx->rank() : 0;
        rank_inv.node_id = 0; // Single-node for now
        rank_inv.local_rank = rank_inv.rank;
        rank_inv.hostname = "localhost";

#ifdef HAVE_CUDA
        IBackend *cuda_backend = getCUDABackend();
        if (cuda_backend)
        {
            int cuda_count = cuda_backend->deviceCount();
            for (int i = 0; i < cuda_count; ++i)
            {
                DeviceInfo gpu;
                gpu.type = DeviceType::CUDA;
                gpu.local_device_id = i;
                gpu.memory_bytes = cuda_backend->deviceMemoryTotal(i);
                gpu.name = cuda_backend->deviceName(i);
                gpu.supports_p2p = true;
                rank_inv.gpus.push_back(gpu);
            }
            LOG_DEBUG("[InferenceRunner] Detected " << cuda_count << " CUDA GPU(s)");
        }
#endif

#ifdef HAVE_ROCM
        IBackend *rocm_backend = getROCmBackend();
        if (rocm_backend)
        {
            int rocm_count = rocm_backend->deviceCount();
            for (int i = 0; i < rocm_count; ++i)
            {
                DeviceInfo gpu;
                gpu.type = DeviceType::ROCm;
                gpu.local_device_id = i;
                gpu.memory_bytes = rocm_backend->deviceMemoryTotal(i);
                gpu.name = rocm_backend->deviceName(i);
                rank_inv.gpus.push_back(gpu);
            }
            LOG_DEBUG("[InferenceRunner] Detected " << rocm_count << " ROCm GPU(s)");
        }
#endif

        inventory.ranks.push_back(rank_inv);
        inventory.world_size = mpi_ctx ? mpi_ctx->world_size() : 1;
        inventory.buildNodeAggregations();

        return inventory;
    }

    // =========================================================================
    // Helper: Resolve effective fused attention backend
    // =========================================================================

    /**
     * @brief Resolve the effective fused attention backend
     *
     * HybridQ16 mode is incompatible with JIT backend (JIT doesn't support Q16_1),
     * so we auto-select Q16_INTEGER in that case.
     *
     * @param precision Activation precision mode
     * @param requested Requested backend
     * @return Effective backend to use
     */
    static FusedAttentionBackend resolveEffectiveAttentionBackend(
        ActivationPrecision precision,
        FusedAttentionBackend requested)
    {
        if (precision == ActivationPrecision::HybridQ16 &&
            requested == FusedAttentionBackend::JIT)
        {
            LOG_DEBUG("[InferenceRunner] HybridQ16 mode: auto-selecting Q16_INTEGER backend (JIT doesn't support Q16_1)");
            return FusedAttentionBackend::Q16_INTEGER;
        }
        return requested;
    }

    static MoERebalanceMode toControllerRebalanceMode(MoERebalanceRuntimeMode mode)
    {
        switch (mode)
        {
        case MoERebalanceRuntimeMode::Observe:
            return MoERebalanceMode::OBSERVE;
        case MoERebalanceRuntimeMode::Dynamic:
            return MoERebalanceMode::DYNAMIC;
        case MoERebalanceRuntimeMode::Off:
        default:
            return MoERebalanceMode::OFF;
        }
    }

    static MoERebalanceRuntimeConfig effectiveMoERebalanceConfig(
        const InferenceRunnerConfig &config)
    {
        MoERebalanceRuntimeConfig result = config.moe_rebalance;
        const auto &env = debugEnv();

        if (std::getenv("LLAMINAR_MOE_REBALANCE"))
        {
            if (auto parsed = parseMoERebalanceRuntimeMode(env.moe_rebalance.mode))
                result.mode = *parsed;
            else
                LOG_WARN("[InferenceRunner] Ignoring invalid LLAMINAR_MOE_REBALANCE='"
                         << env.moe_rebalance.mode << "'");
        }
        if (std::getenv("LLAMINAR_MOE_REBALANCE_WINDOW"))
            result.window_size = env.moe_rebalance.window_size;
        if (std::getenv("LLAMINAR_MOE_REBALANCE_MAX_WINDOW"))
            result.max_window_size = env.moe_rebalance.max_window_size;
        if (std::getenv("LLAMINAR_MOE_REBALANCE_WINDOW_GROWTH"))
            result.window_growth_factor = env.moe_rebalance.window_growth_factor;
        result.release_raw_expert_weights = result.release_raw_expert_weights ||
                                            env.moe_rebalance.release_raw_weights;

        return result;
    }

    // =========================================================================
    // Helper: Set full (non-TP) dimensions on graph config
    // =========================================================================

    /**
     * @brief Configure graph config for single-device (no tensor parallelism)
     *
     * Sets all TP-related fields to use the full model dimensions,
     * disabling column-parallel sharding.
     */
    static void setFullDimensions(GraphConfig &graph_config)
    {
        graph_config.head_start = 0;
        graph_config.local_n_heads = graph_config.n_heads;
        graph_config.local_n_kv_heads = graph_config.n_kv_heads;
        graph_config.qkv_column_parallel = false;
        graph_config.local_rank = 0;

        graph_config.d_ff_local = graph_config.d_ff;
        graph_config.ffn_column_parallel = false;

        graph_config.vocab_local = graph_config.vocab_size;
        graph_config.lm_head_column_parallel = false;
    }

    static bool configureStaticMoEExpertRange(GraphConfig &graph_config)
    {
        graph_config.moe.local_expert_start = 0;
        graph_config.moe.local_expert_count = -1;

        if (!graph_config.moe.enabled())
            return true;

        if (graph_config.moe.expert_mode == MoEExpertMode::TensorParallel)
        {
            LOG_ERROR("[InferenceRunner] Not Implemented: MoE tensor-parallel expert mode is not implemented for the standard Qwen3.5 MoE path.");
            return false;
        }

        if (graph_config.moe.expert_mode != MoEExpertMode::ExpertParallel)
            return true;

        int participants = 1;
        int participant_index = 0;
        if (graph_config.tp_config && graph_config.tp_config->worldSize() > 1)
        {
            participants = graph_config.tp_config->worldSize();
            participant_index = graph_config.local_rank;
        }
        else if (graph_config.tp_ctx && graph_config.tp_ctx->degree() > 1)
        {
            participants = graph_config.tp_ctx->degree();
            participant_index = graph_config.tp_ctx->myIndex();
        }

        if (participants <= 1)
            return true;

        if (graph_config.moe.num_experts < participants)
        {
            LOG_ERROR("[InferenceRunner] Cannot expert-shard " << graph_config.moe.num_experts
                                                               << " routed experts across " << participants << " TP participants");
            return false;
        }
        if (participant_index < 0 || participant_index >= participants)
        {
            LOG_ERROR("[InferenceRunner] Invalid MoE expert participant index " << participant_index
                                                                                << " for " << participants << " participants");
            return false;
        }

        const int base = graph_config.moe.num_experts / participants;
        const int remainder = graph_config.moe.num_experts % participants;
        const int count = base + (participant_index < remainder ? 1 : 0);
        const int start = participant_index * base + std::min(participant_index, remainder);

        graph_config.moe.local_expert_start = start;
        graph_config.moe.local_expert_count = count;

        LOG_INFO("[InferenceRunner] MoE expert mode="
                 << moeExpertModeToString(graph_config.moe.expert_mode)
                 << " participant=" << participant_index << "/" << participants
                 << " expert_range=[" << start << ", " << (start + count) << ")"
                 << " count=" << count << "/" << graph_config.moe.num_experts);
        return true;
    }

    // =========================================================================
    // Helper: Apply GLOBAL TP context assignment to graph config
    // =========================================================================

    /**
     * @brief Apply equal-split global TP assignment from an injected IGlobalTPContext.
     *
     * Used when a pre-created IGlobalTPContext (from DomainCommunicatorRegistry or
     * createTestableInferenceRunner) is provided directly instead of being
     * auto-created from MPI_COMM_WORLD.
     *
     * @param graph_config    Config to modify with TP dimensions
     * @param global_tp_ctx   The pre-created global TP context
     * @param device_idx      This rank's index within the TP domain
     * @return true if assignment succeeded, false on invalid input
     */
    static bool applyGlobalTPContextAssignment(
        GraphConfig &graph_config,
        IGlobalTPContext *global_tp_ctx,
        int device_idx)
    {
        const int degree = global_tp_ctx->degree();
        if (device_idx < 0 || device_idx >= degree)
        {
            LOG_ERROR("[InferenceRunner] Invalid device_idx " << device_idx
                                                              << " for global TP context (degree=" << degree << ")");
            return false;
        }

        // Equal split: build a TensorParallelConfig based on domain degree
        std::vector<DeviceId> dummy_devices(degree, DeviceId::cpu());
        auto tp_config = TensorParallelConfig::equalSplit(
            degree,
            graph_config.n_heads,
            graph_config.n_kv_heads,
            graph_config.d_ff,
            graph_config.vocab_size,
            dummy_devices);
        const auto &assignment = tp_config.forRank(device_idx);

        graph_config.tp_config = std::make_shared<TensorParallelConfig>(tp_config);
        graph_config.local_rank = device_idx;

        graph_config.head_start = assignment.head_start;
        graph_config.local_n_heads = assignment.head_count;
        graph_config.local_n_kv_heads = assignment.kv_head_count;
        graph_config.qkv_column_parallel = true;

        graph_config.d_ff_local = assignment.d_ff_count;
        graph_config.ffn_column_parallel = true;

        graph_config.vocab_local = assignment.vocab_count;
        graph_config.lm_head_column_parallel = true;

        // Store context so graph builders use TPAllreduceStage
        graph_config.tp_ctx = global_tp_ctx;
        graph_config.tp_device_idx = device_idx;

        LOG_INFO("[InferenceRunner] Injected GlobalTPContext: degree=" << degree
                                                                       << " device_idx=" << device_idx
                                                                       << " domainId=" << global_tp_ctx->domainId());
        LOG_DEBUG("[InferenceRunner] Global TP QKV: head_start=" << assignment.head_start
                                                                 << " local_n_heads=" << assignment.head_count << "/" << graph_config.n_heads);
        LOG_DEBUG("[InferenceRunner] Global TP FFN: d_ff_local=" << assignment.d_ff_count << "/" << graph_config.d_ff);
        LOG_DEBUG("[InferenceRunner] Global TP LMHead: vocab_local=" << assignment.vocab_count << "/" << graph_config.vocab_size);

        return true;
    }

    // =========================================================================
    // Helper: Apply LOCAL TP assignment to graph config
    // =========================================================================

    /**
     * @brief Apply LOCAL TP head/FFN/vocab assignment based on device weights
     *
     * Uses proportional assignment from ILocalTPContext to compute which heads,
     * FFN dimensions, and vocab slices this device is responsible for.
     * The last device in the TP group gets remainder to handle rounding.
     *
     * @param graph_config Config to modify with TP dimensions
     * @param local_tp_ctx LOCAL TP context with device list and weights
     * @param device_idx Index of this device within the TP group
     * @return true if assignment succeeded, false on invalid input
     */
    static bool applyLocalTPAssignment(
        GraphConfig &graph_config,
        ILocalTPContext *local_tp_ctx,
        int device_idx)
    {
        const int tp_degree = local_tp_ctx->degree();

        if (device_idx < 0 || device_idx >= tp_degree)
        {
            LOG_ERROR("[InferenceRunner] Invalid local_tp_device_index: " << device_idx
                                                                          << " (degree=" << tp_degree << ")");
            return false;
        }

        // Use TensorParallelConfig for consistent head/FFN/vocab distribution.
        // This ensures the graph config matches the weight slicing exactly
        // (both use distributeProportionally instead of independent std::round).
        auto tp_config = TensorParallelConfig::fromLocalTPContext(
            *local_tp_ctx,
            graph_config.n_heads,
            graph_config.n_kv_heads,
            graph_config.d_ff,
            graph_config.vocab_size);

        const auto &assignment = tp_config.forRank(device_idx);

        graph_config.head_start = assignment.head_start;
        graph_config.local_n_heads = assignment.head_count;
        graph_config.local_n_kv_heads = assignment.kv_head_count;
        graph_config.qkv_column_parallel = true;
        graph_config.local_rank = device_idx;

        graph_config.d_ff_local = assignment.d_ff_count;
        graph_config.ffn_column_parallel = true;

        graph_config.vocab_local = assignment.vocab_count;
        graph_config.lm_head_column_parallel = true;

        // Store TP context for collective operations (polymorphic via ITPContext)
        graph_config.tp_ctx = local_tp_ctx;
        graph_config.tp_device_idx = device_idx;

        // Store the TensorParallelConfig so downstream code can access it
        graph_config.tp_config = std::make_shared<TensorParallelConfig>(tp_config);

        const auto &devices = local_tp_ctx->devices();
        const auto &weights = local_tp_ctx->weights();
        const float my_weight = weights.empty() ? (1.0f / tp_degree) : weights[device_idx];
        LOG_INFO("[InferenceRunner] LOCAL TP enabled: degree=" << tp_degree
                                                               << " device_idx=" << device_idx
                                                               << " device=" << devices[device_idx].toString()
                                                               << " weight=" << (my_weight * 100.0f) << "%"
                                                               << " backend=" << static_cast<int>(local_tp_ctx->backend()));
        LOG_DEBUG("[InferenceRunner] LOCAL TP QKV: head_start=" << assignment.head_start
                                                                << " local_n_heads=" << assignment.head_count << "/" << graph_config.n_heads
                                                                << " local_n_kv_heads=" << assignment.kv_head_count << "/" << graph_config.n_kv_heads);
        LOG_DEBUG("[InferenceRunner] LOCAL TP FFN: d_ff_local=" << assignment.d_ff_count << "/" << graph_config.d_ff);
        LOG_DEBUG("[InferenceRunner] LOCAL TP LMHead: vocab_local=" << assignment.vocab_count << "/" << graph_config.vocab_size);

        return true;
    }

    // =========================================================================
    // Helper: Apply PROPORTIONAL GLOBAL TP assignment to graph config
    // =========================================================================

    /**
     * @brief Apply proportional TP assignment from TensorParallelConfig
     *
     * Used for heterogeneous GLOBAL TP (e.g., NVIDIA 73% + AMD 27%).
     * Gets per-rank DeviceShardingAssignment from the TensorParallelConfig.
     *
     * @param graph_config Config to modify with TP dimensions
     * @param tp_config Tensor parallel configuration with per-device assignments
     * @param current_rank MPI rank index for assignment lookup
     */
    static void applyProportionalGlobalTPAssignment(
        GraphConfig &graph_config,
        const TensorParallelConfig *tp_config,
        int current_rank)
    {
        const DeviceShardingAssignment &assignment = tp_config->forRank(current_rank);

        // Store config and rank in graph_config for downstream use
        graph_config.tp_config = std::make_shared<TensorParallelConfig>(*tp_config);
        graph_config.local_rank = current_rank;

        // QKV head assignment
        graph_config.head_start = assignment.head_start;
        graph_config.local_n_heads = assignment.head_count;
        graph_config.local_n_kv_heads = assignment.kv_head_count;
        graph_config.qkv_column_parallel = true;

        // FFN dimension assignment
        graph_config.d_ff_local = assignment.d_ff_count;
        graph_config.ffn_column_parallel = true;

        // Vocab/LM head assignment
        graph_config.vocab_local = assignment.vocab_count;
        graph_config.lm_head_column_parallel = true;

        LOG_INFO("[InferenceRunner] Using TensorParallelConfig (proportional split): "
                 << "rank=" << current_rank << "/" << tp_config->worldSize()
                 << " device=" << assignment.device.to_string()
                 << " work_fraction=" << (assignment.work_fraction * 100.0f) << "%");
        LOG_DEBUG("[InferenceRunner] QKV: head_start=" << graph_config.head_start
                                                       << " local_n_heads=" << graph_config.local_n_heads << "/" << graph_config.n_heads
                                                       << " local_n_kv_heads=" << graph_config.local_n_kv_heads << "/" << graph_config.n_kv_heads);
        LOG_DEBUG("[InferenceRunner] FFN: d_ff_local=" << graph_config.d_ff_local << "/" << graph_config.d_ff);
        LOG_DEBUG("[InferenceRunner] LMHead: vocab_local=" << graph_config.vocab_local << "/" << graph_config.vocab_size);
    }

    // =========================================================================
    // Helper: Apply EQUAL-SPLIT GLOBAL TP assignment to graph config
    // =========================================================================

    /**
     * @brief Apply equal 1/world_size TP split via MPI context
     *
     * Used for homogeneous GLOBAL TP where all ranks get equal work.
     * Validates that FFN and vocab dimensions are evenly divisible.
     *
     * @param graph_config Config to modify with TP dimensions
     * @param mpi_ctx MPI context with rank/world_size info
     * @return true if assignment succeeded, false if dimensions not divisible
     */
    static bool applyEqualSplitGlobalTPAssignment(
        GraphConfig &graph_config,
        const std::shared_ptr<IMPIContext> &mpi_ctx)
    {
        const int current_rank = mpi_ctx->rank();
        const int world_size = mpi_ctx->world_size();

        std::vector<DeviceId> devices(world_size, graph_config.default_device);
        auto tp_config = TensorParallelConfig::equalSplit(
            world_size,
            graph_config.n_heads,
            graph_config.n_kv_heads,
            graph_config.d_ff,
            graph_config.vocab_size,
            devices);
        const auto &assignment = tp_config.forRank(current_rank);

        graph_config.tp_config = std::make_shared<TensorParallelConfig>(tp_config);
        graph_config.head_start = assignment.head_start;
        graph_config.local_n_heads = assignment.head_count;
        graph_config.local_n_kv_heads = assignment.kv_head_count;
        graph_config.qkv_column_parallel = true;
        graph_config.local_rank = current_rank;

        LOG_DEBUG("[InferenceRunner] QKV Column-Parallel enabled (equal split): "
                  << "head_start=" << graph_config.head_start
                  << ", local_n_heads=" << graph_config.local_n_heads << "/" << graph_config.n_heads
                  << ", local_n_kv_heads=" << graph_config.local_n_kv_heads << "/" << graph_config.n_kv_heads
                  << " (rank " << current_rank << "/" << world_size << ")");

        // FFN dimension (equal split)
        if (graph_config.d_ff % world_size != 0)
        {
            LOG_ERROR("[InferenceRunner] d_ff (" << graph_config.d_ff
                                                 << ") not divisible by world_size (" << world_size << ")");
            return false;
        }
        graph_config.d_ff_local = assignment.d_ff_count;
        graph_config.ffn_column_parallel = true;

        LOG_DEBUG("[InferenceRunner] FFN Column-Parallel enabled (equal split): "
                  << "d_ff_local=" << graph_config.d_ff_local << "/" << graph_config.d_ff
                  << " (rank " << current_rank << "/" << world_size << ")");

        // Vocab/LM head (equal split)
        if (graph_config.vocab_size % world_size != 0)
        {
            LOG_ERROR("[InferenceRunner] vocab_size (" << graph_config.vocab_size
                                                       << ") not divisible by world_size (" << world_size << ")");
            return false;
        }
        graph_config.vocab_local = assignment.vocab_count;
        graph_config.lm_head_column_parallel = true;

        LOG_DEBUG("[InferenceRunner] LM Head Column-Parallel enabled (equal split): "
                  << "vocab_local=" << graph_config.vocab_local << "/" << graph_config.vocab_size
                  << " (rank " << current_rank << "/" << world_size << ")");

        return true;
    }

    // =========================================================================
    // Helper: Select and apply TP assignment strategy
    // =========================================================================

    /**
     * @brief Select and apply the appropriate TP assignment to GraphConfig
     *
     * Precedence (highest to lowest):
     *   1. LOCAL TP — single rank, multiple devices via ILocalTPContext
     *   2. Proportional GLOBAL TP — heterogeneous multi-rank via TensorParallelConfig
     *   3. Equal-split GLOBAL TP — homogeneous multi-rank via MPI equal slicing
     *   4. No TP — single rank or replicated weights, use full dimensions
     *
     * @return true if assignment succeeded, false on error
     */
    static bool applyTPAssignment(
        GraphConfig &graph_config,
        ILocalTPContext *local_tp_ctx,
        int tp_device_idx,
        const TensorParallelConfig *tp_config,
        const std::shared_ptr<IMPIContext> &mpi_ctx,
        bool weights_sharded)
    {
        // LOCAL TP activation: also activate if tp_config is set on WeightManager,
        // which indicates MDO configured it for TP weight slicing within a PP stage
        const bool local_tp_weights_configured = tp_config != nullptr;

        if (local_tp_ctx && local_tp_ctx->degree() > 1 && (weights_sharded || local_tp_weights_configured))
        {
            return applyLocalTPAssignment(graph_config, local_tp_ctx, tp_device_idx);
        }

        if (tp_config && weights_sharded)
        {
            const int current_rank = mpi_ctx ? mpi_ctx->rank() : 0;
            applyProportionalGlobalTPAssignment(graph_config, tp_config, current_rank);
            return true;
        }

        if (mpi_ctx && mpi_ctx->world_size() > 1 && weights_sharded)
        {
            return applyEqualSplitGlobalTPAssignment(graph_config, mpi_ctx);
        }

        // No TP — use full dimensions
        setFullDimensions(graph_config);

        if (mpi_ctx && mpi_ctx->world_size() > 1 && !weights_sharded)
        {
            LOG_WARN("[InferenceRunner] MPI world_size > 1 but weights are REPLICATED, "
                     << "not SHARDED. Using full buffer sizes (no tensor parallelism). "
                     << "Pass WeightDistributionStrategy::SHARDED to ModelContext::create() "
                     << "to enable tensor parallelism.");
        }

        return true;
    }

    // =========================================================================
    // Factory Function
    // =========================================================================

    std::unique_ptr<IInferenceRunner> createInferenceRunner(
        std::shared_ptr<ModelContext> model_ctx,
        std::shared_ptr<IMPIContext> mpi_ctx,
        DeviceId device,
        const InferenceRunnerConfig &config)
    {
        LOG_DEBUG("[InferenceRunner] createInferenceRunner called with mpi_ctx="
                  << (mpi_ctx ? "valid" : "nullptr")
                  << " world_size=" << (mpi_ctx ? mpi_ctx->world_size() : -1));

        if (!model_ctx)
        {
            LOG_ERROR("[InferenceRunner] model_ctx is null");
            return nullptr;
        }

        // Validate device
        if (!device.is_valid())
        {
            LOG_ERROR("[InferenceRunner] Invalid device " << device
                                                          << ". Use DeviceId::cpu() for CPU.");
            return nullptr;
        }
        LOG_DEBUG("[InferenceRunner] Using device " << device);

        // Graph is the only execution path (as of January 2025 cleanup)
        std::string architecture = model_ctx->architecture();
        LOG_DEBUG("[InferenceRunner] Using GRAPH path");
        return createDeviceGraphOrchestratorImpl(model_ctx, mpi_ctx, device, config, architecture);
    }

    // =========================================================================
    // Factory Helper Implementations
    // =========================================================================

    static std::unique_ptr<IInferenceRunner> createDeviceGraphOrchestratorImpl(
        std::shared_ptr<ModelContext> model_ctx,
        std::shared_ptr<IMPIContext> mpi_ctx,
        DeviceId device,
        const InferenceRunnerConfig &config,
        const std::string &architecture)
    {
        // Currently only Qwen2 is supported
        if (!SchemaFactoryRegistry::isSupported(architecture))
        {
            LOG_ERROR("[InferenceRunner] Unsupported architecture: " << architecture
                                                                     << ". Supported: " << [&]()
                      {
                          std::string s;
                          for (const auto& a : SchemaFactoryRegistry::supportedArchitectures()) {
                              if (!s.empty()) s += ", ";
                              s += a;
                          }
                          return s; }());
            return nullptr;
        }

        // Configure weight manager from architecture-specific schema
        auto weight_mgr = model_ctx->concreteWeightManager();

        // Get model metadata
        ModelLoader &loader = model_ctx->concreteLoader();
        const auto &model = loader.getModel();

        // Build GraphConfig via polymorphic builder
        auto config_builder = createGraphConfigBuilder(architecture);
        GraphConfig graph_config;
        config_builder->populateFromModelContext(*model_ctx, graph_config);
        graph_config.moe.expert_mode = config.moe_expert_mode;
        graph_config.moe.hot_expert_cache = config.moe_hot_expert_cache;
        graph_config.moe.rebalance_config = effectiveMoERebalanceConfig(config);
        DomainTPContextMap owned_domain_tp_contexts;
        if (!applyMoEExpertOverlayConfigToGraph(
                *model_ctx,
                config,
                mpi_ctx,
                graph_config,
                owned_domain_tp_contexts,
                "[InferenceRunner]"))
        {
            return nullptr;
        }
        // Apply sharding config + model dimensions in a single call
        // (must be after populateFromModelContext which sets n_heads/n_kv_heads/head_dim)
        if (weight_mgr)
        {
            configureWeightManagerForGraph(*weight_mgr, architecture, graph_config, false);
            LOG_DEBUG("[InferenceRunner] Applied " << architecture << " weight config to WeightManager");
        }

        // Execution-specific settings
        graph_config.max_seq_len = config.max_seq_len;
        graph_config.executor_config.cancellation_requested = config.cancellation_requested;
        graph_config.executor_config.stage_failure_callback = config.stage_failure_callback;

        // CRITICAL: Set default device for kernel dispatch
        // This determines which kernels (CPU vs CUDA) are selected for execution
        graph_config.default_device = device;
        LOG_DEBUG("[InferenceRunner] Default device: " << graph_config.default_device.to_string());

        // Propagate activation precision from runtime config
        // This determines buffer types (FP32/Q8_1) and kernel selection
        graph_config.activation_precision = config.activation_precision;
        LOG_DEBUG("[InferenceRunner] Activation precision: " << activationPrecisionToString(config.activation_precision));

        // Propagate fused attention backend selection
        // This determines which kernel implementation to use for fused attention
        FusedAttentionBackend effective_backend = resolveEffectiveAttentionBackend(
            config.activation_precision, config.fused_attention_backend);
        graph_config.fused_attention_backend = effective_backend;
        LOG_DEBUG("[InferenceRunner] Fused attention backend: " << fusedAttentionBackendToString(effective_backend));

        // Propagate kv_cache_precision for Q16_1 / TQ KV cache selection.
        // kv_cache_scale_k/v are set by the model config builder (e.g. QwenStandardGraphConfigBuilder)
        // which is the sole authority on K/V scale values for each model architecture.
        graph_config.kv_cache_precision = config.kv_cache_precision;
        LOG_DEBUG("[InferenceRunner] KV cache scale: K=" << graph_config.kv_cache_scale_k
                                                         << ", V=" << graph_config.kv_cache_scale_v);
        LOG_DEBUG("[InferenceRunner] KV cache precision mode: "
                  << kvCachePrecisionToString(config.kv_cache_precision));

        // Partial pipeline stages, including global/node-local TP stages built
        // through the concrete inference path, execute only their assigned layer
        // range and consume external hidden state when they do not own embedding.
        if (config.pp_stage_config.has_value())
        {
            graph_config.n_layers = config.pp_stage_config->layerCount();
            graph_config.pp_layer_offset = config.pp_stage_config->first_layer;
            LOG_INFO("[InferenceRunner] PP stage graph config: layers=["
                     << config.pp_stage_config->first_layer << ", "
                     << config.pp_stage_config->last_layer << ")"
                     << " has_embedding=" << (config.pp_stage_config->has_embedding ? "yes" : "no")
                     << " has_lm_head=" << (config.pp_stage_config->has_lm_head ? "yes" : "no"));
        }

        // =====================================================================
        // TurboQuant Context (TQ4 / TQ KV Cache)
        // =====================================================================
        // If TQ4 or TQ (split TQ8/TQ4) KV cache precision is requested,
        // create the shared rotation matrix that all layers use for
        // quantize/dequantize.
        std::shared_ptr<TurboQuantContext> turboquant_ctx;
        if (config.kv_cache_precision == KVCachePrecision::TQ4 ||
            config.kv_cache_precision == KVCachePrecision::TQ)
        {
            turboquant_ctx = std::make_shared<TurboQuantContext>(graph_config.head_dim);
            graph_config.turboquant_ctx = turboquant_ctx.get();
            LOG_INFO("[InferenceRunner] TurboQuant context created for "
                     << kvCachePrecisionToString(config.kv_cache_precision)
                     << " KV cache (head_dim=" << graph_config.head_dim << ")");
        }

        // =====================================================================
        // KV Rotation Context (Q16_1 kurtosis reduction)
        // =====================================================================
        // When Q16_1 KV cache precision is active, create a block-diagonal
        // orthogonal rotation that is applied to K/V before quantization and
        // correspondingly to Q before the attention dot product. This spreads
        // outlier energy (from low-frequency RoPE dimensions) across all dims,
        // dramatically reducing clipping at the fixed Q16_1 scale.
        // =====================================================================
        std::shared_ptr<ActivationRotation> kv_rotation;
        if (config.kv_cache_precision == KVCachePrecision::Q16_1 && debugEnv().kv_rotation)
        {
            kv_rotation = std::make_shared<ActivationRotation>(
                graph_config.head_dim, graph_config.head_dim, /*seed=*/42);
            graph_config.kv_rotation = kv_rotation.get();
            LOG_INFO("[InferenceRunner] KV rotation created for Q16_1 cache"
                     << " (block_dim=" << graph_config.head_dim
                     << ", kv_cache_scale_k=" << graph_config.kv_cache_scale_k
                     << ", kv_cache_scale_v=" << graph_config.kv_cache_scale_v << ")");
        }

        // =====================================================================
        // Phase 3+: Tensor-Parallel Configuration
        // =====================================================================
        // Three modes are supported (in order of precedence):
        //
        // A) LOCAL TP (config.tp_ctx->isLocal()): Single MPI rank, multiple devices
        //    - Configured via ITPContext (polymorphic) in InferenceRunnerConfig
        //    - Collectives via NCCL/RCCL/HOST (high bandwidth for same-vendor)
        //    - Uses ILocalTPContext for proportional head/FFN/vocab assignment
        //
        // B) TensorParallelConfig (Phase 1c): Proportional GLOBAL TP
        //    - Used for heterogeneous GLOBAL TP (e.g., NVIDIA 73% + AMD 27%)
        //    - Assignment comes from DeviceShardingAssignment per MPI rank
        //
        // C) Equal split (legacy): 1/world_size equal GLOBAL TP
        //    - Used for homogeneous GLOBAL setups
        //
        // IMPORTANT: Only enable tensor parallelism if weights are actually sharded.
        // If weights are REPLICATED, each rank has the full weight and should use
        // the full head counts to avoid buffer/weight dimension mismatch.
        // =====================================================================
        const bool weights_sharded = weight_mgr &&
                                     (weight_mgr->strategy() == WeightDistributionStrategy::SHARDED);

        // Check if TP context is provided (LOCAL or GLOBAL)
        ITPContext *tp_ctx = config.tp_ctx;
        ILocalTPContext *local_tp_ctx = tp_ctx && tp_ctx->isLocal()
                                            ? static_cast<ILocalTPContext *>(tp_ctx)
                                            : nullptr;
        IGlobalTPContext *injected_global_tp_ctx = (tp_ctx && !tp_ctx->isLocal())
                                                       ? dynamic_cast<IGlobalTPContext *>(tp_ctx)
                                                       : nullptr;
        const int tp_device_idx = config.tp_device_index;

        // Check if TensorParallelConfig is available from WeightManager (for GLOBAL TP)
        const TensorParallelConfig *tp_config = weight_mgr ? weight_mgr->tensorParallelConfig() : nullptr;

        if (injected_global_tp_ctx && injected_global_tp_ctx->degree() > 1)
        {
            if (!applyGlobalTPContextAssignment(graph_config, injected_global_tp_ctx, tp_device_idx))
            {
                return nullptr;
            }
        }
        else if (!applyTPAssignment(graph_config, local_tp_ctx, tp_device_idx,
                                    tp_config, mpi_ctx, weights_sharded))
        {
            return nullptr;
        }
        if (!configureStaticMoEExpertRange(graph_config))
        {
            return nullptr;
        }

        // =====================================================================
        // Create GlobalTPContext for cross-MPI-rank tensor parallelism
        // =====================================================================
        // When GLOBAL TP is active (multi-rank with sharded weights, no LOCAL TP),
        // either use a pre-created context injected via config.tp_ctx, or
        // auto-create one over MPI_COMM_WORLD (legacy path for simple setups).
        // The injected path is used by StageRunnerFactory (Phase 3) so each global
        // TP stage gets a domain-scoped communicator, not a world-wide one.
        // =====================================================================
        std::shared_ptr<IGlobalTPContext> global_tp_ctx;

        if (injected_global_tp_ctx && injected_global_tp_ctx->degree() > 1)
        {
            // Use the provided domain-scoped context; the caller owns its lifetime.
            graph_config.tp_ctx = injected_global_tp_ctx;
            graph_config.tp_device_idx = config.tp_device_index;
            LOG_INFO("[InferenceRunner] Using injected GlobalTPContext: degree="
                     << injected_global_tp_ctx->degree()
                     << " myIndex=" << config.tp_device_index
                     << " domainId=" << injected_global_tp_ctx->domainId());
        }
        else if (mpi_ctx && mpi_ctx->world_size() > 1 && weights_sharded && !local_tp_ctx)
        {
            auto ctx = GlobalTPContext::createWithSplit(
                MPI_COMM_WORLD,
                /*domain_id=*/0,
                /*color=*/0, // All ranks in same domain
                /*key=*/mpi_ctx->rank(),
                config.hostfile);
            if (ctx && ctx->isValid())
            {
                graph_config.tp_ctx = ctx.get();
                graph_config.tp_device_idx = ctx->myIndex();
                global_tp_ctx = std::move(ctx);
                LOG_INFO("[InferenceRunner] GlobalTPContext created: degree="
                         << global_tp_ctx->degree()
                         << " myIndex=" << global_tp_ctx->myIndex()
                         << " backend=" << static_cast<int>(global_tp_ctx->backend()));
            }
            else
            {
                LOG_WARN("[InferenceRunner] Failed to create GlobalTPContext - "
                         "falling back to direct MPI AllreduceStage path");
            }
        }

        installTPExecutorCancellation(graph_config, graph_config.tp_ctx, "[InferenceRunner]");

        LOG_DEBUG("[InferenceRunner] GraphConfig: "
                  << "vocab=" << graph_config.vocab_size
                  << ", d_model=" << graph_config.d_model
                  << ", n_layers=" << graph_config.n_layers
                  << ", n_heads=" << graph_config.n_heads
                  << ", n_kv_heads=" << graph_config.n_kv_heads
                  << ", d_ff=" << graph_config.d_ff);

        // =====================================================================
        // MoE Expert Rebalance Controller (Phase 4a)
        // =====================================================================
        // Must be set BEFORE graph builder creation so the histogram pointer
        // propagates into MoEExpertComputeStage params during graph construction.
        std::unique_ptr<MoERebalanceController> moe_controller;
        {
            const auto rebalance_config = graph_config.moe.rebalance_config;
            const MoERebalanceMode mode = toControllerRebalanceMode(rebalance_config.mode);
            if (mode != MoERebalanceMode::OFF && graph_config.moe.enabled())
            {
                {
                    int world_size = mpi_ctx ? mpi_ctx->world_size() : 1;
                    std::vector<DeviceId> sockets;
                    if (local_tp_ctx && local_tp_ctx->degree() > 1)
                    {
                        for (const auto &device : local_tp_ctx->devices())
                            sockets.push_back(device.toLocalDeviceId());
                        world_size = local_tp_ctx->degree();
                    }
                    else
                    {
                        for (int s = 0; s < world_size; ++s)
                            sockets.push_back(DeviceId(DeviceType::CPU, s));
                    }

                    std::vector<int> initial_placement(graph_config.moe.num_experts);
                    int experts_per_socket = graph_config.moe.num_experts / std::max(1, world_size);
                    for (int e = 0; e < graph_config.moe.num_experts; ++e)
                        initial_placement[e] = std::min(e / std::max(1, experts_per_socket), world_size - 1);

                    MoERebalanceController::Config ctrl_config;
                    ctrl_config.mode = mode;
                    ctrl_config.num_layers = graph_config.n_layers;
                    ctrl_config.num_experts = graph_config.moe.num_experts;
                    ctrl_config.top_k = graph_config.moe.top_k;
                    ctrl_config.window_size = rebalance_config.window_size;
                    ctrl_config.max_window_size = rebalance_config.max_window_size;
                    ctrl_config.window_growth_factor = rebalance_config.window_growth_factor;
                    int effective_replicas = graph_config.moe.hot_expert_cache.resolveCap(
                        graph_config.moe.num_experts,
                        mode == MoERebalanceMode::DYNAMIC);
                    const auto &env = debugEnv();
                    if (std::getenv("LLAMINAR_MOE_REBALANCE_REPLICAS"))
                        effective_replicas = std::max(0, env.moe_rebalance.max_replicas);
                    ctrl_config.max_replicas = effective_replicas;
                    ctrl_config.sockets = std::move(sockets);
                    ctrl_config.initial_expert_to_socket = std::move(initial_placement);

                    moe_controller = std::make_unique<MoERebalanceController>(std::move(ctrl_config));
                    graph_config.moe.decode_histogram = moe_controller->histogram();
                    graph_config.moe.rebalance_mode = moe_controller->mode();

                    LOG_INFO("[InferenceRunner] MoE rebalance controller: mode="
                             << moeRebalanceRuntimeModeToString(rebalance_config.mode)
                             << " max_replicas=" << effective_replicas
                             << " hot_cache=" << graph_config.moe.hot_expert_cache.toString()
                             << " window=" << rebalance_config.window_size
                             << " experts=" << graph_config.moe.num_experts);
                }
            }
        }

        // Create DeviceGraphOrchestrator with config
        LOG_DEBUG("[InferenceRunner] About to create DeviceGraphOrchestrator with mpi_ctx="
                  << (mpi_ctx ? "valid" : "nullptr")
                  << " world_size=" << (mpi_ctx ? mpi_ctx->world_size() : -1));

        std::unique_ptr<DeviceGraphOrchestrator> orchestrator;
        {
            ScopedWeightLoadDetailTimer timer("graph.build.create_orchestrator");
            auto graph_builder = GraphBuilderRegistry::create(architecture, graph_config, mpi_ctx);
            graph_builder->setModelContext(model_ctx);
            orchestrator = std::make_unique<DeviceGraphOrchestrator>(
                std::move(graph_builder), mpi_ctx);
        }

        if (!owned_domain_tp_contexts.empty())
        {
            orchestrator->setDomainTPContexts(std::move(owned_domain_tp_contexts));
        }

        // Transfer TurboQuant context ownership to orchestrator
        if (turboquant_ctx)
        {
            orchestrator->setTurboQuantContext(std::move(turboquant_ctx));
        }

        // Transfer KV rotation ownership to orchestrator
        if (kv_rotation)
        {
            orchestrator->setKVRotation(std::move(kv_rotation));
        }

        // Transfer MoE rebalance controller ownership to orchestrator
        if (moe_controller)
        {
            orchestrator->setMoERebalanceController(std::move(moe_controller));
        }

        // Transfer GlobalTPContext ownership to orchestrator
        if (global_tp_ctx)
        {
            orchestrator->setGlobalTPContext(std::move(global_tp_ctx));
        }

        if (config.pp_stage_config.has_value())
        {
            orchestrator->setPPStageConfig(config.pp_stage_config.value());
        }

        // Initialize graph cache
        {
            ScopedWeightLoadDetailTimer timer("graph.build.initialize_graph_cache");
            orchestrator->initializeGraphCache(graph_config.n_layers);
        }

        // Initialize inference state via schema-driven BufferArena path
        InferenceStateInitConfig init_config;
        init_config.use_mapped_memory = config.use_mapped_memory;

        {
            ScopedWeightLoadDetailTimer timer("graph.build.initialize_inference_state");
            if (!orchestrator->initializeInferenceStateFromArena(
                    config.batch_size, config.max_seq_len, device, init_config))
            {
                LOG_ERROR("[InferenceRunner] Failed to initialize inference state (arena path)");
                return nullptr;
            }
        }

        // Load weights and configure orchestrator
        {
            ScopedWeightLoadDetailTimer timer("graph.build.configure_weights");
            if (!configureOrchestratorWeightsImpl(orchestrator.get(), model_ctx, device, config, graph_config))
            {
                LOG_ERROR("[InferenceRunner] Failed to configure orchestrator weights");
                return nullptr;
            }
        }

        // =====================================================================
        // Weight Streaming (Option B) - Create streamer from environment
        // =====================================================================
        // If LLAMINAR_WEIGHT_STREAMING=1, create a LayerWeightStreamer.
        // The streamer manages GPU-side weight caching and on-demand transfers.
        // Note: weight_mgr is already declared above (line 91)
        // =====================================================================
        if (weight_mgr)
        {
            auto weight_streamer = WeightStreamerFactory::createFromEnv(
                weight_mgr, graph_config.n_layers);
            if (weight_streamer)
            {
                orchestrator->setWeightStreamer(std::move(weight_streamer));
            }
        }

        // =====================================================================
        // GPU-Native Collectives (NCCL/RCCL/HOST)
        // =====================================================================
        // Create CollectiveContext for GPU-native collective operations.
        // This eliminates GPU→CPU→GPU transfers during tensor-parallel inference:
        // - NCCL for CUDA devices
        // - RCCL for ROCm devices
        // - MPI fallback for CPU-only or heterogeneous setups
        // =====================================================================
        const auto &env = debugEnv();
        const bool local_tp_collectives_enabled =
            (local_tp_ctx != nullptr && local_tp_ctx->degree() > 1);

        {
            ScopedWeightLoadDetailTimer timer("graph.build.collective_setup");
            if (local_tp_collectives_enabled)
            {
                // Build local cluster inventory (detects CUDA/ROCm GPUs)
                ClusterInventory cluster_inventory = buildLocalClusterInventory(mpi_ctx);

                // Only enable GPU collectives if we have GPUs
                if (cluster_inventory.hasAnyGPU())
                {
                    // Create intra-node context which automatically selects:
                    // - NCCL for all-CUDA groups
                    // - RCCL for all-ROCm groups
                    // - MPI fallback for mixed or CPU-only groups
                    auto collective_ctx = CollectiveContextFactory::createIntraNode(
                        cluster_inventory, mpi_ctx);
                    if (collective_ctx)
                    {
                        orchestrator->setCollectiveContext(std::move(collective_ctx));
                        LOG_INFO("[InferenceRunner] GPU-native collectives enabled (NCCL/RCCL)");
                    }
                    else
                    {
                        LOG_WARN("[InferenceRunner] Failed to create CollectiveContext - using CPU MPI fallback");
                    }
                }
                else
                {
                    LOG_DEBUG("[InferenceRunner] No GPUs detected - using CPU MPI for collectives");
                }
            }
            else if (mpi_ctx && mpi_ctx->world_size() > 1)
            {
                // GLOBAL TP path: use MPI-based collectives from compute stages.
                // Optional experiment path: route collective stages through
                // DeviceGraphExecutor intercept + MPI-backed CollectiveContext.
                if (env.execution.force_mpi_collective_context)
                {
                    // Build CPU-only world inventory to force MPI backend selection
                    // without triggering NCCL/RCCL pre-initialization in BackendRouter.
                    ClusterInventory cluster_inventory;
                    cluster_inventory.world_size = mpi_ctx ? mpi_ctx->world_size() : 1;
                    cluster_inventory.ranks.resize(cluster_inventory.world_size);
                    for (int r = 0; r < cluster_inventory.world_size; ++r)
                    {
                        auto &rank_inv = cluster_inventory.ranks[r];
                        rank_inv.rank = r;
                        rank_inv.node_id = 0;
                        rank_inv.local_rank = r;
                        rank_inv.hostname = "localhost";
                    }
                    cluster_inventory.buildNodeAggregations();

                    auto collective_ctx = CollectiveContextFactory::createIntraNode(cluster_inventory, mpi_ctx);
                    if (collective_ctx)
                    {
                        orchestrator->setCollectiveContext(std::move(collective_ctx));
                        LOG_INFO("[InferenceRunner] GLOBAL TP mode: forcing MPI-backed CollectiveContext via LLAMINAR_FORCE_MPI_COLLECTIVE_CONTEXT=1");
                    }
                    else
                    {
                        LOG_WARN("[InferenceRunner] GLOBAL TP mode: failed to create MPI-backed CollectiveContext, using stage MPI path");
                    }
                }
                else
                {
                    // Default GLOBAL TP behavior: use MPI-based collectives from compute stages.
                    LOG_DEBUG("[InferenceRunner] GLOBAL TP mode: using stage MPI collectives (CollectiveContext disabled)");
                }
            }
        }

        LOG_DEBUG("[InferenceRunner] DeviceGraphOrchestrator created successfully");

        if (device.is_cpu() && architecture == "qwen35moe" && graph_config.moe.num_experts > 0)
        {
            ScopedWeightLoadDetailTimer timer("graph.build.eager_moe_expert_materialization");
            if (!orchestrator->materializeForwardGraphForShape(/*seq_len=*/1, config.batch_size))
            {
                LOG_ERROR("[InferenceRunner] Failed eager Qwen35 MoE expert graph materialization on "
                          << device.to_string());
                return nullptr;
            }
        }

        // DeviceGraphOrchestrator implements IInferenceRunner directly
        return orchestrator;
    }

    static bool configureOrchestratorWeightsImpl(
        DeviceGraphOrchestrator *orchestrator,
        std::shared_ptr<ModelContext> model_ctx,
        DeviceId device,
        const InferenceRunnerConfig &config,
        const GraphConfig &graph_config)
    {
        if (!orchestrator || !model_ctx)
        {
            return false;
        }

        auto weight_mgr = model_ctx->concreteWeightManager();
        if (!weight_mgr)
        {
            LOG_ERROR("[InferenceRunner] No weight manager in model context");
            return false;
        }

        // =====================================================================
        // Set WeightManager and PlacementMap for phase-aware weight access
        // (Gap 3: CPU Decode Participation)
        // =====================================================================
        orchestrator->setWeightManager(weight_mgr);
        if (auto placement_map = model_ctx->placementMap())
        {
            orchestrator->setWeightPlacementMap(placement_map);
            LOG_DEBUG("[InferenceRunner] Phase-aware weight access configured with placement map");
        }

        // =====================================================================
        // Eager load ALL layer weights into cache BEFORE preloading
        // =====================================================================
        // Use schema factory to determine which weights are required vs optional.
        // This ensures consistent handling: required weights fail if missing,
        // optional weights (like QKV biases) silently skip.
        const std::string arch = model_ctx->architecture();
        auto schema_factory = SchemaFactoryRegistry::getFactory(arch);

        int n_layers = model_ctx->blockCount();
        LOG_DEBUG("[InferenceRunner] Eagerly loading " << n_layers << " layers of weights...");
        WeightLoadingProfiler::begin(WeightLoadPhase::TENSOR_LOAD);
        ScopedWeightLoadDetailTimer eager_layer_timer("weights.eager_layer_cache_load");

        // Validate all layer weights against schema before loading.
        // Missing required weights are fatal; missing optional weights are skipped.
        auto validation = validateLayerWeights(
            *schema_factory, n_layers,
            [&](const std::string &name)
            { return model_ctx->hasTensor(name); });

        if (!validation.success)
        {
            LOG_ERROR("[InferenceRunner] " << validation.error_message());
            WeightLoadingProfiler::end(WeightLoadPhase::TENSOR_LOAD);
            return false;
        }
        if (!validation.missing_optional.empty())
        {
            LOG_DEBUG("[InferenceRunner] Skipping " << validation.missing_optional.size()
                                                    << " optional weights not present in model");
        }

        // Use validated weight list (only weights that exist in the model)
        auto &weights_to_load = validation.weights_to_load;

        const unsigned hw_threads = std::max(1u, std::thread::hardware_concurrency());
        const unsigned target_workers = std::min<unsigned>(8u, hw_threads);
        const unsigned worker_count = std::min<unsigned>(
            target_workers,
            std::max<unsigned>(1u, static_cast<unsigned>(weights_to_load.size())));

        std::atomic<size_t> next_index{0};
        std::atomic<bool> failed{false};
        std::string first_error;
        std::mutex error_mutex;

        auto load_worker = [&]()
        {
            while (true)
            {
                if (failed.load(std::memory_order_relaxed))
                {
                    return;
                }

                const size_t idx = next_index.fetch_add(1, std::memory_order_relaxed);
                if (idx >= weights_to_load.size())
                {
                    return;
                }

                const auto &[weight_name, is_optional] = weights_to_load[idx];
                auto weight = weight_mgr->getWeightForDevice(weight_name);

                if (!weight)
                {
                    if (is_optional)
                    {
                        // Optional weight missing - this is fine (e.g., model without QKV biases)
                        LOG_TRACE("[InferenceRunner] Optional weight not present: " << weight_name);
                    }
                    else
                    {
                        std::lock_guard<std::mutex> lock(error_mutex);
                        if (!failed.exchange(true, std::memory_order_relaxed))
                        {
                            first_error = weight_name;
                        }
                        return;
                    }
                }
            }
        };

        if (worker_count == 1)
        {
            load_worker();
        }
        else
        {
            LOG_DEBUG("[InferenceRunner] Parallel eager load workers=" << worker_count
                                                                       << " weights=" << weights_to_load.size());
            std::vector<std::future<void>> load_tasks;
            load_tasks.reserve(worker_count);
            for (unsigned i = 0; i < worker_count; ++i)
            {
                load_tasks.emplace_back(std::async(std::launch::async, load_worker));
            }
            for (auto &task : load_tasks)
            {
                task.get();
            }
        }

        if (failed.load(std::memory_order_relaxed))
        {
            LOG_ERROR("[InferenceRunner] Failed to load required weight: " << first_error);
            WeightLoadingProfiler::end(WeightLoadPhase::TENSOR_LOAD);
            return false;
        }

        LOG_DEBUG("[InferenceRunner] All layer weights loaded into cache");
        WeightLoadingProfiler::end(WeightLoadPhase::TENSOR_LOAD);

        // =====================================================================
        // Load global weights BEFORE finalization so they are available for
        // the GPU pipeline (GEMM packing, non-GEMM upload, embedding prep).
        // validateLayerWeights only loads blk.N.* per-layer weights; global
        // weights (output.weight, token_embd.weight, output_norm.weight)
        // must be loaded separately.
        //
        // For tied-embeddings models (Qwen3, Qwen3.5): output.weight is
        // absent and token_embd.weight is reused as the LM head. The GPU
        // GEMM pipeline (packGemmWeightsViaPipeline) has tied-embeddings
        // logic that includes token_embd.weight in the GEMM pipeline when
        // output.weight is missing — but it can only find the tensor if
        // it's already in cache_. Loading it here ensures that.
        //
        // output_norm.weight is also loaded so uploadNonGemmWeights() can
        // upload it to GPU during finalization rather than needing a lazy
        // ensureOnDevice() call later.
        // =====================================================================
        {
            // Load global weights into cache_ using the DEFAULT device (CPU)
            // so they sit alongside per-layer weights. Do NOT pass `device`
            // here — passing CUDA:0 would put them in per_device_cache_,
            // which makes packGemmWeightsViaPipeline() think per_device_cache_
            // is the primary source and skip all per-layer weights from cache_.
            auto lm_head = weight_mgr->getWeightForDevice("output.weight");
            if (!lm_head && model_ctx->hasTensor("token_embd.weight"))
            {
                // Tied embeddings: output.weight absent, token_embd.weight
                // will be reused as the LM head GEMM weight. Load it now so
                // packGemmWeightsViaPipeline() can find it in cache_ and
                // include it in the GPU GEMM pipeline for repacking.
                weight_mgr->getWeightForDevice("token_embd.weight");
                LOG_DEBUG("[InferenceRunner] Tied embeddings: token_embd.weight loaded for GPU GEMM pipeline");
            }
            else if (lm_head)
            {
                LOG_DEBUG("[InferenceRunner] output.weight loaded into cache for GPU pipeline enrollment");
                // Also load token_embd.weight for embedding stage
                if (model_ctx->hasTensor("token_embd.weight"))
                    weight_mgr->getWeightForDevice("token_embd.weight");
            }

            // Load output_norm for GPU upload during finalization
            if (model_ctx->hasTensor("output_norm.weight"))
                weight_mgr->getWeightForDevice("output_norm.weight");

            LOG_DEBUG("[InferenceRunner] Global weights loaded into cache");
        }

        configureWeightManagerForGraph(
            *weight_mgr,
            arch,
            graph_config,
            graph_config.tp_config != nullptr);

        auto weight_plan = buildSingleDeviceWeightPlan(
            *weight_mgr,
            *model_ctx,
            validation,
            device,
            graph_config.tp_config.get(),
            nullptr,
            graph_config.tp_config ? graph_config.local_rank : -1);
        if (!installPreparedWeightStoreForPlan(*weight_mgr, config, weight_plan, "[InferenceRunner]"))
            return false;
        LOG_DEBUG("[InferenceRunner] SingleDevice WeightPlan built with "
                  << weight_plan.size() << " requirements");

        // =====================================================================
        // Weight rotation (activation_rotation) is intentionally NOT registered
        // as a weight preprocessor. GEMM invariance (R(X) @ R(W)^T = X @ W^T)
        // means weight rotation produces identical outputs in exact arithmetic,
        // but the extra quantization step (Q4_0 → FP32 → rotate → INT8) adds
        // noise that compounds with Q16_1 KV cache quantization, degrading
        // parity. KV rotation operates independently on K/V/Q activations.
        // =====================================================================
        const auto &env = debugEnv();

        FrozenModelWeightSet frozen_weights = weight_mgr->materialize(weight_plan);
        auto weight_bindings = makeModelWeightBindings(frozen_weights);
        auto legacy_weights = toLegacyModelWeights(weight_bindings);

        if (!legacy_weights.embedding_table || !legacy_weights.final_norm || !legacy_weights.lm_head)
        {
            LOG_ERROR("[InferenceRunner] Missing global weights");
            return false;
        }

        const bool has_tiered_overlay_plan =
            graph_config.moe.expert_parallel_plan &&
            graph_config.moe.expert_parallel_plan->isTieredOverlay();
        auto overlay_runtime_plan_for_weight_prep = graph_config.moe.expert_overlay_runtime_plan;
        const MoEExpertOverlayExecutionPlan *overlay_execution_plan_for_weight_prep =
            graph_config.moe.expert_overlay_execution_plan.get();
        if (has_tiered_overlay_plan && !overlay_runtime_plan_for_weight_prep)
        {
            overlay_runtime_plan_for_weight_prep = resolveMoEExpertOverlayRuntimePlan(
                graph_config.moe.expert_parallel_plan,
                MoEExpertOverlayRuntimeResolverOptions{
                    .current_world_rank = overlayRankFor(graph_config.moe.overlay_mpi_ctx, nullptr),
                });
            overlay_execution_plan_for_weight_prep = nullptr;
        }

        if (!weight_mgr->prepareWeightsForDevice(
                frozen_weights,
                device,
                /*include_expert_jobs=*/!overlay_runtime_plan_for_weight_prep))
        {
            LOG_WARN("[InferenceRunner] Binding-driven weight preparation had issues for device "
                     << device.to_string());
        }

        if (overlay_runtime_plan_for_weight_prep)
        {
            if (!weight_mgr->prepareMoEExpertOverlayWeights(
                    *overlay_runtime_plan_for_weight_prep,
                    &frozen_weights,
                    overlay_execution_plan_for_weight_prep))
            {
                LOG_ERROR("[InferenceRunner] Failed to prepare MoE expert overlay weights");
                return false;
            }
        }

        orchestrator->setFrozenWeightSet(
            std::make_unique<FrozenModelWeightSet>(std::move(frozen_weights)));
        LOG_DEBUG("[InferenceRunner] Frozen weight bindings configured on orchestrator");

        // Phase 4-5: Populate prepared weight store from frozen bindings
        orchestrator->initializePreparedWeightStore(device);

        // Phase 9: Mark graph materialization complete — all weight bindings resolved
        weight_mgr->markGraphMaterializationComplete();

        return true;
    }

    // =========================================================================
    // Pipeline Parallelism Weight Configuration
    // =========================================================================

    /**
     * @brief Configure orchestrator weights for a Pipeline Parallelism stage
     *
     * Similar to configureOrchestratorWeightsImpl but only loads weights for the
     * layers and components owned by this PP stage.
     *
     * @param orchestrator The DeviceGraphOrchestrator to configure
     * @param model_ctx Model context with weights
     * @param device Target device for weight packing/upload
     * @param pp_config PP stage configuration specifying which layers/components to load
     * @return true on success, false on failure
     */
    static bool configurePPStageWeightsImpl(
        DeviceGraphOrchestrator *orchestrator,
        std::shared_ptr<ModelContext> model_ctx,
        DeviceId device,
        const FactoryPPStageConfig &pp_config,
        const InferenceRunnerConfig &config)
    {
        if (!orchestrator || !model_ctx)
        {
            return false;
        }

        // =====================================================================
        // Use the shared ModelContext's WeightManager (single WM per model).
        // PP layer-range filtering is handled by prepareWeightsForDevice().
        // =====================================================================
        auto weight_mgr = model_ctx->concreteWeightManager();
        if (!weight_mgr)
        {
            LOG_ERROR("[PPStageRunner] No weight manager in model context");
            return false;
        }

        // Apply architecture-specific weight sharding config.
        // setWeightShardingConfig is idempotent and does NOT clear the weight cache
        // (unlike configure() which clears cache_ unconditionally).
        const std::string arch = model_ctx->architecture();
        weight_mgr->setWeightShardingConfig(
            SchemaFactoryRegistry::getWeightShardingConfig(arch));

        // =====================================================================
        // Set WeightManager and PlacementMap for phase-aware weight access
        // =====================================================================
        orchestrator->setWeightManager(weight_mgr);
        if (auto placement_map = model_ctx->placementMap())
        {
            orchestrator->setWeightPlacementMap(placement_map);
            LOG_DEBUG("[PPStageRunner] Phase-aware weight access configured with placement map");
        }

        // =====================================================================
        // Eagerly load ONLY this stage's layer weights into cache
        // =====================================================================
        auto schema_factory = SchemaFactoryRegistry::getFactory(arch);

        const int first_layer = pp_config.first_layer;
        const int last_layer = pp_config.last_layer;
        LOG_DEBUG("[PPStageRunner] Eagerly loading layers [" << first_layer << ", " << last_layer
                                                             << ") of weights...");
        ScopedWeightLoadDetailTimer pp_eager_layer_timer("weights.pp.eager_layer_cache_load");

        // Validate layer weights against schema before loading.
        auto pp_validation = validateLayerWeights(
            *schema_factory, model_ctx->totalBlockCount(),
            [&](const std::string &name)
            { return model_ctx->hasTensor(name); },
            first_layer, last_layer);

        if (!pp_validation.success)
        {
            LOG_ERROR("[PPStageRunner] " << pp_validation.error_message());
            return false;
        }

        // Load global weights this stage owns.
        // IMPORTANT: Pass the target device so first_device_ is set to the GPU
        // device (not cpu). This ensures packGemmWeightsViaPipeline() and
        // buildWeights() use the same tensor pointers, avoiding GPU GEMM cache misses.
        if (pp_config.has_embedding)
        {
            auto embedding = weight_mgr->getWeightForDevice("token_embd.weight", device);
            if (!embedding)
            {
                LOG_ERROR("[PPStageRunner] Stage has_embedding=true but token_embd.weight missing");
                return false;
            }
            LOG_DEBUG("[PPStageRunner] Loaded embedding table for stage");
        }

        if (pp_config.has_lm_head)
        {
            auto final_norm = weight_mgr->getWeightForDevice("output_norm.weight", device);
            auto lm_head = weight_mgr->getWeightForDevice("output.weight", device);
            if (!lm_head)
            {
                auto embedding_fallback = weight_mgr->getWeightForDevice("token_embd.weight", device);
                if (embedding_fallback)
                {
                    LOG_INFO("[PPStageRunner] output.weight not found, using tied embeddings");
                }
            }
            if (!final_norm)
            {
                LOG_ERROR("[PPStageRunner] Stage has_lm_head=true but output_norm weight missing");
                return false;
            }
            LOG_DEBUG("[PPStageRunner] Loaded final_norm and lm_head for stage");
        }

        // Load layer weights
        for (const auto &[weight_name, is_optional] : pp_validation.weights_to_load)
        {
            auto weight = weight_mgr->getWeightForDevice(weight_name, device);

            if (!weight)
            {
                if (is_optional)
                {
                    LOG_TRACE("[PPStageRunner] Optional weight not present: " << weight_name);
                }
                else
                {
                    LOG_ERROR("[PPStageRunner] Failed to load required weight: " << weight_name);
                    return false;
                }
            }
        }
        LOG_DEBUG("[PPStageRunner] All layer weights for stage loaded into cache");

        auto weight_plan = buildSingleDeviceWeightPlan(
            *weight_mgr,
            *model_ctx,
            pp_validation,
            device,
            nullptr,
            &pp_config);
        if (!installPreparedWeightStoreForPlan(*weight_mgr, config, weight_plan, "[PPStageRunner]"))
            return false;
        FrozenModelWeightSet frozen_weights = weight_mgr->materialize(weight_plan);
        auto weight_bindings = makeModelWeightBindings(frozen_weights);
        auto legacy_weights = toLegacyModelWeights(weight_bindings);

        if (pp_config.has_embedding && !legacy_weights.embedding_table)
        {
            LOG_ERROR("[PPStageRunner] Frozen PP stage has_embedding=true but embedding missing");
            return false;
        }
        if (pp_config.has_lm_head && (!legacy_weights.final_norm || !legacy_weights.lm_head))
        {
            LOG_ERROR("[PPStageRunner] Frozen PP stage has_lm_head=true but final_norm/lm_head missing");
            return false;
        }

        // =====================================================================
        // Prepare weights: GEMM pack + upload (layer-filtered, no host release)
        // Host copies are released by the caller after ALL PP stages are prepared.
        // =====================================================================
        bool prepare_ok = weight_mgr->prepareWeightsForDevice(frozen_weights, device);

        if (!prepare_ok)
        {
            LOG_WARN("[PPStageRunner] Weight preparation had issues for device "
                     << device.to_string() << " layers [" << first_layer << ", " << last_layer << ")");
        }

        orchestrator->setFrozenWeightSet(
            std::make_unique<FrozenModelWeightSet>(std::move(frozen_weights)));
        LOG_DEBUG("[PPStageRunner] Weights configured for PP stage [" << first_layer << ", " << last_layer << ")");

        // PP runners build the same frozen graph bindings as full runners, so
        // initialize the model-owned prepared store before graph construction.
        orchestrator->initializePreparedWeightStore(device);
        return true;
    }

    // =========================================================================
    // Unified Pipeline Runner Factory
    // =========================================================================

    /**
     * @brief Configure weights for a unified LOCAL PP pipeline
     *
     * Sets up ModelWeights with device-aware weight loading based on
     * the PipelineConfig. Each layer's weights are loaded for its assigned
     * device (from getDeviceForLayer()).
     *
     * @param orchestrator Orchestrator to configure
     * @param model_ctx Model context with weights
     * @param pipeline_config Pipeline configuration with layer→device mapping
     * @return true on success
     */
    static bool configureUnifiedPipelineWeightsImpl(
        DeviceGraphOrchestrator *orchestrator,
        std::shared_ptr<ModelContext> model_ctx,
        std::shared_ptr<PipelineConfig> pipeline_config)
    {
        if (!orchestrator || !model_ctx || !pipeline_config)
        {
            LOG_ERROR("[UnifiedPipeline] Invalid arguments");
            return false;
        }

        // Get primary device for embedding/lm_head (from first/last stage)
        DeviceId embedding_device = pipeline_config->getDeviceForLayer(0);
        DeviceId lm_head_device = pipeline_config->getDeviceForLayer(pipeline_config->total_layers - 1);

        LOG_DEBUG("[UnifiedPipeline] Embedding device: " << embedding_device.to_string()
                                                         << ", LM head device: " << lm_head_device.to_string());

        // =====================================================================
        // Global weights (embedding, final_norm, lm_head)
        // =====================================================================
        ModelWeights weights;

        auto embedding = model_ctx->getWeightForDevice("token_embd.weight", embedding_device);
        auto final_norm = model_ctx->getWeightForDevice("output_norm.weight", lm_head_device);
        auto lm_head = model_ctx->getWeightForDevice("output.weight", lm_head_device);

        // Tied embeddings: if output.weight is missing, reuse token_embd.weight
        if (!lm_head && embedding)
        {
            LOG_INFO("[UnifiedPipeline] output.weight not found, using tied embeddings (token_embd.weight)");
            lm_head = embedding;
        }

        if (!embedding || !final_norm || !lm_head)
        {
            LOG_ERROR("[UnifiedPipeline] Missing global weights");
            return false;
        }

        weights.embedding_table = embedding.get();
        weights.final_norm = final_norm.get();
        weights.lm_head = lm_head.get();

        // =====================================================================
        // Layer weight accessor - uses PipelineConfig to determine device
        // =====================================================================
        auto model_ctx_ptr = model_ctx;
        auto pipeline_config_ptr = pipeline_config;

        weights.get_layer_weights = [model_ctx_ptr, pipeline_config_ptr](int layer_idx) -> LayerWeights
        {
            // Get device for this layer from pipeline config
            DeviceId layer_device = pipeline_config_ptr->getDeviceForLayer(layer_idx);

            LayerWeights layer;
            std::string prefix = "blk." + std::to_string(layer_idx) + ".";

            // Attention weights - get for layer's specific device
            layer.wq = model_ctx_ptr->getWeightForDevice(prefix + "attn_q.weight", layer_device).get();
            layer.wk = model_ctx_ptr->getWeightForDevice(prefix + "attn_k.weight", layer_device).get();
            layer.wv = model_ctx_ptr->getWeightForDevice(prefix + "attn_v.weight", layer_device).get();
            layer.wo = model_ctx_ptr->getWeightForDevice(prefix + "attn_output.weight", layer_device).get();
            layer.attn_norm = model_ctx_ptr->getWeightForDevice(prefix + "attn_norm.weight", layer_device).get();

            // Attention biases (may be null for Qwen2)
            auto q_bias = model_ctx_ptr->getWeightForDevice(prefix + "attn_q.bias", layer_device);
            auto k_bias = model_ctx_ptr->getWeightForDevice(prefix + "attn_k.bias", layer_device);
            auto v_bias = model_ctx_ptr->getWeightForDevice(prefix + "attn_v.bias", layer_device);
            layer.q_bias = q_bias ? q_bias.get() : nullptr;
            layer.k_bias = k_bias ? k_bias.get() : nullptr;
            layer.v_bias = v_bias ? v_bias.get() : nullptr;

            // QK norm weights (Qwen3: per-head RMSNorm, may be null for Qwen2)
            auto q_norm = model_ctx_ptr->getWeightForDevice(prefix + "attn_q_norm.weight", layer_device);
            auto k_norm = model_ctx_ptr->getWeightForDevice(prefix + "attn_k_norm.weight", layer_device);
            layer.q_norm = q_norm ? q_norm.get() : nullptr;
            layer.k_norm = k_norm ? k_norm.get() : nullptr;

            // FFN weights
            layer.gate_proj = model_ctx_ptr->getWeightForDevice(prefix + "ffn_gate.weight", layer_device).get();
            layer.up_proj = model_ctx_ptr->getWeightForDevice(prefix + "ffn_up.weight", layer_device).get();
            layer.down_proj = model_ctx_ptr->getWeightForDevice(prefix + "ffn_down.weight", layer_device).get();
            layer.ffn_norm = model_ctx_ptr->getWeightForDevice(prefix + "ffn_norm.weight", layer_device).get();

            return layer;
        };

        orchestrator->setWeights(weights);
        LOG_DEBUG("[UnifiedPipeline] Weights configured for " << pipeline_config->total_layers << " layers");
        return true;
    }

    std::unique_ptr<IInferenceRunner> createUnifiedPipelineRunner(
        std::shared_ptr<ModelContext> model_ctx,
        std::shared_ptr<PipelineConfig> pipeline_config,
        const InferenceRunnerConfig &config)
    {
        LOG_DEBUG("[UnifiedPipeline] createUnifiedPipelineRunner called");

        // =====================================================================
        // Validate inputs
        // =====================================================================
        if (!model_ctx)
        {
            LOG_ERROR("[UnifiedPipeline] model_ctx is null");
            return nullptr;
        }

        if (!pipeline_config)
        {
            LOG_ERROR("[UnifiedPipeline] pipeline_config is null");
            return nullptr;
        }

        std::string validation_error;
        if (!pipeline_config->validate(&validation_error))
        {
            LOG_ERROR("[UnifiedPipeline] Invalid PipelineConfig: " << validation_error);
            return nullptr;
        }

        // =====================================================================
        // Validate architecture
        // =====================================================================
        std::string architecture = model_ctx->architecture();
        if (!SchemaFactoryRegistry::isSupported(architecture))
        {
            LOG_ERROR("[UnifiedPipeline] Unsupported architecture: " << architecture);
            return nullptr;
        }

        // =====================================================================
        // Build GraphConfig via polymorphic builder
        // =====================================================================
        auto config_builder = createGraphConfigBuilder(architecture);
        GraphConfig graph_config;
        config_builder->populateFromModelContext(*model_ctx, graph_config);

        // Execution-specific settings
        graph_config.max_seq_len = config.max_seq_len;
        graph_config.activation_precision = config.activation_precision;

        // Non-TP: use full dimensions
        setFullDimensions(graph_config);

        // Primary device is from first PP stage
        DeviceId primary_device = pipeline_config->getDeviceForLayer(0);
        graph_config.default_device = primary_device;

        LOG_DEBUG("[UnifiedPipeline] GraphConfig: "
                  << "n_layers=" << graph_config.n_layers
                  << ", d_model=" << graph_config.d_model
                  << ", primary_device=" << primary_device.to_string());

        // =====================================================================
        // Create DeviceGraphOrchestrator with injected dependencies
        // =====================================================================
        DeviceGraphOrchestrator::Dependencies deps;
        deps.model_ctx = model_ctx;
        deps.graph_builder = GraphBuilderRegistry::create(architecture, graph_config, nullptr);
        deps.graph_builder->setModelContext(model_ctx);
        deps.pipeline_config = pipeline_config;

        auto orchestrator = std::make_unique<DeviceGraphOrchestrator>(
            std::move(deps));

        // =====================================================================
        // Initialize inference state
        // =====================================================================
        InferenceStateInitConfig init_config;
        init_config.use_mapped_memory = config.use_mapped_memory;

        if (!orchestrator->initializeInferenceStateFromArena(
                config.batch_size, config.max_seq_len, primary_device, init_config))
        {
            LOG_ERROR("[UnifiedPipeline] Failed to initialize inference state (arena path)");
            return nullptr;
        }

        // =====================================================================
        // Configure weights (device-aware based on pipeline config)
        // =====================================================================
        if (!configureUnifiedPipelineWeightsImpl(orchestrator.get(), model_ctx, pipeline_config))
        {
            LOG_ERROR("[UnifiedPipeline] Failed to configure weights");
            return nullptr;
        }

        LOG_INFO("[UnifiedPipeline] Created runner with "
                 << pipeline_config->numStages() << " PP stages, "
                 << pipeline_config->total_layers << " layers");

        return orchestrator;
    }

    // =========================================================================
    // Pipeline Parallelism Stage Runner Factory
    // =========================================================================

    std::unique_ptr<IInferenceRunner> createPPStageRunner(
        std::shared_ptr<ModelContext> model_ctx,
        DeviceId device,
        const FactoryPPStageConfig &pp_config,
        const InferenceRunnerConfig &config)
    {
        LOG_DEBUG("[PPStageRunner] createPPStageRunner called: device=" << device.to_string()
                                                                        << " layers=[" << pp_config.first_layer << ", " << pp_config.last_layer << ")"
                                                                        << " has_embedding=" << pp_config.has_embedding
                                                                        << " has_lm_head=" << pp_config.has_lm_head);

        // =====================================================================
        // Validate inputs
        // =====================================================================
        if (!model_ctx)
        {
            LOG_ERROR("[PPStageRunner] model_ctx is null");
            return nullptr;
        }

        if (!device.is_valid())
        {
            LOG_ERROR("[PPStageRunner] Invalid device " << device << ". Use DeviceId::cpu() for CPU.");
            return nullptr;
        }

        if (!pp_config.isValid())
        {
            LOG_ERROR("[PPStageRunner] Invalid FactoryPPStageConfig: first_layer=" << pp_config.first_layer
                                                                                   << " last_layer=" << pp_config.last_layer);
            return nullptr;
        }

        // =====================================================================
        // Validate architecture
        // =====================================================================
        std::string architecture = model_ctx->architecture();
        if (!SchemaFactoryRegistry::isSupported(architecture))
        {
            LOG_ERROR("[PPStageRunner] Unsupported architecture: " << architecture);
            return nullptr;
        }

        // Weight sharding configuration is applied once via the shared WeightManager.

        // =====================================================================
        // Build GraphConfig via polymorphic builder
        // =====================================================================
        auto config_builder = createGraphConfigBuilder(architecture);
        GraphConfig graph_config;
        config_builder->populateFromModelContext(*model_ctx, graph_config);

        // Override n_layers for PP stage: graph builds only this stage's layers,
        // not the full model. total_n_layers retains the full model count for
        // GDN/FA pattern detection etc.
        graph_config.n_layers = pp_config.layerCount();

        // Execution-specific settings
        graph_config.max_seq_len = config.max_seq_len;
        graph_config.default_device = device;
        graph_config.activation_precision = config.activation_precision;

        graph_config.fused_attention_backend = resolveEffectiveAttentionBackend(
            config.activation_precision, config.fused_attention_backend);

        // kv_cache_scale_k/v set by config builder — don't overwrite
        graph_config.kv_cache_precision = config.kv_cache_precision;

        // TurboQuant context for TQ4/TQ KV cache
        std::shared_ptr<TurboQuantContext> turboquant_ctx;
        if (config.kv_cache_precision == KVCachePrecision::TQ4 ||
            config.kv_cache_precision == KVCachePrecision::TQ)
        {
            turboquant_ctx = std::make_shared<TurboQuantContext>(graph_config.head_dim);
            graph_config.turboquant_ctx = turboquant_ctx.get();
        }

        // KV rotation for Q16_1 kurtosis reduction
        std::shared_ptr<ActivationRotation> kv_rotation;
        if (config.kv_cache_precision == KVCachePrecision::Q16_1 && debugEnv().kv_rotation)
        {
            kv_rotation = std::make_shared<ActivationRotation>(
                graph_config.head_dim, graph_config.head_dim, /*seed=*/42);
            graph_config.kv_rotation = kv_rotation.get();
        }

        // PP layer offset for KV cache indexing:
        // When building graphs for PP stage [first_layer, last_layer), this offset
        // is subtracted from global layer index to get local KV cache index.
        graph_config.pp_layer_offset = pp_config.first_layer;

        LOG_DEBUG("[PPStageRunner] GraphConfig: n_layers=" << graph_config.n_layers
                                                           << " (PP stage owns layers ["
                                                           << pp_config.first_layer << ", " << pp_config.last_layer << "))"
                                                           << " pp_layer_offset=" << graph_config.pp_layer_offset);

        // PP stages don't use tensor parallelism (TP) - they use full dimensions
        // Inter-stage communication is handled by the PP orchestrator, not MPI collectives
        setFullDimensions(graph_config);
        if (!configureStaticMoEExpertRange(graph_config))
            return nullptr;

        LOG_DEBUG("[PPStageRunner] GraphConfig (no TP): "
                  << "vocab=" << graph_config.vocab_size
                  << ", d_model=" << graph_config.d_model
                  << ", n_layers=" << graph_config.n_layers
                  << ", n_heads=" << graph_config.n_heads
                  << ", n_kv_heads=" << graph_config.n_kv_heads
                  << ", d_ff=" << graph_config.d_ff
                  << ", rope_theta=" << graph_config.rope_theta
                  << ", rms_norm_eps=" << graph_config.rms_norm_eps
                  << ", activation_precision=" << static_cast<int>(graph_config.activation_precision));

        // =====================================================================
        // Create DeviceGraphOrchestrator
        // Note: No MPI context for PP stages - inter-stage comm handled externally
        // =====================================================================
        auto graph_builder = GraphBuilderRegistry::create(architecture, graph_config, nullptr);
        graph_builder->setModelContext(model_ctx);
        auto orchestrator = std::make_unique<DeviceGraphOrchestrator>(
            std::move(graph_builder), nullptr /* no mpi_ctx */);

        if (turboquant_ctx)
            orchestrator->setTurboQuantContext(std::move(turboquant_ctx));
        if (kv_rotation)
            orchestrator->setKVRotation(std::move(kv_rotation));

        // =====================================================================
        // Set PP stage configuration - CRITICAL for correct graph building
        // This tells executeForward() to use buildPartialForwardGraph() instead
        // of buildFullForwardGraph()
        // =====================================================================
        orchestrator->setPPStageConfig(pp_config);

        // =====================================================================
        // Initialize graph cache for ONLY this stage's layers
        // =====================================================================
        const int stage_layer_count = pp_config.layerCount();
        orchestrator->initializeGraphCache(stage_layer_count);
        LOG_DEBUG("[PPStageRunner] Graph cache initialized for " << stage_layer_count << " layers");

        // =====================================================================
        // Initialize inference state (allocates buffers)
        // =====================================================================
        InferenceStateInitConfig init_config;
        init_config.use_mapped_memory = config.use_mapped_memory;

        if (!orchestrator->initializeInferenceStateFromArena(
                config.batch_size, config.max_seq_len, device, init_config))
        {
            LOG_ERROR("[PPStageRunner] Failed to initialize inference state (arena path)");
            return nullptr;
        }

        // =====================================================================
        // Load weights for this PP stage (partial weight loading)
        // =====================================================================
        if (!configurePPStageWeightsImpl(orchestrator.get(), model_ctx, device, pp_config, config))
        {
            LOG_ERROR("[PPStageRunner] Failed to configure PP stage weights");
            return nullptr;
        }

        // =====================================================================
        // Retain ModelContext to keep the shared ModelLoader and WeightManager
        // alive for the lifetime of this PP stage runner.
        // =====================================================================
        orchestrator->retainModelContext(model_ctx);

        // =====================================================================
        // Note: No GPU collective setup for PP stages
        // PP handles inter-stage communication externally (not via MPI collectives)
        // =====================================================================

        LOG_INFO("[PPStageRunner] PP stage runner created successfully: "
                 << "layers=[" << pp_config.first_layer << ", " << pp_config.last_layer << ") "
                 << "has_embedding=" << pp_config.has_embedding
                 << " has_lm_head=" << pp_config.has_lm_head
                 << " device=" << device.to_string());

        return orchestrator;
    }

    // =========================================================================
    // Testable Factory Function (Interface-Based)
    // =========================================================================

    std::unique_ptr<IInferenceRunner> createTestableInferenceRunner(
        std::shared_ptr<IModelContext> model_ctx,
        DeviceId device,
        const InferenceRunnerConfig &config)
    {
        LOG_DEBUG("[InferenceRunner] createTestableInferenceRunner called");

        if (!model_ctx)
        {
            LOG_ERROR("[InferenceRunner] model_ctx is null");
            return nullptr;
        }

        // Validate device
        if (!device.is_valid())
        {
            LOG_ERROR("[InferenceRunner] Invalid device " << device
                                                          << ". Use DeviceId::cpu() for CPU.");
            return nullptr;
        }
        LOG_DEBUG("[InferenceRunner] Using device " << device);

        std::string architecture = model_ctx->architecture();
        if (!SchemaFactoryRegistry::isSupported(architecture))
        {
            LOG_ERROR("[InferenceRunner] Unsupported architecture: " << architecture);
            return nullptr;
        }

        // Build GraphConfig via polymorphic builder
        auto config_builder = createGraphConfigBuilder(architecture);
        GraphConfig graph_config;
        config_builder->populateFromModelContext(*model_ctx, graph_config);
        graph_config.moe.expert_mode = config.moe_expert_mode;
        graph_config.moe.hot_expert_cache = config.moe_hot_expert_cache;
        graph_config.moe.rebalance_config = effectiveMoERebalanceConfig(config);
        DomainTPContextMap owned_domain_tp_contexts;
        const auto overlay_runner_mpi_ctx = config.moe_expert_overlay_mpi_ctx;
        if (!applyMoEExpertOverlayConfigToGraph(
                *model_ctx,
                config,
                overlay_runner_mpi_ctx,
                graph_config,
                owned_domain_tp_contexts,
                "[InferenceRunner]"))
        {
            return nullptr;
        }
        // Execution-specific settings
        graph_config.max_seq_len = config.max_seq_len;
        graph_config.default_device = device;
        graph_config.activation_precision = config.activation_precision;
        graph_config.fused_attention_backend = config.fused_attention_backend;
        // kv_cache_scale_k/v set by config builder — don't overwrite
        graph_config.kv_cache_precision = config.kv_cache_precision;

        // TurboQuant context for TQ4/TQ KV cache
        std::shared_ptr<TurboQuantContext> turboquant_ctx;
        if (config.kv_cache_precision == KVCachePrecision::TQ4 ||
            config.kv_cache_precision == KVCachePrecision::TQ)
        {
            turboquant_ctx = std::make_shared<TurboQuantContext>(graph_config.head_dim);
            graph_config.turboquant_ctx = turboquant_ctx.get();
        }

        // KV rotation for Q16_1 kurtosis reduction
        std::shared_ptr<ActivationRotation> kv_rotation;
        if (config.kv_cache_precision == KVCachePrecision::Q16_1 && debugEnv().kv_rotation)
        {
            kv_rotation = std::make_shared<ActivationRotation>(
                graph_config.head_dim, graph_config.head_dim, /*seed=*/42);
            graph_config.kv_rotation = kv_rotation.get();
        }

        // PP layer range for nested TP-in-PP (partial graph with layer offset)
        if (config.pp_stage_config.has_value())
        {
            graph_config.n_layers = config.pp_stage_config->layerCount();
            graph_config.pp_layer_offset = config.pp_stage_config->first_layer;
        }

        // Check for TP configuration (LOCAL or GLOBAL)
        ITPContext *tp_ctx = config.tp_ctx;
        ILocalTPContext *local_tp_ctx = tp_ctx && tp_ctx->isLocal()
                                            ? static_cast<ILocalTPContext *>(tp_ctx)
                                            : nullptr;
        IGlobalTPContext *injected_global_tp_ctx = (tp_ctx && !tp_ctx->isLocal())
                                                       ? dynamic_cast<IGlobalTPContext *>(tp_ctx)
                                                       : nullptr;
        const int tp_device_idx = config.tp_device_index;

        if (local_tp_ctx && local_tp_ctx->degree() > 1)
        {
            if (!applyLocalTPAssignment(graph_config, local_tp_ctx, tp_device_idx))
            {
                return nullptr;
            }
        }
        else if (injected_global_tp_ctx && injected_global_tp_ctx->degree() > 1)
        {
            // Injected global TP context (from DomainCommunicatorRegistry or test)
            // — apply equal-split assignment without auto-creating a world context.
            if (!applyGlobalTPContextAssignment(graph_config, injected_global_tp_ctx, tp_device_idx))
            {
                return nullptr;
            }
        }
        else
        {
            // Single rank configuration (testable runner doesn't use MPI by default)
            setFullDimensions(graph_config);
        }
        if (!configureStaticMoEExpertRange(graph_config))
        {
            return nullptr;
        }

        installTPExecutorCancellation(graph_config, graph_config.tp_ctx, "[InferenceRunner] Testable");

        if (auto concrete_model_ctx = std::dynamic_pointer_cast<ModelContext>(model_ctx))
        {
            if (auto concrete_weight_mgr = concrete_model_ctx->concreteWeightManager())
            {
                configureWeightManagerForGraph(
                    *concrete_weight_mgr,
                    architecture,
                    graph_config,
                    graph_config.tp_config != nullptr);
            }
        }

        LOG_DEBUG("[InferenceRunner] TestableGraphConfig: "
                  << "vocab=" << graph_config.vocab_size
                  << ", d_model=" << graph_config.d_model
                  << ", n_layers=" << graph_config.n_layers
                  << ", n_heads=" << graph_config.n_heads
                  << ", n_kv_heads=" << graph_config.n_kv_heads
                  << ", d_ff=" << graph_config.d_ff);

        // Create Dependencies struct
        DeviceGraphOrchestrator::Dependencies deps;
        deps.model_ctx = model_ctx;
        deps.graph_builder = GraphBuilderRegistry::create(architecture, graph_config, nullptr);
        deps.graph_builder->setModelContext(model_ctx);
        deps.turboquant_ctx = std::move(turboquant_ctx);
        deps.kv_rotation = std::move(kv_rotation);
        deps.domain_tp_contexts = std::move(owned_domain_tp_contexts);
        if (config.pp_stage_config.has_value())
            deps.pp_stage_config = config.pp_stage_config.value();
        if (auto concrete_model_ctx = std::dynamic_pointer_cast<ModelContext>(model_ctx))
            deps.weight_manager = concrete_model_ctx->concreteWeightManager();
        // topology and collective_ctx left as nullptr for single-rank testing

        // Create DeviceGraphOrchestrator with injected dependencies
        auto orchestrator = std::make_unique<DeviceGraphOrchestrator>(
            std::move(deps));

        // Initialize graph cache
        orchestrator->initializeGraphCache(graph_config.n_layers);

        // Initialize inference state via schema-driven BufferArena path
        InferenceStateInitConfig init_config;
        init_config.use_mapped_memory = config.use_mapped_memory;

        if (!orchestrator->initializeInferenceStateFromArena(
                config.batch_size, config.max_seq_len, device, init_config))
        {
            LOG_ERROR("[InferenceRunner] Failed to initialize inference state (arena path)");
            return nullptr;
        }

        auto load_weight_for_runner = [model_ctx, device, tp_config = graph_config.tp_config, tp_rank = graph_config.local_rank](const std::string &name)
            -> std::shared_ptr<TensorBase>
        {
            if (tp_config)
            {
                auto weight_mgr = model_ctx->weightManager();
                auto concrete_mgr = std::dynamic_pointer_cast<WeightManager>(weight_mgr);
                if (concrete_mgr)
                {
                    try
                    {
                        const auto &assignment = tp_config->forRank(tp_rank);
                        return concrete_mgr->getShardedWeightForAssignment(name, device, assignment, /*layer_idx=*/-1);
                    }
                    catch (const std::out_of_range &)
                    {
                        LOG_TRACE("[InferenceRunner] Device " << device.to_string()
                                                              << " not in runner TensorParallelConfig, using standard loader for " << name);
                    }
                }
            }

            return model_ctx->getWeightForDevice(name, device);
        };

        // Configure weights: PP-aware vs full.
        // When pp_stage_config is present but model context is non-concrete (e.g. MockModelContext
        // in unit tests), skip PP weight materialization and fall through to the full model path.
        auto pp_concrete_ctx = config.pp_stage_config.has_value()
                                   ? std::dynamic_pointer_cast<ModelContext>(model_ctx)
                                   : nullptr;
        auto pp_concrete_mgr = pp_concrete_ctx ? pp_concrete_ctx->concreteWeightManager() : nullptr;
        if (config.pp_stage_config.has_value() && pp_concrete_ctx && pp_concrete_mgr)
        {
            const auto &pp_cfg = config.pp_stage_config.value();
            auto &concrete_model_ctx = pp_concrete_ctx;
            auto &concrete_weight_mgr = pp_concrete_mgr;

            concrete_weight_mgr->setWeightShardingConfig(
                SchemaFactoryRegistry::getWeightShardingConfig(architecture));

            auto schema_factory = SchemaFactoryRegistry::getFactory(architecture);
            auto validation = validateLayerWeights(
                *schema_factory,
                concrete_model_ctx->totalBlockCount(),
                [concrete_model_ctx](const std::string &name)
                { return concrete_model_ctx->hasTensor(name); },
                pp_cfg.first_layer,
                pp_cfg.last_layer);

            if (!validation.success)
            {
                LOG_ERROR("[InferenceRunner] " << validation.error_message());
                return nullptr;
            }

            auto weight_plan = buildSingleDeviceWeightPlan(
                *concrete_weight_mgr,
                *concrete_model_ctx,
                validation,
                device,
                graph_config.tp_config.get(),
                &pp_cfg,
                graph_config.tp_config ? graph_config.local_rank : -1);
            if (!installPreparedWeightStoreForPlan(*concrete_weight_mgr, config, weight_plan, "[InferenceRunner] PP stage"))
                return nullptr;
            auto frozen_weights = concrete_weight_mgr->materialize(weight_plan);
            auto weight_bindings = makeModelWeightBindings(frozen_weights);
            auto legacy_weights = toLegacyModelWeights(weight_bindings);

            if (pp_cfg.has_embedding && !legacy_weights.embedding_table)
            {
                LOG_ERROR("[InferenceRunner] Frozen PP stage has_embedding=true but embedding missing");
                return nullptr;
            }
            if (pp_cfg.has_lm_head && (!legacy_weights.final_norm || !legacy_weights.lm_head))
            {
                LOG_ERROR("[InferenceRunner] Frozen PP stage has_lm_head=true but final_norm/lm_head missing");
                return nullptr;
            }

            // Nested TP-in-PP RankOrchestrators may have run broad device
            // finalization before this runner materialized its PP-stage frozen
            // bindings. Re-run the stage-filtered preparation here, after
            // materialization, so GPU pipeline handles are registered under the
            // graph binding ids instead of pipeline-local ids.
            bool prepare_ok = concrete_weight_mgr->prepareWeightsForDevice(
                frozen_weights,
                device);

            if (!prepare_ok)
            {
                LOG_WARN("[InferenceRunner] PP stage weight preparation had issues for device "
                         << device.to_string() << " layers [" << pp_cfg.first_layer << ", " << pp_cfg.last_layer << ")");
            }

            orchestrator->setFrozenWeightSet(
                std::make_unique<FrozenModelWeightSet>(std::move(frozen_weights)));
            orchestrator->initializePreparedWeightStore(device);
        }
        else
        {
            if (config.pp_stage_config.has_value())
            {
                // pp_stage_config is set but model context is non-concrete (e.g. MockModelContext):
                // skip PP weight filtering and use the interface/mock weight provider.
                LOG_DEBUG("[InferenceRunner] pp_stage_config present but non-concrete model ctx; "
                          "using interface weight path (test/mock)");
            }
            bool configured_from_frozen = false;
            if (graph_config.tp_config)
            {
                auto concrete_model_ctx = std::dynamic_pointer_cast<ModelContext>(model_ctx);
                auto concrete_weight_mgr = concrete_model_ctx ? concrete_model_ctx->concreteWeightManager() : nullptr;
                if (concrete_model_ctx && concrete_weight_mgr)
                {
                    auto schema_factory = SchemaFactoryRegistry::getFactory(architecture);
                    auto validation = validateLayerWeights(
                        *schema_factory,
                        concrete_model_ctx->totalBlockCount(),
                        [concrete_model_ctx](const std::string &name)
                        { return concrete_model_ctx->hasTensor(name); });

                    if (!validation.success)
                    {
                        LOG_ERROR("[InferenceRunner] " << validation.error_message());
                        return nullptr;
                    }

                    auto weight_plan = buildSingleDeviceWeightPlan(
                        *concrete_weight_mgr,
                        *concrete_model_ctx,
                        validation,
                        device,
                        graph_config.tp_config.get(),
                        nullptr,
                        graph_config.tp_config ? graph_config.local_rank : -1);
                    if (!installPreparedWeightStoreForPlan(*concrete_weight_mgr, config, weight_plan, "[InferenceRunner] LocalTP"))
                        return nullptr;
                    auto frozen_weights = concrete_weight_mgr->materialize(weight_plan);
                    auto weight_bindings = makeModelWeightBindings(frozen_weights);
                    auto legacy_weights = toLegacyModelWeights(weight_bindings);

                    if (!legacy_weights.embedding_table || !legacy_weights.final_norm || !legacy_weights.lm_head)
                    {
                        LOG_ERROR("[InferenceRunner] Missing global weights from materialized LocalTP bindings");
                        return nullptr;
                    }

                    if (!concrete_weight_mgr->prepareWeightsForDevice(
                            frozen_weights,
                            device,
                            /*include_expert_jobs=*/false))
                    {
                        LOG_WARN("[InferenceRunner] LocalTP binding-driven weight preparation had issues for device "
                                 << device.to_string());
                    }

                    if (graph_config.moe.expert_overlay_runtime_plan)
                    {
                        const auto *execution_plan = graph_config.moe.expert_overlay_execution_plan.get();
                        if (!concrete_weight_mgr->prepareMoEExpertOverlayWeights(
                                *graph_config.moe.expert_overlay_runtime_plan,
                                &frozen_weights,
                                execution_plan))
                        {
                            LOG_ERROR("[InferenceRunner] LocalTP failed to prepare MoE expert overlay weights for device "
                                      << device.to_string());
                            return nullptr;
                        }
                    }

                    orchestrator->setFrozenWeightSet(
                        std::make_unique<FrozenModelWeightSet>(std::move(frozen_weights)));
                    orchestrator->initializePreparedWeightStore(device);
                    configured_from_frozen = true;
                }
            }

            if (!configured_from_frozen)
            {
                if (config.prepared_weight_store)
                {
                    if (auto concrete_model_ctx = std::dynamic_pointer_cast<ModelContext>(model_ctx))
                    {
                        if (auto concrete_weight_mgr = concrete_model_ctx->concreteWeightManager())
                        {
                            concrete_weight_mgr->setPreparedWeightStore(config.prepared_weight_store);
                        }
                    }
                }

                // Full model: load all global weights
                auto weights = config_builder->buildWeights(load_weight_for_runner);

                if (!weights.embedding_table || !weights.final_norm || !weights.lm_head)
                {
                    LOG_ERROR("[InferenceRunner] Missing global weights from IModelContext");
                    return nullptr;
                }

                orchestrator->setWeights(weights);
                orchestrator->initializePreparedWeightStore(device);
            }
        }

        if (device.is_cpu() && architecture == "qwen35moe" && graph_config.moe.num_experts > 0)
        {
            if (!orchestrator->materializeForwardGraphForShape(/*seq_len=*/1, config.batch_size))
            {
                LOG_ERROR("[InferenceRunner] Failed eager Qwen35 MoE expert graph materialization on "
                          << device.to_string());
                return nullptr;
            }
        }

        LOG_INFO("[InferenceRunner] Testable DeviceGraphOrchestrator created successfully");

        return orchestrator;
    }

    // =========================================================================
    // Multi-Device Orchestrator Factory Functions
    // =========================================================================

    std::unique_ptr<IRankOrchestrator> createRankOrchestrator(
        std::shared_ptr<IModelContext> model_ctx,
        std::unique_ptr<ILocalTPContext> tp_ctx,
        const RankOrchestrator::Config &config)
    {
        if (!model_ctx)
        {
            LOG_ERROR("[InferenceRunner] model_ctx is null for createRankOrchestrator");
            return nullptr;
        }

        if (!tp_ctx)
        {
            LOG_ERROR("[InferenceRunner] tp_ctx is null for createRankOrchestrator");
            return nullptr;
        }

        if (!config.validate())
        {
            LOG_ERROR("[InferenceRunner] Invalid RankOrchestrator config");
            return nullptr;
        }

        LOG_INFO("[InferenceRunner] Creating RankOrchestrator with "
                 << config.devices.size() << " devices, backend="
                 << static_cast<int>(config.backend));

        try
        {
            return std::make_unique<RankOrchestrator>(
                model_ctx, config, std::move(tp_ctx));
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("[InferenceRunner] Failed to create RankOrchestrator: " << e.what());
            return nullptr;
        }
    }

    std::unique_ptr<IRankOrchestrator> createTestableRankOrchestrator(
        std::shared_ptr<IModelContext> model_ctx,
        std::vector<std::unique_ptr<IInferenceRunner>> device_runners,
        std::unique_ptr<ILocalTPContext> tp_ctx,
        const RankOrchestrator::Config &config)
    {
        if (!model_ctx)
        {
            LOG_ERROR("[InferenceRunner] model_ctx is null for createTestableRankOrchestrator");
            return nullptr;
        }

        if (device_runners.empty())
        {
            LOG_ERROR("[InferenceRunner] device_runners is empty");
            return nullptr;
        }

        LOG_DEBUG("[InferenceRunner] Creating testable RankOrchestrator with "
                  << device_runners.size() << " injected runners");

        try
        {
            return RankOrchestrator::createForTest(
                model_ctx, std::move(device_runners), std::move(tp_ctx), config);
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("[InferenceRunner] Failed to create testable RankOrchestrator: " << e.what());
            return nullptr;
        }
    }

} // namespace llaminar2
