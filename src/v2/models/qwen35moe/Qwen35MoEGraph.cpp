/**
 * @file Qwen35MoEGraph.cpp
 * @brief Qwen 3.5 MoE compute graph builder implementation
 */

#include "Qwen35MoEGraph.h"
#include "Qwen35MoESchema.h"
#include "../../utils/Logger.h"
#include "../../execution/compute_stages/ComputeStageFactory.h"
#include "../../execution/compute_stages/stages/MoERoutingStage.h"
#include "../../execution/compute_stages/stages/MoEExpertComputeStage.h"
#include "../../execution/moe/MoEExpertParallelPlan.h"
#include "../../execution/moe/MoEExpertOverlayExecutionPlan.h"
#include "../../execution/moe/MoEExpertOverlayRuntimePlan.h"
#include "../../memory/BufferId.h"
#include "../../execution/local_execution/graph/GraphResolver.h"
#include "../../tensors/Tensors.h"
#include "../../utils/DebugEnv.h"

#include <algorithm>
#include <cctype>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace llaminar2
{

    namespace
    {
        const ExpertLayerPlacement *findExpertOverlayPlacement(
            const MoEExpertParallelPlan &plan,
            int layer_idx)
        {
            auto it = std::find_if(plan.placements.begin(), plan.placements.end(),
                                   [layer_idx](const ExpertLayerPlacement &placement)
                                   {
                                       return placement.layer == layer_idx;
                                   });
            return it == plan.placements.end() ? nullptr : &(*it);
        }

        bool isUsableExpertOverlayPlacement(
            const ExpertLayerPlacement &placement,
            const MoEExpertParallelPlan &plan,
            int num_experts,
            int layer_idx)
        {
            if (static_cast<int>(placement.routed_expert_tier.size()) != num_experts)
            {
                LOG_ERROR("[Qwen35MoEGraph] Expert overlay placement for layer " << layer_idx
                                                                                 << " covers " << placement.routed_expert_tier.size()
                                                                                 << " experts, expected " << num_experts);
                return false;
            }

            for (int expert = 0; expert < num_experts; ++expert)
            {
                const int tier_index = placement.routed_expert_tier[static_cast<size_t>(expert)];
                if (tier_index < 0 || tier_index >= static_cast<int>(plan.routed_tiers.size()))
                {
                    LOG_ERROR("[Qwen35MoEGraph] Expert overlay placement for layer " << layer_idx
                                                                                     << " maps expert " << expert << " to invalid tier " << tier_index);
                    return false;
                }
            }

            return true;
        }

        std::vector<bool> expertMaskForTier(
            const ExpertLayerPlacement &placement,
            int num_experts,
            int tier_index)
        {
            std::vector<bool> mask(static_cast<size_t>(num_experts), false);
            for (int expert = 0; expert < num_experts; ++expert)
            {
                mask[static_cast<size_t>(expert)] =
                    placement.routed_expert_tier[static_cast<size_t>(expert)] == tier_index;
            }
            return mask;
        }

        std::string nodeSuffixForTier(const ExpertRoutedTier &tier, int tier_index)
        {
            std::string suffix = "tier" + std::to_string(tier_index);
            if (!tier.name.empty())
            {
                suffix += "_";
                for (unsigned char ch : tier.name)
                    suffix += std::isalnum(ch) ? static_cast<char>(std::tolower(ch)) : '_';
            }
            return suffix;
        }

        std::shared_ptr<MoEExpertOverlayRuntimePlan> runtimePlanForGraph(
            const GraphConfig &config)
        {
            if (config.moe.expert_overlay_runtime_plan)
                return config.moe.expert_overlay_runtime_plan;
            if (!config.moe.expert_parallel_plan)
                return nullptr;
            return resolveMoEExpertOverlayRuntimePlan(config.moe.expert_parallel_plan);
        }

        std::shared_ptr<const MoEExpertOverlayExecutionPlan> executionPlanForGraph(
            const GraphConfig &config,
            const std::shared_ptr<MoEExpertOverlayRuntimePlan> &runtime_plan)
        {
            if (config.moe.expert_overlay_execution_plan)
                return config.moe.expert_overlay_execution_plan;
            if (!config.moe.expert_parallel_plan || !config.moe.expert_parallel_plan->isTieredOverlay())
                return nullptr;

            const int current_rank = runtime_plan ? runtime_plan->currentWorldRank() : 0;
            return std::make_shared<MoEExpertOverlayExecutionPlan>(
                resolveMoEExpertOverlayExecutionPlan(config.moe.expert_parallel_plan, current_rank));
        }

#if 0
        std::shared_ptr<DisabledDomainRunner> overlayDomainRuntimeForGraph(
            const GraphConfig &config,
            const std::shared_ptr<MoEExpertOverlayRuntimePlan> &runtime_plan)
        {
            if (config.moe.overlay_domain_runtime)
                return config.moe.overlay_domain_runtime;

            DisabledDomainRunnerImpl::Config runtime_config;
            runtime_config.runtime_plan = runtime_plan;
            runtime_config.execution_plan = executionPlanForGraph(config, runtime_plan);
            runtime_config.enable_compatibility_fallback = true;
            return makeDisabledDomainRunner(std::move(runtime_config));
        }

        const ExpertComputeDomain *sourceDomainForName(
            const MoEExpertParallelPlan &plan,
            const std::string &domain_name)
        {
            auto it = std::find_if(plan.domains.begin(), plan.domains.end(),
                                   [&](const auto &domain)
                                   {
                                       return domain.name == domain_name;
                                   });
            return it == plan.domains.end() ? nullptr : &(*it);
        }

        int participantIndexForDevice(
            const MoEOverlayRuntimeDomain &domain,
            DeviceId device)
        {
            for (const auto &participant : domain.participants)
            {
                if (participant.local_device == device)
                    return participant.participant_index;
            }
            return domain.participants.empty() ? 0 : domain.participants.front().participant_index;
        }

        int ownerParticipantIndexFor(const MoEOverlayRuntimeDomain &domain)
        {
            if (domain.owner_rank >= 0)
            {
                for (const auto &participant : domain.participants)
                {
                    if (participant.world_rank_known && participant.world_rank == domain.owner_rank)
                        return participant.participant_index;
                }
            }
            return domain.participants.empty() ? 0 : domain.participants.front().participant_index;
        }

        int executorParticipantIndexFor(const MoEOverlayRuntimeDomain &domain)
        {
            for (const auto &participant : domain.participants)
            {
                if (participant.address == domain.primary_participant)
                    return participant.participant_index;
            }
            return ownerParticipantIndexFor(domain);
        }

        int stableDispatchGroupId(int domain_id, int layer_idx, int tier_index)
        {
            uint32_t value = static_cast<uint32_t>(domain_id);
            value ^= static_cast<uint32_t>(layer_idx + 1) * 16777619u;
            value ^= static_cast<uint32_t>(tier_index + 1) * 2166136261u;
            return static_cast<int>(value & 0x7fffffff);
        }

        MoEOverlayDispatchGroup dispatchGroupForTier(
            const MoEOverlayRuntimeDomain &runtime_domain,
            int layer_idx,
            int tier_index,
            DeviceId participant_device)
        {
            MoEOverlayDispatchGroup group;
            group.domain_id = DisabledCpuParticipant::stableDomainId(runtime_domain.name);
            group.layer_id = layer_idx;
            group.dispatch_group_id = stableDispatchGroupId(group.domain_id, layer_idx, tier_index);
            group.participant_count = std::max<int>(1, static_cast<int>(runtime_domain.participants.size()));
            group.participant_index = participantIndexForDevice(runtime_domain, participant_device);
            group.owner_participant_index = ownerParticipantIndexFor(runtime_domain);
            group.executor_participant_index = executorParticipantIndexFor(runtime_domain);
            group.stage_sequence = static_cast<uint64_t>(std::max(layer_idx, 0)) * 1024ull +
                                   static_cast<uint64_t>(std::max(tier_index, 0));
            group.microbatch_id = 0;
            group.decode_sequence = 0;
            return group;
        }

        bool isCpuNodeLocalDomain(const MoEOverlayRuntimeDomain &domain)
        {
            if (domain.kind != ExpertDomainKind::NodeLocalTP || domain.participants.empty())
                return false;
            return std::all_of(domain.participants.begin(), domain.participants.end(),
                               [](const auto &participant)
                               {
                                   return participant.address.isCPU();
                               });
        }

        bool isAcceleratorLocalTPTensorParallelDomain(const MoEOverlayRuntimeDomain &domain)
        {
            if (domain.kind != ExpertDomainKind::LocalTP ||
                domain.compute_kind != ExpertDomainComputeKind::TensorParallelExperts ||
                domain.participants.size() < 2)
            {
                return false;
            }

            return std::all_of(domain.participants.begin(), domain.participants.end(),
                               [](const auto &participant)
                               {
                                   return participant.address.isGPU() &&
                                          participant.locally_addressable &&
                                          participant.local_device.is_gpu();
                               });
        }

        int continuationRootWorldRank(const MoEExpertOverlayRuntimePlan &runtime_plan)
        {
            const auto &domain = runtime_plan.continuationDomain();
            if (domain.primary_world_rank_known)
                return domain.primary_world_rank;
            return runtime_plan.currentWorldRank();
        }

        std::string participantContextForDevice(
            const MoEOverlayRuntimeDomain &domain,
            DeviceId device)
        {
            for (const auto &participant : domain.participants)
            {
                if (participant.local_device != device)
                    continue;

                std::ostringstream context;
                context << participant.participant_index;
                if (participant.world_rank_known)
                    context << "/rank" << participant.world_rank;
                return context.str();
            }
            return "unknown";
        }

        [[noreturn]] void failUnsupportedMultiParticipantTier(
            const ExpertRoutedTier &tier,
            const MoEOverlayRuntimeDomain &domain,
            size_t tier_index,
            int layer_idx)
        {
            std::ostringstream message;
            message << "MoE expert overlay routed tier " << tier_index;
            if (!tier.name.empty())
                message << " ('" << tier.name << "')";
            message << " for layer " << layer_idx
                    << " resolves to multi-participant domain '" << domain.name << "'"
                    << " kind=" << toString(domain.kind)
                    << " compute=" << toString(domain.compute_kind)
                    << " participants=" << domain.participants.size()
                    << "; Bridge Phase 5A no longer lowers active routed tier work to primary participant "
                    << domain.primary_device.to_string()
                    << ". Bridge Phase 5D wires accelerator LocalTP TensorParallelExperts tiers only; "
                    << "this domain shape still requires a later production lowering bridge.";
            throw std::runtime_error(message.str());
        }

        [[noreturn]] void failMissingLocalTPContext(
            const ExpertRoutedTier &tier,
            const MoEOverlayRuntimeDomain &domain,
            size_t tier_index,
            int layer_idx)
        {
            std::ostringstream message;
            message << "Bridge Phase 5D missing LocalTP domain context for MoE expert overlay routed tier "
                    << tier_index;
            if (!tier.name.empty())
                message << " ('" << tier.name << "')";
            message << " for layer " << layer_idx
                    << ": domain '" << domain.name << "' requires LocalTP TensorParallelExperts over "
                    << domain.participants.size() << " accelerator participants. "
                    << "Populate GraphConfig::domain_tp_contexts['" << tier.domain
                    << "'] with the domain-scoped ILocalTPContext before graph construction.";
            throw std::runtime_error(message.str());
        }

        ILocalTPContext *requireLocalTPContextForTier(
            const GraphConfig &config,
            const ExpertRoutedTier &tier,
            const MoEOverlayRuntimeDomain &domain,
            size_t tier_index,
            int layer_idx)
        {
            auto it = config.domain_tp_contexts.find(tier.domain);
            if (it == config.domain_tp_contexts.end() || !it->second)
                failMissingLocalTPContext(tier, domain, tier_index, layer_idx);

            std::string reason;
            if (!DisabledLocalTPParticipant::canExecute(domain, *it->second, &reason))
            {
                std::ostringstream message;
                message << "Bridge Phase 5D LocalTP domain context mismatch for MoE expert overlay routed tier "
                        << tier_index;
                if (!tier.name.empty())
                    message << " ('" << tier.name << "')";
                message << " for layer " << layer_idx
                        << ": domain '" << domain.name << "' cannot use GraphConfig::domain_tp_contexts['"
                        << tier.domain << "']: " << reason;
                throw std::runtime_error(message.str());
            }

            return it->second;
        }
#endif
    } // namespace

    // =========================================================================
    // Constructors
    // =========================================================================

    Qwen35MoEGraph::Qwen35MoEGraph(
        std::shared_ptr<ModelContext> model_ctx,
        std::shared_ptr<IMPIContext> mpi_ctx,
        const GraphConfig &config)
        : Qwen35Graph(std::move(model_ctx), std::move(mpi_ctx), config)
    {
    }

    Qwen35MoEGraph::Qwen35MoEGraph(
        const GraphConfig &config,
        std::shared_ptr<IMPIContext> mpi_ctx)
        : Qwen35Graph(config, std::move(mpi_ctx))
    {
    }

    // =========================================================================
    // Schema
    // =========================================================================

    GraphSchema Qwen35MoEGraph::getSchema() const
    {
        Qwen35MoESchemaFactory factory;
        GraphSchema schema = factory.createSchema();

        // Populate layer_template_names from config_.layer_types
        if (!config_.layer_types.empty())
        {
            schema.layer_template_names.resize(config_.n_layers);
            for (int i = 0; i < config_.n_layers; ++i)
            {
                schema.layer_template_names[i] = config_.layer_types[i];
            }
        }

        return schema;
    }

    // =========================================================================
    // Resolver Config (MoE buffer registration)
    // =========================================================================

    GraphResolverConfig Qwen35MoEGraph::getResolverConfig(int seq_len) const
    {
        // Start with base Qwen35 resolver config (GDN buffers, etc.)
        GraphResolverConfig config = Qwen35Graph::getResolverConfig(seq_len);

        // Add MoE-specific custom formulas
        int expert_intermediate = config_.moe.intermediate_size;
        int top_k = config_.moe.top_k;

        config.custom_formulas["moe_top_k"] = static_cast<size_t>(top_k);
        config.custom_formulas["moe_expert_intermediate"] = static_cast<size_t>(expert_intermediate);

        // Add MoE buffer name → BufferId mappings
        config.buffer_name_to_id["moe_expert_indices"] = BufferId::MOE_EXPERT_INDICES;
        config.buffer_name_to_id["moe_expert_weights"] = BufferId::MOE_EXPERT_WEIGHTS;
        config.buffer_name_to_id["moe_combined_output"] = BufferId::MOE_COMBINED_OUTPUT;
        config.buffer_name_to_id["moe_shared_expert_output"] = BufferId::MOE_SHARED_EXPERT_OUTPUT;
        config.buffer_name_to_id["moe_gate_scratch"] = BufferId::MOE_GATE_SCRATCH;
        config.buffer_name_to_id["moe_up_scratch"] = BufferId::MOE_UP_SCRATCH;

        LOG_DEBUG("[Qwen35MoEGraph::getResolverConfig] MoE formulas: "
                  << "moe_top_k=" << top_k
                  << ", moe_expert_intermediate=" << expert_intermediate);

        return config;
    }

    // =========================================================================
    // MoE FFN Graph Building
    // =========================================================================

    ComputeGraph Qwen35MoEGraph::buildFFNGraph(
        const LayerWeights &layer,
        ActivationBuffers &buffers,
        int layer_idx,
        int seq_len,
        int batch_size,
        DeviceId device)
    {
        // If this layer doesn't have MoE weights, fall back to dense FFN
        if (!layer.moe_gate || !layer.moe_gate_exps)
        {
            return Qwen35Graph::buildFFNGraph(layer, buffers, layer_idx, seq_len, batch_size, device);
        }

        ComputeGraph graph;
        std::string prefix = "layer" + std::to_string(layer_idx) + "_";
        std::string ffn_terminal;
        int total_tokens = batch_size * seq_len;
        LayerWeightBindings layer_bindings = layerWeightBindingsForGraph(layer_idx);

        auto overlay_runtime_plan = runtimePlanForGraph(config_);
        const auto &overlay_plan = overlay_runtime_plan
                                       ? overlay_runtime_plan->sourcePlanPtr()
                                       : config_.moe.expert_parallel_plan;
        auto domainContainsDevice = [](const MoEOverlayRuntimeDomain &domain, DeviceId candidate)
        {
            return std::any_of(domain.participants.begin(), domain.participants.end(),
                               [&](const MoEOverlayDomainParticipant &participant)
                               {
                                   return participant.local_device == candidate;
                               });
        };

        if (overlay_runtime_plan)
        {
            const auto &continuation_domain = overlay_runtime_plan->continuationDomain();
            DeviceId continuation_device = overlay_runtime_plan->continuationDevice();
            if (continuation_device.is_valid() && continuation_device != device &&
                !domainContainsDevice(continuation_domain, device))
            {
                LOG_INFO("[Qwen35MoEGraph] Layer " << layer_idx
                                                   << " using MoE overlay continuation_domain root device "
                                                   << continuation_device.to_string()
                                                   << " instead of caller device " << device.to_string());
                device = continuation_device;
            }
        }

        // =====================================================================
        // Stage 1: Pre-FFN RMSNorm (fused with attention residual add)
        // =====================================================================
        {
            FusedResidualNormStage::Params fused_params;
            fused_params.device_id = device;
            fused_params.input = buffers.attn_proj;
            fused_params.residual = buffers.current_hidden;
            fused_params.gamma = layer.ffn_norm;
            fused_params.norm_output = buffers.normalized;
            fused_params.eps = config_.rms_norm_eps;
            fused_params.seq_len = total_tokens;
            fused_params.hidden_dim = config_.d_model;
            fused_params.input_buffer_id = BufferId::ATTN_PROJ;
            fused_params.residual_buffer_id = BufferId::HIDDEN_STATE;
            fused_params.norm_output_buffer_id = BufferId::NORMALIZED;

            graph.addNode(prefix + "ffn_norm",
                          ComputeStageFactory::createFusedResidualNorm(fused_params),
                          device);
            ffn_terminal = prefix + "ffn_norm";
        }

        // =====================================================================
        // Stage 2: MoE Routing (softmax top-k expert selection)
        // =====================================================================
        TensorBase *routing_indices = buffers.get(BufferId::MOE_EXPERT_INDICES);
        TensorBase *routing_weights = buffers.get(BufferId::MOE_EXPERT_WEIGHTS);

        {
            MoERoutingStage::Params route_params;
            route_params.device_id = device;
            route_params.input = buffers.normalized;
            route_params.seq_len = total_tokens;
            route_params.d_model = config_.d_model;
            route_params.gate_weights = layer.moe_gate;
            route_params.num_experts = config_.moe.num_experts;
            route_params.top_k = config_.moe.top_k;
            route_params.norm_topk_prob = config_.moe.norm_topk_prob;
            route_params.layer_idx = layer_idx;
            route_params.decode_histogram = config_.moe.decode_histogram;
            route_params.output_indices = routing_indices;
            route_params.output_weights = routing_weights;
            route_params.input_buffer_id = BufferId::NORMALIZED;
            route_params.output_indices_buffer_id = BufferId::MOE_EXPERT_INDICES;
            route_params.output_weights_buffer_id = BufferId::MOE_EXPERT_WEIGHTS;

            graph.addNode(prefix + "moe_routing",
                          ComputeStageFactory::createMoERouting(route_params),
                          device);
            graph.addDependency(prefix + "moe_routing", prefix + "ffn_norm");
        }

        // =====================================================================
        // Stage 3: MoE Expert Compute (routed expert SwiGLU FFN)
        // =====================================================================
        TensorBase *moe_output = buffers.get(BufferId::MOE_COMBINED_OUTPUT);
        bool overlay_reduce_pending = false;
        std::vector<const ITensor *> overlay_reduce_partials;
        std::vector<std::shared_ptr<TensorBase>> overlay_reduce_partial_lifetimes;
        std::vector<MoEExpertParallelReducePartialInfo> overlay_reduce_partial_infos;
        std::vector<TensorBase *> overlay_reduce_sparse_scratch;
        std::vector<std::shared_ptr<TensorBase>> overlay_reduce_sparse_scratch_lifetimes;
        std::vector<std::string> overlay_reduce_dependencies;
        std::shared_ptr<MoEExpertParallelReduceDiagnostics> overlay_reduce_diagnostics;
        std::string overlay_continuation_domain;
        DeviceId overlay_continuation_device = device;

        {
            // Infer expert intermediate size from weight shape
            int expert_intermediate = config_.moe.intermediate_size;
            if (expert_intermediate == 0 && layer.moe_gate_exps)
            {
                // gate_exps shape: [num_experts, intermediate, d_model] or rows=num_experts*intermediate
                size_t total_rows = layer.moe_gate_exps->rows();
                expert_intermediate = static_cast<int>(total_rows / config_.moe.num_experts);
            }

            auto makeExpertParams = [&](TensorBase *output,
                                        BufferId output_buffer_id,
                                        std::vector<bool> expert_mask,
                                        DeviceId stage_device)
            {
                MoEExpertComputeStage::Params expert_params;
                expert_params.device_id = stage_device;
                expert_params.input = buffers.normalized;
                expert_params.seq_len = total_tokens;
                expert_params.d_model = config_.d_model;
                expert_params.num_experts = config_.moe.num_experts;
                expert_params.top_k = config_.moe.top_k;
                expert_params.gate_exps = layer.moe_gate_exps;
                expert_params.up_exps = layer.moe_up_exps;
                expert_params.down_exps = layer.moe_down_exps;
                expert_params.expert_intermediate = expert_intermediate;
                expert_params.layer_idx = layer_idx;
                expert_params.routing_indices = routing_indices;
                expert_params.routing_weights = routing_weights;
                expert_params.routing_indices_buffer_id = BufferId::MOE_EXPERT_INDICES;
                expert_params.routing_weights_buffer_id = BufferId::MOE_EXPERT_WEIGHTS;
                expert_params.output = output;
                expert_params.output_buffer_id = output_buffer_id;
                expert_params.input_buffer_id = BufferId::NORMALIZED;
                expert_params.prepared_store = prepared_weight_store_;
                expert_params.expert_mask = std::move(expert_mask);

                if (config_.moe.expert_mode == MoEExpertMode::ExpertParallel &&
                    expert_params.expert_mask.empty())
                {
                    expert_params.local_expert_start = config_.moe.local_expert_start;
                    expert_params.local_expert_count = config_.moe.local_expert_count;
                }

                const int gpu_cache_experts = debugEnv().moe_rebalance.gpu_cache_experts_per_layer;
                if (expert_params.expert_mask.empty() && gpu_cache_experts > 0)
                {
                    expert_params.expert_mask.assign(config_.moe.num_experts, false);
                    for (int expert = 0; expert < config_.moe.num_experts; ++expert)
                        expert_params.expert_mask[expert] = !stage_device.is_gpu();
                    LOG_DEBUG("[Qwen35MoEGraph] Initial MoE GPU expert cache bootstrap mask: device="
                              << stage_device.to_string() << " layer=" << layer_idx
                              << " cpu_initial_owner=" << (!stage_device.is_gpu()));
                }

                // GPU scratch buffers
                expert_params.gate_scratch = buffers.get(BufferId::MOE_GATE_SCRATCH);
                expert_params.up_scratch = buffers.get(BufferId::MOE_UP_SCRATCH);

                return expert_params;
            };

            auto prepareExpertParams = [&](MoEExpertComputeStage::Params &expert_params,
                                           DeviceId stage_device,
                                           const std::string &placement_context = {},
                                           const std::string &registry_domain_name = {})
            {
                auto withPlacementContext = [&](const std::string &reason)
                {
                    if (placement_context.empty())
                        return reason;
                    return placement_context + ": " + reason;
                };

                // Extract per-expert 2D views from 3D packed tensors (required)
                if (!MoEExpertComputeStage::extractExpertViews(expert_params))
                {
                    LOG_ERROR("[Qwen35MoEGraph] Failed to extract expert views for layer " << layer_idx);
                    return false;
                }

                // Set expert_registry for dynamic rebalancing registry updates
                if (model_ctx_)
                {
                    auto weight_mgr = model_ctx_->concreteWeightManager();
                    if (weight_mgr)
                        expert_params.expert_registry = &weight_mgr->expertGemmRegistry();
                }

                // CPU prepares engines inline. GPU graph construction must consume the
                // unified pipeline registry and fail before execution if required
                // resident expert engines are absent.
                if (stage_device.is_gpu())
                {
                    const bool has_active_masked_expert = hasActiveExpertMask(expert_params.expert_mask);

                    auto weight_mgr = model_ctx_ ? model_ctx_->concreteWeightManager() : nullptr;
                    if (!weight_mgr)
                    {
                        if (expert_params.expert_mask.empty() || has_active_masked_expert)
                            failMissingGpuExpertGemmEngines(stage_device, layer_idx,
                                                            withPlacementContext(
                                                                "WeightManager unavailable (" +
                                                                describeMissingExpertGemmEngine(config_.moe.num_experts,
                                                                                                expert_params.expert_mask,
                                                                                                expert_params.prepared_gate_gemm,
                                                                                                expert_params.prepared_up_gemm,
                                                                                                expert_params.prepared_down_gemm) +
                                                                ")"));

                        expert_params.prepared_gate_gemm.assign(config_.moe.num_experts, nullptr);
                        expert_params.prepared_up_gemm.assign(config_.moe.num_experts, nullptr);
                        expert_params.prepared_down_gemm.assign(config_.moe.num_experts, nullptr);
                    }
                    else
                    {
                        const auto &registry = weight_mgr->expertGemmRegistry();
                        const bool domain_scoped = !registry_domain_name.empty();
                        const bool complete_layer = domain_scoped
                                                        ? registry.hasCompleteLayerInDomain(
                                                              registry_domain_name, stage_device, layer_idx, config_.moe.num_experts)
                                                        : registry.hasCompleteLayer(stage_device, layer_idx, config_.moe.num_experts);
                        const bool populated = domain_scoped
                                                   ? registry.populateExpertEnginesForDomain(
                                                         registry_domain_name, stage_device, layer_idx, config_.moe.num_experts,
                                                         expert_params.prepared_gate_gemm,
                                                         expert_params.prepared_up_gemm,
                                                         expert_params.prepared_down_gemm)
                                                   : registry.populateExpertEngines(stage_device, layer_idx, config_.moe.num_experts,
                                                                                    expert_params.prepared_gate_gemm,
                                                                                    expert_params.prepared_up_gemm,
                                                                                    expert_params.prepared_down_gemm);

                        if (expert_params.expert_mask.empty())
                        {
                            if (!complete_layer || !populated)
                                failMissingGpuExpertGemmEngines(
                                    stage_device, layer_idx,
                                    withPlacementContext("incomplete ExpertGemmRegistry entry (" +
                                                         describeMissingExpertGemmEngine(config_.moe.num_experts,
                                                                                         expert_params.expert_mask,
                                                                                         expert_params.prepared_gate_gemm,
                                                                                         expert_params.prepared_up_gemm,
                                                                                         expert_params.prepared_down_gemm) +
                                                         ")"));
                        }
                        else if (has_active_masked_expert)
                        {
                            for (int expert = 0; expert < config_.moe.num_experts; ++expert)
                            {
                                if (!expert_params.expert_mask[expert])
                                    continue;
                                if (expert_params.prepared_gate_gemm[expert] == nullptr ||
                                    expert_params.prepared_up_gemm[expert] == nullptr ||
                                    expert_params.prepared_down_gemm[expert] == nullptr)
                                    failMissingGpuExpertGemmEngines(
                                        stage_device, layer_idx,
                                        withPlacementContext("missing active masked expert engine (" +
                                                             describeMissingExpertGemmEngine(config_.moe.num_experts,
                                                                                             expert_params.expert_mask,
                                                                                             expert_params.prepared_gate_gemm,
                                                                                             expert_params.prepared_up_gemm,
                                                                                             expert_params.prepared_down_gemm) +
                                                             ")"));
                            }
                        }

                        LOG_DEBUG("[Qwen35MoEGraph] Layer " << layer_idx
                                                            << ": populated expert GEMM engines from registry"
                                                            << (domain_scoped ? " domain=" + registry_domain_name : std::string())
                                                            << " complete_layer=" << complete_layer
                                                            << " active_masked_expert=" << has_active_masked_expert);
                    }
                }
                else
                {
                    if (!MoEExpertComputeStage::prepareExpertGemmEngines(expert_params))
                        return false;
                }

                return true;
            };

            const bool overlay_requested = overlay_plan && overlay_plan->isTieredOverlay();
            const bool use_expert_overlay = false;

            if (overlay_requested)
            {
                LOG_WARN("[Qwen35MoEGraph] Expert overlay requested for layer " << layer_idx
                                                                                << " but sidecar overlay stages were removed; using graph-native routed expert path");
            }

            if (use_expert_overlay)
            {
#if 0
                auto dispatch_output_lifetime = std::make_shared<MoEExpertDispatchOutput>();

                MoEExpertDispatchStage::Params dispatch_params;
                dispatch_params.device_id = DeviceId::cpu();
                dispatch_params.routing_indices = routing_indices;
                dispatch_params.routing_weights = routing_weights;
                dispatch_params.hidden = buffers.normalized;
                dispatch_params.seq_len = total_tokens;
                dispatch_params.top_k = config_.moe.top_k;
                dispatch_params.d_model = config_.d_model;
                dispatch_params.continuation_domain = overlay_runtime_plan
                                                          ? overlay_runtime_plan->continuationDomain().name
                                                          : overlay_plan->continuation_domain;
                dispatch_params.transfer_mode = debugEnv().moe_expert_overlay.dense_transfer
                                                    ? MoEExpertTransferMode::DenseFullSequence
                                                    : MoEExpertTransferMode::Auto;
                dispatch_params.placement = *overlay_placement;
                dispatch_params.routed_tiers = overlay_plan->routed_tiers;
                dispatch_params.output_lifetime = dispatch_output_lifetime;

                const std::string dispatch_name = prefix + "moe_expert_dispatch";
                graph.addNode(dispatch_name,
                              ComputeStageFactory::createMoEExpertDispatch(dispatch_params),
                              DeviceId::cpu());
                graph.addDependency(dispatch_name, prefix + "moe_routing");

                std::vector<std::string> tier_node_names;
                overlay_reduce_pending = true;
                overlay_continuation_domain = dispatch_params.continuation_domain;
                overlay_continuation_device = device;
                overlay_reduce_diagnostics = std::make_shared<MoEExpertParallelReduceDiagnostics>();
                overlay_reduce_partials.reserve(overlay_plan->routed_tiers.size() + 1);
                overlay_reduce_partial_lifetimes.reserve(overlay_plan->routed_tiers.size());
                overlay_reduce_partial_infos.reserve(overlay_plan->routed_tiers.size() + 1);
                overlay_reduce_runtime_results.reserve(overlay_plan->routed_tiers.size() + 1);
                overlay_reduce_sparse_scratch.reserve(overlay_plan->routed_tiers.size());
                overlay_reduce_sparse_scratch_lifetimes.reserve(overlay_plan->routed_tiers.size());
                tier_node_names.reserve(overlay_plan->routed_tiers.size());

                for (size_t tier_index = 0; tier_index < overlay_plan->routed_tiers.size(); ++tier_index)
                {
                    const auto &tier = overlay_plan->routed_tiers[tier_index];
                    auto tier_mask = expertMaskForTier(*overlay_placement,
                                                       config_.moe.num_experts,
                                                       static_cast<int>(tier_index));
                    if (!hasActiveExpertMask(tier_mask))
                    {
                        LOG_DEBUG("[Qwen35MoEGraph] Skipping inactive MoE expert overlay tier "
                                  << tier.name << " for layer " << layer_idx);
                        continue;
                    }

                    const MoEOverlayRuntimeDomain *runtime_domain = overlay_runtime_plan
                                                                        ? &overlay_runtime_plan->domainForTier(tier_index)
                                                                        : nullptr;
                    const ExpertComputeDomain *source_domain = overlay_runtime_plan
                                                                   ? sourceDomainForName(overlay_runtime_plan->sourcePlan(), tier.domain)
                                                                   : nullptr;

                    const bool use_cpu_node_local_fallback = runtime_domain && source_domain &&
                                                             isCpuNodeLocalDomain(*runtime_domain);
                    const bool use_accelerator_local_tp = runtime_domain &&
                                                          isAcceleratorLocalTPTensorParallelDomain(*runtime_domain);
                    ILocalTPContext *local_tp_context = nullptr;
                    if (use_accelerator_local_tp)
                    {
                        local_tp_context = requireLocalTPContextForTier(
                            config_, tier, *runtime_domain, tier_index, layer_idx);
                    }
                    if (runtime_domain && runtime_domain->participants.size() > 1 &&
                        !use_cpu_node_local_fallback && !use_accelerator_local_tp)
                    {
                        failUnsupportedMultiParticipantTier(tier, *runtime_domain, tier_index, layer_idx);
                    }

                    auto partial = std::make_shared<FP32Tensor>(
                        std::vector<size_t>{static_cast<size_t>(total_tokens), static_cast<size_t>(config_.d_model)});
                    TensorBase *partial_tensor = partial.get();
                    overlay_reduce_partials.push_back(partial_tensor);
                    overlay_reduce_partial_lifetimes.push_back(std::move(partial));
                    auto runtime_result = std::make_shared<MoEOverlayDomainWorkResult>();
                    overlay_reduce_runtime_results.push_back(runtime_result);

                    if (overlay_continuation_device.is_gpu())
                    {
                        auto scratch = std::make_shared<FP32Tensor>(
                            std::vector<size_t>{static_cast<size_t>(total_tokens), static_cast<size_t>(config_.d_model)});
                        overlay_reduce_sparse_scratch.push_back(scratch.get());
                        overlay_reduce_sparse_scratch_lifetimes.push_back(std::move(scratch));
                    }

                    const std::string tier_node = prefix + "moe_expert_ffn_" +
                                                  nodeSuffixForTier(tier, static_cast<int>(tier_index));
                    tier_node_names.push_back(tier_node);

                    auto makeRuntimeRequest = [&](DeviceId output_device)
                    {
                        MoEOverlayDomainWorkRequest request;
                        request.layer_idx = layer_idx;
                        request.tier_index = static_cast<int>(tier_index);
                        request.tier_name = tier.name.empty() ? tier_node : tier.name;
                        request.continuation_domain = overlay_continuation_domain;
                        if (source_domain)
                            request.source_domain = *source_domain;
                        if (runtime_domain)
                            request.runtime_domain = *runtime_domain;
                        request.runtime_plan = overlay_runtime_plan;
                        request.execution_plan = overlay_execution_plan;
                        request.dispatch_output = dispatch_output_lifetime.get();
                        request.dispatch_output_lifetime = dispatch_output_lifetime;
                        request.dispatch_tier_index = static_cast<int>(tier_index);
                        if (runtime_domain)
                        {
                            request.dispatch_group = dispatchGroupForTier(
                                *runtime_domain,
                                layer_idx,
                                static_cast<int>(tier_index),
                                output_device);
                        }
                        request.output = partial_tensor;
                        request.output_device = output_device;
                        return request;
                    };

                    auto addRuntimeNode = [&](MoEOverlayDomainWorkRequest request, DeviceId stage_device)
                    {
                        DisabledDomainRunnerStage::Params runtime_params;
                        runtime_params.device_id = stage_device;
                        runtime_params.mpi_ctx = config_.moe.overlay_mpi_ctx
                                                     ? config_.moe.overlay_mpi_ctx.get()
                                                     : mpi_ctx_.get();
                        runtime_params.runtime = overlay_domain_runtime;
                        runtime_params.request = std::move(request);
                        runtime_params.result_lifetime = runtime_result;
                        runtime_params.result = runtime_result.get();

                        graph.addNode(tier_node,
                                      ComputeStageFactory::createDisabledDomainRunner(runtime_params),
                                      stage_device);
                    };

                    if (use_cpu_node_local_fallback)
                    {
                        DisabledCpuParticipantStage::Params fallback_params;
                        fallback_params.device_id = DeviceId::cpu();
                        fallback_params.mpi_ctx = config_.moe.overlay_mpi_ctx
                                                      ? config_.moe.overlay_mpi_ctx.get()
                                                      : mpi_ctx_.get();
                        fallback_params.domain = *source_domain;
                        fallback_params.root_world_rank = continuationRootWorldRank(*overlay_runtime_plan);
                        fallback_params.domain_id = DisabledCpuParticipant::stableDomainId(source_domain->name);
                        fallback_params.input = buffers.normalized;
                        fallback_params.seq_len = total_tokens;
                        fallback_params.d_model = config_.d_model;
                        fallback_params.num_experts = config_.moe.num_experts;
                        fallback_params.top_k = config_.moe.top_k;
                        fallback_params.gate_exps = layer.moe_gate_exps;
                        fallback_params.up_exps = layer.moe_up_exps;
                        fallback_params.down_exps = layer.moe_down_exps;
                        fallback_params.expert_intermediate = expert_intermediate;
                        fallback_params.layer_idx = layer_idx;
                        fallback_params.routing_indices = routing_indices;
                        fallback_params.routing_weights = routing_weights;
                        fallback_params.output = partial_tensor;
                        fallback_params.expert_mask = std::move(tier_mask);
                        fallback_params.transfer_mode = dispatch_params.transfer_mode;
                        fallback_params.dispatch_output_lifetime = dispatch_output_lifetime;
                        fallback_params.dispatch_tier_index = static_cast<int>(tier_index);
                        fallback_params.prepared_store = prepared_weight_store_;

                        auto request = makeRuntimeRequest(DeviceId::cpu());
                        request.has_cpu_fallback_params = true;
                        request.cpu_fallback_params = std::move(fallback_params);
                        addRuntimeNode(std::move(request), DeviceId::cpu());

                        overlay_reduce_partial_infos.push_back(MoEExpertParallelReducePartialInfo{
                            .name = tier.name.empty() ? tier_node : tier.name,
                            .source_domain = tier.domain,
                            .source_device = DeviceId::cpu(),
                        });
                    }
                    else if (use_accelerator_local_tp)
                    {
                        DisabledLocalTPParticipantStage::Params local_tp_params;
                        local_tp_params.device_id = runtime_domain->primary_device;
                        local_tp_params.domain = *runtime_domain;
                        local_tp_params.local_tp_context = local_tp_context;
                        local_tp_params.input = buffers.normalized;
                        local_tp_params.seq_len = total_tokens;
                        local_tp_params.d_model = config_.d_model;
                        local_tp_params.num_experts = config_.moe.num_experts;
                        local_tp_params.top_k = config_.moe.top_k;
                        local_tp_params.gate_exps = layer.moe_gate_exps;
                        local_tp_params.up_exps = layer.moe_up_exps;
                        local_tp_params.down_exps = layer.moe_down_exps;
                        local_tp_params.expert_intermediate = expert_intermediate;
                        local_tp_params.layer_idx = layer_idx;
                        local_tp_params.routing_indices = routing_indices;
                        local_tp_params.routing_weights = routing_weights;
                        local_tp_params.output = partial_tensor;
                        local_tp_params.expert_mask = std::move(tier_mask);
                        local_tp_params.stage_name_prefix = tier_node;
                        local_tp_params.upload_partials_to_participant_devices =
                            local_tp_context->backend() != CollectiveBackendType::HOST;
                        local_tp_params.dispatch_output_lifetime = dispatch_output_lifetime;
                        local_tp_params.dispatch_tier_index = static_cast<int>(tier_index);
                        local_tp_params.diagnostics_lifetime = std::make_shared<DisabledLocalTPDiagnostics>();

                        if (local_tp_context->backend() != CollectiveBackendType::HOST)
                        {
                            auto weight_mgr = model_ctx_ ? model_ctx_->concreteWeightManager() : nullptr;
                            if (!weight_mgr)
                            {
                                failMissingGpuExpertGemmEngines(
                                    runtime_domain->primary_device,
                                    layer_idx,
                                    "Bridge Phase 5E LocalTP prepared execution requires a concrete WeightManager for tier=" +
                                        std::to_string(tier_index) + " name=" + tier.name +
                                        " domain=" + tier.domain);
                            }

                            const auto &registry = weight_mgr->expertGemmRegistry();
                            local_tp_params.prepared_participants.reserve(runtime_domain->participants.size());
                            for (size_t participant_index = 0; participant_index < runtime_domain->participants.size(); ++participant_index)
                            {
                                const DeviceId participant_device = runtime_domain->participants[participant_index].local_device;
                                DisabledLocalTPPreparedParticipant prepared;
                                prepared.participant_index = static_cast<int>(participant_index);
                                prepared.device = participant_device;

                                (void)registry.populateExpertEnginesForDomain(
                                    tier.domain,
                                    participant_device,
                                    layer_idx,
                                    config_.moe.num_experts,
                                    prepared.gate_gemm,
                                    prepared.up_gemm,
                                    prepared.down_gemm);

                                for (int expert = 0; expert < config_.moe.num_experts; ++expert)
                                {
                                    if (!local_tp_params.expert_mask[static_cast<size_t>(expert)])
                                        continue;
                                    if (prepared.gate_gemm[static_cast<size_t>(expert)] == nullptr ||
                                        prepared.up_gemm[static_cast<size_t>(expert)] == nullptr ||
                                        prepared.down_gemm[static_cast<size_t>(expert)] == nullptr)
                                    {
                                        std::ostringstream placement_context;
                                        placement_context << "Bridge Phase 5E LocalTP prepared execution missing active ExpertGemmRegistry engine"
                                                          << " tier=" << tier_index
                                                          << " name=" << tier.name
                                                          << " domain=" << tier.domain
                                                          << " participant=" << participant_index
                                                          << " device=" << participant_device.to_string()
                                                          << " expert=" << expert << " ("
                                                          << describeMissingExpertGemmEngine(config_.moe.num_experts,
                                                                                             local_tp_params.expert_mask,
                                                                                             prepared.gate_gemm,
                                                                                             prepared.up_gemm,
                                                                                             prepared.down_gemm)
                                                          << ")";
                                        failMissingGpuExpertGemmEngines(participant_device,
                                                                        layer_idx,
                                                                        placement_context.str());
                                    }
                                }

                                local_tp_params.prepared_participants.push_back(std::move(prepared));
                            }
                        }

                        auto request = makeRuntimeRequest(runtime_domain->primary_device);
                        request.has_local_tp_params = true;
                        request.local_tp_params = std::move(local_tp_params);
                        addRuntimeNode(std::move(request), runtime_domain->primary_device);

                        overlay_reduce_partial_infos.push_back(MoEExpertParallelReducePartialInfo{
                            .name = tier.name.empty() ? tier_node : tier.name,
                            .source_domain = tier.domain,
                            .source_device = runtime_domain->primary_device,
                        });
                    }
                    else
                    {
                        const DeviceId tier_device = overlay_runtime_plan
                                                         ? overlay_runtime_plan->tierDeviceForMVP(tier_index, layer_idx)
                                                         : device;
                        auto expert_params = makeExpertParams(partial_tensor,
                                                              BufferId::MOE_EXPERT_OUTPUT,
                                                              std::move(tier_mask),
                                                              tier_device);
                        expert_params.output_registered_in_arena = false;

                        std::ostringstream placement_context;
                        placement_context << "tier=" << tier_index
                                          << " name=" << tier.name
                                          << " domain=" << tier.domain
                                          << " participant="
                                          << (runtime_domain ? participantContextForDevice(*runtime_domain, tier_device) : "unknown")
                                          << " device=" << tier_device.to_string();
                        if (overlay_plan)
                            placement_context << " residency=" << static_cast<int>(overlay_plan->residency_policy);

                        if (!prepareExpertParams(expert_params, tier_device, placement_context.str(), tier.domain))
                        {
                            LOG_ERROR("[Qwen35MoEGraph] Failed to prepare expert overlay tier "
                                      << tier.name << " for layer " << layer_idx
                                      << " on " << tier_device.to_string());
                        }

                        auto request = makeRuntimeRequest(tier_device);
                        request.has_compute_params = true;
                        request.compute_params = std::move(expert_params);
                        addRuntimeNode(std::move(request), tier_device);

                        overlay_reduce_partial_infos.push_back(MoEExpertParallelReducePartialInfo{
                            .name = tier.name.empty() ? tier_node : tier.name,
                            .source_domain = tier.domain,
                            .source_device = tier_device,
                        });
                    }
                    graph.addDependency(tier_node, dispatch_name);
                }
                overlay_reduce_dependencies = std::move(tier_node_names);
#endif
            }
            else
            {
                auto expert_params = makeExpertParams(moe_output,
                                                      BufferId::MOE_COMBINED_OUTPUT,
                                                      {},
                                                      device);
                (void)prepareExpertParams(expert_params, device);

                graph.addNode(prefix + "moe_expert_ffn",
                              ComputeStageFactory::createMoEExpertCompute(expert_params),
                              device);
                graph.addDependency(prefix + "moe_expert_ffn", prefix + "moe_routing");
                ffn_terminal = prefix + "moe_expert_ffn";

                // Qwen35 MoE expert weights are normally replicated, so every rank
                // computes the full routed-expert contribution. Only allreduce this
                // path when an explicit expert range/mask makes the output partial.
                const bool routed_expert_output_is_partial =
                    expert_params.local_expert_count >= 0 || !expert_params.expert_mask.empty();
                if (routed_expert_output_is_partial && needsTPAllreduce())
                {
                    size_t allreduce_count = static_cast<size_t>(total_tokens) * static_cast<size_t>(config_.d_model);
                    std::string ar_name = prefix + "moe_expert_allreduce";
                    auto allreduce_stage = createTPAllreduceStage(
                        moe_output, allreduce_count, device, layer_idx,
                        /*is_attention=*/false, ar_name, BufferId::MOE_COMBINED_OUTPUT);
                    if (allreduce_stage)
                    {
                        graph.addNode(ar_name, std::move(allreduce_stage), device);
                        graph.addDependency(ar_name, prefix + "moe_expert_ffn");
                        ffn_terminal = ar_name;
                    }
                }
            }
        }

        // =====================================================================
        // Stage 4: Shared Expert FFN (always-active dense SwiGLU)
        // =====================================================================
        TensorBase *shared_output = buffers.get(BufferId::MOE_SHARED_EXPERT_OUTPUT);
        std::string shared_ffn_last; // Track last shared expert stage (empty if no shared expert)
        bool shared_expert_reduced_into_moe_output = false;

        if (layer.shared_expert_gate && layer.shared_expert_up && layer.shared_expert_down && shared_output)
        {
            DeviceId shared_device = device;
            if (overlay_runtime_plan)
            {
                const auto &continuation_domain = overlay_runtime_plan->continuationDomain();
                const auto &shared_domain = overlay_runtime_plan->sharedExpertDomain();
                shared_device = overlay_runtime_plan->sharedExpertDeviceForMVP(layer_idx);
                if (domainContainsDevice(shared_domain, device))
                    shared_device = device;

                LOG_DEBUG("[Qwen35MoEGraph] Layer " << layer_idx
                                                    << " shared expert uses domain " << shared_domain.name
                                                    << " on " << shared_device.to_string()
                                                    << "; final output returns to continuation domain "
                                                    << continuation_domain.name << " on " << device.to_string()
                                                    << " via MoE combine");
            }

            // Always use the actual weight dimensions (already sharded for TP).
            // config_.moe.shared_intermediate_size is the FULL size from metadata,
            // but when TP is active, gate/up weights have rows() == intermediate/tp_degree.
            int shared_intermediate = static_cast<int>(layer.shared_expert_gate->rows());

            SharedExpertFFNStage::Params shared_params;
            shared_params.device_id = shared_device;
            shared_params.input = buffers.normalized;
            shared_params.gate_w = layer.shared_expert_gate;
            shared_params.up_w = layer.shared_expert_up;
            shared_params.down_w = layer.shared_expert_down;
            shared_params.output = shared_output;
            shared_params.seq_len = total_tokens;
            shared_params.d_model = config_.d_model;
            shared_params.intermediate = shared_intermediate;
            shared_params.input_buffer_id = BufferId::NORMALIZED;
            shared_params.output_buffer_id = BufferId::MOE_SHARED_EXPERT_OUTPUT;
            shared_params.prepared_ref_gate = preparedRefForGraphWeight(
                layer_bindings.shared_expert_gate, shared_device);
            shared_params.prepared_ref_up = preparedRefForGraphWeight(
                layer_bindings.shared_expert_up, shared_device);
            shared_params.prepared_ref_down = preparedRefForGraphWeight(
                layer_bindings.shared_expert_down, shared_device);
            shared_params.prepared_store = prepared_weight_store_;

            graph.addNode(prefix + "shared_expert_ffn",
                          ComputeStageFactory::createSharedExpertFFN(shared_params),
                          shared_device);
            graph.addDependency(prefix + "shared_expert_ffn", prefix + "ffn_norm");
            shared_ffn_last = prefix + "shared_expert_ffn";

            // Allreduce after shared expert down projection (InputParallel sharding)
            if (needsTPAllreduce())
            {
                size_t allreduce_count = static_cast<size_t>(total_tokens) * static_cast<size_t>(config_.d_model);
                std::string ar_name = prefix + "shared_expert_allreduce";
                auto allreduce_stage = createTPAllreduceStage(
                    shared_output, allreduce_count, shared_device, layer_idx,
                    /*is_attention=*/false, ar_name, BufferId::MOE_SHARED_EXPERT_OUTPUT);
                if (allreduce_stage)
                {
                    graph.addNode(ar_name, std::move(allreduce_stage), shared_device);
                    graph.addDependency(ar_name, prefix + "shared_expert_ffn");
                    shared_ffn_last = ar_name;
                }
            }

            // Stage 4b: Sigmoid gate on shared expert output
            if (layer.shared_expert_gate_inp)
            {
                SharedExpertGateStage::Params gate_params;
                gate_params.device_id = shared_device;
                gate_params.input = buffers.normalized;
                gate_params.gate_inp = layer.shared_expert_gate_inp;
                gate_params.shared_output = shared_output;
                gate_params.seq_len = total_tokens;
                gate_params.d_model = config_.d_model;
                gate_params.input_buffer_id = BufferId::NORMALIZED;
                gate_params.output_buffer_id = BufferId::MOE_SHARED_EXPERT_OUTPUT;

                graph.addNode(prefix + "shared_expert_gate",
                              ComputeStageFactory::createSharedExpertGate(gate_params),
                              shared_device);
                graph.addDependency(prefix + "shared_expert_gate", shared_ffn_last);
                shared_ffn_last = prefix + "shared_expert_gate";
            }

            if (overlay_reduce_pending && overlay_runtime_plan &&
                overlay_runtime_plan->sharedExpertDomain().name != overlay_continuation_domain)
            {
                const auto &shared_domain = overlay_runtime_plan->sharedExpertDomain();
                overlay_reduce_partials.push_back(shared_output);
                overlay_reduce_partial_infos.push_back(MoEExpertParallelReducePartialInfo{
                    .name = "shared_expert",
                    .source_domain = shared_domain.name,
                    .source_device = shared_device,
                });
                overlay_reduce_dependencies.push_back(shared_ffn_last);
                shared_expert_reduced_into_moe_output = true;

                LOG_DEBUG("[Qwen35MoEGraph] Layer " << layer_idx
                                                    << " shared expert domain '" << shared_domain.name
                                                    << "' contributes through cross-domain reduce to continuation domain '"
                                                    << overlay_continuation_domain << "'");
            }
        }

        if (overlay_reduce_pending)
        {
            MoEExpertParallelReduceStage::Params reduce_params;
            reduce_params.device_id = overlay_continuation_device;
            reduce_params.partials = std::move(overlay_reduce_partials);
            reduce_params.partial_lifetimes = std::move(overlay_reduce_partial_lifetimes);
            reduce_params.partial_infos = std::move(overlay_reduce_partial_infos);
            reduce_params.sparse_expansion_scratch = std::move(overlay_reduce_sparse_scratch);
            reduce_params.sparse_expansion_scratch_lifetimes = std::move(overlay_reduce_sparse_scratch_lifetimes);
            reduce_params.output = moe_output;
            reduce_params.output_buffer_id = BufferId::MOE_COMBINED_OUTPUT;
            reduce_params.rows = static_cast<size_t>(total_tokens);
            reduce_params.cols = static_cast<size_t>(config_.d_model);
            reduce_params.layer_idx = layer_idx;
            reduce_params.mode = overlay_continuation_device.is_gpu()
                                     ? MoEExpertParallelReduceMode::ContinuationDeviceOptimized
                                     : MoEExpertParallelReduceMode::HostStagedCorrectness;
            reduce_params.continuation_domain = overlay_continuation_domain;
            reduce_params.continuation_device = overlay_continuation_device;
            reduce_params.diagnostics_lifetime = overlay_reduce_diagnostics;

            const std::string reduce_name = prefix + "moe_expert_parallel_reduce";
            graph.addNode(reduce_name,
                          ComputeStageFactory::createMoEExpertParallelReduce(reduce_params),
                          overlay_continuation_device);
            for (const auto &dependency : overlay_reduce_dependencies)
                graph.addDependency(reduce_name, dependency);
            ffn_terminal = reduce_name;
        }

        // =====================================================================
        // Stage 5: Combine expert output + shared expert output → attn_proj
        // =====================================================================
        // The combined MoE output goes to attn_proj so that the next layer's
        // FusedResidualNormStage handles the residual add automatically.
        {
            if (!shared_expert_reduced_into_moe_output && !shared_ffn_last.empty() && shared_output)
            {
                // Add expert_output + shared_expert_output → attn_proj
                ResidualAddStage::Params add_params;
                add_params.device_id = device;
                add_params.input = shared_output;
                add_params.residual = moe_output;
                add_params.output = buffers.attn_proj;
                add_params.num_elements = static_cast<size_t>(total_tokens) * static_cast<size_t>(config_.d_model);
                add_params.input_buffer_id = BufferId::MOE_SHARED_EXPERT_OUTPUT;
                add_params.residual_buffer_id = BufferId::MOE_COMBINED_OUTPUT;
                add_params.output_buffer_id = BufferId::ATTN_PROJ;

                graph.addNode(prefix + "moe_combine",
                              ComputeStageFactory::createResidualAdd(add_params),
                              device);
                graph.addDependency(prefix + "moe_combine", ffn_terminal);
                graph.addDependency(prefix + "moe_combine", shared_ffn_last);
                ffn_terminal = prefix + "moe_combine";
            }
            else
            {
                // No shared expert — copy expert output to attn_proj directly
                // (or if MOE_COMBINED_OUTPUT IS attn_proj, this is a no-op)
                if (moe_output != buffers.attn_proj)
                {
                    ResidualAddStage::Params copy_params;
                    copy_params.device_id = device;
                    copy_params.input = moe_output;
                    copy_params.residual = nullptr;
                    copy_params.output = buffers.attn_proj;
                    copy_params.num_elements = static_cast<size_t>(total_tokens) * static_cast<size_t>(config_.d_model);
                    copy_params.input_buffer_id = BufferId::MOE_COMBINED_OUTPUT;
                    copy_params.output_buffer_id = BufferId::ATTN_PROJ;

                    graph.addNode(prefix + "moe_combine",
                                  ComputeStageFactory::createResidualAdd(copy_params),
                                  device);
                    graph.addDependency(prefix + "moe_combine", ffn_terminal);
                    ffn_terminal = prefix + "moe_combine";
                }
            }
        }

        // =====================================================================
        // Stage 6: Explicit residual (last layer only)
        // =====================================================================
        const bool skip_ffn_residual = (layer_idx < config_.pp_layer_offset + config_.n_layers - 1);
        if (!skip_ffn_residual)
        {
            ResidualAddStage::Params res_params;
            res_params.device_id = device;
            res_params.input = buffers.attn_proj;
            res_params.residual = buffers.current_hidden;
            res_params.output = buffers.current_hidden;
            res_params.num_elements = static_cast<size_t>(total_tokens) * static_cast<size_t>(config_.d_model);
            res_params.input_buffer_id = BufferId::ATTN_PROJ;
            res_params.residual_buffer_id = BufferId::HIDDEN_STATE;
            res_params.output_buffer_id = BufferId::HIDDEN_STATE;

            graph.addNode(prefix + "ffn_residual",
                          ComputeStageFactory::createResidualAdd(res_params),
                          device);
            graph.addDependency(prefix + "ffn_residual", ffn_terminal);
            ffn_terminal = prefix + "ffn_residual";
        }

        graph.setTerminalNode(ffn_terminal);
        return graph;
    }

} // namespace llaminar2
