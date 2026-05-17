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
#include "../../execution/moe/MoEExpertOwnerMap.h"
#include "../../execution/moe/MoEExpertParallelPlan.h"
#include "../../execution/moe/MoEExpertOverlayExecutionPlan.h"
#include "../../execution/moe/MoEExpertOverlayRuntimePlan.h"
#include "../../memory/BufferId.h"
#include "../../execution/local_execution/graph/GraphResolver.h"
#include "../../tensors/Tensors.h"
#include "../../utils/DebugEnv.h"

#include <algorithm>
#include <cctype>
#include <functional>
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

        int stableMoEOverlayDomainId(const std::string &domain_name)
        {
            const size_t hashed = std::hash<std::string>{}(domain_name);
            return static_cast<int>(hashed & 0x3fffffffU);
        }

        MoEOverlayCollectiveKey makeSparseCollectiveKey(
            int layer_idx,
            int tier_index,
            const std::string &domain_name,
            MoEOverlayCollectiveDirection direction,
            uint64_t sequence)
        {
            MoEOverlayCollectiveKey key;
            key.generation_id = 1;
            key.step_id = 0;
            key.layer_idx = layer_idx;
            key.tier_idx = tier_index;
            key.domain_id = stableMoEOverlayDomainId(domain_name);
            key.direction = direction;
            key.sequence = sequence;
            return key;
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

    IMoERuntimeTable *Qwen35MoEGraph::moeRuntimeTableForDevice(DeviceId device,
                                                               int prefill_token_capacity,
                                                               const std::string &key_suffix)
    {
#if !defined(HAVE_ROCM)
        (void)device;
        (void)prefill_token_capacity;
        (void)key_suffix;
        return nullptr;
#else
        if (!device.is_rocm() || config_.moe.num_experts <= 0 || config_.moe.top_k <= 0 || config_.n_layers <= 0)
            return nullptr;

        const std::string key = key_suffix.empty()
                                    ? device.to_string()
                                    : device.to_string() + "#" + key_suffix;
        auto it = moe_runtime_tables_.find(key);
        if (it != moe_runtime_tables_.end())
        {
            if (prefill_token_capacity > 0)
                it->second->ensurePrefillRouteScratchCapacity(prefill_token_capacity);
            return it->second.get();
        }

        DeviceMoERuntimeTable::Config table_config;
        table_config.device_id = device;
        table_config.num_layers = config_.n_layers;
        table_config.num_experts = config_.moe.num_experts;
        table_config.top_k = config_.moe.top_k;
        table_config.mirror_to_device = true;
        table_config.prefill_token_capacity = std::max(0, prefill_token_capacity);

        auto table = std::make_unique<MoERuntimeTable>(table_config);
        IMoERuntimeTable *ptr = table.get();
        moe_runtime_tables_.emplace(key, std::move(table));
        return ptr;
#endif
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

        IMoERuntimeTable *moe_runtime_table = nullptr;
        const auto &rocm_env = debugEnv().rocm;
        if (total_tokens == 1 && rocm_env.moe_grouped_decode && rocm_env.moe_device_routed_decode)
        {
            moe_runtime_table = moeRuntimeTableForDevice(device);
        }
        else if (total_tokens > 1 && rocm_env.moe_grouped_prefill)
        {
            moe_runtime_table = moeRuntimeTableForDevice(device, total_tokens);
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
            route_params.moe_runtime_table = moe_runtime_table;
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
                expert_params.moe_runtime_table = moe_runtime_table;
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
            const ExpertLayerPlacement *overlay_placement = overlay_requested
                                                                 ? findExpertOverlayPlacement(*overlay_plan, layer_idx)
                                                                 : nullptr;
            const bool use_graph_native_overlay = overlay_requested && overlay_placement &&
                                                  isUsableExpertOverlayPlacement(
                                                      *overlay_placement,
                                                      *overlay_plan,
                                                      config_.moe.num_experts,
                                                      layer_idx);

            if (overlay_requested && !use_graph_native_overlay)
            {
                LOG_WARN("[Qwen35MoEGraph] Expert overlay requested for layer " << layer_idx
                                                                                << " but no usable explicit placement was found; using single-stage routed expert path");
            }

            if (use_graph_native_overlay)
            {
                const auto owner_map = MoEExpertOwnerMap::build(*overlay_plan);
                auto dispatch_output_lifetime = std::make_shared<MoEExpertDispatchOutput>();
                auto collective_lifetime = std::make_shared<MoEOverlayLocalSparseCollectiveContext>(
                    MoEOverlayLocalSparseCollectiveContext::Config{
                        .participant_count = 1,
                        .slot_count = std::max<size_t>(4u, overlay_plan->routed_tiers.size() * 2u + 1u),
                    });
                auto workspace_lifetime = std::make_shared<MoEOverlayCollectiveWorkspace>();
                workspace_lifetime->ensureCapacity(
                    static_cast<size_t>(std::max(total_tokens, 1)),
                    static_cast<size_t>(std::max(total_tokens * config_.moe.top_k, 1)),
                    config_.d_model,
                    config_.moe.top_k,
                    DeviceId::cpu());
                workspace_lifetime->resetForStep(1, 0);

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
                dispatch_params.output = dispatch_output_lifetime.get();
                dispatch_params.output_lifetime = dispatch_output_lifetime;

                const std::string dispatch_name = prefix + "moe_expert_dispatch";
                graph.addNode(dispatch_name,
                              ComputeStageFactory::createMoEExpertDispatch(dispatch_params),
                              DeviceId::cpu());
                graph.addDependency(dispatch_name, prefix + "moe_routing");

                overlay_continuation_domain = dispatch_params.continuation_domain;
                overlay_continuation_device = device;

                std::string previous_return_node;
                bool cleared_sparse_output = false;

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


                    const auto participant_ids = owner_map.participantIdsForTier(static_cast<int>(tier_index));
                    const int target_participant = participant_ids.empty() ? 0 : participant_ids.front();
                    const auto *participant = owner_map.participantForId(target_participant);
                    DeviceId tier_device = participant && participant->device.is_valid()
                                               ? participant->device
                                               : device;
                    if (!tier_device.is_valid())
                        tier_device = DeviceId::cpu();

                    const std::string tier_suffix = nodeSuffixForTier(tier, static_cast<int>(tier_index));
                    const std::string sparse_dispatch_name = prefix + "moe_sparse_dispatch_" + tier_suffix;
                    const std::string local_expert_name = prefix + "moe_local_expert_" + tier_suffix;
                    const std::string sparse_return_name = prefix + "moe_sparse_return_reduce_" + tier_suffix;

                    auto inbound_rows_lifetime = std::make_shared<MoEOverlaySparseRows>(
                        workspace_lifetime->dispatchReceive(layer_idx, static_cast<int>(tier_index)));
                    MoESparseDispatchStage::Params sparse_dispatch_params;
                    sparse_dispatch_params.device_id = DeviceId::cpu();
                    sparse_dispatch_params.collective_context_lifetime = collective_lifetime;
                    sparse_dispatch_params.workspace_lifetime = workspace_lifetime;
                    sparse_dispatch_params.key = makeSparseCollectiveKey(
                        layer_idx,
                        static_cast<int>(tier_index),
                        tier.domain,
                        MoEOverlayCollectiveDirection::Dispatch,
                        100u + static_cast<uint64_t>(tier_index));
                    sparse_dispatch_params.source_participant = 0;
                    sparse_dispatch_params.target_participant = target_participant;
                    sparse_dispatch_params.hidden = buffers.normalized;
                    sparse_dispatch_params.routing_indices = routing_indices;
                    sparse_dispatch_params.routing_weights = routing_weights;
                    sparse_dispatch_params.seq_len = total_tokens;
                    sparse_dispatch_params.top_k = config_.moe.top_k;
                    sparse_dispatch_params.d_model = config_.d_model;
                    sparse_dispatch_params.dispatch_output_lifetime = dispatch_output_lifetime;
                    sparse_dispatch_params.tier_index = static_cast<int>(tier_index);
                    sparse_dispatch_params.inbound_rows_lifetime = inbound_rows_lifetime;

                    graph.addNode(sparse_dispatch_name,
                                  ComputeStageFactory::createMoESparseDispatch(sparse_dispatch_params),
                                  DeviceId::cpu());
                    graph.addDependency(sparse_dispatch_name, dispatch_name);

                    auto outbound_rows_lifetime = std::make_shared<MoEOverlayReturnRows>(
                        workspace_lifetime->localExpertOutput(layer_idx, static_cast<int>(tier_index)));
                    MoELocalExpertStage::Params local_params;
                    local_params.device_id = tier_device;
                    local_params.input_rows_lifetime = inbound_rows_lifetime;
                    local_params.output_rows_lifetime = outbound_rows_lifetime;
                    local_params.workspace_lifetime = workspace_lifetime;
                    local_params.gate_exps = layer.moe_gate_exps;
                    local_params.up_exps = layer.moe_up_exps;
                    local_params.down_exps = layer.moe_down_exps;
                    local_params.num_experts = config_.moe.num_experts;
                    local_params.top_k = config_.moe.top_k;
                    local_params.d_model = config_.d_model;
                    local_params.expert_intermediate = expert_intermediate;
                    local_params.layer_idx = layer_idx;
                    local_params.expert_mask = owner_map.expertMaskForParticipant(
                        layer_idx,
                        target_participant,
                        config_.moe.num_experts);
                    local_params.prepared_store = prepared_weight_store_;
                    if (model_ctx_)
                    {
                        auto weight_mgr = model_ctx_->concreteWeightManager();
                        if (weight_mgr)
                            local_params.expert_registry = &weight_mgr->expertGemmRegistry();
                    }
                    local_params.runtime_participant_index = target_participant;
                    if (tier_device.is_rocm() &&
                        (rocm_env.moe_grouped_decode || rocm_env.moe_grouped_prefill))
                    {
                        local_params.moe_runtime_table = moeRuntimeTableForDevice(
                            tier_device,
                            0,
                            "overlay_p" + std::to_string(target_participant));
                    }

                    graph.addNode(local_expert_name,
                                  ComputeStageFactory::createMoELocalExpert(local_params),
                                  tier_device);
                    graph.addDependency(local_expert_name, sparse_dispatch_name);

                    auto return_rows_lifetime = std::make_shared<MoEOverlayReturnRows>(
                        workspace_lifetime->returnReceive(layer_idx, static_cast<int>(tier_index)));
                    MoESparseReturnReduceStage::Params return_params;
                    return_params.device_id = DeviceId::cpu();
                    return_params.collective_context_lifetime = collective_lifetime;
                    return_params.workspace_lifetime = workspace_lifetime;
                    return_params.key = makeSparseCollectiveKey(
                        layer_idx,
                        static_cast<int>(tier_index),
                        tier.domain,
                        MoEOverlayCollectiveDirection::ReturnReduce,
                        200u + static_cast<uint64_t>(tier_index));
                    return_params.source_participant = 0;
                    return_params.target_participant = 0;
                    return_params.outbound_rows_lifetime = outbound_rows_lifetime;
                    return_params.inbound_rows_lifetime = return_rows_lifetime;
                    return_params.dense_output = moe_output;
                    return_params.seq_len = total_tokens;
                    return_params.d_model = config_.d_model;
                    return_params.clear_output_before_scatter = !cleared_sparse_output;
                    cleared_sparse_output = true;

                    graph.addNode(sparse_return_name,
                                  ComputeStageFactory::createMoESparseReturnReduce(return_params),
                                  DeviceId::cpu());
                    graph.addDependency(sparse_return_name, local_expert_name);
                    if (!previous_return_node.empty())
                        graph.addDependency(sparse_return_name, previous_return_node);
                    previous_return_node = sparse_return_name;
                }

                if (!previous_return_node.empty())
                    ffn_terminal = previous_return_node;
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
