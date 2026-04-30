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
#include "../../memory/BufferId.h"
#include "../../execution/local_execution/graph/GraphResolver.h"

namespace llaminar2
{

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

        {
            // Infer expert intermediate size from weight shape
            int expert_intermediate = config_.moe.intermediate_size;
            if (expert_intermediate == 0 && layer.moe_gate_exps)
            {
                // gate_exps shape: [num_experts, intermediate, d_model] or rows=num_experts*intermediate
                size_t total_rows = layer.moe_gate_exps->rows();
                expert_intermediate = static_cast<int>(total_rows / config_.moe.num_experts);
            }

            MoEExpertComputeStage::Params expert_params;
            expert_params.device_id = device;
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
            expert_params.output = moe_output;
            expert_params.output_buffer_id = BufferId::MOE_COMBINED_OUTPUT;
            expert_params.input_buffer_id = BufferId::NORMALIZED;

            // GPU scratch buffers
            expert_params.gate_scratch = buffers.get(BufferId::MOE_GATE_SCRATCH);
            expert_params.up_scratch = buffers.get(BufferId::MOE_UP_SCRATCH);

            // Extract per-expert 2D views from 3D packed tensors (required)
            if (!MoEExpertComputeStage::extractExpertViews(expert_params))
            {
                LOG_ERROR("[Qwen35MoEGraph] Failed to extract expert views for layer " << layer_idx);
            }

            // Pre-resolve CPU GEMM engines for all local experts.  On GPU, Qwen3.5
            // MoE 35B cannot keep every expert for every layer resident on a 32 GiB
            // device, so expert engines are prepared lazily for routed experts.
            if (device.is_gpu())
            {
                expert_params.prepared_gate_gemm.assign(config_.moe.num_experts, nullptr);
                expert_params.prepared_up_gemm.assign(config_.moe.num_experts, nullptr);
                expert_params.prepared_down_gemm.assign(config_.moe.num_experts, nullptr);
            }
            else
            {
                MoEExpertComputeStage::prepareExpertGemmEngines(expert_params);
            }

            graph.addNode(prefix + "moe_expert_ffn",
                          ComputeStageFactory::createMoEExpertCompute(expert_params),
                          device);
            graph.addDependency(prefix + "moe_expert_ffn", prefix + "moe_routing");
            ffn_terminal = prefix + "moe_expert_ffn";
        }

        // =====================================================================
        // Stage 4: Shared Expert FFN (always-active dense SwiGLU)
        // =====================================================================
        TensorBase *shared_output = buffers.get(BufferId::MOE_SHARED_EXPERT_OUTPUT);
        std::string shared_ffn_last; // Track last shared expert stage (empty if no shared expert)

        if (layer.shared_expert_gate && layer.shared_expert_up && layer.shared_expert_down && shared_output)
        {
            // Always use the actual weight dimensions (already sharded for TP).
            // config_.moe.shared_intermediate_size is the FULL size from metadata,
            // but when TP is active, gate/up weights have rows() == intermediate/tp_degree.
            int shared_intermediate = static_cast<int>(layer.shared_expert_gate->rows());

            SharedExpertFFNStage::Params shared_params;
            shared_params.device_id = device;
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

            graph.addNode(prefix + "shared_expert_ffn",
                          ComputeStageFactory::createSharedExpertFFN(shared_params),
                          device);
            graph.addDependency(prefix + "shared_expert_ffn", prefix + "ffn_norm");
            shared_ffn_last = prefix + "shared_expert_ffn";

            // Allreduce after shared expert down projection (InputParallel sharding)
            if (needsTPAllreduce())
            {
                size_t allreduce_count = static_cast<size_t>(total_tokens) * static_cast<size_t>(config_.d_model);
                std::string ar_name = prefix + "shared_expert_allreduce";
                auto allreduce_stage = createTPAllreduceStage(
                    shared_output, allreduce_count, device, layer_idx,
                    /*is_attention=*/false, ar_name, BufferId::MOE_SHARED_EXPERT_OUTPUT);
                if (allreduce_stage)
                {
                    graph.addNode(ar_name, std::move(allreduce_stage), device);
                    graph.addDependency(ar_name, prefix + "shared_expert_ffn");
                    shared_ffn_last = ar_name;
                }
            }

            // Stage 4b: Sigmoid gate on shared expert output
            if (layer.shared_expert_gate_inp)
            {
                SharedExpertGateStage::Params gate_params;
                gate_params.device_id = device;
                gate_params.input = buffers.normalized;
                gate_params.gate_inp = layer.shared_expert_gate_inp;
                gate_params.shared_output = shared_output;
                gate_params.seq_len = total_tokens;
                gate_params.d_model = config_.d_model;
                gate_params.input_buffer_id = BufferId::NORMALIZED;
                gate_params.output_buffer_id = BufferId::MOE_SHARED_EXPERT_OUTPUT;

                graph.addNode(prefix + "shared_expert_gate",
                              ComputeStageFactory::createSharedExpertGate(gate_params),
                              device);
                graph.addDependency(prefix + "shared_expert_gate", shared_ffn_last);
                shared_ffn_last = prefix + "shared_expert_gate";
            }
        }

        // =====================================================================
        // Stage 5: Combine expert output + shared expert output → attn_proj
        // =====================================================================
        // The combined MoE output goes to attn_proj so that the next layer's
        // FusedResidualNormStage handles the residual add automatically.
        {
            if (!shared_ffn_last.empty())
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
                graph.addDependency(prefix + "moe_combine", prefix + "moe_expert_ffn");
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
                    graph.addDependency(prefix + "moe_combine", prefix + "moe_expert_ffn");
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
