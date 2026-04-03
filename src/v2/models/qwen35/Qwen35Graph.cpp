/**
 * @file Qwen35Graph.cpp
 * @brief Qwen 3.5 compute graph builder implementation
 */

#include "Qwen35Graph.h"
#include "Qwen35Schema.h"
#include "../../execution/compute_stages/ComputeStages.h"
#include "../../execution/compute_stages/stages/GDNProjectionStage.h"
#include "../../execution/compute_stages/stages/ShortConv1dStage.h"
#include "../../execution/compute_stages/stages/GDNRecurrenceStage.h"
#include "../../execution/compute_stages/stages/GatedRMSNormStage.h"
#include "../../execution/compute_stages/stages/AttentionOutputGateStage.h"
#include "../../kernels/cpu/gdn/CPUShortConvolution.h"
#include "../../kernels/cpu/gdn/CPUGatedDeltaNet.h"
#include "../../tensors/TensorKernels.h"
#include "../../utils/Logger.h"

namespace llaminar2
{

    // =========================================================================
    // Constructors
    // =========================================================================

    Qwen35Graph::Qwen35Graph(
        std::shared_ptr<ModelContext> model_ctx,
        std::shared_ptr<MPIContext> mpi_ctx,
        const GraphConfig &config)
        : Qwen2Graph(std::move(model_ctx), std::move(mpi_ctx), config)
    {
        ensureGDNStates();
    }

    Qwen35Graph::Qwen35Graph(
        const GraphConfig &config,
        std::shared_ptr<MPIContext> mpi_ctx)
        : Qwen2Graph(config, std::move(mpi_ctx))
    {
        ensureGDNStates();
    }

    // =========================================================================
    // GDN State Management
    // =========================================================================

    void Qwen35Graph::ensureGDNStates()
    {
        if (!config_.hasGDN())
            return;

        const int n_layers = config_.n_layers;
        const int n_heads = config_.n_heads;
        const int conv_kernel = config_.gdn_conv_kernel_size;

        // Determine GDN key/value dimensions from the SSM inner_size
        // qkv_dim = inner_size, with Q and K each having d_k elements per head,
        // and V having d_v elements per head. For Qwen3.5:
        //   inner_size = 4096, n_heads = 16
        //   Total QKV has 2*n_heads*d_k + n_heads*d_v elements
        // The conv state covers the full QKV dim
        const int inner_size = config_.gdn_inner_size > 0 ? config_.gdn_inner_size : n_heads * config_.gdn_state_size;

        // d_k and d_v from the model: state_size is d_v (or d_k), inner_size = n_heads * (2*d_k + d_v)
        const int d_v = config_.gdn_state_size;
        const int d_k = (inner_size / n_heads - d_v) > 0
                            ? (inner_size / n_heads - d_v) / 2
                            : d_v;
        // Actually: qkv_dim = 2 * n_heads * d_k + n_heads * d_v
        // So d_k = (qkv_dim - n_heads * d_v) / (2 * n_heads)
        // For Qwen3.5-4B: inner=4096, n_heads=16, d_v=128
        // d_k = (4096 - 16*128) / (2*16) = (4096-2048)/32 = 64

        conv_states_.resize(n_layers);
        recurrence_states_.resize(n_layers);
        conv_kernels_.resize(n_layers);
        rec_kernels_.resize(n_layers);

        for (int i = 0; i < n_layers; ++i)
        {
            if (!isGDNLayer(i))
                continue;

            // Conv state: [inner_size, kernel_size - 1]
            const int conv_state_size = inner_size * (conv_kernel - 1);
            conv_states_[i].resize(conv_state_size, 0.0f);

            // Recurrence state: [n_heads, d_k, d_v]
            const int recurrence_state_size = n_heads * d_k * d_v;
            recurrence_states_[i].resize(recurrence_state_size, 0.0f);

            // Create kernel instances (lifetime tied to Qwen35Graph)
            conv_kernels_[i] = std::make_shared<CPUShortConvolution>();
            rec_kernels_[i] = std::make_shared<CPUGatedDeltaNet>();

            LOG_DEBUG("[Qwen35Graph] Layer " << i << " GDN state: conv_state="
                      << conv_state_size << " recurrence_state=" << recurrence_state_size);
        }
    }

    bool Qwen35Graph::isGDNLayer(int layer_idx) const
    {
        if (layer_idx < 0 || layer_idx >= static_cast<int>(config_.layer_types.size()))
            return false;
        return config_.layer_types[layer_idx] == "gdn";
    }

    // =========================================================================
    // Schema
    // =========================================================================

    GraphSchema Qwen35Graph::getSchema() const
    {
        Qwen35SchemaFactory factory;
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
    // Full Forward Graph — inherited from Qwen2Graph.
    // Qwen2Graph::buildFullForwardGraph() calls buildAttentionGraph()
    // virtually, which dispatches to GDN or FA via Qwen35Graph override.
    // No need to override here.
    // =========================================================================

    // =========================================================================
    // Layer Graph
    // =========================================================================

    ComputeGraph Qwen35Graph::buildLayerGraph(const LayerContext &ctx)
    {
        int max_layers = config_.total_n_layers > 0 ? config_.total_n_layers : config_.n_layers;
        if (ctx.layer_idx < 0 || ctx.layer_idx >= max_layers)
        {
            LOG_ERROR("[Qwen35Graph::buildLayerGraph] Invalid layer index: " << ctx.layer_idx);
            return ComputeGraph{};
        }

        if (!weights_.get_layer_weights)
        {
            LOG_ERROR("[Qwen35Graph::buildLayerGraph] Layer weight accessor not set");
            return ComputeGraph{};
        }

        LayerWeights layer_weights = weights_.get_layer_weights(ctx.layer_idx);

        ComputeGraph attn_graph = buildAttentionGraph(
            layer_weights, buffers_.layer_buffers, ctx.layer_idx, ctx.seq_len,
            ctx.batch_size, ctx.kv_cache, ctx.position_ids, ctx.device,
            ctx.sequence_lengths);

        ComputeGraph ffn_graph = buildFFNGraph(
            layer_weights, buffers_.layer_buffers, ctx.layer_idx, ctx.seq_len,
            ctx.batch_size, ctx.device);

        std::string attn_last = attn_graph.terminalNode();
        attn_graph.merge(std::move(ffn_graph), attn_last);

        return attn_graph;
    }

    // =========================================================================
    // Attention Graph Dispatch
    // =========================================================================

    ComputeGraph Qwen35Graph::buildAttentionGraph(
        const LayerWeights &layer,
        ActivationBuffers &buffers,
        int layer_idx,
        int seq_len,
        int batch_size,
        IKVCache *kv_cache,
        const int *position_ids,
        DeviceId device,
        const std::vector<int> *sequence_lengths)
    {
        if (isGDNLayer(layer_idx))
        {
            return buildGDNAttentionGraph(layer, buffers, layer_idx,
                                          seq_len, batch_size, device);
        }
        else
        {
            // Full attention layers — delegate to Qwen2Graph's implementation
            return Qwen2Graph::buildAttentionGraph(
                layer, buffers, layer_idx, seq_len, batch_size,
                kv_cache, position_ids, device, sequence_lengths);
        }
    }

    // =========================================================================
    // GDN Attention Sub-Graph
    // =========================================================================

    ComputeGraph Qwen35Graph::buildGDNAttentionGraph(
        const LayerWeights &layer,
        ActivationBuffers &buffers,
        int layer_idx,
        int seq_len,
        int batch_size,
        DeviceId device)
    {
        ComputeGraph graph;
        std::string prefix = "layer" + std::to_string(layer_idx) + "_";
        int total_tokens = batch_size * seq_len;

        const int n_heads = config_.n_heads;
        const int d_model = config_.d_model;
        const int inner_size = config_.gdn_inner_size > 0
                                   ? config_.gdn_inner_size
                                   : n_heads * config_.gdn_state_size;
        const int d_v = config_.gdn_state_size;
        const int d_k = (inner_size / n_heads - d_v) > 0
                            ? (inner_size / n_heads - d_v) / 2
                            : d_v;
        const int qkv_dim = inner_size; // 2 * n_heads * d_k + n_heads * d_v

        LOG_DEBUG("[Qwen35Graph] Building GDN attention for layer " << layer_idx
                  << ": total_tokens=" << total_tokens
                  << " n_heads=" << n_heads << " d_k=" << d_k << " d_v=" << d_v
                  << " qkv_dim=" << qkv_dim);

        // =====================================================================
        // Stage 1: Pre-attention RMSNorm
        // =====================================================================
        if (layer_idx > config_.pp_layer_offset)
        {
            FusedResidualNormStage::Params fused_params;
            fused_params.device_id = device;
            fused_params.input = buffers.attn_proj;
            fused_params.residual = buffers.current_hidden;
            fused_params.gamma = layer.attn_norm;
            fused_params.norm_output = buffers.normalized;
            fused_params.eps = config_.rms_norm_eps;
            fused_params.seq_len = total_tokens;
            fused_params.hidden_dim = d_model;
            fused_params.input_buffer_id = BufferId::ATTN_PROJ;
            fused_params.residual_buffer_id = BufferId::HIDDEN_STATE;
            fused_params.norm_output_buffer_id = BufferId::NORMALIZED;

            graph.addNode(prefix + "attn_norm",
                          ComputeStageFactory::createFusedResidualNorm(fused_params),
                          device);
        }
        else
        {
            RMSNormStage::Params norm_params;
            norm_params.input = buffers.current_hidden;
            norm_params.output = buffers.normalized;
            norm_params.gamma = layer.attn_norm;
            norm_params.eps = config_.rms_norm_eps;
            norm_params.seq_len = total_tokens;
            norm_params.device_id = device;
            norm_params.input_buffer_id = BufferId::HIDDEN_STATE;
            norm_params.output_buffer_id = BufferId::NORMALIZED;

            graph.addNode(prefix + "attn_norm",
                          ComputeStageFactory::createRMSNorm(norm_params),
                          device);
        }

        // =====================================================================
        // Stage 2: GDN 4-way Projection
        // =====================================================================
        GDNProjectionStage::Params proj_params;
        proj_params.device_id = device;
        proj_params.input = buffers.normalized;
        proj_params.m = total_tokens;
        proj_params.k = d_model;

        proj_params.w_qkv = layer.attn_qkv;
        proj_params.output_qkv = buffers.gdn_qkv;
        proj_params.n_qkv = qkv_dim;

        proj_params.w_z = layer.attn_gate; // Z projection = attn_gate.weight (in_proj_z in HF)
        proj_params.output_z = buffers.gdn_z;
        proj_params.n_z = n_heads * d_v;

        proj_params.w_a = layer.ssm_alpha;
        proj_params.output_a = buffers.gdn_alpha;
        proj_params.n_a = n_heads;

        proj_params.w_b = layer.ssm_beta;
        proj_params.output_b = buffers.gdn_beta;
        proj_params.n_b = n_heads;

        proj_params.input_buffer_id = BufferId::NORMALIZED;
        proj_params.output_qkv_buffer_id = BufferId::GDN_QKV;
        proj_params.output_z_buffer_id = BufferId::GDN_Z;
        proj_params.output_a_buffer_id = BufferId::GDN_ALPHA;
        proj_params.output_b_buffer_id = BufferId::GDN_BETA;

        graph.addNode(prefix + "gdn_proj",
                      std::make_unique<GDNProjectionStage>(proj_params),
                      device);
        graph.addDependency(prefix + "gdn_proj", prefix + "attn_norm");

        // =====================================================================
        // Stage 3: Short Conv1d + SiLU on QKV
        // =====================================================================
        ShortConv1dStage::Params conv_params;
        conv_params.device_id = device;
        conv_params.input = buffers.gdn_qkv;
        conv_params.output = buffers.gdn_qkv; // In-place (conv modifies QKV)
        conv_params.weight = layer.ssm_conv1d;
        conv_params.bias = nullptr; // Conv bias from ssm_dt.bias if available
        conv_params.conv_state = conv_states_[layer_idx].data();
        conv_params.seq_len = total_tokens;
        conv_params.channels = qkv_dim;
        conv_params.kernel_size = config_.gdn_conv_kernel_size;

        // Use stored kernel instance (lifetime tied to Qwen35Graph)
        conv_params.kernel = conv_kernels_[layer_idx].get();

        conv_params.input_buffer_id = BufferId::GDN_QKV;
        conv_params.output_buffer_id = BufferId::GDN_QKV;

        graph.addNode(prefix + "short_conv",
                      std::make_unique<ShortConv1dStage>(conv_params),
                      device);
        graph.addDependency(prefix + "short_conv", prefix + "gdn_proj");

        // =====================================================================
        // Stage 4: GDN Recurrence (delta rule linear attention)
        // =====================================================================
        // Q, K, V are interleaved in gdn_qkv after conv:
        // [seq_len, 2*n_heads*d_k + n_heads*d_v]
        // The recurrence stage splits Q, K, V internally
        GDNRecurrenceStage::Params rec_params;
        rec_params.device_id = device;
        rec_params.layer_idx = layer_idx;
        rec_params.Q = buffers.gdn_qkv;      // Will be split by kernel
        rec_params.K = buffers.gdn_qkv;      // Same tensor, offset by kernel
        rec_params.V = buffers.gdn_qkv;      // Same tensor, offset by kernel
        rec_params.alpha = buffers.gdn_alpha;
        rec_params.beta = buffers.gdn_beta;
        rec_params.A_log = layer.ssm_a;      // Learnable log-space gate
        rec_params.dt_bias = layer.ssm_dt_bias;
        rec_params.output = buffers.attn_output;
        rec_params.recurrence_state = recurrence_states_[layer_idx].data();
        rec_params.seq_len = total_tokens;
        rec_params.n_heads = n_heads;
        rec_params.d_k = d_k;
        rec_params.d_v = d_v;
        rec_params.chunk_size = 64;
        rec_params.use_qk_l2norm = true;
        rec_params.output_buffer_id = BufferId::ATTN_OUTPUT;

        // Use stored kernel instance (lifetime tied to Qwen35Graph)
        rec_params.kernel = rec_kernels_[layer_idx].get();

        graph.addNode(prefix + "gdn_recurrence",
                      std::make_unique<GDNRecurrenceStage>(rec_params),
                      device);
        graph.addDependency(prefix + "gdn_recurrence", prefix + "short_conv");

        // =====================================================================
        // Stage 5: Gated RMSNorm — RMSNorm(output) * SiLU(Z)
        // =====================================================================
        GatedRMSNormStage::Params gnorm_params;
        gnorm_params.device_id = device;
        gnorm_params.input = buffers.attn_output;
        gnorm_params.gate = buffers.gdn_z;
        gnorm_params.output = buffers.attn_output; // In-place
        gnorm_params.gamma = layer.ssm_norm;
        gnorm_params.eps = config_.rms_norm_eps;
        gnorm_params.subtract_one = config_.rms_norm_subtract_one;
        gnorm_params.seq_len = total_tokens;
        gnorm_params.input_buffer_id = BufferId::ATTN_OUTPUT;
        gnorm_params.gate_buffer_id = BufferId::GDN_Z;
        gnorm_params.output_buffer_id = BufferId::ATTN_OUTPUT;

        graph.addNode(prefix + "gated_norm",
                      std::make_unique<GatedRMSNormStage>(gnorm_params),
                      device);
        graph.addDependency(prefix + "gated_norm", prefix + "gdn_recurrence");

        // =====================================================================
        // Stage 6: Output Projection (Wo GEMM)
        // =====================================================================
        GEMMStage::Params wo_params;
        wo_params.device_id = device;
        wo_params.A = buffers.attn_output;
        wo_params.B = layer.ssm_out;
        wo_params.C = buffers.attn_proj;
        wo_params.m = total_tokens;
        if (layer.ssm_out)
        {
            wo_params.n = static_cast<int>(layer.ssm_out->shape()[0]);
            wo_params.k = static_cast<int>(layer.ssm_out->shape()[1]);
        }
        wo_params.alpha = 1.0f;
        wo_params.beta = 0.0f;
        wo_params.transpose_B = false;
        wo_params.a_buffer_id = BufferId::ATTN_OUTPUT;
        wo_params.c_buffer_id = BufferId::ATTN_PROJ;

        graph.addNode(prefix + "gdn_out_proj",
                      ComputeStageFactory::createGEMM(wo_params),
                      device);
        graph.addDependency(prefix + "gdn_out_proj", prefix + "gated_norm");

        // =====================================================================
        // Stage 7: Attention Output Gate — sigmoid(z) * output
        // For GDN layers, the gate is the Z projection output (already computed
        // in GDNProjectionStage). No separate GEMM needed — reuse gdn_z buffer.
        // =====================================================================
        if (layer.attn_gate)
        {
            AttentionOutputGateStage::Params gate_params;
            gate_params.device_id = device;
            gate_params.input = buffers.attn_proj;
            gate_params.gate = buffers.gdn_z;
            gate_params.output = buffers.attn_proj; // In-place
            gate_params.seq_len = total_tokens;
            gate_params.input_buffer_id = BufferId::ATTN_PROJ;
            gate_params.gate_buffer_id = BufferId::GDN_Z;
            gate_params.output_buffer_id = BufferId::ATTN_PROJ;

            graph.addNode(prefix + "attn_output_gate",
                          std::make_unique<AttentionOutputGateStage>(gate_params),
                          device);
            graph.addDependency(prefix + "attn_output_gate", prefix + "gdn_out_proj");
        }

        // =====================================================================
        // Stage 8: Residual Add
        // =====================================================================
        std::string residual_dep = layer.attn_gate
                                       ? (prefix + "attn_output_gate")
                                       : (prefix + "gdn_out_proj");

        ResidualAddStage::Params res_params;
        res_params.device_id = device;
        res_params.input = buffers.attn_proj;
        res_params.residual = buffers.current_hidden;
        res_params.output = buffers.current_hidden; // In-place residual
        res_params.num_elements = static_cast<size_t>(total_tokens) * static_cast<size_t>(d_model);
        res_params.input_buffer_id = BufferId::ATTN_PROJ;
        res_params.residual_buffer_id = BufferId::HIDDEN_STATE;
        res_params.output_buffer_id = BufferId::HIDDEN_STATE;

        graph.addNode(prefix + "attn_residual",
                      ComputeStageFactory::createResidualAdd(res_params),
                      device);
        graph.addDependency(prefix + "attn_residual", residual_dep);

        LOG_DEBUG("[Qwen35Graph] GDN attention graph for layer " << layer_idx
                  << " has " << graph.size() << " nodes");

        return graph;
    }

} // namespace llaminar2
