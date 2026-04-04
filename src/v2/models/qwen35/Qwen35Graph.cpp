/**
 * @file Qwen35Graph.cpp
 * @brief Qwen 3.5 compute graph builder implementation
 */

#include "Qwen35Graph.h"
#include "Qwen35Schema.h"
#include "../../execution/compute_stages/ComputeStages.h"
#include "../../execution/local_execution/graph/GraphBuildUtils.h"
#include "../../kernels/KernelFactory.h"
#include "../../tensors/TensorKernels.h"
#include "../../utils/Logger.h"

namespace llaminar2
{
    using KernelFactory = llaminar::v2::kernels::KernelFactory;

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
        const int n_heads = config_.gdn_group_count > 0
                                ? config_.gdn_group_count
                                : config_.n_heads;
        const int conv_kernel = config_.gdn_conv_kernel_size;

        // GDN dimensions: d_k == d_v == state_size, inner_size == n_heads * d_v
        // QKV merged dim = 3 * inner_size (Q + K + V each = inner_size)
        const int d_v = config_.gdn_state_size;
        const int d_k = d_v;
        const int inner_size = n_heads * d_v;
        const int qkv_dim = 3 * inner_size;

        conv_states_.resize(n_layers);
        recurrence_states_.resize(n_layers);
        conv_kernels_.resize(n_layers);
        rec_kernels_.resize(n_layers);

        for (int i = 0; i < n_layers; ++i)
        {
            if (!isGDNLayer(i))
                continue;

            // Conv state: [qkv_dim, kernel_size - 1] (covers full QKV channels)
            const int conv_state_size = qkv_dim * (conv_kernel - 1);
            conv_states_[i].resize(conv_state_size, 0.0f);

            // Recurrence state: [n_heads, d_k, d_v]
            const int recurrence_state_size = n_heads * d_k * d_v;
            recurrence_states_[i].resize(recurrence_state_size, 0.0f);

            // Create kernel instances via KernelFactory (lifetime tied to Qwen35Graph)
            conv_kernels_[i] = KernelFactory::createShortConvolution(DeviceType::CPU);
            rec_kernels_[i] = KernelFactory::createGatedDeltaNet(DeviceType::CPU);

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
    // setArena — wire GDN-specific buffers after base Qwen2 wiring
    // =========================================================================

    void Qwen35Graph::setArena(BufferArena *arena)
    {
        Qwen2Graph::setArena(arena);
        // GDN extension buffers are automatically discovered by
        // initializeInferenceStateFromArena() via forEachRegistered().
        // No manual wiring needed — the schema + resolver config register
        // them, and the auto-discovery pipeline handles propagation.
    }

    // =========================================================================
    // Resolver Config — extends Qwen2 with GDN-specific formulas + mappings
    // =========================================================================

    GraphResolverConfig Qwen35Graph::getResolverConfig(int seq_len) const
    {
        // Start with Qwen2's resolver config (covers standard dimensions)
        GraphResolverConfig config = Qwen2Graph::getResolverConfig(seq_len);

        // Add GDN-specific shape formulas (used by Qwen35Schema buffer specs)
        const int gdn_n_heads = config_.gdn_group_count > 0
                                    ? config_.gdn_group_count
                                    : config_.n_heads;
        const int gdn_inner = gdn_n_heads * config_.gdn_state_size;
        config.custom_formulas["gdn_inner_size"] =
            static_cast<size_t>(gdn_inner);
        config.custom_formulas["gdn_qkv_dim"] =
            static_cast<size_t>(3 * gdn_inner);
        config.custom_formulas["gdn_time_step_rank"] =
            static_cast<size_t>(config_.gdn_time_step_rank);

        // FA-specific: Q projection outputs query + sigmoid gate (2× normal Q dim)
        config.custom_formulas["fa_q_full_dim"] =
            static_cast<size_t>(config_.n_heads * config_.head_dim * 2);

        // Add GDN buffer name → BufferId mappings
        config.buffer_name_to_id["gdn_qkv"] = BufferId::GDN_QKV;
        config.buffer_name_to_id["gdn_z"] = BufferId::GDN_Z;
        config.buffer_name_to_id["gdn_alpha"] = BufferId::GDN_ALPHA;
        config.buffer_name_to_id["gdn_beta"] = BufferId::GDN_BETA;
        config.buffer_name_to_id["fa_gate"] = BufferId::FA_GATE;
        config.buffer_name_to_id["fa_q_raw"] = BufferId::FA_Q_RAW;

        LOG_DEBUG("[Qwen35Graph::getResolverConfig] GDN formulas: "
                  << "gdn_inner_size=" << gdn_inner
                  << ", gdn_qkv_dim=" << (3 * gdn_inner)
                  << ", gdn_time_step_rank=" << config_.gdn_time_step_rank);

        return config;
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
            // Full attention layers — custom Qwen3.5 FA path with Q gate split
            return buildFAAttentionGraph(
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

        const int n_heads = config_.gdn_group_count > 0
                                ? config_.gdn_group_count
                                : config_.n_heads;
        const int d_model = config_.d_model;
        const int d_v = config_.gdn_state_size;
        const int d_k = d_v;                  // Qwen3.5 GDN: d_k == d_v == state_size
        const int inner_size = n_heads * d_v; // value_dim = key_dim = n_heads * d_v
        const int qkv_dim = 3 * inner_size;   // Q + K + V = 3 * n_heads * d_k

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
        proj_params.output_qkv = buffers.get(BufferId::GDN_QKV);
        proj_params.n_qkv = qkv_dim;

        proj_params.w_z = layer.attn_gate; // Z projection = attn_gate.weight (in_proj_z in HF)
        proj_params.output_z = buffers.get(BufferId::GDN_Z);
        proj_params.n_z = n_heads * d_v;

        proj_params.w_a = layer.ssm_alpha;
        proj_params.output_a = buffers.get(BufferId::GDN_ALPHA);
        proj_params.n_a = n_heads;

        proj_params.w_b = layer.ssm_beta;
        proj_params.output_b = buffers.get(BufferId::GDN_BETA);
        proj_params.n_b = n_heads;

        proj_params.input_buffer_id = BufferId::NORMALIZED;
        proj_params.output_qkv_buffer_id = BufferId::GDN_QKV;
        proj_params.output_z_buffer_id = BufferId::GDN_Z;
        proj_params.output_a_buffer_id = BufferId::GDN_ALPHA;
        proj_params.output_b_buffer_id = BufferId::GDN_BETA;

        graph.addNode(prefix + "gdn_proj",
                      ComputeStageFactory::createGDNProjection(proj_params),
                      device);
        graph.addDependency(prefix + "gdn_proj", prefix + "attn_norm");

        // =====================================================================
        // Stage 3: Short Conv1d + SiLU on QKV
        // =====================================================================
        ShortConv1dStage::Params conv_params;
        conv_params.device_id = device;
        conv_params.input = buffers.get(BufferId::GDN_QKV);
        conv_params.output = buffers.get(BufferId::GDN_QKV); // In-place (conv modifies QKV)
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
                      ComputeStageFactory::createShortConv1d(conv_params),
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
        rec_params.Q = buffers.get(BufferId::GDN_QKV); // Will be split by kernel
        rec_params.K = buffers.get(BufferId::GDN_QKV); // Same tensor, offset by kernel
        rec_params.V = buffers.get(BufferId::GDN_QKV); // Same tensor, offset by kernel
        rec_params.alpha = buffers.get(BufferId::GDN_ALPHA);
        rec_params.beta = buffers.get(BufferId::GDN_BETA);
        rec_params.A_log = layer.ssm_a; // Learnable log-space gate
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
                      ComputeStageFactory::createGDNRecurrence(rec_params),
                      device);
        graph.addDependency(prefix + "gdn_recurrence", prefix + "short_conv");

        // =====================================================================
        // Stage 5: Gated RMSNorm — RMSNorm(output) * SiLU(Z)
        // =====================================================================
        GatedRMSNormStage::Params gnorm_params;
        gnorm_params.device_id = device;
        gnorm_params.input = buffers.attn_output;
        gnorm_params.gate = buffers.get(BufferId::GDN_Z);
        gnorm_params.output = buffers.attn_output; // In-place
        gnorm_params.gamma = layer.ssm_norm;
        gnorm_params.eps = config_.rms_norm_eps;
        gnorm_params.subtract_one = config_.rms_norm_subtract_one;
        gnorm_params.seq_len = total_tokens;
        gnorm_params.norm_dim = d_v;   // Per-head normalization over d_v (128)
        gnorm_params.gate_silu = true; // GDN uses SiLU(Z) as gate
        gnorm_params.input_buffer_id = BufferId::ATTN_OUTPUT;
        gnorm_params.gate_buffer_id = BufferId::GDN_Z;
        gnorm_params.output_buffer_id = BufferId::ATTN_OUTPUT;

        graph.addNode(prefix + "gated_norm",
                      ComputeStageFactory::createGatedRMSNorm(gnorm_params),
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

        // NOTE: GDN layers do NOT apply a sigmoid output gate after out_proj.
        // The Z projection is consumed entirely by GatedRMSNorm (SiLU gating).
        // Only FA layers use a sigmoid output gate (embedded in Q projection).

        // NOTE: No explicit ResidualAdd here. The FFN's FusedResidualNormStage
        // (from buildFFNGraph) fuses the attention residual add with FFN norm:
        //   HIDDEN_STATE = HIDDEN_STATE + ATTN_PROJ, then RMSNorm(HIDDEN_STATE)
        // This matches the Qwen2Graph convention for FA layers.

        LOG_DEBUG("[Qwen35Graph] GDN attention graph for layer " << layer_idx
                                                                 << " has " << graph.size() << " nodes");

        return graph;
    }

    // =========================================================================
    // FA (Full Attention) Sub-Graph — Qwen 3.5 specific
    // =========================================================================
    //
    // Key differences from Qwen2Graph::buildAttentionGraph:
    //   1. Q GEMM outputs [seq, n_heads * head_dim * 2] to fa_q_raw buffer
    //   2. QGateSplitStage deinterleaves into Q [seq, n_heads*head_dim] + fa_gate
    //   3. QK norms, partial RoPE (config_.partial_rotary_factor), KV cache, attention — same as Qwen2
    //   4. AttentionOutputGateStage: attn_output *= sigmoid(fa_gate)  BEFORE Wo
    //   5. Wo GEMM on gated attention output
    // =========================================================================

    ComputeGraph Qwen35Graph::buildFAAttentionGraph(
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
        ComputeGraph graph;
        std::string prefix = "layer" + std::to_string(layer_idx) + "_";
        int total_tokens = batch_size * seq_len;

        LOG_DEBUG("[Qwen35Graph::buildFAAttentionGraph] layer=" << layer_idx
                                                                << " seq_len=" << seq_len << " batch_size=" << batch_size
                                                                << " total_tokens=" << total_tokens);

        // =================================================================
        // Stage 1: Pre-attention RMSNorm (same as Qwen2)
        // =================================================================
        if (!config_.isHybridQ16() && layer_idx > config_.pp_layer_offset)
        {
            FusedResidualNormStage::Params fused_params;
            fused_params.device_id = device;
            fused_params.input = buffers.attn_proj;
            fused_params.residual = buffers.current_hidden;
            fused_params.gamma = layer.attn_norm;
            fused_params.norm_output = buffers.normalized;
            fused_params.eps = config_.rms_norm_eps;
            fused_params.seq_len = total_tokens;
            fused_params.hidden_dim = config_.d_model;
            fused_params.input_buffer_id = BufferId::ATTN_PROJ;
            fused_params.residual_buffer_id = BufferId::HIDDEN_STATE;
            fused_params.norm_output_buffer_id = BufferId::NORMALIZED;

            graph.addNode(prefix + "attn_norm",
                          ComputeStageFactory::createFusedResidualNorm(fused_params),
                          device);
        }
        else
        {
            RMSNormStage::Params attn_norm_params;
            attn_norm_params.input = buffers.current_hidden;
            attn_norm_params.output = buffers.normalized;
            attn_norm_params.gamma = layer.attn_norm;
            attn_norm_params.eps = config_.rms_norm_eps;
            attn_norm_params.seq_len = total_tokens;
            attn_norm_params.device_id = device;
            attn_norm_params.input_buffer_id = BufferId::HIDDEN_STATE;
            attn_norm_params.output_buffer_id = BufferId::NORMALIZED;

            graph.addNode(prefix + "attn_norm",
                          ComputeStageFactory::createRMSNorm(attn_norm_params),
                          device);
        }

        // =================================================================
        // Stage 2: Q/K/V projections — Q outputs to fa_q_raw (2× width)
        // =================================================================
        const bool has_qkv_proj = (layer.wq && layer.wk && layer.wv);

        // Retrieve extension buffers for FA gate path
        TensorBase *fa_q_raw = buffers.get(BufferId::FA_Q_RAW);
        TensorBase *fa_gate = buffers.get(BufferId::FA_GATE);

        if (!fa_q_raw || !fa_gate)
        {
            LOG_ERROR("[Qwen35Graph::buildFAAttentionGraph] Missing FA extension buffers "
                      "(fa_q_raw="
                      << fa_q_raw << ", fa_gate=" << fa_gate << ")");
            return graph;
        }

        if (has_qkv_proj)
        {
            int k = config_.d_model;
            int q_n = static_cast<int>(layer.wq->shape()[0]); // n_heads * head_dim * 2 = 4096
            int k_n = static_cast<int>(layer.wk->shape()[0]);
            int v_n = static_cast<int>(layer.wv->shape()[0]);

            LOG_DEBUG("[Qwen35Graph FA] Layer " << layer_idx << " QKV dims: q_n=" << q_n
                                                << " k_n=" << k_n << " v_n=" << v_n);

            // Q GEMM writes to fa_q_raw (oversized: n_heads * head_dim * 2)
            FusedQKVGEMMStage::Params qkv_params;
            qkv_params.input = buffers.normalized;
            qkv_params.m = total_tokens;
            qkv_params.k = k;
            qkv_params.wq = layer.wq;
            qkv_params.output_q = fa_q_raw; // Write to fa_q_raw, NOT buffers.Q
            qkv_params.n_q = q_n;
            qkv_params.bias_q = layer.q_bias;
            qkv_params.wk = layer.wk;
            qkv_params.output_k = buffers.K;
            qkv_params.n_k = k_n;
            qkv_params.bias_k = layer.k_bias;
            qkv_params.wv = layer.wv;
            qkv_params.output_v = buffers.V;
            qkv_params.n_v = v_n;
            qkv_params.bias_v = layer.v_bias;
            qkv_params.device_id = device;
            qkv_params.input_buffer_id = BufferId::NORMALIZED;
            qkv_params.output_q_buffer_id = BufferId::FA_Q_RAW;
            qkv_params.output_k_buffer_id = BufferId::K_PROJ;
            qkv_params.output_v_buffer_id = BufferId::V_PROJ;

            graph.addNode(prefix + "qkv_proj",
                          ComputeStageFactory::createFusedQKVGEMM(qkv_params),
                          device);
            graph.addDependency(prefix + "qkv_proj", prefix + "attn_norm");
        }

        // =================================================================
        // Stage 2.5: QGateSplit — deinterleave fa_q_raw → Q + fa_gate
        // =================================================================
        {
            int local_n_heads = config_.qkv_column_parallel
                                    ? config_.local_n_heads
                                    : config_.n_heads;
            if (local_n_heads <= 0)
                local_n_heads = config_.n_heads;

            QGateSplitStage::Params split_params;
            split_params.device_id = device;
            split_params.input = fa_q_raw;
            split_params.output_q = buffers.Q;
            split_params.output_gate = fa_gate;
            split_params.seq_len = total_tokens;
            split_params.n_heads = local_n_heads;
            split_params.head_dim = config_.head_dim;
            split_params.input_buffer_id = BufferId::FA_Q_RAW;
            split_params.output_q_buffer_id = BufferId::Q_PROJ;
            split_params.output_gate_buffer_id = BufferId::FA_GATE;

            graph.addNode(prefix + "q_gate_split",
                          ComputeStageFactory::createQGateSplit(split_params),
                          device);

            if (has_qkv_proj)
                graph.addDependency(prefix + "q_gate_split", prefix + "qkv_proj");
        }

        // =================================================================
        // Resolve local head counts for TP
        // =================================================================
        int local_n_heads = config_.qkv_column_parallel
                                ? config_.local_n_heads
                                : config_.n_heads;
        int local_n_kv_heads = config_.qkv_column_parallel
                                   ? config_.local_n_kv_heads
                                   : config_.n_kv_heads;
        if (local_n_heads <= 0)
            local_n_heads = config_.n_heads;
        if (local_n_kv_heads <= 0)
            local_n_kv_heads = config_.n_kv_heads;

        // =================================================================
        // Stage 2.75: Per-head QK RMSNorm (Qwen3.5 FA layers have QK norm)
        // =================================================================
        if (layer.q_norm && layer.k_norm)
        {
            QKNormStage::Params q_norm_params;
            q_norm_params.input = buffers.Q;
            q_norm_params.output = buffers.Q;
            q_norm_params.gamma = layer.q_norm;
            q_norm_params.n_heads = local_n_heads;
            q_norm_params.head_dim = config_.head_dim;
            q_norm_params.eps = config_.rms_norm_eps;
            q_norm_params.seq_len = total_tokens;
            q_norm_params.device_id = device;
            q_norm_params.input_buffer_id = BufferId::Q_PROJ;
            q_norm_params.output_buffer_id = BufferId::Q_PROJ;

            graph.addNode(prefix + "q_norm",
                          ComputeStageFactory::createQKNorm(q_norm_params),
                          device);
            graph.addDependency(prefix + "q_norm", prefix + "q_gate_split");

            QKNormStage::Params k_norm_params;
            k_norm_params.input = buffers.K;
            k_norm_params.output = buffers.K;
            k_norm_params.gamma = layer.k_norm;
            k_norm_params.n_heads = local_n_kv_heads;
            k_norm_params.head_dim = config_.head_dim;
            k_norm_params.eps = config_.rms_norm_eps;
            k_norm_params.seq_len = total_tokens;
            k_norm_params.device_id = device;
            k_norm_params.input_buffer_id = BufferId::K_PROJ;
            k_norm_params.output_buffer_id = BufferId::K_PROJ;

            graph.addNode(prefix + "k_norm",
                          ComputeStageFactory::createQKNorm(k_norm_params),
                          device);
            if (has_qkv_proj)
                graph.addDependency(prefix + "k_norm", prefix + "qkv_proj");
        }

        // =================================================================
        // Stage 3: RoPE on Q and K (with partial_rotary_factor from config)
        // =================================================================
        {
            int pos_offset = position_ids ? position_ids[0] : 0;

            RoPEStage::Params rope_params;
            rope_params.device_id = device;
            rope_params.Q = buffers.Q;
            rope_params.K = buffers.K;
            rope_params.n_heads = local_n_heads;
            rope_params.n_kv_heads = local_n_kv_heads;
            rope_params.head_dim = config_.head_dim;
            rope_params.pos_offset = pos_offset;
            rope_params.position_ids = position_ids;
            rope_params.theta_base = config_.rope_theta;
            rope_params.seq_len = total_tokens;
            rope_params.q_buffer_id = BufferId::Q_PROJ;
            rope_params.k_buffer_id = BufferId::K_PROJ;
            rope_params.partial_rotary_factor = config_.partial_rotary_factor;
            rope_params.skip_k = config_.rope_on_read;

            graph.addNode(prefix + "rope",
                          ComputeStageFactory::createRoPE(rope_params),
                          device);

            if (layer.q_norm && layer.k_norm)
            {
                graph.addDependency(prefix + "rope", prefix + "q_norm");
                graph.addDependency(prefix + "rope", prefix + "k_norm");
            }
            else
            {
                graph.addDependency(prefix + "rope", prefix + "q_gate_split");
                if (has_qkv_proj)
                    graph.addDependency(prefix + "rope", prefix + "qkv_proj");
            }
        }

        // =================================================================
        // Stage 4: KV cache append + attention compute (same as Qwen2)
        // =================================================================
        std::string wo_producer_node;

        {
            int kv_local_layer = layer_idx - config_.pp_layer_offset;

            if (kv_cache)
            {
                KVCacheAppendStage::Params kv_append_params;
                kv_append_params.device_id = device;
                kv_append_params.K = buffers.K;
                kv_append_params.k_buffer_id = BufferId::K_PROJ;
                kv_append_params.V = buffers.V;
                kv_append_params.v_buffer_id = BufferId::V_PROJ;
                kv_append_params.kv_cache = kv_cache;
                kv_append_params.layer_idx = kv_local_layer;
                kv_append_params.seq_idx = 0;
                kv_append_params.num_tokens = total_tokens;
                kv_append_params.batch_size = batch_size;
                kv_append_params.seq_len = seq_len;
                kv_append_params.kv_cache_scale = config_.kv_cache_scale;
                kv_append_params.head_dim = config_.head_dim;
                kv_append_params.turboquant_ctx = config_.turboquant_ctx;

                graph.addNode(prefix + "kv_append",
                              ComputeStageFactory::createKVCacheAppend(kv_append_params),
                              device);

                if (!config_.rope_on_read)
                    graph.addDependency(prefix + "kv_append", prefix + "rope");
                else if (has_qkv_proj)
                    graph.addDependency(prefix + "kv_append", prefix + "qkv_proj");
            }

            // Determine K/V source for attention
            ITensor *K_for_attn = buffers.K;
            ITensor *V_for_attn = buffers.V;
            int kv_len = total_tokens;
            bool use_gather_stage = false;

            if (kv_cache)
            {
                int cached_tokens = kv_cache->get_cached_tokens(kv_local_layer, 0);
                if (cached_tokens > 0 && batch_size == 1)
                {
                    K_for_attn = kv_cache->get_k(kv_local_layer, 0);
                    V_for_attn = kv_cache->get_v(kv_local_layer, 0);
                    kv_len = cached_tokens;
                }
                else if (cached_tokens > 0 && batch_size > 1)
                {
                    if (buffers.gathered_K && buffers.gathered_V)
                    {
                        use_gather_stage = true;
                        K_for_attn = buffers.gathered_K;
                        V_for_attn = buffers.gathered_V;
                        kv_len = cached_tokens;
                    }
                }
            }

            if (use_gather_stage)
            {
                KVCacheGatherStage::Params gather_params;
                gather_params.kv_cache = kv_cache;
                gather_params.layer_idx = kv_local_layer;
                gather_params.batch_size = batch_size;
                gather_params.out_K = buffers.gathered_K;
                gather_params.out_V = buffers.gathered_V;

                graph.addNode(prefix + "kv_gather",
                              ComputeStageFactory::createKVCacheGather(gather_params),
                              device);
                graph.addDependency(prefix + "kv_gather", prefix + "kv_append");
            }

            // Attention compute
            {
                AttentionMode mode = detect_attention_mode(batch_size, seq_len, kv_len);

                AttentionComputeStage::Params attn_params;
                attn_params.Q = buffers.Q;
                attn_params.K = K_for_attn;
                attn_params.V = V_for_attn;
                attn_params.output = buffers.attn_output;
                attn_params.batch_size = batch_size;
                attn_params.seq_len = seq_len;
                attn_params.kv_len = kv_len;
                attn_params.n_heads = local_n_heads;
                attn_params.n_kv_heads = local_n_kv_heads;
                attn_params.head_dim = config_.head_dim;
                attn_params.causal = true;
                attn_params.window_size = -1;
                attn_params.attention_mode = mode;
                attn_params.auto_detect_mode = true;
                attn_params.workspace_scores = buffers.workspace_scores;
                attn_params.workspace_context = buffers.workspace_context;
                attn_params.workspace_mask = buffers.workspace_mask;
                attn_params.kv_cache = kv_cache;
                attn_params.layer_idx = kv_local_layer;
                attn_params.read_kv_from_cache = device.is_gpu() &&
                                                 (!kv_cache || kv_cache->precision() != ActivationPrecision::Q8_1) &&
                                                 (!kv_cache || (kv_cache->precision() != ActivationPrecision::TQ8 &&
                                                                kv_cache->precision() != ActivationPrecision::TQ4));
                attn_params.position_offset = position_ids ? position_ids[0] : 0;
                attn_params.mpi_ctx = mpi_ctx_.get();
                attn_params.device_id = device;
                attn_params.q_buffer_id = BufferId::Q_PROJ;
                attn_params.output_buffer_id = BufferId::ATTN_OUTPUT;
                attn_params.workspace_scores_buffer_id = BufferId::ATTN_SCORES_WORKSPACE;
                attn_params.workspace_context_buffer_id = BufferId::ATTN_CONTEXT_WORKSPACE;
                attn_params.turboquant_ctx = config_.turboquant_ctx;

                if (config_.rope_on_read)
                {
                    attn_params.apply_rope_to_k = true;
                    attn_params.rope_theta = config_.rope_theta;
                }

                graph.addNode(prefix + "attention",
                              ComputeStageFactory::createAttentionCompute(attn_params),
                              device);

                if (use_gather_stage)
                    graph.addDependency(prefix + "attention", prefix + "kv_gather");
                else if (kv_cache)
                    graph.addDependency(prefix + "attention", prefix + "kv_append");
                else
                    graph.addDependency(prefix + "attention", prefix + "rope");
            }

            // =================================================================
            // Stage 4.5: Sigmoid output gate — attn_output *= sigmoid(fa_gate)
            // =================================================================
            // In PyTorch: attn_output = attn_output * torch.sigmoid(gate)
            // Applied BEFORE Wo projection.
            {
                AttentionOutputGateStage::Params gate_params;
                gate_params.device_id = device;
                gate_params.input = buffers.attn_output;
                gate_params.gate = fa_gate;
                gate_params.output = buffers.attn_output; // In-place
                gate_params.seq_len = total_tokens;
                gate_params.input_buffer_id = BufferId::ATTN_OUTPUT;
                gate_params.gate_buffer_id = BufferId::FA_GATE;
                gate_params.output_buffer_id = BufferId::ATTN_OUTPUT;

                graph.addNode(prefix + "attn_output_gate",
                              ComputeStageFactory::createAttentionOutputGate(gate_params),
                              device);
                graph.addDependency(prefix + "attn_output_gate", prefix + "attention");
                // Gate data was produced by q_gate_split (computed much earlier)
                graph.addDependency(prefix + "attn_output_gate", prefix + "q_gate_split");
            }

            // =================================================================
            // Stage 5: Wo projection
            // =================================================================
            if (layer.wo)
            {
                int wo_n = static_cast<int>(layer.wo->shape()[0]);
                int wo_k = static_cast<int>(layer.wo->shape()[1]);

                wo_producer_node = prefix + "wo_proj";
                graph.addNode(wo_producer_node,
                              ComputeStageFactory::createGEMM(
                                  GEMMStage::Params{
                                      .device_id = device,
                                      .A = buffers.attn_output,
                                      .B = layer.wo,
                                      .C = buffers.attn_proj,
                                      .m = total_tokens,
                                      .n = wo_n,
                                      .k = wo_k,
                                      .alpha = 1.0f,
                                      .beta = 0.0f,
                                      .transpose_B = false,
                                      .gemm_context = GemmContext::ATTN,
                                      .a_buffer_id = BufferId::ATTN_OUTPUT,
                                      .c_buffer_id = BufferId::ATTN_PROJ}),
                              device);
                graph.addDependency(wo_producer_node, prefix + "attn_output_gate");
            }

            // TP AllReduce for Wo (same as Qwen2)
            if (layer.wo && !wo_producer_node.empty())
            {
                bool wo_is_sharded = graph_utils::isRowParallelSharded(layer.wo);
                if (wo_is_sharded && needsTPAllreduce())
                {
                    size_t allreduce_count = static_cast<size_t>(total_tokens) * config_.d_model;
                    TensorBase *allreduce_buffer = buffers.attn_proj;
                    BufferId wo_allreduce_bid = BufferId::ATTN_PROJ;
                    std::string stage_name = prefix + "wo_allreduce";

                    auto allreduce_stage = createTPAllreduceStage(
                        allreduce_buffer, allreduce_count, device, layer_idx,
                        /*is_attention=*/true, stage_name, wo_allreduce_bid);

                    if (allreduce_stage)
                    {
                        graph.addNode(stage_name, std::move(allreduce_stage), device);
                        graph.addDependency(stage_name, wo_producer_node);
                        wo_producer_node = stage_name;
                    }
                }
            }
        }

        graph.setTerminalNode(wo_producer_node);

        LOG_DEBUG("[Qwen35Graph] FA attention graph for layer " << layer_idx
                                                                << " has " << graph.size() << " nodes");

        return graph;
    }

} // namespace llaminar2