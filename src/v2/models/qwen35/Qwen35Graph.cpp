/**
 * @file Qwen35Graph.cpp
 * @brief Qwen 3.5 compute graph builder implementation
 */

#include "Qwen35Graph.h"
#include "Qwen35Schema.h"
#include "../../execution/compute_stages/ComputeStages.h"
#include "../../execution/local_execution/graph/GraphBuildUtils.h"
#include "../../kernels/HybridKVCacheConfig.h"
#include "../../kernels/IHybridKVCache.h"
#include "../../kernels/KernelFactory.h"
#include "../../tensors/TensorKernels.h"
#include "../../utils/Assertions.h"
#include "../../utils/Logger.h"

#include <set>

namespace llaminar2
{
    using KernelFactory = llaminar::v2::kernels::KernelFactory;

    // =========================================================================
    // Helper: populate allreduce precision with FA-layer awareness
    // =========================================================================

    static void populateHybridAllreducePrecision(GraphConfig &config)
    {
        if (!config.tp_allreduce_precision.empty() || config.n_layers <= 0)
            return;

        Qwen35SchemaFactory factory;
        auto schema = factory.createSchema();

        // Build the forced-FP32 set from FA layer indices.
        // FA layers are more sensitive to allreduce precision drift because
        // they carry the full sequence-length attention context, whereas
        // GDN layers operate on compressed recurrent state.
        std::set<int> fa_layers;
        for (int i = 0; i < config.n_layers; ++i)
        {
            if (i < static_cast<int>(config.layer_types.size()) &&
                config.layer_types[i] == "full_attention")
            {
                fa_layers.insert(i);
            }
        }

        if (!fa_layers.empty())
        {
            // Use layer-type-aware overload: FA layers always FP32,
            // first N GDN layers FP32, rest GDN layers use schema default
            config.populateAllreducePrecision(
                schema.tp_allreduce_default_precision,
                schema.tp_allreduce_fp32_layer_count,
                fa_layers);

            int gdn_fp32 = 0, gdn_fp16 = 0;
            for (int i = 0; i < config.n_layers; ++i)
            {
                if (!fa_layers.count(i))
                {
                    if (config.tp_allreduce_precision[i] == "fp32")
                        ++gdn_fp32;
                    else
                        ++gdn_fp16;
                }
            }
            LOG_DEBUG("[Qwen35Graph] Hybrid allreduce precision: "
                      << fa_layers.size() << " FA layers=fp32, "
                      << gdn_fp32 << " GDN layers=fp32, "
                      << gdn_fp16 << " GDN layers=" << schema.tp_allreduce_default_precision);
        }
        else
        {
            // No layer types available — fall back to count-based policy
            config.populateAllreducePrecision(
                schema.tp_allreduce_default_precision,
                schema.tp_allreduce_fp32_layer_count);
            LOG_DEBUG("[Qwen35Graph] Allreduce precision (count-based): "
                      << "fp32_layers=" << schema.tp_allreduce_fp32_layer_count
                      << " default=" << schema.tp_allreduce_default_precision);
        }
    }

    // =========================================================================
    // Constructors
    // =========================================================================

    Qwen35Graph::Qwen35Graph(
        std::shared_ptr<ModelContext> model_ctx,
        std::shared_ptr<IMPIContext> mpi_ctx,
        const GraphConfig &config)
        : QwenGraphBase(std::move(model_ctx), std::move(mpi_ctx), config)
    {
        populateHybridAllreducePrecision(config_);
    }

    Qwen35Graph::Qwen35Graph(
        const GraphConfig &config,
        std::shared_ptr<IMPIContext> mpi_ctx)
        : QwenGraphBase(config, std::move(mpi_ctx))
    {
        populateHybridAllreducePrecision(config_);
    }

    // =========================================================================
    // GDN Layer Type Detection
    // =========================================================================

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
        QwenGraphBase::setArena(arena);
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
        GraphResolverConfig config = QwenGraphBase::getResolverConfig(seq_len);

        // Add GDN-specific shape formulas (used by Qwen35Schema buffer specs)
        // Use TP-aware local dimensions: scale GDN head counts by the same ratio
        // as the FA attention path (local_n_heads / n_heads).
        const int n_k_heads_full = config_.gdn.group_count > 0
                                       ? config_.gdn.group_count
                                       : config_.n_heads;
        const int n_v_heads_full = config_.gdn.time_step_rank > 0
                                       ? config_.gdn.time_step_rank
                                       : n_k_heads_full;
        int n_k_heads_local = n_k_heads_full;
        int n_v_heads_local = n_v_heads_full;
        const bool gdn_modular_repeat = (n_v_heads_full > n_k_heads_full);
        if (config_.qkv_column_parallel && config_.local_n_heads > 0 && config_.n_heads > 0)
        {
            // V-heads are always sharded
            n_v_heads_local = n_v_heads_full * config_.local_n_heads / config_.n_heads;
            if (n_v_heads_local <= 0)
                n_v_heads_local = 1;

            // K-heads: replicated for GDN modular repeat, sharded otherwise
            if (!gdn_modular_repeat)
            {
                n_k_heads_local = n_k_heads_full * config_.local_n_heads / config_.n_heads;
                if (n_k_heads_local <= 0)
                    n_k_heads_local = 1;
            }
            // else: n_k_heads_local stays at full count (replicated)
        }
        const int d_k = config_.gdn.state_size;
        const int key_dim = n_k_heads_local * d_k;
        const int gdn_inner = config_.gdn.inner_size > 0
                                  ? (config_.gdn.inner_size * n_v_heads_local / n_v_heads_full)
                                  : n_v_heads_local * config_.gdn.state_size;
        config.custom_formulas["gdn_inner_size"] =
            static_cast<size_t>(gdn_inner);
        config.custom_formulas["gdn_qkv_dim"] =
            static_cast<size_t>(2 * key_dim + gdn_inner);
        config.custom_formulas["gdn_time_step_rank"] =
            static_cast<size_t>(n_v_heads_local);

        // FA-specific: Q projection outputs query + sigmoid gate (2× normal Q dim)
        // Use local head count for TP
        const int local_n_heads_fa = (config_.qkv_column_parallel && config_.local_n_heads > 0)
                                         ? config_.local_n_heads
                                         : config_.n_heads;
        config.custom_formulas["fa_q_full_dim"] =
            static_cast<size_t>(local_n_heads_fa * config_.head_dim * 2);

        // attn_output must be wide enough for BOTH FA (local_qkv_dim) and GDN (gdn_inner_size).
        // GDN layers write n_v_heads * d_v elements per row; FA layers write local_n_heads * head_dim.
        // GatedRMSNorm and GEMV stride are derived from this buffer's column count,
        // so it MUST match the actual data width written by each layer type.
        const size_t local_qkv_dim = static_cast<size_t>(config.local_n_heads * config.head_dim);
        config.custom_formulas["attn_output_dim"] =
            std::max(local_qkv_dim, static_cast<size_t>(gdn_inner));

        // Add GDN buffer name → BufferId mappings
        config.buffer_name_to_id["gdn_qkv"] = BufferId::GDN_QKV;
        config.buffer_name_to_id["gdn_z"] = BufferId::GDN_Z;
        config.buffer_name_to_id["gdn_alpha"] = BufferId::GDN_ALPHA;
        config.buffer_name_to_id["gdn_beta"] = BufferId::GDN_BETA;
        config.buffer_name_to_id["fa_gate"] = BufferId::FA_GATE;
        config.buffer_name_to_id["fa_q_raw"] = BufferId::FA_Q_RAW;

        LOG_DEBUG("[Qwen35Graph::getResolverConfig] GDN formulas (TP-local): "
                  << "gdn_inner_size=" << gdn_inner
                  << ", gdn_qkv_dim=" << (2 * key_dim + gdn_inner)
                  << ", n_k_heads=" << n_k_heads_local << "/" << n_k_heads_full
                  << ", n_v_heads=" << n_v_heads_local << "/" << n_v_heads_full);

        return config;
    }

    // =========================================================================
    // Full Forward Graph — inherited from QwenGraphBase.
    // QwenGraphBase::buildFullForwardGraph() calls buildAttentionGraph()
    // virtually, which dispatches to GDN or FA via Qwen35Graph override.
    // No need to override here.
    // =========================================================================

    // buildLayerGraph() — inherited from QwenGraphBase.
    // The base implementation calls buildAttentionGraph() (virtual dispatch)
    // and buildFFNGraph(), so GDN dispatch is automatic.

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
                                          seq_len, batch_size, kv_cache, device);
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
        IKVCache *kv_cache,
        DeviceId device)
    {
        ComputeGraph graph;
        std::string prefix = "layer" + std::to_string(layer_idx) + "_";
        int total_tokens = batch_size * seq_len;
        LayerWeightBindings layer_bindings = layerWeightBindingsForGraph(layer_idx);

        // Get GDN state from hybrid KV cache
        auto *hybrid_cache = dynamic_cast<IHybridKVCache *>(kv_cache);
        LLAMINAR_ASSERT(hybrid_cache != nullptr,
                        "GDN layers require a hybrid KV cache (IHybridKVCache)");
        auto *gdn_state = hybrid_cache->getGDNState(layer_idx);
        LLAMINAR_ASSERTF(gdn_state != nullptr,
                         "No GDN state found for layer " << layer_idx);

        // Full (global) head counts from model config
        const int n_k_heads_full = config_.gdn.group_count > 0
                                       ? config_.gdn.group_count
                                       : config_.n_heads;
        const int n_v_heads_full = config_.gdn.time_step_rank > 0
                                       ? config_.gdn.time_step_rank
                                       : n_k_heads_full;
        const int d_model = config_.d_model;
        const int d_v = config_.gdn.state_size;
        const int d_k = d_v; // Qwen3.5 GDN: d_k == d_v == state_size

        // TP-aware local dimensions: derive from actual (possibly sharded) weight shapes.
        // When column-parallel TP is active, weights are already sharded by WeightManager
        // so shape[0] reflects the local output dimension for this rank.
        int qkv_dim, value_dim, n_k_heads, n_v_heads;
        if (layer.attn_qkv)
        {
            // Derive from actual weight shape (works for both sharded and full)
            qkv_dim = static_cast<int>(layer.attn_qkv->shape()[0]);
            value_dim = layer.attn_gate
                            ? static_cast<int>(layer.attn_gate->shape()[0])
                            : (config_.gdn.inner_size > 0
                                   ? config_.gdn.inner_size
                                   : n_v_heads_full * d_v);
            // Compute local head counts from local dimensions
            // qkv_dim = 2 * n_k_heads * d_k + value_dim, so:
            n_k_heads = (qkv_dim - value_dim) / (2 * d_k);
            n_v_heads = value_dim / d_v;
        }
        else
        {
            // Fallback to config (no weight tensor available yet)
            n_k_heads = n_k_heads_full;
            n_v_heads = n_v_heads_full;
            const int key_dim = n_k_heads * d_k;
            value_dim = config_.gdn.inner_size > 0
                            ? config_.gdn.inner_size
                            : n_v_heads * d_v;
            qkv_dim = 2 * key_dim + value_dim;
        }

        LOG_DEBUG("[Qwen35Graph] Building GDN attention for layer " << layer_idx
                                                                    << ": total_tokens=" << total_tokens
                                                                    << " n_k_heads=" << n_k_heads << " (full=" << n_k_heads_full << ")"
                                                                    << " n_v_heads=" << n_v_heads << " (full=" << n_v_heads_full << ")"
                                                                    << " d_k=" << d_k << " d_v=" << d_v
                                                                    << " qkv_dim=" << qkv_dim << " value_dim=" << value_dim);

        // =====================================================================
        // Stage 1: Pre-attention RMSNorm
        // =====================================================================
        // GDN layers don't check HybridQ16 (always use fused when not first layer)
        addPreAttentionNorm(graph, prefix, buffers, layer.attn_norm,
                            total_tokens, layer_idx, device, /*check_hybrid_q16=*/false);

        // =====================================================================
        // Stage 2: GDN 4-way Projection
        // =====================================================================
        GDNProjectionStage::Params proj_params;
        proj_params.device_id = device;
        proj_params.prepared_store = prepared_weight_store_;
        proj_params.input = buffers.normalized;
        proj_params.m = total_tokens;
        proj_params.k = d_model;

        proj_params.w_qkv = layer.attn_qkv;
        proj_params.prepared_ref_qkv = preparedRefForGraphWeight(layer_bindings.attn_qkv, device);
        proj_params.output_qkv = buffers.get(BufferId::GDN_QKV);
        proj_params.n_qkv = qkv_dim;

        proj_params.w_z = layer.attn_gate; // Z projection = attn_gate.weight (in_proj_z in HF)
        proj_params.prepared_ref_z = preparedRefForGraphWeight(layer_bindings.attn_gate, device);
        proj_params.output_z = buffers.get(BufferId::GDN_Z);
        proj_params.n_z = value_dim; // Z gate operates on value_dim (n_v_heads * d_v)

        proj_params.w_a = layer.ssm_alpha;
        proj_params.prepared_ref_a = preparedRefForGraphWeight(layer_bindings.ssm_alpha, device);
        proj_params.output_a = buffers.get(BufferId::GDN_ALPHA);
        proj_params.n_a = n_v_heads; // Alpha is per-value-head

        proj_params.w_b = layer.ssm_beta;
        proj_params.prepared_ref_b = preparedRefForGraphWeight(layer_bindings.ssm_beta, device);
        proj_params.output_b = buffers.get(BufferId::GDN_BETA);
        proj_params.n_b = n_v_heads; // Beta is per-value-head

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
        conv_params.conv_state = gdn_state->conv_state.data();
        conv_params.seq_len = total_tokens;
        conv_params.channels = qkv_dim;
        conv_params.kernel_size = config_.gdn.conv_kernel_size;

        // Use kernel instance from hybrid cache (lifetime tied to cache)
        conv_params.kernel = gdn_state->conv_kernel.get();

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
        // [seq_len, 2*n_k_heads*d_k + n_v_heads*d_v]
        // The recurrence stage splits Q, K, V internally and repeat_interleaves
        // Q/K from n_k_heads to n_v_heads when they differ.
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
        rec_params.recurrence_state = gdn_state->recurrence_state.data();
        rec_params.seq_len = total_tokens;
        rec_params.n_heads = n_v_heads;   // Recurrence runs with value head count
        rec_params.n_k_heads = n_k_heads; // Key head count for QKV split
        rec_params.d_k = d_k;
        rec_params.d_v = d_v;
        rec_params.chunk_size = 64;
        rec_params.use_qk_l2norm = true;

        // Under TP, V-heads are always sharded (each rank owns a contiguous
        // slice of global V-heads). The global_v_head_offset tells the
        // recurrence stage which global V-heads this rank owns, so the
        // deinterleave helper can select the correct K-heads:
        //   k_idx = (v_local + offset) % n_k_heads_local
        //
        // This is required in ALL TP modes where V is sharded:
        //   - Selection   (n_k > n_v_local):  K sharded alongside V
        //   - Identity    (n_k == n_v_local, K replicated at full count)
        //   - Expansion   (n_k < n_v_local):  K replicated, modular GQA repeat
        //                                     (e.g. 27B TP=2: n_k=16, n_v_local=24)
        //
        // V is sharded whenever n_v_heads < n_v_heads_full. Previously the
        // expansion case was missed, leaving rank>0 with offset=0 and reading
        // the wrong K-heads for its V-head slice.
        if (mpi_ctx_ && mpi_ctx_->world_size() > 1 && n_v_heads < n_v_heads_full)
        {
            rec_params.global_v_head_offset = mpi_ctx_->rank() * n_v_heads;
        }

        rec_params.output_buffer_id = BufferId::ATTN_OUTPUT;
        rec_params.qkv_buffer_id = BufferId::GDN_QKV;
        rec_params.alpha_buffer_id = BufferId::GDN_ALPHA;
        rec_params.beta_buffer_id = BufferId::GDN_BETA;

        // Use kernel instance from hybrid cache (lifetime tied to cache)
        rec_params.kernel = gdn_state->rec_kernel.get();

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
                                       // PyTorch reshapes to [B*T, n_heads, d_v] before norm
        gnorm_params.gate_silu = true; // GDN uses SiLU(Z) as gate
        gnorm_params.input_buffer_id = BufferId::ATTN_OUTPUT;
        gnorm_params.gate_buffer_id = BufferId::GDN_Z;
        gnorm_params.output_buffer_id = BufferId::ATTN_OUTPUT;

        graph.addNode(prefix + "gated_norm",
                      ComputeStageFactory::createGatedRMSNorm(gnorm_params),
                      device);
        graph.addDependency(prefix + "gated_norm", prefix + "gdn_recurrence");

        // =====================================================================
        // Stage 6: Output Projection (Wo GEMM) + optional TP AllReduce
        // =====================================================================
        std::string terminal_node = addWoProjectionAndAllreduce(
            graph, prefix, buffers, layer.ssm_out, layer_bindings.ssm_out,
            total_tokens, layer_idx, device,
            prefix + "gated_norm",
            "gdn_out_proj", "gdn_wo_allreduce");

        // NOTE: GDN layers do NOT apply a sigmoid output gate after out_proj.
        // The Z projection is consumed entirely by GatedRMSNorm (SiLU gating).
        // Only FA layers use a sigmoid output gate (embedded in Q projection).

        // NOTE: No explicit ResidualAdd here. The FFN's FusedResidualNormStage
        // (from buildFFNGraph) fuses the attention residual add with FFN norm:
        //   HIDDEN_STATE = HIDDEN_STATE + ATTN_PROJ, then RMSNorm(HIDDEN_STATE)
        // This matches the QwenStandardGraph convention for FA layers.

        graph.setTerminalNode(terminal_node);

        LOG_DEBUG("[Qwen35Graph] GDN attention graph for layer " << layer_idx
                                                                 << " has " << graph.size() << " nodes");

        return graph;
    }

    // =========================================================================
    // FA (Full Attention) Sub-Graph — Qwen 3.5 specific
    // =========================================================================
    //
    // Key differences from QwenStandardGraph::buildAttentionGraph:
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
        LayerWeightBindings layer_bindings = layerWeightBindingsForGraph(layer_idx);

        LOG_DEBUG("[Qwen35Graph::buildFAAttentionGraph] layer=" << layer_idx
                                                                << " seq_len=" << seq_len << " batch_size=" << batch_size
                                                                << " total_tokens=" << total_tokens);

        // =================================================================
        // Stage 1: Pre-attention RMSNorm (same as Qwen2)
        // =================================================================
        addPreAttentionNorm(graph, prefix, buffers, layer.attn_norm,
                            total_tokens, layer_idx, device);

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
            graph.addNode(prefix + "qkv_proj",
                          ComputeStageFactory::createFusedQKVGEMM({
                              .device_id = device,
                              .input = buffers.normalized,
                              .m = total_tokens,
                              .k = k,
                              .wq = layer.wq,
                              .output_q = fa_q_raw,
                              .n_q = q_n,
                              .bias_q = layer.q_bias,
                              .wk = layer.wk,
                              .output_k = buffers.K,
                              .n_k = k_n,
                              .bias_k = layer.k_bias,
                              .wv = layer.wv,
                              .output_v = buffers.V,
                              .n_v = v_n,
                              .bias_v = layer.v_bias,
                              .input_buffer_id = BufferId::NORMALIZED,
                              .output_q_buffer_id = BufferId::FA_Q_RAW,
                              .output_k_buffer_id = BufferId::K_PROJ,
                              .output_v_buffer_id = BufferId::V_PROJ,
                              .prepared_ref_q = preparedRefForGraphWeight(layer_bindings.wq, device),
                              .prepared_ref_k = preparedRefForGraphWeight(layer_bindings.wk, device),
                              .prepared_ref_v = preparedRefForGraphWeight(layer_bindings.wv, device),
                              .prepared_store = prepared_weight_store_,
                          }),
                          device);
            graph.addDependency(prefix + "qkv_proj", prefix + "attn_norm");
        }

        // =================================================================
        // Stage 2.5: QGateSplit — deinterleave fa_q_raw → Q + fa_gate
        // =================================================================
        auto [local_n_heads, local_n_kv_heads] = resolveLocalHeadCounts();
        {
            graph.addNode(prefix + "q_gate_split",
                          ComputeStageFactory::createQGateSplit({
                              .device_id = device,
                              .input = fa_q_raw,
                              .output_q = buffers.Q,
                              .output_gate = fa_gate,
                              .seq_len = total_tokens,
                              .n_heads = local_n_heads,
                              .head_dim = config_.head_dim,
                              .input_buffer_id = BufferId::FA_Q_RAW,
                              .output_q_buffer_id = BufferId::Q_PROJ,
                              .output_gate_buffer_id = BufferId::FA_GATE,
                          }),
                          device);

            if (has_qkv_proj)
                graph.addDependency(prefix + "q_gate_split", prefix + "qkv_proj");
        }

        // =================================================================
        // Stage 2.75: Per-head QK RMSNorm
        // =================================================================
        bool has_qk_norms = addQKNorms(
            graph, prefix, buffers, layer,
            local_n_heads, local_n_kv_heads, total_tokens, device,
            prefix + "q_gate_split",
            has_qkv_proj ? prefix + "qkv_proj" : prefix + "attn_norm");

        // =================================================================
        // Stage 3: RoPE on Q and K
        // =================================================================
        std::string rope_node = addRoPE(
            graph, prefix, buffers,
            local_n_heads, local_n_kv_heads, total_tokens,
            position_ids, device);

        if (has_qk_norms)
        {
            graph.addDependency(rope_node, prefix + "q_norm");
            graph.addDependency(rope_node, prefix + "k_norm");
        }
        else
        {
            graph.addDependency(rope_node, prefix + "q_gate_split");
            if (has_qkv_proj)
                graph.addDependency(rope_node, prefix + "qkv_proj");
        }

        // =================================================================
        // Stage 4: KV cache + attention compute
        // =================================================================
        std::string attn_node = addKVCacheAndAttention(
            graph, prefix, buffers, layer_idx,
            seq_len, batch_size, local_n_heads, local_n_kv_heads,
            kv_cache, position_ids, device, has_qkv_proj, rope_node);

        // =================================================================
        // Stage 4.5: Sigmoid output gate — attn_output *= sigmoid(fa_gate)
        // =================================================================
        {
            graph.addNode(prefix + "attn_output_gate",
                          ComputeStageFactory::createAttentionOutputGate({
                              .device_id = device,
                              .input = buffers.attn_output,
                              .gate = fa_gate,
                              .output = buffers.attn_output,
                              .seq_len = total_tokens,
                              .input_buffer_id = BufferId::ATTN_OUTPUT,
                              .gate_buffer_id = BufferId::FA_GATE,
                              .output_buffer_id = BufferId::ATTN_OUTPUT,
                          }),
                          device);
            graph.addDependency(prefix + "attn_output_gate", attn_node);
            graph.addDependency(prefix + "attn_output_gate", prefix + "q_gate_split");
        }

        // =================================================================
        // Stage 5: Wo projection + optional TP allreduce
        // =================================================================
        std::string terminal = addWoProjectionAndAllreduce(
            graph, prefix, buffers, layer.wo, layer_bindings.wo,
            total_tokens, layer_idx, device,
            prefix + "attn_output_gate");

        graph.setTerminalNode(terminal);

        LOG_DEBUG("[Qwen35Graph] FA attention graph for layer " << layer_idx
                                                                << " has " << graph.size() << " nodes");

        return graph;
    }

} // namespace llaminar2