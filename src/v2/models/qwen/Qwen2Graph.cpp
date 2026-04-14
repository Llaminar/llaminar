/**
 * @file Qwen2Graph.cpp
 * @brief Qwen2-specific graph builder implementation
 * @author David Sanftenberg
 * @date December 2025
 *
 * Implements the Qwen2-specific attention pattern and schema.
 * All shared infrastructure is in QwenGraphBase.cpp.
 */

#include "Qwen2Graph.h"
#include "Qwen2Schema.h"
#include "../../utils/Logger.h"
#include "../../utils/DebugEnv.h"
#include "../../tensors/TensorSlice.h"
#include "../../tensors/Tensors.h"
#include "../../utils/MPIContext.h"
#include "../../execution/local_execution/graph/GraphBuildUtils.h"
#include "../../execution/config/RuntimeConfig.h"
#include "../../collective/ILocalTPContext.h"
#include "../../execution/compute_stages/stages/QKNormStage.h"
#include "../../execution/compute_stages/stages/FusedResidualNormStage.h"
#include "../../memory/BufferId.h"
#include <stdexcept>

namespace llaminar2
{

    // Import graph_utils for cleaner code
    using namespace graph_utils;

    // =============================================================================
    // Constructors (delegate to QwenGraphBase, populate Qwen2 schema)
    // =============================================================================

    Qwen2Graph::Qwen2Graph(std::shared_ptr<ModelContext> model_ctx,
                           std::shared_ptr<IMPIContext> mpi_ctx,
                           const GraphConfig &config)
        : QwenGraphBase(std::move(model_ctx), std::move(mpi_ctx), config)
    {
        // Qwen2-specific: populate per-layer allreduce precision from schema
        if (config_.tp_allreduce_precision.empty() && config_.n_layers > 0)
        {
            Qwen2SchemaFactory factory;
            auto schema = factory.createSchema();
            config_.populateAllreducePrecision(
                schema.tp_allreduce_default_precision,
                schema.tp_allreduce_fp32_layer_count);
            LOG_DEBUG("[Qwen2Graph] Populated per-layer allreduce precision: "
                      << "fp32_layers=" << schema.tp_allreduce_fp32_layer_count
                      << " default=" << schema.tp_allreduce_default_precision);
        }
    }

    Qwen2Graph::Qwen2Graph(const GraphConfig &config,
                           std::shared_ptr<IMPIContext> mpi_ctx)
        : QwenGraphBase(config, std::move(mpi_ctx))
    {
        // Qwen2-specific: populate per-layer allreduce precision from schema
        if (config_.tp_allreduce_precision.empty() && config_.n_layers > 0)
        {
            Qwen2SchemaFactory factory;
            auto schema = factory.createSchema();
            config_.populateAllreducePrecision(
                schema.tp_allreduce_default_precision,
                schema.tp_allreduce_fp32_layer_count);
        }
    }

    // =============================================================================
    // Schema
    // =============================================================================

    GraphSchema Qwen2Graph::getSchema() const
    {
        Qwen2SchemaFactory factory;
        return factory.createSchema();
    }

    // =============================================================================
    // Attention Graph Building (Qwen2-specific)
    // =============================================================================

    ComputeGraph Qwen2Graph::buildAttentionGraph(
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

        // Total tokens for GEMM m dimension = batch_size * seq_len
        int total_tokens = batch_size * seq_len;

        LOG_DEBUG("[buildAttentionGraph] layer_idx=" << layer_idx << " seq_len=" << seq_len
                                                     << " batch_size=" << batch_size
                                                     << " total_tokens=" << total_tokens
                                                     << " layer.wq=" << static_cast<const void *>(layer.wq)
                                                     << " layer.wo=" << layer.wo << " world_size="
                                                     << (mpi_ctx_ ? mpi_ctx_->world_size() : 1)
                                                     << " device=" << device.to_string()
                                                     << " sequence_lengths=" << (sequence_lengths ? "valid" : "nullptr")
                                                     << (sequence_lengths ? " size=" + std::to_string(sequence_lengths->size()) : ""));

        // Stage 1: Pre-attention RMSNorm
        // Non-first layers: Fused ResidualAdd + RMSNorm (previous layer's ffn output + this norm)
        // First layer: standalone RMSNorm (no prior residual to fuse)
        {
            if (!config_.isHybridQ16() && layer_idx > config_.pp_layer_offset)
            {
                // Non-first layers: Fused ResidualAdd + RMSNorm
                // Combines previous layer's ffn_residual with this layer's attn_norm
                // On GPU: single fused kernel. On CPU: sequential ResidualAdd + RMSNorm via KernelFactory.
                FusedResidualNormStage::Params fused_params;
                fused_params.device_id = device;
                fused_params.input = buffers.attn_proj;         // Previous layer's down_proj output
                fused_params.residual = buffers.current_hidden; // Hidden state (in-place update)
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
                // First layer: standalone RMSNorm (no prior residual to fuse)
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
        }

        // Stage 2: Q/K/V projections using FusedQKVGEMMStage
        const bool has_qkv_proj = (layer.wq && layer.wk && layer.wv);
        if (has_qkv_proj)
        {
            LOG_DEBUG("[Qwen2Graph] Using FusedQKVGEMMStage");

            int k = config_.d_model;
            int q_n = static_cast<int>(layer.wq->shape()[0]);
            int k_n = static_cast<int>(layer.wk->shape()[0]);
            int v_n = static_cast<int>(layer.wv->shape()[0]);

            LOG_DEBUG("[Qwen2Graph] Layer " << layer_idx << " QKV dims: q_n=" << q_n
                                            << " k_n=" << k_n << " v_n=" << v_n
                                            << " wq_shape=[" << layer.wq->shape()[0] << "," << layer.wq->shape()[1] << "]"
                                            << " wk_shape=[" << layer.wk->shape()[0] << "," << layer.wk->shape()[1] << "]");

            FusedQKVGEMMStage::Params qkv_params;
            qkv_params.input = buffers.normalized;
            qkv_params.m = total_tokens; // Use total_tokens = batch_size * seq_len
            qkv_params.k = k;
            qkv_params.wq = layer.wq;
            qkv_params.output_q = buffers.Q;
            qkv_params.n_q = q_n;
            qkv_params.bias_q = layer.q_bias; // TensorBase* for tensor-aware GPU path
            qkv_params.wk = layer.wk;
            qkv_params.output_k = buffers.K;
            qkv_params.n_k = k_n;
            qkv_params.bias_k = layer.k_bias; // TensorBase* for tensor-aware GPU path
            qkv_params.wv = layer.wv;
            qkv_params.output_v = buffers.V;
            qkv_params.n_v = v_n;
            qkv_params.bias_v = layer.v_bias; // TensorBase* for tensor-aware GPU path
            qkv_params.device_id = device;
            qkv_params.input_buffer_id = BufferId::NORMALIZED;
            qkv_params.output_q_buffer_id = BufferId::Q_PROJ;
            qkv_params.output_k_buffer_id = BufferId::K_PROJ;
            qkv_params.output_v_buffer_id = BufferId::V_PROJ;
            LOG_DEBUG("[Qwen2Graph] Creating FusedQKVGEMM with device_id=" << device.to_string());

            graph.addNode(prefix + "qkv_proj",
                          ComputeStageFactory::createFusedQKVGEMM(qkv_params),
                          device);

            graph.addDependency(prefix + "qkv_proj", prefix + "attn_norm");
        }

        // =================================================================
        // Resolve local head counts for tensor-parallel attention
        // =================================================================
        // When qkv_column_parallel is enabled, each rank processes a subset of heads.
        // The weight shapes from QKV projection already reflect local dimensions,
        // and RoPE/Attention stages must use matching local head counts.
        // =================================================================
        int local_n_heads = config_.qkv_column_parallel
                                ? config_.local_n_heads
                                : config_.n_heads;
        int local_n_kv_heads = config_.qkv_column_parallel
                                   ? config_.local_n_kv_heads
                                   : config_.n_kv_heads;

        // Validate local head counts (safety check)
        if (local_n_heads <= 0)
            local_n_heads = config_.n_heads;
        if (local_n_kv_heads <= 0)
            local_n_kv_heads = config_.n_kv_heads;

        // Stage 2.5: Per-head QK RMSNorm (Qwen3)
        // Applied to Q and K projections independently before RoPE.
        // Each head is normalized separately with gamma of shape [head_dim].
        if (layer.q_norm && layer.k_norm)
        {
            LOG_DEBUG("[Qwen2Graph] Layer " << layer_idx << " using QK norm (Qwen3)");

            // Q norm: normalize each Q head independently
            QKNormStage::Params q_norm_params;
            q_norm_params.input = buffers.Q;
            q_norm_params.output = buffers.Q; // In-place
            q_norm_params.gamma = layer.q_norm;
            q_norm_params.n_heads = local_n_heads;
            q_norm_params.head_dim = config_.head_dim;
            q_norm_params.eps = config_.rms_norm_eps;
            q_norm_params.seq_len = total_tokens;
            q_norm_params.device_id = device;
            q_norm_params.input_buffer_id = BufferId::Q_PROJ;
            q_norm_params.output_buffer_id = BufferId::Q_PROJ; // In-place

            graph.addNode(prefix + "q_norm",
                          ComputeStageFactory::createQKNorm(q_norm_params),
                          device);

            if (has_qkv_proj)
            {
                graph.addDependency(prefix + "q_norm", prefix + "qkv_proj");
            }

            // K norm: normalize each K head independently
            QKNormStage::Params k_norm_params;
            k_norm_params.input = buffers.K;
            k_norm_params.output = buffers.K; // In-place
            k_norm_params.gamma = layer.k_norm;
            k_norm_params.n_heads = local_n_kv_heads;
            k_norm_params.head_dim = config_.head_dim;
            k_norm_params.eps = config_.rms_norm_eps;
            k_norm_params.seq_len = total_tokens;
            k_norm_params.device_id = device;
            k_norm_params.input_buffer_id = BufferId::K_PROJ;
            k_norm_params.output_buffer_id = BufferId::K_PROJ; // In-place

            graph.addNode(prefix + "k_norm",
                          ComputeStageFactory::createQKNorm(k_norm_params),
                          device);

            if (has_qkv_proj)
            {
                graph.addDependency(prefix + "k_norm", prefix + "qkv_proj");
            }
        }

        // Stage 3: RoPE on Q and K
        {
            // For batched execution, pass the full position_ids array
            // This enables correct per-token position encoding for variable-length sequences
            // Fallback pos_offset is still used for compatibility with single-sequence execution
            int pos_offset = position_ids ? position_ids[0] : 0;

            RoPEStage::Params rope_params;
            rope_params.device_id = device; // Use graph's target device for kernel dispatch
            rope_params.Q = buffers.Q;
            rope_params.K = buffers.K;
            rope_params.n_heads = local_n_heads;       // Use local head count for TP
            rope_params.n_kv_heads = local_n_kv_heads; // Use local KV head count for TP
            rope_params.head_dim = config_.head_dim;
            rope_params.pos_offset = pos_offset;
            rope_params.position_ids = position_ids; // Pass full array for batched execution
            rope_params.theta_base = config_.rope_theta;
            rope_params.seq_len = total_tokens; // Use total_tokens = batch_size * seq_len
            rope_params.q_buffer_id = BufferId::Q_PROJ;
            rope_params.k_buffer_id = BufferId::K_PROJ;

            // Partial rotary: some models (Qwen3.5) only rotate a fraction of head_dim
            rope_params.partial_rotary_factor = config_.partial_rotary_factor;

            // RoPE-on-read: skip K in RoPE stage; it will be applied during attention
            rope_params.skip_k = config_.rope_on_read;

            graph.addNode(prefix + "rope",
                          ComputeStageFactory::createRoPE(rope_params),
                          device);

            // If QK norms are present, RoPE depends on norms (which depend on qkv_proj)
            if (layer.q_norm && layer.k_norm)
            {
                graph.addDependency(prefix + "rope", prefix + "q_norm");
                graph.addDependency(prefix + "rope", prefix + "k_norm");
            }
            else if (has_qkv_proj)
            {
                graph.addDependency(prefix + "rope", prefix + "qkv_proj");
            }
        }

        // Stage 4: Attention computation with KV cache integration
        // NOTE: Decomposed attention (Phase 9) is now the ONLY supported path.
        // Legacy AttentionWithKVCacheStage has been removed (Phase 7 cleanup).
        std::string wo_producer_node;

        // Phase 9 Decomposed Path: KVCacheAppendStage + AttentionComputeStage
        if (kv_cache)
        {
            // For batched execution, K/V are [batch_size * seq_len, kv_dim]
            // Each sequence's K/V is appended to its own seq_idx in the cache
            int total_tokens = batch_size * seq_len;

            KVCacheAppendStage::Params kv_append_params;
            kv_append_params.device_id = device;
            kv_append_params.K = buffers.K;
            kv_append_params.k_buffer_id = BufferId::K_PROJ;

            kv_append_params.V = buffers.V;
            kv_append_params.v_buffer_id = BufferId::V_PROJ;
            kv_append_params.kv_cache = kv_cache;
            // For PP stages: map global layer index to local KV cache index
            kv_append_params.layer_idx = layer_idx - config_.pp_layer_offset;
            kv_append_params.seq_idx = 0; // Starting seq_idx
            kv_append_params.num_tokens = total_tokens;
            kv_append_params.batch_size = batch_size; // Phase 3: Per-sequence append
            kv_append_params.seq_len = seq_len;       // Phase 3: Tokens per sequence

            // Phase 5.4: VNNI-safe Q16 KV cache quantization parameters
            kv_append_params.kv_cache_scale_k = config_.kv_cache_scale_k;
            kv_append_params.kv_cache_scale_v = config_.kv_cache_scale_v;
            kv_append_params.head_dim = config_.head_dim;
            kv_append_params.turboquant_ctx = config_.turboquant_ctx;
            kv_append_params.kv_rotation = config_.kv_rotation;

            graph.addNode(prefix + "kv_append",
                          ComputeStageFactory::createKVCacheAppend(kv_append_params),
                          device);

            if (!config_.rope_on_read)
            {
                // Standard mode: K needs RoPE before caching
                graph.addDependency(prefix + "kv_append", prefix + "rope");
            }
            else
            {
                // RoPE-on-read: K stored pre-RoPE, depend on QKV projection
                if (has_qkv_proj)
                    graph.addDependency(prefix + "kv_append", prefix + "qkv_proj");
            }
        }

        // For Hybrid mode attention path selection:
        // - Decomposed attention: Use FP32 K_rope/V_dequant for best precision
        // - Fused attention: Use Q8_1 K/V (kernel only supports Q8_1 K/V currently)
        // KV cache always gets FP32 K_rope (stored separately from attention path)
        ITensor *K_for_attn = buffers.K;
        ITensor *V_for_attn = buffers.V;

        int total_query_tokens = batch_size * seq_len;
        int kv_len = total_query_tokens; // Static hint for mode detection (actual queried at runtime)
        bool use_gather_stage = false;

        // For attention K/V source:
        // - Prefill (cached_tokens == 0): Use projected K/V directly
        // - Decode (cached_tokens > 0 and batch_size == 1): Use cache (single-sequence only)
        // - Batched decode (cached_tokens > 0 and batch_size > 1): Gather K/V from multiple cache slots
        // Map global layer index to local KV cache index for PP stages
        int kv_local_layer = layer_idx - config_.pp_layer_offset;

        if (kv_cache)
        {
            int cached_tokens = kv_cache->get_cached_tokens(kv_local_layer, 0);
            if (cached_tokens > 0 && batch_size == 1)
            {
                // Single-sequence decode: read K/V from cache
                // Use ITensor* directly (works for both CPU and GPU caches)
                K_for_attn = kv_cache->get_k(kv_local_layer, 0);
                V_for_attn = kv_cache->get_v(kv_local_layer, 0);
                kv_len = cached_tokens;
                LOG_TRACE("[Qwen2Graph] Layer " << layer_idx << " (local=" << kv_local_layer << ") using cached K/V (decode mode)");
            }
            else if (cached_tokens > 0 && batch_size > 1)
            {
                // Batched decode: gather K/V from multiple cache slots
                if (buffers.gathered_K && buffers.gathered_V)
                {
                    use_gather_stage = true;
                    K_for_attn = buffers.gathered_K;
                    V_for_attn = buffers.gathered_V;
                    // kv_len will be updated by gather stage; use cache max for now
                    kv_len = cached_tokens; // Approximate - actual max determined at gather
                    LOG_TRACE("[Qwen2Graph] Layer " << layer_idx << " using gathered K/V (batched decode mode)");
                }
                else
                {
                    // Fallback: use projected K/V if gather buffers not provided
                    LOG_WARN("[Qwen2Graph] Layer " << layer_idx
                                                   << " batched decode but no gather buffers - using projected K/V");
                }
            }
            else
            {
                // Prefill or batched prefill: use projected K/V directly
                // KV cache will be populated but attention uses fresh projections
                LOG_TRACE("[Qwen2Graph] Layer " << layer_idx << " using projected K/V (prefill/batch mode)");
            }
        }

        // Add KVCacheGatherStage if batched decode
        if (use_gather_stage)
        {
            KVCacheGatherStage::Params gather_params;
            gather_params.kv_cache = kv_cache;
            // For PP stages: map global layer index to local KV cache index
            gather_params.layer_idx = layer_idx - config_.pp_layer_offset;
            gather_params.batch_size = batch_size;
            gather_params.out_K = buffers.gathered_K;
            gather_params.out_V = buffers.gathered_V;
            // Note: out_max_kv_len and out_per_seq_kv_lens can be retrieved from stage after execute

            graph.addNode(prefix + "kv_gather",
                          ComputeStageFactory::createKVCacheGather(gather_params),
                          device);

            // Gather depends on append (must append new tokens before gathering full history)
            graph.addDependency(prefix + "kv_gather", prefix + "kv_append");
        }

        // Decomposed Path: Attention -> Wo projection
        {
            AttentionMode mode = detect_attention_mode(batch_size, seq_len, kv_len);
            LOG_TRACE("[Qwen2Graph] Layer " << layer_idx
                                            << " attention mode: " << attention_mode_name(mode)
                                            << " (batch_size=" << batch_size << ", seq_len=" << seq_len << ", kv_len=" << kv_len << ")");

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
            attn_params.head_start = config_.head_start;
            // Only set gqa_n_rep when KV heads are REPLICATED (each device has all KV heads).
            // When KV is column-parallel (each device gets local_n_kv_heads < global n_kv_heads),
            // the local ratio (local_n_heads / local_n_kv_heads) is correct.
            if (local_n_kv_heads == config_.n_kv_heads && config_.n_kv_heads > 0 && local_n_heads != config_.n_heads)
            {
                attn_params.gqa_n_rep = config_.n_heads / config_.n_kv_heads;
            }
            else
            {
                attn_params.gqa_n_rep = 0; // use local ratio
            }
            attn_params.causal = true;
            attn_params.window_size = -1;
            attn_params.attention_mode = mode;
            attn_params.auto_detect_mode = true;
            attn_params.workspace_scores = buffers.workspace_scores;
            attn_params.workspace_context = buffers.workspace_context;
            attn_params.workspace_mask = buffers.workspace_mask;
            attn_params.kv_cache = kv_cache;
            attn_params.layer_idx = layer_idx - config_.pp_layer_offset;
            attn_params.read_kv_from_cache = device.is_gpu() &&
                                             (!kv_cache || kv_cache->precision() != ActivationPrecision::Q8_1) &&
                                             (!kv_cache || (kv_cache->precision() != ActivationPrecision::TQ8 &&
                                                            kv_cache->precision() != ActivationPrecision::TQ4));
            attn_params.position_offset = position_ids ? position_ids[0] : 0;
            attn_params.mpi_ctx = mpi_ctx_.get();
            attn_params.device_id = device;
            attn_params.q_buffer_id = BufferId::Q_PROJ;
            attn_params.output_buffer_id = BufferId::ATTN_OUTPUT;
            // GPU flash attention doesn't use score/context workspace buffers —
            // only register them for CPU attention to avoid wasting 544 MB VRAM.
            if (!device.is_gpu())
            {
                attn_params.workspace_scores_buffer_id = BufferId::ATTN_SCORES_WORKSPACE;
                attn_params.workspace_context_buffer_id = BufferId::ATTN_CONTEXT_WORKSPACE;
            }
            attn_params.turboquant_ctx = config_.turboquant_ctx;
            attn_params.kv_rotation = config_.kv_rotation;

            // RoPE-on-read: apply RoPE to K in the attention stage
            // (fused with TQ4 dequant for decode, in-place for FP32 prefill)
            if (config_.rope_on_read)
            {
                attn_params.apply_rope_to_k = true;
                attn_params.rope_theta = config_.rope_theta;
                attn_params.partial_rotary_factor = config_.partial_rotary_factor;
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

            LOG_DEBUG("[Qwen2Graph] Using decomposed attention path");
        }

        // Stage 5: Output projection (Wo)
        if (layer.wo)
        {
            int wo_n = static_cast<int>(layer.wo->shape()[0]);
            int wo_k = static_cast<int>(layer.wo->shape()[1]);

            LOG_DEBUG("[Qwen2Graph] Layer " << layer_idx << " Wo dims: wo_n=" << wo_n
                                            << " wo_k=" << wo_k
                                            << " wo_shape=[" << layer.wo->shape()[0] << "," << layer.wo->shape()[1] << "]"
                                            << " attn_output_shape=" << buffers.attn_output->shape()[0] << "x" << buffers.attn_output->shape()[1]);

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

            graph.addDependency(wo_producer_node, prefix + "attention");
        }

        // Common AllReduce for Wo
        if (layer.wo && !wo_producer_node.empty())
        {
            bool wo_is_sharded = isRowParallelSharded(layer.wo);

            if (wo_is_sharded && needsTPAllreduce())
            {
                size_t allreduce_count = static_cast<size_t>(total_tokens) * static_cast<size_t>(config_.d_model);

                TensorBase *allreduce_buffer = buffers.attn_proj;
                BufferId wo_allreduce_bid = BufferId::ATTN_PROJ;

                std::string stage_name = prefix + "wo_allreduce";
                auto allreduce_stage = createTPAllreduceStage(
                    allreduce_buffer, allreduce_count, device, layer_idx, /*is_attention=*/true, stage_name,
                    wo_allreduce_bid);

                if (allreduce_stage)
                {
                    graph.addNode(stage_name, std::move(allreduce_stage), device);
                    graph.addDependency(stage_name, wo_producer_node);
                    wo_producer_node = stage_name;

                    LOG_TRACE("[Qwen2Graph] Layer " << layer_idx
                                                    << " Wo: row-parallel sharded, adding allreduce");
                }
            }
        }

        // Attention residual is now fused into FusedResidualNormStage in buildFFNGraph
        // (for non-first layers) or handled by the first layer's standalone RMSNorm path.

        graph.setTerminalNode(wo_producer_node);
        return graph;
    }

} // namespace llaminar2
