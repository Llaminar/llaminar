/**
 * @file Qwen2Graph.cpp
 * @brief Implementation of Qwen2 compute graph builder
 * @author David Sanftenberg
 * @date December 2025
 *
 * This file implements Qwen2Graph, the unified graph builder for Qwen2
 * architecture models. It combines the functionality previously split
 * across Qwen2LayerExecutor and Qwen2ModelExecutor.
 */

#include "Qwen2Graph.h"
#include "Qwen2Pipeline.h"
#include "../../utils/Logger.h"
#include "../../utils/DebugEnv.h"
#include "../../backends/ComputeBackend.h"
#include "../../tensors/TensorSlice.h"
#include "../../tensors/Tensors.h"
#include "../../utils/MPIContext.h"
#include "../MPIStrategy.h"
#include <chrono>
#include <stdexcept>

namespace llaminar2
{

    // =============================================================================
    // Helper Functions
    // =============================================================================

    /**
     * @brief Check if a weight tensor is sharded row-parallel
     */
    static bool isRowParallelSharded(const TensorBase *weight)
    {
        if (!weight)
            return false;

        const auto *slice = dynamic_cast<const TensorSlice *>(weight);
        bool result = slice && slice->is_row_parallel();
        LOG_DEBUG("[isRowParallelSharded] weight=" << weight << " is_slice=" << (slice != nullptr)
                                                   << " is_row_parallel=" << (slice ? slice->is_row_parallel() : false)
                                                   << " result=" << result);
        return result;
    }

    /**
     * @brief Get MPI communicator as void* for AllreduceStage
     */
    static void *getMPIComm(const MPIContext *mpi_ctx)
    {
        if (!mpi_ctx)
        {
            LOG_WARN("[getMPIComm] mpi_ctx is null!");
            return nullptr;
        }
        MPI_Comm comm = mpi_ctx->comm();
        void *result = static_cast<void *>(comm);
        LOG_DEBUG("[getMPIComm] mpi_ctx=" << mpi_ctx << " comm=" << comm << " result=" << result);
        return result;
    }

    // =============================================================================
    // Constructors
    // =============================================================================

    Qwen2Graph::Qwen2Graph(std::shared_ptr<ModelContext> model_ctx,
                           std::shared_ptr<MPIContext> mpi_ctx,
                           const Qwen2GraphConfig &config)
        : config_(config),
          model_ctx_(std::move(model_ctx)),
          mpi_ctx_(std::move(mpi_ctx))
    {
        LOG_INFO("[Qwen2Graph] Initializing (full): n_layers=" << config_.n_layers
                                                               << " d_model=" << config_.d_model
                                                               << " vocab_size=" << config_.vocab_size
                                                               << " d_ff=" << config_.d_ff
                                                               << " ffn_column_parallel=" << config_.ffn_column_parallel
                                                               << " n_heads=" << config_.n_heads
                                                               << " n_kv_heads=" << config_.n_kv_heads);

        // Configure underlying GraphExecutor
        GraphExecutorConfig exec_config = config.executor_config;
        exec_config.default_device = config.default_device;
        exec_config.enable_profiling = config.enable_profiling;
        exec_config.enable_validation = config.enable_validation;

        // Parse execution mode from environment
        const auto &env = debugEnv();
        if (env.execution.execution_mode == "parallel")
        {
            exec_config.mode = ExecutionMode::PARALLEL;
        }
        else if (env.execution.execution_mode == "pipelined")
        {
            exec_config.mode = ExecutionMode::PIPELINED;
        }
        else
        {
            exec_config.mode = ExecutionMode::SEQUENTIAL;
        }

        executor_ = GraphExecutor(exec_config);

        // Propagate MPI rank to executor for stage dumping
        if (mpi_ctx_)
        {
            executor_.setMPIRank(mpi_ctx_->rank());
        }

        LOG_INFO("[Qwen2Graph] Initialized (full) with " << config_.n_layers
                                                         << " layers, precision="
                                                         << (config_.activation_precision == ActivationPrecision::Q8_1 ? "Q8_1" : "FP32")
                                                         << ", mode=" << executionModeName(exec_config.mode));
    }

    Qwen2Graph::Qwen2Graph(const Qwen2GraphConfig &config,
                           std::shared_ptr<MPIContext> mpi_ctx)
        : config_(config),
          model_ctx_(nullptr),
          mpi_ctx_(std::move(mpi_ctx))
    {
        // This constructor is for layer-level operations only (no model context needed)
        LOG_INFO("[Qwen2Graph] Initializing (layer-only): d_model=" << config_.d_model
                                                                    << " d_ff=" << config_.d_ff
                                                                    << " ffn_column_parallel=" << config_.ffn_column_parallel
                                                                    << " n_heads=" << config_.n_heads
                                                                    << " n_kv_heads=" << config_.n_kv_heads);

        // Configure underlying GraphExecutor
        GraphExecutorConfig exec_config = config.executor_config;
        exec_config.default_device = config.default_device;
        exec_config.enable_profiling = config.enable_profiling;
        exec_config.enable_validation = config.enable_validation;

        // Parse execution mode from environment
        const auto &env = debugEnv();
        if (env.execution.execution_mode == "parallel")
        {
            exec_config.mode = ExecutionMode::PARALLEL;
        }
        else if (env.execution.execution_mode == "pipelined")
        {
            exec_config.mode = ExecutionMode::PIPELINED;
        }
        else
        {
            exec_config.mode = ExecutionMode::SEQUENTIAL;
        }

        executor_ = GraphExecutor(exec_config);

        // Propagate MPI rank to executor for stage dumping
        if (mpi_ctx_)
        {
            executor_.setMPIRank(mpi_ctx_->rank());
        }

        LOG_DEBUG("[Qwen2Graph] Initialized (layer-only), mode=" << executionModeName(exec_config.mode));
    }

    // =============================================================================
    // Device Context Management
    // =============================================================================

    IDeviceContext *Qwen2Graph::getDeviceContext(int device_idx)
    {
        auto it = device_contexts_.find(device_idx);
        if (it != device_contexts_.end())
        {
            return it->second.get();
        }

        // Create new context
        auto ctx = IDeviceContext::create(device_idx);
        if (!ctx)
        {
            LOG_ERROR("[Qwen2Graph] Failed to create device context for device " << device_idx);
            return nullptr;
        }

        IDeviceContext *raw_ptr = ctx.get();
        device_contexts_[device_idx] = std::move(ctx);
        return raw_ptr;
    }

    // =============================================================================
    // Model-Level Graph Building
    // =============================================================================

    ComputeGraph Qwen2Graph::buildFullForwardGraph(
        const Qwen2ForwardInput &input,
        Qwen2ForwardOutput &output)
    {
        LOG_DEBUG("[Qwen2Graph] Building full forward graph: "
                  << "batch_size=" << input.batch_size
                  << ", seq_len=" << input.seq_len);

        if (!weights_.embedding_table || !weights_.final_norm || !weights_.lm_head)
        {
            LOG_ERROR("[Qwen2Graph] Weights not set! Call setWeights() first.");
            throw std::runtime_error("Qwen2Graph weights not initialized");
        }

        if (!buffers_.current_hidden || !buffers_.logits)
        {
            LOG_ERROR("[Qwen2Graph] Buffers not set! Call setBuffers() first.");
            throw std::runtime_error("Qwen2Graph buffers not initialized");
        }

        // Store current dimensions
        current_batch_size_ = input.batch_size;
        current_seq_len_ = input.seq_len;

        int device_idx = config_.default_device;
        int total_tokens = input.batch_size * input.seq_len;

        ComputeGraph graph;

        // -------------------------------------------------------------------------
        // Stage 1: Embedding Lookup
        // -------------------------------------------------------------------------
        EmbeddingStage::Params embed_params;
        embed_params.embed_table = weights_.embedding_table;
        embed_params.token_ids = input.token_ids;
        embed_params.output = buffers_.current_hidden;
        embed_params.num_tokens = total_tokens;
        embed_params.d_model = config_.d_model;
        embed_params.vocab_size = config_.vocab_size;
        embed_params.device_idx = device_idx;

        graph.addNode("embedding",
                      ComputeStageFactory::createEmbedding(embed_params),
                      device_idx);

        // -------------------------------------------------------------------------
        // Stage 2: Transformer Layers (as placeholder nodes)
        // -------------------------------------------------------------------------
        std::string prev_node = "embedding";

        // Build position IDs
        position_ids_buffer_ = buildPositionIds(input.seq_len, input.batch_size, input.position_offset);

        // Each layer is a placeholder node - actual execution uses executeLayer()
        for (int layer = 0; layer < config_.n_layers; ++layer)
        {
            std::string layer_name = "layer_" + std::to_string(layer);

            // Add node with nullptr stage - serves as a sequencing marker
            graph.addNode(layer_name, nullptr, device_idx);
            graph.addDependency(layer_name, prev_node);

            prev_node = layer_name;
        }

        // -------------------------------------------------------------------------
        // Stage 3: Final RMSNorm
        // -------------------------------------------------------------------------
        addFinalNormToGraph(graph, buffers_.current_hidden, prev_node, total_tokens, device_idx);
        prev_node = "final_norm";

        // -------------------------------------------------------------------------
        // Stage 4: LM Head
        // -------------------------------------------------------------------------
        LMHeadStage::Params lm_params;
        lm_params.hidden_states = buffers_.current_hidden;
        lm_params.lm_head_weight = weights_.lm_head;
        lm_params.logits = buffers_.logits;
        lm_params.seq_len = total_tokens;
        lm_params.d_model = config_.d_model;
        lm_params.vocab_size = config_.vocab_size;
        lm_params.bias = nullptr; // Qwen2 has no LM head bias
        lm_params.device_idx = device_idx;

        graph.addNode("lm_head",
                      ComputeStageFactory::createLMHead(lm_params),
                      device_idx);
        graph.addDependency("lm_head", prev_node);

        // Set output
        output.logits = buffers_.logits;

        LOG_DEBUG("[Qwen2Graph] Built full forward graph with "
                  << graph.size() << " nodes");

        return graph;
    }

    ComputeGraph Qwen2Graph::buildEmbeddingGraph(
        const Qwen2ForwardInput &input,
        TensorBase *output_hidden)
    {
        LOG_DEBUG("[Qwen2Graph] Building embedding graph for "
                  << (input.batch_size * input.seq_len) << " tokens");

        ComputeGraph graph;

        EmbeddingStage::Params params;
        params.embed_table = weights_.embedding_table;
        params.token_ids = input.token_ids;
        params.output = output_hidden;
        params.num_tokens = input.batch_size * input.seq_len;
        params.d_model = config_.d_model;
        params.vocab_size = config_.vocab_size;
        params.device_idx = config_.default_device;

        graph.addNode("embedding",
                      ComputeStageFactory::createEmbedding(params),
                      config_.default_device);

        return graph;
    }

    ComputeGraph Qwen2Graph::buildTransformerLayersGraph(
        TensorBase *input_hidden,
        IUnifiedKVCache *kv_cache,
        const int *position_ids,
        int device_idx)
    {
        LOG_DEBUG("[Qwen2Graph] Building transformer layers graph: "
                  << config_.n_layers << " layers");

        ComputeGraph graph;
        std::string prev_node;

        // Build placeholder nodes for layer sequencing
        for (int layer = 0; layer < config_.n_layers; ++layer)
        {
            std::string layer_name = "layer_" + std::to_string(layer);

            // Add node with nullptr stage - placeholder for sequencing
            graph.addNode(layer_name, nullptr, device_idx);

            if (!prev_node.empty())
            {
                graph.addDependency(layer_name, prev_node);
            }

            prev_node = layer_name;
        }

        return graph;
    }

    ComputeGraph Qwen2Graph::buildLayerGraph(
        int layer_idx,
        TensorBase *input_hidden,
        IUnifiedKVCache *kv_cache,
        const int *position_ids,
        int device_idx)
    {
        LOG_DEBUG("[Qwen2Graph] Building layer " << layer_idx << " graph");

        ComputeGraph graph;
        graph.addNode("layer_" + std::to_string(layer_idx), nullptr, device_idx);

        return graph;
    }

    ComputeGraph Qwen2Graph::buildLMHeadGraph(
        TensorBase *hidden_states,
        TensorBase *output_logits,
        int device_idx)
    {
        LOG_DEBUG("[Qwen2Graph] Building LM head graph");

        ComputeGraph graph;
        int total_tokens = current_batch_size_ * current_seq_len_;

        // Final RMSNorm
        RMSNormStage::Params norm_params;
        norm_params.input = hidden_states;
        norm_params.output = hidden_states; // In-place norm
        norm_params.gamma = weights_.final_norm;
        norm_params.eps = config_.rms_norm_eps;
        norm_params.seq_len = total_tokens;
        norm_params.device_idx = device_idx;

        graph.addNode("final_norm",
                      ComputeStageFactory::createRMSNorm(norm_params),
                      device_idx);

        // LM Head projection
        LMHeadStage::Params lm_params;
        lm_params.hidden_states = hidden_states;
        lm_params.lm_head_weight = weights_.lm_head;
        lm_params.logits = output_logits;
        lm_params.seq_len = total_tokens;
        lm_params.d_model = config_.d_model;
        lm_params.vocab_size = config_.vocab_size;
        lm_params.bias = nullptr;
        lm_params.device_idx = device_idx;

        graph.addNode("lm_head",
                      ComputeStageFactory::createLMHead(lm_params),
                      device_idx);
        graph.addDependency("lm_head", "final_norm");

        return graph;
    }

    // =============================================================================
    // Layer-Level Graph Building
    // =============================================================================

    ComputeGraph Qwen2Graph::buildAttentionGraph(
        const Qwen2LayerWeights &layer,
        Qwen2ActivationBuffers &buffers,
        int layer_idx,
        int seq_len,
        IUnifiedKVCache *kv_cache,
        const int *position_ids,
        int device_idx)
    {
        ComputeGraph graph;
        const auto &env = debugEnv();
        std::string prefix = "layer" + std::to_string(layer_idx) + "_";

        LOG_DEBUG("[buildAttentionGraph] layer_idx=" << layer_idx << " seq_len=" << seq_len
                                                     << " layer.wq=" << static_cast<const void *>(layer.wq)
                                                     << " layer.wo=" << layer.wo << " world_size="
                                                     << (mpi_ctx_ ? mpi_ctx_->world_size() : 1));

        // Determine backend type for stage creation
        auto &dm = DeviceManager::instance();
        ComputeBackendType backend = ComputeBackendType::CPU;
        if (static_cast<size_t>(device_idx) < dm.devices().size())
        {
            backend = dm.devices()[device_idx].type;
        }

        // Stage 1: Pre-attention RMSNorm
        if (env.execution.exec_rmsnorm)
        {
            RMSNormStage::Params attn_norm_params;
            attn_norm_params.input = buffers.current_hidden;
            attn_norm_params.output = buffers.normalized;
            attn_norm_params.gamma = layer.attn_norm;
            attn_norm_params.eps = config_.rms_norm_eps;
            attn_norm_params.seq_len = seq_len;

            graph.addNode(prefix + "attn_norm",
                          ComputeStageFactory::createRMSNorm(attn_norm_params),
                          device_idx);
        }

        // Stage 2: Q/K/V projections using FusedQKVGEMMStage
        if (env.execution.exec_gemm && layer.wq && layer.wk && layer.wv)
        {
            LOG_DEBUG("[Qwen2Graph] Using FusedQKVGEMMStage");

            int k = config_.d_model;
            int q_n = static_cast<int>(layer.wq->shape()[0]);
            int k_n = static_cast<int>(layer.wk->shape()[0]);
            int v_n = static_cast<int>(layer.wv->shape()[0]);

            // Extract bias pointers
            const float *q_bias_ptr = nullptr;
            const float *k_bias_ptr = nullptr;
            const float *v_bias_ptr = nullptr;

            if (layer.q_bias)
            {
                auto *q_bias_fp32 = dynamic_cast<FP32Tensor *>(layer.q_bias);
                if (q_bias_fp32)
                    q_bias_ptr = q_bias_fp32->data();
            }
            if (layer.k_bias)
            {
                auto *k_bias_fp32 = dynamic_cast<FP32Tensor *>(layer.k_bias);
                if (k_bias_fp32)
                    k_bias_ptr = k_bias_fp32->data();
            }
            if (layer.v_bias)
            {
                auto *v_bias_fp32 = dynamic_cast<FP32Tensor *>(layer.v_bias);
                if (v_bias_fp32)
                    v_bias_ptr = v_bias_fp32->data();
            }

            FusedQKVGEMMStage::Params qkv_params;
            qkv_params.input = buffers.normalized;
            qkv_params.m = seq_len;
            qkv_params.k = k;
            qkv_params.wq = layer.wq;
            qkv_params.output_q = buffers.Q;
            qkv_params.n_q = q_n;
            qkv_params.bias_q = q_bias_ptr;
            qkv_params.wk = layer.wk;
            qkv_params.output_k = buffers.K;
            qkv_params.n_k = k_n;
            qkv_params.bias_k = k_bias_ptr;
            qkv_params.wv = layer.wv;
            qkv_params.output_v = buffers.V;
            qkv_params.n_v = v_n;
            qkv_params.bias_v = v_bias_ptr;

            graph.addNode(prefix + "qkv_proj",
                          ComputeStageFactory::createFusedQKVGEMM(qkv_params),
                          device_idx);

            if (env.execution.exec_rmsnorm)
            {
                graph.addDependency(prefix + "qkv_proj", prefix + "attn_norm");
            }
        }

        // Stage 3: RoPE on Q and K
        if (env.execution.exec_rope)
        {
            int pos_offset = position_ids ? position_ids[0] : 0;

            RoPEStage::Params rope_params;
            rope_params.Q = buffers.Q;
            rope_params.K = buffers.K;
            rope_params.n_heads = config_.n_heads;
            rope_params.n_kv_heads = config_.n_kv_heads;
            rope_params.head_dim = config_.head_dim;
            rope_params.pos_offset = pos_offset;
            rope_params.theta_base = config_.rope_theta;

            graph.addNode(prefix + "rope",
                          ComputeStageFactory::createRoPE(rope_params),
                          device_idx);

            if (env.execution.exec_gemm)
            {
                graph.addDependency(prefix + "rope", prefix + "qkv_proj");
            }
        }

        // Stage 4: Attention computation with KV cache integration
        if (env.execution.exec_attention)
        {
            if (config_.use_decomposed_attention)
            {
                // Phase 9 Decomposed Path: KVCacheAppendStage + AttentionComputeStage
                if (kv_cache)
                {
                    KVCacheAppendStage::Params kv_append_params;
                    kv_append_params.K = buffers.K;
                    kv_append_params.V = buffers.V;
                    kv_append_params.kv_cache = kv_cache;
                    kv_append_params.layer_idx = layer_idx;
                    kv_append_params.seq_idx = 0;
                    kv_append_params.num_tokens = seq_len;

                    graph.addNode(prefix + "kv_append",
                                  ComputeStageFactory::createKVCacheAppend(kv_append_params),
                                  device_idx);

                    if (env.execution.exec_rope)
                    {
                        graph.addDependency(prefix + "kv_append", prefix + "rope");
                    }
                    else if (env.execution.exec_gemm)
                    {
                        graph.addDependency(prefix + "kv_append", prefix + "qkv_proj");
                    }
                }

                TensorBase *K_for_attn = buffers.K;
                TensorBase *V_for_attn = buffers.V;
                int kv_len = seq_len; // Static hint for mode detection (actual queried at runtime)

                if (kv_cache)
                {
                    K_for_attn = kv_cache->get_k_base(layer_idx, 0);
                    V_for_attn = kv_cache->get_v_base(layer_idx, 0);
                    int cached_tokens = kv_cache->get_cached_tokens(layer_idx, 0);
                    // Use static cached_tokens for initial mode hint
                    // Actual kv_len will be queried dynamically at execution time
                    kv_len = (cached_tokens > 0) ? cached_tokens : seq_len;
                }

                AttentionMode mode = detect_attention_mode(1, seq_len, kv_len);
                LOG_TRACE("[Qwen2Graph] Layer " << layer_idx
                                                << " attention mode: " << attention_mode_name(mode)
                                                << " (seq_len=" << seq_len << ", kv_len=" << kv_len << ")");

                AttentionComputeStage::Params attn_params;
                attn_params.Q = buffers.Q;
                attn_params.K = K_for_attn;
                attn_params.V = V_for_attn;
                attn_params.output = buffers.attn_output;
                attn_params.batch_size = 1;
                attn_params.seq_len = seq_len;
                attn_params.kv_len = kv_len; // Static hint, actual queried from kv_cache at runtime
                attn_params.n_heads = config_.n_heads;
                attn_params.n_kv_heads = config_.n_kv_heads;
                attn_params.head_dim = config_.head_dim;
                attn_params.causal = true;
                attn_params.window_size = -1;
                attn_params.attention_mode = mode;
                attn_params.auto_detect_mode = true; // Re-detect at runtime with actual kv_len
                attn_params.workspace_scores = buffers.workspace_scores;
                attn_params.workspace_context = buffers.workspace_context;
                attn_params.workspace_mask = buffers.workspace_mask;
                attn_params.kv_cache = kv_cache; // Pass for dynamic kv_len query at execution
                attn_params.layer_idx = layer_idx;
                attn_params.position_offset = position_ids ? position_ids[0] : 0;
                attn_params.mpi_ctx = mpi_ctx_.get();
                attn_params.device_idx = device_idx;

                graph.addNode(prefix + "attention",
                              ComputeStageFactory::createAttentionCompute(attn_params),
                              device_idx);

                if (kv_cache)
                {
                    graph.addDependency(prefix + "attention", prefix + "kv_append");
                }
                else if (env.execution.exec_rope)
                {
                    graph.addDependency(prefix + "attention", prefix + "rope");
                }
                else if (env.execution.exec_gemm)
                {
                    graph.addDependency(prefix + "attention", prefix + "qkv_proj");
                }

                LOG_DEBUG("[Qwen2Graph] Using decomposed attention path (Phase 9)");
            }
            else
            {
                // Legacy Path: AttentionWithKVCacheStage
                AttentionWithKVCacheStage::Params attn_params;
                attn_params.Q = buffers.Q;
                attn_params.K = buffers.K;
                attn_params.V = buffers.V;
                attn_params.output = buffers.attn_output;
                attn_params.kv_cache = kv_cache;
                attn_params.layer_idx = layer_idx;
                attn_params.mode = AttentionWithKVCacheStage::Mode::AUTO;
                attn_params.batch_size = 1;
                attn_params.seq_len = seq_len;
                attn_params.n_heads = config_.n_heads;
                attn_params.n_kv_heads = config_.n_kv_heads;
                attn_params.head_dim = config_.head_dim;
                attn_params.causal = true;
                attn_params.window_size = -1;
                attn_params.mpi_ctx = mpi_ctx_;
                MPIStrategy mpi_strategy = MPIStrategy::None;
                if (mpi_ctx_ && mpi_ctx_->world_size() > 1)
                {
                    mpi_strategy = MPIStrategy::TensorParallel;
                }
                attn_params.mpi_strategy = static_cast<int>(mpi_strategy);
                attn_params.workspace_scores = buffers.workspace_scores;
                attn_params.workspace_context = buffers.workspace_context;
                attn_params.workspace_mask = buffers.workspace_mask;
                attn_params.sequence_lengths = nullptr;
                attn_params.position_offset = position_ids ? position_ids[0] : 0;

                graph.addNode(prefix + "attention",
                              ComputeStageFactory::createAttentionWithKVCache(attn_params),
                              device_idx);

                if (env.execution.exec_rope)
                {
                    graph.addDependency(prefix + "attention", prefix + "rope");
                }
                else if (env.execution.exec_gemm)
                {
                    graph.addDependency(prefix + "attention", prefix + "q_proj");
                    graph.addDependency(prefix + "attention", prefix + "k_proj");
                    graph.addDependency(prefix + "attention", prefix + "v_proj");
                }
            }
        }

        // Stage 5: Output projection (Wo)
        if (env.execution.exec_gemm && layer.wo)
        {
            int wo_n = static_cast<int>(layer.wo->shape()[0]);
            int wo_k = static_cast<int>(layer.wo->shape()[1]);

            graph.addNode(prefix + "wo_proj",
                          ComputeStageFactory::createGEMM(
                              GEMMStage::Params{
                                  .A = buffers.attn_output,
                                  .B = layer.wo,
                                  .C = buffers.attn_proj,
                                  .m = seq_len,
                                  .n = wo_n,
                                  .k = wo_k,
                                  .alpha = 1.0f,
                                  .beta = 0.0f,
                                  .transpose_B = false}),
                          device_idx);

            bool wo_is_sharded = isRowParallelSharded(layer.wo);
            bool has_multi_rank = mpi_ctx_ && mpi_ctx_->world_size() > 1;

            if (wo_is_sharded && has_multi_rank)
            {
                graph.addNode(prefix + "wo_allreduce",
                              ComputeStageFactory::createAllreduce(
                                  AllreduceStage::Params{
                                      buffers.attn_proj,
                                      getMPIComm(mpi_ctx_.get())}),
                              device_idx);

                graph.addDependency(prefix + "wo_allreduce", prefix + "wo_proj");

                LOG_TRACE("[Qwen2Graph] Layer " << layer_idx
                                                << " Wo: row-parallel sharded, adding allreduce");
            }

            if (env.execution.exec_attention)
            {
                graph.addDependency(prefix + "wo_proj", prefix + "attention");
            }
        }

        // Stage 6: Residual connection
        if (env.execution.exec_residual)
        {
            ResidualAddStage::Params res_params;
            res_params.input = buffers.attn_proj;
            res_params.residual = buffers.current_hidden;
            res_params.output = buffers.current_hidden;
            res_params.num_elements = static_cast<size_t>(seq_len) * static_cast<size_t>(config_.d_model);

            graph.addNode(prefix + "attn_residual",
                          ComputeStageFactory::createResidualAdd(res_params),
                          device_idx);

            if (env.execution.exec_gemm && layer.wo)
            {
                bool wo_is_sharded = isRowParallelSharded(layer.wo);
                bool has_multi_rank = mpi_ctx_ && mpi_ctx_->world_size() > 1;

                if (wo_is_sharded && has_multi_rank)
                {
                    graph.addDependency(prefix + "attn_residual", prefix + "wo_allreduce");
                }
                else
                {
                    graph.addDependency(prefix + "attn_residual", prefix + "wo_proj");
                }
            }
        }

        return graph;
    }

    ComputeGraph Qwen2Graph::buildFFNGraph(
        const Qwen2LayerWeights &layer,
        Qwen2ActivationBuffers &buffers,
        int layer_idx,
        int seq_len,
        int device_idx)
    {
        ComputeGraph graph;
        const auto &env = debugEnv();
        std::string prefix = "layer" + std::to_string(layer_idx) + "_";

        // Determine backend type
        auto &dm = DeviceManager::instance();
        ComputeBackendType backend = ComputeBackendType::CPU;
        if (static_cast<size_t>(device_idx) < dm.devices().size())
        {
            backend = dm.devices()[device_idx].type;
        }

        // Stage 1: Pre-FFN RMSNorm
        if (env.execution.exec_rmsnorm)
        {
            RMSNormStage::Params ffn_norm_params;
            ffn_norm_params.input = buffers.current_hidden;
            ffn_norm_params.output = buffers.normalized;
            ffn_norm_params.gamma = layer.ffn_norm;
            ffn_norm_params.eps = config_.rms_norm_eps;
            ffn_norm_params.seq_len = seq_len;

            graph.addNode(prefix + "ffn_norm",
                          ComputeStageFactory::createRMSNorm(ffn_norm_params),
                          device_idx);
        }

        // Stage 2: Gate and Up projections using FusedGateUpGEMMStage
        if (env.execution.exec_gemm && layer.gate_proj && layer.up_proj)
        {
            LOG_DEBUG("[Qwen2Graph] FFN using FusedGateUpGEMMStage");

            int k = config_.d_model;
            int gate_n = static_cast<int>(layer.gate_proj->shape()[0]);
            int up_n = static_cast<int>(layer.up_proj->shape()[0]);

            FusedGateUpGEMMStage::Params gate_up_params;
            gate_up_params.input = buffers.normalized;
            gate_up_params.m = seq_len;
            gate_up_params.k = k;
            gate_up_params.w_gate = layer.gate_proj;
            gate_up_params.output_gate = buffers.gate;
            gate_up_params.n_gate = gate_n;
            gate_up_params.w_up = layer.up_proj;
            gate_up_params.output_up = buffers.up;
            gate_up_params.n_up = up_n;
            gate_up_params.mpi_ctx = mpi_ctx_.get();
            gate_up_params.device_idx = device_idx;

            graph.addNode(prefix + "gate_up_proj",
                          ComputeStageFactory::createFusedGateUpGEMM(gate_up_params),
                          device_idx);

            if (env.execution.exec_rmsnorm)
            {
                graph.addDependency(prefix + "gate_up_proj", prefix + "ffn_norm");
            }
        }

        // Stage 3: SwiGLU activation
        if (env.execution.exec_swiglu)
        {
            SwiGLUStage::Params swiglu_params;
            swiglu_params.gate = buffers.gate;
            swiglu_params.up = buffers.up;
            swiglu_params.output = buffers.up; // In-place

            graph.addNode(prefix + "swiglu",
                          ComputeStageFactory::createSwiGLU(swiglu_params),
                          device_idx);

            if (env.execution.exec_gemm)
            {
                graph.addDependency(prefix + "swiglu", prefix + "gate_up_proj");
            }
        }

        // Stage 4: Down projection
        if (env.execution.exec_gemm && layer.down_proj)
        {
            int down_n = static_cast<int>(layer.down_proj->shape()[0]);
            int down_k = static_cast<int>(layer.down_proj->shape()[1]);

            graph.addNode(prefix + "down_proj",
                          ComputeStageFactory::createGEMM(
                              GEMMStage::Params{
                                  .A = buffers.up,
                                  .B = layer.down_proj,
                                  .C = buffers.attn_proj,
                                  .m = seq_len,
                                  .n = down_n,
                                  .k = down_k,
                                  .alpha = 1.0f,
                                  .beta = 0.0f,
                                  .transpose_B = false}),
                          device_idx);

            if (env.execution.exec_swiglu)
            {
                graph.addDependency(prefix + "down_proj", prefix + "swiglu");
            }

            bool down_is_row_sharded = isRowParallelSharded(layer.down_proj);
            bool needs_allreduce = (down_is_row_sharded || config_.ffn_column_parallel);
            bool has_multi_rank = mpi_ctx_ && mpi_ctx_->world_size() > 1;

            if (needs_allreduce && has_multi_rank)
            {
                size_t allreduce_count = static_cast<size_t>(seq_len) * down_n;
                MPI_Comm comm = static_cast<MPI_Comm>(getMPIComm(mpi_ctx_.get()));

                LOG_DEBUG("[buildFFNGraph] Adding down_allreduce: ffn_column_parallel="
                          << config_.ffn_column_parallel << " down_is_row_sharded=" << down_is_row_sharded
                          << " count=" << allreduce_count);

                graph.addNode(prefix + "down_allreduce",
                              ComputeStageFactory::createAllreduce(
                                  AllreduceStage::Params{
                                      buffers.attn_proj,
                                      comm}),
                              device_idx);

                graph.addDependency(prefix + "down_allreduce", prefix + "down_proj");
            }
        }

        // Stage 5: Residual connection
        if (env.execution.exec_residual)
        {
            ResidualAddStage::Params res_params;
            res_params.input = buffers.attn_proj;
            res_params.residual = buffers.current_hidden;
            res_params.output = buffers.current_hidden;
            res_params.num_elements = static_cast<size_t>(seq_len) * static_cast<size_t>(config_.d_model);

            graph.addNode(prefix + "ffn_residual",
                          ComputeStageFactory::createResidualAdd(res_params),
                          device_idx);

            if (env.execution.exec_gemm && layer.down_proj)
            {
                bool down_is_row_sharded = isRowParallelSharded(layer.down_proj);
                bool needs_allreduce = (down_is_row_sharded || config_.ffn_column_parallel);
                bool has_multi_rank = mpi_ctx_ && mpi_ctx_->world_size() > 1;

                if (needs_allreduce && has_multi_rank)
                {
                    graph.addDependency(prefix + "ffn_residual", prefix + "down_allreduce");
                }
                else
                {
                    graph.addDependency(prefix + "ffn_residual", prefix + "down_proj");
                }
            }
        }

        return graph;
    }

    // =============================================================================
    // Execution Methods
    // =============================================================================

    bool Qwen2Graph::executeForward(
        const Qwen2ForwardInput &input,
        Qwen2ForwardOutput &output)
    {
        auto start = std::chrono::high_resolution_clock::now();

        if (!input.token_ids && !input.batches)
        {
            LOG_ERROR("[Qwen2Graph] No token input provided");
            return false;
        }

        if (input.seq_len <= 0)
        {
            LOG_ERROR("[Qwen2Graph] Invalid sequence length: " << input.seq_len);
            return false;
        }

        LOG_TRACE("[Qwen2Graph] executeForward: batch_size=" << input.batch_size
                                                             << ", seq_len=" << input.seq_len
                                                             << ", device=" << input.device_idx);

        // Build and execute full forward graph
        ComputeGraph graph = buildFullForwardGraph(input, output);

        if (graph.size() == 0)
        {
            LOG_ERROR("[Qwen2Graph] Empty forward graph");
            return false;
        }

        // Get device context
        IDeviceContext *ctx = getDeviceContext(input.device_idx);
        if (!ctx)
        {
            LOG_ERROR("[Qwen2Graph] Failed to get device context");
            return false;
        }

        // Execute
        bool success = executor_.execute(graph, ctx);

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;

        LOG_DEBUG("[Qwen2Graph] Forward completed in " << ms << "ms, success=" << success);

        return success;
    }

    bool Qwen2Graph::execute(ComputeGraph &graph, IDeviceContext *ctx)
    {
        return executor_.execute(graph, ctx);
    }

    bool Qwen2Graph::executeAttention(
        const Qwen2LayerWeights &layer,
        Qwen2ActivationBuffers &buffers,
        int layer_idx,
        int seq_len,
        IUnifiedKVCache *kv_cache,
        const int *position_ids,
        int device_idx)
    {
        const auto &env = debugEnv();

        // Debug: dump input to attention (for layer 0 only)
        if (layer_idx == 0)
        {
            const float *input = buffers.current_hidden->fp32_data();
            LOG_INFO("[EXEC_ATTN_INPUT] layer=" << layer_idx << " seq_len=" << seq_len
                                                << " input[0:4]=" << input[0] << "," << input[1] << "," << input[2] << "," << input[3]);
        }

        // Get device context
        IDeviceContext *ctx = getDeviceContext(device_idx);
        if (!ctx)
        {
            return false;
        }

        // =============================================================================
        // Phase 10: Graph Caching for Decode Mode
        // =============================================================================
        // For decode mode (seq_len=1), we can cache and reuse the attention graph.
        // Only the position offset changes between decode steps.
        // =============================================================================
        int pos_offset = position_ids ? position_ids[0] : 0;

        if (graph_caching_enabled_ && seq_len == 1 &&
            layer_idx >= 0 && static_cast<size_t>(layer_idx) < layer_graph_cache_.size())
        {
            auto &cache = layer_graph_cache_[layer_idx];

            // Check if we have a valid cached graph for decode mode
            if (cache.attention_decode && cache.cached_seq_len == 1 && cache.valid)
            {
                LOG_DEBUG("[Qwen2Graph] Reusing cached attention graph for layer " << layer_idx << " (pos_offset=" << pos_offset << ")");

                // Update dynamic parameters (position offset)
                updateCachedGraphParams(*cache.attention_decode, pos_offset, seq_len);

                // Execute cached graph
                bool success = executor_.execute(*cache.attention_decode, ctx);
                if (!success)
                {
                    LOG_ERROR("[Qwen2Graph] Cached attention graph failed at layer " << layer_idx);
                }
                cache.attention_decode->reset(); // Reset for next execution
                return success;
            }

            // Build and cache the graph
            LOG_DEBUG("[Qwen2Graph] Building and caching attention graph for layer " << layer_idx << " (decode mode)");
            cache.attention_decode = std::make_unique<ComputeGraph>(
                buildAttentionGraph(layer, buffers, layer_idx, seq_len, kv_cache, position_ids, device_idx));
            cache.cached_seq_len = seq_len;
            cache.valid = true;

            // Execute the newly built graph
            bool success = executor_.execute(*cache.attention_decode, ctx);
            if (!success)
            {
                LOG_ERROR("[Qwen2Graph] Attention block failed at layer " << layer_idx);
            }
            cache.attention_decode->reset();
            return success;
        }

        // =============================================================================
        // Non-cached path (prefill or caching disabled)
        // =============================================================================
        ComputeGraph graph = buildAttentionGraph(layer, buffers, layer_idx, seq_len,
                                                 kv_cache, position_ids, device_idx);

        // Debug: log graph structure
        if (layer_idx == 0)
        {
            auto order = graph.getExecutionOrder();
            LOG_INFO("[EXEC_ATTN] Graph has " << graph.size() << " nodes, execution order:");
            for (const auto &name : order)
            {
                LOG_INFO("[EXEC_ATTN]   - " << name);
            }
        }

        bool success = executor_.execute(graph, ctx);

        // Debug: dump intermediate buffers (for layer 0 only)
        if (layer_idx == 0 && buffers.normalized && buffers.Q && buffers.attn_output && buffers.attn_proj)
        {
            const float *normalized = buffers.normalized->fp32_data();
            const float *Q = buffers.Q->fp32_data();
            const float *attn_output = buffers.attn_output->fp32_data();
            const float *attn_proj = buffers.attn_proj->fp32_data();
            const float *output = buffers.current_hidden->fp32_data();
            LOG_INFO("[EXEC_ATTN_OUTPUT] layer=" << layer_idx << " seq_len=" << seq_len
                                                 << " output[0:4]=" << output[0] << "," << output[1] << "," << output[2] << "," << output[3]);
        }

        if (!success)
        {
            LOG_ERROR("[Qwen2Graph] Attention block failed at layer " << layer_idx);
        }

        return success;
    }

    bool Qwen2Graph::executeFFN(
        const Qwen2LayerWeights &layer,
        Qwen2ActivationBuffers &buffers,
        int layer_idx,
        int seq_len,
        int device_idx)
    {
        // Get device context
        IDeviceContext *ctx = getDeviceContext(device_idx);
        if (!ctx)
        {
            return false;
        }

        // =============================================================================
        // Phase 10: Graph Caching for Decode Mode
        // =============================================================================
        // FFN graphs have no dynamic parameters (no position offset).
        // For decode mode (seq_len=1), we can cache and reuse directly.
        // =============================================================================
        if (graph_caching_enabled_ && seq_len == 1 &&
            layer_idx >= 0 && static_cast<size_t>(layer_idx) < layer_graph_cache_.size())
        {
            auto &cache = layer_graph_cache_[layer_idx];

            // Check if we have a valid cached FFN graph
            if (cache.ffn_decode && cache.valid)
            {
                LOG_DEBUG("[Qwen2Graph] Reusing cached FFN graph for layer " << layer_idx);

                // Execute cached graph (no params to update for FFN)
                bool success = executor_.execute(*cache.ffn_decode, ctx);
                if (!success)
                {
                    LOG_ERROR("[Qwen2Graph] Cached FFN graph failed at layer " << layer_idx);
                }
                cache.ffn_decode->reset();
                return success;
            }

            // Build and cache the graph
            LOG_DEBUG("[Qwen2Graph] Building and caching FFN graph for layer " << layer_idx << " (decode mode)");
            cache.ffn_decode = std::make_unique<ComputeGraph>(
                buildFFNGraph(layer, buffers, layer_idx, seq_len, device_idx));

            // Execute the newly built graph
            bool success = executor_.execute(*cache.ffn_decode, ctx);
            if (!success)
            {
                LOG_ERROR("[Qwen2Graph] FFN block failed at layer " << layer_idx);
            }
            cache.ffn_decode->reset();
            return success;
        }

        // =============================================================================
        // Non-cached path (prefill or caching disabled)
        // =============================================================================
        ComputeGraph graph = buildFFNGraph(layer, buffers, layer_idx, seq_len, device_idx);

        bool success = executor_.execute(graph, ctx);

        if (!success)
        {
            LOG_ERROR("[Qwen2Graph] FFN block failed at layer " << layer_idx);
        }

        return success;
    }

    bool Qwen2Graph::executeLayer(
        const Qwen2LayerWeights &layer,
        Qwen2ActivationBuffers &buffers,
        int layer_idx,
        int seq_len,
        IUnifiedKVCache *kv_cache,
        const int *position_ids,
        int device_idx)
    {
        LOG_INFO("[Qwen2Graph::executeLayer] LAYER_EXEC_ENTERED layer_idx=" << layer_idx << " seq_len=" << seq_len);

        // Execute attention block
        if (!executeAttention(layer, buffers, layer_idx, seq_len, kv_cache, position_ids, device_idx))
        {
            return false;
        }

        // Execute FFN block
        if (!executeFFN(layer, buffers, layer_idx, seq_len, device_idx))
        {
            return false;
        }

        return true;
    }

    // =============================================================================
    // Helper Methods
    // =============================================================================

    std::vector<int> Qwen2Graph::buildPositionIds(int seq_len, int batch_size, int offset)
    {
        std::vector<int> pos_ids(batch_size * seq_len);

        for (int b = 0; b < batch_size; ++b)
        {
            for (int s = 0; s < seq_len; ++s)
            {
                pos_ids[b * seq_len + s] = offset + s;
            }
        }

        return pos_ids;
    }

    void Qwen2Graph::addFinalNormToGraph(
        ComputeGraph &graph,
        TensorBase *hidden,
        const std::string &prev_node,
        int n_tokens,
        int device_idx)
    {
        RMSNormStage::Params norm_params;
        norm_params.input = hidden;
        norm_params.output = hidden; // In-place norm
        norm_params.gamma = weights_.final_norm;
        norm_params.eps = config_.rms_norm_eps;
        norm_params.seq_len = n_tokens;
        norm_params.device_idx = device_idx;

        graph.addNode("final_norm",
                      ComputeStageFactory::createRMSNorm(norm_params),
                      device_idx);

        if (!prev_node.empty())
        {
            graph.addDependency("final_norm", prev_node);
        }
    }

    void Qwen2Graph::clearCache()
    {
        current_batch_size_ = 0;
        current_seq_len_ = 0;
        position_ids_buffer_.clear();

        // Clear cached graphs (Phase 10)
        for (auto &cache : layer_graph_cache_)
        {
            cache.attention_decode.reset();
            cache.ffn_decode.reset();
            cache.cached_seq_len = 0;
            cache.valid = false;
        }
        last_pos_offset_ = -1;

        LOG_DEBUG("[Qwen2Graph] Cache cleared (including " << layer_graph_cache_.size() << " layer graph caches)");
    }

    bool Qwen2Graph::hasValidCachedGraph(int layer_idx, bool is_attention) const
    {
        if (!graph_caching_enabled_)
            return false;
        if (layer_idx < 0 || static_cast<size_t>(layer_idx) >= layer_graph_cache_.size())
            return false;

        const auto &cache = layer_graph_cache_[layer_idx];
        if (!cache.valid)
            return false;

        if (is_attention)
            return cache.attention_decode != nullptr;
        else
            return cache.ffn_decode != nullptr;
    }

    // =============================================================================
    // Graph Buffer Management (Phase 5)
    // =============================================================================

    bool Qwen2Graph::initializeBuffers(int seq_len)
    {
        if (!config_.use_graph_buffer_management)
        {
            LOG_WARN("[Qwen2Graph] initializeBuffers called but use_graph_buffer_management=false");
            return false;
        }

        LOG_INFO("[Qwen2Graph] Initializing buffers with graph management, seq_len=" << seq_len);

        // Create buffer spec builder if not already created
        if (!buffer_spec_builder_)
        {
            // Determine activation tensor type
            BufferTensorType activation_type =
                (config_.activation_precision == ActivationPrecision::Q8_1)
                    ? BufferTensorType::Q8_1
                    : BufferTensorType::FP32;

            buffer_spec_builder_ = std::make_unique<Qwen2BufferSpecBuilder>(
                config_.d_model,
                config_.n_heads,
                config_.n_kv_heads,
                config_.head_dim,
                config_.d_ff,
                config_.vocab_size,
                activation_type,
                config_.default_device);
        }

        // Verify TensorFactory is set
        if (!tensor_factory_)
        {
            LOG_ERROR("[Qwen2Graph] TensorFactory not set. Call setTensorFactory() before initializeBuffers()");
            return false;
        }

        // Create buffer manager with TensorFactory
        buffer_manager_ = std::make_unique<GraphBufferManager>(
            tensor_factory_, mpi_ctx_.get());

        // Build layer buffer specifications
        auto layer_specs = buffer_spec_builder_->buildLayerSpecs(seq_len);
        auto layer_reqs = toBufferRequirements(layer_specs);

        // Allocate layer buffers (iterate over vector)
        for (const auto &desc : layer_reqs.buffers)
        {
            if (!buffer_manager_->allocateBuffer("layer", desc))
            {
                LOG_ERROR("[Qwen2Graph] Failed to allocate layer buffer: " << desc.name);
                return false;
            }
        }

        // Build model-level buffer specifications (current_hidden, logits)
        auto model_specs = buffer_spec_builder_->buildModelSpecs(1, seq_len);
        auto model_reqs = toBufferRequirements(model_specs);

        for (const auto &desc : model_reqs.buffers)
        {
            if (!buffer_manager_->allocateBuffer("model", desc))
            {
                LOG_ERROR("[Qwen2Graph] Failed to allocate model buffer: " << desc.name);
                return false;
            }
        }

        // Bind allocated buffers to buffers_ struct
        bindGraphManagedBuffers(seq_len);

        auto &stats = buffer_manager_->stats();
        LOG_INFO("[Qwen2Graph] Buffer initialization complete: "
                 << "total=" << (stats.total_bytes / (1024.0 * 1024.0)) << " MB, "
                 << "scratch=" << (stats.scratch_bytes / (1024.0 * 1024.0)) << " MB, "
                 << "output=" << (stats.output_bytes / (1024.0 * 1024.0)) << " MB");

        // Log theoretical aliasing savings
        auto [original, optimized] = buffer_spec_builder_->estimateMemorySavings(seq_len);
        double savings = (original > 0) ? 100.0 * (original - optimized) / original : 0.0;
        LOG_INFO("[Qwen2Graph] Theoretical aliasing savings: "
                 << (original / 1024.0) << " KB -> " << (optimized / 1024.0) << " KB"
                 << " (" << savings << "% reduction)");

        // =============================================================================
        // Phase 10: Enable Graph Caching
        // =============================================================================
        // With graph-managed buffers, buffer pointers are stable across executions.
        // This means we can cache the built graphs and reuse them.
        // =============================================================================
        graph_caching_enabled_ = true;
        layer_graph_cache_.resize(config_.n_layers);
        LOG_INFO("[Qwen2Graph] Graph caching ENABLED for " << config_.n_layers << " layers");

        return true;
    }

    void Qwen2Graph::releaseBuffers()
    {
        // Clear cached graphs first (they reference buffers)
        for (auto &cache : layer_graph_cache_)
        {
            cache.attention_decode.reset();
            cache.ffn_decode.reset();
            cache.cached_seq_len = 0;
            cache.valid = false;
        }
        graph_caching_enabled_ = false;

        if (buffer_manager_)
        {
            buffer_manager_->releaseAll();
            buffer_manager_.reset();
            LOG_INFO("[Qwen2Graph] Buffers released");
        }

        owned_buffers_.clear();
        buffer_spec_builder_.reset();

        // Clear buffer pointers
        buffers_ = Qwen2ModelBuffers{};
    }

    const BufferAllocationStats *Qwen2Graph::bufferStats() const
    {
        if (!buffer_manager_)
        {
            return nullptr;
        }
        return &buffer_manager_->stats();
    }

    void Qwen2Graph::bindGraphManagedBuffers(int seq_len)
    {
        if (!buffer_manager_)
        {
            LOG_ERROR("[Qwen2Graph] bindGraphManagedBuffers: buffer_manager_ is null");
            return;
        }

        // Bind layer buffers
        auto &lb = buffers_.layer_buffers;

        lb.residual = buffer_manager_->getBuffer("layer", BufferNames::RESIDUAL);
        lb.normalized = buffer_manager_->getBuffer("layer", BufferNames::NORMALIZED);

        // Attention buffers
        lb.Q = buffer_manager_->getBuffer("layer", BufferNames::Q);
        lb.K = buffer_manager_->getBuffer("layer", BufferNames::K);
        lb.V = buffer_manager_->getBuffer("layer", BufferNames::V);
        lb.attn_output = buffer_manager_->getBuffer("layer", BufferNames::ATTN_OUTPUT);
        lb.attn_proj = buffer_manager_->getBuffer("layer", BufferNames::ATTN_PROJ);
        lb.workspace_scores = buffer_manager_->getBuffer("layer", BufferNames::WORKSPACE_SCORES);
        lb.workspace_context = buffer_manager_->getBuffer("layer", BufferNames::WORKSPACE_CONTEXT);
        lb.workspace_mask = buffer_manager_->getBuffer("layer", BufferNames::WORKSPACE_MASK);

        // FFN buffers
        lb.gate = buffer_manager_->getBuffer("layer", BufferNames::GATE);
        lb.up = buffer_manager_->getBuffer("layer", BufferNames::UP);
        lb.ffn_output = buffer_manager_->getBuffer("layer", BufferNames::FFN_OUTPUT);

        // Model-level buffers
        buffers_.current_hidden = buffer_manager_->getBuffer("model", BufferNames::CURRENT_HIDDEN);
        buffers_.logits = buffer_manager_->getBuffer("model", BufferNames::LOGITS);

        LOG_DEBUG("[Qwen2Graph] Bound graph-managed buffers: "
                  << "residual=" << lb.residual
                  << " Q=" << lb.Q
                  << " gate=" << lb.gate
                  << " current_hidden=" << buffers_.current_hidden
                  << " logits=" << buffers_.logits);
    }

    // =============================================================================
    // Graph Caching Helper Methods (Phase 10)
    // =============================================================================

    bool Qwen2Graph::canUseCachedGraph(int layer_idx, int seq_len) const
    {
        if (!graph_caching_enabled_)
            return false;
        if (layer_idx < 0 || static_cast<size_t>(layer_idx) >= layer_graph_cache_.size())
            return false;

        const auto &cache = layer_graph_cache_[layer_idx];
        return cache.valid && cache.cached_seq_len == seq_len;
    }

    void Qwen2Graph::updateCachedGraphParams(ComputeGraph &graph, int pos_offset, int seq_len)
    {
        // Update all stages in the graph that have dynamic parameters
        // The graph structure allows us to iterate through all nodes
        auto order = graph.getExecutionOrder();

        for (const auto &node_name : order)
        {
            ComputeNode *node = graph.getNode(node_name);
            if (!node || !node->stage)
                continue;

            // Update dynamic params (pos_offset, seq_len)
            // Only stages that override updateDynamicParams will actually do anything
            node->stage->updateDynamicParams(pos_offset, seq_len);
        }

        LOG_TRACE("[Qwen2Graph] Updated cached graph params: pos_offset=" << pos_offset << " seq_len=" << seq_len);
    }

} // namespace llaminar2
