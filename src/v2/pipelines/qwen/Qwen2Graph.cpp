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
#include "../../utils/Logger.h"
#include "../../utils/DebugEnv.h"
#include "../../backends/ComputeBackend.h"
#include "../../tensors/TensorSlice.h"
#include "../../tensors/Tensors.h"
#include "../../utils/MPIContext.h"
#include "../MPIStrategy.h"
#include <algorithm>
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
                                                                    << " n_kv_heads=" << config_.n_kv_heads
                                                                    << " mpi_ctx=" << (mpi_ctx_ ? "valid" : "nullptr")
                                                                    << " world_size=" << (mpi_ctx_ ? mpi_ctx_->world_size() : -1));

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
    // IGraphBuilder Interface Implementation
    // =============================================================================

    ComputeGraph Qwen2Graph::buildForwardGraph(
        const ForwardInput &input,
        ForwardOutput &output)
    {
        // Adapt generic ForwardInput to Qwen2ForwardInput
        Qwen2ForwardInput qwen_input;
        qwen_input.token_ids = input.token_ids;
        qwen_input.position_ids = input.position_ids;
        qwen_input.batch_size = input.batch_size;
        qwen_input.seq_len = input.seq_len;
        qwen_input.position_offset = input.position_offset;
        qwen_input.device_idx = input.device_idx;
        qwen_input.kv_cache = input.kv_cache;

        // Adapt generic ForwardOutput to Qwen2ForwardOutput
        Qwen2ForwardOutput qwen_output;
        qwen_output.logits = output.logits;
        qwen_output.hidden = output.hidden;

        // Build the graph
        ComputeGraph graph = buildFullForwardGraph(qwen_input, qwen_output);

        // Copy back output pointers (in case they were set by the builder)
        output.logits = qwen_output.logits;
        output.hidden = qwen_output.hidden;

        return graph;
    }

    ComputeGraph Qwen2Graph::buildLayerGraph(const LayerContext &ctx)
    {
        // Validate layer index
        if (ctx.layer_idx < 0 || ctx.layer_idx >= config_.n_layers)
        {
            LOG_ERROR("[Qwen2Graph::buildLayerGraph] Invalid layer index: " << ctx.layer_idx);
            return ComputeGraph{};
        }

        // Get layer weights
        if (!weights_.get_layer_weights)
        {
            LOG_ERROR("[Qwen2Graph::buildLayerGraph] Layer weight accessor not set");
            return ComputeGraph{};
        }

        Qwen2LayerWeights layer_weights = weights_.get_layer_weights(ctx.layer_idx);

        // Build attention graph
        ComputeGraph attn_graph = buildAttentionGraph(
            layer_weights, buffers_.layer_buffers, ctx.layer_idx, ctx.seq_len,
            ctx.batch_size, ctx.kv_cache, ctx.position_ids, ctx.device_idx,
            ctx.sequence_lengths);

        // Build FFN graph
        ComputeGraph ffn_graph = buildFFNGraph(
            layer_weights, buffers_.layer_buffers, ctx.layer_idx, ctx.seq_len,
            ctx.batch_size, ctx.device_idx);

        // Merge: attention -> FFN
        std::string attn_last = "layer" + std::to_string(ctx.layer_idx) + "_attn_residual";
        attn_graph.merge(std::move(ffn_graph), attn_last);

        return attn_graph;
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
        // Stage 2: Transformer Layers (complete graphs, not placeholders)
        // -------------------------------------------------------------------------
        std::string prev_node = "embedding";

        // Position IDs must be provided externally (or use fallback for backward compat)
        const int *position_ids = input.position_ids;
        if (!position_ids)
        {
            // Fallback: build position IDs internally (deprecated path)
            position_ids_buffer_ = buildPositionIds(input.seq_len, input.batch_size, input.position_offset);
            position_ids = position_ids_buffer_.data();
            LOG_DEBUG("[Qwen2Graph] Position IDs built internally (deprecated - prefer external input)");
        }

        // Check if we have layer weight accessor
        if (!weights_.get_layer_weights)
        {
            LOG_ERROR("[Qwen2Graph] Layer weight accessor not set!");
            throw std::runtime_error("Qwen2Graph layer weight accessor not initialized");
        }

        // Build complete graphs for each layer
        for (int layer = 0; layer < config_.n_layers; ++layer)
        {
            // Get layer weights
            Qwen2LayerWeights layer_weights = weights_.get_layer_weights(layer);

            // Build attention graph for this layer
            // Pass sequence_lengths for proper batch-aware attention masking
            ComputeGraph attn_graph = buildAttentionGraph(
                layer_weights, buffers_.layer_buffers, layer, input.seq_len,
                input.batch_size, input.kv_cache, position_ids, device_idx,
                input.sequence_lengths);

            // Get the leaf node(s) of attention graph before merging
            auto attn_leaves = attn_graph.getLeafNodes();
            std::string attn_last;
            if (!attn_leaves.empty())
            {
                std::string prefix = "layer" + std::to_string(layer) + "_";
                attn_last = prefix + "attn_residual";
                if (std::find(attn_leaves.begin(), attn_leaves.end(), attn_last) == attn_leaves.end())
                {
                    attn_last = attn_leaves.front();
                }
            }
            else
            {
                // Fallback: use previous node if attention graph is empty
                attn_last = prev_node;
            }

            // Merge attention graph, connecting to previous node
            graph.merge(std::move(attn_graph), prev_node);

            // Build FFN graph for this layer
            ComputeGraph ffn_graph = buildFFNGraph(
                layer_weights, buffers_.layer_buffers, layer, input.seq_len,
                input.batch_size, device_idx);

            // Get the leaf node(s) of FFN graph before merging (we need the last node)
            auto ffn_leaves = ffn_graph.getLeafNodes();
            std::string ffn_last;
            if (!ffn_leaves.empty())
            {
                // Prefer ffn_residual if it exists, otherwise use first leaf
                std::string prefix = "layer" + std::to_string(layer) + "_";
                ffn_last = prefix + "ffn_residual";
                if (std::find(ffn_leaves.begin(), ffn_leaves.end(), ffn_last) == ffn_leaves.end())
                {
                    ffn_last = ffn_leaves.front(); // Fall back to first leaf
                }
            }
            else
            {
                // Fallback: use attention residual if FFN graph is empty
                ffn_last = attn_last;
            }

            // Merge FFN graph, connecting to attention residual
            graph.merge(std::move(ffn_graph), attn_last);

            // Use the determined FFN leaf as the prev node for next layer
            prev_node = ffn_last;
        }

        // -------------------------------------------------------------------------
        // Stage 3: Final RMSNorm
        // -------------------------------------------------------------------------
        addFinalNormToGraph(graph, buffers_.current_hidden, prev_node, total_tokens, device_idx);
        prev_node = "final_norm";

        // -------------------------------------------------------------------------
        // Stage 4: LM Head (with optional Column-Parallel + AllGather)
        // -------------------------------------------------------------------------
        // When lm_head_column_parallel is enabled:
        // - LM head weight is sharded by vocab: [vocab_local, d_model]
        // - LM head outputs to buffers_.logits_local: [seq_len, vocab_local]
        // - AllGather collects to buffers_.logits: [seq_len, vocab_size]
        // -------------------------------------------------------------------------
        bool use_column_parallel = config_.lm_head_column_parallel && buffers_.logits_local != nullptr;

        // Determine output buffer and vocab size for LM head stage
        TensorBase *lm_head_output = use_column_parallel ? buffers_.logits_local : buffers_.logits;
        int lm_head_vocab_size = use_column_parallel ? config_.vocab_local : config_.vocab_size;

        LOG_DEBUG("[Qwen2Graph] LM head in buildFullForwardGraph: use_column_parallel="
                  << use_column_parallel << " lm_head_vocab_size=" << lm_head_vocab_size);

        LMHeadStage::Params lm_params;
        lm_params.hidden_states = buffers_.current_hidden;
        lm_params.lm_head_weight = weights_.lm_head;
        lm_params.logits = lm_head_output;
        lm_params.seq_len = total_tokens;
        lm_params.d_model = config_.d_model;
        lm_params.vocab_size = lm_head_vocab_size;
        lm_params.bias = nullptr; // Qwen2 has no LM head bias
        lm_params.device_idx = device_idx;

        graph.addNode("lm_head",
                      ComputeStageFactory::createLMHead(lm_params),
                      device_idx);
        graph.addDependency("lm_head", prev_node);
        prev_node = "lm_head";

        // Phase 5: AllGather stage for column-parallel LM head
        if (use_column_parallel && mpi_ctx_)
        {
            LOG_DEBUG("[Qwen2Graph] Adding lm_head_allgather in buildFullForwardGraph: world_size="
                      << mpi_ctx_->world_size() << " total_tokens=" << total_tokens);

            AllGatherStage::Params allgather_params;
            allgather_params.local_input = buffers_.logits_local;
            allgather_params.full_output = buffers_.logits;
            allgather_params.mpi_comm = mpi_ctx_->comm();
            allgather_params.world_size = mpi_ctx_->world_size();
            allgather_params.actual_seq_len = static_cast<size_t>(total_tokens);

            graph.addNode("lm_head_allgather",
                          ComputeStageFactory::createAllGather(allgather_params),
                          device_idx);
            graph.addDependency("lm_head_allgather", prev_node);
        }

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
        int total_tokens,
        int device_idx,
        TensorBase *logits_local)
    {
        LOG_DEBUG("[Qwen2Graph] Building LM head graph for " << total_tokens << " tokens"
                                                             << " lm_head_column_parallel=" << config_.lm_head_column_parallel
                                                             << " vocab_local=" << config_.vocab_local);

        ComputeGraph graph;

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

        // =================================================================
        // LM Head Projection - Column-Parallel or Full
        // =================================================================
        // When lm_head_column_parallel is enabled:
        // - LM head weight is sharded by vocab: [vocab_local, d_model]
        // - LM head outputs to logits_local: [seq_len, vocab_local]
        // - AllGather collects to output_logits: [seq_len, vocab_size]
        // =================================================================

        bool use_column_parallel = config_.lm_head_column_parallel && logits_local != nullptr;

        // Determine output buffer and vocab size for LM head stage
        TensorBase *lm_head_output = use_column_parallel ? logits_local : output_logits;
        int lm_head_vocab_size = use_column_parallel ? config_.vocab_local : config_.vocab_size;

        LOG_DEBUG("[Qwen2Graph] LM head: use_column_parallel=" << use_column_parallel
                                                               << " lm_head_vocab_size=" << lm_head_vocab_size
                                                               << " lm_head_output=" << lm_head_output);

        // LM Head projection
        LMHeadStage::Params lm_params;
        lm_params.hidden_states = hidden_states;
        lm_params.lm_head_weight = weights_.lm_head;
        lm_params.logits = lm_head_output;
        lm_params.seq_len = total_tokens;
        lm_params.d_model = config_.d_model;
        lm_params.vocab_size = lm_head_vocab_size;
        lm_params.bias = nullptr;
        lm_params.device_idx = device_idx;

        graph.addNode("lm_head",
                      ComputeStageFactory::createLMHead(lm_params),
                      device_idx);
        graph.addDependency("lm_head", "final_norm");

        // =================================================================
        // AllGather stage for column-parallel LM head
        // =================================================================
        if (use_column_parallel && mpi_ctx_)
        {
            LOG_DEBUG("[Qwen2Graph] Adding lm_head_allgather: world_size=" << mpi_ctx_->world_size()
                                                                           << " total_tokens=" << total_tokens);

            AllGatherStage::Params allgather_params;
            allgather_params.local_input = logits_local;
            allgather_params.full_output = output_logits;
            allgather_params.mpi_comm = mpi_ctx_->comm();
            allgather_params.world_size = mpi_ctx_->world_size();
            allgather_params.actual_seq_len = static_cast<size_t>(total_tokens);

            graph.addNode("lm_head_allgather",
                          ComputeStageFactory::createAllGather(allgather_params),
                          device_idx);
            graph.addDependency("lm_head_allgather", "lm_head");
        }

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
        int batch_size,
        IUnifiedKVCache *kv_cache,
        const int *position_ids,
        int device_idx,
        const std::vector<int> *sequence_lengths)
    {
        ComputeGraph graph;
        const auto &env = debugEnv();
        std::string prefix = "layer" + std::to_string(layer_idx) + "_";

        // Total tokens for GEMM m dimension = batch_size * seq_len
        int total_tokens = batch_size * seq_len;

        LOG_DEBUG("[buildAttentionGraph] layer_idx=" << layer_idx << " seq_len=" << seq_len
                                                     << " batch_size=" << batch_size
                                                     << " total_tokens=" << total_tokens
                                                     << " layer.wq=" << static_cast<const void *>(layer.wq)
                                                     << " layer.wo=" << layer.wo << " world_size="
                                                     << (mpi_ctx_ ? mpi_ctx_->world_size() : 1)
                                                     << " sequence_lengths=" << (sequence_lengths ? "valid" : "nullptr")
                                                     << (sequence_lengths ? " size=" + std::to_string(sequence_lengths->size()) : ""));

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
            attn_norm_params.seq_len = total_tokens; // Use total_tokens = batch_size * seq_len

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

            LOG_DEBUG("[Qwen2Graph] Layer " << layer_idx << " QKV dims: q_n=" << q_n
                                            << " k_n=" << k_n << " v_n=" << v_n
                                            << " wq_shape=[" << layer.wq->shape()[0] << "," << layer.wq->shape()[1] << "]"
                                            << " wk_shape=[" << layer.wk->shape()[0] << "," << layer.wk->shape()[1] << "]");

            // Extract bias pointers
            // Note: Use TensorBase::data() directly - works for both raw tensors and TensorSlice
            // (TensorSlice delegates data() to its inner tensor)
            const float *q_bias_ptr = layer.q_bias ? layer.q_bias->data() : nullptr;
            const float *k_bias_ptr = layer.k_bias ? layer.k_bias->data() : nullptr;
            const float *v_bias_ptr = layer.v_bias ? layer.v_bias->data() : nullptr;

            FusedQKVGEMMStage::Params qkv_params;
            qkv_params.input = buffers.normalized;
            qkv_params.m = total_tokens; // Use total_tokens = batch_size * seq_len
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

        // Stage 3: RoPE on Q and K
        if (env.execution.exec_rope)
        {
            int pos_offset = position_ids ? position_ids[0] : 0;

            RoPEStage::Params rope_params;
            rope_params.Q = buffers.Q;
            rope_params.K = buffers.K;
            rope_params.n_heads = local_n_heads;       // Use local head count for TP
            rope_params.n_kv_heads = local_n_kv_heads; // Use local KV head count for TP
            rope_params.head_dim = config_.head_dim;
            rope_params.pos_offset = pos_offset;
            rope_params.theta_base = config_.rope_theta;
            rope_params.seq_len = total_tokens; // Use total_tokens = batch_size * seq_len

            graph.addNode(prefix + "rope",
                          ComputeStageFactory::createRoPE(rope_params),
                          device_idx);

            if (env.execution.exec_gemm)
            {
                graph.addDependency(prefix + "rope", prefix + "qkv_proj");
            }
        }

        // Stage 4: Attention computation with KV cache integration
        // NOTE: Decomposed attention (Phase 9) is now the ONLY supported path.
        // Legacy AttentionWithKVCacheStage has been removed (Phase 7 cleanup).
        if (env.execution.exec_attention)
        {
            // Phase 9 Decomposed Path: KVCacheAppendStage + AttentionComputeStage
            if (kv_cache)
            {
                // For batched execution, K/V are [batch_size * seq_len, kv_dim]
                // Each sequence's K/V is appended to its own seq_idx in the cache
                int total_tokens = batch_size * seq_len;

                KVCacheAppendStage::Params kv_append_params;
                kv_append_params.K = buffers.K;
                kv_append_params.V = buffers.V;
                kv_append_params.kv_cache = kv_cache;
                kv_append_params.layer_idx = layer_idx;
                kv_append_params.seq_idx = 0; // Starting seq_idx
                kv_append_params.num_tokens = total_tokens;
                kv_append_params.batch_size = batch_size; // Phase 3: Per-sequence append
                kv_append_params.seq_len = seq_len;       // Phase 3: Tokens per sequence

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
            int total_query_tokens = batch_size * seq_len;
            int kv_len = total_query_tokens; // Static hint for mode detection (actual queried at runtime)
            bool use_gather_stage = false;

            // For attention K/V source:
            // - Prefill (cached_tokens == 0): Use projected K/V directly [batch_size * seq_len, kv_dim]
            // - Decode (cached_tokens > 0 and batch_size == 1): Use cache (single-sequence only)
            // - Batched decode (cached_tokens > 0 and batch_size > 1): Gather K/V from multiple cache slots
            if (kv_cache)
            {
                int cached_tokens = kv_cache->get_cached_tokens(layer_idx, 0);
                if (cached_tokens > 0 && batch_size == 1)
                {
                    // Single-sequence decode: read K/V from cache
                    K_for_attn = kv_cache->get_k_base(layer_idx, 0);
                    V_for_attn = kv_cache->get_v_base(layer_idx, 0);
                    kv_len = cached_tokens;
                    LOG_TRACE("[Qwen2Graph] Layer " << layer_idx << " using cached K/V (decode mode)");
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
                gather_params.layer_idx = layer_idx;
                gather_params.batch_size = batch_size;
                gather_params.out_K = buffers.gathered_K;
                gather_params.out_V = buffers.gathered_V;
                // Note: out_max_kv_len and out_per_seq_kv_lens can be retrieved from stage after execute

                graph.addNode(prefix + "kv_gather",
                              ComputeStageFactory::createKVCacheGather(gather_params),
                              device_idx);

                // Gather depends on append (must append new tokens before gathering full history)
                graph.addDependency(prefix + "kv_gather", prefix + "kv_append");
            }

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
            attn_params.kv_len = kv_len;               // Static hint, actual queried from kv_cache at runtime
            attn_params.n_heads = local_n_heads;       // Use local head count for TP
            attn_params.n_kv_heads = local_n_kv_heads; // Use local KV head count for TP
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

            if (use_gather_stage)
            {
                // Batched decode: attention depends on gather
                graph.addDependency(prefix + "attention", prefix + "kv_gather");
            }
            else if (kv_cache)
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

        // Stage 5: Output projection (Wo)
        if (env.execution.exec_gemm && layer.wo)
        {
            int wo_n = static_cast<int>(layer.wo->shape()[0]);
            int wo_k = static_cast<int>(layer.wo->shape()[1]);

            LOG_DEBUG("[Qwen2Graph] Layer " << layer_idx << " Wo dims: wo_n=" << wo_n
                                            << " wo_k=" << wo_k
                                            << " wo_shape=[" << layer.wo->shape()[0] << "," << layer.wo->shape()[1] << "]"
                                            << " attn_output_shape=" << buffers.attn_output->shape()[0] << "x" << buffers.attn_output->shape()[1]);

            graph.addNode(prefix + "wo_proj",
                          ComputeStageFactory::createGEMM(
                              GEMMStage::Params{
                                  .A = buffers.attn_output,
                                  .B = layer.wo,
                                  .C = buffers.attn_proj,
                                  .m = total_tokens, // Use total_tokens = batch_size * seq_len
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
                // AllReduce the actual data, not full buffer capacity
                size_t allreduce_count = static_cast<size_t>(total_tokens) * static_cast<size_t>(config_.d_model);

                graph.addNode(prefix + "wo_allreduce",
                              ComputeStageFactory::createAllreduce(
                                  AllreduceStage::Params{
                                      buffers.attn_proj,
                                      getMPIComm(mpi_ctx_.get()),
                                      allreduce_count}),
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
            res_params.num_elements = static_cast<size_t>(total_tokens) * static_cast<size_t>(config_.d_model);

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
        int batch_size,
        int device_idx)
    {
        ComputeGraph graph;
        const auto &env = debugEnv();
        std::string prefix = "layer" + std::to_string(layer_idx) + "_";

        // Compute total tokens for GEMM m parameter
        int total_tokens = batch_size * seq_len;

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
            ffn_norm_params.seq_len = total_tokens; // Use total_tokens = batch_size * seq_len

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
            gate_up_params.m = total_tokens; // Use total_tokens = batch_size * seq_len
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
            swiglu_params.output = buffers.up;    // In-place
            swiglu_params.seq_len = total_tokens; // Use total_tokens = batch_size * seq_len

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
                                  .m = total_tokens, // Use total_tokens = batch_size * seq_len
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
                size_t allreduce_count = static_cast<size_t>(total_tokens) * down_n;
                MPI_Comm comm = static_cast<MPI_Comm>(getMPIComm(mpi_ctx_.get()));

                LOG_DEBUG("[buildFFNGraph] Adding down_allreduce: ffn_column_parallel="
                          << config_.ffn_column_parallel << " down_is_row_sharded=" << down_is_row_sharded
                          << " count=" << allreduce_count);

                graph.addNode(prefix + "down_allreduce",
                              ComputeStageFactory::createAllreduce(
                                  AllreduceStage::Params{
                                      buffers.attn_proj,
                                      comm,
                                      allreduce_count}),
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
            res_params.num_elements = static_cast<size_t>(total_tokens) * static_cast<size_t>(config_.d_model);

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
                buildAttentionGraph(layer, buffers, layer_idx, seq_len, 1, kv_cache, position_ids, device_idx, nullptr));
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
                                                 1, kv_cache, position_ids, device_idx, nullptr);

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
                buildFFNGraph(layer, buffers, layer_idx, seq_len, 1, device_idx)); // batch_size=1 for decode

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
        ComputeGraph graph = buildFFNGraph(layer, buffers, layer_idx, seq_len, 1, device_idx); // batch_size=1 for deprecated path

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

            // =================================================================
            // Phase 3: Use local head counts for buffer allocation when TP is enabled
            // =================================================================
            // When qkv_column_parallel is enabled:
            // - Q buffer: [seq_len, local_n_heads * head_dim]
            // - K/V buffers: [seq_len, local_n_kv_heads * head_dim]
            // - Attention output: [seq_len, local_n_heads * head_dim]
            // =================================================================
            int buf_n_heads = config_.qkv_column_parallel
                                  ? config_.local_n_heads
                                  : config_.n_heads;
            int buf_n_kv_heads = config_.qkv_column_parallel
                                     ? config_.local_n_kv_heads
                                     : config_.n_kv_heads;

            // Validate (safety check for uninitialized config)
            if (buf_n_heads <= 0)
                buf_n_heads = config_.n_heads;
            if (buf_n_kv_heads <= 0)
                buf_n_kv_heads = config_.n_kv_heads;

            // =================================================================
            // Phase 4: Use local FFN dimension for buffer allocation when TP is enabled
            // =================================================================
            // When ffn_column_parallel is enabled:
            // - Gate buffer: [seq_len, d_ff_local]
            // - Up buffer: [seq_len, d_ff_local]
            // - FFN output buffer: [seq_len, d_ff_local]
            // =================================================================
            int buf_d_ff = config_.ffn_column_parallel
                               ? config_.d_ff_local
                               : config_.d_ff;

            // Validate (safety check for uninitialized config)
            if (buf_d_ff <= 0)
                buf_d_ff = config_.d_ff;

            buffer_spec_builder_ = std::make_unique<Qwen2BufferSpecBuilder>(
                config_.d_model,
                buf_n_heads,    // Use local head count for TP
                buf_n_kv_heads, // Use local KV head count for TP
                config_.head_dim,
                buf_d_ff, // Use local FFN dimension for TP
                config_.vocab_size,
                activation_type,
                config_.default_device);

            LOG_DEBUG("[Qwen2Graph] Buffer spec builder: "
                      << "n_heads=" << buf_n_heads << " (config_.n_heads=" << config_.n_heads << "), "
                      << "n_kv_heads=" << buf_n_kv_heads << " (config_.n_kv_heads=" << config_.n_kv_heads << "), "
                      << "d_ff=" << buf_d_ff << " (config_.d_ff=" << config_.d_ff << "), "
                      << "qkv_column_parallel=" << config_.qkv_column_parallel << ", "
                      << "ffn_column_parallel=" << config_.ffn_column_parallel);
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
