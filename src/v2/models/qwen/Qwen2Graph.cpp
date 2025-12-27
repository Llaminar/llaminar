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
#include "../../execution/InferenceMode.h"
#include "../../tensors/TensorSlice.h"
#include "../../tensors/Tensors.h"
#include "../../utils/MPIContext.h"
#include "../../utils/MPIStrategy.h"
#include "../../execution/GraphBuildUtils.h"
#include "../../execution/RuntimeConfig.h"
#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace llaminar2
{

    // Import graph_utils for cleaner code
    using namespace graph_utils;

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

        LOG_INFO("[Qwen2Graph] Initialized (full) with " << config_.n_layers
                                                         << " layers, precision="
                                                         << activationPrecisionToString(config_.activation_precision));
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

        LOG_DEBUG("[Qwen2Graph] Initialized (layer-only)");
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
        // For HybridQ16 mode: output to Q16_1 residual buffer (the residual stream)
        // For other modes: output to FP32 current_hidden
        InferenceMode full_pass_inference_mode(config_.activation_precision);
        TensorBase *embed_output = buffers_.layer_buffers.residual &&
                                           full_pass_inference_mode.isHybridQ16()
                                       ? buffers_.layer_buffers.residual
                                       : buffers_.current_hidden;

        EmbeddingStage::Params embed_params;
        embed_params.embed_table = weights_.embedding_table;
        embed_params.token_ids = input.token_ids;
        embed_params.output = embed_output;
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
        std::vector<int> local_position_ids;
        if (!position_ids)
        {
            // Fallback: build position IDs internally (deprecated path)
            local_position_ids = buildPositionIds(input.seq_len, input.batch_size, input.position_offset);
            position_ids = local_position_ids.data();
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

    // =========================================================================
    // Schema-Based Graph Building (Phase 4d)
    // =========================================================================

    ComputeGraph Qwen2Graph::buildForwardGraphFromSchema(
        const Qwen2ForwardInput &input,
        Qwen2ForwardOutput &output)
    {
        LOG_DEBUG("[Qwen2Graph] Building forward graph from schema: "
                  << "batch_size=" << input.batch_size
                  << ", seq_len=" << input.seq_len);

        // =====================================================================
        // Step 1: Create the declarative schema
        // =====================================================================
        Qwen2SchemaFactory factory;
        GraphSchema schema = factory.createSchema();

        // =====================================================================
        // Step 2: Build runtime configuration
        // =====================================================================
        GraphResolverConfig config;

        // MPI / Tensor Parallelism
        config.world_size = mpi_ctx_ ? mpi_ctx_->world_size() : 1;
        config.rank = mpi_ctx_ ? mpi_ctx_->rank() : 0;
        config.mpi_comm = mpi_ctx_ ? mpi_ctx_->comm() : MPI_COMM_NULL;

        // Execution dimensions
        config.batch_size = input.batch_size;
        config.seq_len = input.seq_len;
        config.default_device = config_.default_device;

        // Model dimensions
        config.n_layers = config_.n_layers;
        config.d_model = config_.d_model;
        config.n_heads = config_.n_heads;
        config.n_kv_heads = config_.n_kv_heads;
        config.head_dim = config_.head_dim;
        config.d_ff = config_.d_ff;
        config.vocab_size = config_.vocab_size;

        // TP-adjusted dimensions
        config.local_n_heads = config_.qkv_column_parallel ? config_.local_n_heads : config_.n_heads;
        config.local_n_kv_heads = config_.qkv_column_parallel ? config_.local_n_kv_heads : config_.n_kv_heads;
        config.local_d_ff = config_.ffn_column_parallel ? config_.d_ff_local : config_.d_ff;
        config.local_vocab = config_.lm_head_column_parallel ? config_.vocab_local : config_.vocab_size;

        // Model parameters
        config.rms_norm_eps = config_.rms_norm_eps;
        config.rope_theta = config_.rope_theta;

        // KV cache state
        config.has_kv_cache = (input.kv_cache != nullptr);
        if (config.has_kv_cache)
        {
            config.cached_tokens = input.kv_cache->get_cached_tokens(0, 0);
        }

        // Execution policy from debugEnv
        config.exec_policy = ExecutionPolicyFlags::fromDebugEnv();

        // =====================================================================
        // Step 3: Build TensorContext for name → tensor resolution
        // =====================================================================
        TensorContext tensors;

        // Activation buffers
        tensors.buffers["hidden"] = buffers_.current_hidden;
        tensors.buffers["normalized"] = buffers_.layer_buffers.normalized;
        tensors.buffers["Q"] = buffers_.layer_buffers.Q;
        tensors.buffers["K"] = buffers_.layer_buffers.K;
        tensors.buffers["V"] = buffers_.layer_buffers.V;
        tensors.buffers["attn_output"] = buffers_.layer_buffers.attn_output;
        tensors.buffers["attn_proj"] = buffers_.layer_buffers.attn_proj;
        tensors.buffers["gate"] = buffers_.layer_buffers.gate;
        tensors.buffers["up"] = buffers_.layer_buffers.up;
        tensors.buffers["logits"] = buffers_.logits;
        tensors.buffers["logits_local"] = buffers_.logits_local;

        // Model-level weights
        tensors.model_weights["embedding_table"] = weights_.embedding_table;
        tensors.model_weights["final_norm"] = weights_.final_norm;
        tensors.model_weights["lm_head"] = weights_.lm_head;

        // Layer weight accessor
        tensors.get_layer_weight = [this](int layer_idx, const std::string &name) -> TensorBase *
        {
            if (!weights_.get_layer_weights)
                return nullptr;

            Qwen2LayerWeights layer = weights_.get_layer_weights(layer_idx);

            if (name == "wq")
                return layer.wq;
            if (name == "wk")
                return layer.wk;
            if (name == "wv")
                return layer.wv;
            if (name == "wo")
                return layer.wo;
            if (name == "attn_norm")
                return layer.attn_norm;
            if (name == "q_bias")
                return layer.q_bias;
            if (name == "k_bias")
                return layer.k_bias;
            if (name == "v_bias")
                return layer.v_bias;
            if (name == "gate_proj")
                return layer.gate_proj;
            if (name == "up_proj")
                return layer.up_proj;
            if (name == "down_proj")
                return layer.down_proj;
            if (name == "ffn_norm")
                return layer.ffn_norm;

            LOG_WARN("[TensorContext] Unknown layer weight: " << name);
            return nullptr;
        };

        // =====================================================================
        // Step 4: Resolve schema to concrete graph spec
        // =====================================================================
        GraphResolver resolver;
        ResolvedGraphSpec resolved = resolver.resolve(schema, config, tensors);

        LOG_DEBUG("[Qwen2Graph] Schema resolved: " << resolved.stages.size() << " stages"
                                                   << " (emitted=" << resolved.stats.stages_emitted
                                                   << ", skipped=" << resolved.stats.stages_skipped
                                                   << ", allreduce=" << resolved.stats.allreduce_inserted
                                                   << ", allgather=" << resolved.stats.allgather_inserted << ")");

        // =====================================================================
        // Step 5: Build the ComputeGraph
        // =====================================================================
        ComputeGraph graph = GraphBuilder::build(resolved);

        // Set output
        output.logits = buffers_.logits;

        LOG_DEBUG("[Qwen2Graph] Built schema-based forward graph with "
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
        // For HybridQ16: read from Q16_1 residual (auto-dispatches to Q16_1->FP32 kernel)
        // For other modes: read from FP32 current_hidden
        if (env.execution.exec_rmsnorm)
        {
            InferenceMode inference_mode(config_.activation_precision);

            RMSNormStage::Params attn_norm_params;
            attn_norm_params.input = (inference_mode.isHybridQ16() && buffers.residual)
                                         ? buffers.residual
                                         : buffers.current_hidden;
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

        // Create InferenceMode for centralized mode logic
        InferenceMode inference_mode(config_.activation_precision);

        // Check if we're in Hybrid mode with required buffers available
        bool use_hybrid_mode = isHybridModeActive(inference_mode, buffers);

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

            // Hybrid mode: output to separate FP32 buffers to avoid requantization
            if (use_hybrid_mode)
            {
                rope_params.Q_out = buffers.Q_rope;
                rope_params.K_out = buffers.K_rope;
                LOG_DEBUG("[Qwen2Graph] Layer " << layer_idx << " using Hybrid RoPE: Q8_1→FP32 output");
            }

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
        std::string wo_producer_node;
        if (env.execution.exec_attention)
        {
            // Phase 9 Decomposed Path: KVCacheAppendStage + AttentionComputeStage
            if (kv_cache)
            {
                // For batched execution, K/V are [batch_size * seq_len, kv_dim]
                // Each sequence's K/V is appended to its own seq_idx in the cache
                int total_tokens = batch_size * seq_len;

                KVCacheAppendStage::Params kv_append_params;

                // Hybrid mode: Use K_rope (post-RoPE FP32) instead of K (pre-RoPE Q8_1)
                // The KV cache in Hybrid mode stores post-RoPE values
                if (use_hybrid_mode && inference_mode.needsKRope())
                {
                    kv_append_params.K = buffers.K_rope;
                    LOG_DEBUG("[Qwen2Graph] Layer " << layer_idx
                                                    << " KVCacheAppend using K_rope (post-RoPE FP32)");
                }
                else
                {
                    kv_append_params.K = buffers.K;
                }

                kv_append_params.V = buffers.V;
                kv_append_params.kv_cache = kv_cache;
                kv_append_params.layer_idx = layer_idx;
                kv_append_params.seq_idx = 0; // Starting seq_idx
                kv_append_params.num_tokens = total_tokens;
                kv_append_params.batch_size = batch_size; // Phase 3: Per-sequence append
                kv_append_params.seq_len = seq_len;       // Phase 3: Tokens per sequence

                // Hybrid mode: populate V_dequant during KV cache append (V→FP32 conversion)
                // This avoids a separate dequantization pass since KVCacheAppend already converts V
                if (use_hybrid_mode && inference_mode.needsVDequant() && buffers.V_dequant)
                {
                    kv_append_params.V_dequant_out = buffers.V_dequant;
                    LOG_DEBUG("[Qwen2Graph] Layer " << layer_idx
                                                    << " KVCacheAppend will populate V_dequant for attention");
                }

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

            // For Hybrid mode attention path selection:
            // - Decomposed attention: Use FP32 K_rope/V_dequant for best precision
            // - Fused attention: Use Q8_1 K/V (kernel only supports Q8_1 K/V currently)
            // KV cache always gets FP32 K_rope (stored separately from attention path)
            //
            // For Q8_1 mode: always use Q8_1 K/V
            // For FP32 mode: always use FP32 K/V (but FP32 doesn't use fused attention)
            //
            // Note: Hybrid mode's main accuracy gain is from FP32 Q (via Q_rope) and
            // FP32 streaming dequant for Wo projection, not from FP32 K/V.
            TensorBase *K_for_attn = buffers.K; // Default to Q8_1
            TensorBase *V_for_attn = buffers.V; // Default to Q8_1

            // For decomposed attention path (FP32 or Hybrid without fused), use FP32 K/V if available
            // The fused attention path check below will override this for fused attention
            bool use_decomposed_fp32_kv = use_hybrid_mode && inference_mode.usesDecomposedFP32Attention() && buffers.V_dequant;
            if (use_decomposed_fp32_kv)
            {
                // Only set FP32 K/V here - will be reset to Q8_1 if fused attention is used
                K_for_attn = buffers.K_rope;
                V_for_attn = buffers.V_dequant;
            }

            int total_query_tokens = batch_size * seq_len;
            int kv_len = total_query_tokens; // Static hint for mode detection (actual queried at runtime)
            bool use_gather_stage = false;

            // For attention K/V source:
            // - Prefill (cached_tokens == 0): Use projected K/V directly [batch_size * seq_len, kv_dim]
            //   In Hybrid mode: K_rope (FP32) and V_dequant (FP32) from above
            // - Decode (cached_tokens > 0 and batch_size == 1): Use cache (single-sequence only)
            //   In Hybrid mode: KV cache is FP32
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

            // Determine if we can use Fused Attention + Wo
            // Requirements:
            // 1. fused_wo flag enabled (env.attention.fused_wo)
            // 2. exec_attention and exec_gemm stages enabled
            // 3. Wo weights available
            // 4. CPU backend (JIT kernel is CPU-only)
            // 5. Q8_1 or HybridQ16 activation precision
            //    - Q8_1: Uses Q8_1 Q directly, outputs to FP32
            //    - HybridQ16: Uses FP32 Q_rope, fuses Q16_1 residual add
            // Note: Regular Hybrid mode should use decomposed attention for FP32 attention scoring.
            // The fused path quantizes Q to Q8_1 and uses Q8_1 K/V, losing the precision benefit.
            // HybridQ16 enables fused path with Q16_1 residual fusion for better precision.
            bool use_fused_wo = env.attention.fused_wo &&
                                env.execution.exec_attention &&
                                env.execution.exec_gemm &&
                                layer.wo &&
                                backend == ComputeBackendType::CPU &&
                                (config_.activation_precision == ActivationPrecision::Q8_1 ||
                                 config_.activation_precision == ActivationPrecision::HybridQ16);

            LOG_TRACE("[Qwen2Graph] Layer " << layer_idx << " fused_wo check: env.fused_wo=" << env.attention.fused_wo
                                            << " exec_attention=" << env.execution.exec_attention
                                            << " exec_gemm=" << env.execution.exec_gemm
                                            << " has_wo=" << (layer.wo != nullptr)
                                            << " cpu_backend=" << (backend == ComputeBackendType::CPU)
                                            << " activation_precision=" << activationPrecisionToString(config_.activation_precision)
                                            << " => use_fused_wo=" << use_fused_wo);

            if (use_fused_wo)
            {
                LOG_DEBUG("[Qwen2Graph] Layer " << layer_idx << " using FusedAttentionWoStage");

                // FusedAttentionWoKernel only supports Q8_1 K/V currently
                // For Hybrid mode, we use Q8_1 K/V for attention but store FP32 to cache
                // Reset to Q8_1 K/V for fused attention path (overriding FP32 K/V if set above)
                TensorBase *K_for_fused = buffers.K;
                TensorBase *V_for_fused = buffers.V;

                // Check decode mode - need to get K/V from cache
                if (kv_cache)
                {
                    int cached_tokens = kv_cache->get_cached_tokens(layer_idx, 0);
                    if (cached_tokens > 0 && batch_size == 1)
                    {
                        // Decode mode: use cached K/V
                        // Note: In Hybrid mode, KV cache stores FP32 but fused kernel needs Q8_1
                        // This is a known limitation - Hybrid decode will fail until we add
                        // FP32 K/V support to FusedAttentionWoKernel
                        K_for_fused = kv_cache->get_k_base(layer_idx, 0);
                        V_for_fused = kv_cache->get_v_base(layer_idx, 0);
                    }
                }

                FusedAttentionWoStage::Params fused_params;
                // For Hybrid mode: use Q_rope (FP32) instead of Q (Q8_1)
                // The FusedAttentionWoKernel will detect FP32 Q and quantize on-the-fly
                fused_params.Q = (use_hybrid_mode && inference_mode.needsQRope()) ? buffers.Q_rope : buffers.Q;
                fused_params.K = K_for_fused;
                fused_params.V = V_for_fused;
                fused_params.Wo = layer.wo;
                fused_params.output = buffers.attn_proj; // Direct output to projection buffer
                fused_params.batch_size = batch_size;
                fused_params.seq_len = seq_len;
                fused_params.kv_len = kv_len;
                fused_params.n_heads = local_n_heads;
                fused_params.n_kv_heads = local_n_kv_heads;
                fused_params.head_dim = config_.head_dim;
                fused_params.d_model = config_.d_model;
                fused_params.causal = true;
                fused_params.position_offset = position_ids ? position_ids[0] : 0;
                fused_params.backend = FusedAttentionBackend::JIT;
                fused_params.kv_cache = kv_cache;
                fused_params.layer_idx = layer_idx;
                fused_params.mpi_ctx = mpi_ctx_.get();
                fused_params.device_idx = device_idx;

                // Hybrid/HybridQ16 mode: enable streaming dequantization for FP32-equivalent Wo projection
                // This gives highest numerical precision by dequantizing VNNI-packed weights to FP32
                // row-by-row during GEMM, avoiding the precision loss of quantizing FP32 context.
                fused_params.use_hybrid_wo = inference_mode.isAnyHybrid();

                // HybridQ16 mode: enable Q16_1 residual fusion
                // The JIT kernel will fuse residual addition after Wo projection:
                // - Wo output stays in registers as FP32
                // - Q16_1 residual is loaded, dequantized, added
                // - Result is quantized to Q16_1 and stored directly
                // This eliminates FP32 intermediate memory traffic (2.8× reduction).
                if (inference_mode.isHybridQ16())
                {
                    fused_params.fuse_residual_add = true;
                    fused_params.output = buffers.residual; // Output directly to Q16_1 residual
                    LOG_DEBUG("[Qwen2Graph] Layer " << layer_idx << " HybridQ16: fused residual add enabled");
                }

                // Optional context snapshot for debugging (enabled via buffer allocation)
                fused_params.context_snapshot = buffers.context_snapshot;
                LOG_DEBUG("[Qwen2Graph] Layer " << layer_idx << " fused_attn_wo context_snapshot="
                                                << (buffers.context_snapshot ? "allocated" : "NULL")
                                                << " use_hybrid_wo=" << fused_params.use_hybrid_wo
                                                << " fuse_residual_add=" << fused_params.fuse_residual_add);

                wo_producer_node = prefix + "fused_attn_wo";
                graph.addNode(wo_producer_node,
                              ComputeStageFactory::createFusedAttentionWo(fused_params),
                              device_idx);

                if (use_gather_stage)
                    graph.addDependency(wo_producer_node, prefix + "kv_gather");
                else if (kv_cache)
                    graph.addDependency(wo_producer_node, prefix + "kv_append");
                else if (env.execution.exec_rope)
                    graph.addDependency(wo_producer_node, prefix + "rope");
                else if (env.execution.exec_gemm)
                    graph.addDependency(wo_producer_node, prefix + "qkv_proj");
            }
            else
            {
                // Standard Decomposed Path: Attention -> Wo
                if (env.execution.exec_attention)
                {
                    AttentionMode mode = detect_attention_mode(batch_size, seq_len, kv_len);
                    LOG_TRACE("[Qwen2Graph] Layer " << layer_idx
                                                    << " attention mode: " << attention_mode_name(mode)
                                                    << " (batch_size=" << batch_size << ", seq_len=" << seq_len << ", kv_len=" << kv_len << ")");

                    AttentionComputeStage::Params attn_params;
                    // For Hybrid mode: use Q_rope (FP32) instead of Q (Q8_1)
                    attn_params.Q = (use_hybrid_mode && inference_mode.needsQRope()) ? buffers.Q_rope : buffers.Q;
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
                        graph.addDependency(prefix + "attention", prefix + "kv_gather");
                    else if (kv_cache)
                        graph.addDependency(prefix + "attention", prefix + "kv_append");
                    else if (env.execution.exec_rope)
                        graph.addDependency(prefix + "attention", prefix + "rope");
                    else if (env.execution.exec_gemm)
                        graph.addDependency(prefix + "attention", prefix + "qkv_proj");

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

                    wo_producer_node = prefix + "wo_proj";
                    graph.addNode(wo_producer_node,
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

                    if (env.execution.exec_attention)
                    {
                        graph.addDependency(wo_producer_node, prefix + "attention");
                    }
                }
            }

            // Common AllReduce for Wo
            if (env.execution.exec_gemm && layer.wo && !wo_producer_node.empty())
            {
                bool wo_is_sharded = isRowParallelSharded(layer.wo);
                bool has_multi_rank = hasMultipleRanks(mpi_ctx_.get());

                if (wo_is_sharded && has_multi_rank)
                {
                    // AllReduce the actual data, not full buffer capacity
                    size_t allreduce_count = static_cast<size_t>(total_tokens) * static_cast<size_t>(config_.d_model);

                    // For HybridQ16 fusion: allreduce the Q16_1 residual buffer
                    // For standard path: allreduce the FP32 attn_proj buffer
                    TensorBase *allreduce_buffer = inference_mode.isHybridQ16()
                                                       ? buffers.residual
                                                       : buffers.attn_proj;

                    graph.addNode(prefix + "wo_allreduce",
                                  ComputeStageFactory::createAllreduce(
                                      AllreduceStage::Params{
                                          allreduce_buffer,
                                          getMPICommPtr(mpi_ctx_.get()),
                                          allreduce_count}),
                                  device_idx);

                    graph.addDependency(prefix + "wo_allreduce", wo_producer_node);
                    wo_producer_node = prefix + "wo_allreduce";

                    LOG_TRACE("[Qwen2Graph] Layer " << layer_idx
                                                    << " Wo: row-parallel sharded, adding allreduce"
                                                    << (inference_mode.isHybridQ16() ? " (Q16_1 residual)" : " (FP32 proj)"));
                }
            }
        }

        // Stage 6: Residual connection
        // Skip when HybridQ16 fusion is enabled - the fused kernel already did residual add
        if (env.execution.exec_residual && !inference_mode.isHybridQ16())
        {
            ResidualAddStage::Params res_params;
            res_params.input = buffers.attn_proj;
            res_params.residual = buffers.current_hidden;
            res_params.output = buffers.current_hidden;
            res_params.num_elements = static_cast<size_t>(total_tokens) * static_cast<size_t>(config_.d_model);

            graph.addNode(prefix + "attn_residual",
                          ComputeStageFactory::createResidualAdd(res_params),
                          device_idx);

            if (env.execution.exec_gemm && layer.wo && !wo_producer_node.empty())
            {
                graph.addDependency(prefix + "attn_residual", wo_producer_node);
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
            // For HybridQ16, read from Q16_1 residual buffer
            InferenceMode inference_mode(config_.activation_precision);

            RMSNormStage::Params ffn_norm_params;
            ffn_norm_params.input = (inference_mode.isHybridQ16() && buffers.residual)
                                        ? buffers.residual
                                        : buffers.current_hidden;
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
            bool has_multi_rank = hasMultipleRanks(mpi_ctx_.get());

            if (needs_allreduce && has_multi_rank)
            {
                size_t allreduce_count = static_cast<size_t>(total_tokens) * down_n;
                MPI_Comm comm = static_cast<MPI_Comm>(getMPICommPtr(mpi_ctx_.get()));

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
            // For HybridQ16, FFN residual uses Q16_1 residual buffer
            InferenceMode inference_mode_ffn_res(config_.activation_precision);

            ResidualAddStage::Params res_params;
            res_params.input = buffers.attn_proj;
            res_params.residual = (inference_mode_ffn_res.isHybridQ16() && buffers.residual)
                                      ? buffers.residual
                                      : buffers.current_hidden;
            res_params.output = (inference_mode_ffn_res.isHybridQ16() && buffers.residual)
                                    ? buffers.residual
                                    : buffers.current_hidden;
            res_params.num_elements = static_cast<size_t>(total_tokens) * static_cast<size_t>(config_.d_model);

            graph.addNode(prefix + "ffn_residual",
                          ComputeStageFactory::createResidualAdd(res_params),
                          device_idx);

            if (env.execution.exec_gemm && layer.down_proj)
            {
                bool down_is_row_sharded = isRowParallelSharded(layer.down_proj);
                bool needs_allreduce = (down_is_row_sharded || config_.ffn_column_parallel);
                bool has_multi_rank = hasMultipleRanks(mpi_ctx_.get());

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

    GraphSchema Qwen2Graph::getSchema() const
    {
        Qwen2SchemaFactory factory;
        return factory.createSchema();
    }

    GraphResolverConfig Qwen2Graph::getResolverConfig(int seq_len) const
    {
        GraphResolverConfig config;

        // MPI context
        if (mpi_ctx_)
        {
            config.world_size = mpi_ctx_->world_size();
            config.rank = mpi_ctx_->rank();
            config.mpi_comm = mpi_ctx_->comm();
            config.mpi_ctx = mpi_ctx_.get();
        }

        // Sequence configuration
        config.seq_len = seq_len;
        config.batch_size = 1;

        // Device configuration
        config.default_device = config_.default_device;

        // Model architecture
        config.n_layers = config_.n_layers;
        config.d_model = config_.d_model;
        config.n_heads = config_.n_heads;
        config.n_kv_heads = config_.n_kv_heads;
        config.head_dim = config_.head_dim;
        config.d_ff = config_.d_ff;
        config.vocab_size = config_.vocab_size;
        config.rms_norm_eps = config_.rms_norm_eps;
        config.rope_theta = config_.rope_theta;

        // TP-adjusted local dimensions
        // Use local head counts when QKV is column-parallel
        config.local_n_heads = config_.qkv_column_parallel
                                   ? config_.local_n_heads
                                   : config_.n_heads;
        config.local_n_kv_heads = config_.qkv_column_parallel
                                      ? config_.local_n_kv_heads
                                      : config_.n_kv_heads;

        // Validate (safety check for uninitialized config)
        if (config.local_n_heads <= 0)
            config.local_n_heads = config_.n_heads;
        if (config.local_n_kv_heads <= 0)
            config.local_n_kv_heads = config_.n_kv_heads;

        // Use local FFN dimension when FFN is column-parallel
        config.local_d_ff = config_.ffn_column_parallel
                                ? config_.d_ff_local
                                : config_.d_ff;
        if (config.local_d_ff <= 0)
            config.local_d_ff = config_.d_ff;

        // Use local vocab when LM head is column-parallel
        config.local_vocab = config_.lm_head_column_parallel
                                 ? config_.vocab_local
                                 : config_.vocab_size;
        if (config.local_vocab <= 0)
            config.local_vocab = config_.vocab_size;

        LOG_DEBUG("[Qwen2Graph::getResolverConfig] Created config: "
                  << "seq_len=" << config.seq_len << ", "
                  << "local_n_heads=" << config.local_n_heads << " (n_heads=" << config.n_heads << "), "
                  << "local_n_kv_heads=" << config.local_n_kv_heads << " (n_kv_heads=" << config.n_kv_heads << "), "
                  << "local_d_ff=" << config.local_d_ff << " (d_ff=" << config.d_ff << "), "
                  << "local_vocab=" << config.local_vocab << " (vocab_size=" << config.vocab_size << ")");

        return config;
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

} // namespace llaminar2
