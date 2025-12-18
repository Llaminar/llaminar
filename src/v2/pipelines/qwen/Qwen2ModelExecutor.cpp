/**
 * @file Qwen2ModelExecutor.cpp
 * @brief Implementation of Qwen2-specific model executor
 * @author David Sanftenberg
 * @date December 2025
 */

#include "Qwen2ModelExecutor.h"
#include "../../utils/Logger.h"
#include <stdexcept>
#include <chrono>

namespace llaminar2
{

    // =========================================================================
    // Constructor
    // =========================================================================

    Qwen2ModelExecutor::Qwen2ModelExecutor(std::shared_ptr<ModelContext> model_ctx,
                                           std::shared_ptr<MPIContext> mpi_ctx,
                                           const Qwen2ModelExecutorConfig &config)
        : ModelExecutor(model_ctx, mpi_ctx, config.base_config), qwen2_config_(config)
    {
        LOG_DEBUG("[Qwen2ModelExecutor] Initializing for "
                  << config.n_layers << " layers, d_model=" << config.d_model
                  << ", vocab_size=" << config.vocab_size);

        // Build Qwen2ExecutorConfig from Qwen2ModelExecutorConfig
        Qwen2ExecutorConfig executor_config;
        executor_config.n_heads = config.n_heads;
        executor_config.n_kv_heads = config.n_kv_heads;
        executor_config.head_dim = config.head_dim;
        executor_config.d_model = config.d_model;
        executor_config.d_ff = config.d_ff;
        executor_config.ffn_column_parallel = config.ffn_column_parallel;
        executor_config.rms_norm_eps = config.rms_norm_eps;
        executor_config.rope_theta = config.rope_theta;
        executor_config.activation_precision = config.activation_precision;
        executor_config.default_device = config.base_config.layer_config.default_device;

        // Create layer executor for building per-layer graphs
        layer_executor_ = std::make_unique<Qwen2LayerExecutor>(executor_config, mpi_ctx);

        LOG_INFO("[Qwen2ModelExecutor] Initialized with " << config.n_layers
                                                          << " layers, precision="
                                                          << (config.activation_precision == ActivationPrecision::Q8_1 ? "Q8_1" : "FP32"));
    }

    // =========================================================================
    // Full Forward Graph
    // =========================================================================

    ComputeGraph Qwen2ModelExecutor::buildFullForwardGraph(
        const ForwardInput &input,
        ForwardOutput &output)
    {
        LOG_DEBUG("[Qwen2ModelExecutor] Building full forward graph: "
                  << "batch_size=" << input.batch_size
                  << ", seq_len=" << input.seq_len);

        if (!weights_.embedding_table || !weights_.final_norm || !weights_.lm_head)
        {
            LOG_ERROR("[Qwen2ModelExecutor] Weights not set! Call setWeights() first.");
            throw std::runtime_error("Qwen2ModelExecutor weights not initialized");
        }

        if (!buffers_.current_hidden || !buffers_.logits)
        {
            LOG_ERROR("[Qwen2ModelExecutor] Buffers not set! Call setBuffers() first.");
            throw std::runtime_error("Qwen2ModelExecutor buffers not initialized");
        }

        // Store current dimensions
        current_batch_size_ = input.batch_size;
        current_seq_len_ = input.seq_len;

        int device_idx = qwen2_config_.base_config.layer_config.default_device;
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
        embed_params.d_model = qwen2_config_.d_model;
        embed_params.vocab_size = qwen2_config_.vocab_size;
        embed_params.device_idx = device_idx;

        graph.addNode("embedding",
                      ComputeStageFactory::createEmbedding(embed_params),
                      device_idx);

        // -------------------------------------------------------------------------
        // Stage 2: Transformer Layers (as opaque execution nodes)
        // -------------------------------------------------------------------------
        std::string prev_node = "embedding";

        // Build position IDs
        position_ids_buffer_ = buildPositionIds(input.seq_len, input.batch_size, input.position_offset);

        // For now, we represent each layer as a placeholder node.
        // The actual execution will use Qwen2LayerExecutor::executeLayer.
        // In Phase 2, we can inline attention+FFN subgraphs for full declarativity.
        for (int layer = 0; layer < qwen2_config_.n_layers; ++layer)
        {
            std::string layer_name = "layer_" + std::to_string(layer);

            // Add node with nullptr stage - serves as a sequencing marker
            // Note: The graph will skip execution of nullptr stages
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
        lm_params.d_model = qwen2_config_.d_model;
        lm_params.vocab_size = qwen2_config_.vocab_size;
        lm_params.bias = nullptr; // Qwen2 has no LM head bias
        lm_params.device_idx = device_idx;

        graph.addNode("lm_head",
                      ComputeStageFactory::createLMHead(lm_params),
                      device_idx);
        graph.addDependency("lm_head", prev_node);

        // Set output
        output.logits = buffers_.logits;

        LOG_DEBUG("[Qwen2ModelExecutor] Built full forward graph with "
                  << graph.size() << " nodes");

        return graph;
    }

    // =========================================================================
    // Embedding Graph
    // =========================================================================

    ComputeGraph Qwen2ModelExecutor::buildEmbeddingGraph(
        const ForwardInput &input,
        TensorBase *output_hidden)
    {
        LOG_DEBUG("[Qwen2ModelExecutor] Building embedding graph for "
                  << (input.batch_size * input.seq_len) << " tokens");

        ComputeGraph graph;

        EmbeddingStage::Params params;
        params.embed_table = weights_.embedding_table;
        params.token_ids = input.token_ids;
        params.output = output_hidden;
        params.num_tokens = input.batch_size * input.seq_len;
        params.d_model = qwen2_config_.d_model;
        params.vocab_size = qwen2_config_.vocab_size;
        params.device_idx = qwen2_config_.base_config.layer_config.default_device;

        graph.addNode("embedding",
                      ComputeStageFactory::createEmbedding(params),
                      qwen2_config_.base_config.layer_config.default_device);

        return graph;
    }

    // =========================================================================
    // Transformer Layers Graph
    // =========================================================================

    ComputeGraph Qwen2ModelExecutor::buildTransformerLayersGraph(
        TensorBase *input_hidden,
        IUnifiedKVCache *kv_cache,
        const int *position_ids,
        int device_idx)
    {
        LOG_DEBUG("[Qwen2ModelExecutor] Building transformer layers graph: "
                  << qwen2_config_.n_layers << " layers");

        ComputeGraph graph;
        std::string prev_node;

        // Build placeholder nodes for layer sequencing
        // Each layer will be executed via layer_executor_->executeLayer() at runtime
        for (int layer = 0; layer < qwen2_config_.n_layers; ++layer)
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

    // =========================================================================
    // Single Layer Graph
    // =========================================================================

    ComputeGraph Qwen2ModelExecutor::buildLayerGraph(
        int layer_idx,
        TensorBase *input_hidden,
        IUnifiedKVCache *kv_cache,
        const int *position_ids,
        int device_idx)
    {
        LOG_DEBUG("[Qwen2ModelExecutor] Building layer " << layer_idx << " graph");

        // Build a minimal graph with just a layer placeholder
        // Actual execution uses layer_executor_->executeLayer()
        ComputeGraph graph;

        // Add placeholder node with nullptr stage
        graph.addNode("layer_" + std::to_string(layer_idx), nullptr, device_idx);

        return graph;
    }

    // =========================================================================
    // LM Head Graph
    // =========================================================================

    ComputeGraph Qwen2ModelExecutor::buildLMHeadGraph(
        TensorBase *hidden_states,
        TensorBase *output_logits,
        int device_idx)
    {
        LOG_DEBUG("[Qwen2ModelExecutor] Building LM head graph");

        ComputeGraph graph;
        int total_tokens = current_batch_size_ * current_seq_len_;

        // Final RMSNorm
        RMSNormStage::Params norm_params;
        norm_params.input = hidden_states;
        norm_params.output = hidden_states; // In-place norm
        norm_params.gamma = weights_.final_norm;
        norm_params.eps = qwen2_config_.rms_norm_eps;
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
        lm_params.d_model = qwen2_config_.d_model;
        lm_params.vocab_size = qwen2_config_.vocab_size;
        lm_params.bias = nullptr;
        lm_params.device_idx = device_idx;

        graph.addNode("lm_head",
                      ComputeStageFactory::createLMHead(lm_params),
                      device_idx);
        graph.addDependency("lm_head", "final_norm");

        return graph;
    }

    // =========================================================================
    // Helper: Build Position IDs
    // =========================================================================

    std::vector<int> Qwen2ModelExecutor::buildPositionIds(int seq_len, int batch_size, int offset)
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

    // =========================================================================
    // Helper: Add Final Norm
    // =========================================================================

    void Qwen2ModelExecutor::addFinalNormToGraph(
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
        norm_params.eps = qwen2_config_.rms_norm_eps;
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

    // =========================================================================
    // State Management
    // =========================================================================

    void Qwen2ModelExecutor::clearCache()
    {
        ModelExecutor::clearCache();

        current_batch_size_ = 0;
        current_seq_len_ = 0;
        position_ids_buffer_.clear();

        LOG_DEBUG("[Qwen2ModelExecutor] Cache cleared");
    }

} // namespace llaminar2
