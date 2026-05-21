/**
 * @file IModelExecutor.h
 * @brief Interface for model-level execution orchestration
 * @author David Sanftenberg
 * @date December 2025
 *
 * IModelExecutor defines the abstract interface for executing complete
 * transformer forward passes using declarative compute graphs. This interface
 * enables the gradual migration from imperative pipeline code to fully
 * declarative graph-based execution.
 *
 * Key Design Goals:
 * - **Full Forward Pass**: Handles embedding, all transformer layers, and LM head
 * - **Declarative Graphs**: All operations expressed as ComputeGraph DAGs
 * - **Composable**: Builds on LayerExecutor for layer-level parallelism
 * - **Testable**: Interface enables mocking and alternative implementations
 * - **Gradual Migration**: Pipelines can incrementally adopt executor patterns
 *
 * Architecture:
 * @code
 *   Pipeline
 *      |
 *      v
 *   IModelExecutor  <-- This interface
 *      |
 *      +-- buildFullForwardGraph() -> ComputeGraph
 *      +-- buildEmbeddingGraph() -> ComputeGraph
 *      +-- buildTransformerLayersGraph() -> ComputeGraph
 *      +-- buildLMHeadGraph() -> ComputeGraph
 *      |
 *      v
 *   ILayerExecutor (for per-layer orchestration)
 *      |
 *      v
 *   ComputeStage[] (device-specific implementations)
 * @endcode
 *
 * Migration Path from Pipeline:
 * 1. Pipeline::forward() calls ModelExecutor::executeForward()
 * 2. ModelExecutor builds graph and executes via LayerExecutor
 * 3. Eventually, Pipeline becomes thin wrapper around ModelExecutor
 */

#pragma once

#include "ILayerExecutor.h"
#include "../../compute_stages/ComputeStages.h"
#include "../device/DeviceContext.h"
#include "../../../backends/DeviceId.h"
#include <memory>
#include <vector>
#include <string>
#include <functional>

namespace llaminar2
{

    // Forward declarations
    class ComputeGraph;
    class TensorBase;
    class ICPUKVCache;
    class IMPIContext;

    /**
     * @brief Configuration for ModelExecutor
     *
     * Extends LayerExecutorConfig with model-level settings.
     */
    struct ModelExecutorConfig
    {
        // Inherit layer executor settings
        LayerExecutorConfig layer_config;

        // Model-level settings
        int n_layers = 0;            ///< Number of transformer layers
        int max_batch_size = 1;      ///< Maximum batch size supported
        int max_seq_len = 2048;      ///< Maximum sequence length
        bool enable_kv_cache = true; ///< Enable KV cache (disable for testing)

        // Execution granularity
        bool fuse_embedding = true;    ///< Include embedding in forward graph
        bool fuse_lm_head = true;      ///< Include LM head in forward graph
        bool per_layer_graphs = false; ///< Build separate graphs per layer (for debugging)

        // Profiling
        bool profile_per_phase = false; ///< Profile embedding/layers/lm_head separately
    };

    /**
     * @brief Model execution statistics
     *
     * Extends LayerExecutorStats with model-level metrics.
     */
    struct ModelExecutorStats
    {
        // Phase timing
        double embedding_time_ms = 0.0;
        double layers_time_ms = 0.0;
        double lm_head_time_ms = 0.0;
        double total_time_ms = 0.0;

        // Token throughput
        size_t total_tokens_processed = 0;
        size_t total_forward_passes = 0;

        // Aggregate layer stats
        LayerExecutorStats layer_stats;

        void reset()
        {
            embedding_time_ms = 0.0;
            layers_time_ms = 0.0;
            lm_head_time_ms = 0.0;
            total_time_ms = 0.0;
            total_tokens_processed = 0;
            total_forward_passes = 0;
            layer_stats.reset();
        }
    };

    /**
     * @brief Input specification for forward pass
     *
     * Encapsulates all inputs needed for a forward pass, supporting
     * both single-sequence and batched execution.
     */
    struct ForwardInput
    {
        // Token input (one of these must be set)
        const int *token_ids = nullptr;                         ///< Token IDs [seq_len] (single sequence)
        const std::vector<std::vector<int>> *batches = nullptr; ///< Batched tokens

        // Dimensions
        int batch_size = 1;
        int seq_len = 0;            ///< Per-sequence length (single) or max length (batched)
        int *seq_lengths = nullptr; ///< Per-sequence lengths for batched (nullptr = all same)
        int real_seq_len = 0;       ///< Real tokens in a bucketed prefill chunk (0 = seq_len)
        int bucket_seq_len = 0;     ///< Fixed bucket length for graph shape (0 = seq_len)
        int token_offset = 0;       ///< Chunk offset within the original prompt

        // Position information (for decode mode)
        int *position_ids = nullptr; ///< Position IDs for RoPE (nullptr = auto)
        int position_offset = 0;     ///< Offset for decode mode (current position)

        // KV cache state
        IKVCache *kv_cache = nullptr; ///< KV cache (nullptr = no caching)

        // Device placement
        DeviceId device = DeviceId::cpu(); ///< Target device
    };

    /**
     * @brief Output specification for forward pass
     *
     * Encapsulates all outputs from a forward pass.
     */
    struct ForwardOutput
    {
        // Output tensors (set by executor)
        TensorBase *logits = nullptr;        ///< Output logits [batch_size * seq_len, vocab_size]
        TensorBase *hidden_states = nullptr; ///< Final hidden states (if requested)

        // Flags
        bool return_hidden_states = false; ///< Whether to return hidden states
    };

    /**
     * @brief Abstract interface for model-level execution
     *
     * IModelExecutor defines the contract for components that execute
     * complete forward passes through declarative compute graphs.
     *
     * Usage:
     * @code
     * // In production
     * auto executor = Qwen2ModelExecutor::create(model_ctx, mpi_ctx, config);
     *
     * ForwardInput input{.token_ids = tokens, .seq_len = seq_len};
     * ForwardOutput output{.logits = logits_buffer};
     *
     * executor->executeForward(input, output);
     * @endcode
     */
    class IModelExecutor
    {
    public:
        virtual ~IModelExecutor() = default;

        // =========================================================================
        // Configuration
        // =========================================================================

        /**
         * @brief Get current configuration
         */
        virtual const ModelExecutorConfig &config() const = 0;

        /**
         * @brief Set snapshot callback for debugging
         *
         * Propagates to underlying layer executor for stage-level capture.
         */
        virtual void setSnapshotCallback(StageSnapshotCallback callback) = 0;

        // =========================================================================
        // Graph Building
        // =========================================================================

        /**
         * @brief Build compute graph for full forward pass
         *
         * Creates a DAG containing:
         * 1. Embedding lookup
         * 2. All transformer layers (attention + FFN)
         * 3. Final normalization
         * 4. LM head projection
         *
         * @param input Forward input specification
         * @param output Forward output specification
         * @return Complete compute graph
         */
        virtual ComputeGraph buildFullForwardGraph(
            const ForwardInput &input,
            ForwardOutput &output) = 0;

        /**
         * @brief Build compute graph for embedding only
         *
         * Useful for debugging or when embedding runs on different device.
         *
         * @param input Forward input specification
         * @param output_hidden Output tensor for hidden states
         * @return Embedding compute graph
         */
        virtual ComputeGraph buildEmbeddingGraph(
            const ForwardInput &input,
            TensorBase *output_hidden) = 0;

        /**
         * @brief Build compute graph for all transformer layers
         *
         * @param input_hidden Hidden states from embedding
         * @param kv_cache KV cache for attention
         * @param position_ids Position IDs for RoPE
         * @param device Target device
         * @return Transformer layers compute graph
         */
        virtual ComputeGraph buildTransformerLayersGraph(
            TensorBase *input_hidden,
            IKVCache *kv_cache,
            const int *position_ids,
            DeviceId device) = 0;

        /**
         * @brief Build compute graph for single transformer layer
         *
         * Used when per_layer_graphs is enabled for debugging.
         *
         * @param layer_idx Layer index (0-based)
         * @param input_hidden Hidden states input
         * @param kv_cache KV cache
         * @param position_ids Position IDs
         * @param device Target device
         * @return Single layer compute graph
         */
        virtual ComputeGraph buildLayerGraph(
            int layer_idx,
            TensorBase *input_hidden,
            IKVCache *kv_cache,
            const int *position_ids,
            DeviceId device) = 0;

        /**
         * @brief Build compute graph for LM head
         *
         * @param hidden_states Final hidden states
         * @param output_logits Output tensor for full logits [seq_len, vocab_size]
         * @param total_tokens Number of tokens (batch_size * seq_len)
         * @param device Target device
         * @param logits_local Optional local logits buffer for column-parallel LM head.
         *                     When provided and column-parallel is enabled, LM head
         *                     writes to logits_local then AllGather to output_logits.
         * @return LM head compute graph
         */
        virtual ComputeGraph buildLMHeadGraph(
            TensorBase *hidden_states,
            TensorBase *output_logits,
            int total_tokens,
            DeviceId device,
            TensorBase *logits_local = nullptr) = 0;

        // =========================================================================
        // Execution
        // =========================================================================

        /**
         * @brief Execute complete forward pass
         *
         * This is the primary entry point. Builds and executes the full
         * forward graph in one call.
         *
         * @param input Forward input specification
         * @param output Forward output specification
         * @return true on success
         */
        virtual bool executeForward(
            const ForwardInput &input,
            ForwardOutput &output) = 0;

        /**
         * @brief Execute a pre-built compute graph
         *
         * Lower-level interface for when you want to build graphs manually.
         *
         * @param graph Compute graph to execute
         * @param ctx Device context
         * @return true on success
         */
        virtual bool execute(ComputeGraph &graph, IDeviceContext *ctx) = 0;

        // =========================================================================
        // Statistics
        // =========================================================================

        /**
         * @brief Get execution statistics
         */
        virtual const ModelExecutorStats &stats() const = 0;

        /**
         * @brief Reset statistics
         */
        virtual void resetStats() = 0;

        // =========================================================================
        // State Management
        // =========================================================================

        /**
         * @brief Clear KV cache and reset state
         *
         * Call between independent inference runs.
         */
        virtual void clearCache() = 0;
    };

} // namespace llaminar2
