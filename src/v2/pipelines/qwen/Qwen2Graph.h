/**
 * @file Qwen2Graph.h
 * @brief Qwen2 compute graph builder for declarative forward pass execution
 * @author David Sanftenberg
 * @date December 2025
 *
 * Qwen2Graph is the unified graph builder for Qwen2 architecture models,
 * combining the functionality previously split across Qwen2LayerExecutor
 * and Qwen2ModelExecutor.
 *
 * Key Design:
 * - This class BUILDS compute graphs (it's a graph builder, not executor)
 * - The actual execution is delegated to GraphExecutor
 * - Supports both model-level and layer-level graph construction
 *
 * Graph Building Methods:
 * - buildFullForwardGraph(): Complete embedding → layers → LM head
 * - buildEmbeddingGraph(): Token embedding lookup
 * - buildTransformerLayersGraph(): All transformer layers
 * - buildLayerGraph(): Single transformer layer
 * - buildAttentionGraph(): Attention block within a layer
 * - buildFFNGraph(): FFN block within a layer
 * - buildLMHeadGraph(): Final projection to vocabulary
 *
 * Migration Path:
 * - Qwen2Pipeline creates Qwen2Graph
 * - Qwen2Pipeline::forward() delegates to executeForward()
 * - Eventually, all forward logic moves to this graph-based approach
 */

#pragma once

#include "../../execution/GraphExecutor.h"
#include "../../execution/GraphBufferManager.h"
#include "../../execution/ComputeStage.h"
#include "../../execution/DeviceContext.h"
#include "../../execution/ExecutionPolicy.h"
#include "../../execution/IGraphBuilder.h"
#include "../../pipelines/RuntimeConfig.h"
#include "../../tensors/Tensors.h"
#include "../../tensors/TensorFactory.h"
#include "../../tensors/UnifiedKVCache.h"
#include "../../loaders/ModelContext.h"
#include "../../utils/MPIContext.h"
#include "Qwen2BufferSpec.h"
#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <unordered_map>

namespace llaminar2
{

    // Forward declarations
    class Qwen2Pipeline;

    // =========================================================================
    // Configuration
    // =========================================================================

    /**
     * @brief Configuration for Qwen2Graph
     *
     * Combines architecture parameters with execution settings.
     */
    struct Qwen2GraphConfig
    {
        // Model architecture
        int n_layers = 0;   ///< Number of transformer layers
        int d_model = 0;    ///< Model hidden dimension
        int n_heads = 0;    ///< Number of attention heads
        int n_kv_heads = 0; ///< Number of KV heads (GQA)
        int head_dim = 0;   ///< Dimension per head
        int d_ff = 0;       ///< FFN intermediate dimension
        int vocab_size = 0; ///< Vocabulary size

        // FFN sharding (for tensor parallelism)
        int d_ff_local = 0; ///< Local FFN dim per rank
        bool ffn_column_parallel = false;

        // =================================================================
        // LM Head TP Parameters (Phase 5: Column-Parallel LM Head)
        // =================================================================
        /// Local vocab size per rank (vocab_size / world_size)
        int vocab_local = 0;

        /// Enable column-parallel LM head projection (weights sharded by vocab)
        bool lm_head_column_parallel = false;

        // =================================================================
        // Attention TP Parameters (Phase 3: Column-Parallel QKV)
        // =================================================================
        /// First query head for this rank (0-indexed, default 0 = no sharding)
        int head_start = 0;

        /// Number of query heads for this rank (default -1 = use full n_heads)
        int local_n_heads = -1;

        /// Number of KV heads for this rank (default -1 = use full n_kv_heads)
        /// For GQA: may equal local_n_heads / gqa_ratio
        int local_n_kv_heads = -1;

        /// Enable column-parallel QKV projection (weights sharded by head)
        bool qkv_column_parallel = false;

        // Precision and execution
        float rms_norm_eps = 1e-6f;
        float rope_theta = 10000.0f;
        ActivationPrecision activation_precision = ActivationPrecision::FP32;

        // Execution settings
        int default_device = 0;
        bool enable_profiling = false;
        bool enable_validation = false;

        // NOTE: Decomposed attention (Phase 9: KVCacheAppendStage + AttentionComputeStage)
        // is now the ONLY supported path. The legacy AttentionWithKVCacheStage path has been
        // removed as part of Phase 7 cleanup. See DISTRIBUTED_ARCHITECTURE_PROPOSAL.md.

        /// Use graph-managed buffer allocation with aliasing optimization.
        /// When true, Qwen2Graph will use GraphBufferManager to allocate activation
        /// buffers with automatic aliasing of non-overlapping SCRATCH buffers.
        /// NOTE: As of December 2025, this defaults to true (Graph is primary path).
        bool use_graph_buffer_management = true;

        /// Maximum sequence length for buffer allocation (when use_graph_buffer_management=true)
        int max_seq_len = 2048;

        /// Execution policy controlling which operations run
        ExecutionPolicy execution_policy = ExecutionPolicy::allEnabled();

        /// Base GraphExecutor configuration
        GraphExecutorConfig executor_config = GraphExecutorConfig{};
    };

    // =========================================================================
    // Weight Structures
    // =========================================================================

    /**
     * @brief Layer weights for attention and FFN blocks
     *
     * Raw pointers since Qwen2Graph does NOT own these weights.
     */
    struct Qwen2LayerWeights
    {
        // Attention weights
        TensorBase *wq = nullptr;        ///< Query projection
        TensorBase *wk = nullptr;        ///< Key projection
        TensorBase *wv = nullptr;        ///< Value projection
        TensorBase *wo = nullptr;        ///< Output projection
        TensorBase *attn_norm = nullptr; ///< Pre-attention norm gamma

        // Attention biases (Qwen2 uses Q/K/V biases)
        TensorBase *q_bias = nullptr; ///< Query bias [d_model]
        TensorBase *k_bias = nullptr; ///< Key bias [n_kv_heads * head_dim]
        TensorBase *v_bias = nullptr; ///< Value bias [n_kv_heads * head_dim]

        // FFN weights
        TensorBase *gate_proj = nullptr; ///< FFN gate projection
        TensorBase *up_proj = nullptr;   ///< FFN up projection
        TensorBase *down_proj = nullptr; ///< FFN down projection
        TensorBase *ffn_norm = nullptr;  ///< Pre-FFN norm gamma
    };

    /**
     * @brief Model-level weights
     *
     * Provides access to global weights and per-layer accessor.
     */
    struct Qwen2ModelWeights
    {
        TensorBase *embedding_table = nullptr; ///< [vocab_size, d_model]
        TensorBase *final_norm = nullptr;      ///< [d_model]
        TensorBase *lm_head = nullptr;         ///< [vocab_size, d_model]

        /// Accessor for per-layer weights
        std::function<Qwen2LayerWeights(int layer_idx)> get_layer_weights;
    };

    // =========================================================================
    // Activation Buffers
    // =========================================================================

    /**
     * @brief Activation buffers for layer execution
     *
     * ## Buffer Lifecycle Semantics
     *
     * **INOUT (Input modified in-place)**:
     *   - `residual`: Accumulates attention/FFN outputs via +=
     *   - `normalized`: Receives norm output, may be reused
     *
     * **SCRATCH (Temporary workspace)**:
     *   - `Q`, `K`, `V`: Projection outputs, consumed by attention
     *   - `attn_output`: Pre-Wo output, consumed by projection
     *   - `gate`, `up`: FFN projections, consumed by SwiGLU
     *   - `ffn_output`: SwiGLU output, consumed by down projection
     *   - `workspace_scores`, `workspace_context`, `workspace_mask`: Attention scratch
     *
     * **OUTPUT (Write-only)**:
     *   - `current_hidden`: Final hidden state output
     *   - `attn_proj`: After Wo projection, before residual add
     */
    struct Qwen2ActivationBuffers
    {
        // === INOUT Buffers ===
        TensorBase *residual = nullptr;
        TensorBase *normalized = nullptr;

        // === SCRATCH Buffers ===
        TensorBase *Q = nullptr;
        TensorBase *K = nullptr;
        TensorBase *V = nullptr;
        TensorBase *attn_output = nullptr;
        TensorBase *gate = nullptr;
        TensorBase *up = nullptr;
        TensorBase *ffn_output = nullptr;
        TensorBase *workspace_scores = nullptr;
        TensorBase *workspace_context = nullptr;
        TensorBase *workspace_mask = nullptr;

        // === Batched Decode Buffers (for gather from multiple cache slots) ===
        TensorBase *gathered_K = nullptr; ///< [batch_size * max_kv_len, kv_dim]
        TensorBase *gathered_V = nullptr; ///< [batch_size * max_kv_len, kv_dim]

        // === OUTPUT Buffers ===
        TensorBase *attn_proj = nullptr;
        TensorBase *current_hidden = nullptr;
    };

    /**
     * @brief Model-level buffers for full forward pass
     */
    struct Qwen2ModelBuffers
    {
        TensorBase *current_hidden = nullptr; ///< [batch_size * seq_len, d_model]
        TensorBase *logits = nullptr;         ///< [batch_size * seq_len, vocab_size]

        /// Local logits for column-parallel LM head [batch_size * seq_len, vocab_local]
        /// Only used when lm_head_column_parallel is enabled
        TensorBase *logits_local = nullptr;

        /// Per-layer activation buffers
        Qwen2ActivationBuffers layer_buffers;
    };

    // =========================================================================
    // Forward Input/Output
    // =========================================================================

    /**
     * @brief Input for forward pass
     */
    struct Qwen2ForwardInput
    {
        const int *token_ids = nullptr;      ///< Token IDs [batch_size * seq_len]
        const int *position_ids = nullptr;   ///< Position IDs [batch_size * seq_len] (required)
        int batch_size = 1;                  ///< Number of sequences
        int seq_len = 0;                     ///< Sequence length
        int position_offset = 0;             ///< KV cache position offset (legacy, used if position_ids == nullptr)
        int device_idx = 0;                  ///< Target device
        IUnifiedKVCache *kv_cache = nullptr; ///< KV cache for attention (optional)

        /// Sequence lengths for variable-length batching (nullptr = all equal to seq_len)
        /// When set, this enables proper batch-separating attention masks that
        /// prevent cross-sequence attention in batched execution.
        const std::vector<int> *sequence_lengths = nullptr;

        /// For batched input (alternative to token_ids)
        struct Batch
        {
            const int *tokens;
            int len;
            int offset;
        };
        const Batch *batches = nullptr;
        int num_batches = 0;
    };

    /**
     * @brief Output from forward pass
     */
    struct Qwen2ForwardOutput
    {
        TensorBase *logits = nullptr; ///< Output logits [batch_size * seq_len, vocab_size]
        TensorBase *hidden = nullptr; ///< Optional: final hidden states
    };

    // =========================================================================
    // Qwen2Graph Class
    // =========================================================================

    /**
     * @brief Qwen2 compute graph builder
     *
     * Builds ComputeGraph instances for Qwen2 architecture,
     * delegating execution to GraphExecutor.
     *
     * Implements IGraphBuilder interface for polymorphic graph building
     * and testability via MockGraphBuilder.
     */
    class Qwen2Graph : public IGraphBuilder
    {
    public:
        /**
         * @brief Construct Qwen2Graph with full context
         *
         * @param model_ctx Model context with GGUF metadata
         * @param mpi_ctx MPI context (nullptr for single-rank)
         * @param config Qwen2-specific configuration
         */
        Qwen2Graph(std::shared_ptr<ModelContext> model_ctx,
                   std::shared_ptr<MPIContext> mpi_ctx,
                   const Qwen2GraphConfig &config);

        /**
         * @brief Construct Qwen2Graph for layer-level operations only
         *
         * This constructor is used when only layer-level graph building is needed,
         * without model-level operations like embedding or LM head.
         * Backward compatible with Qwen2LayerExecutor(config, mpi_ctx).
         *
         * @param config Qwen2-specific configuration
         * @param mpi_ctx MPI context (nullptr for single-rank)
         */
        Qwen2Graph(const Qwen2GraphConfig &config,
                   std::shared_ptr<MPIContext> mpi_ctx = nullptr);

        ~Qwen2Graph() = default;

        // Non-copyable
        Qwen2Graph(const Qwen2Graph &) = delete;
        Qwen2Graph &operator=(const Qwen2Graph &) = delete;

        // =====================================================================
        // Configuration
        // =====================================================================

        const Qwen2GraphConfig &config() const { return config_; }

        /**
         * @brief Set weight accessors (called by pipeline)
         */
        void setWeights(const Qwen2ModelWeights &weights) { weights_ = weights; }

        /**
         * @brief Set activation buffers (called by pipeline)
         *
         * Use this for manual buffer management (default behavior).
         * Alternative: Use initializeBuffers() for graph-managed allocation.
         */
        void setBuffers(const Qwen2ModelBuffers &buffers) { buffers_ = buffers; }

        /**
         * @brief Set TensorFactory for graph-managed buffer allocation
         * @param factory TensorFactory pointer (not owned)
         */
        void setTensorFactory(TensorFactory *factory) { tensor_factory_ = factory; }

        // =====================================================================
        // Graph-Managed Buffer Allocation (Phase 5)
        // =====================================================================

        /**
         * @brief Initialize activation buffers using GraphBufferManager
         *
         * Allocates all activation buffers with automatic aliasing optimization
         * for SCRATCH buffers. This is an alternative to setBuffers().
         *
         * Requires config.use_graph_buffer_management = true.
         *
         * @param seq_len Maximum sequence length for buffer allocation
         * @return true if allocation successful
         */
        bool initializeBuffers(int seq_len);

        /**
         * @brief Release all graph-managed buffers
         *
         * Call this when buffers are no longer needed to free memory.
         */
        void releaseBuffers();

        /**
         * @brief Check if graph buffer management is enabled
         */
        bool hasGraphManagedBuffers() const { return buffer_manager_ != nullptr; }

        /**
         * @brief Get internal activation buffers (for graph-managed mode)
         *
         * When using graph-managed buffers, the pipeline should use these
         * instead of creating its own buffer mappings.
         *
         * @return Reference to internal activation buffers
         */
        Qwen2ActivationBuffers &getInternalBuffers() { return buffers_.layer_buffers; }
        const Qwen2ActivationBuffers &getInternalBuffers() const { return buffers_.layer_buffers; }

        /**
         * @brief Get model-level buffers (current_hidden, logits)
         *
         * When using graph-managed buffers, these are allocated by the graph.
         *
         * @return Reference to model buffers
         */
        const Qwen2ModelBuffers &getModelBuffers() const { return buffers_; }

        /**
         * @brief Get buffer manager statistics
         *
         * @return BufferAllocationStats or nullptr if not using graph buffer management
         */
        const BufferAllocationStats *bufferStats() const;

        /**
         * @brief Set snapshot callback for debugging
         */
        void setSnapshotCallback(StageSnapshotCallback callback)
        {
            snapshot_callback_ = std::move(callback);
            executor_.setSnapshotCallback(snapshot_callback_);
        }

        // =====================================================================
        // IGraphBuilder Interface Implementation
        // =====================================================================

        /**
         * @brief Build complete forward graph (IGraphBuilder interface)
         *
         * Adapts Qwen2ForwardInput/Output to generic ForwardInput/Output.
         */
        ComputeGraph buildForwardGraph(
            const ForwardInput &input,
            ForwardOutput &output) override;

        /**
         * @brief Build single transformer layer graph (IGraphBuilder interface)
         *
         * Uses layer context to build attention + FFN graph.
         */
        ComputeGraph buildLayerGraph(const LayerContext &ctx) override;

        /**
         * @brief Get number of transformer layers
         */
        int numLayers() const override { return config_.n_layers; }

        /**
         * @brief Get model hidden dimension
         */
        int hiddenDim() const override { return config_.d_model; }

        /**
         * @brief Check if the builder is properly initialized
         */
        bool isInitialized() const override
        {
            return weights_.get_layer_weights != nullptr;
        }

        // =====================================================================
        // Model-Level Graph Building
        // =====================================================================

        /**
         * @brief Build complete forward graph (embedding → layers → LM head)
         */
        ComputeGraph buildFullForwardGraph(
            const Qwen2ForwardInput &input,
            Qwen2ForwardOutput &output);

        /**
         * @brief Build embedding lookup graph
         */
        ComputeGraph buildEmbeddingGraph(
            const Qwen2ForwardInput &input,
            TensorBase *output_hidden);

        /**
         * @brief Build all transformer layers graph
         */
        ComputeGraph buildTransformerLayersGraph(
            TensorBase *input_hidden,
            IUnifiedKVCache *kv_cache,
            const int *position_ids,
            int device_idx);

        /**
         * @brief Build single transformer layer graph
         */
        ComputeGraph buildLayerGraph(
            int layer_idx,
            TensorBase *input_hidden,
            IUnifiedKVCache *kv_cache,
            const int *position_ids,
            int device_idx);

        /**
         * @brief Build LM head projection graph
         * @param hidden_states Final hidden states from transformer layers
         * @param output_logits Output tensor for full logits [seq_len, vocab_size]
         * @param total_tokens Number of tokens (batch_size * seq_len)
         * @param device_idx Target device
         * @param logits_local Optional local logits buffer for column-parallel LM head
         *                     [seq_len, vocab_local]. When lm_head_column_parallel is
         *                     enabled and this is non-null, the LM head writes to
         *                     logits_local, then AllGather collects to output_logits.
         * @return LM head compute graph
         */
        ComputeGraph buildLMHeadGraph(
            TensorBase *hidden_states,
            TensorBase *output_logits,
            int total_tokens,
            int device_idx,
            TensorBase *logits_local = nullptr);

        // =====================================================================
        // Layer-Level Graph Building
        // =====================================================================

        /**
         * @brief Build attention block graph
         * @param batch_size Number of sequences in batch (1 for single-sequence)
         * @param sequence_lengths Actual lengths per sequence (nullptr = all equal to seq_len)
         */
        ComputeGraph buildAttentionGraph(
            const Qwen2LayerWeights &layer,
            Qwen2ActivationBuffers &buffers,
            int layer_idx,
            int seq_len,
            int batch_size,
            IUnifiedKVCache *kv_cache,
            const int *position_ids,
            int device_idx,
            const std::vector<int> *sequence_lengths = nullptr);

        /**
         * @brief Build FFN block graph
         */
        ComputeGraph buildFFNGraph(
            const Qwen2LayerWeights &layer,
            Qwen2ActivationBuffers &buffers,
            int layer_idx,
            int seq_len,
            int batch_size,
            int device_idx);

        // =====================================================================
        // Execution Methods (DEPRECATED - Use GraphOrchestrator)
        // =====================================================================
        //
        // These methods are deprecated. Qwen2Graph should only BUILD graphs.
        // Use GraphOrchestrator for execution, which provides:
        // - Graph caching with cache hit statistics
        // - Device context management
        // - Batched execution support
        // =====================================================================

        /**
         * @brief Execute full forward pass
         *
         * @deprecated Use GraphOrchestrator::executeForward() instead.
         *             Qwen2Graph should only build graphs, not execute them.
         *
         * Builds and executes the complete forward graph.
         */
        [[deprecated("Use GraphOrchestrator::executeForward() instead")]]
        bool executeForward(
            const Qwen2ForwardInput &input,
            Qwen2ForwardOutput &output);

        /**
         * @brief Execute a pre-built graph
         *
         * @deprecated Use GraphOrchestrator::execute() instead.
         */
        [[deprecated("Use GraphOrchestrator::execute() instead")]]
        bool execute(ComputeGraph &graph, IDeviceContext *ctx);

        /**
         * @brief Execute attention block
         *
         * @deprecated Use GraphOrchestrator::executeAttention() instead.
         */
        [[deprecated("Use GraphOrchestrator::executeAttention() instead")]]
        bool executeAttention(
            const Qwen2LayerWeights &layer,
            Qwen2ActivationBuffers &buffers,
            int layer_idx,
            int seq_len,
            IUnifiedKVCache *kv_cache,
            const int *position_ids,
            int device_idx);

        /**
         * @brief Execute FFN block
         *
         * @deprecated Use GraphOrchestrator::executeFFN() instead.
         */
        [[deprecated("Use GraphOrchestrator::executeFFN() instead")]]
        bool executeFFN(
            const Qwen2LayerWeights &layer,
            Qwen2ActivationBuffers &buffers,
            int layer_idx,
            int seq_len,
            int device_idx);

        /**
         * @brief Execute full transformer layer (attention + FFN)
         *
         * @deprecated Use GraphOrchestrator::executeLayer() instead.
         */
        [[deprecated("Use GraphOrchestrator::executeLayer() instead")]]
        bool executeLayer(
            const Qwen2LayerWeights &layer,
            Qwen2ActivationBuffers &buffers,
            int layer_idx,
            int seq_len,
            IUnifiedKVCache *kv_cache,
            const int *position_ids,
            int device_idx);

        // =====================================================================
        // Statistics and State
        // =====================================================================

        const GraphExecutorStats &stats() const { return executor_.stats(); }
        void resetStats() { executor_.resetStats(); }
        void clearCache();

        // =====================================================================
        // Graph Caching Accessors (for testing)
        // =====================================================================

        /**
         * @brief Check if graph caching is enabled
         * @return true if caching is enabled (after initializeBuffers() with graph buffer management)
         */
        bool isGraphCachingEnabled() const { return graph_caching_enabled_; }

        /**
         * @brief Get the size of the layer graph cache
         * @return Number of layers in the cache (0 if caching disabled)
         */
        size_t getCacheSize() const { return layer_graph_cache_.size(); }

        /**
         * @brief Check if a valid cached graph exists for a layer
         * @param layer_idx Layer index (0-based)
         * @param is_attention true for attention graph, false for FFN graph
         * @return true if a valid cached graph exists
         */
        bool hasValidCachedGraph(int layer_idx, bool is_attention) const;

    private:
        // Configuration
        Qwen2GraphConfig config_;
        std::shared_ptr<ModelContext> model_ctx_;
        std::shared_ptr<MPIContext> mpi_ctx_;

        // TensorFactory for buffer allocation (not owned)
        TensorFactory *tensor_factory_ = nullptr;

        // Weights and buffers (not owned)
        Qwen2ModelWeights weights_;
        Qwen2ModelBuffers buffers_;

        // Graph executor for actual execution
        GraphExecutor executor_;

        // Device contexts (created lazily)
        std::unordered_map<int, std::unique_ptr<IDeviceContext>> device_contexts_;

        // Snapshot callback
        StageSnapshotCallback snapshot_callback_;

        // Position IDs buffer - DEPRECATED: Position IDs should be provided externally via Qwen2ForwardInput
        // Kept temporarily for backward compatibility with execute*() methods
        std::vector<int> position_ids_buffer_;

        // =====================================================================
        // Graph Caching (Phase 10: Execution Optimization)
        // =====================================================================
        //
        // When graph_caching_enabled_ is true, we cache pre-built graphs per layer
        // and reuse them across executions. This avoids the overhead of:
        // - Creating ComputeGraph objects
        // - Allocating stage unique_ptrs
        // - Building dependency maps
        //
        // Cached graphs work because:
        // 1. Buffer pointers are stable (graph-managed buffers)
        // 2. Weight pointers are stable (owned by ModelLoader)
        // 3. Only dynamic params (seq_len, pos_offset) change
        //
        // For each layer, we cache:
        // - Attention graph (for seq_len=1 decode mode)
        // - FFN graph (always reusable)
        //
        // Dynamic params are updated via stage setters before execution.
        // =====================================================================

        /// Enable graph caching (auto-enabled when using graph-managed buffers)
        bool graph_caching_enabled_ = false;

        /// Cached attention graphs per layer [layer_idx]
        /// Key is: (layer_idx, seq_len) for different graph variants
        struct CachedLayerGraphs
        {
            std::unique_ptr<ComputeGraph> attention_decode; ///< seq_len=1
            std::unique_ptr<ComputeGraph> ffn_decode;       ///< seq_len=1
            int cached_seq_len = 0;                         ///< seq_len used for cached graphs
            bool valid = false;                             ///< Whether cache is valid
        };
        std::vector<CachedLayerGraphs> layer_graph_cache_;

        /// Last position offset used (for RoPE update detection)
        int last_pos_offset_ = -1;

        /**
         * @brief Check if we can use cached graph for this execution
         */
        bool canUseCachedGraph(int layer_idx, int seq_len) const;

        /**
         * @brief Get or build attention graph (with caching)
         */
        ComputeGraph &getOrBuildAttentionGraph(
            const Qwen2LayerWeights &layer,
            Qwen2ActivationBuffers &buffers,
            int layer_idx,
            int seq_len,
            IUnifiedKVCache *kv_cache,
            const int *position_ids,
            int device_idx);

        /**
         * @brief Get or build FFN graph (with caching)
         */
        ComputeGraph &getOrBuildFFNGraph(
            const Qwen2LayerWeights &layer,
            Qwen2ActivationBuffers &buffers,
            int layer_idx,
            int seq_len,
            int device_idx);

        /**
         * @brief Update dynamic parameters in cached graph
         */
        void updateCachedGraphParams(ComputeGraph &graph, int pos_offset, int seq_len);

        // =====================================================================
        // Graph Buffer Management (Phase 5)
        // =====================================================================

        /// Buffer manager for graph-managed allocation (nullptr if using setBuffers())
        std::unique_ptr<GraphBufferManager> buffer_manager_;

        /// Owned tensors when using graph-managed allocation
        std::vector<std::unique_ptr<TensorBase>> owned_buffers_;

        /// Buffer spec builder for generating buffer specifications
        std::unique_ptr<Qwen2BufferSpecBuilder> buffer_spec_builder_;

        // =====================================================================
        // Helper Methods
        // =====================================================================

        IDeviceContext *getDeviceContext(int device_idx);

    public:
        /**
         * @brief Build position IDs array for RoPE
         * @param seq_len Sequence length
         * @param batch_size Number of sequences
         * @param offset Position offset (for KV cache continuation)
         * @return Vector of position IDs [batch_size * seq_len]
         *
         * Static utility function that can be used by callers to build position IDs.
         * For batch_size=1: [offset, offset+1, ..., offset+seq_len-1]
         * For batch_size>1: Repeated pattern per batch
         */
        static std::vector<int> buildPositionIds(int seq_len, int batch_size, int offset);

        void addFinalNormToGraph(
            ComputeGraph &graph,
            TensorBase *hidden,
            const std::string &prev_node,
            int seq_len,
            int device_idx);

        /**
         * @brief Populate buffers_ from graph-managed allocations
         */
        void bindGraphManagedBuffers(int seq_len);
    };

    // =========================================================================
    // Backward Compatibility Aliases
    // =========================================================================

    // NOTE: These aliases are DEPRECATED. Use Qwen2Graph directly.
    using Qwen2LayerExecutor = Qwen2Graph;
    using Qwen2ModelExecutor = Qwen2Graph;
    using Qwen2ExecutorConfig = Qwen2GraphConfig;
    using Qwen2ModelExecutorConfig = Qwen2GraphConfig;

} // namespace llaminar2
