/**
 * @file Qwen2Graph.h
 * @brief Qwen2 compute graph builder for declarative forward pass execution
 * @author David Sanftenberg
 * @date December 2025
 *
 * Qwen2Graph is the graph builder for Qwen2 architecture models.
 *
 * Key Design:
 * - This class BUILDS compute graphs (it's a graph builder, not executor)
 * - The actual execution is delegated to DeviceGraphExecutor
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
 */

#pragma once

#include "../../execution/local_execution/graph/DeviceGraphExecutor.h"
#include "../../execution/local_execution/graph/DeviceGraphBufferManager.h"
#include "../../execution/compute_stages/ComputeStages.h"
#include "../../execution/local_execution/device/DeviceContext.h"
#include "../../execution/config/ExecutionPolicy.h"
#include "../../execution/local_execution/graph/IGraphBuilder.h"
#include "../../execution/config/RuntimeConfig.h"
#include "../../execution/local_execution/graph/GraphResolver.h"
#include "../../backends/DeviceId.h"
#include "../../tensors/Tensors.h"
#include "../../tensors/TensorFactory.h"
#include "../../kernels/cpu/CPUKVCache.h"
#include "../../loaders/ModelContext.h"
#include "../../utils/MPIContext.h"
#include "../../config/TensorParallelConfig.h"
#include "../../config/TPDomain.h"
#include "Qwen2Schema.h"
#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <unordered_map>

namespace llaminar2
{

    // Forward declarations
    class ILocalTPContext;
    class ILocalPPContext;
    struct PipelineConfig;

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
        int n_layers = 0;   ///< Number of transformer layers (full model count for validation)
        int d_model = 0;    ///< Model hidden dimension
        int n_heads = 0;    ///< Number of attention heads
        int n_kv_heads = 0; ///< Number of KV heads (GQA)
        int head_dim = 0;   ///< Dimension per head
        int d_ff = 0;       ///< FFN intermediate dimension
        int vocab_size = 0; ///< Vocabulary size

        /// Pipeline Parallelism layer offset for KV cache indexing.
        /// When building graphs for PP stage [first_layer, last_layer), this offset
        /// is subtracted from the global layer index to get the local KV cache index.
        /// E.g., for PP stage 1 with layers [12, 24), pp_layer_offset=12, so layer 12
        /// maps to KV cache layer 0.
        int pp_layer_offset = 0;

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

        // =================================================================
        // Q16 KV Cache VNNI Safety (Phase 5.4)
        // =================================================================
        /// Fixed scale for Q16 KV cache quantization (FP32 range: ±kv_cache_scale).
        /// All K/V values are quantized with this scale, producing INT16 values in
        /// range [-32767 * kv_cache_scale, +32767 * kv_cache_scale]. The value must
        /// match the expected activation range to avoid clipping. Default 8.0f works
        /// for typical Qwen2 models. See VNNISafetyConstants.h for VNNI overflow limits.
        float kv_cache_scale = 256.0f; ///< Fixed Q16 scale. Must cover Q projection max (~130)

        /// Explicit KV cache precision mode (AUTO preserves legacy behavior).
        KVCachePrecision kv_cache_precision = KVCachePrecision::AUTO;

        // Execution settings
        DeviceId default_device = DeviceId::cpu(); ///< Default device for execution
        bool enable_profiling = false;
        bool enable_validation = false;

        // NOTE: Decomposed attention (Phase 9: KVCacheAppendStage + AttentionComputeStage)
        // is now the ONLY supported path. The legacy AttentionWithKVCacheStage path has been
        // removed as part of Phase 7 cleanup. See DISTRIBUTED_ARCHITECTURE_PROPOSAL.md.

        /// Use graph-managed buffer allocation with aliasing optimization.
        /// When true, Qwen2Graph will use DeviceGraphBufferManager to allocate activation
        /// buffers with automatic aliasing of non-overlapping SCRATCH buffers.
        /// NOTE: As of December 2025, this defaults to true (Graph is primary path).
        bool use_graph_buffer_management = true;

        /// Maximum sequence length for buffer allocation (when use_graph_buffer_management=true)
        int max_seq_len = 2048;

        /// Execution policy controlling which operations run
        ExecutionPolicy execution_policy = ExecutionPolicy::allEnabled();

        /// Base DeviceGraphExecutor configuration
        GraphExecutorConfig executor_config = GraphExecutorConfig{};

        /// Fused attention backend selection
        /// - JIT: AVX-512 VNNI optimized (default)
        /// - REFERENCE: Pure C++ implementation for testing
        /// - TILED: Cache-blocked implementation
        /// - Q16_INTEGER: Pure Q16 integer-domain kernel (experimental, requires HybridQ16)
        FusedAttentionBackend fused_attention_backend = FusedAttentionBackend::JIT;

        // =================================================================
        // Heterogeneous Tensor Parallelism (Phase 1c: Proportional TP)
        // =================================================================
        /// Optional TensorParallelConfig for proportional head/FFN/vocab assignment.
        /// When set, overrides equal-split head/KV/FFN computation.
        /// Use for heterogeneous GPU setups (e.g., NVIDIA 73% + AMD 27%).
        std::shared_ptr<TensorParallelConfig> tp_config = nullptr;

        /// Local rank index (used with tp_config to look up assignment)
        int local_rank = 0;

        // =================================================================
        // Multi-Domain Tensor Parallelism (Phase 4.3: Heterogeneous TP)
        // =================================================================
        /// Optional MultiDomainTPConfig for domain-based collective routing.
        /// When set, collective operations (AllreduceStage, AllGatherStage) will
        /// use the domain communicator instead of the global MPI communicator.
        /// This enables heterogeneous TP with separate GPU and CPU domains.
        MultiDomainTPConfig *multi_domain_tp_config = nullptr;

        // =================================================================
        // LOCAL Tensor Parallelism (Intra-Rank Multi-Device)
        // =================================================================
        /// Optional ILocalTPContext for LOCAL tensor parallelism.
        /// When set, collective operations use TPAllreduceStage with the local
        /// TP context. LOCAL TP runs on a single MPI rank with multiple devices,
        /// using high-bandwidth backends like NCCL, RCCL, or PCIeBAR for collectives.
        ///
        /// Distinction from GLOBAL TP:
        /// - GLOBAL TP: Multiple MPI ranks (world_size > 1), MPI collectives
        /// - LOCAL TP: Single MPI rank (world_size = 1), ILocalTPContext collectives
        ///
        /// Note: Either mpi_ctx (for GLOBAL TP) OR local_tp_ctx (for LOCAL TP) should
        /// be active, not both. If both are set, LOCAL TP takes precedence for collectives.
        ILocalTPContext *local_tp_ctx = nullptr;

        /// Device index within the LOCAL TP context (0 to degree-1).
        /// Each device runs a separate graph instance with sharded weights.
        int local_tp_device_idx = 0;

        // =================================================================
        // Unified Pipeline Parallel Configuration (Phase 2)
        // =================================================================
        /// Pipeline configuration for PP + TP composition.
        /// When set, graph building uses multi-stage PP-aware logic.
        /// See docs/v2/UNIFIED_PP_GRAPH_ARCHITECTURE_PLAN.md
        std::shared_ptr<PipelineConfig> pipeline_config = nullptr;

        /// PP contexts for inter-stage activation transfers.
        /// Key: {from_stage_id, to_stage_id}
        /// Created by the orchestrator and passed to Qwen2Graph.
        std::map<std::pair<int, int>, ILocalPPContext *> pp_contexts;

        /// TP contexts for each domain (one per domain name).
        /// Each domain may have internal tensor parallelism.
        /// Created by the orchestrator and passed to Qwen2Graph.
        std::map<std::string, ILocalTPContext *> domain_tp_contexts;

        /// Helper: Check if unified PP mode is enabled
        /// Implementation in Qwen2Graph.cpp (requires full PipelineConfig definition)
        bool hasUnifiedPP() const;

        /// Helper: Get the device for a specific layer in unified PP mode
        /// Implementation in Qwen2Graph.cpp (requires full PipelineConfig definition)
        DeviceId getDeviceForLayer(int layer_idx) const;

        /**
         * @brief Helper to get DeviceShardingAssignment for current rank
         * @return Pointer to assignment, or nullptr if tp_config is not set
         */
        const DeviceShardingAssignment *getAssignment() const
        {
            if (!tp_config)
                return nullptr;
            return &tp_config->forRank(local_rank);
        }

        // =================================================================
        // Per-Layer TP Allreduce Precision
        // =================================================================

        /// Per-layer allreduce precision map: layer_idx → precision string.
        /// Populated from the GraphSchema precision policy (first N layers FP32,
        /// rest use schema default). Layers not in the map fall back to the
        /// global DebugEnv::allreduce_precision ("fp32" by default).
        std::unordered_map<int, std::string> tp_allreduce_precision;

        /**
         * @brief Get allreduce precision for a specific layer
         *
         * Resolution order:
         * 1. Per-layer override from tp_allreduce_precision map
         * 2. Global fallback from debugEnv().allreduce_precision
         *
         * @param layer_idx Transformer layer index (0-based)
         * @return Precision string ("fp32", "fp16", "bf16")
         */
        std::string getAllreducePrecisionForLayer(int layer_idx) const
        {
            auto it = tp_allreduce_precision.find(layer_idx);
            if (it != tp_allreduce_precision.end())
                return it->second;
            return ""; // Empty = defer to global DebugEnv default
        }

        /**
         * @brief Populate the precision map from a GraphSchema precision policy
         *
         * Applies the schema's fp32_layer_count + default_precision to build
         * the per-layer map for this model's n_layers.
         *
         * @param schema_default Default precision for layers beyond fp32 count
         * @param fp32_count Number of initial layers forced to FP32
         */
        void populateAllreducePrecision(const std::string &schema_default, int fp32_count)
        {
            tp_allreduce_precision.clear();
            for (int i = 0; i < n_layers; ++i)
            {
                tp_allreduce_precision[i] = (i < fp32_count) ? "fp32" : schema_default;
            }
        }
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

        // QK norm weights (Qwen3: per-head RMSNorm before RoPE)
        TensorBase *q_norm = nullptr; ///< Q norm gamma [head_dim]
        TensorBase *k_norm = nullptr; ///< K norm gamma [head_dim]

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

        // === Hybrid Mode Buffers ===
        /// FP32 Q after RoPE (Hybrid mode only - avoids requantization)
        /// When set, RoPE outputs to this buffer instead of modifying Q in-place
        TensorBase *Q_rope = nullptr;
        /// FP32 K after RoPE (Hybrid mode only - avoids requantization)
        /// When set, RoPE outputs to this buffer instead of modifying K in-place
        TensorBase *K_rope = nullptr;
        /// FP32 V dequantized (Hybrid mode only - for KV cache when V is Q8_1)
        /// V doesn't go through RoPE but needs to match KV cache precision
        TensorBase *V_dequant = nullptr;

        // === Batched Decode Buffers (for gather from multiple cache slots) ===
        TensorBase *gathered_K = nullptr; ///< [batch_size * max_kv_len, kv_dim]
        TensorBase *gathered_V = nullptr; ///< [batch_size * max_kv_len, kv_dim]

        // === HybridQ16 K Precision Fix: Per-head K scales ===
        /// Per-head dynamic scales from RoPE Q16→Q16 path
        /// Shape: [seq_len * n_kv_heads] for prefill, stored in KV cache for decode
        /// Only used when GEMM outputs K as Q16_1 (K precision fix mode)
        float *K_head_scales = nullptr;
        /// Capacity of K_head_scales buffer (in floats)
        size_t K_head_scales_capacity = 0;

        // === OUTPUT Buffers ===
        TensorBase *attn_proj = nullptr;
        TensorBase *current_hidden = nullptr;

        // === SNAPSHOT Buffers (for debugging, enabled with ENABLE_PIPELINE_SNAPSHOTS) ===
        /// Optional buffer to capture attention context before Wo projection
        /// Shape: [batch_size * seq_len, n_heads * head_dim]
        TensorBase *context_snapshot = nullptr;

        /// Optional buffer to capture attention output (Wo projection result, before residual add)
        /// Shape: [batch_size * seq_len, d_model] - corresponds to ATTENTION_OUTPUT
        TensorBase *attention_output_snapshot = nullptr;

        /// Optional buffer to capture attention residual (after residual add)
        /// Shape: [batch_size * seq_len, d_model] - corresponds to ATTENTION_RESIDUAL
        TensorBase *attention_residual_snapshot = nullptr;
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
        const int *token_ids = nullptr;    ///< Token IDs [batch_size * seq_len]
        const int *position_ids = nullptr; ///< Position IDs [batch_size * seq_len] (required)
        int batch_size = 1;                ///< Number of sequences
        int seq_len = 0;                   ///< Sequence length
        int position_offset = 0;           ///< KV cache position offset (legacy, used if position_ids == nullptr)
        DeviceId device = DeviceId::cpu(); ///< Target device
        IKVCache *kv_cache = nullptr;      ///< KV cache for attention (optional)

        /// Per-device KV caches for Pipeline Parallelism (PP).
        /// When set (non-empty), each PP stage uses the KV cache for its device.
        /// Key: DeviceId (e.g., cuda:0, cuda:1), Value: IKVCache* for that device.
        /// Takes precedence over kv_cache when device is found in this map.
        const std::unordered_map<DeviceId, IKVCache *> *pp_kv_caches = nullptr;

        /// External hidden state input (for PP middle stages that don't have embedding)
        /// When set, embedding is skipped and this tensor is used as initial hidden state.
        /// For HybridQ16 mode: expected to be Q16_1 tensor
        /// For FP32 mode: expected to be FP32 tensor
        TensorBase *external_hidden_state = nullptr;

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

        /// Get the KV cache for a specific device (PP) or the default (non-PP)
        IKVCache *getKVCacheForDevice(const DeviceId &device) const
        {
            // For PP mode: look up per-device KV cache
            if (pp_kv_caches && !pp_kv_caches->empty())
            {
                auto it = pp_kv_caches->find(device);
                if (it != pp_kv_caches->end())
                {
                    return it->second;
                }
                // Fallback to default kv_cache if device not found
            }
            return kv_cache;
        }
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
     * delegating execution to DeviceGraphExecutor.
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

        // =====================================================================
        // Unified PP Configuration Mutators (Phase 6)
        // =====================================================================

        /**
         * @brief Set pipeline configuration for unified PP graph
         *
         * Enables buildUnifiedPipelineGraph() to create multi-stage graphs.
         *
         * @param pipeline_config Shared ownership of pipeline configuration
         */
        void setPipelineConfig(std::shared_ptr<PipelineConfig> pipeline_config)
        {
            config_.pipeline_config = std::move(pipeline_config);
        }

        /**
         * @brief Register a PP context for inter-stage activation transfers
         *
         * Called by DeviceGraphOrchestrator after initializing LocalPPContexts.
         *
         * @param from_stage Source PP stage ID
         * @param to_stage Destination PP stage ID
         * @param pp_ctx Pointer to ILocalPPContext (must remain valid during graph execution)
         */
        void setPPContext(int from_stage, int to_stage, ILocalPPContext *pp_ctx)
        {
            config_.pp_contexts[{from_stage, to_stage}] = pp_ctx;
        }

        /**
         * @brief Register a TP context for a named domain
         *
         * Called by DeviceGraphOrchestrator after initializing LocalTPContexts.
         *
         * @param domain_name Name of the TP domain
         * @param tp_ctx Pointer to ILocalTPContext (must remain valid during graph execution)
         */
        void setTPContext(const std::string &domain_name, ILocalTPContext *tp_ctx)
        {
            config_.domain_tp_contexts[domain_name] = tp_ctx;
        }

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

        /**
         * @brief Get the declarative schema for this architecture
         *
         * Returns the Qwen2 GraphSchema which defines all buffers, stages,
         * and their relationships declaratively.
         *
         * @return GraphSchema for Qwen2 architecture
         */
        GraphSchema getSchema() const;

        /**
         * @brief Get resolver config for buffer allocation
         *
         * Creates a GraphResolverConfig populated with this graph's
         * configuration, including tensor-parallel local dimensions.
         * Used by BufferAllocator to resolve buffer shapes.
         *
         * @param seq_len Sequence length for buffer sizing
         * @return GraphResolverConfig with model dimensions
         */
        GraphResolverConfig getResolverConfig(int seq_len) const;

        // =====================================================================
        // Buffer Management
        // =====================================================================
        // NOTE: Buffer lifecycle management (initializeBuffers, releaseBuffers, etc.)
        // has been moved to DeviceGraphOrchestrator as part of the declarative refactor.
        // Use DeviceGraphOrchestrator::initializeBuffers() for graph-managed allocation.
        // =====================================================================

        /**
         * @brief Set snapshot callback for debugging
         *
         * Note: The callback is stored for use by DeviceGraphOrchestrator when it executes
         * graphs built by this Qwen2Graph.
         */
        void setSnapshotCallback(StageSnapshotCallback callback)
        {
            snapshot_callback_ = std::move(callback);
            // Note: DeviceGraphOrchestrator will call its own executor_.setSnapshotCallback()
        }

        /**
         * @brief Get the current snapshot callback
         */
        const StageSnapshotCallback &getSnapshotCallback() const { return snapshot_callback_; }

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
         *
         * LEGACY: This method uses imperative graph building.
         * For new code, prefer buildForwardGraphFromSchema().
         */
        ComputeGraph buildFullForwardGraph(
            const Qwen2ForwardInput &input,
            Qwen2ForwardOutput &output);

        /**
         * @brief Build partial forward graph for Pipeline Parallelism
         *
         * Builds a compute graph for a subset of the model, enabling PP execution
         * where each stage runs only its assigned components:
         *
         * Stage 0: [embedding] → layers[0, mid) → hidden output
         * Stage 1: hidden input → layers[mid, n) → [LM head] → logits
         *
         * @param input Forward pass input (tokens for embedding stage, hidden for others)
         * @param output Forward pass output (hidden for non-final stages, logits for final)
         * @param first_layer First layer to include (inclusive)
         * @param last_layer Last layer to include (exclusive)
         * @param has_embedding Include embedding lookup stage
         * @param has_lm_head Include final_norm and LM head projection
         * @return ComputeGraph for partial forward pass
         */
        ComputeGraph buildPartialForwardGraph(
            const Qwen2ForwardInput &input,
            Qwen2ForwardOutput &output,
            int first_layer,
            int last_layer,
            bool has_embedding,
            bool has_lm_head);

        /**
         * @brief Build unified forward graph with PP + TP composition
         *
         * Creates a single ComputeGraph spanning all PP stages with:
         * - Per-layer device assignment based on PP stage → TP domain mapping
         * - TPAllreduceStage nodes within each TP domain
         * - LocalPPTransferStage nodes at PP stage boundaries
         *
         * Requires: config_.pipeline_config must be set and valid
         *
         * @param input Forward pass input
         * @param output Forward pass output (populated)
         * @return ComputeGraph spanning all PP stages
         */
        ComputeGraph buildUnifiedPipelineGraph(
            const Qwen2ForwardInput &input,
            Qwen2ForwardOutput &output);

        /**
         * @brief Build forward graph using declarative schema
         *
         * This method uses the new three-layer architecture:
         * 1. Qwen2Schema - Declarative graph structure
         * 2. GraphResolver - Evaluates runtime conditionals
         * 3. GraphBuilder - Constructs the ComputeGraph
         *
         * Benefits:
         * - All TP/MPI logic shared with other models
         * - All debugEnv toggling handled uniformly
         * - Graph structure defined declaratively (could be YAML)
         *
         * @param input Forward pass input
         * @param output Forward pass output (modified)
         * @return Constructed ComputeGraph
         */
        ComputeGraph buildForwardGraphFromSchema(
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
            IKVCache *kv_cache,
            const int *position_ids,
            DeviceId device);

        /**
         * @brief Build single transformer layer graph
         */
        ComputeGraph buildLayerGraph(
            int layer_idx,
            TensorBase *input_hidden,
            IKVCache *kv_cache,
            const int *position_ids,
            DeviceId device);

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
            DeviceId device,
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
            IKVCache *kv_cache,
            const int *position_ids,
            DeviceId device,
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
            DeviceId device);

    private:
        // =====================================================================
        // Configuration
        // =====================================================================
        Qwen2GraphConfig config_;
        std::shared_ptr<ModelContext> model_ctx_;
        std::shared_ptr<MPIContext> mpi_ctx_;

        // TensorFactory for buffer allocation (not owned)
        TensorFactory *tensor_factory_ = nullptr;

        // Weights and buffers (not owned)
        Qwen2ModelWeights weights_;
        Qwen2ModelBuffers buffers_;

        // Snapshot callback
        StageSnapshotCallback snapshot_callback_;

        // =====================================================================
        // Helper: TP Allreduce Stage Creation
        // =====================================================================
        /**
         * @brief Check if TP (LOCAL or GLOBAL) requires allreduce collective
         * @return True if either LOCAL TP (degree > 1) or GLOBAL TP (world_size > 1) is active
         */
        bool needsTPAllreduce() const;

        /**
         * @brief Create an allreduce stage appropriate for the active TP mode
         *
         * Creates TPAllreduceStage for LOCAL TP (single rank, multiple devices)
         * or AllreduceStage for GLOBAL TP (multiple MPI ranks).
         *
         * @param buffer Tensor to reduce
         * @param count Number of elements
         * @param device Target device
         * @param layer_idx Layer index for domain routing
         * @param is_attention True for attention, false for FFN
         * @param stage_name Name for the stage (used for BAR-backed tensor lookup)
         * @return Unique pointer to the allreduce stage
         */
        std::unique_ptr<IComputeStage> createTPAllreduceStage(
            TensorBase *buffer,
            size_t count,
            DeviceId device,
            int layer_idx,
            bool is_attention,
            const std::string &stage_name = "") const;

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
            TensorBase *normalized_out,
            const std::string &prev_node,
            int seq_len,
            DeviceId device);

        /**
         * @brief Get TPDomain for collective operations in a specific layer
         *
         * Queries the MultiDomainTPConfig (if set) to determine which domain
         * should handle collective operations for the given layer.
         *
         * @param layer_idx Layer index (0 to n_layers-1)
         * @param is_attention True for attention Wo allreduce, false for FFN down allreduce
         * @return Pointer to TPDomain, or nullptr if no domain config (legacy MPI path)
         */
        const TPDomain *getDomainForLayer(int layer_idx, bool is_attention) const;
    };

} // namespace llaminar2
