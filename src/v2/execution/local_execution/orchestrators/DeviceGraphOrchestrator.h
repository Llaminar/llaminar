/**
 * @file DeviceGraphOrchestrator.h
 * @brief Generic orchestrator for compute graph execution
 * @author David Sanftenberg
 * @date December 2025
 *
 * This file implements the execution layer for transformer models, separating
 * graph execution concerns from graph definition (IGraphBuilder implementations).
 *
 * Design Philosophy:
 * - Graph Builders (Qwen2Graph, etc.): Declarative, stateless, build ComputeGraph DAGs
 * - DeviceGraphOrchestrator: Imperative executor (manages state, caching, device contexts)
 *
 * The orchestrator owns:
 * - DeviceGraphExecutor (for DAG execution)
 * - Device context cache (lazy initialization)
 * - Graph cache (decode optimization)
 * - Execution state (position offset tracking)
 *
 * Usage:
 * @code
 * auto graph_builder = std::make_shared<Qwen2Graph>(config, mpi_ctx);
 * DeviceGraphOrchestrator orchestrator(graph_builder, mpi_ctx);
 *
 * // Execute a layer
 * orchestrator.executeLayer(weights, buffers, layer_idx, seq_len, kv_cache, pos_ids, device_idx);
 * @endcode
 */

#pragma once

#include "../../../models/qwen/Qwen2Graph.h"
#include "../../../backends/DeviceId.h"
#include "../../../backends/IGPUGraphCapture.h"
#include "IInferenceRunner.h"
#include "../graph/DeviceGraphExecutor.h"
#include "../device/DeviceContext.h"
#include "../device/WorkspaceAllocator.h"
#include "../../mpi_orchestration/PlacementStrategy.h" // For InferencePhase
#include "../../compute_stages/ComputeStages.h"        // For StageDumpInfo
#include "../../factory/InferenceRunnerFactory.h"      // For FactoryPPStageConfig
#include "../../../tensors/FP16Utils.h"                // For fp16_to_fp32, bf16_to_fp32
#include "../../../tensors/BlockStructures.h"          // For Q8_1Block, Q16_1Block
#include "../../../loaders/IWeightStreamer.h"          // For weight streaming (Option B)
#include "../../../interfaces/IModelContext.h"         // For interface-based construction
#include "../../../memory/BufferArena.h"               // Phase 2: unified buffer management
#include "../../../interfaces/IMPITopology.h"          // For interface-based construction
#include "../../../interfaces/ICollectiveContext.h"    // For interface-based construction
#include "../../../config/TPDomain.h"                  // For MultiDomainTPConfig (Phase 6.3)
#include "../../../config/PipelineConfig.h"            // For unified PP+TP configuration (Phase 6)
#include "../../../collective/ILocalPPContext.h"       // For unique_ptr<ILocalPPContext> in maps
#include "../../../collective/ILocalTPContext.h"       // For unique_ptr<ILocalTPContext> in maps
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <vector>
#include <string>
#include <cstring>

namespace llaminar2
{

    // Forward declarations
    class MPIContext;
    class IKVCache;
    class WeightManager;
    class WeightPlacementMap;
    class TensorParallelConfig;
    class TurboQuantContext;

    /**
     * @brief Configuration for graph caching behavior
     */
    struct GraphCacheConfig
    {
        bool enabled = true;         ///< Enable graph caching (Phase 10)
        int decode_seq_len = 1;      ///< Sequence length that triggers decode caching
        bool cache_attention = true; ///< Cache attention graphs
        bool cache_ffn = true;       ///< Cache FFN graphs
    };

    /**
     * @brief Cached graphs for a single transformer layer
     *
     * Stores pre-built attention and FFN graphs for decode mode (seq_len=1).
     * These graphs have stable buffer pointers and only need parameter updates
     * (position offset) between executions.
     */
    struct LayerGraphCache
    {
        std::unique_ptr<ComputeGraph> attention_decode; ///< Cached attention graph for decode
        std::unique_ptr<ComputeGraph> ffn_decode;       ///< Cached FFN graph for decode
        int cached_seq_len = 0;                         ///< Sequence length when cached
        bool valid = false;                             ///< Whether cache entries are valid

        void invalidate()
        {
            attention_decode.reset();
            ffn_decode.reset();
            cached_seq_len = 0;
            valid = false;
        }
    };

    /**
     * @brief Configuration for inference state initialization
     *
     * Controls how buffers are allocated during initializeInferenceState().
     */
    struct InferenceStateInitConfig
    {
        /**
         * @brief Use mapped memory for GPU tensor allocation
         *
         * When true and the target device is a GPU (CUDA or ROCm), FP32 activation
         * buffers will be allocated using zero-copy mapped memory:
         * - CUDA: cudaHostAllocMapped | cudaHostAllocWriteCombined
         * - ROCm: hipHostMallocMapped | hipHostMallocWriteCombined
         *
         * This enables the host to read GPU tensor data without memcpy, which is
         * essential for:
         * - Snapshot capture mode (parity testing, debugging)
         * - Any scenario where host needs frequent access to GPU tensors
         *
         * Tradeoffs:
         * - Slightly slower GPU access (PCIe vs VRAM bandwidth)
         * - But eliminates ~5-10 second sync delays for snapshot callbacks
         *
         * Default: false (use device memory for best GPU performance)
         */
        bool use_mapped_memory = false;

        /**
         * @brief Use BAR-backed memory for hidden state tensor
         *
         * When true and the device is ROCm, the hidden state output tensor will
         * be allocated in PCIe BAR memory, enabling zero-copy reads from CUDA
         * devices. This is required for cross-vendor PP transfers (ROCm→CUDA).
         *
         * Conditions for BAR-backed allocation:
         * 1. Device is ROCm (source of cross-vendor transfer)
         * 2. Next PP stage is CUDA (destination)
         * 3. DirectP2PEngine is available with active BAR mapping
         *
         * Default: false (use standard VRAM allocation)
         */
        bool use_bar_backed_hidden = false;
    };

    /**
     * @brief Inference state owned by DeviceGraphOrchestrator (Phase 5)
     *
     * This struct encapsulates all mutable inference state, allowing the
     * orchestrator to manage state internally rather than requiring the
     * pipeline to pass buffers for each forward call.
     *
     * State includes:
     * - Hidden state buffer (current layer activations)
     * - Logits buffer (output vocabulary scores)
     * - KV cache (attention key/value history)
     * - Position tracking (per-sequence position offsets)
     * - Sequence lengths (for variable-length batches)
     * - Activation buffers (intermediate tensors for attention/FFN)
     */
    struct InferenceState
    {
        // === Core Buffers ===
        std::shared_ptr<TensorBase> hidden; ///< [batch_size * seq_len, d_model]
        std::shared_ptr<TensorBase> logits; ///< [batch_size * seq_len, vocab_size]

        /// Local logits for column-parallel LM head [batch_size * seq_len, vocab_local]
        /// Only allocated when lm_head_column_parallel is enabled
        std::shared_ptr<TensorBase> logits_local;

        // === KV Cache ===
        std::unique_ptr<IKVCache> kv_cache; ///< Attention KV history (single-device mode)

        /// Per-device KV caches for Pipeline Parallelism
        /// When PP is enabled, each PP stage device has its own KV cache containing
        /// only the layers processed by that stage. Key is DeviceId, value is the cache.
        /// Only populated when pipeline_config->hasPP() is true.
        std::unordered_map<DeviceId, std::unique_ptr<IKVCache>> pp_kv_caches;

        // === Position Tracking ===
        std::vector<int> positions;        ///< Per-sequence position offset
        std::vector<int> sequence_lengths; ///< Per-sequence length (for padding)

        // === Activation Buffers (shared with ActivationBuffers) ===
        std::shared_ptr<TensorBase> normalized;
        std::shared_ptr<TensorBase> residual;
        std::shared_ptr<TensorBase> Q;
        std::shared_ptr<TensorBase> K;
        std::shared_ptr<TensorBase> V;
        std::shared_ptr<TensorBase> attn_output;
        std::shared_ptr<TensorBase> attn_proj;
        std::shared_ptr<TensorBase> gate;
        std::shared_ptr<TensorBase> up;
        std::shared_ptr<TensorBase> ffn_output;

        // === Hybrid Mode Buffers ===
        /// FP32 Q after RoPE (Hybrid mode only - avoids requantization)
        std::shared_ptr<TensorBase> Q_rope;
        /// FP32 K after RoPE (Hybrid mode only - avoids requantization)
        std::shared_ptr<TensorBase> K_rope;
        /// FP32 V dequantized (Hybrid mode only - for KV cache append when V is Q8_1)
        std::shared_ptr<TensorBase> V_dequant;

        // === HybridQ16 K Precision Fix Buffers ===
        /// Per-head dynamic scales for K vectors from RoPE Q16→Q16 path
        /// Shape: [batch_size * max_seq_len * n_kv_heads]
        /// Only used when GEMM outputs K as Q16_1 (K precision fix mode)
        std::vector<float> K_head_scales;

        // === Attention Workspace ===
        std::shared_ptr<TensorBase> workspace_scores;
        std::shared_ptr<TensorBase> workspace_context;
        std::shared_ptr<TensorBase> workspace_mask;

        // === Snapshot Buffers (for E2E debugging) ===
        /// Optional buffer to capture attention context before Wo projection
        /// Allocated when ENABLE_PIPELINE_SNAPSHOTS is defined
        std::shared_ptr<TensorBase> context_snapshot;

        /// Optional buffer to capture attention output (Wo projection, before residual)
        /// Shape: [batch_size * max_seq_len, d_model] - corresponds to ATTENTION_OUTPUT
        std::shared_ptr<TensorBase> attention_output_snapshot;

        /// Optional buffer to capture attention residual (after residual add)
        /// Shape: [batch_size * max_seq_len, d_model] - corresponds to ATTENTION_RESIDUAL
        std::shared_ptr<TensorBase> attention_residual_snapshot;

        // === Configuration ===
        int batch_size = 0;
        int max_seq_len = 0;
        int d_model = 0;
        int vocab_size = 0;
        DeviceId device_id = DeviceId::cpu();

        /**
         * @brief Check if state is initialized
         */
        bool isInitialized() const
        {
            return hidden != nullptr && logits != nullptr && batch_size > 0;
        }

        /**
         * @brief Clear state (reset positions, clear KV cache)
         */
        void clear()
        {
            if (kv_cache)
                kv_cache->clear();
            for (auto &[device, cache] : pp_kv_caches)
            {
                if (cache)
                    cache->clear();
            }
            std::fill(positions.begin(), positions.end(), 0);
            std::fill(sequence_lengths.begin(), sequence_lengths.end(), 0);
        }
    };

    /**
     * @brief Generic orchestrator for compute graph execution
     *
     * Separates execution concerns from graph definition, implementing:
     * - Graph execution via DeviceGraphExecutor
     * - Device context management with lazy initialization
     * - Graph caching for decode mode
     * - Execution state tracking
     *
     * This class is the imperative counterpart to declarative graph builders.
     * Currently supports Qwen2Graph, designed for extension to other architectures.
     *
     * Implements IInferenceRunner for unified inference API.
     */
    class DeviceGraphOrchestrator : public IInferenceRunner
    {
    public:
        // =========================================================================
        // Dependencies Struct for Interface-Based Construction (Testing Support)
        // =========================================================================

        /**
         * @brief Dependency injection container for testable construction
         *
         * Allows injection of interface implementations for unit testing:
         * - IModelContext: Provides model metadata and weights
         * - IMPITopology: Provides MPI topology information (optional)
         * - ICollectiveContext: Provides collective operations (optional)
         *
         * Usage:
         * @code
         * // In tests with mock dependencies
         * DeviceGraphOrchestrator::Dependencies deps;
         * deps.model_ctx = std::make_shared<MockModelContext>(config);
         * deps.topology = std::make_shared<MockMPITopology>(rank, world_size);
         * auto orchestrator = DeviceGraphOrchestrator(std::move(deps), graph_config);
         * @endcode
         */
        struct Dependencies
        {
            /// Model context providing weights and metadata (required)
            std::shared_ptr<IModelContext> model_ctx;

            /// MPI topology for work distribution (optional - nullptr for single-rank)
            std::shared_ptr<IMPITopology> topology = nullptr;

            /// Collective context for MPI operations (optional - nullptr for single-rank)
            std::shared_ptr<ICollectiveContext> collective_ctx = nullptr;
        };

        // =========================================================================
        // Constructors
        // =========================================================================

        /**
         * @brief Construct orchestrator with injected dependencies (preferred for testing)
         *
         * This constructor allows full dependency injection for unit testing:
         * - Mock model context provides weights and metadata without GGUF files
         * - Mock topology enables testing distributed logic without MPI
         * - Mock collective context enables testing sync stages in isolation
         *
         * @param deps Dependency injection container
         * @param graph_config Configuration for Qwen2Graph
         * @param cache_config Graph caching configuration
         */
        DeviceGraphOrchestrator(
            Dependencies deps,
            const GraphConfig &graph_config,
            const GraphCacheConfig &cache_config = {});

        /**
         * @brief Construct orchestrator with graph builder
         *
         * @param graph_builder Shared pointer to Qwen2Graph (graph definition)
         * @param mpi_ctx MPI context for distributed execution
         * @param cache_config Graph caching configuration
         */
        DeviceGraphOrchestrator(
            std::shared_ptr<Qwen2Graph> graph_builder,
            std::shared_ptr<MPIContext> mpi_ctx = nullptr,
            const GraphCacheConfig &cache_config = {});

        /**
         * @brief Construct orchestrator with graph config (creates internal graph builder)
         *
         * @param graph_config Configuration for Qwen2Graph
         * @param mpi_ctx MPI context for distributed execution
         * @param cache_config Graph caching configuration
         */
        DeviceGraphOrchestrator(
            const GraphConfig &graph_config,
            std::shared_ptr<MPIContext> mpi_ctx = nullptr,
            const GraphCacheConfig &cache_config = {});

        ~DeviceGraphOrchestrator() = default;

        // Non-copyable, movable
        DeviceGraphOrchestrator(const DeviceGraphOrchestrator &) = delete;
        DeviceGraphOrchestrator &operator=(const DeviceGraphOrchestrator &) = delete;
        DeviceGraphOrchestrator(DeviceGraphOrchestrator &&) = default;
        DeviceGraphOrchestrator &operator=(DeviceGraphOrchestrator &&) = default;

        // =========================================================================
        // Execution Methods (moved from Qwen2Graph)
        // =========================================================================

        /**
         * @brief Execute full forward pass
         *
         * Builds and executes the complete forward graph including:
         * - Embedding lookup
         * - All transformer layers
         * - Final normalization
         * - LM head projection
         *
         * @param input Forward pass input (tokens, sequence info)
         * @param output Forward pass output (logits buffer)
         * @return true if execution succeeded
         */
        bool executeForward(
            const Qwen2ForwardInput &input,
            Qwen2ForwardOutput &output);

        /**
         * @brief Execute attention block for a single layer
         *
         * Builds and executes attention graph:
         * - Pre-attention RMSNorm
         * - Q/K/V projections
         * - RoPE application
         * - Attention computation with KV cache
         * - Output projection
         * - Residual connection
         *
         * Uses cached graph for decode mode (seq_len=1) when enabled.
         *
         * @param layer Layer weights
         * @param buffers Activation buffers
         * @param layer_idx Layer index
         * @param seq_len Sequence length
         * @param kv_cache KV cache for attention
         * @param position_ids Position IDs for RoPE
         * @param device_idx Target device
         * @return true if execution succeeded
         */
        bool executeAttention(
            const LayerWeights &layer,
            ActivationBuffers &buffers,
            int layer_idx,
            int seq_len,
            IKVCache *kv_cache,
            const int *position_ids,
            DeviceId device);

        /**
         * @brief Execute FFN block for a single layer
         *
         * Builds and executes FFN graph:
         * - Pre-FFN RMSNorm
         * - Gate and Up projections
         * - SwiGLU activation
         * - Down projection
         * - Residual connection
         *
         * Uses cached graph for decode mode (seq_len=1) when enabled.
         *
         * @param layer Layer weights
         * @param buffers Activation buffers
         * @param layer_idx Layer index
         * @param seq_len Sequence length
         * @param device Target device
         * @return true if execution succeeded
         */
        bool executeFFN(
            const LayerWeights &layer,
            ActivationBuffers &buffers,
            int layer_idx,
            int seq_len,
            DeviceId device);

        /**
         * @brief Execute complete transformer layer (attention + FFN)
         *
         * Convenience method that executes both attention and FFN blocks.
         *
         * @param layer Layer weights
         * @param buffers Activation buffers
         * @param layer_idx Layer index
         * @param seq_len Sequence length
         * @param kv_cache KV cache for attention
         * @param position_ids Position IDs for RoPE
         * @param device Target device
         * @return true if execution succeeded
         */
        bool executeLayer(
            const LayerWeights &layer,
            ActivationBuffers &buffers,
            int layer_idx,
            int seq_len,
            IKVCache *kv_cache,
            const int *position_ids,
            DeviceId device);

        /**
         * @brief Execute a pre-built compute graph
         *
         * Low-level method for executing arbitrary graphs.
         *
         * @param graph ComputeGraph to execute
         * @param ctx Device context for execution
         * @return true if execution succeeded
         */
        bool execute(ComputeGraph &graph, IDeviceContext *ctx);

        // =========================================================================
        // Cache Management
        // =========================================================================

        /**
         * @brief Clear all caches (graphs, device contexts, state)
         */
        void clearCache();

        /**
         * @brief Invalidate graph cache for a specific layer
         *
         * @param layer_idx Layer index to invalidate (-1 for all)
         */
        void invalidateGraphCache(int layer_idx = -1);

        /**
         * @brief Check if a cached graph exists for a layer
         *
         * @param layer_idx Layer index
         * @param is_attention true for attention, false for FFN
         * @return true if valid cached graph exists
         */
        bool hasValidCachedGraph(int layer_idx, bool is_attention) const;

        /**
         * @brief Enable or disable graph caching
         *
         * @param enabled Whether caching should be enabled
         */
        void setGraphCachingEnabled(bool enabled);

        /**
         * @brief Check if graph caching is enabled
         */
        bool isGraphCachingEnabled() const { return cache_config_.enabled; }

        /**
         * @brief Initialize graph cache for n_layers
         *
         * Must be called before caching can be used.
         *
         * @param n_layers Number of transformer layers
         */
        void initializeGraphCache(int n_layers);

        // =========================================================================
        // Weight and Buffer Configuration
        // =========================================================================

        /**
         * @brief Set model weights for full forward pass
         *
         * Must be called before executeForward() to enable embedding lookup,
         * final normalization, and LM head projection.
         *
         * @param weights Model weights including embedding_table, final_norm, lm_head
         */
        void setWeights(const ModelWeights &weights);

        /**
         * @brief Set activation buffers for full forward pass
         *
         * @param buffers Model buffers including current_hidden, logits, layer_buffers
         */
        void setBuffers(const ModelBuffers &buffers);

        /**
         * @brief Check if weights are configured for full forward
         */
        bool hasGlobalWeights() const;

        // =========================================================================
        // Weight Manager and Phase-Aware Weight Access (Gap 3)
        // =========================================================================

        // =========================================================================
        // Weight Streaming (Option B)
        // =========================================================================

        /**
         * @brief Set weight streamer for on-demand layer weight transfer
         *
         * When set, the orchestrator will call streaming hooks during layer
         * execution to ensure weights are on-device and to prefetch upcoming layers.
         *
         * @param streamer Shared pointer to IWeightStreamer (nullptr to disable streaming)
         */
        void setWeightStreamer(std::shared_ptr<IWeightStreamer> streamer);

        /**
         * @brief Get weight streamer
         * @return Shared pointer to IWeightStreamer (may be nullptr)
         */
        std::shared_ptr<IWeightStreamer> weightStreamer() const { return weight_streamer_; }

        /**
         * @brief Check if weight streaming is active
         *
         * Returns true if a weight streamer is set and the current residency
         * mode is STREAMING (not RESIDENT or UNIFIED).
         *
         * @return true if streaming hooks are active
         */
        bool isWeightStreamingEnabled() const;

        // =========================================================================
        // Collective Context (NCCL/RCCL/PCIeBAR)
        // =========================================================================

        /**
         * @brief Set collective context for GPU-native collective operations
         *
         * When set, AllreduceStage and AllGatherStage execution will be intercepted
         * by DeviceGraphExecutor and routed through the BackendRouter for device-native
         * collectives (NCCL for CUDA, RCCL for ROCm, PCIeBAR for P2P).
         *
         * This eliminates the need for GPU→CPU→GPU transfers during tensor-parallel
         * inference, significantly reducing coherence overhead.
         *
         * @param collective_ctx Shared pointer to ICollectiveContext (nullptr to disable)
         */
        void setCollectiveContext(std::shared_ptr<ICollectiveContext> collective_ctx);

        /**
         * @brief Get collective context
         * @return Shared pointer to ICollectiveContext (may be nullptr)
         */
        std::shared_ptr<ICollectiveContext> collectiveContext() const { return injected_collective_ctx_; }

        /**
         * @brief Check if GPU-native collectives are enabled
         * @return true if CollectiveContext is set and ready
         */
        bool isGpuCollectivesEnabled() const { return injected_collective_ctx_ != nullptr; }

        /**
         * @brief Set TurboQuant context for TQ4 KV cache quantization.
         *
         * The context holds the rotation matrix used during KV cache quantization
         * and dequantization. Ownership is shared — the orchestrator keeps the
         * context alive for the entire inference session.
         *
         * @param ctx Shared pointer to TurboQuantContext
         */
        void setTurboQuantContext(std::shared_ptr<TurboQuantContext> ctx)
        {
            turboquant_ctx_ = std::move(ctx);
        }

        // =========================================================================
        // Weight Manager and Phase-Aware Weight Access (Gap 3)
        // =========================================================================

        /**
         * @brief Set weight manager for phase-aware weight access
         *
         * The weight manager provides access to both full weights (prefill)
         * and decode shards (decode) for CPU decode participation.
         *
         * @param weight_manager Shared pointer to WeightManager
         */
        void setWeightManager(std::shared_ptr<WeightManager> weight_manager);

        /**
         * @brief Retain model context to prevent dangling WeightManager references
         *
         * WeightManager stores an IModelLoader& reference to a ModelContext member.
         * For PP stages, the ModelContext must outlive the orchestrator to prevent
         * use-after-free when loading weights during inference.
         *
         * @param ctx Shared pointer to ModelContext (extends lifetime)
         */
        void retainModelContext(std::shared_ptr<IModelContext> ctx)
        {
            injected_model_ctx_ = std::move(ctx);
        }

        /**
         * @brief Get weight manager
         * @return Shared pointer to WeightManager (may be nullptr)
         */
        std::shared_ptr<WeightManager> weightManager() const { return weight_manager_; }

        /**
         * @brief Set weight placement map for decode device selection
         *
         * The placement map provides device info for phase-aware weight selection.
         *
         * @param placement_map Shared pointer to WeightPlacementMap
         */
        void setWeightPlacementMap(std::shared_ptr<WeightPlacementMap> placement_map);

        /**
         * @brief Get weight placement map
         * @return Shared pointer to WeightPlacementMap (may be nullptr)
         */
        std::shared_ptr<WeightPlacementMap> weightPlacementMap() const { return weight_placement_map_; }

        // =========================================================================
        // Tensor Parallel Configuration (Phase 1c: Proportional TP)
        // =========================================================================

        /**
         * @brief Set tensor parallelism configuration for proportional head assignment
         *
         * When set, the orchestrator uses TensorParallelConfig to determine
         * per-device head/FFN/vocab assignments instead of equal 1/world_size splits.
         * This enables heterogeneous GPU setups (e.g., NVIDIA 73% + AMD 27%).
         *
         * The config is propagated to the graph builder and used for:
         * - Buffer allocation sizing (Q/K/V/attention output)
         * - KV cache creation (local KV heads)
         * - Weight sharding hints
         *
         * @param config Shared pointer to TensorParallelConfig (nullptr to disable)
         */
        void setTensorParallelConfig(std::shared_ptr<TensorParallelConfig> config);

        /**
         * @brief Get tensor parallelism configuration
         * @return Shared pointer to TensorParallelConfig (may be nullptr)
         */
        std::shared_ptr<TensorParallelConfig> tensorParallelConfig() const { return tp_config_; }

        /**
         * @brief Check if proportional tensor parallelism is active
         * @return true if TensorParallelConfig is set and has proportional splits
         */
        bool isProportionalTPEnabled() const { return tp_config_ && tp_config_->isProportional(); }

        // =========================================================================
        // Pipeline Parallelism Configuration
        // =========================================================================

        /**
         * @brief Set pipeline parallelism stage configuration
         *
         * When set, this orchestrator runs as a PP stage, executing only a subset
         * of transformer layers. The configuration specifies:
         * - Layer range [first_layer, last_layer)
         * - Whether this stage owns embedding lookup
         * - Whether this stage owns final norm and LM head
         *
         * When PP config is set, executeForward() uses buildPartialForwardGraph()
         * instead of buildFullForwardGraph().
         *
         * @param config PP stage configuration
         */
        void setPPStageConfig(const FactoryPPStageConfig &config);

        /**
         * @brief Get pipeline parallelism stage configuration
         * @return Optional containing FactoryPPStageConfig if this is a PP stage
         */
        const std::optional<FactoryPPStageConfig> &ppStageConfig() const { return pp_stage_config_; }

        /**
         * @brief Check if this orchestrator is running as a PP stage
         * @return true if PP stage configuration is set
         */
        bool isPPStage() const { return pp_stage_config_.has_value(); }

        // =====================================================================
        // Unified Pipeline Configuration (Phase 6: Full PP+TP Integration)
        // =====================================================================

        /**
         * @brief Set unified pipeline configuration for PP+TP composition
         *
         * When set, the orchestrator can build and execute unified graphs that
         * span multiple PP stages with internal TP. This replaces the need for
         * external coordinators that manually sequence PP stages.
         *
         * The orchestrator will:
         * - Create ILocalPPContext for each inter-stage transfer
         * - Create ILocalTPContext for each TP domain
         * - Build unified graphs via buildUnifiedPipelineGraph()
         * - Execute the full pipeline in a single forward() call
         *
         * @param config PipelineConfig with TP domains and PP stages
         */
        void setPipelineConfig(std::shared_ptr<PipelineConfig> config);

        /**
         * @brief Get the unified pipeline configuration
         * @return Shared pointer to PipelineConfig (may be nullptr)
         */
        std::shared_ptr<PipelineConfig> pipelineConfig() const { return pipeline_config_; }

        /**
         * @brief Check if unified PP mode is enabled
         * @return true if PipelineConfig is set with multiple PP stages
         */
        bool hasUnifiedPP() const { return pipeline_config_ && pipeline_config_->hasPP(); }

        /**
         * @brief Initialize PP contexts for inter-stage activation transfers
         *
         * Creates ILocalPPContext instances for each pair of adjacent PP stages.
         * Must be called after setPipelineConfig() and before forward().
         *
         * @return true if initialization succeeded
         */
        bool initializePPContexts();

        /**
         * @brief Initialize TP contexts for each domain
         *
         * Creates ILocalTPContext instances for each TP domain in the config.
         * Must be called after setPipelineConfig() and before forward().
         *
         * @return true if initialization succeeded
         */
        bool initializeTPContexts();

        // =====================================================================
        // Hidden State API (for Pipeline Parallelism)
        // =====================================================================

        TensorBase *getHiddenState() override;
        const TensorBase *getHiddenState() const override;
        void setHiddenState(TensorBase *hidden_state) override;
        bool hasHiddenStateInput() const override;
        void clearHiddenStateInput() override;

        // =========================================================================
        // Multi-Domain Tensor Parallel Configuration (Phase 6.3: Heterogeneous TP)
        // =========================================================================

        /**
         * @brief Set multi-domain tensor parallelism configuration
         *
         * When set, enables heterogeneous tensor parallelism with separate
         * domains for different compute operations (e.g., GPU domain for attention,
         * CPU domain for FFN). Each domain has its own MPI communicator.
         *
         * The config is:
         * - Propagated to graph builder's config.multi_domain_tp_config
         * - Used by getDomainForLayer() to route AllreduceStage calls
         *
         * @param config Shared pointer to MultiDomainTPConfig (nullptr to disable)
         */
        void setDomainConfig(std::shared_ptr<MultiDomainTPConfig> config);

        /**
         * @brief Get multi-domain tensor parallelism configuration
         * @return Shared pointer to MultiDomainTPConfig (may be nullptr)
         */
        std::shared_ptr<MultiDomainTPConfig> domainConfig() const { return domain_config_; }

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

        /**
         * @brief Set current inference phase (low-level, no logging)
         *
         * Changes the inference phase which affects weight selection:
         * - PREFILL: Uses full weights from GPU (compute-bound)
         * - DECODE: May use CPU decode shards if participation enabled
         *
         * Prefer using transitionToPhase() for explicit transitions with logging.
         *
         * @param phase The inference phase to set
         */
        void setPhase(InferencePhase phase) { current_phase_ = phase; }

        void setSuppressTimeline(bool suppress) override { suppress_timeline_ = suppress; }

        void setAccumulatePrefill(bool accumulate) override { accumulate_prefill_ = accumulate; }

        void flushStageTimeline() override
        {
            if (!debugEnv().gpu_stage_timing)
                return;

            auto &timeline = executor_.stageTimeline();
            std::string dev_str = state_.device_id.toString();

            // Print accumulated prefill summary if any
            if (timeline.hasAccumulatedPrefillData())
            {
                timeline.printAccumulatedPrefillSummary(dev_str.c_str());
            }

            // Print accumulated decode summary if any
            if (timeline.hasAccumulatedData())
            {
                timeline.printAccumulatedSummary("DECODE", dev_str.c_str());
            }
        }

        /**
         * @brief Transition to a new inference phase with logging
         *
         * Use this method for explicit phase transitions (e.g., from tests or
         * when manually controlling prefill/decode phases). Logs the transition
         * at DEBUG level if the phase actually changes.
         *
         * The phase affects weight selection via getPhaseAwareWeight():
         * - PREFILL: Uses full weights from GPU (compute-bound)
         * - DECODE: May use CPU decode shards if participation enabled
         *
         * @param phase The new inference phase
         */
        void transitionToPhase(InferencePhase phase);

        /**
         * @brief Get current inference phase
         */
        InferencePhase getPhase() const { return current_phase_; }

        /**
         * @brief Get weight tensor appropriate for current inference phase
         *
         * For PREFILL: Returns full weight from GPU (compute-bound - needs all weights)
         * For DECODE: Returns decode shard if CPU is participating, else full weight
         *
         * This enables "Option A: Selective Duplication" where CPU only participates
         * in decode phase with a subset of weights.
         *
         * @param name Weight tensor name (e.g., "blk.0.attn_q.weight")
         * @param layer_idx Layer index for placement lookup
         * @param phase Inference phase (overrides current_phase_ if provided)
         * @return Shared pointer to weight tensor, or nullptr on error
         */
        std::shared_ptr<TensorBase> getPhaseAwareWeight(
            const std::string &name,
            int layer_idx,
            InferencePhase phase) const;

        /**
         * @brief Get weight for current phase (uses current_phase_)
         *
         * Convenience overload that uses the orchestrator's current phase.
         *
         * @param name Weight tensor name
         * @param layer_idx Layer index for placement lookup
         * @return Shared pointer to weight tensor, or nullptr on error
         */
        std::shared_ptr<TensorBase> getPhaseAwareWeight(
            const std::string &name,
            int layer_idx) const
        {
            return getPhaseAwareWeight(name, layer_idx, current_phase_);
        }

        /**
         * @brief Check if this rank should participate in CPU decode
         *
         * Returns true if:
         * - Phase is DECODE
         * - WeightPlacementMap indicates CPU decode participation
         * - This MPI rank is the designated CPU decode participant
         *
         * @param name Weight tensor name
         * @param layer_idx Layer index
         * @return true if this rank handles CPU decode shard for this weight
         */
        bool shouldUseCPUDecodeWeight(const std::string &name, int layer_idx) const;

        // =========================================================================
        // Graph Buffer Management (Phase 3 - moved from Qwen2Graph)
        // =========================================================================

        /**
         * @brief Set TensorFactory for graph-managed buffer allocation
         * @param factory TensorFactory pointer (not owned)
         */
        void setTensorFactory(TensorFactory *factory) { tensor_factory_ = factory; }

        /**
         * @brief Get TensorFactory
         * @return TensorFactory pointer (nullptr if not set)
         */
        TensorFactory *tensorFactory() const { return tensor_factory_; }

        /**
         * @brief Initialize activation buffers using BufferArena
         *
         * Allocates all activation buffers with automatic aliasing optimization
         * for SCRATCH buffers. This is an alternative to manual buffer allocation.
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
         * @brief Check if graph buffer management is active
         */
        bool hasGraphManagedBuffers() const { return arena_ != nullptr; }

        /**
         * @brief Get internal activation buffers (for graph-managed mode)
         *
         * When using graph-managed buffers, the pipeline should use these
         * instead of creating its own buffer mappings.
         *
         * @return Reference to internal activation buffers
         */
        ActivationBuffers &getInternalBuffers();
        const ActivationBuffers &getInternalBuffers() const;

        /**
         * @brief Get model-level buffers (current_hidden, logits)
         *
         * When using graph-managed buffers, these are allocated by the orchestrator.
         *
         * @return Reference to model buffers
         */
        const ModelBuffers &getModelBuffers() const;

        /**
         * @brief Get buffer arena allocation statistics
         *
         * @return ArenaAllocationStats or nullptr if not using graph buffer management
         */
        const ArenaAllocationStats *bufferStats() const;

        // =========================================================================
        // Inference State Management (Phase 5)
        // =========================================================================

        /**
         * @brief Initialize inference state owned by orchestrator
         *
         * Allocates all buffers needed for inference. After calling this,
         * the simplified forward() API can be used without passing buffers.
         *
         * @param batch_size Maximum batch size
         * @param max_seq_len Maximum sequence length
         * @param device_id Device for buffer allocation (default: CPU)
         * @param init_config Configuration for buffer allocation (mapped memory, etc.)
         * @return true if initialization succeeded
         */
        bool initializeInferenceState(
            int batch_size,
            int max_seq_len,
            DeviceId device_id = DeviceId::cpu(),
            const InferenceStateInitConfig &init_config = InferenceStateInitConfig{});

        /**
         * @brief Check if inference state is initialized
         */
        bool hasInferenceState() const { return state_.isInitialized(); }

        /**
         * @brief Get inference state (read-only)
         */
        const InferenceState &inferenceState() const { return state_; }

        /**
         * @brief Simplified forward pass using orchestrator-owned state
         *
         * This is the high-level API for inference. The orchestrator manages
         * all buffers and state internally.
         *
         * @param tokens Token IDs [batch_size * seq_len]
         * @param seq_len Sequence length per batch item
         * @param batch_size Number of sequences (default 1)
         * @return Pointer to logits buffer, or nullptr on failure
         */
        const float *forward(
            const int *tokens,
            int seq_len,
            int batch_size = 1);

        /**
         * @brief Get logits from last forward pass
         *
         * @return Pointer to logits buffer, or nullptr if not available
         */
        const float *logits() const override;

        /**
         * @brief Get current position offset for a sequence
         *
         * @param seq_idx Sequence index (default 0)
         * @return Current position offset
         */
        int getPosition(int seq_idx = 0) const;

        /**
         * @brief Clear inference state (reset positions, clear KV cache)
         */
        void clearInferenceState();

        // =========================================================================
        // Fluent Graph Building API
        // =========================================================================

        /**
         * @brief Result of a graph build operation (nested class)
         */
        class GraphBuildResult
        {
        public:
            GraphBuildResult() = default;
            GraphBuildResult(ComputeGraph graph, Qwen2ForwardOutput output)
                : graph_(std::move(graph)), output_(output), success_(true) {}
            explicit GraphBuildResult(std::string error)
                : error_(std::move(error)), success_(false) {}

            [[nodiscard]] bool success() const { return success_; }
            [[nodiscard]] bool failed() const { return !success_; }
            [[nodiscard]] const std::string &error() const { return error_; }
            [[nodiscard]] ComputeGraph &graph() { return graph_; }
            [[nodiscard]] const ComputeGraph &graph() const { return graph_; }
            [[nodiscard]] const Qwen2ForwardOutput &output() const { return output_; }
            [[nodiscard]] ComputeGraph takeGraph() { return std::move(graph_); }
            explicit operator bool() const { return success_; }

        private:
            ComputeGraph graph_;
            Qwen2ForwardOutput output_{};
            std::string error_;
            bool success_ = false;
        };

        /**
         * @brief Fluent builder for compute graph composition (nested class)
         */
        class GraphBuildSession
        {
        public:
            explicit GraphBuildSession(DeviceGraphOrchestrator &orchestrator)
                : orchestrator_(orchestrator) {}

            // Input configuration
            GraphBuildSession &forInput(const Qwen2ForwardInput &input);
            GraphBuildSession &withPositionIds(const int *position_ids);
            GraphBuildSession &withExternalHiddenState(TensorBase *hidden_state);

            // Pipeline configuration
            GraphBuildSession &withPipelineConfig(std::shared_ptr<PipelineConfig> config);
            GraphBuildSession &forPPStage(int first_layer, int last_layer,
                                          bool has_embedding = false, bool has_lm_head = false);
            GraphBuildSession &withPPContext(int from_stage, int to_stage, ILocalPPContext *context);
            GraphBuildSession &withTPContext(const std::string &domain_name, ILocalTPContext *context);

            // Resource configuration
            GraphBuildSession &withWeights(const ModelWeights &weights);
            GraphBuildSession &withBuffers(const ModelBuffers &buffers);
            GraphBuildSession &withKVCache(IKVCache *kv_cache);

            // Build methods (terminal operations)
            [[nodiscard]] GraphBuildResult buildForward();
            [[nodiscard]] GraphBuildResult buildPartial();
            [[nodiscard]] GraphBuildResult buildUnified();
            [[nodiscard]] GraphBuildResult build();

            // Validation
            [[nodiscard]] bool isValid() const;
            [[nodiscard]] std::string validationError() const;

        private:
            DeviceGraphOrchestrator &orchestrator_;
            std::optional<Qwen2ForwardInput> input_;
            const int *explicit_position_ids_ = nullptr;
            TensorBase *external_hidden_state_ = nullptr;
            std::shared_ptr<PipelineConfig> pipeline_config_;
            struct PPStageSpec
            {
                int first_layer;
                int last_layer;
                bool has_embedding;
                bool has_lm_head;
            };
            std::optional<PPStageSpec> pp_stage_;
            std::map<std::pair<int, int>, ILocalPPContext *> pp_contexts_;
            std::map<std::string, ILocalTPContext *> tp_contexts_;
            std::optional<ModelWeights> weights_;
            std::optional<ModelBuffers> buffers_;
            IKVCache *kv_cache_ = nullptr;

            Qwen2ForwardInput prepareInput() const;
            void applyConfiguration();
        };

        /**
         * @brief Result of a sub-graph build operation (attention, FFN)
         *
         * Lightweight result type for sub-graph building that doesn't need output tracking.
         */
        class SubGraphBuildResult
        {
        public:
            SubGraphBuildResult() = default;
            explicit SubGraphBuildResult(ComputeGraph graph)
                : graph_(std::move(graph)), success_(true) {}
            explicit SubGraphBuildResult(std::string error)
                : error_(std::move(error)), success_(false) {}

            [[nodiscard]] bool success() const { return success_; }
            [[nodiscard]] bool failed() const { return !success_; }
            [[nodiscard]] const std::string &error() const { return error_; }
            [[nodiscard]] ComputeGraph &graph() { return graph_; }
            [[nodiscard]] const ComputeGraph &graph() const { return graph_; }
            [[nodiscard]] ComputeGraph takeGraph() { return std::move(graph_); }
            explicit operator bool() const { return success_; }

        private:
            ComputeGraph graph_;
            std::string error_;
            bool success_ = false;
        };

        /**
         * @brief Fluent builder for attention sub-graph
         *
         * Provides a clear, chainable API for building attention block graphs.
         *
         * @code
         * auto result = buildAttentionGraph()
         *     .forLayer(layer, layer_idx)
         *     .withBuffers(buffers)
         *     .withSequence(seq_len)
         *     .onDevice(device)
         *     .withKVCache(kv_cache)
         *     .withPositionIds(position_ids)
         *     .build();
         * @endcode
         */
        class AttentionGraphSession
        {
        public:
            explicit AttentionGraphSession(DeviceGraphOrchestrator &orchestrator)
                : orchestrator_(orchestrator) {}

            // Required configuration
            AttentionGraphSession &forLayer(const LayerWeights &layer, int layer_idx);
            AttentionGraphSession &withBuffers(ActivationBuffers &buffers);
            AttentionGraphSession &withSequence(int seq_len, int batch_size = 1);
            AttentionGraphSession &onDevice(DeviceId device);

            // Optional configuration
            AttentionGraphSession &withKVCache(IKVCache *kv_cache);
            AttentionGraphSession &withPositionIds(const int *position_ids);
            AttentionGraphSession &withSequenceLengths(const std::vector<int> *lengths);

            // Build (terminal operation)
            [[nodiscard]] SubGraphBuildResult build();

            // Validation
            [[nodiscard]] bool isValid() const;
            [[nodiscard]] std::string validationError() const;

        private:
            DeviceGraphOrchestrator &orchestrator_;

            // Required
            const LayerWeights *layer_ = nullptr;
            ActivationBuffers *buffers_ = nullptr;
            int layer_idx_ = -1;
            int seq_len_ = 0;
            int batch_size_ = 1;
            std::optional<DeviceId> device_;

            // Optional
            IKVCache *kv_cache_ = nullptr;
            const int *position_ids_ = nullptr;
            const std::vector<int> *sequence_lengths_ = nullptr;
        };

        /**
         * @brief Fluent builder for FFN sub-graph
         *
         * Provides a clear, chainable API for building FFN block graphs.
         *
         * @code
         * auto result = buildFFNGraph()
         *     .forLayer(layer, layer_idx)
         *     .withBuffers(buffers)
         *     .withSequence(seq_len)
         *     .onDevice(device)
         *     .build();
         * @endcode
         */
        class FFNGraphSession
        {
        public:
            explicit FFNGraphSession(DeviceGraphOrchestrator &orchestrator)
                : orchestrator_(orchestrator) {}

            // Required configuration
            FFNGraphSession &forLayer(const LayerWeights &layer, int layer_idx);
            FFNGraphSession &withBuffers(ActivationBuffers &buffers);
            FFNGraphSession &withSequence(int seq_len, int batch_size = 1);
            FFNGraphSession &onDevice(DeviceId device);

            // Build (terminal operation)
            [[nodiscard]] SubGraphBuildResult build();

            // Validation
            [[nodiscard]] bool isValid() const;
            [[nodiscard]] std::string validationError() const;

        private:
            DeviceGraphOrchestrator &orchestrator_;

            // Required
            const LayerWeights *layer_ = nullptr;
            ActivationBuffers *buffers_ = nullptr;
            int layer_idx_ = -1;
            int seq_len_ = 0;
            int batch_size_ = 1;
            std::optional<DeviceId> device_;
        };

        /**
         * @brief Start a fluent graph build session
         *
         * Returns a GraphBuildSession for composing and building compute graphs
         * with a clear, chainable API.
         *
         * @code
         * auto result = buildGraph()
         *     .forInput(input)
         *     .build();
         *
         * if (result.success()) {
         *     executor.execute(result.graph(), context);
         * }
         * @endcode
         *
         * @return GraphBuildSession for fluent configuration
         */
        [[nodiscard]] GraphBuildSession buildGraph() { return GraphBuildSession(*this); }

        /**
         * @brief Start a fluent attention graph build session
         *
         * @return AttentionGraphSession for fluent configuration
         */
        [[nodiscard]] AttentionGraphSession buildAttentionGraph() { return AttentionGraphSession(*this); }

        /**
         * @brief Start a fluent FFN graph build session
         *
         * @return FFNGraphSession for fluent configuration
         */
        [[nodiscard]] FFNGraphSession buildFFNGraph() { return FFNGraphSession(*this); }

        // =========================================================================
        // Accessors
        // =========================================================================

        /**
         * @brief Get the underlying graph builder
         */
        Qwen2Graph *graphBuilder() { return graph_builder_.get(); }
        const Qwen2Graph *graphBuilder() const { return graph_builder_.get(); }

        /**
         * @brief Get the underlying executor
         */
        DeviceGraphExecutor &executor() { return executor_; }
        const DeviceGraphExecutor &executor() const { return executor_; }

        /**
         * @brief Get device context for a device (creates if needed)
         *
         * @param device Device identifier
         * @return Device context pointer (owned by orchestrator)
         */
        IDeviceContext *getDeviceContext(DeviceId device);

        // =========================================================================
        // IInferenceRunner Interface Implementation
        // =========================================================================

        /**
         * @brief Run forward pass (IInferenceRunner override)
         */
        bool forward(const int *tokens, int seq_len) override
        {
            return forward(tokens, seq_len, 1) != nullptr;
        }

        /**
         * @brief Get logits (IInferenceRunner override - already declared above)
         */
        // const float *logits() const; - declared above

        /**
         * @brief Get vocabulary size (IInferenceRunner override)
         */
        int vocab_size() const override { return graph_builder_ ? graph_builder_->config().vocab_size : 0; }

        /**
         * @brief Clear KV cache (IInferenceRunner override)
         */
        void clear_cache() override
        {
            for (auto &[dev, ctx] : device_contexts_)
            {
                if (ctx && dev.is_gpu())
                    ctx->synchronize();
            }
            // Invalidate graph caches but preserve device contexts
            for (auto &entry : layer_graph_cache_)
            {
                entry.valid = false;
            }
            forward_graph_cache_.clear();
            last_pos_offset_ = -1;
            cache_stats_ = CacheStats{};
            state_.clear();
            arena_.reset();
            // Reset input-dependent cached state on all kernels (e.g., stale token IDs)
            resetKernelDynamicState();
            ++session_epoch_;
        }

        /**
         * @brief Get current position (IInferenceRunner override)
         */
        int get_position() const override { return getPosition(0); }

        /**
         * @brief Get execution path (always GRAPH)
         */
        ExecutionPath executionPath() const override { return ExecutionPath::GRAPH; }

        /**
         * @brief Get architecture name
         */
        const char *architecture() const override { return "qwen2"; }

        /**
         * @brief Get executor statistics for profiling
         */
        const GraphExecutorStats *executorStats() const override { return &executor_.stats(); }

        /**
         * @brief Reset executor statistics
         */
        void resetExecutorStats() override { executor_.resetStats(); }

        // =========================================================================
        // Batch Interface (IInferenceRunner overrides)
        // =========================================================================

        /**
         * @brief Batched forward pass with variable-length sequences
         *
         * @param token_batches Vector of token sequences
         * @return true if forward pass succeeded
         */
        bool forward_batch(const std::vector<std::vector<int>> &token_batches) override;

        /**
         * @brief Get logits for a specific sequence in batch
         *
         * @param seq_idx Sequence index in batch (default=0)
         * @return Pointer to logits [padded_seq_len, vocab_size], or nullptr
         */
        const float *getLogits(int seq_idx = 0) const override;

        /**
         * @brief Get current batch size
         */
        int batch_size() const override { return state_.batch_size; }

        /**
         * @brief Get padded sequence length for current batch
         */
        int padded_seq_len() const override { return padded_seq_len_; }

        /**
         * @brief Get sequence lengths for current batch
         */
        const std::vector<int> &sequence_lengths() const override { return state_.sequence_lengths; }

        // =========================================================================
        // Snapshot Capture API (Pipeline-compatible for E2E testing)
        // =========================================================================

        /**
         * @brief Extract FP32 data from a StageDumpInfo output buffer
         *
         * Handles dequantization for Q8_1, Q16_1, and other quantized formats.
         * For FP32 outputs, performs a simple copy.
         *
         * @param out Output buffer from StageDumpInfo
         * @return Vector of FP32 values, or empty if extraction fails
         */
        static std::vector<float> extractFp32FromOutput(const StageDumpInfo::OutputBuffer &out)
        {
            if (!out.data)
                return {};

            size_t count = out.rows * out.cols;
            if (count == 0)
                return {};

            std::vector<float> data(count);
            std::string dtype_str = out.dtype ? out.dtype : "FP32";

            LOG_TRACE("[extractFp32FromOutput] name=" << (out.name ? out.name : "?")
                                                      << " dtype=" << dtype_str << " rows=" << out.rows << " cols=" << out.cols);

            // FP32: direct copy
            if (dtype_str == "FP32")
            {
                std::memcpy(data.data(), out.data, count * sizeof(float));
                return data;
            }

            // Q8_1: dequantize blocks
            if (dtype_str == "Q8_1")
            {
                const Q8_1Block *blocks = static_cast<const Q8_1Block *>(out.data);
                constexpr int BLOCK_SIZE = 32;
                size_t num_blocks = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;

                for (size_t b = 0; b < num_blocks; ++b)
                {
                    const Q8_1Block &block = blocks[b];
                    float scale = fp16_to_fp32(block.d);
                    for (int i = 0; i < BLOCK_SIZE && b * BLOCK_SIZE + i < count; ++i)
                    {
                        data[b * BLOCK_SIZE + i] = static_cast<float>(block.qs[i]) * scale;
                    }
                }
                return data;
            }

            // Q16_1 variants: dequantize blocks (block sizes 32, 64, 128)
            if (dtype_str.find("Q16_1") == 0)
            {
                // Determine block size from dtype string (Q16_1_32, Q16_1_64, Q16_1_128)
                int block_size = 32; // Default
                if (dtype_str.find("_64") != std::string::npos)
                    block_size = 64;
                else if (dtype_str.find("_128") != std::string::npos)
                    block_size = 128;

                const Q16_1Block *blocks = static_cast<const Q16_1Block *>(out.data);
                size_t num_blocks = (count + block_size - 1) / block_size;

                for (size_t b = 0; b < num_blocks; ++b)
                {
                    const Q16_1Block &block = blocks[b];
                    float scale = fp16_to_fp32(block.d);
                    for (int i = 0; i < block_size && b * block_size + i < count; ++i)
                    {
                        data[b * block_size + i] = static_cast<float>(block.qs[i]) * scale;
                    }
                }
                return data;
            }

            // BF16 or FP16: convert to FP32
            if (dtype_str == "BF16" || dtype_str == "FP16")
            {
                const uint16_t *half_data = static_cast<const uint16_t *>(out.data);
                for (size_t i = 0; i < count; ++i)
                {
                    if (dtype_str == "BF16")
                    {
                        data[i] = simd::bf16_to_fp32(half_data[i]);
                    }
                    else
                    {
                        data[i] = simd::fp16_to_fp32(half_data[i]);
                    }
                }
                return data;
            }

            // Unknown dtype - warn and try FP32 (may be garbage)
            LOG_WARN("[extractFp32FromOutput] Unknown dtype '" << dtype_str << "', assuming FP32");
            std::memcpy(data.data(), out.data, count * sizeof(float));
            return data;
        }

        /**
         * @brief Enable snapshot capture of intermediate activations
         *
         * When enabled, orchestrator stores copies of intermediate tensors from
         * each stage execution. Used for parity testing against reference implementations.
         *
         * @param output_dir Optional directory to save snapshots (currently unused)
         */
        void enableSnapshotCapture(const std::string &output_dir = "") override
        {
            (void)output_dir; // Reserved for future disk output
            snapshots_.clear();
            snapshot_enabled_ = true;

            LOG_INFO("[DeviceGraphOrchestrator::enableSnapshotCapture] Setting callback on executor_");
            executor_.setSnapshotCallback(
                [this](const std::string &name, const StageDumpInfo &dump)
                {
                    LOG_TRACE("[Snapshot] Callback invoked for stage: " << name << " outputs.size=" << dump.outputs.size());
                    // Handle fused QKV stage specially - split into separate Q, K, V snapshots
                    // The fused stage outputs are ordered: Q (output_q), K (output_k), V (output_v)
                    if (name.find("_qkv_proj") != std::string::npos)
                    {
                        // Extract layer prefix (e.g., "layer0" from "layer0_qkv_proj")
                        size_t qkv_pos = name.find("_qkv_proj");
                        std::string prefix = name.substr(0, qkv_pos);

                        // Store Q, K, V separately with pipeline-compatible keys
                        if (dump.outputs.size() >= 3)
                        {
                            // Output 0 = Q
                            if (dump.outputs[0].data)
                            {
                                auto data = extractFp32FromOutput(dump.outputs[0]);
                                if (!data.empty())
                                    snapshots_[prefix + "_Q_PROJECTION"] = {std::move(data), dump.outputs[0].rows, dump.outputs[0].cols};
                            }
                            // Output 1 = K
                            if (dump.outputs[1].data)
                            {
                                auto data = extractFp32FromOutput(dump.outputs[1]);
                                if (!data.empty())
                                    snapshots_[prefix + "_K_PROJECTION"] = {std::move(data), dump.outputs[1].rows, dump.outputs[1].cols};
                            }
                            // Output 2 = V
                            if (dump.outputs[2].data)
                            {
                                auto data = extractFp32FromOutput(dump.outputs[2]);
                                if (!data.empty())
                                    snapshots_[prefix + "_V_PROJECTION"] = {std::move(data), dump.outputs[2].rows, dump.outputs[2].cols};
                            }
                        }
                        return;
                    }

                    // Handle fused Gate/Up stage specially - split into separate GATE and UP snapshots
                    if (name.find("_gate_up") != std::string::npos)
                    {
                        size_t pos = name.find("_gate_up");
                        std::string prefix = name.substr(0, pos);

                        if (dump.outputs.size() >= 2)
                        {
                            // Output 0 = gate
                            if (dump.outputs[0].data)
                            {
                                auto data = extractFp32FromOutput(dump.outputs[0]);
                                if (!data.empty())
                                    snapshots_[prefix + "_FFN_GATE"] = {std::move(data), dump.outputs[0].rows, dump.outputs[0].cols};
                            }
                            // Output 1 = up
                            if (dump.outputs[1].data)
                            {
                                auto data = extractFp32FromOutput(dump.outputs[1]);
                                if (!data.empty())
                                    snapshots_[prefix + "_FFN_UP"] = {std::move(data), dump.outputs[1].rows, dump.outputs[1].cols};
                            }
                        }
                        return;
                    }

                    // Handle RoPE stage - captures Q_ROPE and K_ROPE
                    if (name.find("_rope") != std::string::npos && name.find("_q_rope") == std::string::npos && name.find("_k_rope") == std::string::npos)
                    {
                        // Generic "_rope" stage (fused Q/K RoPE)
                        size_t pos = name.find("_rope");
                        std::string prefix = name.substr(0, pos);

                        // RoPE stage outputs are Q and K after RoPE application
                        if (dump.outputs.size() >= 2)
                        {
                            if (dump.outputs[0].data)
                            {
                                auto data = extractFp32FromOutput(dump.outputs[0]);
                                if (!data.empty())
                                    snapshots_[prefix + "_Q_ROPE"] = {std::move(data), dump.outputs[0].rows, dump.outputs[0].cols};
                            }
                            if (dump.outputs[1].data)
                            {
                                auto data = extractFp32FromOutput(dump.outputs[1]);
                                if (!data.empty())
                                    snapshots_[prefix + "_K_ROPE"] = {std::move(data), dump.outputs[1].rows, dump.outputs[1].cols};
                            }
                        }
                        return;
                    }

                    // Handle fused attention+Wo stage - captures ATTENTION_CONTEXT, ATTENTION_OUTPUT, and ATTENTION_RESIDUAL
                    // FusedAttentionWoStage::getDumpInfo() outputs are named (in order):
                    //   "context" (ATTENTION_CONTEXT) - pre-Wo attention output
                    //   "attention_output" (ATTENTION_OUTPUT) - post-Wo projection, before residual add
                    //   "attention_residual" (ATTENTION_RESIDUAL) - after residual add
                    //   "output" - primary output tensor (may be Q16_1 for HybridQ16)
                    // The outputs vector preserves this order for non-null snapshots.
                    if (name.find("_fused_attn_wo") != std::string::npos)
                    {
                        size_t pos = name.find("_fused_attn_wo");
                        std::string prefix = name.substr(0, pos);

                        LOG_TRACE("[Snapshot] fused_attn_wo handler: prefix=" << prefix
                                                                              << " dump.outputs.size()=" << dump.outputs.size());

                        // Iterate through outputs and map by name to snapshot keys
                        for (size_t i = 0; i < dump.outputs.size(); ++i)
                        {
                            const auto &out = dump.outputs[i];
                            if (!out.data || !out.name)
                                continue;

                            std::string key;
                            std::string out_name(out.name);

                            // Match by name from StageDumpInfo::OutputBuffer
                            if (out_name == "context")
                            {
                                key = prefix + "_ATTENTION_CONTEXT";
                            }
                            else if (out_name == "attention_output")
                            {
                                key = prefix + "_ATTENTION_OUTPUT";
                            }
                            else if (out_name == "attention_residual")
                            {
                                key = prefix + "_ATTENTION_RESIDUAL";
                            }
                            else if (out_name == "output")
                            {
                                // Skip primary output - it's handled by residual stage or is Q16_1
                                continue;
                            }
                            else
                            {
                                LOG_WARN("[Snapshot] fused_attn_wo unknown output name: " << out_name);
                                continue;
                            }

                            LOG_TRACE("[Snapshot] Storing " << key << ": rows=" << out.rows
                                                            << " cols=" << out.cols);
                            auto data = extractFp32FromOutput(out);
                            if (!data.empty())
                                snapshots_[key] = {std::move(data), out.rows, out.cols};
                        }
                        return;
                    }

                    // Handle lm_head_allgather - store as LM_HEAD (overwrites partial output)
                    // In TP mode, lm_head produces partial vocab logits, but parity tests expect
                    // full vocab. The allgather output is the correct comparison point.
                    if (name == "lm_head_allgather")
                    {
                        if (!dump.outputs.empty() && dump.outputs[0].data)
                        {
                            const auto &out = dump.outputs[0];
                            auto data = extractFp32FromOutput(out);
                            LOG_DEBUG("[Snapshot] lm_head_allgather handler: storing as LM_HEAD (overwriting partial), count=" << data.size());
                            if (!data.empty())
                                snapshots_["LM_HEAD"] = {std::move(data), out.rows, out.cols}; // Overwrite partial with full
                        }
                        return;
                    }

                    // Standard single-output stages
                    LOG_DEBUG("[Snapshot] Standard path: stage=" << name
                                                                 << " outputs.size=" << dump.outputs.size()
                                                                 << " out[0].data=" << (dump.outputs.empty() ? nullptr : dump.outputs[0].data));
                    if (!dump.outputs.empty() && dump.outputs[0].data)
                    {
                        const auto &out = dump.outputs[0];
                        auto data = extractFp32FromOutput(out);

                        // Convert graph stage name to pipeline-style key
                        std::string key = convertStageNameToSnapshotKey(name);
                        LOG_DEBUG("[Snapshot] Storing key=" << key << " count=" << data.size());

                        // Debug: Log first few values of extracted data
                        if (data.size() >= 8 && key == "EMBEDDING")
                        {
                            LOG_DEBUG("[Snapshot] " << key << " first 8 values: "
                                                    << data[0] << "," << data[1] << "," << data[2] << "," << data[3] << ","
                                                    << data[4] << "," << data[5] << "," << data[6] << "," << data[7]);
                        }

                        if (!data.empty())
                            snapshots_[key] = {std::move(data), out.rows, out.cols};
                    }
                });
        }

        /**
         * @brief Disable snapshot capture and clear stored snapshots
         */
        void disableSnapshotCapture() override
        {
            snapshot_enabled_ = false;
            snapshots_.clear();
            executor_.setSnapshotCallback(nullptr);
        }

        /**
         * @brief Clear stored snapshots but keep capture enabled
         */
        void clearSnapshots() override
        {
            snapshots_.clear();
        }

        /**
         * @brief Retrieve a captured snapshot by key
         *
         * @param key Snapshot identifier (e.g., "layer0_Q_PROJECTION", "EMBEDDING")
         * @param out_size Output parameter for tensor size (number of float elements)
         * @return Pointer to snapshot data, or nullptr if key doesn't exist
         */
        const float *getSnapshot(const std::string &key, size_t &out_size) const override
        {
            auto it = snapshots_.find(key);
            if (it == snapshots_.end())
            {
                LOG_DEBUG("[DeviceGraphOrchestrator::getSnapshot] Key NOT FOUND: " << key
                                                                                   << " (have " << snapshots_.size() << " snapshots)");
                out_size = 0;
                return nullptr;
            }
            out_size = it->second.data.size();
            LOG_TRACE("[DeviceGraphOrchestrator::getSnapshot] Key found: " << key << " size=" << out_size);
            return it->second.data.data();
        }

        /**
         * @brief Retrieve a captured snapshot with 2D shape metadata
         *
         * Returns the snapshot data along with the rows/cols that the stage
         * reported via getDumpInfo() at capture time.
         */
        SnapshotInfo getSnapshotWithShape(const std::string &key) const override
        {
            auto it = snapshots_.find(key);
            if (it == snapshots_.end())
                return {};
            const auto &snap = it->second;
            return {snap.data.data(), snap.data.size(), snap.rows, snap.cols};
        }

        /**
         * @brief Get list of all captured snapshot keys
         *
         * @return Vector of snapshot identifiers
         */
        std::vector<std::string> getSnapshotKeys() const override
        {
            std::vector<std::string> keys;
            keys.reserve(snapshots_.size());
            for (const auto &p : snapshots_)
            {
                keys.push_back(p.first);
            }
            return keys;
        }

        /**
         * @brief Check if snapshot capture is enabled
         */
        bool isSnapshotCaptureEnabled() const { return snapshot_enabled_; }

        /**
         * @brief Convert graph stage name to pipeline-style snapshot key
         *
         * Graph stages use snake_case (e.g., "layer0_q_proj")
         * Pipeline snapshots use SCREAMING_CASE (e.g., "layer0_Q_PROJECTION")
         *
         * This is public to allow unit testing of the conversion logic.
         *
         * @param stage_name Graph stage name
         * @return Pipeline-compatible snapshot key
         */
        static std::string convertStageNameToSnapshotKey(const std::string &stage_name)
        {
            // Stage name mappings (graph → pipeline)
            static const std::unordered_map<std::string, std::string> suffix_map = {
                {"_attn_norm", "_ATTENTION_NORM"},
                {"_q_norm", "_Q_NORM"}, // Qwen3 per-head QK RMSNorm
                {"_k_norm", "_K_NORM"}, // Qwen3 per-head QK RMSNorm
                {"_q_proj", "_Q_PROJECTION"},
                {"_k_proj", "_K_PROJECTION"},
                {"_v_proj", "_V_PROJECTION"},
                {"_q_rope", "_Q_ROPE"},
                {"_k_rope", "_K_ROPE"},
                {"_attention", "_ATTENTION_CONTEXT"},   // Graph uses "attention" not "attn_compute"
                {"_wo_proj", "_ATTENTION_OUTPUT"},      // Graph uses "wo_proj" not "attn_proj"
                {"_wo_allreduce", "_ATTENTION_OUTPUT"}, // LOCAL TP: allreduce overwrites partial Wo result
                {"_attn_residual", "_ATTENTION_RESIDUAL"},
                {"_ffn_norm", "_FFN_NORM"},
                {"_ffn_gate", "_FFN_GATE"},
                {"_ffn_up", "_FFN_UP"},
                {"_swiglu", "_FFN_SWIGLU"},
                {"_down_proj", "_FFN_DOWN"},      // Graph uses "down_proj" not "ffn_down"
                {"_down_allreduce", "_FFN_DOWN"}, // LOCAL TP: allreduce overwrites partial down_proj result
                {"_ffn_residual", "_FFN_RESIDUAL"},
            };

            // Global stages
            if (stage_name == "embedding")
                return "EMBEDDING";
            if (stage_name == "final_norm")
                return "FINAL_NORM";
            if (stage_name == "lm_head")
                return "LM_HEAD";

            // Layer-specific stages: extract layer prefix and convert suffix
            for (const auto &[suffix, replacement] : suffix_map)
            {
                size_t pos = stage_name.find(suffix);
                if (pos != std::string::npos)
                {
                    // Extract "layerN" prefix
                    std::string prefix = stage_name.substr(0, pos);
                    return prefix + replacement;
                }
            }

            // Fallback: return original name (uppercase)
            std::string result = stage_name;
            for (char &c : result)
            {
                if (c >= 'a' && c <= 'z')
                    c = c - 'a' + 'A';
                if (c == '_')
                    c = '_'; // Keep underscores
            }
            return result;
        }

        // =========================================================================
        // Model Metadata Accessors (Convenience)
        // =========================================================================

        /**
         * @brief Get model hidden dimension
         */
        int d_model() const { return graph_builder_ ? graph_builder_->config().d_model : 0; }

        /**
         * @brief Get number of transformer layers
         */
        int n_layers() const { return graph_builder_ ? graph_builder_->config().n_layers : 0; }

        /**
         * @brief Get maximum sequence length (from config)
         */
        int max_seq_len() const { return graph_builder_ ? graph_builder_->config().max_seq_len : 0; }

        /**
         * @brief Get number of attention heads
         */
        int n_heads() const { return graph_builder_ ? graph_builder_->config().n_heads : 0; }

        /**
         * @brief Get number of KV heads (GQA)
         */
        int n_kv_heads() const { return graph_builder_ ? graph_builder_->config().n_kv_heads : 0; }

        /**
         * @brief Get cache statistics
         */
        struct CacheStats
        {
            size_t attention_cache_hits = 0;
            size_t attention_cache_misses = 0;
            size_t ffn_cache_hits = 0;
            size_t ffn_cache_misses = 0;
            size_t cached_layers = 0;
        };

        CacheStats getCacheStats() const { return cache_stats_; }

    private:
        // =========================================================================
        // Private Helpers
        // =========================================================================

        /**
         * @brief Update dynamic parameters in a cached graph
         *
         * Updates position offset and sequence length in all stages
         * that have dynamic parameters.
         *
         * @param graph Graph to update
         * @param pos_offset New position offset
         * @param seq_len New sequence length
         */
        void updateCachedGraphParams(ComputeGraph &graph, int pos_offset, int seq_len);

        /**
         * @brief Synchronize the GPU stream and mark logits as synced at forward
         *        pass boundary.
         *
         * This ensures the caller receives logits without any per-access
         * coherence or device synchronization.  The stream sync happens once
         * here; subsequent data()/fp32_data() calls on the logits tensor
         * return the mapped pointer immediately.
         *
         * @param ctx Device context whose stream to synchronize
         */
        void syncLogitsAtBoundary(IDeviceContext *ctx);

        /**
         * @brief Build decode-time capture policy from runtime and graph context
         */
        DeviceGraphExecutor::DecodeCapturePolicy buildDecodeCapturePolicy(
            bool has_collective_nodes,
            IDeviceContext *ctx,
            int segment_consecutive_failures) const;

        /**
         * @brief Check whether collective segmented replay is backend-supported
         */
        bool collectivesSupportSegmentedReplay() const;

        /**
         * @brief Check if we can use cached graph for current execution
         *
         * @param layer_idx Layer index
         * @param seq_len Current sequence length
         * @return true if cached graph can be reused
         */
        bool canUseCachedGraph(int layer_idx, int seq_len) const;

        // =========================================================================
        // Members
        // =========================================================================

        /// Graph builder (declarative layer)
        std::shared_ptr<Qwen2Graph> graph_builder_;

        /// Graph executor
        DeviceGraphExecutor executor_;

        /// MPI context for distributed execution
        std::shared_ptr<MPIContext> mpi_ctx_;

        /// Graph caching configuration
        GraphCacheConfig cache_config_;

        /// Per-layer graph cache
        std::vector<LayerGraphCache> layer_graph_cache_;

        /// Device context cache (lazy initialization)
        std::unordered_map<DeviceId, std::unique_ptr<IDeviceContext>> device_contexts_;

        /// Cache statistics
        mutable CacheStats cache_stats_;

        /// Last position offset (for cache validation)
        int last_pos_offset_ = -1;

        /// When true, GPU stage timeline output is suppressed (warmup runs)
        bool suppress_timeline_ = false;

        /// When true, prefill timelines are accumulated instead of printed immediately (benchmark mode)
        bool accumulate_prefill_ = false;

        /// Inference state (Phase 5 - owned buffers)
        InferenceState state_;

        // =========================================================================
        // Full Forward Graph Cache (Decode Optimization)
        // =========================================================================

        /**
         * @brief Signature for caching full forward graphs.
         *
         * Phase 1 goal: avoid rebuilding stage/kernel objects on repeated forwards
         * with the same execution shape/path.
         */
        struct ForwardGraphSignature
        {
            int seq_len = 0;
            int batch_size = 0;
            DeviceId device = DeviceId::cpu();
            bool decode = false;
            bool standard_path = true;
            bool pp_stage_enabled = false;
            int pp_first_layer = -1;
            int pp_last_layer = -1;
            bool pp_has_embedding = false;
            bool pp_has_lm_head = false;

            bool operator==(const ForwardGraphSignature &other) const
            {
                return seq_len == other.seq_len &&
                       batch_size == other.batch_size &&
                       device == other.device &&
                       decode == other.decode &&
                       standard_path == other.standard_path &&
                       pp_stage_enabled == other.pp_stage_enabled &&
                       pp_first_layer == other.pp_first_layer &&
                       pp_last_layer == other.pp_last_layer &&
                       pp_has_embedding == other.pp_has_embedding &&
                       pp_has_lm_head == other.pp_has_lm_head;
            }
        };

        struct ForwardGraphSignatureHash
        {
            size_t operator()(const ForwardGraphSignature &sig) const
            {
                size_t h = std::hash<int>{}(sig.seq_len);
                h ^= (std::hash<int>{}(sig.batch_size) + 0x9e3779b9 + (h << 6) + (h >> 2));
                h ^= (std::hash<DeviceId>{}(sig.device) + 0x9e3779b9 + (h << 6) + (h >> 2));
                h ^= (std::hash<bool>{}(sig.decode) + 0x9e3779b9 + (h << 6) + (h >> 2));
                h ^= (std::hash<bool>{}(sig.standard_path) + 0x9e3779b9 + (h << 6) + (h >> 2));
                h ^= (std::hash<bool>{}(sig.pp_stage_enabled) + 0x9e3779b9 + (h << 6) + (h >> 2));
                h ^= (std::hash<int>{}(sig.pp_first_layer) + 0x9e3779b9 + (h << 6) + (h >> 2));
                h ^= (std::hash<int>{}(sig.pp_last_layer) + 0x9e3779b9 + (h << 6) + (h >> 2));
                h ^= (std::hash<bool>{}(sig.pp_has_embedding) + 0x9e3779b9 + (h << 6) + (h >> 2));
                h ^= (std::hash<bool>{}(sig.pp_has_lm_head) + 0x9e3779b9 + (h << 6) + (h >> 2));
                return h;
            }
        };

        /**
         * @brief Cached full forward graph for decode mode
         *
         * During decode (seq_len=1), the graph structure is identical between
         * steps — only token_ids, position_ids, and position_offset change.
         * Instead of rebuilding hundreds of stage objects every forward() call,
         * we cache the graph and its stages after the first decode step.
         *
         * Stable buffers (token_ids, position_ids) are owned here so that
         * cached stages' pointers remain valid across calls.
         */
        struct ForwardGraphCache
        {
            std::unique_ptr<ComputeGraph> graph; ///< Cached compute graph
            Qwen2ForwardOutput output;           ///< Cached output (logits pointer)
            bool valid = false;                  ///< Whether cache is usable

            // Stable buffers — stages point to these, contents updated each step
            std::vector<int> token_ids;    ///< Persistent decode token storage
            std::vector<int> position_ids; ///< Persistent decode position IDs

            // PP hidden state copy — for non-embedding PP stages, the external
            // hidden state must be copied to the working buffer on every forward.
            // During graph build (cache MISS) this copy happens inline in
            // Qwen2Graph::buildPartialForwardGraph(). On cache HIT we must redo
            // the copy here because the graph build code is not re-executed.
            TensorBase *pp_external_hidden_state = nullptr; ///< Source (stage N-1 output)
            TensorBase *pp_working_buffer = nullptr;        ///< Destination (local residual/hidden)
            size_t pp_copy_bytes = 0;
            DeviceId pp_device;
            bool pp_needs_copy = false;

            // Pre-computed collective stage names for fast decode intercept
            std::unordered_set<std::string> collective_nodes;

            // Pre-cached pointers to stages that override updateDynamicParams().
            // Only ~4 stages (RoPE, Attention, FusedAttention, KVCacheAppend) need
            // updating — avoids iterating all ~339 stages with hash lookups each step.
            std::vector<IComputeStage *> dynamic_param_stages;
            bool dynamic_param_stages_cached = false;

            // Tracks whether setGPUStream has been applied to all stages.
            // The capture_stream never changes once set, so we skip the
            // 339-stage loop on subsequent decode steps.
            bool gpu_stream_applied = false;

            // Tracks whether Phase 3 graph replay is active (no markCompleted calls),
            // allowing us to skip the 339-node graph.reset() since flags are already clear.
            bool phase3_active = false;

            /// GPU graph capture/replay for eliminating per-kernel launch overhead
            std::unique_ptr<IGPUGraphCapture> gpu_graph;

            /// Segmented GPU graph cache — excludes non-capturable stages (attention, KV cache)
            /// and captures contiguous runs of capturable stages into separate graphs
            DeviceGraphExecutor::GraphSegmentCache segment_cache;

            /// GPU stream (from IWorkerGPUContext::defaultStream()) for kernel dispatch
            /// Set when gpu_graph is created; used by stages to dispatch on correct stream
            void *gpu_stream = nullptr;

            /// GPU context for creating new graph captures (not owned)
            IWorkerGPUContext *gpu_ctx = nullptr;

            /// Number of consecutive graph update failures (fallback heuristic)
            int gpu_graph_update_failures = 0;

            /// Maximum consecutive update failures before disabling graph capture
            static constexpr int kMaxGraphUpdateFailures = 4;

            void invalidate()
            {
                if (gpu_graph)
                {
                    gpu_graph->reset();
                    gpu_graph.reset();
                }
                segment_cache.reset();
                gpu_stream = nullptr;
                gpu_ctx = nullptr;
                gpu_graph_update_failures = 0;
                graph.reset();
                valid = false;
                token_ids.clear();
                position_ids.clear();
                collective_nodes.clear();
                dynamic_param_stages.clear();
                dynamic_param_stages_cached = false;
                gpu_stream_applied = false;
                phase3_active = false;
                pp_external_hidden_state = nullptr;
                pp_working_buffer = nullptr;
                pp_copy_bytes = 0;
                pp_needs_copy = false;
            }
        };

        std::unordered_map<ForwardGraphSignature, ForwardGraphCache, ForwardGraphSignatureHash> forward_graph_cache_;

        /// Padded sequence length from last forward_batch() call
        int padded_seq_len_ = 0;

        // =========================================================================
        // Snapshot Capture Members
        // =========================================================================

        /// Whether snapshot capture is enabled
        bool snapshot_enabled_ = false;

        /// Internal storage for a captured snapshot with shape metadata
        struct StoredSnapshot
        {
            std::vector<float> data;
            size_t rows = 0;
            size_t cols = 0;
        };

        /// Captured snapshots (key -> FP32 data with shape)
        std::unordered_map<std::string, StoredSnapshot> snapshots_;

        // =========================================================================
        // Graph Buffer Management Members (Phase 3 - moved from Qwen2Graph)
        // =========================================================================

        /// TensorFactory for buffer allocation (not owned, set via setTensorFactory())
        TensorFactory *tensor_factory_ = nullptr;

        /// Owned TensorFactory, created during initializeInferenceState() when
        /// no external factory is provided via setTensorFactory().
        std::unique_ptr<TensorFactory> owned_tensor_factory_;

        /// Standalone workspace allocator
        std::unique_ptr<WorkspaceAllocator> workspace_allocator_;

        /// Unified buffer arena — owns and tracks coherence for all activation buffers
        std::unique_ptr<BufferArena> arena_;

        /// Owned tensors when using graph-managed allocation
        std::vector<std::unique_ptr<TensorBase>> owned_buffers_;

        /// Model-level buffers (when using graph-managed allocation)
        ModelBuffers managed_buffers_;

        /**
         * @brief Allocate GPU workspace for stages in a graph
         *
         * This is called lazily on first graph execution to bind workspace
         * to GEMM kernels, eliminating hot-path allocations on GPU.
         *
         * @param graph The compute graph whose stages need workspace
         * @return true if allocation succeeded (or was already done)
         */
        bool ensureDeviceWorkspaceAllocated(const ComputeGraph &graph);

        /**
         * @brief Populate managed_buffers_ from arena-owned tensors.
         *
         * Maps BufferId → TensorBase* from the arena into the legacy
         * ModelBuffers/ActivationBuffers struct for Qwen2Graph compatibility.
         */
        void bindArenaToManagedBuffers();

        /**
         * @brief Initialize the BufferArena from existing managed buffers (legacy path)
         *
         * For paths where managed_buffers_ is populated externally (e.g., PP stages),
         * this registers all managed_buffers_ activation buffers as external (non-owning)
         * entries in the arena. In the primary flow, initializeBuffers() creates the
         * arena directly and this method is a no-op.
         */
        void initializeArena();

        // =========================================================================
        // Phase-Aware Weight Access Members (Gap 3 - CPU Decode Participation)
        // =========================================================================

        /// Weight manager for full weights and decode shards
        std::shared_ptr<WeightManager> weight_manager_;

        /// Weight placement map for decode device selection
        std::shared_ptr<WeightPlacementMap> weight_placement_map_;

        /// Current inference phase (PREFILL or DECODE)
        InferencePhase current_phase_ = InferencePhase::PREFILL;

        // =========================================================================
        // Tensor Parallel Configuration (Phase 1c: Proportional TP)
        // =========================================================================

        /// Tensor parallelism configuration for proportional head/FFN/vocab assignment
        /// When set, overrides equal 1/world_size splits for heterogeneous GPU setups
        std::shared_ptr<TensorParallelConfig> tp_config_;

        // =========================================================================
        // Multi-Domain Tensor Parallel Configuration (Phase 6.3: Heterogeneous TP)
        // =========================================================================

        /// Multi-domain tensor parallelism configuration for heterogeneous TP
        /// When set, enables separate domains for attention (GPU) and FFN (CPU)
        std::shared_ptr<MultiDomainTPConfig> domain_config_;

        // =========================================================================
        // Weight Streaming Members (Option B)
        // =========================================================================

        /// Weight streamer for on-demand layer transfer (nullptr = disabled)
        std::shared_ptr<IWeightStreamer> weight_streamer_;

        // =========================================================================
        // Injected Dependencies (for testing - Phase 4)
        // =========================================================================

        /// Injected model context interface (nullptr if using concrete types)
        std::shared_ptr<IModelContext> injected_model_ctx_;

        /// Injected topology interface (nullptr if using MPIContext directly)
        std::shared_ptr<IMPITopology> injected_topology_;

        /// Injected collective context (nullptr if using default)
        std::shared_ptr<ICollectiveContext> injected_collective_ctx_;

        /// TurboQuant context for TQ4 KV cache (owns rotation matrix lifetime)
        std::shared_ptr<TurboQuantContext> turboquant_ctx_;

        // =========================================================================
        // Pipeline Parallelism Configuration (Legacy - Single Stage)
        // =========================================================================

        /// PP stage configuration (empty = full model, has value = PP stage)
        /// When set, executeForward() uses buildPartialForwardGraph()
        std::optional<FactoryPPStageConfig> pp_stage_config_;

        /// External hidden state input for PP middle/final stages
        TensorBase *external_hidden_state_input_ = nullptr;

        // =========================================================================
        // Unified Pipeline Configuration (Phase 6 - Full PP+TP)
        // =========================================================================

        /// Unified pipeline configuration for PP+TP composition
        /// When set, orchestrator builds/executes unified graphs spanning all stages
        std::shared_ptr<PipelineConfig> pipeline_config_;

        /// PP contexts for inter-stage activation transfers
        /// Key: {from_stage_id, to_stage_id}
        std::map<std::pair<int, int>, std::unique_ptr<ILocalPPContext>> pp_contexts_;

        /// TP contexts for each domain (one per domain name)
        /// Each domain may have internal tensor parallelism
        /// NOTE: Uses shared_ptr because PPStage can hold a reference to the TP context
        std::map<std::string, std::shared_ptr<ILocalTPContext>> domain_tp_contexts_;

        /// Whether PP contexts have been initialized
        bool pp_contexts_initialized_ = false;

        /// Whether TP contexts have been initialized
        bool tp_contexts_initialized_ = false;

        /// Session epoch counter — incremented on each clear_cache() call
        /// Used to detect stale kernel state across inference sessions
        uint64_t session_epoch_ = 0;

        /// Reset input-dependent dynamic state on all cached kernels
        /// Implemented in .cpp to avoid including KernelFactory.h in the header
        void resetKernelDynamicState();
    };

} // namespace llaminar2
