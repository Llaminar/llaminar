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
 * - Graph Builders (QwenStandardGraph, etc.): Declarative, stateless, build ComputeGraph DAGs
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
 * auto graph_builder = std::make_shared<QwenStandardGraph>(config, mpi_ctx);
 * DeviceGraphOrchestrator orchestrator(graph_builder, mpi_ctx);
 *
 * // Execute a layer
 * orchestrator.executeLayer(weights, buffers, layer_idx, seq_len, kv_cache, pos_ids, device_idx);
 * @endcode
 */

#pragma once

#include "../graph/IGraphBuilder.h"
#include "../../../backends/DeviceId.h"
#include "../../../backends/IGPUGraphCapture.h"
#include "../../../backends/GPUDeviceContextPool.h"
#include "IInferenceRunner.h"
#include "../graph/DeviceGraphExecutor.h"
#include "../device/DeviceContext.h"
#include "../device/WorkspaceAllocator.h"
#include "../../mpi_orchestration/PlacementStrategy.h" // For InferencePhase
#include "../../compute_stages/ComputeStages.h"        // For StageDumpInfo
#include "../../moe/ExpertWeightTransfer.h"            // For ReceivedWeightsMap, ExpertMigration
#include "../../moe/MoEExpertOverlayProfiler.h"        // For overlay profiling summary flush
#include "../../factory/InferenceRunnerFactory.h"      // For FactoryPPStageConfig
#include "../../../snapshots/SnapshotCapture.h"        // Snapshot capture (extracted Phase 2)
#include "../engine/ForwardExecutionEngine.h"          // Forward execution engine (extracted Phase 3)
#include "../../../loaders/IWeightStreamer.h"          // For weight streaming (Option B)
#include "../../../interfaces/IModelContext.h"         // For interface-based construction
#include "../../../memory/BufferArena.h"               // Phase 2: unified buffer management
#include "../../prefix_cache/PrefixCacheFingerprint.h"
#include "../../prefix_cache/PrefixCacheStats.h"
#include "../../mtp/MTPSpecDecodeMetadata.h"
#include "../../../interfaces/IMPITopology.h"          // For interface-based construction
#include "../../../interfaces/ICollectiveContext.h"    // For interface-based construction
#include "../../../config/TPDomain.h"                  // For MultiDomainTPConfig (Phase 6.3)
#include "../../../config/PipelineConfig.h"            // For unified PP+TP configuration (Phase 6)
#include "../../../collective/ILocalPPContext.h"       // For unique_ptr<ILocalPPContext> in maps
#include "../../../collective/ITPContext.h"            // For polymorphic TP context ownership
#include "../../../collective/ILocalTPContext.h"       // For unique_ptr<ILocalTPContext> in maps
#include "../../../collective/IGlobalTPContext.h"      // For shared_ptr<IGlobalTPContext> ownership
#include <memory>
#include <optional>
#include <array>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <vector>
#include <string>
#include <cstring>
#include <cstdint>

namespace llaminar2
{

    // Forward declarations
    class IMPIContext;
    class IKVCache;
    class IWeightManager;
    class IWeightPlacementMap;
    class WeightManager;
    class WeightPlacementMap;
    class TensorParallelConfig;
    class TurboQuantContext;
    class MoERebalanceController;
    class ExpertWeightPayloadProvider;
    class PreparedWeightStore;
    class FrozenModelWeightSet;
    class PrefixStateCache;
    class RamPrefixStorageBackend;
    class DiskPrefixStorageBackend;
    class DeviceHotPrefixStorageBackend;
    struct PrefixBlockHandle;
    struct ExpertReplicaSet;
    class ActivationRotation;
    enum class KVCacheLayoutMode : uint8_t;

    /**
     * @brief Configuration for graph caching behavior
     */
    // GraphCacheConfig moved to ForwardGraphTypes.h (Phase 3 extraction)
    // Alias retained for backward compatibility with existing code.

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
        int cached_seq_len = 0;                         ///< Legacy/shared sequence length for compatibility.
        int attention_cached_seq_len = 0;               ///< Sequence length for cached attention graph.
        int ffn_cached_seq_len = 0;                     ///< Sequence length for cached FFN graph.
        bool attention_cached_all_position_logits = false;
        bool ffn_cached_all_position_logits = false;
        bool valid = false;                             ///< Whether cache entries are valid

        void invalidate()
        {
            attention_decode.reset();
            ffn_decode.reset();
            cached_seq_len = 0;
            attention_cached_seq_len = 0;
            ffn_cached_seq_len = 0;
            attention_cached_all_position_logits = false;
            ffn_cached_all_position_logits = false;
            valid = false;
        }

        /**
         * @brief Reset request-scoped state while preserving cached graph topology.
         *
         * The decode graph cache is intended to survive prompt boundaries. Stages
         * inside the cached graphs can still own request-derived metadata (for
         * example stashed routing snapshots, dynamic stream pointers, or replay
         * callbacks), so clear_cache() must reset that state explicitly instead
         * of marking every cached graph invalid and rebuilding the topology.
         */
        void resetSessionState()
        {
            auto reset_graph = [](std::unique_ptr<ComputeGraph> &graph)
            {
                if (!graph)
                    return;

                graph->reset();
                for (const auto &node_name : graph->getExecutionOrder())
                {
                    ComputeNode *node = graph->getNode(node_name);
                    if (node && node->stage)
                        node->stage->resetSessionState();
                }
            };

            reset_graph(attention_decode);
            reset_graph(ffn_decode);
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
         * @brief Sequence length used for transient graph/activation buffers.
         *
         * This may be smaller than max_seq_len. KV caches, positions, and request
         * context validation still use max_seq_len; graph execution shapes must
         * fit within this activation capacity or go through explicit chunked
         * prefill scheduling.
         *
         * Default: 0 (use max_seq_len, preserving historical behavior).
         */
        int activation_seq_len = 0;
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
        std::shared_ptr<TensorBase> all_position_logits; ///< Runtime verifier logits [tokens, vocab_size]
        std::shared_ptr<TensorBase> all_position_logits_local; ///< Runtime verifier logits [tokens, vocab_local]
        std::unordered_map<int, std::shared_ptr<TensorBase>> all_position_logits_by_rows; ///< Stable verifier logits by row count.
        std::unordered_map<int, std::shared_ptr<TensorBase>> all_position_logits_local_by_rows; ///< Stable local verifier logits by row count.

        /// Local logits for column-parallel LM head [batch_size * seq_len, vocab_local]
        /// Only allocated when lm_head_column_parallel is enabled
        std::shared_ptr<TensorBase> logits_local;

        // === KV Cache ===
        std::unique_ptr<IKVCache> kv_cache; ///< Attention KV history (single-device mode)

        /// Request-local MTP sidecar KV caches, one cache per MTP depth.
        /// These are shifted relative to the main cache and are populated only
        /// when MTP support is explicitly enabled.
        std::vector<std::unique_ptr<IKVCache>> mtp_kv_caches;

        /// Terminal state restored from a prefix hit. MTP full hits require the
        /// hidden row as well as logits; dense prefix-cache reuse can use logits
        /// alone to preserve first-token semantics.
        std::shared_ptr<TensorBase> prefix_terminal_hidden;
        std::shared_ptr<TensorBase> prefix_terminal_logits;

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

        // === Dynamic Extension Buffers ===
        /// Model-specific buffers keyed by BufferId (GDN, MoE, etc.)
        /// Auto-populated from BufferArena for non-core BufferIds.
        std::unordered_map<BufferId, std::shared_ptr<TensorBase>> extension_buffers;

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
        int activation_seq_len = 0;
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
            for (auto &cache : mtp_kv_caches)
            {
                if (cache)
                    cache->clear();
            }
            for (auto &[device, cache] : pp_kv_caches)
            {
                if (cache)
                    cache->clear();
            }
            std::fill(positions.begin(), positions.end(), 0);
            std::fill(sequence_lengths.begin(), sequence_lengths.end(), 0);
        }

        /**
         * @brief Build ModelBuffers from this state (mechanical mapping)
         *
         * Populates a ModelBuffers struct with raw pointers from owned shared_ptrs.
         * This replaces the ~30 lines of manual field-by-field assignment that
         * previously lived in forward().
         */
        ModelBuffers toModelBuffers() const
        {
            ModelBuffers mb;
            mb.current_hidden = hidden.get();
            mb.logits = logits.get();
            mb.logits_local = logits_local.get();

            auto &lb = mb.layer_buffers;
            lb.current_hidden = hidden.get();
            lb.normalized = normalized.get();
            lb.residual = residual.get();
            lb.Q = Q.get();
            lb.K = K.get();
            lb.V = V.get();
            lb.attn_output = attn_output.get();
            lb.attn_proj = attn_proj.get();
            lb.gate = gate.get();
            lb.up = up.get();
            lb.ffn_output = ffn_output.get();
            lb.workspace_scores = workspace_scores.get();
            lb.workspace_context = workspace_context.get();
            lb.workspace_mask = workspace_mask.get();

            lb.Q_rope = Q_rope.get();
            lb.K_rope = K_rope.get();
            lb.V_dequant = V_dequant.get();

            if (!K_head_scales.empty())
            {
                lb.K_head_scales = const_cast<float *>(K_head_scales.data());
                lb.K_head_scales_capacity = K_head_scales.size();
            }

            // Dynamic extensions (model-specific buffers flow through automatically)
            for (const auto &[id, tensor] : extension_buffers)
            {
                if (tensor)
                    lb.extensions[id] = tensor.get();
            }

#ifdef ENABLE_PIPELINE_SNAPSHOTS
            lb.context_snapshot = context_snapshot.get();
            lb.attention_output_snapshot = attention_output_snapshot.get();
            lb.attention_residual_snapshot = attention_residual_snapshot.get();
#endif

            return mb;
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
     * Currently supports QwenStandardGraph, designed for extension to other architectures.
     *
     * Implements IInferenceRunner for unified inference API.
     */
    class DeviceGraphOrchestrator : public IInferenceRunner, public IForwardExecutionHost
    {
    public:
        // =========================================================================
        // Dependencies Struct for Interface-Based Construction (Testing Support)
        // =========================================================================

        /**
         * @brief Dependency injection container for construction
         *
         * Consolidates all configuration-time dependencies into a single struct.
         * Required fields must be set before construction; optional fields have
         * sensible defaults (nullptr / empty).
         *
         * Usage:
         * @code
         * DeviceGraphOrchestrator::Dependencies deps;
         * deps.model_ctx = model_ctx;
         * deps.graph_builder = std::make_shared<QwenStandardGraph>(config, nullptr);
         * deps.turboquant_ctx = turboquant_ctx;              // optional
         * deps.pp_stage_config = pp_config;                  // optional
         * auto orchestrator = DeviceGraphOrchestrator(std::move(deps));
         * @endcode
         */
        struct Dependencies
        {
            // ---- Required ----

            /// Model context providing weights and metadata (required)
            std::shared_ptr<IModelContext> model_ctx;

            /// Graph builder for constructing compute graphs (required)
            std::shared_ptr<IGraphBuilder> graph_builder;

            // ---- Optional: distributed execution ----

            /// MPI topology for work distribution (nullptr for single-rank)
            std::shared_ptr<IMPITopology> topology = nullptr;

            /// Collective context for GPU-native collectives (nullptr for single-rank)
            std::shared_ptr<ICollectiveContext> collective_ctx = nullptr;

            // ---- Optional: pipeline parallelism ----

            /// PP stage bounds (empty = full model)
            std::optional<FactoryPPStageConfig> pp_stage_config;

            /// Unified pipeline config for multi-stage PP+TP
            std::shared_ptr<PipelineConfig> pipeline_config = nullptr;

            // ---- Optional: additional config ----

            /// TurboQuant context for TQ KV cache (owns rotation matrix lifetime)
            std::shared_ptr<TurboQuantContext> turboquant_ctx = nullptr;

            /// KV rotation for Q16_1 kurtosis reduction
            std::shared_ptr<ActivationRotation> kv_rotation = nullptr;

            /// Weight streamer for on-demand layer transfer (nullptr = disabled)
            std::shared_ptr<IWeightStreamer> weight_streamer = nullptr;

            /// Weight manager for full weights and decode shards
            std::shared_ptr<IWeightManager> weight_manager = nullptr;

            /// Weight placement map for decode device selection
            std::shared_ptr<IWeightPlacementMap> weight_placement_map = nullptr;

            /// Tensor parallelism configuration (proportional TP splits)
            std::shared_ptr<TensorParallelConfig> tp_config = nullptr;

            /// Multi-domain TP configuration (heterogeneous TP domains)
            std::shared_ptr<MultiDomainTPConfig> domain_config = nullptr;

            /// Domain-scoped TP contexts whose raw pointers are installed
            /// into GraphConfig / graph builders before graph construction.
            std::map<std::string, std::shared_ptr<ITPContext>> domain_tp_contexts;

            // ---- Optional: graph caching ----

            /// Graph caching configuration
            GraphCacheConfig cache_config;
        };

        // =========================================================================
        // Constructors
        // =========================================================================

        /**
         * @brief Construct orchestrator with injected dependencies (preferred)
         *
         * Accepts all configuration-time dependencies in a single struct.
         * Required fields: model_ctx, graph_builder.
         * Optional: topology, collective_ctx, pp_stage_config, pipeline_config,
         *           turboquant_ctx, weight_streamer, weight_manager, etc.
         *
         * @param deps Dependency injection container
         */
        DeviceGraphOrchestrator(Dependencies deps);

        /**
         * @brief Construct orchestrator with graph builder
         *
         * @param graph_builder Shared pointer to IGraphBuilder (graph definition)
         * @param mpi_ctx MPI context for distributed execution
         * @param cache_config Graph caching configuration
         */
        DeviceGraphOrchestrator(
            std::shared_ptr<IGraphBuilder> graph_builder,
            std::shared_ptr<IMPIContext> mpi_ctx = nullptr,
            const GraphCacheConfig &cache_config = {});

        ~DeviceGraphOrchestrator();

        // Non-copyable, movable
        DeviceGraphOrchestrator(const DeviceGraphOrchestrator &) = delete;
        DeviceGraphOrchestrator &operator=(const DeviceGraphOrchestrator &) = delete;
        DeviceGraphOrchestrator(DeviceGraphOrchestrator &&) noexcept;
        DeviceGraphOrchestrator &operator=(DeviceGraphOrchestrator &&) noexcept;

        // =========================================================================
        // Execution Methods (moved from QwenStandardGraph)
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
            const ForwardInput &input,
            ForwardOutput &output);

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
         * @brief Destructively invalidate execution caches and device contexts.
         *
         * This is a topology/workspace lifetime reset, not a request boundary.
         * Normal prompt/session resets must use IInferenceRunner::clear_cache(),
         * which preserves reusable graph topology and prepared runtime resources.
         */
        void invalidateExecutionCaches();

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
         * @brief Set pre-materialized frozen model weight bindings.
         *
         * Used by planned weight setup paths where WeightManager has already
         * resolved all graph-bound tensors into a FrozenModelWeightSet.
         */
        void setFrozenWeightSet(std::unique_ptr<FrozenModelWeightSet> weight_set);

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
        // Collective Context (NCCL/RCCL/HOST)
        // =========================================================================

        /**
         * @brief Set collective context for GPU-native collective operations
         *
         * When set, AllreduceStage and AllGatherStage execution will be intercepted
         * by DeviceGraphExecutor and routed through the BackendRouter for device-native
         * collectives (NCCL for CUDA, RCCL for ROCm, HOST for cross-vendor).
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

        /**
         * @brief Set KV rotation for Q16_1 kurtosis reduction
         *
         * Block-diagonal orthogonal rotation applied to K/V before Q16_1
         * quantization and to Q/output during attention. Ownership is shared.
         *
         * @param rot Shared pointer to ActivationRotation
         */
        void setKVRotation(std::shared_ptr<ActivationRotation> rot)
        {
            kv_rotation_ = std::move(rot);
        }

        // =========================================================================
        // MoE Expert Rebalance Controller
        // =========================================================================

        /// Set MoE rebalance controller (ownership transfer)
        void setMoERebalanceController(std::unique_ptr<MoERebalanceController> controller);
        void addMoERebalanceController(std::unique_ptr<MoERebalanceController> controller);

        /// Initialize expert weight payload provider for metadata-based host retention.
        /// Creates the provider, wires it to all cached MoE stages and the WeightManager.
        /// Call after graph cache is populated and MoE stages are cached.
        void initializeExpertPayloadProvider();

        // =========================================================================
        // Prepared Weight Store (Phase 4-5)
        // =========================================================================

        /// Initialize the model-context-owned prepared weight store.
        /// Creates the store and populates it from already-prepared GEMM weights.
        /// Call after weight finalization (finalizeForDevice) completes.
        void initializePreparedWeightStore(DeviceId device);

        /// Get prepared weight store (may be null if no GEMM weights prepared)
        PreparedWeightStore *preparedWeightStore() const { return prepared_weight_store_.get(); }

        /// Get frozen model weight set (may be null before setWeights is called)
        const FrozenModelWeightSet *frozenWeightSet() const { return frozen_weight_set_.get(); }

        /// Get expert weight payload provider (may be null for non-MoE models)
        ExpertWeightPayloadProvider *expertPayloadProvider() const { return expert_payload_provider_.get(); }

        /// Get MoE rebalance controller (for post-decode logging)
        MoERebalanceController *moeRebalanceController() const
        {
            return moe_rebalance_controller_ ? moe_rebalance_controller_.get() : nullptr;
        }
        std::vector<MoERebalanceController *> moeRebalanceControllers() const override;
        MoERebalanceController *moeRebalanceControllerForDomain(
            const std::string &domain_id) const override;
        int moeRebalanceParticipantId() const override;

        /// Apply expert masks to all MoEExpertComputeStages in cached FFN graphs.
        /// Called after rebalancing to update which experts each rank computes.
        /// @param masks Per-layer expert masks (masks[layer][expert] == true means active)
        /// @param received_weights Optional transferred packed weights from MPI transfer
        void applyExpertMasks(
            const std::vector<std::vector<bool>> &masks,
            const ReceivedWeightsMap &received_weights = {});
        void applyExpertMasksForDomain(
            const std::string &domain_id,
            const std::vector<std::vector<bool>> &masks,
            const ReceivedWeightsMap &received_weights = {});

        /// Non-destructively collect packed expert weights for experts requested
        /// by masks. Used for intra-rank migration between composed TP domains
        /// without falling back to raw GGUF host tensors.
        ReceivedWeightsMap collectExpertWeightsForMasks(
            const std::vector<std::vector<bool>> &masks) const;

        /// Set expert replica info on all MoE stages for per-token dispatch.
        /// Call after applyExpertMasks() so GEMM engines are already prepared.
        void setExpertReplicaSetForParticipant(const ExpertReplicaSet &replicas, int participant_id);

        /// Compatibility wrapper for older socket-oriented call sites.
        void setExpertReplicaSet(const ExpertReplicaSet &replicas, int socket_id);

        /// Release raw expert weight data from all MoE stages after initial VNNI packing.
        /// For mmap: confirms DONTNEED already applied. For heap: frees raw_data_ vectors.
        /// After this, fallback VNNI repacking from raw data is no longer possible —
        /// only prepacked MPI transfer can provide weights for rebalanced experts.
        /// @return Total bytes freed (or already DONTNEED'd)
        size_t releaseRawExpertWeights();

        /// Build and discard a forward graph for the requested shape without executing it.
        /// This is used to force graph-build-time weight materialization, including
        /// Qwen3.5 MoE expert GEMM preparation, before host weight data is released.
        bool materializeForwardGraphForShape(int seq_len, int batch_size = 1);

        /// Transfer packed weights for migrating experts via MPI.
        /// Returns received weights map: [layer_idx][expert_id] → blobs.
        ReceivedWeightsMap transferExpertWeights(
            const std::vector<ExpertMigration> &manifest,
            int num_layers);

        /// Transfer packed weights for replicated experts via MPI.
        /// Unlike transferExpertWeights(), the sender keeps its weights (non-destructive).
        /// The receiver gets pre-packed weights to avoid VNNI repacking from raw.
        /// @param replicas The active replica set describing which experts to transfer
        /// @param num_layers Number of MoE layers
        /// @return Received weights map for this rank's new replicas
        ReceivedWeightsMap transferReplicaWeights(
            const ExpertReplicaSet &replicas,
            int num_layers);

        /**
         * @brief Set GlobalTPContext for cross-MPI-rank tensor parallelism
         *
         * The orchestrator takes shared ownership to keep the context alive
         * for the entire inference session. The context is passed through
         * GraphConfig to graph builders, enabling TPAllreduceStage usage
         * for global TP (same polymorphic path as local TP via ITPContext).
         *
         * @param ctx Shared pointer to GlobalTPContext
         */
        void setGlobalTPContext(std::shared_ptr<IGlobalTPContext> ctx)
        {
            global_tp_ctx_ = std::move(ctx);
        }

        /// Retain domain-scoped TP contexts and wire them to the graph builder.
        void setDomainTPContexts(std::map<std::string, std::shared_ptr<ITPContext>> contexts);

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
        void setWeightManager(std::shared_ptr<IWeightManager> weight_manager);

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
        std::shared_ptr<IWeightManager> weightManager() const { return weight_manager_; }

        /**
         * @brief Set weight placement map for decode device selection
         *
         * The placement map provides device info for phase-aware weight selection.
         *
         * @param placement_map Shared pointer to WeightPlacementMap
         */
        void setWeightPlacementMap(std::shared_ptr<IWeightPlacementMap> placement_map);

        /**
         * @brief Get weight placement map
         * @return Shared pointer to WeightPlacementMap (may be nullptr)
         */
        std::shared_ptr<IWeightPlacementMap> weightPlacementMap() const { return weight_placement_map_; }

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

        /// Enable or disable this DGO's post-prefill host-resident weight release.
        /// RankOrchestrator disables this for child DGOs because it owns the
        /// synchronization point across sibling TP/PP runners.
        void setHostResidentReleaseEnabled(bool enabled) { release_host_resident_after_forward_ = enabled; }

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

        void setSuppressTimeline(bool suppress) override
        {
            suppress_timeline_ = suppress;
            if (forward_engine_)
                forward_engine_->setSuppressTimeline(suppress);
        }

        void setAccumulatePrefill(bool accumulate) override
        {
            accumulate_prefill_ = accumulate;
            if (forward_engine_)
                forward_engine_->setAccumulatePrefill(accumulate);
        }

        void flushStageTimeline() override
        {
            // Print forward pass wall-clock profiler (always, when profiling is on)
            if (forward_engine_ && forward_engine_->forwardPassProfiler().hasData())
            {
                if (debugEnv().profile.enabled)
                {
                    std::string dev_str = state_.device_id.toString();
                    forward_engine_->forwardPassProfiler().printAndReset(dev_str.c_str());
                }
                else
                {
                    forward_engine_->forwardPassProfiler().reset();
                }
            }

            if (!debugEnv().gpu_stage_timing)
            {
                MoEExpertOverlayProfiler::flush();
                return;
            }

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
            MoEExpertOverlayProfiler::flush();
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
        // Graph Buffer Management (Phase 3 - moved from QwenStandardGraph)
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
        // Inference State Management
        // =========================================================================

        /**
         * @brief Initialize inference state from BufferArena (schema-driven path)
         *
         * Populates InferenceState by pulling shared_ptrs from the arena allocated
         * by initializeBuffers(). This replaces the manual buffer allocation in
         * initializeInferenceState() with a schema-driven approach.
         *
         * If initializeBuffers() has not yet been called, this method will:
         * 1. Create a TensorFactory (if not externally set)
         * 2. Call initializeBuffers(max_seq_len) to create the arena
         * 3. Pull tensors from the arena into InferenceState
         * 4. Initialize KV caches (not arena-managed)
         *
         * @param batch_size Maximum batch size
         * @param max_seq_len Maximum sequence length
         * @param device Target device for KV cache allocation
         * @param init_config Configuration for special allocation modes
         * @return true if all required buffers were found and state initialized
         */
        bool initializeInferenceStateFromArena(
            int batch_size,
            int max_seq_len,
            DeviceId device,
            const InferenceStateInitConfig &init_config = InferenceStateInitConfig{});

        /**
         * @brief Check if inference state is initialized
         */
        bool hasInferenceState() const { return state_.isInitialized(); }

        /**
         * @brief Get inference state (read-only)
         */
        const InferenceState &inferenceState() const { return state_; }

        // =====================================================================
        // IInferenceRunner: Device & Logits Local API overrides
        // =====================================================================

        DeviceId primaryDeviceId() const override { return state_.device_id; }

        bool hasLogitsLocal() const override { return state_.logits_local != nullptr; }

        LogitsLocalInfo getLogitsLocalInfo() const override
        {
            if (!state_.logits_local)
                return {};
            const auto &shape = state_.logits_local->shape();
            auto device_opt = state_.logits_local->current_device();
            // Resolve the explicit worker stream for this device to avoid NULL stream races
            void *stream = nullptr;
            if (device_opt.has_value() && device_opt->is_gpu())
            {
                stream = GPUDeviceContextPool::instance().getContext(*device_opt).defaultStream();
            }
            return LogitsLocalInfo{
                state_.logits_local->gpu_data_ptr(),
                device_opt,
                shape.size() >= 2 ? shape[1] : 0,
                state_.logits_local.get(),
                stream,
                // Expose this runner's arena-owned argmax scratch so the
                // multi-device sampler can drive the multi-block reduction
                // without any hot-path allocation.
                argmax_partial_vals_dev_,
                argmax_partial_idxs_dev_,
                argmax_partial_capacity_};
        }

        bool hasMTPLogitsLocal() const override
        {
            if (!graph_builder_ || !graph_builder_->config().lm_head_column_parallel)
                return false;
            auto it = state_.extension_buffers.find(BufferId::MTP_LOGITS);
            return it != state_.extension_buffers.end() && it->second != nullptr;
        }

        LogitsLocalInfo getMTPLogitsLocalInfo() const override
        {
            if (!hasMTPLogitsLocal())
                return {};

            auto it = state_.extension_buffers.find(BufferId::MTP_LOGITS);
            TensorBase *mtp_logits = it->second.get();
            const auto &shape = mtp_logits->shape();
            auto device_opt = mtp_logits->current_device();
            void *stream = nullptr;
            if (device_opt.has_value() && device_opt->is_gpu())
            {
                stream = GPUDeviceContextPool::instance().getContext(*device_opt).defaultStream();
            }
            return LogitsLocalInfo{
                mtp_logits->gpu_data_ptr(),
                device_opt,
                shape.size() >= 2 ? shape[1] : 0,
                mtp_logits,
                stream,
                argmax_partial_vals_dev_,
                argmax_partial_idxs_dev_,
                argmax_partial_capacity_};
        }

        bool hasAllPositionLogitsLocal() const override
        {
            if (!graph_builder_ || !graph_builder_->config().lm_head_column_parallel)
                return false;
            return state_.all_position_logits_local != nullptr;
        }

        LogitsLocalInfo getAllPositionLogitsLocalInfo() const override
        {
            if (!hasAllPositionLogitsLocal())
                return {};

            const auto &shape = state_.all_position_logits_local->shape();
            auto device_opt = state_.all_position_logits_local->current_device();
            void *stream = nullptr;
            if (device_opt.has_value() && device_opt->is_gpu())
            {
                stream = GPUDeviceContextPool::instance().getContext(*device_opt).defaultStream();
            }
            return LogitsLocalInfo{
                state_.all_position_logits_local->gpu_data_ptr(),
                device_opt,
                shape.size() >= 2 ? shape[1] : 0,
                state_.all_position_logits_local.get(),
                stream,
                argmax_partial_vals_dev_,
                argmax_partial_idxs_dev_,
                argmax_partial_capacity_};
        }

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
        bool forwardWithDeviceTokenIds(
            const int *token_shadow,
            const void *token_ids_device,
            int seq_len) override;
        const void *prepareMTPVerifierInputTokensOnDevice(
            int32_t first_token,
            int first_draft_slot,
            int draft_token_count,
            int total_verifier_input_tokens) override;

        /**
         * @brief Get logits from last forward pass
         *
         * @return Pointer to logits buffer, or nullptr if not available
         */
        const float *logits() const override;

        bool forwardMTP(int32_t draft_condition_token) override;
        bool forwardMTPForDeviceSampling(int32_t draft_condition_token) override;
        bool supportsChainedMTPDrafts() const override { return true; }
        bool supportsMTPSidecarSampleFusion() const override;
        bool supportsMTPSidecarLogitsStreamHandoff() const override;
        bool supportsMTPDeviceDraftTokenInput() const override;
        bool supportsMTPSidecarPreservesMainState() const override;
        bool forwardMTPFromLastDraft(int32_t draft_condition_token, int position_id) override;
        bool forwardMTPFromLastDraftForDeviceSampling(
            int32_t draft_condition_token,
            int position_id) override;
        bool forwardMTPFromDeviceDraftForDeviceSampling(
            int draft_sample_slot,
            int position_id) override;
        bool forwardMTPFromDeviceTargetForDeviceSampling(
            int target_sample_slot,
            int position_id) override;
        bool forwardMTPAndSampleGreedy(int32_t draft_condition_token, int32_t *out_token) override;
        bool forwardMTPFromLastDraftAndSampleGreedy(
            int32_t draft_condition_token,
            int position_id,
            int32_t *out_token) override;
        bool flushPendingMTPWork() override;
        void setMTPAllPositionVerifierSyncDeferralEnabled(bool enabled) override;
        bool supportsMTPSpecStatePublication() const override;
        bool publishAcceptedMTPSpecState(
            const MTPSpecStepPlan &plan,
            std::string *error = nullptr) override;
        bool commitMTPShiftedRowsFromLastForward(
            const int32_t *tokens,
            int token_count,
            int already_appended_tokens) override;
        bool commitMTPShiftedRowsFromPartialForward(
            const int32_t *tokens,
            int token_count,
            int already_appended_tokens,
            int main_forward_token_count,
            bool allow_speculative_discard = false,
            int position_offset_override = -1) override;
        bool commitMTPShiftedRowFromCurrentTerminalHidden(
            int32_t token,
            int already_appended_tokens,
            bool allow_speculative_discard = false,
            int position_offset_override = -1) override;
        uint64_t forwardReplayLiveStateEpoch() const
        {
            return live_replay_state_epoch_;
        }
        std::vector<ForwardExecutionEngine::ReplayCacheObservation>
            forwardReplayCacheObservations() const;
        const float *mtpLogits() const override;
        bool setComputeAllPositionLogits(bool enabled) override;
        bool setComputeRowIndexedAllPositionLogits(bool enabled, int row_count) override;
        bool setMTPSpecVerifierInputPlan(
            const MTPSpecDecodeVerifierInputPlan &plan) override;
        void clearMTPSpecVerifierInputPlan() override;
        const float *getAllPositionLogits() const override;
        std::string mtpDecodeUnsupportedReason() const override;
        bool supportsMTPTokenCoordination() const override;
        int sampleGreedyFromMTPLogitsOnDevice() override;
        int sampleGreedyFromAllPositionLogitsOnDevice(int row) override;
        bool sampleGreedyFromAllPositionLogitsOnDeviceRows(
            int start_row,
            int row_count,
            int32_t *out_tokens) override;

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
         * @brief Result of a graph build operation
         *
         * Type alias to the standalone GraphBuildResult (extracted to ForwardGraphTypes.h).
         * Kept as a nested type alias for backward compatibility with existing code.
         */
        using GraphBuildResult = llaminar2::GraphBuildResult;

        /**
         * @brief Fluent builder for compute graph composition (nested class)
         */
        class GraphBuildSession
        {
        public:
            explicit GraphBuildSession(DeviceGraphOrchestrator &orchestrator)
                : orchestrator_(orchestrator) {}

            // Input configuration
            GraphBuildSession &forInput(const ForwardInput &input);
            GraphBuildSession &withPositionIds(const int *position_ids);
            GraphBuildSession &withExternalHiddenState(TensorBase *hidden_state);

            // Pipeline configuration
            GraphBuildSession &withPipelineConfig(std::shared_ptr<PipelineConfig> config);
            GraphBuildSession &forPPStage(int first_layer, int last_layer,
                                          bool has_embedding = false, bool has_lm_head = false);
            GraphBuildSession &withPPContext(int from_stage, int to_stage, ILocalPPContext *context);
            GraphBuildSession &withTPContext(const std::string &domain_name, ITPContext *context);

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
            std::optional<ForwardInput> input_;
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
            std::map<std::string, ITPContext *> tp_contexts_;
            std::optional<ModelWeights> weights_;
            std::optional<ModelBuffers> buffers_;
            IKVCache *kv_cache_ = nullptr;

            ForwardInput prepareInput() const;
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
        IGraphBuilder *graphBuilder() { return graph_builder_.get(); }
        const IGraphBuilder *graphBuilder() const { return graph_builder_.get(); }

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
        IDeviceContext *getDeviceContext(DeviceId device) override;

        /** Check if MoE dynamic rebalancing is active (blocks prefill graph capture). */
        bool isMoeRebalancingActive() const override;

        /** Return the active MoE placement epoch for graph-cache keying. */
        uint64_t moePlacementEpoch() const override;
        std::string prefillGraphDomainId() const override;
        int prefillGraphParticipantId() const override;

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

        bool supportsPrefillChunkSchedule(int seq_len) const override;

        bool forwardPrefillChunkSchedule(
            const int *tokens,
            int seq_len,
            const PrefillChunkSchedulerPolicy &policy,
            int pad_token_id,
            bool allow_padded_execution) override;

        /**
         * @brief GPU-side greedy argmax for single-device inference
         *
         * Uses IBackend::argmaxF32() on the logits tensor's GPU buffer,
         * transferring only 8 bytes (float value + int index) instead of
         * the full vocab row (~600 KB for 152K vocab).
         *
         * @return Token ID (>= 0) on success, -1 if not GPU or backend unavailable
         */
        int sampleGreedyOnDevice() override;
        int sampleOnDevice(const SamplingParams &params) override;

        /**
         * @brief Apply sparse logit penalties on device
         */
        bool applyPenaltiesOnDevice(const std::vector<LogitPenalty> &penalties,
                                    int vocab_size) override;
        bool applyPenaltiesToMTPLogitsOnDevice(const std::vector<LogitPenalty> &penalties,
                                               int vocab_size) override;
        bool applyPenaltiesToAllPositionLogitsOnDeviceRow(
            int row,
            const std::vector<LogitPenalty> &penalties,
            int vocab_size) override;
        bool supportsDeviceStochasticMTPVerification() const override;
        bool buildStochasticDistributionOnDevice(
            DeviceLogitsSource source,
            int row,
            DeviceDistributionBuffer buffer,
            int slot,
            const SamplingParams &params,
            int vocab_size) override;
        int sampleStochasticDistributionOnDevice(
            DeviceDistributionBuffer buffer,
            int slot,
            float threshold) override;
        bool sampleStochasticDistributionOnDeviceDeferred(
            DeviceDistributionBuffer buffer,
            int slot,
            float threshold) override;
        const void *prepareMTPVerifierInputTokensOnDeviceFromDeviceFirstToken(
            int first_target_sample_slot,
            int first_draft_slot,
            int draft_token_count,
            int total_verifier_input_tokens) override;
        bool verifyStochasticDistributionsOnDevice(
            int target_slot,
            int draft_slot,
            int draft_token,
            float accept_threshold,
            float residual_threshold,
            DeviceSpeculativeVerifyResult *out) override;
        bool verifyStochasticDistributionsBatchOnDevice(
            int first_target_slot,
            int first_draft_slot,
            const int32_t *draft_tokens,
            const float *accept_thresholds,
            const float *residual_thresholds,
            int row_count,
            DeviceSpeculativeVerifyResult *out) override;
        bool verifyStochasticDistributionsBatchOutcomeOnDevice(
            int first_target_slot,
            int first_draft_slot,
            const int32_t *draft_tokens,
            const float *accept_thresholds,
            const float *residual_thresholds,
            int row_count,
            int32_t first_token,
            const int32_t *stop_tokens,
            int stop_token_count,
            int bonus_target_slot,
            float bonus_threshold,
            DeviceSpeculativeVerifyBatchOutcome *out) override;
        bool verifyStochasticDistributionsBatchOutcomeOnDeviceFirstToken(
            int first_target_slot,
            int first_draft_slot,
            const int32_t *draft_tokens,
            const float *accept_thresholds,
            const float *residual_thresholds,
            int row_count,
            int first_target_sample_slot,
            const int32_t *stop_tokens,
            int stop_token_count,
            int bonus_target_slot,
            float bonus_threshold,
            DeviceSpeculativeVerifyBatchOutcome *out) override;

        /**
         * @brief Get logits (IInferenceRunner override - already declared above)
         */
        // const float *logits() const; - declared above

        /**
         * @brief Get vocabulary size (IInferenceRunner override)
         */
        int vocab_size() const override { return graph_builder_ ? graph_builder_->config().vocab_size : 0; }

        /**
         * @brief Reset request/session state for a new sequence.
         *
         * Resets inference state (KV cache, positions, model recurrence) while
         * preserving cached ComputeGraphs. Cached stages and GPU replay segments
         * are reset in-place so the next prompt reuses the graph topology without
         * inheriting request-scoped KV/GDN/RoPE/kernel state.
         */
        void clear_cache() override
        {
            for (auto &[dev, ctx] : device_contexts_)
            {
                if (ctx && dev.is_gpu())
                    ctx->synchronize();
            }
            for (auto &entry : layer_graph_cache_)
            {
                entry.resetSessionState();
            }
            if (forward_engine_)
            {
                forward_engine_->resetSessionReplayState();
            }
            mtp_sidecar_depth0_cache_.resetSessionState();
            mtp_sidecar_depth0_chained_cache_.resetSessionState();
            mtp_sidecar_depth0_kv_only_cache_.resetSessionState();
            for (auto &cache : mtp_sidecar_depth0_kv_only_batch_caches_)
                cache.resetSessionState();
            mtp_terminal_hidden_row_select_cache_.resetSessionState();
            last_pos_offset_ = -1;
            pending_mtp_logits_stream_ = nullptr;
            stochastic_target_distribution_streams_.fill(nullptr);
            stochastic_draft_distribution_streams_.fill(nullptr);
            clearStochasticTargetSampleReadySlots();
            clearStochasticDraftSampleReadySlots();
            cache_stats_ = CacheStats{};
            state_.clear();
            // NOTE: Do NOT reset arena_ here. Buffer registrations and allocations
            // are expensive and model-specific (e.g., GDN buffers for Qwen3.5).
            // The arena is created once in initializeBuffers() and persists for
            // the lifetime of the orchestrator.
            // Reset input-dependent cached state on all kernels (e.g., stale token IDs)
            resetKernelDynamicState();
            // Reset model-internal state (no-op for models with hybrid cache since
            // state_.clear() → kv_cache->clear() already handles GDN state reset)
            if (graph_builder_)
                graph_builder_->resetState();
            // Note: host_resident_released_ is NOT reset here —
            // the host data is gone and cannot be re-uploaded.
            device_sampling_counter_ = 0;
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
        const char *architecture() const override
        {
            static thread_local std::string arch_name;
            arch_name = graph_builder_ ? graph_builder_->architectureName() : "unknown";
            return arch_name.c_str();
        }

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

        PrefixLookupResult lookupPrefix(const std::vector<int32_t> &tokens) override;
        bool populatePrefix(const PrefixLookupResult &hit, int seq_idx = 0) override;
        bool harvestPrefix(const std::vector<int32_t> &tokens, int prompt_token_count) override;
        bool restorePrefixTerminalState(const PrefixLookupResult &hit) override;
        PrefixStateSnapshot captureLivePrefixState(int seq_idx = 0) const override;
        PrefixStateSnapshot captureLivePrefixCheckpoint(int seq_idx = 0) const override;
        bool restoreLivePrefixState(const PrefixStateSnapshot &snapshot, int seq_idx = 0) override;
        bool truncateLivePrefixState(int cached_tokens, int seq_idx = 0) override;
        bool requiresMTPDecodeEquivalentVerifierReplay() const override;

        /**
         * @brief Inspect request-local runtime state for prefix-cache/MTP probes.
         */
        PrefixRuntimeStateSnapshot prefixStateProbe() const override;

        // =========================================================================
        // Snapshot Capture API (delegated to SnapshotCapture — Phase 2 extract)
        // =========================================================================

        void enableSnapshotCapture(const std::string &output_dir = "") override
        {
            (void)output_dir;
            snapshot_capture_.clear();
            snapshot_enabled_ = true;

            LOG_DEBUG("[DeviceGraphOrchestrator::enableSnapshotCapture] Setting callback on executor_");
            executor_.setSnapshotCallback(
                [this](const std::string &name, const StageDumpInfo &dump)
                {
                    if (!snapshot_context_.empty())
                        snapshot_capture_.captureStage(snapshot_context_ + "::" + name, dump);
                    snapshot_capture_.captureStage(name, dump);
                });
        }

        void disableSnapshotCapture() override
        {
            snapshot_enabled_ = false;
            snapshot_capture_.clear();
            executor_.setSnapshotCallback(nullptr);
        }

        void clearSnapshots() override
        {
            snapshot_capture_.clear();
        }

        const float *getSnapshot(const std::string &key, size_t &out_size) const override
        {
            const auto *snap = snapshot_capture_.get(key);
            if (!snap)
            {
                LOG_DEBUG("[DeviceGraphOrchestrator::getSnapshot] Key NOT FOUND: " << key
                                                                                   << " (have " << snapshot_capture_.all().size() << " snapshots)");
                out_size = 0;
                return nullptr;
            }
            out_size = snap->data.size();
            LOG_TRACE("[DeviceGraphOrchestrator::getSnapshot] Key found: " << key << " size=" << out_size);
            return snap->data.data();
        }

        SnapshotInfo getSnapshotWithShape(const std::string &key) const override
        {
            const auto *snap = snapshot_capture_.get(key);
            if (!snap)
                return {};
            return {snap->data.data(), snap->data.size(), snap->rows, snap->cols};
        }

        std::vector<std::string> getSnapshotKeys() const override
        {
            return snapshot_capture_.keys();
        }

        bool isSnapshotCaptureEnabled() const { return snapshot_enabled_; }

        /// @brief Convert graph stage name to pipeline-style snapshot key (delegates to SnapshotCapture)
        static std::string convertStageNameToSnapshotKey(const std::string &stage_name)
        {
            return SnapshotCapture::convertStageNameToSnapshotKey(stage_name);
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
         * @brief Shared implementation for host-token and device-token forwards.
         *
         * `tokens` is always the host shadow used for request bookkeeping. When
         * `token_ids_device` is non-null, graph embedding stages read token IDs
         * from that stable device buffer instead of uploading from `tokens`.
         */
        const float *forwardImpl(
            const int *tokens,
            const void *token_ids_device,
            int seq_len,
            int batch_size);

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
         * @brief Configure executor from graph builder config and environment.
         *
         * Shared setup extracted from all constructors to eliminate duplication.
         */
        void configureExecutor();

        /**
         * @brief Validate that the orchestrator has all required configuration
         *        for executing forward passes.
         *
         * Checks that inference state, weights, and graph builder are properly
         * initialized. Logs specific errors for each missing requirement.
         *
         * @return true if configuration is complete and forward() can proceed
         */
        bool validateConfigurationForForward() const;

        /**
         * @brief Initialize KV caches for the current configuration.
         *
         * Creates single KV cache (non-PP) or per-device KV caches (PP mode).
         * Handles sharding for tensor parallelism (LOCAL and GLOBAL TP).
         *
         * @param batch_size Batch size for KV cache allocation
         * @param max_seq_len Maximum sequence length
         * @param n_layers Number of layers (already adjusted for PP stage)
         * @param device Target device
         * @param local_mpi_ctx MPI context (may be single-rank default)
         * @return true if KV caches initialized successfully
         */
        bool initializeKVCaches(int batch_size, int max_seq_len, int n_layers,
                                DeviceId device, const std::shared_ptr<IMPIContext> &local_mpi_ctx);

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
        void syncLogitsAtBoundary(IDeviceContext *ctx) override;

        /**
         * @brief Build decode-time capture policy from runtime and graph context
         */
        DeviceGraphExecutor::DecodeCapturePolicy buildDecodeCapturePolicy(
            bool has_collective_nodes,
            IDeviceContext *ctx,
            int segment_consecutive_failures) const override;

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
        // IForwardExecutionHost Overrides (Phase 3)
        // =========================================================================

        /** Build forward graph via fluent builder API. */
        GraphBuildResult buildForwardGraph(const ForwardInput &input) override;

        /** Get device contexts for all PP pipeline devices. */
        std::unordered_map<DeviceId, IDeviceContext *> getPipelineDeviceContexts() override;

        /** Access the logits tensor for mapped-memory sync checks. */
        TensorBase *logitsTensor() override;

        /** Resolve PP copy info for cache-miss builds. */
        PPCopyInfo resolvePPCopyInfo(const ForwardInput &input) const override;

        /** Whether forward graph construction should emit all-position logits. */
        bool computeAllPositionLogitsEnabled() const override { return compute_all_position_logits_; }

        /** Compact verifier logits row count; 0 means full all-position logits. */
        int allPositionLogitRows() const override
        {
            return compute_row_indexed_all_position_logits_
                       ? row_indexed_all_position_logits_row_count_
                       : 0;
        }

        /** Upload any pending compact verifier row metadata before graph execution. */
        bool prepareAllPositionVerifierGraphMetadata(
            const ForwardInput &input,
            void *execution_stream,
            DeviceId execution_device) override;

        /**
         * @brief Materialize pending verifier token IDs on the graph execution stream.
         *
         * The all-position verifier metadata hook receives the exact stream that
         * the forward engine will use for graph replay/capture.  Copying the
         * compact token row here keeps the device-token embedding input ordered
         * with the verifier graph without relying on the default stream or a
         * host-side synchronization.
         */
        bool materializePendingMTPVerifierInputTokensOnDevice(
            void *execution_stream,
            DeviceId execution_device);

        /**
         * @brief Clear the device-side "sample token is ready" marker for one draft slot.
         *
         * Draft sample slots are reused across decode iterations.  Clearing
         * the marker before a new distribution build prevents later verifier or
         * sidecar stages from accidentally waiting on a previous sample's event.
         */
        void clearStochasticDraftSampleReadySlot(int slot);

        /** @brief Clear every draft sample readiness marker for request reset paths. */
        void clearStochasticDraftSampleReadySlots();

        /**
         * @brief Clear the device-side "sample token is ready" marker for one target slot.
         *
         * Target sample slots feed the first sidecar and the stochastic batch
         * reducer. Clearing stale state keeps the next decode iteration from
         * waiting on a previous sample event or consuming an old token.
         */
        void clearStochasticTargetSampleReadySlot(int slot);

        /** @brief Clear every target sample readiness marker for request reset paths. */
        void clearStochasticTargetSampleReadySlots();

        /**
         * @brief Record that a deferred GPU sampler has written a draft token slot.
         *
         * The deferred stochastic path avoids a host D2H token read, so the host
         * can no longer act as an ordering point.  This helper records a backend
         * event on the producer stream immediately after the sample kernel so
         * later GPU consumers can wait without synchronizing the CPU.
         */
        bool recordStochasticDraftSampleReady(int slot, void *producer_stream);

        /**
         * @brief Record that a deferred GPU sampler has written a target token slot.
         *
         * This mirrors recordStochasticDraftSampleReady(), but target samples
         * live in a separate arena buffer so verifier row outputs can overwrite
         * STOCHASTIC_VERIFY_TOKENS without corrupting the first generated token.
         */
        bool recordStochasticTargetSampleReady(int slot, void *producer_stream);

        /**
         * @brief Make a consumer stream wait for deferred draft sample tokens.
         *
         * @param first_slot First draft-token slot consumed by the next stage.
         * @param slot_count Number of contiguous slots consumed.
         * @param consumer_stream Explicit stream used by the consuming copy/kernel.
         * @param consumer_name Short perf/log tag naming the consumer.
         * @return true if all required waits were queued successfully.
         */
        bool waitForStochasticDraftSampleReadyRange(
            int first_slot,
            int slot_count,
            void *consumer_stream,
            const char *consumer_name);

        /**
         * @brief Make a consumer stream wait for a deferred target sample token.
         *
         * @param slot Target-token sample slot consumed by the next stage.
         * @param consumer_stream Explicit stream used by the consuming copy/kernel.
         * @param consumer_name Short perf/log tag naming the consumer.
         * @return true if the required wait was queued successfully.
         */
        bool waitForStochasticTargetSampleReady(
            int slot,
            void *consumer_stream,
            const char *consumer_name);

        /**
         * @brief Shared implementation for compact device-side stochastic sampling.
         *
         * When `out_token_host` is non-null the helper also performs the trusted
         * fast D2H scalar read used by legacy host-driven callers.  When it is
         * null, the sampled token remains only in the runner-owned device slot
         * for later sidecar/verifier graph stages to consume.
         */
        bool sampleStochasticDistributionOnDeviceImpl(
            DeviceDistributionBuffer buffer,
            int slot,
            float threshold,
            int32_t *out_token_host);

        /**
         * @brief Shared implementation for host-token and device-token batch summaries.
         *
         * `first_token_from_device=false` preserves the legacy host-scalar
         * summary contract.  `true` reads the first sampled token from
         * STOCHASTIC_TARGET_SAMPLE_TOKENS inside the backend summary kernel.
         */
        bool verifyStochasticDistributionsBatchOutcomeOnDeviceCommon(
            int first_target_slot,
            int first_draft_slot,
            const int32_t *draft_tokens,
            const float *accept_thresholds,
            const float *residual_thresholds,
            int row_count,
            int32_t first_token,
            int first_target_sample_slot,
            bool first_token_from_device,
            const int32_t *stop_tokens,
            int stop_token_count,
            int bonus_target_slot,
            float bonus_threshold,
            DeviceSpeculativeVerifyBatchOutcome *out);

        /** Monotonic live-state epoch used by versioned decode replay. */
        uint64_t liveReplayStateEpoch() const override { return live_replay_state_epoch_; }

        /** Whether the current all-position verifier may hand its stream to the sampler. */
        bool shouldDeferAllPositionVerifierFinalSync() const override;

        /** Store the verifier replay stream for the next all-position logits consumer. */
        void setPendingAllPositionVerifierStream(void *stream) override;

        /** Report host-side safety state for chunk-boundary maintenance. */
        PrefillChunkMaintenanceState prefillChunkMaintenanceState(
            const PrefillChunkPlan &chunk) const override;

        /** Run local chunk-boundary maintenance after the engine gate allows it. */
        bool onPrefillChunkMaintenance(
            const PrefillChunkPlan &chunk,
            const PrefillChunkMaintenanceDecision &decision) override;

        /** Create and return the ForwardExecutionEngine with current config. */
        void ensureForwardEngine();

        // =========================================================================
        // Members
        // =========================================================================

        /// Graph builder (declarative layer)
        std::shared_ptr<IGraphBuilder> graph_builder_;

        /// Graph executor
        DeviceGraphExecutor executor_;

        /// MPI context for distributed execution
        std::shared_ptr<IMPIContext> mpi_ctx_;

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

        /// Persistent prefix cache state. This intentionally lives outside
        /// InferenceState so clear_cache() preserves cross-request blocks.
        std::shared_ptr<PrefixStateCache> prefix_cache_;
        std::shared_ptr<RamPrefixStorageBackend> prefix_ram_backend_;
        std::shared_ptr<DiskPrefixStorageBackend> prefix_disk_backend_;
        std::shared_ptr<DeviceHotPrefixStorageBackend> prefix_device_hot_backend_;
        PrefixPayloadLayout prefix_layout_;
        PrefixCacheStats prefix_cache_stats_;
        uint64_t prefix_fingerprint_ = 0;
        bool prefix_cache_bypassed_ = false;
        std::string prefix_cache_bypass_reason_;
        struct LiveHybridCheckpointStorageSlot
        {
            size_t host_capacity_bytes = 0;
            size_t device_capacity_bytes = 0;
            DeviceId device = DeviceId::invalid();
            std::shared_ptr<std::vector<uint8_t>> host_storage;
            std::shared_ptr<TensorBase> device_storage;
        };
        mutable std::vector<LiveHybridCheckpointStorageSlot> live_hybrid_checkpoint_storage_pool_;
        bool ensurePrefixCacheReady();
        bool isPrefixCacheMoEModel() const;
        bool mtpSpecStatePublicationRequiresCapturedStage() const;
        void *explicitGPUStreamForOperation(const char *operation) const;
        void handleLivePrefixReplayStateAfterMutation(
            const char *operation,
            bool preserve_gpu_replay_state = false);
        PrefixCacheFingerprintResult buildCurrentPrefixFingerprint(
            const PrefixCacheRuntimeConfig &prefix_config) const;
        PrefixCacheKey makePrefixKeyForBlock(
            const std::vector<int32_t> &tokens,
            int block_index,
            uint64_t parent_hash) const;
        void disablePrefixCacheForRunner(const std::string &reason);
        bool acquireLiveHybridCheckpointStorage(PrefixBlockHandle &handle) const;
        bool ensureLiveHybridCheckpointStorage(PrefixBlockHandle &handle) const;
        bool initializeMTPKVCaches(
            int batch_size,
            int max_seq_len,
            ActivationPrecision kv_cache_prec,
            KVCacheLayoutMode kv_layout_mode,
            DeviceId device,
            const std::shared_ptr<IMPIContext> &local_mpi_ctx,
            bool use_sharded_cache,
            bool has_tp,
            bool is_global_tp);
        bool selectMTPTerminalHiddenRow(int row_idx, int seq_len);
        bool executeMTPDepth0(int32_t draft_condition_token,
                              TensorBase *terminal_hidden,
                              int position_id,
                              const char *sidecar_perf_context,
                              bool kv_cache_only = false,
                              BufferId terminal_hidden_buffer_id = BufferId::PREFIX_TERMINAL_HIDDEN,
                              bool defer_final_sync = false);
        bool executeMTPDepth0Batched(const int32_t *draft_condition_tokens,
                                     int token_count,
                                     TensorBase *terminal_hidden,
                                     int position_id,
                                     const char *sidecar_perf_context,
                                     bool kv_cache_only = false,
                                     BufferId terminal_hidden_buffer_id = BufferId::PREFIX_TERMINAL_HIDDEN,
                                     bool defer_final_sync = false,
                                     const void *draft_condition_tokens_device = nullptr,
                                     int draft_condition_ready_slot = -1,
                                     bool draft_condition_ready_is_target = false);
        bool populateMTPShiftedCacheFromPrefill(const int *tokens,
                                                int seq_len,
                                                int batch_size,
                                                int position_offset);
        void updateMTPShiftedCacheMetadata(int active_batch_size);

        struct MTPSidecarGraphCache
        {
            std::unique_ptr<ComputeGraph> graph;
            DeviceGraphExecutor::GraphSegmentCache segment_cache;
            std::vector<IComputeStage *> dynamic_param_stages;
            std::unordered_set<std::string> collective_nodes;
            TensorBase *terminal_hidden = nullptr;
            uint64_t workspace_generation = 0;
            uint64_t moe_placement_epoch = 0;
            bool moe_epoch_sensitive = false;
            std::vector<int32_t> token_ids;
            std::vector<int> position_ids;
            int32_t token_id = 0;
            int position_id = 0;
            int seq_len = 0;
            bool uses_device_token_ids = false;
            bool valid = false;

            void resetReplayState()
            {
                segment_cache.reset(DeviceGraphExecutor::GraphSegmentCache::StreamResetPolicy::Preserve);
            }

            void resetReplayStateAfterWorkspaceRebind()
            {
                resetReplayState();
            }

            void resetSessionState()
            {
                resetReplayState();
                if (graph)
                {
                    graph->reset();
                    for (const auto &node_name : graph->getExecutionOrder())
                    {
                        ComputeNode *node = graph->getNode(node_name);
                        if (node && node->stage)
                            node->stage->resetSessionState();
                    }
                }
            }

            void invalidate()
            {
                segment_cache.reset(DeviceGraphExecutor::GraphSegmentCache::StreamResetPolicy::Destroy);
                graph.reset();
                dynamic_param_stages.clear();
                collective_nodes.clear();
                terminal_hidden = nullptr;
                workspace_generation = 0;
                moe_placement_epoch = 0;
                moe_epoch_sensitive = false;
                token_ids.clear();
                position_ids.clear();
                token_id = 0;
                position_id = 0;
                seq_len = 0;
                uses_device_token_ids = false;
                valid = false;
            }
        };

        MTPSidecarGraphCache mtp_sidecar_depth0_cache_;
        MTPSidecarGraphCache mtp_sidecar_depth0_chained_cache_;
        MTPSidecarGraphCache mtp_sidecar_depth0_kv_only_cache_;
        std::array<MTPSidecarGraphCache, 5> mtp_sidecar_depth0_kv_only_batch_caches_;
        void *pending_mtp_logits_stream_ = nullptr;
        bool defer_all_position_verifier_sync_ = false;
        void *pending_all_position_logits_stream_ = nullptr;

        struct MTPTerminalHiddenRowSelectGraphCache
        {
            std::unique_ptr<ComputeGraph> graph;
            HiddenStateRowSelectStage *stage = nullptr;
            TensorBase *input = nullptr;
            TensorBase *output = nullptr;
            DeviceId device = DeviceId::invalid();
            int seq_capacity = 0;
            int d_model = 0;
            bool valid = false;

            void resetSessionState()
            {
                if (!graph)
                    return;
                graph->reset();
                for (const auto &node_name : graph->getExecutionOrder())
                {
                    ComputeNode *node = graph->getNode(node_name);
                    if (node && node->stage)
                        node->stage->resetSessionState();
                }
            }

            void invalidate()
            {
                graph.reset();
                stage = nullptr;
                input = nullptr;
                output = nullptr;
                device = DeviceId::invalid();
                seq_capacity = 0;
                d_model = 0;
                valid = false;
            }
        };

        MTPTerminalHiddenRowSelectGraphCache mtp_terminal_hidden_row_select_cache_;

        // =========================================================================
        // Full Forward Graph Cache (Decode Optimization)
        // =========================================================================

        /**
         * @brief Signature for caching full forward graphs.
         *
         * Phase 1 goal: avoid rebuilding stage/kernel objects on repeated forwards
         * with the same execution shape/path.
         */
        // =========================================================================
        // Forward Execution Engine (Phase 3: extracted from executeForward)
        // =========================================================================

        /// Forward graph execution engine — owns the forward graph cache
        /// and handles cache HIT/MISS dispatch, GPU graph replay, timeline collection.
        std::unique_ptr<ForwardExecutionEngine> forward_engine_;

        /// Padded sequence length from last forward_batch() call
        int padded_seq_len_ = 0;

        // =========================================================================
        // Snapshot Capture Members
        // =========================================================================

        /// Whether snapshot capture is enabled
        bool snapshot_enabled_ = false;

        /// Optional execution context prefix for disambiguating repeated stage keys.
        std::string snapshot_context_;

        /// Snapshot capture engine (owns storage + routing logic)
        SnapshotCapture snapshot_capture_;

        // =========================================================================
        // Graph Buffer Management Members (Phase 3 - moved from QwenStandardGraph)
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

        /// Cached device pointers for the arena-owned argmax partial-reduction
        /// scratch (two-pass GPU greedy sampling). Resolved once after arena
        /// allocation so the per-decode-step hot path avoids any arena lookups.
        void *argmax_partial_vals_dev_ = nullptr; ///< FP32 [1, argmax_partial_capacity_]
        void *argmax_partial_idxs_dev_ = nullptr; ///< INT32 [1, argmax_partial_capacity_]
        int argmax_partial_capacity_ = 0;         ///< Entries in the partial scratch (0 = unavailable)

        void *stochastic_target_token_ids_dev_ = nullptr; ///< INT32 [4, 256]
        void *stochastic_target_probs_dev_ = nullptr;     ///< FP32 [4, 256]
        void *stochastic_draft_token_ids_dev_ = nullptr;  ///< INT32 [3, 256]
        void *stochastic_draft_probs_dev_ = nullptr;      ///< FP32 [3, 256]
        void *stochastic_target_sample_tokens_dev_ = nullptr; ///< INT32 [1, 4]
        void *stochastic_draft_sample_tokens_dev_ = nullptr; ///< INT32 [1, 3]
        void *mtp_sidecar_condition_token_dev_ = nullptr; ///< INT32 [1], stable sidecar token input
        void *mtp_verifier_input_tokens_dev_ = nullptr; ///< INT32 [1, 4], stable verifier token input
        void *stochastic_topk_partial_vals_dev_ = nullptr; ///< FP32 [partial_blocks, 32]
        void *stochastic_topk_partial_idxs_dev_ = nullptr; ///< INT32 [partial_blocks, 32]
        int stochastic_topk_partial_capacity_ = 0;
        void *stochastic_verify_tokens_dev_ = nullptr;    ///< INT32 [1, 4]
        void *stochastic_verify_accepted_dev_ = nullptr;  ///< INT32 [1, 4]
        void *stochastic_verify_accept_probs_dev_ = nullptr; ///< FP32 [1, 4]
        void *stochastic_verify_thresholds_dev_ = nullptr;   ///< FP32 [1, 4]
        void *stochastic_batch_output_tokens_dev_ = nullptr; ///< INT32 [1, 5]
        void *stochastic_batch_output_meta_dev_ = nullptr;   ///< INT32 [1, 10]
        std::array<int, 4> stochastic_target_top_k_ = {0, 0, 0, 0};
        std::array<int, 3> stochastic_draft_top_k_ = {0, 0, 0};

        /**
         * @brief Producer stream for each compact stochastic distribution slot.
         *
         * A distribution build can consume logits produced by captured replay on
         * a non-context stream. The later sample kernel must run on that same
         * explicit stream so it observes the finished token/probability arrays
         * without inserting a host synchronization. Slots are cleared when the
         * paired sampler consumes them or when request-scoped state is reset.
         */
        std::array<void *, 4> stochastic_target_distribution_streams_ =
            {nullptr, nullptr, nullptr, nullptr};

        /**
         * @brief Producer streams for MTP draft compact distributions.
         *
         * Draft distributions are normally built immediately after a sidecar
         * graph. Keeping the stream here lets stochastic sampling follow a
         * deferred sidecar graph replay on CUDA/ROCm without touching the device
         * null stream or synchronizing the whole context.
         */
        std::array<void *, 3> stochastic_draft_distribution_streams_ =
            {nullptr, nullptr, nullptr};

        /**
         * @brief Event-backed readiness state for deferred stochastic sample tokens.
         *
         * Each slot corresponds to one row in a stochastic sample-token buffer.
         * The event is recorded after the sampler kernel writes the token; GPU
         * consumers such as chained sidecars and all-position verifiers wait on
         * it before reading the slot. `std::shared_ptr<void>` gives the raw
         * backend event a move-safe owner, which matters because this
         * orchestrator is movable in tests and factory paths.
         */
        struct StochasticSampleReadyState
        {
            std::shared_ptr<void> event;
            void *producer_stream = nullptr;
            bool valid = false;
        };
        std::array<StochasticSampleReadyState, 4>
            stochastic_target_sample_ready_;
        std::array<StochasticSampleReadyState, 3>
            stochastic_draft_sample_ready_;

        /**
         * @brief Deferred device-token composition plan for the target verifier.
         *
         * OrchestrationRunner knows which host token and sampled draft slots
         * should feed the verifier, but DeviceGraphOrchestrator knows the graph
         * execution stream.  Keeping this plan pending until
         * prepareAllPositionVerifierGraphMetadata() lets us compose the compact
         * `[first_token, draft_0, ...]` device row on that same stream.
         */
        struct PendingMTPVerifierDeviceTokenPlan
        {
            int32_t first_token = -1;
            int first_target_sample_slot = -1;
            bool first_token_from_device = false;
            int first_draft_slot = 0;
            int draft_token_count = 0;
            int total_verifier_input_tokens = 0;
        };
        std::optional<PendingMTPVerifierDeviceTokenPlan>
            pending_mtp_verifier_device_token_plan_;

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
         * @param workspace_seq_len Optional active execution length for shape-sized scratch.
         * @return true if allocation succeeded (or was already done)
         */
        bool ensureDeviceWorkspaceAllocated(
            const ComputeGraph &graph,
            int workspace_seq_len = 0) override;

        /**
         * @brief Return the current workspace generation for a device.
         */
        uint64_t workspaceGeneration(DeviceId device) const override;

        /**
         * @brief Called once after the first graph build + workspace allocation.
         *
         * Graph-ready is too early for mmap page reclaim on GPU/TP paths because
         * participant-local graph materialization can still be resolving captured
         * transfers. mmap reclaim is performed after first successful prefill.
         */
        void onFirstGraphReady() override;

        /**
         * @brief Advise mmap pages away after first successful prefill.
         */
        void adviseMmapDontneedAfterFirstPrefill();

        // =========================================================================
        // Phase-Aware Weight Access Members (Gap 3 - CPU Decode Participation)
        // =========================================================================

        /// Weight manager for full weights and decode shards
        std::shared_ptr<IWeightManager> weight_manager_;

        /// Weight placement map for decode device selection
        std::shared_ptr<IWeightPlacementMap> weight_placement_map_;

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

        /// Injected topology interface (nullptr if using IMPIContext directly)
        std::shared_ptr<IMPITopology> injected_topology_;

        /// Injected collective context (nullptr if using default)
        std::shared_ptr<ICollectiveContext> injected_collective_ctx_;

        /// TurboQuant context for TQ4 KV cache (owns rotation matrix lifetime)
        std::shared_ptr<TurboQuantContext> turboquant_ctx_;

        /// KV rotation for Q16_1 kurtosis reduction
        std::shared_ptr<ActivationRotation> kv_rotation_;

        /// Global TP context for cross-MPI-rank tensor parallelism (owns communicator lifetime)
        std::shared_ptr<IGlobalTPContext> global_tp_ctx_;

        /// Resolve the active GlobalTP/NodeLocalTP domain context for scalar MTP coordination.
        IGlobalTPContext *globalTPContextForMTPCoordination() const;

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
        std::map<std::string, std::shared_ptr<ITPContext>> domain_tp_contexts_;

        /// Whether PP contexts have been initialized
        bool pp_contexts_initialized_ = false;

        /// Whether TP contexts have been initialized
        bool tp_contexts_initialized_ = false;

        /// Session epoch counter — incremented on each clear_cache() call
        /// Used to detect stale kernel state across inference sessions
        uint64_t session_epoch_ = 0;
        uint64_t live_replay_state_epoch_ = 1;
        uint64_t device_sampling_counter_ = 0;

        bool compute_all_position_logits_ = false;
        bool compute_row_indexed_all_position_logits_ = false;
        int row_indexed_all_position_logits_row_count_ = 0;

        /// Runner-owned graph metadata workspace for vLLM-style MTP verification.
        MTPSpecDecodeMetadataWorkspaceBinding mtp_spec_decode_metadata_binding_{
            MTPSpecDecodeMetadataShape{1, 3}};

        /// Host copy of the verifier row plan waiting to be uploaded before replay.
        std::optional<MTPSpecDecodeVerifierInputPlan>
            pending_mtp_spec_verifier_input_plan_;

        /// Whether host-resident weight data has been released after first prefill
        bool host_resident_released_ = false;
        bool release_host_resident_after_forward_ = true;
        bool mmap_dontneed_advised_ = false;

        /// Whether raw MoE expert tensors were released after eager graph-build packing.
        bool raw_expert_weights_released_after_graph_build_ = false;

        // =========================================================================
        // MoE Expert Rebalance Controller
        // =========================================================================

        /// Optional primary MoE expert rebalance controller (owned)
        std::unique_ptr<MoERebalanceController> moe_rebalance_controller_;

        /// Additional routed overlay-domain rebalance controllers (owned).
        std::vector<std::unique_ptr<MoERebalanceController>> moe_rebalance_extra_controllers_;

        /// Optional expert weight payload provider for metadata-based host retention (owned)
        std::unique_ptr<ExpertWeightPayloadProvider> expert_payload_provider_;

        /// Model-context-owned prepared weight store (Phase 4-5)
        std::shared_ptr<PreparedWeightStore> prepared_weight_store_;

        /// Frozen model weight set for audit/validation (Phase 6)
        std::unique_ptr<FrozenModelWeightSet> frozen_weight_set_;

        /// Build FrozenModelWeightSet from pre-resolved layer weights (Phase 6)
        void buildFrozenWeightSet(
            const ModelWeights &weights,
            const std::unordered_map<int, LayerWeights> &resolved_layers,
            int first_layer, int last_layer);

        /// Ensure a stable one-row terminal hidden buffer exists for MTP sidecar input.
        bool ensureMTPTerminalHiddenBuffer();

        /// Execute the cached graph-native row select used for MTP terminal hidden refresh.
        bool executeMTPTerminalHiddenRowSelect(int row_idx, int seq_len);

        /// Execute a cached graph-native hidden row select into an MTP buffer.
        bool executeMTPHiddenRowSelect(
            TensorBase *input,
            BufferId input_buffer_id,
            TensorBase *output,
            BufferId output_buffer_id,
            MTPTerminalHiddenRowSelectGraphCache &cache,
            const char *node_name,
            int row_idx,
            int seq_len);

        /// Copy the latest forward pass terminal hidden row into the stable MTP input buffer.
        bool refreshMTPTerminalHiddenState(int seq_len, int batch_size);

        /// Reset input-dependent dynamic state on all cached kernels
        /// Implemented in .cpp to avoid including KernelFactory.h in the header
        void resetKernelDynamicState();
    };

} // namespace llaminar2
