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
#include "../../moe/MoERebalanceController.h"          // For ExpertReplicaSet
#include "../../moe/CPUCurrentBatchLLEP.h"             // CPU transient LLEP authority
#include "../../moe/MoEExpertOverlayProfiler.h"        // For overlay profiling summary flush
#include "../../moe/MoEOverlayEpochLeaseLifecycle.h"  // Typed epoch submission ownership
#include "../../factory/InferenceRunnerFactory.h"      // For FactoryPPStageConfig
#include "../../../snapshots/SnapshotCapture.h"        // Snapshot capture (extracted Phase 2)
#include "../engine/ForwardExecutionEngine.h"          // Forward execution engine (extracted Phase 3)
#include "../../../loaders/IWeightStreamer.h"          // For weight streaming (Option B)
#include "../../../interfaces/IModelContext.h"         // For interface-based construction
#include "../../../memory/BufferArena.h"               // Phase 2: unified buffer management
#include "../../prefix_cache/PrefixCacheFingerprint.h"
#include "../../prefix_cache/PrefixCacheStats.h"
#include "../../prefix_cache/PrefixStorageBackend.h"   // PrefixBlockHandle restore-source ownership
#include "../../mtp/MTPSpecDecodeMetadata.h"
#include "../../mtp/MTPSidecarCaptureLayout.h"
#include "../../mtp/MTPVerifierPolicy.h"
#include "../../mtp/HostedDeviceGenerationLifecycle.h"
#include "../../../interfaces/IMPITopology.h"          // For interface-based construction
#include "../../../interfaces/ICollectiveContext.h"    // For interface-based construction
#include "../../../config/TPDomain.h"                  // For MultiDomainTPConfig (Phase 6.3)
#include "../../../config/PipelineConfig.h"            // For unified PP+TP configuration (Phase 6)
#include "../../../collective/ILocalPPContext.h"       // For unique_ptr<ILocalPPContext> in maps
#include "../../../collective/ITPContext.h"            // For polymorphic TP context ownership
#include "../../../collective/ILocalTPContext.h"       // For unique_ptr<ILocalTPContext> in maps
#include "../../../collective/IGlobalTPContext.h"      // For shared_ptr<IGlobalTPContext> ownership
#include <memory>
#include <atomic>
#include <optional>
#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <exception>
#include <limits>
#include <mutex>
#include <shared_mutex>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <vector>
#include <string>
#include <cstring>
#include <cstdint>

namespace llaminar2
{
    enum class DeviceTimelineRole : uint8_t;

    // Forward declarations
    class IMPIContext;
    class GpuExpertTransferStagingPool;
    class IKVCache;
    class IWeightManager;
    class IWeightPlacementMap;
    class WeightManager;
    class WeightPlacementMap;
    class TensorParallelConfig;
    class TurboQuantContext;
    class MoERebalanceController;
    class MappedTransferProgressEpoch;
    class ExpertWeightPayloadProvider;
    class IMoEOverlayInferenceCompletionEvent;
    struct MoEOverlayInferenceParticipantGraphBinding;
    class PreparedWeightStore;
    class FrozenModelWeightSet;
    class PrefixStateCache;
    class RamPrefixStorageBackend;
    class DiskPrefixStorageBackend;
    class DeviceHotPrefixStorageBackend;
    class ActivationRotation;
    enum class KVCacheLayoutMode : uint8_t;

    class ForwardGraphExecutionRendezvous
    {
    public:
        explicit ForwardGraphExecutionRendezvous(
            size_t expected_participants,
            std::string label = {});

        ForwardGraphExecutionRendezvous(const ForwardGraphExecutionRendezvous &) = delete;
        ForwardGraphExecutionRendezvous &operator=(const ForwardGraphExecutionRendezvous &) = delete;

        bool arriveAndWait(DeviceId device, int timeout_ms);

    private:
        const size_t expected_participants_;
        const std::string label_;
        size_t arrivals_ = 0;
        bool released_ = false;
        bool failed_ = false;
        std::mutex mutex_;
        std::condition_variable cv_;
    };

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

        /// Shape of the most recent main-model forward that populated `hidden`.
        /// MTP sidecars use this to decide whether `hidden` is already a single
        /// terminal row or whether a compact row-select is required first.
        int last_forward_seq_len = 0;
        int last_forward_batch_size = 0;
        /// Per-request token counts written by the most recent main forward.
        ///
        /// For padded request batches `last_forward_seq_len` is the padded row
        /// width, not every request's true length.  MTP sidecar input refresh
        /// uses this vector to gather each request's real terminal hidden row.
        std::vector<int> last_forward_request_lengths;

        /// True when `prefix_terminal_hidden` is the accepted terminal row for
        /// the current live state. Ordinary forward makes this stale; prefix
        /// restore, accepted-state publication, or an explicit terminal refresh
        /// makes it current again.
        bool mtp_terminal_hidden_current = false;

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
         * @brief Reset committed KV plus hybrid GDN/short-conv state.
         *
         * Every cache consumes the same explicit reset context.  On GPU this
         * means PP shards, main KV metadata, GDN recurrence, and short-conv
         * history all enqueue on one caller-owned stream.  The orchestrator can
         * therefore publish one transitive completion event after this method
         * and the shifted-MTP reset below have both succeeded.
         */
        bool resetCommittedKVAndRecurrentState(
            const IKVCache::StateResetContext &context)
        {
            if (kv_cache && !kv_cache->resetRequestState(context))
                return false;
            for (auto &[device, cache] : pp_kv_caches)
            {
                (void)device;
                if (cache && !cache->resetRequestState(context))
                    return false;
            }
            return true;
        }

        /**
         * @brief Reset request-local shifted-MTP sidecar state.
         *
         * MTP sidecars are shifted relative to the main cache and are owned by
         * the active speculative transaction, not by the main decode stream.
         */
        bool resetMTPShiftedSidecarState(
            const IKVCache::StateResetContext &context)
        {
            for (auto &cache : mtp_kv_caches)
            {
                if (cache && !cache->resetRequestState(context))
                    return false;
            }
            return true;
        }

        /**
         * @brief Clear host logical sequence mirrors and terminal-row freshness.
         *
         * These mirrors are request-boundary metadata. They must be reset when
         * KV/GDN/MTP live state is reset, but they are not themselves cache
         * payloads and should be named separately in boundary code.
         */
        void clearLogicalSequenceState()
        {
            std::fill(positions.begin(), positions.end(), 0);
            std::fill(sequence_lengths.begin(), sequence_lengths.end(), 0);
            last_forward_seq_len = 0;
            last_forward_batch_size = 0;
            last_forward_request_lengths.clear();
            mtp_terminal_hidden_current = false;
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
            if (auto it = extension_buffers.find(BufferId::K_FULL_PREFILL);
                it != extension_buffers.end() && it->second)
            {
                lb.K_full_prefill = it->second.get();
            }
            if (auto it = extension_buffers.find(BufferId::V_FULL_PREFILL);
                it != extension_buffers.end() && it->second)
            {
                lb.V_full_prefill = it->second.get();
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
     * @brief One compact byte-level observation from a failed mirrored MTP pass.
     *
     * Mirrored LocalTP MTP heads are required to produce identical logits on
     * every participant.  When that invariant fails, copying complete tensors
     * into a log would be both noisy and expensive.  This record instead names
     * one logical row, reports its byte count, and carries a deterministic
     * FNV-1a digest.  Records are produced only after the rank has already
     * rejected a mismatched proposal; they are never part of normal inference.
     */
    struct MTPMirroredTensorDigest
    {
        std::string name;       ///< Stable semantic boundary name.
        uint64_t hash = 0;      ///< FNV-1a digest of the copied device bytes.
        size_t byte_count = 0;  ///< Number of bytes included in @ref hash.
        bool available = false; ///< True when the device-to-host diagnostic copy succeeded.
        std::string value_preview; ///< Optional bounded scalar values for control metadata.
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
    class DeviceGraphOrchestrator : public IInferenceRunner,
                                    public IForwardExecutionHost,
                                    public ICPUCurrentBatchLLEPPhysicalExecutor,
                                    public IMoEOverlayDeviceInferenceBoundary
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

        /**
         * @brief Arm or clear a one-shot pre-execution rendezvous for the next forward.
         *
         * RankOrchestrator uses this for LocalTP fan-out so all child production
         * graphs finish cache-miss materialization before any child enters the
         * first collective. Passing nullptr clears the rendezvous.
         */
        void setForwardGraphExecutionRendezvous(
            std::shared_ptr<ForwardGraphExecutionRendezvous> rendezvous);

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
         * @brief Install alternate full dense bindings for phase-split decode.
         *
         * In ExpertOverlay phase-split modes, prefill may use tensor-parallel
         * dense weights while decode uses a replicated dense sidecar to preserve
         * serial row semantics.  MTP verifier rows are part of that decode
         * sidecar: if MTP is enabled, this weight set must carry complete MTP
         * bindings for terminal decode participants.
         */
        void setDecodeReplicatedDenseWeightSet(std::unique_ptr<FrozenModelWeightSet> weight_set);

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
        MoEOverlayAuthorityExecutionKind
        moeOverlayAuthorityExecution() const override;
        bool deviceResidentMoEOverlayMaintenanceReady() const override;
        int moeRebalanceParticipantId() const override;

        /// Apply expert masks to all MoEExpertComputeStages in cached FFN graphs.
        /// Called after rebalancing to update which experts each rank computes.
        /// @param masks Per-layer expert masks (masks[layer][expert] == true means active)
        /// @param received_weights Optional transferred packed weights from MPI transfer
        void applyExpertMasks(
            const std::vector<std::vector<bool>> &masks,
            const ReceivedWeightsMap &received_weights = {},
            const ReceivedPreparedExpertsMap &received_prepared_experts = {});
        void applyExpertMasksForDomain(
            const std::string &domain_id,
            const std::vector<std::vector<bool>> &masks,
            const ReceivedWeightsMap &received_weights = {},
            const ReceivedPreparedExpertsMap &received_prepared_experts = {});

        /// Non-destructively collect packed expert weights for experts requested
        /// by masks. Used for intra-rank migration between composed TP domains
        /// without falling back to raw GGUF host tensors.
        ReceivedWeightsMap collectExpertWeightsForMasks(
            const std::vector<std::vector<bool>> &masks) const;

        /// Try to satisfy requested expert arrivals by directly copying GPU
        /// packed descriptors from a sibling device orchestrator. Returns a
        /// copy of masks with successfully copied experts cleared; remaining
        /// true bits should use collectExpertWeightsForMasks() fallback.
        std::vector<std::vector<bool>> transferExpertWeightsDirectForMasksFrom(
            DeviceGraphOrchestrator &source,
            const std::vector<std::vector<bool>> &masks);

        struct PendingGpuDirectTransferSlotArrival
        {
            MoEExpertComputeStage *destination_stage = nullptr;
            GpuDirectTransferSlotArrivals arrivals;

            bool empty() const
            {
                return destination_stage == nullptr || arrivals.empty();
            }
        };

        struct PreparedDirectExpertTransfer
        {
            std::vector<std::vector<bool>> remaining_masks;
            std::vector<PendingGpuDirectTransferSlotArrival> staged_arrivals;

            bool empty() const
            {
                return staged_arrivals.empty();
            }
        };

        /// Stage same-backend GPU expert arrivals into transfer slots without
        /// publishing active GEMM engines. Call activatePreparedGpuDirectExpertTransfers()
        /// before applying masks that depend on these arrivals.
        PreparedDirectExpertTransfer prepareExpertWeightsDirectForMasksFrom(
            DeviceGraphOrchestrator &source,
            const std::vector<std::vector<bool>> &masks);

        bool activatePreparedGpuDirectExpertTransfers(
            const std::vector<PendingGpuDirectTransferSlotArrival> &staged_arrivals);

        void clearPendingGpuDirectExpertTransfers();
        bool activatePendingGpuDirectExpertTransfers();
        void retireCompletedGpuDirectActivationCompletions();

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

        /**
         * @brief Materialize the complete serial workspace family without inference.
         *
         * The requested shape supplies the representative decode participant;
         * the implementation also declares the maximum admitted prefill,
         * grouped verifier, MTP sidecar, and maintenance participants before it
         * publishes one immutable workspace generation. Repeated calls are
         * idempotent and reject a changed generation. This also forces
         * graph-build-time expert preparation before raw host weights may be
         * released.
         *
         * @param seq_len Positive representative forward width.
         * @param batch_size Positive configured request capacity.
         * @param pipeline_hidden_input Stable previous-stage activation owner
         *        required by a non-embedding PP graph; null for graph roots.
         * @return True once the complete family owns stable workspace addresses.
         */
        bool materializeForwardGraphForShape(
            int seq_len,
            int batch_size = 1,
            TensorBase *pipeline_hidden_input = nullptr);

        /**
         * @brief Prepare CPU MoE expert slabs used only by MTP/nextn sidecars.
         *
         * Qwen nextn/MTP blocks can live outside the main transformer layer loop,
         * so ordinary eager forward-graph materialization may never construct a
         * MoE stage for those expert tensors before raw mmap-backed host data is
         * released.  This hook materializes the same local expert range that the
         * sidecar graph will later request, ensuring sidecar graph construction
         * resolves persistent PreparedWeightStore slabs instead of attempting a
         * late raw repack.
         *
         * @return true when no MTP MoE slabs are needed or all required slabs
         *         were prepared successfully.
         */
        bool prepareMTPMoEExpertSlabs(DeviceId device);

        /// Transfer packed weights for migrating experts via MPI.
        /// Homogeneous CPU arrivals are final prepared engines; conversion
        /// domains use explicit serialized payloads.
        ExpertTransferResult transferExpertWeights(
            const std::vector<ExpertMigration> &manifest);

        /// Transfer packed weights for replicated experts via MPI.
        /// Unlike transferExpertWeights(), the sender keeps its weights (non-destructive).
        /// The receiver gets pre-packed weights to avoid VNNI repacking from raw.
        /// @param replicas The active replica set describing which experts to transfer
        /// @param num_layers Number of MoE layers
        /// @return Backend-explicit arrivals for this rank's new replicas
        ExpertTransferResult transferReplicaWeights(
            const ExpertReplicaSet &replicas);

        /** @copydoc ICPUCurrentBatchLLEPPhysicalExecutor::materializeCPUCurrentBatchLLEPTransaction */
        bool materializeCPUCurrentBatchLLEPTransaction(
            CPUCurrentBatchLLEPTransactionState &state,
            ICPUCurrentBatchLLEPExpertConsumer &expert_consumer,
            IGlobalTPContext &tp_ctx) override;

        /** @copydoc ICPUCurrentBatchLLEPPhysicalExecutor::restoreCPUCurrentBatchLLEPPhysicalState */
        bool restoreCPUCurrentBatchLLEPPhysicalState(
            CPUCurrentBatchLLEPTransactionState &state,
            ICPUCurrentBatchLLEPExpertConsumer &expert_consumer,
            IGlobalTPContext &tp_ctx) override;

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

        /**
         * @brief Mark the current hidden tensor as freshly produced by main forward for tests.
         *
         * Unit tests that inject `state_.hidden` directly bypass the normal forward
         * path that records terminal-hidden freshness for MTP sidecars. This helper
         * preserves the production invariant: sidecar execution must only consume a
         * hidden tensor that was explicitly published as current.
         *
         * @param seq_len Number of rows represented by the injected hidden tensor.
         * @param batch_size Batch size represented by the injected hidden tensor.
         */
        void markMainForwardHiddenProducedForTesting(int seq_len, int batch_size)
        {
            noteMainForwardHiddenProducedForMTP(seq_len, batch_size);
        }

        /**
         * @brief Mark a padded batched forward for terminal-hidden tests.
         *
         * `request_lengths[i]` is the number of real, non-padding tokens in
         * request `i` from the most recent forward.
         */
        void markMainForwardHiddenProducedForTesting(
            int seq_len,
            int batch_size,
            const std::vector<int> &request_lengths)
        {
            noteMainForwardHiddenProducedForMTP(
                seq_len,
                batch_size,
                request_lengths);
        }

        /**
         * @brief Refresh the stable MTP terminal-hidden buffer for tests.
         *
         * This exposes the same production helper used by MTP sidecar
         * resolution, allowing unit tests to verify row selection and batch
         * shape guards without constructing a full speculative scheduler.
         */
        bool refreshMTPTerminalHiddenForTesting(int seq_len, int batch_size)
        {
            return refreshMTPTerminalHiddenState(seq_len, batch_size);
        }

        /**
         * @brief Inspect the stable MTP terminal-hidden buffer published by tests.
         */
        const TensorBase *mtpTerminalHiddenForTesting() const
        {
            return state_.prefix_terminal_hidden.get();
        }

        /**
         * @brief Capture compact row digests after a mirrored MTP proposal fails.
         *
         * This error-path diagnostic samples the device-owned terminal-hidden
         * input, every semantically important sidecar boundary, live device
         * positions, and resident proposal slots.  It is deliberately absent
         * from the successful hot path: the caller invokes it only after two
         * mirrored participants have already returned different draft tokens.
         *
         * Each tensor contributes its first logical row because production
         * proposal sidecars execute one condition row.  The copy uses this
         * runner's explicit GPU stream and leaves TensorBase coherence metadata
         * untouched, so collecting diagnostics cannot make a stale host mirror
         * authoritative.
         *
         * @return Ordered digest inventory suitable for participant comparison.
         */
        std::vector<MTPMirroredTensorDigest>
        captureFailedMirroredMTPDigests();

        /**
         * @brief Mark the active compact request-prefill logits transaction.
         *
         * Production code sets this count while `forward_batch()` owns one
         * terminal full-vocabulary row per request and clears it after the
         * resident sampler consumes those rows. Unit tests use this hook to
         * exercise LM-head ownership classification without performing GPU
         * work or manufacturing device tensors.
         *
         * @param row_count Active compact row count, or zero to end the test
         *        transaction.
         */
        void markRequestBatchedPrefillLogitsForTesting(int row_count)
        {
            request_batched_prefill_logits_row_count_ =
                std::max(0, row_count);
        }

        // =====================================================================
        // IInferenceRunner: Device & Logits Local API overrides
        // =====================================================================

        DeviceId primaryDeviceId() const override { return state_.device_id; }

        bool hasLogitsLocal() const override { return activeMainLogitsAreColumnParallel(); }

        LogitsLocalInfo getLogitsLocalInfo() const override
        {
            if (!activeMainLogitsAreColumnParallel())
                return {};
            auto device_opt = state_.logits_local->current_device();
            const size_t vocab_local = localLogitsVocabColumns(state_.logits_local.get());
            const size_t row_stride = localLogitsRowStrideColumns(state_.logits_local.get());
            /*
             * This is a metadata-only view. An explicit stream would falsely
             * imply that the caller consumed the graph producer handoff. GPU
             * samplers and host gathers must use their consume*() APIs.
             */
            void *stream = nullptr;
            return LogitsLocalInfo{
                state_.logits_local->gpu_data_ptr(),
                device_opt,
                vocab_local,
                static_cast<size_t>(localLogitsVocabOffset()),
                state_.logits_local.get(),
                stream,
                // Expose this runner's arena-owned argmax scratch so the
                // multi-device sampler can drive the multi-block reduction
                // without any hot-path allocation.
                argmax_partial_vals_dev_,
                argmax_partial_idxs_dev_,
                argmax_partial_capacity_,
                row_stride};
        }

        LogitsLocalInfo consumeLogitsLocalInfoForSampling() override
        {
            if (!activeMainLogitsAreColumnParallel())
                return {};

            auto device_opt = state_.logits_local->current_device();
            const size_t vocab_local = localLogitsVocabColumns(state_.logits_local.get());
            const size_t row_stride = localLogitsRowStrideColumns(state_.logits_local.get());

            /*
             * TP sampling is the semantic consumer of main-decode logits.  Use
             * the stream published by graph replay when present; otherwise use
             * this runner's explicit worker stream.  Sampling on any unrelated
             * stream can race graph-captured decode and read an older logits
             * row, which is both nondeterministic and very hard to diagnose.
             */
            void *stream = nullptr;
            if (device_opt.has_value() && device_opt->is_gpu())
            {
                stream = consumePendingLogitsStream(
                    PendingLogitsStreamRole::MainDecode,
                    "consumeLogitsLocalInfoForSampling");
                if (!stream)
                    stream = explicitGPUStreamForOperation(
                        "consumeLogitsLocalInfoForSampling");
            }

            return LogitsLocalInfo{
                state_.logits_local->gpu_data_ptr(),
                device_opt,
                vocab_local,
                static_cast<size_t>(localLogitsVocabOffset()),
                state_.logits_local.get(),
                stream,
                argmax_partial_vals_dev_,
                argmax_partial_idxs_dev_,
                argmax_partial_capacity_,
                row_stride};
        }

        LogitsLocalInfo consumeLogitsLocalInfoForHostGather() override
        {
            LogitsLocalInfo info = getLogitsLocalInfo();
            if (info.device.has_value() && info.device->is_gpu())
            {
                info.stream = prepareLogitsHostObservation(
                    HostLogitsSurface::Main,
                    "consumeLogitsLocalInfoForHostGather");
            }
            return info;
        }

        bool hasMTPLogitsLocal() const override
        {
            if (!mtpSidecarLogitsAreColumnParallel())
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
            auto device_opt = mtp_logits->current_device();
            const size_t vocab_local = localLogitsVocabColumns(mtp_logits);
            const size_t row_stride = localLogitsRowStrideColumns(mtp_logits);
            // Metadata-only views never advertise an unearned producer stream.
            void *stream = nullptr;
            return LogitsLocalInfo{
                mtp_logits->gpu_data_ptr(),
                device_opt,
                vocab_local,
                static_cast<size_t>(localLogitsVocabOffset()),
                mtp_logits,
                stream,
                argmax_partial_vals_dev_,
                argmax_partial_idxs_dev_,
                argmax_partial_capacity_,
                row_stride};
        }

        LogitsLocalInfo consumeMTPLogitsLocalInfoForSampling() override
        {
            if (!hasMTPLogitsLocal())
                return {};

            auto it = state_.extension_buffers.find(BufferId::MTP_LOGITS);
            TensorBase *mtp_logits = it->second.get();
            auto device_opt = mtp_logits->current_device();
            const size_t vocab_local = localLogitsVocabColumns(mtp_logits);
            const size_t row_stride = localLogitsRowStrideColumns(mtp_logits);

            /*
             * LocalTP MTP sampling is the semantic consumer of sidecar logits.
             * Carry the exact sidecar producer stream into the TP sampler so
             * each shard is sampled after its local MTP graph replay.  Falling
             * back to the runner's explicit operation stream keeps CPU/no-graph
             * paths simple without ever touching a null/default GPU stream.
             */
            void *stream = nullptr;
            if (device_opt.has_value() && device_opt->is_gpu())
            {
                stream = consumePendingLogitsStream(
                    PendingLogitsStreamRole::MTPSidecar,
                    "consumeMTPLogitsLocalInfoForSampling");
                if (!stream)
                    stream = explicitGPUStreamForOperation(
                        "consumeMTPLogitsLocalInfoForSampling");
            }

            return LogitsLocalInfo{
                mtp_logits->gpu_data_ptr(),
                device_opt,
                vocab_local,
                static_cast<size_t>(localLogitsVocabOffset()),
                mtp_logits,
                stream,
                argmax_partial_vals_dev_,
                argmax_partial_idxs_dev_,
                argmax_partial_capacity_,
                row_stride};
        }

        LogitsLocalInfo consumeMTPLogitsLocalInfoForHostGather() override
        {
            LogitsLocalInfo info = getMTPLogitsLocalInfo();
            if (info.device.has_value() && info.device->is_gpu())
            {
                info.stream = prepareLogitsHostObservation(
                    HostLogitsSurface::MTPSidecar,
                    "consumeMTPLogitsLocalInfoForHostGather");
            }
            return info;
        }

        bool hasAllPositionLogitsLocal() const override
        {
            return activeAllPositionLogitsAreColumnParallel();
        }

        LogitsLocalInfo getAllPositionLogitsLocalInfo() const override
        {
            if (!hasAllPositionLogitsLocal())
                return {};

            auto device_opt = state_.all_position_logits_local->current_device();
            const size_t vocab_local =
                localLogitsVocabColumns(state_.all_position_logits_local.get());
            const size_t row_stride =
                localLogitsRowStrideColumns(state_.all_position_logits_local.get());
            // Metadata-only views never advertise an unearned producer stream.
            void *stream = nullptr;
            return LogitsLocalInfo{
                state_.all_position_logits_local->gpu_data_ptr(),
                device_opt,
                vocab_local,
                static_cast<size_t>(localLogitsVocabOffset()),
                state_.all_position_logits_local.get(),
                stream,
                argmax_partial_vals_dev_,
                argmax_partial_idxs_dev_,
                argmax_partial_capacity_,
                row_stride};
        }

        LogitsLocalInfo consumeAllPositionLogitsLocalInfoForSampling() override
        {
            if (!hasAllPositionLogitsLocal())
                return {};

            LogitsLocalInfo info = getAllPositionLogitsLocalInfo();
            if (!info)
                return {};

            if (info.device.has_value() && info.device->is_gpu())
            {
                /*
                 * All-position verifier sampling is the semantic consumer of
                 * verifier graph replay.  Consume the replay stream here so
                 * LocalTP argmax reads this shard only after graph replay has
                 * produced it; non-captured execution still uses an explicit
                 * non-null worker stream.
                 */
                void *stream = consumePendingLogitsStream(
                    PendingLogitsStreamRole::AllPositionVerifier,
                    "consumeAllPositionLogitsLocalInfoForSampling");
                if (!stream)
                {
                    stream = explicitGPUStreamForOperation(
                        "consumeAllPositionLogitsLocalInfoForSampling");
                }
                info.stream = stream;
            }

            return info;
        }

        LogitsLocalInfo consumeAllPositionLogitsLocalInfoForHostGather() override
        {
            LogitsLocalInfo info = getAllPositionLogitsLocalInfo();
            if (info.device.has_value() && info.device->is_gpu())
            {
                info.stream = prepareLogitsHostObservation(
                    HostLogitsSurface::AllPositionVerifier,
                    "consumeAllPositionLogitsLocalInfoForHostGather");
            }
            return info;
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
        bool forwardGroupedMTPVerifierWithHostTokenIds(
            const std::vector<std::vector<int>> &token_batches) override;
        bool forwardGroupedMTPVerifierWithDeviceTokenIds(
            const int *token_shadow,
            const void *token_ids_device,
            int seq_len) override;
        bool advanceMTPMainConditionFromDeviceResidentLogicalState(
            int32_t token_shadow,
            const DeviceResidentLogicalSequenceStateHandle &logical_state,
            int request_index = 0) override;
        bool advanceMTPMainConditionFromDeviceTargetSample(
            int32_t token_shadow,
            int target_sample_slot) override;

        /**
         * @brief Batched verifier forward from a flat device token buffer.
         *
         * The host vectors are a logical shadow only; GPU embedding reads the
         * padded `[batch, padded_seq_len]` INT32 rows from `token_ids_device`.
         * This is the SingleDevice building block for request-batched MTP graph
         * capture. Multi-participant topologies must provide participant-local
         * device pointers before using the same contract.
         */
        bool forwardBatchWithDeviceTokenIds(
            const std::vector<std::vector<int>> &token_batches,
            const void *token_ids_device,
            int padded_seq_len) override;

        const void *prepareMTPVerifierInputTokensOnDevice(
            int32_t first_token,
            int first_draft_slot,
            int draft_token_count,
            int total_verifier_input_tokens) override;
        const void *prepareMTPVerifierInputTokenBatchOnDevice(
            const DeviceMTPVerifierInputBatchRequest *requests,
            int request_count,
            int logical_padded_seq_len) override;
        const void *prepareMTPVerifierInputTokensOnDeviceFromHostRow(
            const int32_t *verifier_tokens,
            int total_verifier_input_tokens,
            int draft_token_count) override;

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
        bool supportsMTPShiftedRowReuseFromSidecar() const override;
        bool forwardMTPFromLastDraft(int32_t draft_condition_token, int position_id) override;
        bool forwardMTPFromLastDraftForDeviceSampling(
            int32_t draft_condition_token,
            int position_id) override;
        bool forwardMTPFromDeviceDraftAtLivePositionForDeviceSampling(
            int draft_sample_slot,
            int position_offset) override;
        bool forwardMTPFromDeviceTargetAtLivePositionForDeviceSampling(
            int target_sample_slot) override;
        bool forwardMTPFromDeviceResidentLogicalStateForDeviceSampling(
            const DeviceResidentLogicalSequenceStateHandle &logical_state,
            int request_index = 0) override;
        bool forwardMTPAndSampleGreedy(int32_t draft_condition_token, int32_t *out_token) override;
        bool forwardMTPAndSampleGreedyToDeviceDraftSlot(
            int32_t draft_condition_token,
            int draft_sample_slot,
            int32_t *out_token) override;
        bool forwardMTPBatchAndSampleGreedy(
            const int32_t *draft_condition_tokens,
            const int *position_ids,
            int request_batch,
            int32_t *out_tokens) override;
        bool advanceMTPRequestBatchConditionOnDevice(
            const DeviceResidentLogicalSequenceStateHandle &logical_state,
            int request_batch,
            const SamplingParams &params,
            const uint64_t *stochastic_position_seeds = nullptr) override;
        bool observeDeviceResidentNextConditionTokens(
            const DeviceResidentLogicalSequenceStateHandle &logical_state,
            int request_count,
            int32_t *out_tokens) override;
        bool forwardMTPBatchFromDeviceResidentLogicalStateAndSampleGreedyToDeviceDraftSlots(
            const DeviceResidentLogicalSequenceStateHandle &logical_state,
            int request_batch,
            int first_draft_slot,
            int slot_stride) override;
        bool forwardMTPBatchFromLastDraftAndSampleGreedy(
            const int32_t *draft_condition_tokens,
            const int *position_ids,
            int request_batch,
            int32_t *out_tokens) override;
        bool forwardMTPBatchFromDeviceDraftSlotsAndSampleGreedyToDeviceDraftSlots(
            const DeviceResidentLogicalSequenceStateHandle &logical_state,
            int request_batch,
            int first_condition_slot,
            int condition_slot_stride,
            int position_offset,
            int first_draft_slot,
            int draft_slot_stride) override;
        bool forwardMTPFromLastDraftAndSampleGreedy(
            int32_t draft_condition_token,
            int position_id,
            int32_t *out_token) override;
        bool forwardMTPFromLastDraftAndSampleGreedyToDeviceDraftSlot(
            int32_t draft_condition_token,
            int position_id,
            int draft_sample_slot,
            int32_t *out_token) override;
        bool flushPendingMTPWork() override;
        void setMTPAllPositionVerifierSyncDeferralEnabled(bool enabled) override;
        void setMTPMainDecodeSyncDeferralEnabled(bool enabled) override;
        bool consumeUnusedReplicatedMainLogitsPublication() override;
        bool supportsMTPSpecStatePublication() const override;
        bool supportsDeviceResidentMTPSpecStatePublication() const override;
        bool supportsLogicalMTPVerifierBaseCheckpoint() const override;
        MTPVerifierRowCapability mtpVerifierRowCapability() const override;
        bool publishAcceptedMTPSpecState(
            const MTPSpecStepPlan &plan,
            std::string *error = nullptr) override;
        bool publishAcceptedMTPSpecStateBatch(
            const MTPSpecStepPlanBatch &plans,
            std::string *error = nullptr) override;
        /**
         * @brief Publish accepted state for a grouped decode-equivalent outcome.
         *
         * This method is intentionally implemented as a scoped entry point over
         * the existing batch publisher.  The internal scope lets the batch
         * publisher reuse the same KV/GDN/short-conv/terminal-hidden mutation
         * code while ordinary direct calls still fail unless
         * supportsMTPSpecStatePublication() is true.
         *
         * @param plans Accepted-row publication plan derived from grouped
         *        decode-equivalent verifier rows.
         * @param error Optional destination for the first failure reason.
         * @return true when the live runner state and host mirrors were advanced
         *         exactly to the accepted prefix described by @p plans.
         */
        bool publishGroupedDecodeEquivalentMTPSpecStateBatch(
            const MTPSpecStepPlanBatch &plans,
            std::string *error = nullptr) override;
        bool publishAcceptedMTPSpecStateBatchFromDeviceOutcome(
            const DeviceSpeculativePublicationRequest &request,
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
            int position_offset_override = -1,
            int already_appended_shifted_kv_tokens = -1) override;
        bool commitMTPShiftedRowFromCurrentTerminalHidden(
            int32_t token,
            int already_appended_tokens,
            bool allow_speculative_discard = false,
            int position_offset_override = -1) override;
        bool commitMTPShiftedRowFromCheckpointTerminalHidden(
            const PrefixStateSnapshot &checkpoint,
            int32_t token,
            int already_appended_tokens,
            bool allow_speculative_discard = false,
            int position_offset_override = -1) override;
        bool commitMTPInitialShiftedRowFromDeviceOutcome(
            const PrefixStateSnapshot &checkpoint,
            const DeviceSpeculativeOutcomeHandle &outcome,
            int request_index,
            int main_forward_token_count,
            bool allow_speculative_discard = false) override;
        bool commitMTPShiftedRowFromDeviceTargetSample(
            int target_sample_slot,
            int already_appended_tokens,
            bool allow_speculative_discard = false) override;
        bool commitMTPShiftedRowFromDeviceResidentLogicalState(
            const DeviceResidentLogicalSequenceStateHandle &logical_state,
            int request_index,
            int already_appended_tokens,
            bool allow_speculative_discard = false) override;
        bool commitMTPShiftedRowsFromDeviceOutcome(
            const DeviceSpeculativeOutcomeHandle &outcome,
            int request_index,
            int already_appended_tokens,
            int max_state_commit_rows,
            int main_forward_token_count,
            bool allow_speculative_discard = false) override;
        bool ensureMTPCheckpointTerminalHidden() override;
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
        bool sampleGreedyFromMTPLogitsToDeviceDraftSlot(
            int draft_sample_slot,
            int32_t *out_token) override;
        bool sampleGreedyFromMainLogitsToDeviceTargetSlot(
            int target_sample_slot,
            int32_t *out_token) override;
        int sampleGreedyFromAllPositionLogitsOnDevice(int row) override;
        bool sampleGreedyFromAllPositionLogitsOnDeviceRows(
            int start_row,
            int row_count,
            int32_t *out_tokens) override;
        bool supportsGreedyAllPositionBatchOutcomeOnDevice() const override;
        bool usesMirroredMTPHeadForVerifier() const override;
        bool verifyGreedyAllPositionBatchOutcomeOnDevice(
            const int32_t *draft_tokens,
            int draft_token_count,
            const int32_t *stop_tokens,
            int stop_token_count,
            DeviceSpeculativeVerifyBatchOutcome *out) override;
        bool verifyGreedyAllPositionBatchOutcomeOnDeviceResident(
            const int32_t *draft_tokens,
            int draft_token_count,
            const int32_t *stop_tokens,
            int stop_token_count,
            DeviceSpeculativeOutcomeHandle *out_handle) override;
        bool configureMTPRequestStopTokens(
            const std::vector<int32_t> &stop_tokens) override;
        bool configureMTPRequestPenaltyPolicy(
            const MTPRequestPenaltyPolicy &policy) override;
        bool prepareGreedyAllPositionBatchOutcomeGraph(
            int verifier_token_count,
            const int32_t *stop_tokens,
            int stop_token_count,
            const MTPRequestPenaltyPolicy &penalty_policy =
                MTPRequestPenaltyPolicy{}) override;
        bool verifyGreedyAllPositionRequestBatchOutcomesOnDeviceResident(
            const DeviceGreedyBatchOutcomeRequest *requests,
            int request_count,
            DeviceSpeculativeOutcomeHandle *out_handle) override;

        /**
         * @brief Get current position offset for a sequence
         *
         * @param seq_idx Sequence index (default 0)
         * @return Current position offset
         */
        int getPosition(int seq_idx = 0) const;

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
            /**
             * @brief Override the host-visible RoPE position rows for this graph build.
             *
             * Host positions are still used by CPU execution, graph-key metadata,
             * and legacy attention offsets.  GPU execution may additionally carry
             * a device-resident position pointer through withDevicePositionIds().
             */
            GraphBuildSession &withPositionIds(const int *position_ids);
            /**
             * @brief Override the device-resident INT32 RoPE position rows.
             *
             * This pointer is consumed by GPU RoPE stages on their explicit graph
             * stream.  It does not imply ownership transfer and must outlive the
             * captured/replayed graph segment that consumes it.
             */
            GraphBuildSession &withDevicePositionIds(const void *position_ids_device);
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
            const void *explicit_position_ids_device_ = nullptr;
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
            /**
             * @brief Set host-visible RoPE position rows for this attention graph.
             */
            AttentionGraphSession &withPositionIds(const int *position_ids);
            /**
             * @brief Set device-resident INT32 RoPE position rows for GPU attention.
             */
            AttentionGraphSession &withDevicePositionIds(const void *position_ids_device);
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
            const void *position_ids_device_ = nullptr;
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
         *     .withDeviceStatePublicationStream(stream)
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
            FFNGraphSession &withDeviceStatePublicationStream(void *stream);

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
            void *device_state_publication_stream_ = nullptr;
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

        /** @brief Resolve the production worker that owns a GPU device runtime. */
        IWorkerGPUContext *getWorkerGPUContext(DeviceId device) override;
        bool workerGPUContextUsesProcessPool(DeviceId device) const override;

        /** Check whether MoE dynamic rebalancing is active for this forward domain. */
        bool isMoeRebalancingActive() const override;
        bool isMoeRebalancingGraphStableForPrefillCapture() const override;
        bool prefillGraphCaptureDisabledByHost() const override;

        /** Return the active MoE placement epoch for graph-cache keying. */
        uint64_t moePlacementEpoch() const override;
        uint64_t moeRuntimeMovementEpoch() const override;
        std::string prefillGraphDomainId() const override;
        int prefillGraphParticipantId() const override;

        /**
         * @brief Test hook for graph-stable GPU MoE movement.
         *
         * Graph-stable rebalance keeps moePlacementEpoch() pinned so captured
         * graphs do not recapture on expert movement, but prefix-cache identity
         * must still observe runtime expert movement.
         */
        void markMoERuntimeMovementForTesting() { ++moe_runtime_movement_epoch_; }

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

        bool waitForLastForwardCompletionForBenchmark() override;

        bool forwardPrefill(const int *tokens, int seq_len) override;

        /** @copydoc IInferenceRunner::forwardRestoredPrefixMTPDecodeBridge */
        bool forwardRestoredPrefixMTPDecodeBridge(
            const RestoredPrefixMTPDecodeBridgeRequest &request) override;

        /** @copydoc IInferenceRunner::servingGraphPreparationKind */
        ServingGraphPreparationKind
        servingGraphPreparationKind() const noexcept override
        {
            return state_.device_id.is_gpu()
                       ? ServingGraphPreparationKind::
                             NativeDeviceExecutableFamily
                       : ServingGraphPreparationKind::EagerHostGraph;
        }

        /** @copydoc IInferenceRunner::materializeServingGraphFamilyWithoutLaunch */
        bool materializeServingGraphFamilyWithoutLaunch(
            const ServingGraphFamilyMaterializationPlan &plan) override;

        /**
         * @brief Install the root-published request generation for sparse MoE keys.
         *
         * This value is deliberately separate from @ref session_epoch_: the
         * latter authenticates local device state and can advance during local
         * graph/cache lifecycle work, whereas this generation must remain
         * equal on independently captured overlay ranks.
         */
        bool setMoEOverlayCollectiveRequestGeneration(
            uint64_t generation_id) override;

        /**
         * @brief Bind this participant to rank-wide heterogeneous graph submission.
         *
         * The coordinator owns only retained-graph tickets and symmetric local
         * submission accounting. Placement policy and durable epochs remain
         * with the topology-selected ExpertOverlay authority. The binding is
         * installed once during setup; its participant index is stable for the
         * LocalTP cell and proves that every symmetric continuation graph
         * entered and finished the same remote transaction.
         */
        bool setMoEOverlayInferenceTransactionCoordinator(
            std::shared_ptr<MoEOverlayInferenceTransactionCoordinator>
                coordinator,
            int continuation_participant_index) override;

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
        int sampleOnDeviceAtLogicalPosition(
            const SamplingParams &params,
            int logical_position) override;
        bool requiresMPICoordinatedDecodeSampling(const SamplingParams &params) const override;
        bool publishMainLogitsBatchSamplesToDeviceResidentState(
            int request_count,
            const SamplingParams &params,
            const uint64_t *stochastic_position_seeds = nullptr) override;

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
        bool applyDeviceOwnedMTPPenaltiesToLogitRows(
            DeviceLogitsSource source,
            int row_count,
            const MTPRequestPenaltyPolicy &penalty_policy) override;
        bool applyDeviceOwnedMTPBranchPenaltiesToLogits(
            int prior_draft_count,
            const MTPRequestPenaltyPolicy &penalty_policy) override;
        bool supportsRowLocalAllPositionPenaltyApplication() const override;
        bool supportsDeviceStochasticMTPVerification() const override;
        bool buildStochasticDistributionOnDevice(
            DeviceLogitsSource source,
            int row,
            DeviceDistributionBuffer buffer,
            int slot,
            const SamplingParams &params,
            int vocab_size) override;
        bool buildStochasticDistributionsOnDevice(
            DeviceLogitsSource source,
            int first_row,
            DeviceDistributionBuffer buffer,
            int first_slot,
            int row_count,
            const SamplingParams &params,
            int vocab_size) override;
        bool buildCapturedStochasticVerifierTargetDistributions(
            int row_count,
            const SamplingParams &params,
            const MTPRequestPenaltyPolicy &penalty_policy,
            int vocab_size) override;
        bool buildStochasticProcessedLogitRowsOnDevice(
            DeviceLogitsSource source,
            int first_row,
            DeviceDistributionBuffer buffer,
            int first_slot,
            int row_count,
            const SamplingParams &params,
            int vocab_size) override;
        int sampleStochasticDraftProposalOnDevice(
            DeviceLogitsSource source,
            int row,
            int slot,
            const SamplingParams &params,
            int vocab_size,
            float threshold) override;
        bool publishCapturedMTPDraftToken(
            int row,
            int slot,
            const MTPRequestPenaltyPolicy &penalty_policy) override;
        bool sampleStochasticDraftProposalOnDeviceDeferred(
            DeviceLogitsSource source,
            int row,
            int slot,
            const SamplingParams &params,
            int vocab_size,
            float threshold) override;
        DeviceStochasticTargetSampleSlotHandle
        deviceStochasticTargetSampleProducerSlot(int slot) override;
        DeviceStochasticTargetSampleSlotHandle
        deviceStochasticTargetSampleBroadcastDestinationSlot(int slot) override;
        bool recordStochasticTargetSampleSlotReadyFromDevice(
            int slot,
            void *producer_stream,
            bool verifier_consumer_pending = true) override;
        bool stageStochasticDraftTokensForDeviceVerification(
            const int32_t *draft_tokens,
            int draft_token_count,
            int first_draft_slot = 0) override;
        bool stageStochasticTargetTokenForDeviceSampling(
            int32_t target_token,
            int target_sample_slot = 0) override;
        bool publishDeviceResidentConditionTokenToTargetSampleSlot(
            const DeviceResidentLogicalSequenceStateHandle &logical_state,
            int request_index,
            int target_sample_slot = 0) override;
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
            DeviceSpeculativeVerifyBatchOutcome *out,
            uint64_t inverse_sample_seed = 0,
            int inverse_sample_first_logical_position = 0,
            bool use_vllm_probability_rejection = false) override;
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
            DeviceSpeculativeVerifyBatchOutcome *out,
            uint64_t inverse_sample_seed = 0,
            int inverse_sample_first_logical_position = 0,
            bool use_vllm_probability_rejection = false) override;
        /**
         * @brief Reduce a logical stochastic request batch through compact GPU output rows.
         *
         * The method consumes the pending verifier stream once, queues each
         * request's row verifier and summary into a distinct arena output row,
         * then copies the compact `[request, fields]` summaries back in one
         * host-visible boundary.  This keeps request-batched decode from
         * repeatedly draining the GPU stream for every request.
         */
        bool verifyStochasticDistributionsRequestBatchOutcomesOnDevice(
            const DeviceStochasticBatchOutcomeRequest *requests,
            int request_count,
            DeviceSpeculativeVerifyBatchOutcome *outcomes) override;
        /**
         * @brief Enqueue request-batch stochastic verification without host copy.
         *
         * The returned handle references runner-owned compact output rows in the
         * activation arena.  It is valid until another stochastic outcome batch
         * is staged on this runner.
         */
        bool verifyStochasticDistributionsRequestBatchOutcomesOnDeviceResident(
            const DeviceStochasticBatchOutcomeRequest *requests,
            int request_count,
            DeviceSpeculativeOutcomeHandle *out_handle) override;
        bool beginDeviceResidentGeneration(
            const DeviceGenerationAdmissionRequest &request) override;

        /**
         * @brief Select native-parent or hosted-transaction MTP execution.
         *
         * The decision is made from the preallocated graph owner's live
         * conditional-node capabilities. CUDA must provide the requested
         * native topology. ROCm may use host-scheduled captured transactions
         * until HIP exposes equivalent conditional graph nodes; it is selected
         * before admission and is never entered after a native launch failure.
         *
         * @param topology Fixed-depth WHILE or dynamic SWITCH-in-WHILE shape.
         * @return Explicit execution policy, or `Unsupported` for a fatal
         *         incomplete backend/capture configuration.
         */
        DeviceGenerationExecutionPolicy deviceGenerationExecutionPolicy(
            DeviceGenerationLoopTopology topology) const noexcept override;
        bool materializeDeviceResidentGeneration(
            int request_count,
            int draft_depth,
            DeviceGenerationLoopTopology topology,
            DeviceGenerationSamplingMode sampling_mode) override;
        bool launchDeviceResidentGeneration() override;
        bool observeDeviceGenerationDispatchTicket(
            sampling_math::DeviceGenerationDispatchTicket *out_ticket)
            override;
        bool beginHostScheduledDeviceGenerationAdvance(
            const sampling_math::DeviceGenerationDispatchTicket &ticket,
            size_t *out_fragment_count) override;
        bool submitHostScheduledDeviceGenerationFragment(
            const sampling_math::DeviceGenerationDispatchTicket &ticket,
            size_t fragment_index) override;
        bool finishHostScheduledDeviceGenerationAdvance(
            const sampling_math::DeviceGenerationDispatchTicket &ticket)
            override;
        bool submitHostScheduledDeviceGenerationAdvance(
            const sampling_math::DeviceGenerationDispatchTicket &ticket)
            override;
        bool finishDeviceResidentGeneration(
            DeviceGenerationTerminalResult *out_result) override;

        /**
         * @brief Borrow one replay-ready MTP sidecar for device-loop composition.
         *
         * Only sidecars whose condition tokens and positions are both owned by
         * persistent device storage are addressable through this API. The
         * returned capture remains owned by the sidecar cache and may only be
         * cloned into a parent graph. Missing warmup/capture, segmentation,
         * host-authored launch metadata, and stale graph bindings are hard
         * failures reported through @p error; this method never selects a host
         * token cache or eager execution path.
         *
         * @param role Semantic sidecar role in the fixed-depth transaction.
         * @param row_count Immutable flattened row geometry of the sidecar.
         * @param error Optional diagnostic describing the first violated contract.
         * @return Borrowed monolithic graph template when every invariant holds.
         */
        std::optional<
            DeviceGraphExecutor::GraphSegmentCache::DeviceLoopGraphTemplateView>
        mtpSidecarDeviceLoopGraphTemplate(
            MTPSidecarCaptureRole role,
            int row_count,
            std::string *error = nullptr) const;

        /**
         * @brief Borrow one captured proposal publication for parent-loop composition.
         *
         * Slot identity also fixes branch depth: slot zero sees no prior draft,
         * slot one sees draft zero, and so on.  The returned graph consumes the
         * fresh sidecar logit row and writes the same persistent token slot used
         * by verifier input preparation.  Eager and segmented representations
         * are rejected.
         */
        std::optional<
            DeviceGraphExecutor::GraphSegmentCache::DeviceLoopGraphTemplateView>
        mtpDraftTokenPublicationDeviceLoopGraphTemplate(
            int slot,
            std::string *error = nullptr) const;

        /**
         * @brief Borrow the exact grouped-verifier preparation graph that just ran.
         *
         * Preparation capture is a bounded family because request count, padded
         * verifier width, and pointer topology are immutable graph identity.  This
         * API deliberately exports only the family member retained by the most
         * recent successful preparation transaction.  Looking up another member by
         * geometry would allow a stale token/KV checkpoint producer to be cloned
         * into the parent generation loop.
         *
         * @param request_count Exact logical request count of the parent loop.
         * @param padded_seq_len Exact grouped-verifier width of the parent loop.
         * @param error Optional first violated identity or capture invariant.
         * @return Borrowed monolithic capture owned by the bounded graph family.
         */
        std::optional<
            DeviceGraphExecutor::GraphSegmentCache::DeviceLoopGraphTemplateView>
        mtpVerifierPreparationDeviceLoopGraphTemplate(
            int request_count,
            int padded_seq_len,
            std::string *error = nullptr) const;

        /**
         * @brief Borrow the all-position forward paired with the exact registry key.
         *
         * This is intentionally not a general forwarding wrapper around
         * ForwardExecutionEngine. The requested preparation registry slot must
         * validate first, and its retained forward signature must consume that exact
         * request geometry and compact row count through persistent device token,
         * position, and length inputs.
         *
         * @param request_count Exact logical request count of the parent loop.
         * @param padded_seq_len Exact grouped-verifier width of the parent loop.
         * @param error Optional first violated paired-graph invariant.
         * @return Borrowed monolithic all-position verifier capture.
         */
        std::optional<ForwardExecutionEngine::DeviceLoopGraphTemplateView>
        mtpAllPositionVerifierDeviceLoopGraphTemplate(
            int request_count,
            int padded_seq_len,
            DeviceGenerationSamplingMode sampling_mode,
            std::string *error = nullptr) const;

        /**
         * @brief Borrow the accepted-row terminal-hidden publication graph.
         *
         * The request count selects one immutable launch geometry from the
         * pre-materialized device-indexed graph family. The graph reads row
         * indices derived by the compact stochastic reducer directly from the
         * MTP metadata workspace. Host-authored row vectors, stale workspace
         * generations, eager execution, and segmented capture are rejected.
         *
         * @param request_count Number of accepted terminal rows to publish.
         * @param error Optional first violated composition contract.
         * @return Borrowed monolithic capture suitable for parent-loop cloning.
         */
        std::optional<
            DeviceGraphExecutor::GraphSegmentCache::DeviceLoopGraphTemplateView>
        mtpAcceptedTerminalHiddenDeviceLoopGraphTemplate(
            int request_count,
            std::string *error = nullptr) const;

        /**
         * @brief Borrow the seeded stochastic sample-and-summary transaction.
         *
         * This fragment consumes compact target distributions and resident
         * verifier/control rows.  It contains one fused sampler/reducer launch
         * per request plus the optional device MoE maintenance-clock advance.
         * The active graph must match request count and comparison depth exactly;
         * no eager or segmented substitute is returned.
         */
        std::optional<
            DeviceGraphExecutor::GraphSegmentCache::DeviceLoopGraphTemplateView>
        mtpStochasticSerialOutcomeDeviceLoopGraphTemplate(
            int request_count,
            int comparison_rows_per_request,
            std::string *error = nullptr) const;

        /**
         * @brief Borrow one captured ExpertOverlay reader boundary for composition.
         *
         * The returned acquire/release graph is the same executable used by the
         * externally orchestrated first transaction. Native CUDA and hosted HIP
         * generation loops clone it as the first/last unconditional child of
         * every transaction branch, so maintenance can publish only after the
         * complete main/MTP reader family has drained.
         *
         * @param operation Acquire or release edge to export.
         * @param error Optional first violated capture/lifetime invariant.
         * @return Strict monolithic child capture, or empty for no overlay binding.
         */
        std::optional<
            DeviceGraphExecutor::GraphSegmentCache::DeviceLoopGraphTemplateView>
        moeOverlayEpochBoundaryDeviceLoopGraphTemplate(
            MoEOverlayEpochBoundaryStage::Operation operation,
            std::string *error = nullptr) const;

        /**
         * @brief Borrow captured penalty-plus-distribution verifier preparation.
         *
         * Only the canonical zero-based all-position row and target-slot
         * geometry is exportable. The graph owns optional device-history
         * penalties and compact target distribution construction together, so
         * parent-loop composition cannot accidentally omit or reorder either
         * half of serial-equivalent stochastic sampling.
         */
        std::optional<
            DeviceGraphExecutor::GraphSegmentCache::DeviceLoopGraphTemplateView>
        mtpStochasticTargetDistributionDeviceLoopGraphTemplate(
            int row_count,
            std::string *error = nullptr) const;

        /**
         * @brief Borrow the complete accepted-state publication transaction.
         *
         * The exported graph contains the fused response/controller commit,
         * primary and shifted KV publication, committed MoE routing history,
         * stochastic penalty-history commit, and recurrent verifier-row
         * restoration.  It contains no host lifecycle callback or replay-time
         * launch preparation and is therefore suitable for cloning immediately
         * before the accepted terminal-hidden selector in a parent device loop.
         *
         * @param request_count Immutable request geometry represented by capture.
         * @param verifier_rows_per_request Immutable grouped-verifier row width.
         * @param error Optional first violated composition invariant.
         * @return Borrowed monolithic capture when the active cache is exact.
         */
        std::optional<
            DeviceGraphExecutor::GraphSegmentCache::DeviceLoopGraphTemplateView>
        mtpSpeculativeStatePublicationDeviceLoopGraphTemplate(
            int request_count,
            int verifier_rows_per_request,
            std::string *error = nullptr) const;
        /**
         * @brief Republish this runner's compact outcome after rank collection.
         *
         * This is deliberately runner-owned because only the per-device runner
         * owns the backend event and exact stream stored in the handle.
         */
        /**
         * @brief Diagnostic-only host probe for a device-resident outcome.
         */
        bool copyDeviceSpeculativeOutcomesToHostForDiagnostics(
            const DeviceSpeculativeOutcomeHandle &handle,
            DeviceSpeculativeVerifyBatchOutcome *outcomes) override;
        /**
         * @brief Get logits (IInferenceRunner override - already declared above)
         */
        // const float *logits() const; - declared above

        /**
         * @brief Get vocabulary size (IInferenceRunner override)
         */
        int vocab_size() const override { return graph_builder_ ? graph_builder_->config().vocab_size : 0; }

        /**
         * @brief Policy for clearing deferred sampled-token readiness markers.
         *
         * Verifier-owned sample slots are part of the active MTP transaction:
         * a GPU sidecar, verifier input materializer, and outcome reducer may
         * all consume the same device token without a host read. Generic
         * stream-handoff cleanup must preserve those slots, while request
         * resets, new samples, and final outcome consumption force-clear them.
         */
        enum class StochasticSampleReadyClearMode
        {
            PreserveVerifierConsumer,
            Force,
        };

        /**
         * @brief Scope guard for one request-state reset publication.
         *
         * The transaction captures exactly one execution stream and exposes it
         * through the IKVCache reset context.  The same stream first consumes
         * every old-request producer, then owns KV/GDN/MTP mutation, and finally
         * records RequestStateResetReady.  Destroying an armed transaction
         * without that publication is fatal because the next captured graph
         * would otherwise have no valid dependency to consume.
         */
        class RequestStateResetTransaction final
        {
        public:
            /**
             * @brief Ordered phases of one device-owned request reset.
             *
             * The phase is diagnostic state only; device ordering remains
             * expressed by the explicit stream and timeline events. Keeping
             * this typed list beside the transaction makes every reset owner
             * visible when a fatal exception crosses the boundary.
             */
            enum class Phase : uint8_t
            {
                Constructed,
                DrainMaintenanceDiagnostics,
                JoinPriorProducers,
                ReleaseOverlayEpoch,
                ResetReplaySessions,
                ResetMaintenanceRequestState,
                ResetMaintenanceGraph,
                ClearDeferredPublications,
                ResetCommittedKVAndGDN,
                ResetShiftedMTP,
                ResetMTPHistory,
                ResetLogicalSequence,
                ResetKernelDynamicState,
                ResetModelRuntime,
                PublishResetReady,
                FinalizeSessionMetadata,
                Completed,
            };

            RequestStateResetTransaction(
                IKVCache::StateResetBoundary boundary,
                void *execution_stream,
                const char *reason,
                bool publication_required)
                : context_{
                      .boundary = boundary,
                      .execution_stream = execution_stream,
                      .reason = reason,
                  },
                  publication_required_(publication_required)
            {
                if (!context_.permitsRequestReset() ||
                    !context_.hasReason() ||
                    (publication_required_ && !context_.execution_stream))
                {
                    throw std::invalid_argument(
                        "RequestStateResetTransaction requires a named boundary "
                        "and an explicit GPU stream");
                }
            }

            /**
             * @brief Enforce publication and preserve the original fatal diagnostic.
             *
             * An exception raised while resetting one of the device-owned state
             * participants unwinds through this guard before it reaches the
             * request handler.  Calling `std::terminate()` without first naming
             * that exception hides the operation that violated the transaction.
             * The reset remains unconditionally fatal, but the active exception
             * is surfaced before termination so production E2E failures identify
             * the actual owner instead of reporting only an unpublished event.
             */
            ~RequestStateResetTransaction() noexcept
            {
                if (publication_required_ && (!published_ || !completed_))
                {
                    const int active_exceptions = std::uncaught_exceptions();
                    if (active_exceptions > 0)
                    {
                        try
                        {
                            const std::exception_ptr active =
                                std::current_exception();
                            if (active)
                                std::rethrow_exception(active);
                            std::fprintf(
                                stderr,
                                "[FATAL] GPU request-state reset aborted before "
                                "transaction completion: phase=%s published=%s "
                                "exception=<active exception unavailable outside "
                                "catch> active_exceptions=%d\n",
                                phaseName(phase_),
                                published_ ? "true" : "false",
                                active_exceptions);
                        }
                        catch (const std::exception &error)
                        {
                            std::fprintf(
                                stderr,
                                "[FATAL] GPU request-state reset aborted before "
                                "transaction completion: phase=%s published=%s "
                                "exception=%s "
                                "active_exceptions=%d\n",
                                phaseName(phase_),
                                published_ ? "true" : "false",
                                error.what(),
                                active_exceptions);
                        }
                        catch (...)
                        {
                            std::fprintf(
                                stderr,
                                "[FATAL] GPU request-state reset aborted before "
                                "transaction completion: phase=%s published=%s "
                                "exception=<non-standard> "
                                "active_exceptions=%d\n",
                                phaseName(phase_),
                                published_ ? "true" : "false",
                                active_exceptions);
                        }
                    }
                    else
                    {
                        std::fprintf(
                            stderr,
                            "[FATAL] GPU request-state reset exited without "
                            "transaction completion: phase=%s published=%s\n",
                            phaseName(phase_),
                            published_ ? "true" : "false");
                    }
                    std::fflush(stderr);
                    std::terminate();
                }
            }

            RequestStateResetTransaction(
                const RequestStateResetTransaction &) = delete;
            RequestStateResetTransaction &operator=(
                const RequestStateResetTransaction &) = delete;
            RequestStateResetTransaction(
                RequestStateResetTransaction &&) = delete;
            RequestStateResetTransaction &operator=(
                RequestStateResetTransaction &&) = delete;

            const IKVCache::StateResetContext &cacheContext() const
            {
                return context_;
            }

            void *executionStream() const
            {
                return context_.execution_stream;
            }

            void markPublished()
            {
                if (!publication_required_ || published_)
                    std::terminate();
                published_ = true;
            }

            /**
             * @brief Enter the next named reset phase.
             *
             * @param phase Phase whose work is about to begin.
             */
            void enter(Phase phase)
            {
                if (completed_ || phase == Phase::Constructed)
                    std::terminate();
                phase_ = phase;
            }

            /**
             * @brief Commit the complete reset transaction.
             *
             * Completion is distinct from event publication because graph and
             * model-runtime metadata still cross the same request boundary
             * after device zeroing has been published.
             */
            void markCompleted()
            {
                if (completed_ ||
                    (publication_required_ && !published_))
                {
                    std::terminate();
                }
                phase_ = Phase::Completed;
                completed_ = true;
            }

        private:
            static const char *phaseName(Phase phase)
            {
                switch (phase)
                {
                case Phase::Constructed:
                    return "constructed";
                case Phase::DrainMaintenanceDiagnostics:
                    return "drain_maintenance_diagnostics";
                case Phase::JoinPriorProducers:
                    return "join_prior_producers";
                case Phase::ReleaseOverlayEpoch:
                    return "release_overlay_epoch";
                case Phase::ResetReplaySessions:
                    return "reset_replay_sessions";
                case Phase::ResetMaintenanceRequestState:
                    return "reset_maintenance_request_state";
                case Phase::ResetMaintenanceGraph:
                    return "reset_maintenance_graph";
                case Phase::ClearDeferredPublications:
                    return "clear_deferred_publications";
                case Phase::ResetCommittedKVAndGDN:
                    return "reset_committed_kv_and_gdn";
                case Phase::ResetShiftedMTP:
                    return "reset_shifted_mtp";
                case Phase::ResetMTPHistory:
                    return "reset_mtp_history";
                case Phase::ResetLogicalSequence:
                    return "reset_logical_sequence";
                case Phase::PublishResetReady:
                    return "publish_reset_ready";
                case Phase::ResetKernelDynamicState:
                    return "reset_kernel_dynamic_state";
                case Phase::ResetModelRuntime:
                    return "reset_model_runtime";
                case Phase::FinalizeSessionMetadata:
                    return "finalize_session_metadata";
                case Phase::Completed:
                    return "completed";
                }
                return "unknown";
            }

            IKVCache::StateResetContext context_;
            bool publication_required_ = false;
            bool published_ = false;
            bool completed_ = false;
            Phase phase_ = Phase::Constructed;
        };

        /**
         * @brief IInferenceRunner request-boundary reset.
         *
         * Resets live sequence state (KV cache, positions, model recurrence,
         * pending stream handoffs, resident mailboxes, and request-local stage
         * metadata) while preserving cached ComputeGraphs, BufferArena
         * bindings, prepared weights, and proven replay-safe kernel dynamic
         * objects.  Captured GPU graphs may hold argument pointers into
         * kernel-owned dynamic buffers, so this path must not globally wipe
         * kernel dynamic state while it keeps those graph executables alive.
         *
         * Replay-safe GPU graphs, including prefill, single-token decode,
         * all-position verifier, and MTP sidecars, keep their captured
         * executables. Their stages reset request metadata and rebind to
         * explicit streams before the next launch; state content changes behind
         * stable addresses are never treated as a reason to recapture.
         */
        void resetInferenceState(const InferenceStateResetRequest &request) override
        {
            const bool request_boundary =
                request.boundary == InferenceStateResetRequest::Boundary::Request;
            const bool prefix_restore_boundary =
                request.boundary == InferenceStateResetRequest::Boundary::PrefixRestore;
            const bool serving_graph_setup_boundary =
                request.boundary ==
                InferenceStateResetRequest::Boundary::ServingGraphSetup;
            if (!request.resetsAllLiveRequestOwners())
            {
                throw std::invalid_argument(
                    "DeviceGraphOrchestrator::resetInferenceState requires KV, GDN, MTP, "
                    "and logical sequence owners to cross the reset boundary together");
            }
            if (request_boundary || serving_graph_setup_boundary)
            {
                if (!request.reset_model_runtime || !request.preserve_replay_safe_graphs)
                {
                    throw std::invalid_argument(
                        "DeviceGraphOrchestrator::resetInferenceState request/setup boundary must "
                        "reset model runtime state and preserve replay-safe graph captures");
                }
            }
            else if (prefix_restore_boundary)
            {
                if (!request.preserve_replay_safe_graphs)
                {
                    throw std::invalid_argument(
                        "DeviceGraphOrchestrator::resetInferenceState prefix restore must "
                        "preserve replay-safe graph captures while importing cached state");
                }
            }
            else
            {
                throw std::invalid_argument(
                    "DeviceGraphOrchestrator::resetInferenceState does not yet implement "
                    "the requested hard reset boundary");
            }
            const char *reset_reason = request.reason ? request.reason : "request-boundary";
            const bool preserve_replay_safe_graphs = request.preserve_replay_safe_graphs;
            void *const reset_stream =
                state_.device_id.is_gpu()
                    ? explicitGPUStreamForOperation(
                          "requestStateResetTransaction")
                    : nullptr;
            RequestStateResetTransaction reset_transaction(
                prefix_restore_boundary
                    ? IKVCache::StateResetBoundary::PrefixReplacement
                    : IKVCache::StateResetBoundary::RequestBoundary,
                reset_stream,
                reset_reason,
                state_.device_id.is_gpu());
            reset_transaction.enter(
                RequestStateResetTransaction::Phase::
                    DrainMaintenanceDiagnostics);
            {
                /*
                 * Request reset is not a whole-device ownership boundary.
                 * Ordinary generation has already surfaced its terminal
                 * result, while graph-native MoE maintenance publishes a
                 * distinct terminal event on its own stream. Draining that
                 * exact transaction below queues the final status D2H on the
                 * maintenance stream and waits only for that readback.
                 *
                 * A former implementation synchronized every GPU here and
                 * then cleared the backend's sticky error. Besides stalling
                 * unrelated streams, that made an asynchronous device failure
                 * disappear before the owning transaction could attribute it.
                 * The exact epilogue must observe and fail on such errors.
                 */
                PerfStatsCollector::ScopedTimer reset_epilogue_timer(
                    "orchestrator",
                    "request_reset_exact_device_epilogue",
                    "request_reset",
                    state_.device_id.toString(),
                    {{"reason", reset_reason}});
                try
                {
                    drainCompletedDeviceMoERebalanceMaintenanceDiagnostics(
                        "request_reset",
                        reset_reason);
                }
                catch (const std::exception &error)
                {
                    /*
                     * The transaction destructor intentionally terminates when
                     * this phase fails, but stack unwinding cannot recover the
                     * active exception text outside a catch handler. Surface
                     * the exact maintenance invariant here, then terminate
                     * before any cache mutation can begin.
                     */
                    std::fprintf(
                        stderr,
                        "[FATAL] GPU request-state reset maintenance epilogue "
                        "failed: device=%s reason=%s error=%s\n",
                        state_.device_id.toString().c_str(),
                        reset_reason,
                        error.what());
                    std::fflush(stderr);
                    std::terminate();
                }
                catch (...)
                {
                    std::fprintf(
                        stderr,
                        "[FATAL] GPU request-state reset maintenance epilogue "
                        "failed: device=%s reason=%s "
                        "error=<non-standard exception>\n",
                        state_.device_id.toString().c_str(),
                        reset_reason);
                    std::fflush(stderr);
                    std::terminate();
                }
            }
            if (state_.device_id.is_gpu())
            {
                reset_transaction.enter(
                    RequestStateResetTransaction::Phase::
                        JoinPriorProducers);
                /*
                 * A surfaced token result does not transfer ownership of every
                 * live-state producer to the host. Accepted-state publication,
                 * verifier replay, terminal archival, and ordinary forward
                 * graphs may still have device work queued. Join all named
                 * publications onto the reset stream before cache-owned
                 * zeroing starts; the reset-ready event published below then
                 * forms one transitive edge from the old request to the first
                 * graph of the new request.
                 */
                if (!joinPriorDeviceWorkForRequestStateReset(
                        reset_transaction.executionStream(),
                        reset_reason))
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Request-state reset could not join all prior GPU producers"
                              << " reason=" << reset_reason
                              << " device=" << state_.device_id.toString());
                    std::terminate();
                }
                reset_transaction.enter(
                    RequestStateResetTransaction::Phase::
                        ReleaseOverlayEpoch);
                const auto overlay_lease_state =
                    moe_overlay_epoch_lease_lifecycle_->load();
                if (overlay_lease_state !=
                        MoEOverlayEpochLeaseState::Idle &&
                    overlay_lease_state !=
                        MoEOverlayEpochLeaseState::ExternalReader &&
                    overlay_lease_state !=
                        MoEOverlayEpochLeaseState::ReleasePublished)
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Request-state reset reached an unterminated ExpertOverlay lease"
                              << " reason=" << reset_reason
                              << " device=" << state_.device_id.toString()
                              << " lease_state="
                              << moeOverlayEpochLeaseStateName(
                                     overlay_lease_state));
                    std::terminate();
                }
                if (overlay_lease_state ==
                        MoEOverlayEpochLeaseState::ExternalReader &&
                    !releaseMoEOverlayEpochForExternalTransaction(
                        reset_transaction.executionStream(),
                        "request_state_reset"))
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Request-state reset could not release its durable ExpertOverlay reader"
                              << " reason=" << reset_reason
                              << " device=" << state_.device_id.toString());
                    std::terminate();
                }
            }
            reset_transaction.enter(
                RequestStateResetTransaction::Phase::
                    ResetReplaySessions);
            for (auto &entry : layer_graph_cache_)
            {
                entry.resetSessionState();
            }
            if (forward_engine_)
            {
                forward_engine_->resetSessionReplayState(
                    preserve_replay_safe_graphs);
            }
            if (preserve_replay_safe_graphs)
            {
                mtp_sidecar_depth0_cache_.resetSessionStatePreservingGraphReplay();
                mtp_sidecar_depth0_device_token_cache_.resetSessionStatePreservingGraphReplay();
                mtp_sidecar_depth0_chained_cache_.resetSessionStatePreservingGraphReplay();
                mtp_sidecar_depth0_chained_device_token_cache_.resetSessionStatePreservingGraphReplay();
                mtp_sidecar_depth0_kv_only_cache_.resetSessionStatePreservingGraphReplay();
                mtp_sidecar_depth0_kv_only_device_token_cache_.resetSessionStatePreservingGraphReplay();
            }
            else
            {
                mtp_sidecar_depth0_cache_.resetSessionState();
                mtp_sidecar_depth0_device_token_cache_.resetSessionState();
                mtp_sidecar_depth0_chained_cache_.resetSessionState();
                mtp_sidecar_depth0_chained_device_token_cache_.resetSessionState();
                mtp_sidecar_depth0_kv_only_cache_.resetSessionState();
                mtp_sidecar_depth0_kv_only_device_token_cache_.resetSessionState();
            }
            for (auto &cache : mtp_sidecar_depth0_kv_only_batch_caches_)
            {
                if (preserve_replay_safe_graphs)
                    cache->resetSessionStatePreservingGraphReplay();
                else
                    cache->resetSessionState();
            }
            const bool preserves_maintenance_bindings =
                preserve_replay_safe_graphs || prefix_restore_boundary;
            if (preserves_maintenance_bindings &&
                device_moe_rebalance_maintenance_graph_.graph)
            {
                reset_transaction.enter(
                    RequestStateResetTransaction::Phase::
                        ResetMaintenanceRequestState);
                if (!device_moe_rebalance_maintenance_graph_
                         .resetRequestOwnedDeviceTransaction(
                             reset_transaction.executionStream()) ||
                    session_epoch_ == std::numeric_limits<uint64_t>::max() ||
                    !initializeDeviceMoERebalanceDispatchTicketOnStream(
                        reset_transaction.executionStream(),
                        session_epoch_ + 1u))
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Device MoE maintenance transaction could not begin a fresh request"
                              << " reason=" << reset_reason
                              << " device=" << state_.device_id.toString());
                    std::terminate();
                }
            }
            reset_transaction.enter(
                RequestStateResetTransaction::Phase::
                    ResetMaintenanceGraph);
            if (preserves_maintenance_bindings)
            {
                /*
                 * Both ordinary request reset and prefix restore mutate only
                 * contents behind model-lifetime device addresses.  Prefix
                 * restore writes portable placement state into the existing
                 * runtime-table banks. Portable capture has already
                 * canonicalized every rolling transfer-slot replica to
                 * immutable model placement, so restore neither resolves nor
                 * republishes transient slot bytes. It does not replace the
                 * runtime table, directory, collective lane, transfer state,
                 * or workspace owner captured by this graph.
                 *
                 * Treating a content import as a binding-identity change used
                 * to destroy the maintenance executable once per restored
                 * request.  A maintenance wave runs only once per scheduling
                 * window, so the graph could remain forever in warmup.  Route
                 * both content-only boundaries through the typed stable-
                 * binding reset below.  Actual workspace generation changes
                 * are checked immediately before launch and explicitly force
                 * recapture; topology identity changes destroy the cache.
                 */
                device_moe_rebalance_maintenance_graph_.reset(
                    DeviceMoERebalanceMaintenanceGraphCache::ResetBoundary::
                        StableDeviceStateContents);
            }
            else
            {
                device_moe_rebalance_maintenance_graph_.reset(
                    DeviceMoERebalanceMaintenanceGraphCache::ResetBoundary::
                        BindingIdentityChanged);
            }
            reset_transaction.enter(
                RequestStateResetTransaction::Phase::
                    ClearDeferredPublications);
            /*
             * Request reset mutates contents behind model-lifetime arena and
             * workspace addresses. Preserve the terminal-hidden publication
             * graphs just like the sidecar graphs above; workspace-generation
             * or tensor-identity changes are the only valid invalidation
             * boundaries and are handled during graph-family materialization.
             */
            mtp_terminal_hidden_row_select_cache_.resetSessionState();
            mtp_terminal_hidden_rows_select_cache_.resetSessionState();
            for (auto &cache :
                 mtp_terminal_hidden_contiguous_rows_select_caches_)
            {
                if (cache)
                    cache->resetSessionState();
            }
            for (auto &cache :
                 mtp_terminal_hidden_device_accepted_rows_select_caches_)
            {
                if (cache)
                    cache->resetSessionState();
            }
            for (auto &cache :
                 mtp_terminal_hidden_request_rows_select_caches_)
            {
                if (cache)
                    cache->resetSessionState();
            }
            for (auto &cache : mtp_draft_token_publication_graphs_)
            {
                if (cache)
                    cache->resetSessionState();
            }
            mtp_speculative_state_publication_graph_.resetSessionState();
            mtp_stochastic_serial_outcome_graph_.resetSessionState();
            mtp_stochastic_target_distribution_graph_.resetSessionState();
            last_pos_offset_ = -1;
            defer_next_mtp_main_decode_sync_ = false;
            defer_all_position_verifier_sync_ = false;
            clearAllPendingLogitsStreams(reset_reason);
            clearCurrentMainLogitsPublication(reset_reason);
            std::fill(stochastic_target_distribution_streams_.begin(),
                      stochastic_target_distribution_streams_.end(),
                      nullptr);
            std::fill(stochastic_draft_distribution_streams_.begin(),
                      stochastic_draft_distribution_streams_.end(),
                      nullptr);
            std::fill(stochastic_target_row_formats_.begin(),
                      stochastic_target_row_formats_.end(),
                      StochasticRowFormat::Empty);
            std::fill(stochastic_draft_row_formats_.begin(),
                      stochastic_draft_row_formats_.end(),
                      StochasticRowFormat::Empty);
            std::fill(stochastic_target_top_k_.begin(),
                      stochastic_target_top_k_.end(),
                      0);
            std::fill(stochastic_draft_top_k_.begin(),
                      stochastic_draft_top_k_.end(),
                      0);
            clearStochasticTargetSampleReadySlots(StochasticSampleReadyClearMode::Force);
            clearStochasticDraftSampleReadySlots(StochasticSampleReadyClearMode::Force);
            clearMTPVerifierTransactionStateForBoundary(reset_reason);
            shifted_mtp_kv_ready_.valid = false;
            shifted_mtp_kv_ready_.producer_stream = nullptr;
            clearPendingAllPositionVerifierStateReady();
            clearDeviceResidentLogicalSequenceStateMailbox();
            retireDeviceResidentMTPTransaction();
            if (device_generation_storage_.active_request_count != 0 ||
                device_generation_state_ready_.valid)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Request reset reached mutation after an unjoined device-generation transaction"
                          << " active_requests="
                          << device_generation_storage_.active_request_count
                          << " ready="
                          << device_generation_state_ready_.valid);
                std::terminate();
            }
            device_generation_state_ready_.producer_stream = nullptr;
            device_generation_state_ready_.request_count = 0;
            cache_stats_ = CacheStats{};
            reset_transaction.enter(
                RequestStateResetTransaction::Phase::
                    ResetCommittedKVAndGDN);
            if ((request.reset_kv || request.reset_gdn) &&
                !state_.resetCommittedKVAndRecurrentState(
                    reset_transaction.cacheContext()))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Committed KV/GDN request reset failed"
                          << " reason=" << reset_reason);
                std::fprintf(
                    stderr,
                    "[FATAL] Committed KV/GDN request reset failed: "
                    "device=%s reason=%s\n",
                    state_.device_id.toString().c_str(),
                    reset_reason);
                std::fflush(stderr);
                std::terminate();
            }
            reset_transaction.enter(
                RequestStateResetTransaction::Phase::
                    ResetShiftedMTP);
            if (request.reset_mtp &&
                !state_.resetMTPShiftedSidecarState(
                    reset_transaction.cacheContext()))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Shifted-MTP request reset failed"
                          << " reason=" << reset_reason);
                std::terminate();
            }
            reset_transaction.enter(
                RequestStateResetTransaction::Phase::
                    ResetMTPHistory);
            if (request.reset_mtp && state_.device_id.is_gpu())
            {
                /*
                 * Sampling history is part of the same request-owned MTP
                 * transaction as shifted KV and recurrent state.  Zero it on
                 * the reset stream before publishing reset-ready; the next
                 * verifier graph then consumes one explicit transitive event
                 * instead of trusting a host-side sampler mirror.
                 */
                if (!zeroAndPublishMTPGeneratedTokenHistoryOnStream(
                        reset_transaction.executionStream(),
                        reset_reason))
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Device-owned MTP generated-token history reset failed"
                              << " reason=" << reset_reason);
                    std::terminate();
                }
            }
            reset_transaction.enter(
                RequestStateResetTransaction::Phase::
                    ResetLogicalSequence);
            if (request.reset_logical_sequence)
                state_.clearLogicalSequenceState();
            // NOTE: Do NOT reset arena_ here. Buffer registrations and allocations
            // are expensive and model-specific (e.g., GDN buffers for Qwen3.5).
            // The arena is created once in initializeBuffers() and persists for
            // the lifetime of the orchestrator.
            reset_transaction.enter(
                RequestStateResetTransaction::Phase::
                    ResetKernelDynamicState);
            if (preserve_replay_safe_graphs)
            {
                /*
                 * Do not call resetKernelDynamicState() here.  The request
                 * reset above deliberately preserves replay-safe captured
                 * CUDA/HIP graph executables, and those executable nodes can
                 * reference kernel-owned dynamic pointer tables populated
                 * during graph warmup.  Prefix restore takes the other branch:
                 * it discards replay state before importing cached KV/GDN/MTP
                 * payloads, so dynamic pointer tables can be reset safely.
                 */
                recordKernelDynamicStatePreservedForCapturedReplay(reset_reason);
            }
            else
            {
                resetKernelDynamicState();
            }
            // Reset model-internal state only when this boundary owns that
            // reset. Prefix restore with a model-runtime snapshot restores the
            // graph builder immediately after this reset instead. Prefix
            // restore without a snapshot uses a separate hook because an
            // ordinary request-boundary reset may preserve graph-compatible
            // runtime baseline state that would be stale for a restored prefix.
            reset_transaction.enter(
                RequestStateResetTransaction::Phase::
                    ResetModelRuntime);
            if (request.reset_model_runtime && graph_builder_)
            {
                if (prefix_restore_boundary)
                    graph_builder_->resetPrefixCacheRuntimeStateWithoutSnapshot(
                        reset_transaction.executionStream());
                else
                    graph_builder_->resetState(
                        reset_transaction.executionStream());
            }
            if (state_.device_id.is_gpu())
            {
                /*
                 * This is the sole publication point for the reset
                 * transaction. It follows every device mutation, including
                 * model-owned runtime tables and transfer directories, so the
                 * next graph inherits the complete boundary through one
                 * transitive event wait.
                 */
                reset_transaction.enter(
                    RequestStateResetTransaction::Phase::
                        PublishResetReady);
                publishRequestStateResetReady(
                    reset_transaction.executionStream(),
                    reset_reason);
                reset_transaction.markPublished();
            }
            reset_transaction.enter(
                RequestStateResetTransaction::Phase::
                    FinalizeSessionMetadata);
            // Note: host_resident_released_ is NOT reset here —
            // the host data is gone and cannot be re-uploaded.
            device_sampling_counter_ = 0;
            ++session_epoch_;
            recordLivePrefixSessionReset(reset_reason,
                                         preserve_replay_safe_graphs);
            reset_transaction.markCompleted();
        }

        void clear_cache() override
        {
            resetInferenceState(
                InferenceStateResetRequest::requestBoundary("clear_cache"));
        }

        void drainCompletedDecodeBoundaryMaintenanceDiagnostics() override;

        /**
         * @brief Schedule graph-captured device MoE maintenance after a committed decode step.
         *
         * A raw forward call is not a transaction boundary for MTP: grouped
         * verification still has to publish accepted recurrent/KV state or
         * roll back rejected rows.  Serving and benchmark drivers invoke this
         * hook only after that transaction has completed, so expert movement
         * cannot race verifier rollback or be omitted when MTP performs no
         * ordinary one-token main-model forward.
         */
        bool maybeApplyDecodeBoundaryMaintenance(
            uint64_t committed_tokens) override;

        /**
         * @brief Select this participant's backend-exact MoE scheduler.
         * @return Native CUDA, hosted-ticket HIP, inactive, or unsupported.
         */
        DeviceMoERebalanceMaintenanceExecutionPolicy
        deviceMoERebalanceMaintenanceExecutionPolicy()
            const noexcept override;

        /** @brief Return the setup-authenticated HIP observation cadence. */
        DeviceMoERebalanceHostedObservationSchedule
        deviceMoERebalanceHostedObservationSchedule()
            const noexcept override;

        /** @brief Return this graph's one canonical overlay runtime source. */
        std::vector<MoEOverlayDeviceControllerRuntimeBinding>
        moeOverlayDeviceControllerRuntimeBindings() const override;

        /**
         * @brief Join committed inference and release its overlay reader.
         *
         * The topology-wide controller calls this only on this device's worker
         * with its exact dedicated maintenance stream.  It queues event edges
         * and the retained epoch-release graph, but performs no host or device
         * synchronization.
         */
        MoEOverlayInferenceBoundaryStatus
        enqueueMoEOverlayDeviceInferenceBoundary(
            void *maintenance_stream,
            MoEOverlayInferenceBoundaryRequest request) override;

        /** @copydoc IMoEOverlayDeviceInferenceBoundary::installMoEOverlayTransferProgressEpoch */
        [[nodiscard]] bool installMoEOverlayTransferProgressEpoch(
            std::shared_ptr<MappedTransferProgressEpoch> epoch) override;

        /** @brief Validate one graph-embedded, D2H-free boundary known not to be due. */
        bool submitHostScheduledDeviceMoERebalanceKnownNonDueBoundary()
            override;

        /**
         * @brief Publish and authenticate one HIP cadence decision.
         * @param out_ticket Non-null host destination for the immutable ticket.
         * @return true only after the exact ticket-ready event has completed
         *         and the snapshot matches this request, arena, and participant.
         */
        bool observeDeviceMoERebalanceDispatchTicket(
            DeviceMoERebalanceDispatchTicket *out_ticket) override;

        /**
         * @brief Submit the captured branch selected by a validated rank ticket.
         * @param ticket Rank-validated decision for this exact participant.
         * @return true only when maintenance or acknowledgement was enqueued.
         */
        bool submitHostScheduledDeviceMoERebalanceMaintenance(
            const DeviceMoERebalanceDispatchTicket &ticket) override;

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

        /**
         * @brief Expose the latest resident logical-state mailbox, if current.
         *
         * This is the structured device-side counterpart to get_position() and
         * sequence_lengths().  It stays invalid until a verifier outcome has
         * produced resident publication metadata on an explicit stream.
        */
        DeviceResidentLogicalSequenceStateHandle deviceResidentLogicalSequenceState() const override;

        PrefixLookupResult lookupPrefix(const std::vector<int32_t> &tokens) override;
        bool populatePrefix(const PrefixLookupResult &hit, int seq_idx = 0) override;
        bool harvestPrefix(const std::vector<int32_t> &tokens, int prompt_token_count) override;
        bool restorePrefixTerminalState(const PrefixLookupResult &hit) override;
        PrefixStateSnapshot captureLivePrefixState(int seq_idx = 0) const override;
        PrefixStateSnapshot captureLivePrefixCheckpoint(
            const PrefixCheckpointCaptureRequest &request) const override;
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

        /**
         * @brief Enable graph-stable diagnostic snapshots for subsequent forwards.
         * @param output_dir Retained for the runner interface; snapshots are held in memory.
         *
         * Repeating this call clears request-local values without changing the
         * executor's snapshot topology epoch. Captured graphs therefore remain
         * reusable when a parity caller restates an unchanged diagnostics policy.
         */
        void enableSnapshotCapture(const std::string &output_dir = "") override
        {
            (void)output_dir;
            snapshot_capture_.clear();
            prefill_chunk_snapshot_previous_context_.reset();
            prefill_chunk_snapshot_sequence_.clear();

            /*
             * Enabling an already-enabled sink clears its request-local values,
             * but it does not change the set of D2D snapshot nodes embedded in
             * a captured graph.  Reinstalling an equivalent std::function would
             * advance DeviceGraphExecutor's topology epoch and needlessly throw
             * away a proven decode executable before every parity token.
             */
            if (snapshot_enabled_)
                return;

            snapshot_enabled_ = true;

            LOG_DEBUG("[DeviceGraphOrchestrator::enableSnapshotCapture] Setting callback on executor_");
            applySnapshotCaptureFilter();
            executor_.setSnapshotCallback(
                [this](const std::string &name, const StageDumpInfo &dump)
                {
                    if (!snapshot_context_.empty())
                        snapshot_capture_.captureStage(snapshot_context_ + "::" + name, dump);
                    snapshot_capture_.captureStage(name, dump);
                });
        }

        /**
         * @brief Select the stage keys copied into graph-stable snapshot slots.
         * @param keys Snapshot keys; order, duplicates, and empty entries have no meaning.
         *
         * A semantic filter change invalidates captured graph topology exactly
         * once. Reapplying an equivalent set preserves the current executable.
         */
        void setSnapshotCaptureFilter(const std::vector<std::string> &keys) override
        {
            std::unordered_set<std::string> requested_filter;
            requested_filter.reserve(keys.size());
            for (const auto &key : keys)
            {
                if (!key.empty())
                    requested_filter.insert(key);
            }

            /*
             * Filter order and duplicates have no graph meaning.  Compare the
             * normalized semantic set before touching the executor so a caller
             * may restate its diagnostics policy on every request without
             * turning steady replay into perpetual recapture.
             */
            if (requested_filter == snapshot_capture_filter_)
                return;

            snapshot_capture_filter_ = std::move(requested_filter);
            if (!snapshot_enabled_)
                return;
            applySnapshotCaptureFilter();
        }

        /**
         * @brief Disable diagnostic snapshots and remove their captured graph nodes.
         *
         * The first transition advances the executor topology epoch; repeating
         * the disabled state is an idempotent request-local cleanup operation.
         */
        void disableSnapshotCapture() override
        {
            if (!snapshot_enabled_)
            {
                snapshot_capture_.clear();
                snapshot_capture_filter_.clear();
                prefill_chunk_snapshot_previous_context_.reset();
                prefill_chunk_snapshot_sequence_.clear();
                return;
            }

            snapshot_enabled_ = false;
            snapshot_capture_.clear();
            snapshot_capture_filter_.clear();
            prefill_chunk_snapshot_previous_context_.reset();
            prefill_chunk_snapshot_sequence_.clear();
            executor_.setSnapshotStageFilter(nullptr);
            executor_.setSnapshotCallback(nullptr);
        }

        /**
         * @brief Clear published snapshot values without changing graph topology.
         */
        void clearSnapshots() override
        {
            snapshot_capture_.clear();
            prefill_chunk_snapshot_previous_context_.reset();
            prefill_chunk_snapshot_sequence_.clear();
        }

        const float *getSnapshot(const std::string &key, size_t &out_size) const override
        {
            const StoredSnapshotHandle snap = snapshot_capture_.getShared(key);
            if (!snap)
            {
                LOG_DEBUG("[DeviceGraphOrchestrator::getSnapshot] Key NOT FOUND: " << key
                                                                                   << " (have " << snapshot_capture_.size() << " snapshots)");
                out_size = 0;
                return nullptr;
            }
            out_size = snap->data.size();
            LOG_TRACE("[DeviceGraphOrchestrator::getSnapshot] Key found: " << key << " size=" << out_size);
            return snap->data.data();
        }

        SnapshotInfo getSnapshotWithShape(const std::string &key) const override
        {
            const StoredSnapshotHandle snap = snapshot_capture_.getShared(key);
            if (!snap)
                return {};
            return SnapshotInfo{
                .data = snap->data.data(),
                .size = snap->data.size(),
                .rows = snap->rows,
                .cols = snap->cols,
                /*
                 * Keep the immutable publication alive after this lookup
                 * returns. Rank/global parity routers intentionally copy the
                 * FP32 payload outside SnapshotCapture's map lock, while a
                 * later graph callback may replace or clear that map entry.
                 */
                .lifetime_owner = std::move(snap),
            };
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

        /**
         * @brief Match requested semantic keys against one concrete stage output.
         *
         * Stage names alone are insufficient for fused and policy-selected MoE
         * producers. The output descriptor makes the real producer explicit and
         * prevents graph capture from allocating a slot for stale tensor state.
         */
        bool snapshotStageMatchesFilter(
            const std::string &stage_name,
            const StageDumpInfo &dump_info) const
        {
            if (snapshot_capture_filter_.empty())
                return true;
            for (const auto &key :
                 SnapshotCapture::possibleKeysForStage(stage_name, dump_info))
            {
                if (snapshot_capture_filter_.count(key) > 0)
                    return true;
            }
            return false;
        }

        void applySnapshotCaptureFilter()
        {
            if (snapshot_capture_filter_.empty())
            {
                executor_.setSnapshotStageFilter(nullptr);
                return;
            }
            executor_.setSnapshotStageFilter(
                [this](const std::string &stage_name,
                       const StageDumpInfo &dump_info)
                {
                    return snapshotStageMatchesFilter(stage_name, dump_info);
                });
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
         * @brief Backend-owned pinned host scratch for compact stochastic outcomes.
         *
         * GPU D2H copies are only truly asynchronous when the host destination is
         * pinned/page-locked. This helper owns the small token and metadata arrays
         * used by the host-visible compatibility bridge so decode does not allocate
         * or pageable-stage these buffers on every MTP step.
         */
        struct PinnedHostScratch;

        /**
         * @brief Fixed pinned destination for the narrow HIP dispatch ticket.
         *
         * This allocation is distinct from compact-outcome and terminal-result
         * scratch so the hosted graph scheduler cannot accidentally observe a
         * broader inference payload while deciding which captured graph follows.
         */
        struct PinnedDispatchTicketScratch;

        /**
         * @brief Fixed pinned destination for HIP MoE cadence decisions.
         *
         * This allocation is one ticket wide and exists only to let the rank
         * compare participant predicates before submitting an RCCL graph. It is
         * never a source buffer and exposes no routing or expert payload state.
         */
        struct PinnedMoERebalanceDispatchTicketScratch;

        /**
         * @brief Remote retained-graph width for one exact-shape CPU prefill.
         *
         * A CPU continuation executes only logical rows, while an expert-only
         * GPU follower may replay a padded retained bucket.  This descriptor
         * keeps that authenticated remote width distinct from `seq_len`, which
         * remains the local arithmetic shape.  It is present only for an
         * explicitly scheduled prefill chunk.
         */
        struct RemotePrefillTransactionGeometry
        {
            int physical_rows_per_request = 0;
        };

        /**
         * @brief Shared implementation for host-token and device-token forwards.
         *
         * `tokens` is always the host shadow used for request bookkeeping. When
         * `token_ids_device` is non-null, graph embedding stages read token IDs
         * from that stable device buffer instead of uploading from `tokens`.
         * Device position and sequence-length overrides travel together for
         * resident request batches: recurrent stages need both rows to select
         * the correct request-owned live-state bank without observing a host
         * length mirror. `request_real_lengths` is the immutable logical row
         * geometry for this transaction. It is deliberately separate from
         * `state_.sequence_lengths`, whose values are mutable request progress
         * and therefore cannot safely serve as asynchronous admission input.
         * `execution_role` is mandatory because graph shape does not distinguish
         * prefill, grouped verification, and request-batched decode.
         * `invocation` is the single typed phase/state policy; in particular,
         * RestoredPrefixMTPDecodeBridge selects decode math and the additional
         * shifted-MTP transaction without an invalid pair of force booleans.
         * `remote_prefill_geometry` is supplied only when an eager CPU graph
         * executes exact rows while its retained remote follower uses the
         * scheduler's larger physical bucket.
         */
        const float *forwardImpl(
            const int *tokens,
            const void *token_ids_device,
            int seq_len,
            int batch_size,
            ForwardExecutionRole execution_role,
            ForwardInvocationKind invocation =
                ForwardInvocationKind::Automatic,
            const void *position_ids_device_override = nullptr,
            const int32_t *sequence_lengths_device_override = nullptr,
            std::span<const int> request_real_lengths = {},
            std::optional<RemotePrefillTransactionGeometry>
                remote_prefill_geometry = std::nullopt);

        /**
         * @brief Execute a host-token rectangular batch under an explicit role.
         *
         * Ordinary request prefill and CPU grouped verification share padding,
         * logical-length, and request-progress mechanics, but they must never
         * infer semantic graph ownership from those dimensions. This helper is
         * the single place that flattens host rows and forwards the caller's
         * typed role into graph construction.
         *
         * @param token_batches Logical host token rows.
         * @param execution_role Main inference or grouped-MTP verification.
         * @return true when the complete forward transaction succeeds.
         */
        bool forwardHostTokenBatchImpl(
            const std::vector<std::vector<int>> &token_batches,
            ForwardExecutionRole execution_role);

        size_t localLogitsVocabColumns(const TensorBase *tensor) const;
        size_t localLogitsRowStrideColumns(const TensorBase *tensor) const;
        int localLogitsVocabOffset() const;
        bool activeMainLogitsAreColumnParallel() const;
        bool mtpSidecarLogitsAreColumnParallel() const;

        /**
         * @brief Decide whether one MTP sidecar graph owns a logits allgather.
         * @param kv_cache_only True for the shifted-prefill cache-population graph.
         * @return True only for a full sidecar whose participant output is a
         *         vocabulary shard inside a multi-rank GlobalTP domain.
         *
         * The decision deliberately delegates participant output ownership to
         * `GraphConfig::mtpTerminalLogitsLayout()`. This prevents orchestration
         * from reconstructing terminal-head placement from the primary LM-head
         * sharding bit and accidentally gathering a mirrored full-vocabulary
         * result.
         */
        bool mtpSidecarRequiresGlobalLogitsGather(bool kv_cache_only) const;

        bool allPositionVerifierGraphWritesLocalLogits(int graph_token_count = -1) const;
        bool activeAllPositionLogitsAreColumnParallel(int graph_token_count = -1) const;

        /**
         * @brief Return the preplanned MTP logits capacity for a typed main graph.
         *
         * The draft sidecar has already been consumed before grouped verifier
         * or condition execution starts. Its stable maximum-row logits
         * allocation can therefore become the main graph's output owner without
         * overlap. The graph and every sampler use the invocation's explicit row
         * count; the tensor's first dimension is capacity. This keeps one pointer
         * valid for M-total capture/replay and removes per-row-count allocations
         * from the GPU hot path.
         *
         * This is an output-storage query, not an input-coherence query. The
         * allocation is intentionally UNINITIALIZED before its producer graph
         * first runs, so eligibility requires a stable device-local allocation
         * but does not require deviceValid(). Graph execution publishes device
         * authority only after the exact producer stream has written the rows.
         *
         * @param execution_role Semantic role of the current forward transaction.
         * @param buffer_id Sidecar logits allocation to inspect.
         * @param rows Minimum required output row capacity.
         * @param columns Required output column count.
         * @return Shared owner of the preplanned device-local allocation, or empty.
         */
        std::shared_ptr<TensorBase> preplannedMTPLogitsBuffer(
            ForwardExecutionRole execution_role,
            BufferId buffer_id,
            size_t rows,
            size_t columns) const;

        /**
         * @brief Bind the sole all-position logits surface for a graph role.
         *
         * GPU MTP condition and grouped-verifier graphs must reuse the
         * schema-owned maximum-row logits family allocated before capture.
         * CPU and non-MTP diagnostic graphs may retain an exact row-count
         * owner.  This method performs both selection and arena publication so
         * setup-only materialization and live execution cannot construct
         * different output topologies.
         *
         * @param execution_role Typed producer role for the graph.
         * @param rows Number of logical logits rows written by the graph.
         * @param logits_output Receives the full-vocabulary output owner.
         * @param logits_local_output Receives the vocabulary-local owner when
         *        the graph writes a sharded LM-head result.
         * @return true when every required stable owner is allocated and bound.
         */
        bool bindAllPositionLogitsOutputs(
            ForwardExecutionRole execution_role,
            size_t rows,
            TensorBase *&logits_output,
            TensorBase *&logits_local_output);

        /**
         * @brief Return whether stable epoch tickets decouple placement from graph identity.
         *
         * ExpertOverlay mutates model-lifetime runtime banks and publishes an
         * epoch selector. Captured graphs retain those stable addresses, so
         * neither host-resident nor device-resident publication requires
         * graph recapture.
         */
        bool usesGraphStableMoEOverlayResidency() const;

        /**
         * @brief Return whether this participant runs the homogeneous device backend.
         *
         * The predicate is derived exclusively from the frozen graph topology
         * and sole-authority contract. Hot-cache capacity and diagnostic
         * environment settings must never change publication ownership.
         */
        bool usesDeviceResidentMoEOverlayAuthority() const;

        /**
         * @brief Return whether maintenance belongs to the mapped topology authority.
         *
         * A mapped multi-group overlay has one topology-wide controller service.
         * Participant-local CUDA conditional graphs and ROCm ticket graphs are
         * different authorities and must remain absent in this regime.  The
         * participant contributes only its stable runtime table and exact
         * inference-boundary event interface to the topology-wide service.
         */
        bool usesTopologyWideDeviceMoEOverlayController() const noexcept;

        /**
         * @brief Return whether this participant owns an embedded maintenance clock.
         *
         * Single-domain homogeneous Dynamic execution advances one retained
         * participant-local controller from its decode and MTP graphs. A mapped
         * topology-wide controller instead admits maintenance only after the
         * orchestration layer publishes a complete inference transaction, so it
         * must never be bound as an in-transaction row budget or epoch writer.
         *
         * @return True only for native CUDA-conditional or HIP-ticket maintenance.
         */
        bool usesParticipantLocalDeviceMoERebalanceController() const noexcept;

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
         * @brief Publish graph-selected GPU logits through an exact producer event.
         *
         * ForwardOutput, rather than mutable orchestrator mode flags, names the
         * tensor produced by the graph. The tensor records a reusable completion
         * event on the exact producer stream and remains device-authoritative.
         * Explicit result access owns any later host materialization.
         *
         * @param logits Graph-declared terminal logits tensor.
         * @param ctx Device context owning the logits.
         * @param producer_stream Exact non-null logits producer stream.
         * @return true when device ownership and completion are published.
         */
        bool publishForwardResultAtBoundary(
            const ForwardOutput &output,
            IDeviceContext *ctx) override;

        /**
         * @brief Build decode-time capture policy from runtime and graph context
         */
        DeviceGraphExecutor::DecodeCapturePolicy buildDecodeCapturePolicy(
            bool has_collective_nodes,
            IDeviceContext *ctx) const override;

        /**
         * @brief Build capture policy constrained by one graph's native envelope.
         *
         * Runtime topology selects whether heterogeneous segmentation is
         * generally available, while the graph declares whether its event DAG
         * is ordinary, one indivisible device-owned transaction, or an
         * explicitly ticket-segmented heterogeneous transaction. The capture
         * controller applies that typed declaration here so initial capture and
         * retained replay cannot derive different executable identities.
         *
         * @param graph Exact graph whose native envelope constrains replay.
         * @param has_collective_nodes Whether @p graph contains collectives.
         * @param ctx Exact device context that owns capture and replay.
         * @param consumer Human-readable owner used in fatal diagnostics.
         * @return Complete immutable policy, or no value when the graph's
         *         required native execution mode is unavailable.
         */
        std::optional<DeviceGraphExecutor::DecodeCapturePolicy>
        buildDecodeCapturePolicyForGraph(
            const ComputeGraph &graph,
            bool has_collective_nodes,
            IDeviceContext *ctx,
            const char *consumer) const;

        /**
         * @brief Check whether this graph spans genuinely mixed device types
         *        in a collective execution domain.
         *
         * Segmented execution is forbidden for homogeneous CUDA-only and
         * ROCm-only domains. The proof considers LocalTP, multi-domain TP, and
         * routed ExpertOverlay participants. It is kept separate from backend
         * capability so neither an environment variable nor a backend feature
         * bit can accidentally authorize segmentation.
         */
        bool hasHeterogeneousCollectiveExecutionDomain() const;

        /**
         * @brief Check whether a proven heterogeneous collective domain can
         *        execute manual collective boundaries safely.
         */
        bool supportsHeterogeneousSegmentedReplay() const;

        /**
         * @brief Check whether this graph's homogeneous LocalTP subdomain can
         *        capture its collectives inside GPU graph segments.
         *
         * Explicit heterogeneous ExpertOverlay boundaries do not weaken this
         * capability: they remain separate manual segments while the local
         * NCCL/RCCL domain retains native collective capture.
         */
        bool collectivesSupportCapturedGraph(std::string *reason_out = nullptr) const;

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

        /**
         * @brief Restore the complete active MoE placement into a newly built graph.
         *
         * A graph-cache miss may occur after dynamic ownership and hot-replica
         * publication. Fresh stages are initially built from static model
         * placement, so they must adopt the current prepared engines and mask
         * before receiving replica metadata. Publishing only the replica set
         * would allow decode to select an expert whose local GEMM engine was
         * never made resident in that graph.
         *
         * @param graph Newly built forward graph, before it becomes executable.
         * @param publication_stream Exact stream that owns placement publication.
         */
        void applyCurrentExpertPlacementToGraph(
            ComputeGraph &graph,
            void *publication_stream);

        /** Get device contexts for all PP pipeline devices. */
        std::unordered_map<DeviceId, IDeviceContext *> getPipelineDeviceContexts() override;

        /** Resolve PP copy info for cache-miss builds. */
        PPCopyInfo resolvePPCopyInfo(const ForwardInput &input) const override;

        /** Whether forward graph construction should emit all-position logits. */
        bool computeAllPositionLogitsEnabled() const override { return compute_all_position_logits_; }

        /** Whether this forward owns one live main-model condition row per request. */
        bool liveMTPRequestBatchConditionEnabled() const override
        {
            return live_mtp_request_batch_condition_;
        }

        /** Compact verifier logits row count; 0 means full all-position logits. */
        int allPositionLogitRows() const override
        {
            return compute_row_indexed_all_position_logits_
                       ? row_indexed_all_position_logits_row_count_
                       : 0;
        }

        MTPVerifierOutcomeGraphMode
        mtpVerifierOutcomeGraphMode() const override
        {
            return mtp_verifier_outcome_graph_mode_;
        }

        /** Whether the next all-position verifier forward has explicit MTP row metadata. */
        bool mtpSpecVerifierInputPlanActive() const override
        {
            return pending_mtp_spec_verifier_input_plan_.has_value();
        }

        /** Upload any pending compact verifier row metadata before graph execution. */
        bool prepareAllPositionVerifierGraphMetadata(
            const ForwardInput &input,
            void *execution_stream,
            DeviceId execution_device) override;

        /** Queue live-state publication waits before any forward graph reads live KV/GDN state. */
        bool prepareLiveStateForForwardGraphExecution(
            const ForwardInput &input,
            void *execution_stream,
            DeviceId execution_device) override;

        /** @copydoc IForwardExecutionHost::forwardGraphAuxiliaryBranchFactory */
        GraphCaptureAuxiliaryBranchFactory
        forwardGraphAuxiliaryBranchFactory(
            const ForwardInput &input,
            DeviceId execution_device) override;

        /** Queue the cold graph-build publication onto a setup capture stream. */
        bool prepareGraphBuildStateForMaterialization(
            const ForwardInput &input,
            void *capture_stream,
            DeviceId execution_device) override;

        /** Complete the exact-stream live-state reader transaction begun by the prelude. */
        bool completeLiveStateForForwardGraphExecution(
            const ForwardInput &input,
            void *execution_stream,
            DeviceId execution_device) override;

        /** Publish any staged persistent device token row on the consuming graph stream. */
        bool prepareDeviceTokenInputsForForwardGraphExecution(
            const ForwardInput &input,
            void *execution_stream,
            DeviceId execution_device) override;

        /** Wait at an optional rank-level rendezvous immediately before graph execution. */
        bool waitBeforeForwardGraphExecution(
            const ForwardInput &input,
            DeviceId execution_device,
            bool cache_miss) override;

        /** Wait at a LocalTP-aware prefill graph-capture lifecycle boundary. */
        bool waitAtPrefillGraphCaptureBoundary(
            const ForwardInput &input,
            DeviceId execution_device,
            const std::string &boundary_name,
            void *capture_stream) override;

        /** Wait at a LocalTP-aware decode graph-capture lifecycle boundary. */
        bool waitAtDecodeGraphCaptureBoundary(
            const ForwardInput &input,
            DeviceId execution_device,
            const std::string &boundary_name,
            void *capture_stream) override;

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
         * @brief Publish one raw-written arena buffer as a graph input.
         *
         * Verifier control rows are filled through backend copy/kernel APIs
         * because their device addresses are captured permanently.  A raw
         * launch alone cannot update the arena's coherence state.  This
         * boundary proves that the raw destination is still the exact arena
         * binding and records the producer stream before the graph may read it.
         *
         * @param id Arena buffer whose device bytes were just produced.
         * @param expected_device_ptr Raw destination passed to the backend.
         * @param producer_stream Exact stream carrying the write.
         * @param device Device that owns both storage and stream.
         * @param producer Stable diagnostic name for the write transaction.
         * @return true only when identity and event publication both succeed.
         */
        bool publishPreparedArenaGraphInput(
            BufferId id,
            const void *expected_device_ptr,
            void *producer_stream,
            DeviceId device,
            const char *producer);

        /**
         * @brief Establish an empty device-owned generated-token histogram.
         *
         * Construction and request reset share this single ownership boundary:
         * enqueue the zero-fill on the exact producer stream, then publish that
         * stream's event through BufferArena before any verifier graph may read
         * the histogram.  Allocation alone is deliberately insufficient because
         * newly allocated GPU storage has no initialized device-authoritative
         * bytes.
         *
         * Kept out of the inline reset transaction so backend selection and
         * IBackend's complete type remain private to the implementation unit.
         *
         * @param producer_stream Explicit stream carrying the zero-fill.
         * @param reason Stable lifecycle reason used by fatal diagnostics.
         * @return true only after the initialized bytes and producer event have
         *         both been published to the arena.
         */
        bool zeroAndPublishMTPGeneratedTokenHistoryOnStream(
            void *producer_stream,
            const char *reason);

        /**
         * @brief Initialize the graph-stable terminal-hidden archive on device.
         *
         * The shifted-prefill stage reads and overwrites the same arena tensor
         * across graph replays. Static graph contracts therefore require valid
         * input bytes before the first capture even though a fresh request's
         * device cache counters select the branch that does not consume the
         * prior terminal row. This setup operation establishes byte validity;
         * it deliberately does not mark the archive semantically current for a
         * request.
         *
         * @param producer_stream Explicit setup stream carrying the zero-fill.
         * @param reason Stable lifecycle reason used by fatal diagnostics.
         * @return true after initialized device authority has been published.
         */
        bool initializeAndPublishMTPTerminalHiddenArchiveOnStream(
            void *producer_stream,
            const char *reason);

        /**
         * @brief Initialize persistent sample and verifier token banks before decode.
         *
         * The sample banks are range-published by independent sampler
         * operations, while the verifier row is transaction-published by its
         * preparation graph. BufferArena tracks authority at tensor
         * granularity, so setup fills every token slot with an invalid sentinel
         * and publishes each backing tensor once. This establishes stable
         * completion-event storage before the hot path; later readiness
         * publication only re-records existing events on the exact producer
         * stream.
         *
         * Per-slot readiness remains the semantic validity gate. Initializing
         * the backing tensors does not make any slot consumable.
         *
         * @param producer_stream Explicit setup stream carrying all fills.
         * @param reason Stable lifecycle reason used by fatal diagnostics.
         * @return true after every bank has initialized device authority.
         */
        bool initializeAndPublishMTPSampleBanksOnStream(
            void *producer_stream,
            const char *reason);

        /**
         * @brief Initialize the persistent response ledger and controller on device.
         *
         * Arena allocation establishes stable addresses but does not establish
         * initialized bytes or device authority.  This setup-only operation
         * fills the complete response ledger with the invalid-token sentinel,
         * clears every controller row, and publishes both writes through their
         * exact producer stream.  Performing the first publication during
         * setup also creates the persistent tensor completion events before a
         * request enters the generation hot path.
         *
         * @param producer_stream Explicit setup stream carrying both fills.
         * @param reason Stable lifecycle reason used by fatal diagnostics.
         * @return true only after both arena tensors hold initialized,
         *         event-published device authority.
         */
        bool initializeAndPublishDeviceGenerationStorageOnStream(
            void *producer_stream,
            const char *reason);

        /**
         * @brief Publish the core device-generation arena tensors as one edge.
         *
         * Every device-generation producer writes the response ledger,
         * controller, and authenticated dispatch-ticket bank as one logical
         * transaction. Keeping their arena publication behind this helper
         * prevents the private controller event and the graph executor's arena
         * frontier from observing different generations of that transaction.
         * The verifier identity row is deliberately added by
         * `publishDeviceGenerationStateReady()` only after a committed or
         * terminal transition; admission has not materialized it yet.
         *
         * @param producer_stream Exact stream carrying the complete transaction.
         * @param producer Stable producer name used by diagnostics.
         * @return true only when all core arena tensors publish the exact stream.
         */
        bool publishDeviceGenerationArenaState(
            void *producer_stream,
            const char *producer);

        /**
         * @brief Enqueue the next ticket-only HIP scheduling observation.
         *
         * The retained publication graph, one D2H copy into persistent pinned
         * storage, and one exact completion event are submitted in stream order.
         * This method never waits and cannot access compact outcomes or mutable
         * inference payloads.
         */
        bool enqueueDeviceGenerationDispatchTicketObservation();

        /**
         * @brief Clear the device-side "sample token is ready" marker for one draft slot.
         *
         * Draft sample slots are reused across decode iterations.  Clearing
         * the marker before a new distribution build prevents later verifier or
         * sidecar stages from accidentally waiting on a previous sample's event.
         */
        void clearStochasticDraftSampleReadySlot(
            int slot,
            StochasticSampleReadyClearMode mode =
                StochasticSampleReadyClearMode::PreserveVerifierConsumer);

        /** @brief Clear every draft sample readiness marker for request reset paths. */
        void clearStochasticDraftSampleReadySlots(
            StochasticSampleReadyClearMode mode =
                StochasticSampleReadyClearMode::PreserveVerifierConsumer);

        /**
         * @brief Clear the device-side "sample token is ready" marker for one target slot.
         *
         * Target sample slots feed the first sidecar and the stochastic batch
         * reducer.  They are deliberately separate from target distribution
         * slots: all-position verifier rows can reuse distribution slot 0
         * after the first token is sampled, but must keep this ready marker so
         * the summary kernel still waits for STOCHASTIC_TARGET_SAMPLE_TOKENS.
         * Only request-reset paths and actual target-token sampling should
         * clear this marker.
         */
        void clearStochasticTargetSampleReadySlot(
            int slot,
            StochasticSampleReadyClearMode mode =
                StochasticSampleReadyClearMode::PreserveVerifierConsumer);

        /** @brief Clear every target sample readiness marker for request reset paths. */
        void clearStochasticTargetSampleReadySlots(
            StochasticSampleReadyClearMode mode =
                StochasticSampleReadyClearMode::PreserveVerifierConsumer);

        /**
         * @brief Record that a GPU sampler has written a draft token slot.
         *
         * Host-visible and deferred stochastic paths both write the same
         * runner-owned device slot.  A host D2H scalar read is only an optional
         * observation of that slot; it must not erase the device readiness edge.
         * This helper records a backend event on the producer stream immediately
         * after the sample kernel and publishes the draft bank through
         * BufferArena on that same stream. Later GPU consumers can therefore
         * validate device authority and wait for the exact slot without using
         * the device-default stream or synchronizing the CPU.
         */
        bool recordStochasticDraftSampleReady(
            int slot,
            void *producer_stream,
            bool verifier_consumer_pending = false);

        /**
         * @brief Record that a GPU sampler has written a target token slot.
         *
         * This mirrors recordStochasticDraftSampleReady(). Target samples live
         * in a separate arena buffer so verifier row outputs can overwrite
         * STOCHASTIC_VERIFY_TOKENS without corrupting the first generated token,
         * and a host read of the first token remains independent from device
         * summary-kernel ownership. The same call also publishes tensor-level
         * arena authority on the exact target-sampler stream.
         */
        bool recordStochasticTargetSampleReady(
            int slot,
            void *producer_stream,
            bool verifier_consumer_pending = false);

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
         * @brief Wait for draft sample slots that must have a deferred producer.
         *
         * Device-token MTP paths consume sample slots without a host readback.
         * In those paths a missing ready event is a correctness bug, not a
         * synchronized fallback, because the consumer would otherwise read an
         * uninitialized or stale token from the arena slot.
         */
        bool waitForRequiredStochasticDraftSampleReadyRange(
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
         * @brief Wait for a target sample slot that must have a deferred producer.
         *
         * This is the target-token counterpart to
         * waitForRequiredStochasticDraftSampleReadyRange(). It is used by the
         * vLLM-style first-token fast lane where the MTP sidecar consumes the
         * sampled target token directly on device.
         */
        bool waitForRequiredStochasticTargetSampleReady(
            int slot,
            void *consumer_stream,
            const char *consumer_name);

        /**
         * @brief Preallocate every event used by the resident GPU MTP timeline.
         *
         * The resident verifier transaction records readiness many times per
         * generated token. Creating CUDA/HIP events at those publication sites
         * is both an allocation in the decode hot path and an ambiguous lifetime
         * boundary. This initializer runs after the fixed MTP row capacities are
         * known and creates all sample, verifier, publication, transaction,
         * logical-state, response, and profiling events up front.
         *
         * @return true when CPU execution needs no events or every GPU event
         *         required by the configured MTP capacity was created.
         */
        bool initializePersistentMTPDeviceEvents();

        /**
         * @brief Borrow one preallocated response-ready event for an outcome handle.
         *
         * Outcome handles may overlap the next transaction fence while the host
         * response bridge drains the previous compact row. The fixed pool makes
         * that overlap explicit. Exhaustion is a fatal lifecycle error for the
         * current operation; the caller must never allocate a replacement.
         */
        std::shared_ptr<void> acquirePersistentMTPOutcomeReadyEvent(
            const char *consumer_name);

        /**
         * @brief Borrow one preallocated start/stop timing pair.
         *
         * Timing events are optional when perfstats are disabled. When perfstats
         * are enabled, pool exhaustion means the configured transaction has more
         * concurrent measurements than its declared row capacity and must fail
         * loudly instead of allocating or silently dropping evidence.
         */
        bool acquirePersistentMTPGpuTimingEvents(
            const char *measurement_name,
            std::shared_ptr<void> *out_start_event,
            std::shared_ptr<void> *out_stop_event);

        /**
         * @brief Allocate the reusable shifted-MTP-KV completion event.
         *
         * KV-only sidecars publish this event on both first-use execution and
         * captured replay. Allocation during runner initialization keeps the
         * decode and shifted-prefill hot paths allocation free.
         */
        bool initializeShiftedMTPKVReadyEvent();

        /**
         * @brief Record that an asynchronous shifted-MTP-KV append has been queued.
         *
         * KV-only MTP sidecar replay can avoid a CPU stream synchronization when
         * the next operation does not need the cache immediately.  This helper
         * records a backend event on the producer stream so the next MTP-KV
         * consumer can establish a GPU-side dependency instead of racing or
         * falling back to the legacy default stream.
         */
        bool recordShiftedMTPKVReady(void *producer_stream, const char *producer_name);

        /**
         * @brief Queue a wait for any deferred shifted-MTP-KV append.
         *
         * @param consumer_stream Explicit stream that will read, append, truncate,
         *        or publish the shifted MTP KV cache.
         * @param consumer_name Short perf/log tag naming the consumer boundary.
         * @return true if no wait was needed or if the backend wait was queued.
         */
        bool waitForPendingShiftedMTPKVReady(
            void *consumer_stream,
            const char *consumer_name);

        /**
         * @brief Queue an observation-only wait for deferred shifted MTP KV writes.
         *
         * Prefix probes and checkpoint exporters must observe any shifted-cache
         * append before reading live MTP state, but they must not consume the
         * event that the next verifier, sidecar, restore, or truncate boundary
         * may still own as a semantic state transition.
         */
        bool waitForPendingShiftedMTPKVReadyForObservation(
            void *consumer_stream,
            const char *consumer_name) const;

        /**
         * @brief Record that a deferred all-position verifier finished writing rows.
         *
         * The verifier produces two different surfaces on the same replay stream:
         * logits consumed by samplers and row-local KV/GDN/short-conv state later
         * consumed by accepted-state publication. Logits use a one-shot stream
         * handoff because exactly one sampler owns them. Publication needs a
         * separate event-backed dependency because samplers may consume that
         * stream before publication restores the accepted verifier row.
         */
        bool recordAllPositionVerifierStateReady(
            void *producer_stream,
            const char *producer_name);

        /**
         * @brief Queue accepted-state publication after deferred verifier state.
         *
         * This consumes the verifier-state readiness marker. Call it before any
         * publication path reads verifier-captured KV/recurrent/hidden rows.
         */
        bool waitForPendingAllPositionVerifierStateReady(
            void *consumer_stream,
            const char *consumer_name);

        /**
         * @brief Queue an observation-only wait for deferred verifier row state.
         *
         * Snapshot/probe reads need verifier-produced KV/GDN/short-conv rows to
         * be visible, but they must not clear the event that publication still
         * owns for accepted-state restore.
         */
        bool waitForPendingAllPositionVerifierStateReadyForObservation(
            void *consumer_stream,
            const char *consumer_name) const;

        /** Clear stale verifier-state readiness after reset or disabled deferral. */
        void clearPendingAllPositionVerifierStateReady();

        /** Record that accepted spec-state publication has finished queuing live-state writes. */
        bool recordAcceptedSpecPublicationReady(void *producer_stream, const char *producer_name);

        /** Clear accepted-publication ownership when a hard restore/reset replaces live state. */
        void clearPendingAcceptedSpecPublicationReady() const;

        /**
         * @brief Record that a logical prefix checkpoint owns queued GPU payload copies.
         *
         * The event is stored both in the returned snapshot and in the runner.
         * Snapshot restore waits on the snapshot-owned event before importing
         * from device payload storage; the runner-owned copy prevents the next
         * forward/restore/truncate mutation from overwriting the live source
         * state while the asynchronous checkpoint export is still in flight.
         */
        bool recordLivePrefixCheckpointReady(
            PrefixStateSnapshot *snapshot,
            void *producer_stream,
            const char *producer_name) const;

        /**
         * @brief Queue a wait for an event-backed prefix snapshot payload.
         *
         * This is a non-consuming wait because snapshots may be restored more
         * than once by replay checkers or parity harnesses. The event is owned
         * by the snapshot until the snapshot is destroyed.
         */
        bool waitForSnapshotReady(
            const PrefixStateSnapshot &snapshot,
            void *consumer_stream,
            const char *consumer_name) const;

        /**
         * @brief Observe the previous checkpoint copy without adopting ownership.
         *
         * A later checkpoint may reuse a pooled destination after the previous
         * snapshot releases its shared storage reference. The new producer must
         * therefore wait for the old copy before overwriting that slot, while the
         * next live-state mutation still needs the same event to protect the old
         * source. This non-consuming wait satisfies both ownership edges.
         */
        bool waitForPendingLivePrefixCheckpointReadyForObservation(
            void *consumer_stream,
            const char *consumer_name) const;

        /**
         * @brief Import only a checkpoint's MTP terminal-hidden row.
         *
         * The verifier-base shifted-row repair must not restore KV, GDN,
         * positions, sampler state, or MTP sidecar KV.  It only needs the
         * terminal hidden row that was current when the checkpoint was captured.
         * This helper validates the snapshot payload, uploads it to
         * PREFIX_TERMINAL_HIDDEN on the runner's explicit stream, and marks that
         * tensor as the current sidecar input.
         */
        bool importMTPCheckpointTerminalHidden(
            const PrefixStateSnapshot &snapshot,
            void *stream,
            const char *consumer_name);

        /**
         * @brief Restore terminal hidden from a transient live-state checkpoint.
         *
         * A live checkpoint belongs to the inference transaction, rather than to
         * the durable prefix-cache hierarchy. GPU transactions therefore require
         * a device-resident source and enqueue a device-to-device copy on the
         * caller's explicit stream. CPU transactions copy between host-owned
         * buffers. This contract deliberately rejects a GPU checkpoint carrying
         * only host bytes so stale host mirrors cannot re-enter MTP rollback.
         *
         * @param handle Checkpoint block that owns the terminal-hidden payload.
         * @param stream Explicit GPU stream, or nullptr for CPU execution.
         * @param consumer_name Stable diagnostic name for counters and failures.
         * @param publish_mailbox Whether to publish the terminal-hidden mailbox
         *        event after the copy. Shifted-row repair requests publication;
         *        whole-state restore publishes its enclosing mutation event.
         * @return true when the copy was queued and ownership was published.
         */
        bool restoreLiveCheckpointTerminalHidden(
            const PrefixBlockHandle &handle,
            void *stream,
            const char *consumer_name,
            bool publish_mailbox);

        /** Consume the pending live-source checkpoint handoff before mutation. */
        bool waitForPendingLivePrefixCheckpointReady(
            void *consumer_stream,
            const char *consumer_name);

        /** Clear the runner-side checkpoint handoff after it has been consumed or invalidated. */
        void clearPendingLivePrefixCheckpointReady();

        /**
         * @brief Allocate the reusable shifted-prefill hidden-selection event.
         *
         * This is called during runner initialization, before inference can
         * enqueue a row-selection kernel.  Event creation is therefore never a
         * hot-path side effect and a later publication failure is a hard
         * backend contract violation rather than a reason to synchronize or
         * choose another execution path.
         */
        bool initializeMTPPrefillTerminalArchiveReadyEvent();

        /**
         * @brief Acquire the terminal-hidden mailbox for one GPU writer.
         *
         * `PREFIX_TERMINAL_HIDDEN` is reused by prefill archival, grouped
         * verifier publication, checkpoint import, and decode catch-up.  Every
         * GPU writer must call this method on the exact stream that produces its
         * source hidden state before replacing the mailbox payload.  The method
         * consumes any older payload publication and waits for a deferred
         * shifted-KV sidecar that may still be reading the mailbox.
         *
         * CPU writers have ordinary synchronous ownership and therefore pass
         * through without an event.
         *
         * @param writer_stream Exact GPU stream that will write the mailbox.
         * @param writer_name Stable diagnostic name for the ownership transfer.
         * @return true when the write can be queued without racing an older
         *         payload or reader.
         */
        bool beginMTPTerminalHiddenMailboxWrite(
            void *writer_stream,
            const char *writer_name);

        /**
         * @brief Publish completion of a terminal-hidden mailbox write.
         *
         * Grouped shifted prefill repeatedly selects one or more rows from the
         * main graph's hidden tensor into `PREFIX_TERMINAL_HIDDEN`, then gives
         * that stable buffer to an MTP sidecar graph. Checkpoint imports use the
         * same method after their H2D copy. Every production writer therefore
         * publishes the same event-backed handoff, so the sidecar cannot read
         * before its payload is complete and no caller can rely on host timing.
         */
        bool publishMTPTerminalHiddenMailboxReady(
            void *producer_stream,
            const char *producer_name);

        /**
         * @brief Consume the terminal-archive source-protection event before mutation.
         *
         * Forward graphs and prefix restore/truncate operations call this method
         * on their execution stream before writing either the source hidden tensor
         * or the archived terminal row.
         */
        bool waitForPendingMTPPrefillTerminalArchiveReady(
            void *consumer_stream,
            const char *consumer_name);

        /**
         * @brief Observe terminal-archive readiness without consuming ownership.
         *
         * Prefix harvest needs the archived row to be complete, but a later
         * forward remains the semantic owner that releases source protection.
         */
        bool waitForPendingMTPPrefillTerminalArchiveReadyForObservation(
            void *consumer_stream,
            const char *consumer_name) const;

        /** @brief Clear terminal-archive readiness after replacement or teardown. */
        void clearPendingMTPPrefillTerminalArchiveReady();

        /**
         * @brief Allocate and validate the live-mutation event before GPU work is queued.
         *
         * Event allocation is a recoverable preflight failure because no live
         * state or cache payload has been touched yet. Once asynchronous restore
         * reads are queued, failure to record this prepared event is an
         * unrecoverable backend contract violation; production never switches
         * to stream synchronization or another execution path.
         *
         * @param producer_stream Explicit stream that will own the mutation.
         * @param producer_name Stable diagnostic owner for error reporting.
         * @return true when an event is prepared for this exact stream.
         */
        bool prepareLivePrefixMutationReadyEvent(
            void *producer_stream,
            const char *producer_name);

        /**
         * @brief Record that a prefix restore/truncate finished queuing live-state writes.
         *
         * Payload checkpoint restore imports KV, hybrid recurrent state, and
         * terminal hidden through explicit GPU streams. Those imports are
         * asynchronous: the next graph stream must wait on this event before
         * reading live state. Keeping this as a separate handoff from accepted
         * publication makes restore semantics explicit and testable.
         *
         * `retained_payload_sources` transfers ownership of any RAM or
         * device-hot cache allocations read by the queued restore. The owners
         * remain alive until this exact event reports completion. This is not a
         * cache lease: lookup, eviction, and request execution acquire no cache
         * mutex and update no shared reference counter. Ordinary shared-pointer
         * moves plus a nonblocking event query are sufficient because the
         * completion event already exists for graph-stream ordering.
         *
         * @param producer_stream Explicit GPU stream containing the mutation.
         * @param producer_name Stable diagnostic label for perf counters.
         * @param retained_payload_sources Cache payload owners consumed by the
         *        asynchronous work already queued on `producer_stream`.
         * @return true when the completion event and ownership handoff were
         *         published successfully. A failure after retained payload
         *         reads were queued is unrecoverable and terminates the process.
         */
        bool recordLivePrefixMutationReady(
            void *producer_stream,
            const char *producer_name,
            std::vector<PrefixBlockHandle> retained_payload_sources = {},
            std::vector<std::shared_ptr<void>>
                retained_device_sources = {});

        /**
         * @brief Join every published live-state event through one typed API.
         *
         * The caller supplies only its semantic timeline role. This method
         * validates that role against the declarative device timeline, queues
         * durable event waits, and decides whether the role observes or consumes
         * each publication. There are deliberately no lower-level consuming or
         * observation entry points: callers cannot accidentally steal an event
         * from the main graph or leave an invalidated timeline live.
         *
         * @param consumer_stream Explicit stream that will launch the graph.
         * @param consumer_role Semantic owner declared in DeviceExecutionTimeline.
         * @param consumer_name Stable diagnostic label for both event joins.
         * @return true after every pending dependency is queued.
         */
        bool joinPublishedLiveStateHandoffs(
            void *consumer_stream,
            DeviceTimelineRole consumer_role,
            const char *consumer_name) const;

        /** Clear any pending prefix restore/truncate handoff. */
        void clearPendingLivePrefixMutationReady() const;

        /**
         * @brief Shared implementation for compact device-side stochastic sampling.
         *
         * The sample kernel always writes a runner-owned device slot and records
         * a readiness event for later GPU consumers. When `out_token_host` is
         * non-null the helper also performs the trusted fast D2H scalar read used
         * by legacy host-driven callers; that read does not transfer ownership
         * away from the device slot.
         */
        bool sampleStochasticDistributionOnDeviceImpl(
            DeviceDistributionBuffer buffer,
            int slot,
            float threshold,
            int32_t *out_token_host,
            uint64_t threshold_seed = 0,
            const int32_t *threshold_position_device = nullptr,
            int threshold_position_offset = 0);

        /**
         * @brief Shared vLLM-style draft proposal implementation.
         *
         * This writes the sampled draft token plus q(sampled_token). It follows
         * vLLM's default greedy draft branch, so later rejection verification
         * treats the draft distribution as one-hot at the sampled token.
         */
        bool sampleStochasticDraftProposalOnDeviceImpl(
            DeviceLogitsSource source,
            int row,
            int slot,
            const SamplingParams &params,
            int vocab_size,
            float threshold,
            int32_t *out_token_host,
            bool verifier_consumer_pending);

        /**
         * @brief Sample batched MTP logits into contiguous stochastic draft slots.
         *
         * The sidecar graph leaves logits in `MTP_LOGITS`.  This helper binds
         * the backend's device-output batched argmax to the runner-owned
         * `STOCHASTIC_DRAFT_SAMPLE_TOKENS` / `STOCHASTIC_DRAFT_SAMPLE_PROBS`
         * arena buffers and records one readiness event per slot.  The optional
         * host shadow exists only for legacy metadata assembly and must not be
         * used as verifier input.
         */
        bool sampleMTPBatchGreedyLogitsToDeviceDraftSlots(
            int request_batch,
            int first_draft_slot,
            int slot_stride,
            const char *context,
            int32_t *out_tokens);

        /**
         * @brief Shared implementation for host-token and device-token batch summaries.
         *
         * `first_token_from_device=false` preserves the legacy host-scalar
         * summary contract.  `true` reads entry 0 either from
         * STOCHASTIC_TARGET_SAMPLE_TOKENS or from the prepared verifier-token
         * matrix that fed the graph replay.  Request-batched GPU MTP uses the
         * matrix source so summary and verifier observe the exact same
         * device-resident row zero.
         */
        bool verifyStochasticDistributionsBatchOutcomeOnDeviceCommon(
            int first_target_slot,
            int first_draft_slot,
            const int32_t *draft_tokens,
            const float *accept_thresholds,
            const float *residual_thresholds,
            const float *sample_thresholds,
            int row_count,
            int32_t first_token,
            int first_target_sample_slot,
            int first_token_row_offset,
            int first_token_row_stride,
            bool first_token_from_device,
            const int32_t *stop_tokens,
            int stop_token_count,
            int bonus_target_slot,
            float bonus_threshold,
            DeviceSpeculativeVerifyBatchOutcome *out,
            uint64_t inverse_sample_seed,
            int inverse_sample_first_logical_position,
            const int32_t *threshold_base_position_device,
            int threshold_position_offset,
            bool use_vllm_probability_rejection,
            bool serial_sample_equivalent,
            const int *generation_control_device,
            int output_request_slot,
            void *stream_override,
            bool copy_summary_to_host);

        /** Build processed verifier rows without the full-softmax writeback. */
        bool buildStochasticProcessedLogitRowsOnDevice(
            DeviceLogitsSource source,
            int first_row,
            DeviceDistributionBuffer buffer,
            int first_slot,
            int row_count,
            const SamplingParams &params,
            int vocab_size,
            void *stream,
            const char *operation_name);

        /**
         * @brief Pending GPU event timing measurement drained at an existing sync boundary.
         *
         * These records let profiling distinguish enqueue cost from true device
         * work without inserting new synchronizations.  A producer queues start
         * and stop events around kernels on an explicit stream; the next
         * already-required host boundary calls drainPendingGpuTimingMeasurements().
         */
        struct PendingGpuTimingMeasurement
        {
            std::string name;
            std::map<std::string, std::string> tags;
            std::shared_ptr<void> start_event;
            std::shared_ptr<void> stop_event;
        };

        /** Queue a timing start event when perfstats are enabled on a GPU device. */
        bool beginPendingGpuTimingMeasurement(
            const std::string &name,
            void *stream,
            std::shared_ptr<void> *out_start_event,
            std::shared_ptr<void> *out_stop_event);

        /** Queue the matching stop event and remember the measurement for later drain. */
        void finishPendingGpuTimingMeasurement(
            const std::string &name,
            void *stream,
            std::map<std::string, std::string> tags,
            std::shared_ptr<void> start_event,
            std::shared_ptr<void> stop_event);

        /** Convert completed pending GPU timing events into structured perfstats. */
        void drainPendingGpuTimingMeasurements(IBackend *backend);

        /**
         * @brief Reclaim only timing measurements whose stop events are complete.
         *
         * Mirrored non-root participants do not own the final compact D2H, so
         * they cannot rely on that host boundary to drain profiler events. This
         * method queries stop events without blocking, emits timing records for
         * completed work, and leaves unfinished measurements in place. It
         * performs no allocation and is safe to call before borrowing another
         * persistent timing-event pair.
         */
        bool reclaimCompletedGpuTimingMeasurementsNonblocking(
            IBackend *backend);

        /** Monotonic live-state epoch used by versioned decode replay. */
        uint64_t liveReplayStateEpoch() const override { return live_replay_state_epoch_; }

        /** Whether the current all-position verifier may hand its stream to the sampler. */
        bool shouldDeferAllPositionVerifierFinalSync() const override;

        /** Store the verifier replay stream for the next all-position logits consumer. */
        void setPendingAllPositionVerifierStream(void *stream) override;

        /** Whether the next MTP main condition forward may hand its stream to the sampler. */
        bool shouldDeferMainDecodeFinalSync() const override;

        /** Store the main-decode replay stream for the next main-logits consumer. */
        void setPendingMainDecodeStream(void *stream) override;

        struct DeviceMoERebalanceMaintenanceGraphCache;

        /**
         * @brief Materialize the production MoE maintenance graph before workspace publication.
         *
         * Homogeneous-domain Dynamic residency maintenance is a real member of
         * the participant graph family, even though its scheduler launches it
         * only after a decode window closes. A mapped multi-group overlay
         * instead certifies the stable runtime/publication binding consumed by
         * its sole topology-wide controller; materializing another local graph
         * there would create a second placement writer. Current-batch LLEP is
         * ordinary prefill graph work and does not enter either maintenance
         * family. Eager declaration therefore proves exactly one applicable
         * ownership contract before generation one is allocated.
         *
         * @return true when maintenance is disabled/inapplicable or the retained
         *         production graph is complete and ready to join family planning.
         */
        bool materializeDeviceMoERebalanceMaintenanceGraphForFamily();

        /**
         * @brief Append HIP serial-decode cadence ownership to a main graph.
         *
         * ROCm has no native conditional graph node. Its ordinary decode graph
         * therefore ends with the allocation-free device publish/acknowledge
         * pair, while the host observes an authenticated ticket only at a
         * conservative potentially-due boundary. Grouped MTP is excluded
         * because accepted-state publication and the next verifier admission
         * already own the same device marker transaction.
         *
         * @param graph Fully built participant-local production graph.
         * @param input Typed role and mathematical phase for that graph.
         * @param error Optional construction diagnostic.
         * @return true when inapplicable or the exact terminal stage was added.
         */
        bool appendHostedDeviceMoEDecodeCommitBoundary(
            ComputeGraph &graph,
            const ForwardInput &input,
            std::string *error = nullptr);

        /**
         * @brief Enclose one topology-wide main forward in its device epoch lease.
         *
         * Ordinary prefill and decode are complete production transactions, so
         * their reader acquire must be a graph root and their release must be
         * the sole terminal after every model, collective, and auxiliary-stream
         * join.  Embedding both stages avoids launching a separate release graph
         * beside a still-running captured MoE lane.  MTP parents already own an
         * equivalent complete transaction and are deliberately excluded.
         *
         * @param graph Fully built participant-local production graph.
         * @param input Typed role and mathematical phase for that graph.
         * @param error Optional construction diagnostic.
         * @return true when inapplicable or the exact root/terminal pair exists.
         */
        bool appendCapturedMainForwardMoEOverlayEpochTransaction(
            ComputeGraph &graph,
            const ForwardInput &input,
            std::string *error = nullptr);

        /**
         * @brief Resolve the persistent device-owned MoE controller record.
         *
         * The returned value is a device address used only as a kernel
         * argument. The host never dereferences or mirrors the record.
         *
         * @param error Optional diagnostic explaining which immutable graph-
         *              family binding is incomplete.
         * @return Device address of the controller record, or nullptr when the
         *         retained maintenance participant is not fully bound.
         */
        DeviceMoERebalanceGraphControllerState *
        deviceMoERebalanceControllerStateDevice(
            std::string *error = nullptr);

        /**
         * @brief Capture the cheap serial-boundary fragments for sparse maintenance.
         *
         * CUDA owns one native conditional parent whose unconditional head
         * publishes (or recognizes an MTP-published) commit edge and whose
         * unconditional tail acknowledges a skipped non-due edge. HIP captures
         * a ticket-only publisher and a due-boundary acknowledgement; ordinary
         * non-due publication is already part of its complete decode graph.
         * This method preflights those exact sources before the expensive
         * maintenance capture is composed.
         *
         * @param gpu_ctx Persistent worker context owning graph resources.
         * @param backend Backend used to enqueue the graph-capturable kernels.
         * @param maintenance_stream Exact non-null scheduler stream.
         * @param error Optional construction diagnostic.
         * @return true when sources are ready or native composition is unsupported.
         */
        bool prepareDeviceMoERebalanceCadenceTransactionSources(
            IWorkerGPUContext *gpu_ctx,
            IBackend *backend,
            void *maintenance_stream,
            std::string *error = nullptr);

        /**
         * @brief Compose the captured maintenance graph behind its device predicate.
         *
         * The ordered transaction is `publish -> IF(due) maintenance -> ack`.
         * Source captures and every embedded controller address are persistent
         * model-lifetime bindings. The resulting graph is instantiated once and
         * reused; no host predicate read, callback, recapture, or eager fallback
         * participates in replay.
         *
         * @param error Optional construction diagnostic.
         * @return true when native composition is unavailable or fully ready.
         */
        bool composeDeviceMoERebalanceCadenceTransaction(
            std::string *error = nullptr);

        /**
         * @brief Initialize the persistent MoE ticket on a request-reset stream.
         *
         * The controller must already have been reset on @p stream. This method
         * binds its embedded ticket to the current session/workspace generation
         * and clears all host observation bookkeeping before reset readiness is
         * published.
         */
        bool initializeDeviceMoERebalanceDispatchTicketOnStream(
            void *stream,
            uint64_t ticket_session_epoch);

        /**
         * @brief Launch graph-captured device-side MoE maintenance when due.
         *
         * The caller invokes this method only after a decode transaction has
         * committed.  The graph reads readiness, routing statistics, planning
         * state, and publication state from device-resident buffers.  It must
         * therefore never depend on the legacy host position mirror, which is
         * deliberately not updated by fully device-owned grouped MTP decode.
         *
         * @return true when no launch was due or the scheduled graph completed
         *         successfully; false when graph preparation or launch failed.
         */
        bool maybeRunDeviceMoERebalanceMaintenanceGraph(
            MoEOverlayEpochMaintenanceBoundarySource boundary_source =
                MoEOverlayEpochMaintenanceBoundarySource::
                    GraphLaunchDependency);

        /**
         * @brief Order a live graph consumer after pending device-side MoE maintenance.
         *
         * The atomic maintenance graph executes on a dedicated stream so its
         * controller and transfer work can overlap unrelated device work. Any
         * graph that consumes routing histograms, runtime ownership tables, or
         * transferred expert payloads must nevertheless observe the completed
         * publication. This method queues a manifest-validated backend event
         * wait on every consumer's exact execution stream. The completion event
         * is a durable multi-consumer publication: one MTP sidecar stream can
         * never consume or retire the ordering edge owed by a main forward
         * stream, and diagnostic host observation cannot change production
         * ordering semantics.
         *
         * @param consumer_stream Explicit CUDA/HIP stream that will execute the
         *        consuming graph.
         * @param consumer_role Typed owner of the consuming graph. The role must
         *        be admitted by the centralized device execution timeline.
         * @param consumer_name Stable diagnostic name describing the consumer.
         * @return true when no maintenance is pending or all event waits were
         *         queued successfully.
         * @throws std::runtime_error if a pending publication has no exact
         *         producer event/stream/backend or the backend rejects its
         *         device-side event wait. Such a failure makes the ownership
         *         timeline unknowable and must stop inference immediately.
         */
        bool waitForPendingDeviceMoERebalanceMaintenance(
            void *consumer_stream,
            DeviceTimelineRole consumer_role,
            const char *consumer_name,
            bool acquire_overlay_epoch = true);

        /** @return Stable diagnostic name for a lease lifecycle phase. */
        static const char *moeOverlayEpochLeaseStateName(
            MoEOverlayEpochLeaseState state) noexcept;

        /**
         * @brief Complete identity of one externally submitted graph sequence.
         *
         * A valid identity exists exactly while an @c ExternalSequence lease,
         * or its final verifier's @c ExternalForward phase, is live.  Keeping
         * the expected next ordinal here makes skipped, duplicated, or crossed
         * MTP graphs unrepresentable as a successful residency transaction.
         */
        struct MoEOverlayExternalSequenceIdentity
        {
            std::uint64_t request_generation = 0; ///< Outer request generation.
            std::uint64_t sequence_id = 0; ///< Coordinator-assigned sequence identity.
            std::uint64_t placement_epoch = 0; ///< Placement pinned for every graph.
            int graph_count = 0; ///< Exact number of graphs admitted by the coordinator.
            int last_graph_ordinal = -1; ///< Last graph admitted to device execution.

            /** @return True only when every identity component is meaningful. */
            [[nodiscard]] bool valid() const noexcept
            {
                return request_generation != 0 && sequence_id != 0 &&
                       placement_epoch != 0 && graph_count > 1 &&
                       last_graph_ordinal >= 0 &&
                       last_graph_ordinal < graph_count;
            }

            /** @brief Return this object to its sole invalid/empty state. */
            void reset() noexcept
            {
                *this = {};
            }
        };

        /**
         * @brief Lexical owner for one direct MTP graph's sequence admission.
         *
         * The first draft acquires the residency epoch, intermediate drafts
         * retain it, and the grouped verifier converts it into the ordinary
         * forward terminal.  Any early return releases the epoch on the exact
         * admitted stream so a failed graph cannot strand maintenance.
         */
        class MoEOverlayExternalSequenceGraphLease final
        {
        public:
            /** @brief Construct an inactive graph lease. */
            MoEOverlayExternalSequenceGraphLease() = default;

            /** @brief Abort and release an admitted graph left unfinished. */
            ~MoEOverlayExternalSequenceGraphLease() noexcept;

            MoEOverlayExternalSequenceGraphLease(
                const MoEOverlayExternalSequenceGraphLease &) = delete;
            MoEOverlayExternalSequenceGraphLease &operator=(
                const MoEOverlayExternalSequenceGraphLease &) = delete;

            /**
             * @brief Join one coordinator binding to its device residency lease.
             * @param owner Device orchestrator owning the epoch arena.
             * @param binding Immutable coordinator graph binding.
             * @param execution_stream Exact stream that will launch the graph.
             * @param consumer_role Typed device-timeline role for diagnostics.
             * @param consumer_name Stable diagnostic owner.
             * @return True when the graph is inactive or successfully admitted.
             */
            bool admit(
                DeviceGraphOrchestrator &owner,
                const MoEOverlayInferenceParticipantGraphBinding &binding,
                void *execution_stream,
                DeviceTimelineRole consumer_role,
                const char *consumer_name);

            /**
             * @brief Publish this graph's successful or failed terminal exactly once.
             * @param execution_succeeded Whether the graph completed normally.
             * @return True when the sequence lease reached its required state.
             */
            bool finish(bool execution_succeeded) noexcept;

            /** @return Whether this object currently owes a terminal action. */
            [[nodiscard]] bool active() const noexcept
            {
                return owner_ != nullptr && !finished_;
            }

        private:
            DeviceGraphOrchestrator *owner_ = nullptr; ///< Epoch authority.
            const MoEOverlayInferenceParticipantGraphBinding *binding_ = nullptr; ///< Immutable graph identity.
            void *execution_stream_ = nullptr; ///< Exact graph stream.
            const char *consumer_name_ = nullptr; ///< Stable diagnostic label.
            bool finished_ = false; ///< True after the sole terminal action.
        };

        /**
         * @brief Resolve and validate the model-lifetime epoch binding at setup.
         *
         * This method may allocate the graph-builder-owned arena and the one
         * persistent acquired-event. It is never called from inference replay.
         * Empty bindings are an explicit non-overlay topology.
         */
        bool initializeMoEOverlayEpochExecutionBinding();

        /**
         * @brief Materialize immutable acquire and release graph definitions.
         * @return true when both definitions match the current binding.
         */
        bool materializeMoEOverlayEpochBoundaryGraphs();

        /**
         * @brief Execute one retained epoch boundary on an exact consumer stream.
         *
         * The auxiliary capture stream is joined to and from @p execution_stream
         * with backend events. No host synchronization or device-state readback
         * occurs.
         */
        bool executeMoEOverlayEpochBoundaryCaptured(
            MoEOverlayEpochBoundaryStage::Operation operation,
            void *execution_stream,
            const char *consumer_name);

        /**
         * @brief Acquire once for an externally orchestrated inference transaction.
         *
         * Later graph streams wait on the persistent acquired-event instead of
         * attempting to mutate the same request ticket twice.
         */
        bool acquireMoEOverlayEpochForExternalTransaction(
            void *consumer_stream,
            DeviceTimelineRole consumer_role,
            const char *consumer_name,
            MoEOverlayEpochLeaseState acquired_state =
                MoEOverlayEpochLeaseState::ExternalReader);

        /**
         * @brief Admit one graph into a complete direct MTP residency sequence.
         *
         * The first graph acquires once; later graphs consume the immutable
         * acquire event and must arrive in exact ordinal order with an unchanged
         * placement epoch.  No intermediate graph may release the ticket.
         */
        bool admitMoEOverlayExternalSequenceGraph(
            const MoEOverlayInferenceParticipantGraphBinding &binding,
            void *execution_stream,
            DeviceTimelineRole consumer_role,
            const char *consumer_name);

        /**
         * @brief Complete or abort one admitted direct MTP sequence graph.
         *
         * Successful draft graphs retain the lease for their successor.  A
         * failure releases immediately.  The successful final verifier is
         * completed by the ordinary forward epilogue after it publishes all
         * inference state.
         */
        bool finishMoEOverlayExternalSequenceGraph(
            const MoEOverlayInferenceParticipantGraphBinding &binding,
            void *execution_stream,
            bool execution_succeeded,
            const char *consumer_name);

        /**
         * @brief Claim the exact inference terminal for placement maintenance.
         *
         * The authenticated due-ticket path may encounter either an ambient
         * external reader or the immutable release already published by a
         * self-contained forward.  This method selects and submits exactly one
         * legal action while holding the lifecycle submission authority: close
         * the reader once, or join the existing release event.  It performs no
         * host or device synchronization.
         *
         * @param maintenance_stream Exact retained maintenance stream.
         * @param boundary_name Stable diagnostic identity for the claim.
         * @return true after the release edge is constructed or joined.
         */
        bool claimMoEOverlayEpochForPlacementMaintenance(
            void *maintenance_stream,
            const char *boundary_name);

        /**
         * @brief Submit a validated external release under existing authority.
         *
         * The caller owns @p submission for the whole event-wait, retained
         * release launch, terminal-event record, and semantic commit recipe.
         * Sharing this primitive keeps ordinary inference release and an
         * authenticated maintenance close byte-for-byte identical without
         * making the public release operation idempotent.
         *
         * @param submission Exclusive lifecycle submission authority.
         * @param release_stream Exact stream which owns the release recipe.
         * @param boundary_name Stable diagnostic identity for the release.
         * @param expected_state Exact live external owner to close.
         * @return true after publishing the immutable release receipt.
         */
        bool submitMoEOverlayEpochExternalRelease(
            MoEOverlayEpochLeaseLifecycle::Submission &submission,
            void *release_stream,
            const char *boundary_name,
            MoEOverlayEpochLeaseState expected_state);

        /**
         * @brief Release an active external reader after its complete state commit.
         *
         * The caller must first join every main/MTP producer onto @p release_stream.
         * Maintenance then follows on that same stream, making publication unable
         * to overlap an older placement reader while inference itself remains
         * asynchronous.
         */
        bool releaseMoEOverlayEpochForExternalTransaction(
            void *release_stream,
            const char *boundary_name,
            MoEOverlayEpochLeaseState expected_state =
                MoEOverlayEpochLeaseState::ExternalReader);

        /**
         * @brief Acquire the sole reader owned by one hosted MTP transaction.
         *
         * A heterogeneous hosted branch cannot clone its segmented sidecar into
         * one native parent graph. Its full sidecar therefore submits the same
         * retained acquire capture on the sidecar's exact stream and publishes
         * one durable event for every later child. The method joins the prior
         * branch release before launching the new acquire and rejects nested
         * parent or externally owned readers.
         *
         * @param acquire_stream Exact full-sidecar replay stream.
         * @param acquire_capture Retained epoch-acquire executable.
         * @param parent_name Stable diagnostic owner of the transaction.
         * @return true when the acquire and its publication event were enqueued.
         */
        bool acquireMoEOverlayEpochForHostedParent(
            void *acquire_stream,
            const IGPUGraphCapture *acquire_capture,
            const char *parent_name);

        /**
         * @brief Join a hosted parent's acquired reader at one child launch.
         *
         * Hosted sidecars and grouped verifiers replay retained executables
         * directly, so they do not pass through the ordinary forward-engine
         * live-state prelude. Every such executable must consume the durable
         * acquire event on its exact launch stream before ticket publication
         * or graph launch. Repeated children may wait on the same immutable
         * event; none of them gains authority to release the parent ticket.
         *
         * @param consumer_stream Exact retained-executable launch stream.
         * @param child_name Stable diagnostic identity for the child graph.
         * @return true when the acquire dependency was enqueued or no overlay
         *         epoch binding exists.
         */
        bool consumeMoEOverlayEpochForHostedChild(
            void *consumer_stream,
            const char *child_name);

        /**
         * @brief Close the sole reader owned by one hosted MTP transaction.
         *
         * The typed release fragment runs only after every sidecar, verifier,
         * sampler, and publication child has joined the scheduler stream. Its
         * event becomes the only legal predecessor of the next hosted acquire.
         * No child forward may call this method or clear the parent's ticket.
         *
         * @param release_stream Exact hosted scheduler stream.
         * @param release_capture Retained epoch-release executable.
         * @param parent_name Stable diagnostic owner of the transaction.
         * @return true when the release and its publication event were enqueued.
         */
        bool releaseMoEOverlayEpochForHostedParent(
            void *release_stream,
            const IGPUGraphCapture *release_capture,
            const char *parent_name);

        /**
         * @brief Close/join the host-scheduled reader before an internal parent.
         *
         * Main prefill/decode is submitted outside the resident MTP parent and
         * therefore acquires through the host lifecycle. Every parent branch
         * contains its own captured acquire/release pair. Before launching that
         * parent, the exact loop stream must either release the still-active
         * external ticket (static/no-maintenance topology) or consume the prior
         * external release event (maintenance topology). This prevents the first
         * child acquire from double-acquiring one request slot.
         *
         * @param parent_stream Exact stream that will launch the parent/branch.
         * @param parent_name Stable diagnostic owner of the boundary.
         * @return true when the parent stream is ordered after external release.
         */
        bool prepareMoEOverlayEpochForInternalParent(
            void *parent_stream,
            const char *parent_name);

        struct DeviceMoERebalanceMaintenanceOutcome
        {
            bool valid = false;
            bool useful_work = false;
            bool controller_valid = false;
            bool copy_status_present = false;
            bool copy_status_valid = false;
            bool apply_status_present = false;
            bool apply_status_valid = false;
            uint32_t status_code = 0;
            uint32_t invalid_runtime_layers = 0;
            uint32_t plan_overflow = 0;
            uint32_t capacity_limited_candidates = 0;
            uint32_t payload_bucket_overflow = 0;
            uint32_t controller_last_error_code = 0;
            uint32_t controller_error_waves = 0;
            uint32_t controller_last_error_wave_index =
                kDeviceMoEInvalidSlot;
            uint32_t controller_last_error_epoch = 0;
            uint32_t controller_last_error_expected_arrivals = 0;
            uint32_t controller_last_error_copied_arrivals = 0;
            uint32_t controller_last_error_copy_status_code = 0;
            uint32_t controller_last_error_copy_failure_flags = 0;
            uint32_t controller_last_error_copy_plan_entries_seen = 0;
            uint32_t
                controller_last_error_copy_skipped_wrong_destination = 0;
            uint32_t controller_last_error_missing_destination_slot =
                kDeviceMoEInvalidSlot;
            uint32_t controller_last_error_missing_destination_layer =
                kDeviceMoEInvalidSlot;
            uint32_t controller_last_error_missing_destination_expert =
                kDeviceMoEInvalidSlot;
            uint32_t controller_last_error_missing_destination_source =
                kDeviceMoEInvalidSlot;
            uint32_t controller_last_error_local_transfer_slot_count = 0;
            uint32_t controller_last_error_participant =
                kDeviceMoEInvalidSlot;
            uint32_t copy_status_code = 0;
            uint32_t copy_transaction_wave_index =
                kDeviceMoEInvalidSlot;
            uint32_t copy_transaction_epoch = 0;
            uint32_t copy_transaction_command_count = 0;
            uint32_t copy_plan_entries_seen = 0;
            uint32_t copy_copied_arrivals = 0;
            uint32_t copy_skipped_wrong_destination = 0;
            uint32_t copy_invalid_plan_entries = 0;
            uint32_t copy_missing_source_descriptors = 0;
            uint32_t copy_missing_destination_slots = 0;
            uint32_t copy_descriptor_mismatches = 0;
            uint32_t copy_incomplete = 0;
            uint32_t copy_required_local_arrivals = 0;
            uint32_t copy_ready_local_arrivals = 0;
            uint32_t apply_status_code = 0;
            uint32_t apply_invalid_plan_entries = 0;
            uint32_t apply_missing_source_descriptors = 0;
            uint32_t apply_missing_destination_slots = 0;
            uint32_t apply_descriptor_mismatches = 0;
            uint32_t apply_copy_incomplete = 0;
            uint32_t apply_required_local_arrivals = 0;
            uint32_t apply_ready_local_arrivals = 0;
            uint32_t windows_applied = 0;
            uint32_t selected_replicas = 0;
            uint32_t planned_arrivals = 0;
            uint32_t runtime_changed_layers = 0;
            uint32_t runtime_applied_arrivals = 0;
            /**
             * Runtime layers carrying sticky proof of an applied payload move.
             *
             * The marker survives later transfer-slot retirement, so this is
             * the request-level movement proof. The active-slot fields below
             * describe terminal storage integrity instead of movement history.
             */
            uint32_t prefill_current_batch_movement_layers = 0;
            /**
             * Runtime layers whose current-batch span consumer redistributed
             * rows to a resident non-owner participant without moving payload.
             */
            uint32_t prefill_current_batch_non_owner_assignment_layers = 0;
            /**
             * Transfer-slot-backed experts still active in the device runtime.
             *
             * This is terminal-state storage evidence collected by the
             * captured maintenance controller. It validates unique physical
             * slot ownership but may legitimately be zero after a later
             * maintenance wave retires an earlier prefill placement.
             */
            uint32_t prefill_active_transfer_slot_experts = 0;
            uint32_t prefill_unique_transfer_slot_claims = 0;
            uint32_t prefill_duplicate_transfer_slot_claims = 0;
            uint32_t prefill_invalid_transfer_slot_claims = 0;
            uint32_t prefill_max_transfer_slot = 0;
            uint32_t prefill_max_transfer_slot_layer = 0;
            uint32_t prefill_max_transfer_slot_expert = 0;
            uint32_t prefill_first_duplicate_transfer_slot = 0;
            uint32_t prefill_first_duplicate_layer = 0;
            uint32_t prefill_first_duplicate_expert = 0;
            uint32_t prefill_first_invalid_transfer_slot =
                kDeviceMoEInvalidSlot;
            uint32_t prefill_first_invalid_layer =
                kDeviceMoEInvalidSlot;
            uint32_t prefill_first_invalid_expert =
                kDeviceMoEInvalidSlot;
            uint32_t prefill_first_invalid_reasons = 0;
            uint32_t prefill_first_invalid_flags = 0;
            uint32_t prefill_first_invalid_resident_mask = 0;
            int32_t prefill_first_invalid_owner = -1;
            uint32_t local_transfer_slot_count = 0;
            uint32_t payload_bucket_slots = 0;
            uint32_t payload_source_participant_mask = 0;
            uint32_t payload_destination_participant_mask = 0;
            uint64_t payload_edge_mask = 0;

            /**
             * @brief Whether device maintenance observed an unrecoverable invariant violation.
             *
             * WindowNotReady is an ordinary no-work outcome. Every malformed
             * status object, controller error, invalid command, missing
             * descriptor, undersized destination, or incomplete copy is fatal:
             * inference must not continue with a partially published expert map.
             */
            bool hasFatalError() const noexcept
            {
                const bool status_error =
                    !valid ||
                    status_code >
                        static_cast<uint32_t>(
                            DeviceMoERebalanceStatusCode::WindowNotReady) ||
                    invalid_runtime_layers != 0u ||
                    plan_overflow != 0u ||
                    payload_bucket_overflow != 0u ||
                    prefill_duplicate_transfer_slot_claims != 0u ||
                    prefill_invalid_transfer_slot_claims != 0u ||
                    prefill_active_transfer_slot_experts !=
                        prefill_unique_transfer_slot_claims ||
                    (prefill_active_transfer_slot_experts != 0u &&
                     (local_transfer_slot_count == 0u ||
                      prefill_max_transfer_slot >=
                          local_transfer_slot_count));
                const bool copy_error =
                    copy_status_present &&
                    (!copy_status_valid ||
                     copy_status_code !=
                         static_cast<uint32_t>(
                             DeviceMoERebalanceApplyStatusCode::Ok) ||
                     copy_invalid_plan_entries != 0u ||
                     copy_missing_source_descriptors != 0u ||
                     copy_missing_destination_slots != 0u ||
                     copy_descriptor_mismatches != 0u ||
                     copy_incomplete != 0u ||
                     copy_ready_local_arrivals !=
                         copy_required_local_arrivals);
                const bool apply_error =
                    apply_status_present &&
                    (!apply_status_valid ||
                     apply_status_code !=
                         static_cast<uint32_t>(
                             DeviceMoERebalanceApplyStatusCode::Ok) ||
                     apply_invalid_plan_entries != 0u ||
                     apply_missing_source_descriptors != 0u ||
                     apply_missing_destination_slots != 0u ||
                     apply_descriptor_mismatches != 0u ||
                     apply_copy_incomplete != 0u ||
                     apply_ready_local_arrivals !=
                         apply_required_local_arrivals);
                return status_error ||
                       !controller_valid ||
                       controller_last_error_code != 0u ||
                       controller_error_waves != 0u ||
                       copy_error ||
                       apply_error;
            }
        };

        /** Export completed device-side MoE rebalance diagnostic status to PerfStats. */
        bool exportCompletedDeviceMoERebalanceMaintenanceStats(
            DeviceMoERebalanceMaintenanceGraphCache &cache,
            void *maintenance_stream,
            const std::string &device_key,
            const std::map<std::string, std::string> &maintenance_tags,
            DeviceMoERebalanceMaintenanceOutcome *outcome = nullptr);

        /**
         * @brief Publish the newest completed device maintenance status at an epilogue.
         *
         * Decode keeps maintenance planning, payload movement, and runtime-table
         * publication device-owned. This method is called only at an explicit
         * request epilogue or request reset, where surfacing final diagnostics
         * to the host is allowed. It waits on the maintenance stream only when
         * its newest launch is still in flight, performs one stream-scoped D2H
         * publication for a launch that has not already been observed, and
         * leaves all live inference state intact.
         *
         * @param boundary Stable diagnostic tag such as `request_epilogue` or
         *        `request_reset`.
         * @param reset_operation Concrete reset operation, such as
         *        `clear_cache`, when @p boundary is `request_reset`. Request
         *        epilogues pass `nullptr` because they do not mutate inference
         *        state. Keeping the operation separate from the generic
         *        boundary lets lifecycle tests prove that a particular reset
         *        drained device-owned maintenance without introducing a host
         *        mirror of the live controller state.
         */
        void drainCompletedDeviceMoERebalanceMaintenanceDiagnostics(
            const char *boundary,
            const char *reset_operation = nullptr);

        /**
         * @brief Export current-batch LLEP evidence at a PerfStats epilogue.
         *
         * Production MTP generation appends the same device reduction to its
         * existing terminal-control bridge. Ordinary decode, parity, and
         * benchmark execution do not launch that controller, so an explicitly
         * requested PerfStats run performs one ordered two-word readback after
         * measured inference has ended. This is a terminal observer only: it
         * never runs in a graph or token hot path and never mutates placement.
         */
        void publishTerminalCurrentBatchLLEPEvidenceDiagnostics();

        /** Report host-side safety state for chunk-boundary maintenance. */
        PrefillChunkMaintenanceState prefillChunkMaintenanceState(
            const PrefillChunkPlan &chunk) const override;

        /** Run local chunk-boundary maintenance after the engine gate allows it. */
        bool onPrefillChunkMaintenance(
            const PrefillChunkPlan &chunk,
            const PrefillChunkMaintenanceDecision &decision) override;

        /**
         * @brief Bind one scheduled prefill chunk to the ExpertOverlay authority.
         *
         * Multi-chunk prefill reuses one captured physical bucket, but every
         * replay is a distinct sparse transaction with its own authenticated
         * remote-follower ticket.  This hook opens that transaction, stamps the
         * coordinator-owned wire identity into the chunk input, and returns a
         * lease whose lifetime covers the exact executable launch.
         *
         * @param chunk_input Chunk-specific forward input about to execute.
         * @param chunk Scheduler-owned real and physical row geometry.
         * @param lease Receives the active transaction lease, or remains empty
         *        when this runner has no heterogeneous overlay authority.
         * @param error Receives a precise admission or lifecycle diagnostic.
         * @return True when the graph may be submitted.
         */
        bool beginPrefillChunkGraphSubmission(
            ForwardInput &chunk_input,
            const PrefillChunkPlan &chunk,
            std::unique_ptr<IPrefillChunkGraphSubmissionLease> *lease,
            std::string *error) override;

        /**
         * @brief Scope parity snapshots to one scheduled prefill chunk.
         *
         * This host-only diagnostic hook runs immediately before the graph
         * executor publishes its immutable snapshot slots. It changes only the
         * capture-key namespace; graph inputs, device state, and stream order
         * remain exactly those selected by ForwardExecutionEngine.
         *
         * @param chunk Real/bucket geometry currently being executed.
         */
        void beginPrefillChunkSnapshotDiagnostics(
            const PrefillChunkPlan &chunk) override;

        /**
         * @brief Restore the snapshot namespace that preceded one prefill chunk.
         * @param chunk Chunk whose diagnostic scope has completed.
         */
        void endPrefillChunkSnapshotDiagnostics(
            const PrefillChunkPlan &chunk) override;

        /**
         * @brief Join every sequence-shaped chunk snapshot into the normal parity keys.
         *
         * The graph executor has already projected padded bucket rows to their
         * real prefixes. This validates and concatenates those host-visible
         * values after the final chunk, restoring the full-prompt checkpoint
         * contract consumed by the existing CSV comparator.
         *
         * @param schedule Successfully executed chunk order.
         * @return False when a chunk is missing or has incompatible geometry.
         */
        bool finalizePrefillChunkSnapshotDiagnostics(
            const PrefillChunkSchedule &schedule) override;

        /**
         * @brief Forget a partial segmented-prefill capture after a failed request.
         *
         * Diagnostic data is discarded as a sequence candidate; no captured
         * graph allocation, model weight, or live KV state is modified here.
         */
        void cancelPrefillChunkSnapshotDiagnostics() noexcept override;

        /** Create and return the ForwardExecutionEngine with current config. */
        void ensureForwardEngine();

        // =========================================================================
        // Members
        // =========================================================================

        /// Graph builder (declarative layer)
        std::shared_ptr<IGraphBuilder> graph_builder_;

        /// Graph executor
        DeviceGraphExecutor executor_;

        /// GPU timing events that should be read at the next existing sync boundary.
        std::vector<PendingGpuTimingMeasurement> pending_gpu_timing_measurements_;

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
        struct PrefixArchiveDeviceStaging;
        std::unique_ptr<PrefixArchiveDeviceStaging> prefix_archive_device_staging_;
        PrefixPayloadLayout prefix_layout_;
        PrefixCacheStats prefix_cache_stats_;
        uint64_t prefix_fingerprint_ = 0;
        bool prefix_cache_bypassed_ = false;
        std::string prefix_cache_bypass_reason_;
        /**
         * @brief One reusable payload slot within the live MTP checkpoint pool.
         *
         * A complete transaction checkpoint can consume one slot for the main
         * cache plus one slot for every shifted MTP cache that owns recurrent
         * GDN/short-conv state. Handles retain shared ownership of the fields
         * they consume. A slot is reusable only after every exported
         * PrefixStateSnapshot releases those references, making overwrite
         * exclusion structural instead of timing-dependent. GPU slots contain
         * pure VRAM allocations and no host terminal-hidden mirror.
         */
        struct LiveCheckpointStorageSlot
        {
            size_t hybrid_host_capacity_bytes = 0;
            size_t hybrid_device_capacity_bytes = 0;
            size_t terminal_hidden_host_capacity_bytes = 0;
            size_t terminal_hidden_device_capacity_bytes = 0;
            DeviceId device = DeviceId::invalid();
            std::shared_ptr<std::vector<uint8_t>> hybrid_host_storage;
            std::shared_ptr<void> hybrid_device_storage;
            std::shared_ptr<std::vector<uint8_t>> terminal_hidden_host_storage;
            std::shared_ptr<void> terminal_hidden_device_storage;
            std::shared_ptr<void> ready_event;
        };
        mutable std::vector<LiveCheckpointStorageSlot> live_checkpoint_storage_pool_;
        /**
         * @brief Reusable VRAM slot for one cache's canonical sequence metadata.
         *
         * This pool is separate from recurrent/terminal payload storage because
         * every GPU cache requires metadata rollback, including plain
         * full-attention shifted caches with no hybrid payload. Slots are
         * allocated during runner initialization and retained by snapshot
         * descriptors until their asynchronous capture/restore lifetime ends.
         */
        struct LiveDeviceSequenceStateStorageSlot
        {
            size_t capacity_bytes = 0;
            DeviceId device = DeviceId::invalid();
            std::shared_ptr<void> storage;
            std::shared_ptr<void> ready_event;
        };
        mutable std::vector<LiveDeviceSequenceStateStorageSlot>
            live_device_sequence_state_storage_pool_;
        bool live_checkpoint_storage_pool_initialized_ = false;
        bool ensurePrefixCacheReady();
        /**
         * @brief Allocate and arena-register persistent GPU archive staging.
         *
         * The buffers are sized from the native logical payload layout and
         * reused for every block, layer, harvest, and restore. They are never
         * allocated in decode or prefix-transfer hot loops.
         */
        bool ensurePrefixArchiveDeviceStaging();

        /**
         * @brief Prepare one archive handle's completion event before any copy.
         *
         * Event creation is recoverable only at this boundary because no DMA
         * may yet reference the handle's pinned or device allocations.
         */
        bool preparePrefixArchivePayloadReady(
            PrefixBlockHandle *handle,
            void *producer_stream,
            const char *producer_name) const;

        /**
         * @brief Record and publish a preflighted archive completion event.
         *
         * This method is called after the final queued copy, including on
         * partial submission failure. A missing or unrecordable prepared event
         * is therefore fatal; production never synchronizes the stream as a
         * cleanup path.
         */
        bool recordPrefixArchivePayloadReady(
            PrefixBlockHandle *handle,
            void *producer_stream,
            const char *producer_name) const;

        /**
         * @brief Promote one pinned lower-tier archive into a persistent VRAM replica.
         *
         * Every present payload section is uploaded on `producer_stream`; one
         * readiness event is published after the final transfer, and only then
         * is the completed handle installed in the device-hot LRU. The method
         * performs no stream or device synchronization.
         */
        bool promotePrefixBlockToDeviceHotForRestore(
            const PrefixBlockHandle &lower_tier_handle,
            void *producer_stream,
            PrefixBlockHandle *promoted_handle);

        bool refreshPrefixPayloadLayoutForLiveHybridState(const char *operation);
        bool isPrefixCacheMoEModel() const;
        bool mtpSpecStatePublicationRequiresCapturedStage() const;
        void *explicitGPUStreamForOperation(const char *operation) const;
        /**
         * @brief Typed reasons for live inference-state mutations.
         *
         * Phase 5 treats live state as a versioned object.  Each mutation must
         * say whether it came from accepted speculative publication, a rejected
         * correction, prefix restore/truncate, or a full session reset.  This
         * keeps graph replay invalidation and perf diagnostics aligned.
         */
        enum class LivePrefixMutationReason
        {
            Unknown,
            AcceptedSpecPublication,
            RejectedCorrection,
            PrefixRestore,
            PrefixTruncate,
            SessionReset,
        };

        struct LivePrefixMutationRecord
        {
            uint64_t previous_epoch = 0;
            uint64_t live_state_epoch = 0;
            const char *reason_name = "unknown";
        };

        static const char *livePrefixMutationReasonName(
            LivePrefixMutationReason reason);
        LivePrefixMutationRecord recordLivePrefixMutation(
            LivePrefixMutationReason reason,
            const char *operation);
        /**
         * @brief Record a request/session state reset in live-state telemetry.
         *
         * A typed request-boundary reset clears live KV/GDN and request-local
         * metadata while deliberately preserving replay-safe CUDA/HIP graph
         * executables and their captured dynamic pointer tables. Keeping that
         * boundary explicit avoids misleading Phase 10 perf counters and
         * makes the ownership contract visible to future reset call sites.
         */
        void recordLivePrefixSessionReset(
            const char *operation,
            bool preserve_gpu_replay_state = false);
        /**
         * @brief Reset every cached depth-0 MTP sidecar graph replay handle.
         *
         * This does not mean the sidecar mutated main-model state.  It only
         * says the already-instantiated sidecar graph may have captured
         * per-step metadata that is not safe after an accepted-state
         * publication boundary.  Keeping the operation centralized makes the
         * handoff rule easy to audit.
         */
        void resetMTPSidecarDepth0ReplayState();
        /**
         * @brief Prepare the shifted MTP KV boundary consumed by one commit.
         *
         * A GPU transaction lease is an event-ordered device ownership
         * contract. In that mode this helper validates request/depth binding,
         * session ownership, and the cache's canonical device counter, then
         * deliberately avoids every host cache getter. CPU calls retain their
         * ordinary synchronous count validation and optional truncation.
         *
         * @param cache Shifted MTP cache that will receive the next row.
         * @param depth MTP cache depth represented by @p cache.
         * @param request_index Logical request inside the cache batch.
         * @param expected_cached_tokens CPU boundary. Ignored for GPU owners.
         * @param allow_speculative_discard Whether CPU state may be
         *        truncated to @p expected_cached_tokens.
         * @param stream Explicit GPU stream, or nullptr for CPU execution.
         * @param transaction Explicit child-local GPU transaction lease. GPU
         *        callers must provide a lease owned by this runner.
         * @param operation Stable diagnostic/perfstats operation name.
         * @return true when the boundary is ordered and ready for append.
         */
        bool prepareShiftedMTPKVCommitBoundary(
            IKVCache &cache,
            int depth,
            int request_index,
            int expected_cached_tokens,
            bool allow_speculative_discard,
            void *stream,
            const DeviceResidentMTPTransactionLease *transaction,
            const char *operation);
        /**
         * @brief Apply the graph-lifetime policy for a live inference-state mutation.
         *
         * The mutation reason is the complete policy input. Callers cannot
         * request blanket preservation or invalidation: the forward engine
         * classifies each cached graph by its device-state ownership contract,
         * preserves byte-proven stable-address captures through an explicit
         * stream rebind, and resets only live-state-versioned captures.
         *
         * @param reason Typed mutation boundary that selected the lifetime rule.
         * @param operation Stable diagnostic name for PerfStats attribution.
         */
        void handleLivePrefixReplayStateAfterMutation(
            LivePrefixMutationReason reason,
            const char *operation);
        PrefixCacheFingerprintResult buildCurrentPrefixFingerprint(
            const PrefixCacheRuntimeConfig &prefix_config) const;
        PrefixCacheKey makePrefixKeyForBlock(
            const std::vector<int32_t> &tokens,
            int block_index,
            uint64_t parent_hash) const;
        void disablePrefixCacheForRunner(const std::string &reason);
        bool acquireLiveCheckpointStorage(PrefixBlockHandle &handle) const;
        bool ensureLiveCheckpointStorage(PrefixBlockHandle &handle) const;
        bool acquireLiveDeviceSequenceStateStorage(
            int cache_depth,
            int sequence_index,
            const IKVCache &cache,
            DeviceKVSequenceStateCheckpoint *checkpoint) const;

        /**
         * @brief Preallocate every transient MTP checkpoint slot before inference.
         *
         * Production can retain transaction-base, verifier-base, and post-sidecar
         * snapshots concurrently. A fourth slot covers the optional commit-replay
         * diagnostic without changing hot-path allocation behavior.
         */
        bool initializeLiveCheckpointStoragePool();
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
        bool selectMTPTerminalHiddenRow(int row_idx, int seq_len, void *stream = nullptr);
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
                                     bool draft_condition_ready_is_target = false,
                                     int request_batch = 1,
                                     const int *position_ids_override = nullptr,
                                     const void *position_ids_device_override = nullptr,
                                     const void *speculative_outcome_meta_device = nullptr,
                                     int speculative_outcome_meta_stride = 0,
                                     const void *speculative_outcome_output_tokens_device = nullptr,
                                     int speculative_outcome_output_token_stride = 0,
                                     int speculative_outcome_request_index = -1,
                                     int speculative_first_output_token_index = 0,
                                     void *speculative_outcome_ready_event = nullptr,
                                     int draft_condition_token_stride = 1,
                                     int draft_condition_ready_count = 1,
                                     int draft_condition_ready_stride = 1,
                                     int device_position_offset = 0,
                                     int first_seq_idx = 0);

        /**
         * @brief Populate CPU shifted-MTP KV from one completed main prefill.
         *
         * CPU execution is host-owned, so this helper consumes host token and
         * position rows directly. GPU shifted prefill is a different production
         * architecture: `bindShiftedMTPPrefillTransaction()` installs it inside
         * the complete captured main graph and no GPU caller may enter this
         * method.
         *
         * @param tokens Host-owned token rows for the completed CPU prefill.
         * @param seq_len Number of rows produced per request.
         * @param batch_size Number of logical requests represented by the
         *        hidden-state tensor.
         * @param position_offset Logical position of the first prefill row.
         * @param request_lengths Optional real row count for each padded request.
         * @param position_ids Optional flattened host position rows.
         * @return true after every shifted pair is published.
         */
        bool populateCPUMTPShiftedCacheFromPrefill(
            const int *tokens,
            int seq_len,
            int batch_size,
            int position_offset,
            const std::vector<int> *request_lengths = nullptr,
            const std::vector<int> *position_ids = nullptr);

        /**
         * @brief Bind shifted MTP prefill into a main GPU forward graph.
         *
         * The binding contains only model-lifetime device addresses. Live token
         * values, request geometry, and main/shifted KV progress remain in those
         * allocations and are consumed by the captured graph at replay time.
         * Consequently this method performs no transfer and creates no host
         * shadow of request state.
         *
         * @param input Main-prefill input receiving the typed graph binding.
         * @param request_count Number of independent request rows in the graph.
         * @param purpose Runtime execution or setup-only graph-family
         *        declaration. Declaration bindings expose identical stage
         *        topology but cannot enter ForwardExecutionEngine.
         * @return true when the binding is either unnecessary or complete;
         *         false when enabled GPU MTP lacks a required owner.
         */
        bool bindShiftedMTPPrefillTransaction(
            ForwardInput &input,
            int request_count,
            ShiftedMTPPrefillGraphBinding::Purpose purpose =
                ShiftedMTPPrefillGraphBinding::Purpose::RuntimeExecution);

        /**
         * @brief Bind graph-integrated terminal-hidden publication for MTP decode.
         *
         * Main prefill already archives its terminal row through
         * @ref bindShiftedMTPPrefillTransaction. Grouped verification retains
         * candidate rows until accepted-state publication. This method binds
         * the remaining canonical transition: one main decoded row per request
         * copied into the persistent terminal archive by the same forward graph.
         *
         * @param input Forward input whose typed role/phase determines whether
         *        the binding is required.
         * @param request_count Number of decoded request rows.
         * @param purpose Runtime execution or declaration-only workspace sizing.
         * @return true when the binding is absent by policy or complete.
         */
        bool bindMTPMainTerminalHiddenTransaction(
            ForwardInput &input,
            int request_count,
            MTPMainTerminalHiddenGraphBinding::Purpose purpose =
                MTPMainTerminalHiddenGraphBinding::Purpose::RuntimeExecution);

        /**
         * @brief Construct the one canonical captured prefill-chunk binding.
         *
         * Setup materialization and live request scheduling both call this
         * helper.  Consequently the workspace generation, backend, KV counter,
         * arena addresses, bucket geometry, and pad token are hashed in one
         * place and cannot drift into distinct graph-cache identities.
         *
         * @param bucket_seq_len Physical captured row count.
         * @param pad_token_id Token written to inactive rows.
         * @return Complete binding, or std::nullopt when any permanent owner is
         *         absent or the requested bucket exceeds admitted capacity.
         */
        std::optional<DevicePrefillChunkGraphBinding>
        makeDevicePrefillChunkGraphBinding(
            int bucket_seq_len,
            int pad_token_id);

        /**
         * @brief Publish setup-only request rows before native family capture.
         *
         * Native capture records kernels that read the production request bank,
         * even though the retained executable is not launched. This method
         * initializes that graph frontier with model-lifetime sentinel bytes
         * and publishes their exact producer event before the engine performs
         * its pre-capture dependency pass. It does not admit a request or alter
         * KV, prefix, sampler, cursor, or response state.
         *
         * @param row_count Physical bucket rows about to be captured.
         * @param producer_stream Exact non-null setup publication stream.
         * @return True when all request-frontier members are device-valid.
         */
        bool prepareServingGraphSetupInputs(
            int row_count,
            void *producer_stream);

        /**
         * @brief Record the hidden-state ownership produced by main forward.
         *
         * Ordinarily a new main forward invalidates the compact terminal-hidden
         * cache, so a later MTP sidecar must explicitly select the latest row
         * from `HIDDEN_STATE`.  Single-request grouped prefill is the important
         * CPU grouped prefill is the important exception:
         * `populateCPUMTPShiftedCacheFromPrefill()` archives the true terminal
         * row after publishing every internal shifted pair. That row remains
         * current because a following CPU prefill segment consumes it to publish
         * the cross-segment shifted pair. GPU publication is part of the captured
         * main graph and reports the same ownership through its typed binding.
         *
         * @param seq_len Number of rows produced per request.
         * @param batch_size Number of logical requests represented by the
         *        hidden-state tensor.
         * @param request_lengths Optional real row count for each padded request.
         * @param terminal_hidden_archived True only when the caller has already
         *        copied the latest terminal row into `PREFIX_TERMINAL_HIDDEN`.
         */
        void noteMainForwardHiddenProducedForMTP(
            int seq_len,
            int batch_size,
            std::vector<int> request_lengths = {},
            bool terminal_hidden_archived = false);

        /**
         * @brief Resolve the terminal-hidden tensor that should seed a first MTP sidecar.
         *
         * The resolver centralizes freshness rules so ordinary forward never
         * runs helper row-select graphs just because MTP is enabled, while MTP
         * sidecar entry points cannot accidentally consume a stale prefix row.
         */
        bool resolveMTPTerminalHiddenInput(TensorBase **terminal_hidden,
                                           BufferId *terminal_hidden_buffer_id,
                                           const char *operation);

        struct MTPSidecarGraphCache
        {
            std::unique_ptr<ComputeGraph> graph;
            DeviceGraphExecutor::GraphSegmentCache segment_cache;
            /**
             * @brief Immutable physical identity of the graph held by this cache.
             *
             * One captured sidecar executable may serve several logical MTP
             * invocations when they share the same typed role, stable input
             * slots, shape, and workspace bindings. PerfStats lifecycle records
             * must remain attached to this physical identity even when the
             * caller's diagnostic context changes between launches.
             */
            std::string capture_perf_context;
            std::vector<IComputeStage *> dynamic_param_stages;
            /** Sparse stages whose wire identity changes for every ticket. */
            std::vector<IComputeStage *>
                moe_overlay_collective_runtime_stages;
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
            int batch_size = 1;
            int first_seq_idx = 0;
            bool uses_device_token_ids = false;
            bool uses_device_position_ids = false;
            int condition_token_slot = -1;
            void *condition_token_device = nullptr;
            const void *position_ids_device = nullptr;
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

            /**
             * @brief Clear request-local sidecar metadata without dropping replay graphs.
             *
             * MTP sidecar graphs read stable token/position/device-mailbox
             * buffers that are rewritten before every launch.  Request reset
             * must still clear stage-owned dynamic metadata and explicit stream
             * bindings, but preserving the graph replay cache avoids paying
             * warmup/capture again for every served request with the same shape.
             */
            void resetSessionStatePreservingGraphReplay()
            {
                if (graph)
                {
                    graph->reset();
                    for (const auto &node_name : graph->getExecutionOrder())
                    {
                        ComputeNode *node = graph->getNode(node_name);
                        if (node && node->stage)
                            node->stage->resetSessionStatePreservingCapturedReplay();
                    }
                }
            }

            void invalidate()
            {
                segment_cache.reset(DeviceGraphExecutor::GraphSegmentCache::StreamResetPolicy::Destroy);
                graph.reset();
                capture_perf_context.clear();
                dynamic_param_stages.clear();
                moe_overlay_collective_runtime_stages.clear();
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
                batch_size = 1;
                uses_device_token_ids = false;
                uses_device_position_ids = false;
                condition_token_slot = -1;
                condition_token_device = nullptr;
                position_ids_device = nullptr;
                valid = false;
            }
        };

        /**
         * @brief Depth-0 MTP sidecar graph caches partitioned by input source.
         *
         * Host token IDs and device-resident token IDs build different graph
         * signatures because the embedding stage reads from different stable
         * buffers.  Keeping them in separate caches lets both paths progress
         * from warmup to capture/replay independently; sharing one cache would
         * make stochastic decode alternate signatures and repeatedly rebuild.
         */
        MTPSidecarGraphCache mtp_sidecar_depth0_cache_;
        MTPSidecarGraphCache mtp_sidecar_depth0_device_token_cache_;
        MTPSidecarGraphCache mtp_sidecar_depth0_chained_cache_;
        MTPSidecarGraphCache mtp_sidecar_depth0_chained_device_token_cache_;
        MTPSidecarGraphCache mtp_sidecar_depth0_kv_only_cache_;
        MTPSidecarGraphCache mtp_sidecar_depth0_kv_only_device_token_cache_;
        /**
         * @brief Shape-specific KV-only graph caches, allocated before inference.
         *
         * Entry `i` owns flattened row count `i + 2`. Unique ownership keeps every
         * cache address stable even if the orchestrator itself moves, while the
         * immutable capture layout resolves both this index and the corresponding
         * condition-token publication slot.
         */
        MTPSidecarCaptureLayout mtp_sidecar_capture_layout_{2};
        std::vector<std::unique_ptr<MTPSidecarGraphCache>>
            mtp_sidecar_depth0_kv_only_batch_caches_;

        /**
         * @brief Identifies the logits stream handoff slot being manipulated.
         *
         * Captured GPU graph replay may finish asynchronously and hand its
         * stream to the next logits consumer.  Keeping those handoffs behind a
         * role-specific API avoids scattered raw pointer writes, which are hard
         * to reason about and easy to turn into cross-stream races.
         */
        enum class PendingLogitsStreamRole
        {
            MTPSidecar,
            MainDecode,
            AllPositionVerifier,
        };

        /**
         * @brief Whether a main-logits consumer observes or takes a stream handoff.
         *
         * Some GPU consumers mutate or inspect logits before a later sampler
         * completes the transaction. They must retain the role-specific stream
         * handoff. Terminal samplers take that handoff exactly once. Both modes
         * still join the durable ForwardGraphOutputReady event, so correctness
         * never depends on the optional raw-stream fast path being present.
         */
        enum class MainLogitsHandoffMode
        {
            Observe,
            Consume,
        };

        /**
         * @brief Typed host-observation surface for logits publication.
         *
         * The surface determines both the one-shot stream handoff and the
         * semantic kind required from the durable forward-output event. Keeping
         * the mapping here prevents callers from pairing main logits with a
         * verifier event or consuming the MTP sidecar handoff.
         */
        enum class HostLogitsSurface
        {
            Main,
            MTPSidecar,
            AllPositionVerifier,
        };

        /**
         * @brief Join one GPU logits producer timeline onto the host bridge.
         *
         * The returned stream is always explicit and non-null. The method first
         * waits on the lifecycle-owned forward event, then consumes any newer
         * role-specific stream publication onto the same bridge. Ordering
         * failures are fatal because continuing would expose stale logits.
         *
         * @throws std::runtime_error when no publication or explicit stream is
         *         available, or when either event edge cannot be established.
         */
        void *prepareLogitsHostObservation(
            HostLogitsSurface surface,
            const char *operation);

        /**
         * @brief Publish one logits tensor to host through the ordered bridge.
         *
         * CPU tensors are read directly. GPU tensors must be device
         * authoritative and are downloaded only after
         * prepareLogitsHostObservation() establishes the complete dependency.
         *
         * @throws std::runtime_error for invalid GPU ownership or failed D2H.
         */
        const float *publishLogitsTensorToHost(
            TensorBase *tensor,
            HostLogitsSurface surface,
            const char *operation) const;

        /**
         * @brief Publish a stream that produced logits for the next consumer.
         *
         * @param role Which logits buffer owns the handoff.
         * @param stream Explicit backend stream; must be non-null to create a
         *        real handoff.
         * @param producer Human-readable producer name for debug/profiling.
         *
         * Publishing is a one-shot ownership transfer. A producer may refresh
         * the same stream after an in-place logits mutation, but replacing an
         * unconsumed stream with a different stream is a logic error because it
         * would lose the ordering edge between graph replay and sampling.
         */
        void publishPendingLogitsStream(
            PendingLogitsStreamRole role,
            void *stream,
            const char *producer);

        /**
         * @brief Consume and clear a one-shot pending logits stream.
         *
         * Use this for samplers or reducers that become the next ordered GPU
         * operation after logits production.  The returned stream is never the
         * device default/null stream; callers must request an explicit fallback
         * if no handoff is pending.
         */
        void *consumePendingLogitsStream(
            PendingLogitsStreamRole role,
            const char *consumer);

        /**
         * @brief Acquire an ordered stream for a device consumer of main logits.
         *
         * This is the only valid device-consumer entry point for main logits.
         * It resolves the optional same-stream handoff, otherwise selects an
         * explicit owned stream, and always queues a wait for the durable
         * ForwardGraphOutputReady publication. The returned stream therefore
         * cannot observe a previous invocation's logits after graph replay.
         *
         * @param consumer Semantic operation name used for diagnostics.
         * @param mode Whether to retain or consume the one-shot stream handoff.
         * @return Ordered explicit GPU stream, or nullptr after a fatal
         *         dependency/publication error.
         */
        void *prepareMainLogitsDeviceConsumer(
            const char *consumer,
            MainLogitsHandoffMode mode,
            bool require_main_forward = true);

        /**
         * @brief Inspect a pending stream without clearing ownership.
         *
         * This is intentionally rare.  It is used for in-place mutations such
         * as logit penalties, where the mutator must remain in the same stream
         * chain and then republish the stream for the final sampler.
         */
        void *peekPendingLogitsStream(PendingLogitsStreamRole role) const;

        /** @brief Clear a pending stream because the associated state is reset. */
        void clearPendingLogitsStream(
            PendingLogitsStreamRole role,
            const char *reason);

        /**
         * @brief Order a consumer stream after a pending logits producer stream.
         *
         * This is the graph-captured equivalent of `flushPendingMTPWork()` for
         * code paths that need ordering but must not block the host.  The
         * producer handoff is consumed exactly once, then the GPU context queues
         * an event dependency from `consumer_stream` to the producer stream.
         */
        bool waitForPendingLogitsStream(
            PendingLogitsStreamRole role,
            void *consumer_stream,
            const char *consumer);

        /**
         * @brief Order an observation stream after a pending logits producer.
         *
         * Prefix probes, prefix harvest, and checkpoint capture read the same
         * live KV/GDN/MTP surfaces as normal forward graphs, but they are not
         * the semantic owner of deferred logits.  This helper queues the same
         * GPU stream dependency as waitForPendingLogitsStream() while leaving
         * the role handoff intact for the sampler, reducer, or later mutation
         * boundary that owns consumption.
         */
        bool waitForPendingLogitsStreamForObservation(
            PendingLogitsStreamRole role,
            void *consumer_stream,
            const char *consumer) const;

        /** @brief Clear every pending logits handoff during session teardown. */
        void clearAllPendingLogitsStreams(const char *reason);

        /**
         * @brief Queue a restore/truncate stream after any live graph producers.
         *
         * A normal forward can return with KV/GDN/logits writes still queued on
         * the captured graph stream.  Prefix restore and truncate both overwrite
         * the same live KV/GDN surface, so they must consume those stream
         * handoffs and make the mutation stream wait before touching state.
         */
        bool waitForPendingLiveGraphProducersBeforePrefixMutation(
            void *mutation_stream,
            const char *mutation_name);

        /**
         * @brief Queue an observation stream after all live graph producers.
         *
         * Unlike waitForPendingLiveGraphProducersBeforePrefixMutation(), this
         * method is non-consuming. It is the read-only side of the live-state
         * access contract: host-visible KV exports, prefix-cache harvest, and
         * diagnostic probes must observe graph-produced KV/GDN/MTP writes in
         * stream order without stealing sampler or mutation ownership. The
         * producer set includes every successful ordinary prefill/decode graph,
         * not only role-specific MTP logits handoffs.
         */
        bool waitForPendingLiveGraphProducersForObservation(
            void *observation_stream,
            const char *observation_name,
            DeviceTimelineRole observation_role) const;

        /**
         * @brief Queue an observation stream after every live inference-state producer.
         *
         * This is the first-class owner boundary for host-visible reads of live
         * inference state. It covers accepted verifier publication, prefix
         * restore/truncate mutation events, ordinary graph replay streams,
         * shifted MTP KV events, all-position verifier row-state events, and
         * device-resident logical sequence-state publication. Callers that
         * export KV, GDN, or MTP state must go through this helper instead of
         * assembling partial waits locally.
         *
         * @param observation_stream Exact non-null consumer stream.
         * @param observation_name Stable diagnostic identity.
         * @param observation_role Typed timeline consumer role.
         * @param logical_state_mailbox_already_joined True only when the caller
         *        already queued the same mailbox root as an admission preflight.
         * @return True after every remaining producer edge is enqueued.
         */
        bool waitForLiveInferenceStateReadyForObservation(
            void *observation_stream,
            const char *observation_name,
            DeviceTimelineRole observation_role,
            bool logical_state_mailbox_already_joined = false) const;

        /**
         * @brief Allocate the durable main-forward completion event.
         *
         * The event has orchestrator lifetime rather than graph-cache lifetime,
         * so a captured stream may be invalidated after publication without
         * invalidating the dependency token. Initialization happens before the
         * first request and therefore never allocates in a forward hot path.
         */
        bool initializeForwardGraphOutputReadyEvent();

        /**
         * @brief Commit the semantic result of one successful forward transaction.
         *
         * Every successful entry into `ForwardExecutionEngine`, including the
         * multi-chunk prefill scheduler, must pass through this single boundary.
         * It closes an armed grouped-verifier outcome transaction and publishes
         * both the exact current-logits tensor identity and its durable producer
         * event. Keeping those actions together prevents a caller from returning
         * usable device logits without making them consumable by the sampler.
         *
         * @param output Terminal output and producer provenance from the final
         *        graph invocation in the transaction.
         * @throws std::runtime_error if the output does not satisfy an armed
         *         verifier transaction or cannot publish its durable event.
         */
        void commitSuccessfulForwardOutput(
            const ForwardOutput &output) override;

        /**
         * @brief Publish one successful GPU forward into the device timeline.
         *
         * @param producer Invocation-scoped producer metadata returned by the
         *        forward engine. Its raw stream is consumed only during this call.
         * @return true after the persistent event was recorded successfully.
         */
        bool publishForwardGraphOutputReady(const ForwardOutput &output);

        /**
         * @brief Producer class for the authoritative current main-logits row(s).
         *
         * Forward graphs and prefix-terminal restores have different event
         * timelines, but both can establish the bytes sampled by the next token
         * transaction. The source remains typed so a consumer never treats a
         * restored row as an unrecorded forward or vice versa.
         */
        enum class MainLogitsPublicationSource
        {
            None,
            ForwardGraph,
            PrefixTerminalRestore,
        };

        /**
         * @brief Exact tensor authority for current main-model logits.
         *
         * This record contains no stream. Ordering remains owned by the durable
         * forward event or live-prefix mutation event. Its sole responsibility
         * is selecting the exact stable tensor written by that ordered producer.
         */
        struct MainLogitsPublicationState
        {
            TensorBase *tensor = nullptr;
            ForwardLogitsPublicationDescriptor descriptor{};
            MainLogitsPublicationSource source =
                MainLogitsPublicationSource::None;
            uint64_t session_epoch = 0;
            bool valid = false;
        };

        /** Classify an owned tensor without inferring from graph role or M. */
        [[nodiscard]] ForwardLogitsStorageSurface classifyForwardLogitsSurface(
            const TensorBase *tensor) const noexcept;

        /** Publish exact current-main tensor identity after its ordering edge. */
        bool publishCurrentMainLogits(
            TensorBase *tensor,
            const ForwardLogitsPublicationDescriptor &descriptor,
            MainLogitsPublicationSource source,
            const char *producer_name);

        /** Invalidate tensor identity at a request or restore boundary. */
        void clearCurrentMainLogitsPublication(const char *reason) noexcept;

        /** Resolve the exact published tensor for a semantic main consumer. */
        [[nodiscard]] TensorBase *requireCurrentMainLogits(
            DeviceLogitsSource source,
            const char *consumer_name) const;

        /** Resolve any sampling source while preserving main publication rules. */
        [[nodiscard]] TensorBase *resolveDeviceLogitsTensor(
            DeviceLogitsSource source,
            const char *consumer_name) const;

        /** @brief Required semantic kind for a latest-forward consumer. */
        enum class ForwardGraphOutputKind
        {
            Any,
            MainForward,
            GroupedVerifier,
        };

        /**
         * @brief Queue a consumer behind the durable latest-forward publication.
         *
         * No producer stream is retained or consulted here. The persistent event
         * remains valid even if the graph cache has already destroyed its capture
         * stream.
         */
        bool waitForForwardGraphOutputReady(
            void *consumer_stream,
            DeviceTimelineRole consumer_role,
            ForwardGraphOutputKind required_kind,
            const char *consumer_name) const;

        /**
         * @brief Drop transient stream/mailbox handoffs after restoring live state.
         *
         * `restoreLivePrefixState()` replaces the request-visible KV/GDN/logits
         * timeline with a captured snapshot.  Any pending handoff from the
         * abandoned timeline would describe ordering for old producers, not for
         * the restored state, so restore must clear them all before the next graph
         * is allowed to consume live state.
         */
        void clearLivePrefixRestoreTransientHandoffs(const char *reason);

        /**
         * @brief Clear request-scoped MTP verifier transaction state.
         *
         * MTP verification spans several owner surfaces: a host verifier row
         * plan, optional device-resident first-token materialization, compact
         * all-position logit row mode, and a publication-time base-KV snapshot.
         * Prefix restore and request reset abandon that transaction as a unit,
         * so this helper is the single boundary that returns the runner to an
         * ordinary continuation state.
         */
        void clearMTPVerifierTransactionStateForBoundary(const char *reason);

        /**
         * @brief Storage for one pending logits stream handoff.
         *
         * The raw pointer is deliberately nested so production code cannot
         * casually grab a role-specific member such as "main decode stream".
         * All ownership checks and perf counters live in the helper API below.
         */
        struct PendingLogitsStreamHandoff
        {
            /** @brief True when a producer has handed off a stream. */
            bool hasStream() const { return stream_ != nullptr; }

            /**
             * @brief Return whether `candidate` may replace the current state.
             *
             * A producer may republish the same stream after mutating logits in
             * place. Publishing a different non-null stream before consumption
             * would break the one-producer/one-consumer ordering contract.
             */
            bool canPublish(void *candidate) const
            {
                return !stream_ || stream_ == candidate;
            }

            /** @brief Publish a new pending stream after the caller validates it. */
            void publish(void *candidate) { stream_ = candidate; }

            /** @brief Observe the pending stream without consuming ownership. */
            void *peek() const { return stream_; }

            /** @brief Transfer ownership to the consumer and clear the slot. */
            void *consume()
            {
                void *pending = stream_;
                stream_ = nullptr;
                return pending;
            }

            /** @brief Drop any pending stream without transferring ownership. */
            void clear() { stream_ = nullptr; }

        private:
            void *stream_ = nullptr;
        };

        /** @brief Map a role to its storage slot. Only handoff helpers use this. */
        PendingLogitsStreamHandoff &pendingLogitsStreamHandoff(PendingLogitsStreamRole role);

        /** @brief Const view of a role's storage slot. */
        const PendingLogitsStreamHandoff &pendingLogitsStreamHandoff(PendingLogitsStreamRole role) const;

        /** @brief Const view of a role's raw storage slot. */
        void *pendingLogitsStreamValue(PendingLogitsStreamRole role) const;

        /** @brief Stable label used in profiling/debug records. */
        static const char *pendingLogitsStreamRoleName(PendingLogitsStreamRole role);

        std::array<PendingLogitsStreamHandoff, 3> pending_logits_streams_{};
        std::vector<PendingGpuDirectTransferSlotArrival> pending_gpu_direct_transfer_slot_arrivals_;
        std::vector<GpuDirectTransferCompletion> pending_gpu_direct_activation_completions_;
        std::vector<std::shared_ptr<GpuExpertTransferStagingPool>> gpu_direct_transfer_staging_pools_;
        bool defer_next_mtp_main_decode_sync_ = false;
        bool defer_all_position_verifier_sync_ = false;

        struct MTPTerminalHiddenRowSelectGraphCache
        {
            std::unique_ptr<ComputeGraph> graph;
            HiddenStateRowSelectStage *stage = nullptr;
            TensorBase *input = nullptr;
            TensorBase *output = nullptr;
            DeviceId device = DeviceId::invalid();
            uint64_t workspace_generation = 0;
            int seq_capacity = 0;
            int d_model = 0;
            bool valid = false;

            /**
             * @brief Reset dynamic stage state while preserving topology.
             *
             * This is safe only inside one live allocator generation. Request
             * boundaries use invalidate() instead because terminal-hidden helper
             * graphs cache graph-workspace scalar pointers and arena tensor
             * identities that are cheap to rebuild but dangerous to reuse after a
             * cache clear or workspace rebind.
             */
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
                workspace_generation = 0;
                seq_capacity = 0;
                d_model = 0;
                valid = false;
            }
        };

        struct MTPTerminalHiddenRowsSelectGraphCache
        {
            std::unique_ptr<ComputeGraph> graph;
            DeviceGraphExecutor::GraphSegmentCache segment_cache;
            HiddenStateRowsSelectStage *stage = nullptr;
            TensorBase *input = nullptr;
            TensorBase *output = nullptr;
            DeviceId device = DeviceId::invalid();
            uint64_t workspace_generation = 0;
            int seq_capacity = 0;
            int d_model = 0;
            int selected_row_count = 0;
            int fixed_contiguous_row_start = 0;
            int request_row_stride = 0;
            const int32_t *request_sequence_lengths_device = nullptr;
            HiddenStateRowsSelectStage::RequestRowStrideSource
                request_row_stride_source =
                    HiddenStateRowsSelectStage::RequestRowStrideSource::
                        StaticGraphGeometry;
            const int32_t *request_row_stride_device = nullptr;
            const int32_t *main_cached_tokens_device = nullptr;
            const int32_t *shifted_cached_tokens_device = nullptr;
            int request_index = -1;
            std::string row_buffer_name;
            const int32_t *external_device_row_indices = nullptr;
            HiddenStateRowsSelectStage::DeviceRowIndexSource row_index_source =
                HiddenStateRowsSelectStage::DeviceRowIndexSource::
                    StageOwnedIndices;
            bool valid = false;

            /**
             * @brief Reset dynamic stage state while preserving topology.
             *
             * Multi-row terminal-hidden selection also reads workspace-hosted row
             * metadata.  It may keep topology within a request, but clear_cache()
             * and allocator-generation changes must invalidate the cache outright.
             */
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
                segment_cache.reset(
                    DeviceGraphExecutor::GraphSegmentCache::
                        StreamResetPolicy::Destroy);
                graph.reset();
                stage = nullptr;
                input = nullptr;
                output = nullptr;
                device = DeviceId::invalid();
                workspace_generation = 0;
                seq_capacity = 0;
                d_model = 0;
                selected_row_count = 0;
                fixed_contiguous_row_start = 0;
                request_row_stride = 0;
                request_sequence_lengths_device = nullptr;
                request_row_stride_source =
                    HiddenStateRowsSelectStage::RequestRowStrideSource::
                        StaticGraphGeometry;
                request_row_stride_device = nullptr;
                main_cached_tokens_device = nullptr;
                shifted_cached_tokens_device = nullptr;
                request_index = -1;
                row_buffer_name.clear();
                external_device_row_indices = nullptr;
                row_index_source =
                    HiddenStateRowsSelectStage::DeviceRowIndexSource::
                        StageOwnedIndices;
                valid = false;
            }
        };

        /**
         * @brief Capture owner for one complete accepted-state mutation graph.
         *
         * Dynamic outcome/state values live at persistent device addresses and
         * are intentionally absent from this host cache.  The stage compares
         * every captured pointer and scalar before reuse; workspace generation
         * adds allocator lifetime to that identity.  Any mismatch destroys the
         * native executable before a replacement is built.
         */
        struct MTPSpeculativeStatePublicationGraphCache
        {
            std::unique_ptr<ComputeGraph> graph;
            DeviceGraphExecutor::GraphSegmentCache segment_cache;
            MTPSpeculativeStatePublicationStage *stage = nullptr;
            uint64_t workspace_generation = 0;
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
                    {
                        node->stage
                            ->resetSessionStatePreservingCapturedReplay();
                    }
                }
            }

            void invalidate()
            {
                segment_cache.reset(
                    DeviceGraphExecutor::GraphSegmentCache::
                        StreamResetPolicy::Destroy);
                graph.reset();
                stage = nullptr;
                workspace_generation = 0;
                valid = false;
            }
        };

        /**
         * @brief Capture owner for one fixed-depth MTP proposal publication.
         *
         * The cache is indexed by destination draft slot.  That slot is also
         * the number of prior proposals visible to branch penalties, making
         * depth and destination one structural identity rather than two
         * independently mutable integers.
         */
        struct MTPDraftTokenPublicationGraphCache
        {
            std::unique_ptr<ComputeGraph> graph;
            DeviceGraphExecutor::GraphSegmentCache segment_cache;
            MTPDraftTokenPublicationStage *stage = nullptr;
            uint64_t workspace_generation = 0;
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
                    {
                        node->stage
                            ->resetSessionStatePreservingCapturedReplay();
                    }
                }
            }

            void invalidate()
            {
                segment_cache.reset(
                    DeviceGraphExecutor::GraphSegmentCache::
                        StreamResetPolicy::Destroy);
                graph.reset();
                stage = nullptr;
                workspace_generation = 0;
                valid = false;
            }
        };

        /**
         * @brief Immutable controller policy embedded by verifier preparation.
         *
         * Standalone grouped verification derives its logical width from the
         * staged row plan. A device-generation parent instead reads live depth
         * from its resident controller. Those kernels have different pointer
         * topology and must therefore own distinct capture slots even when
         * request count and physical row geometry match.
         */
        enum class MTPVerifierPreparationControlPolicy : uint8_t
        {
            Standalone = 0,
            DeviceGenerationControlled = 1,
            Count = 2,
        };

        static constexpr size_t
            kMTPVerifierPreparationControlPolicyCount =
                static_cast<size_t>(
                    MTPVerifierPreparationControlPolicy::Count);

        /**
         * @brief Exact structural address of one verifier-preparation capture.
         *
         * Controller policy, request count, and physical row stride are known
         * before graph lookup. They therefore address one deterministic slot in
         * the bounded registry; no caller may search by recency, compare nearest
         * geometries, or replace a same-generation slot with different pointer
         * topology.
         */
        struct MTPVerifierPreparationGraphKey
        {
            MTPVerifierPreparationControlPolicy control_policy =
                MTPVerifierPreparationControlPolicy::Standalone;
            int request_count = 0;
            int padded_seq_len = 0;

            [[nodiscard]] bool populated() const noexcept
            {
                return control_policy !=
                           MTPVerifierPreparationControlPolicy::Count &&
                       request_count > 0 && padded_seq_len > 0;
            }

            bool operator==(
                const MTPVerifierPreparationGraphKey &) const = default;
        };

        /**
         * @brief Capture owner for one immutable grouped-verifier registry key.
         *
         * Dynamic token values and KV metadata remain behind persistent device
         * addresses.  The graph identity contains only pointer topology,
         * request geometry, and request-constant policy.  A bounded family is
         * preallocated with the MTP workspace so decode never allocates a cache
         * owner while selecting a previously captured depth or batch shape.
         */
        struct MTPVerifierPreparationGraphCache
        {
            MTPVerifierPreparationGraphKey key;
            std::unique_ptr<ComputeGraph> graph;
            DeviceGraphExecutor::GraphSegmentCache segment_cache;
            MTPVerifierPreparationStage *stage = nullptr;
            /** Exact replay-ready forward capture paired with this preparation. */
            std::optional<ForwardGraphSignature> paired_forward_signature;
            uint64_t workspace_generation = 0;
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
                    {
                        node->stage
                            ->resetSessionStatePreservingCapturedReplay();
                    }
                }
            }

            void invalidate()
            {
                segment_cache.reset(
                    DeviceGraphExecutor::GraphSegmentCache::
                        StreamResetPolicy::Destroy);
                graph.reset();
                key = {};
                stage = nullptr;
                paired_forward_signature.reset();
                workspace_generation = 0;
                valid = false;
            }
        };

        /**
         * @brief One semantic unit in a hosted retained generation branch.
         *
         * Local controller/sampler/publication units are ordinary captured
         * children. Sidecar and grouped-verifier units retain their production
         * segmented ComputeGraph plan because a heterogeneous sparse boundary
         * cannot be cloned into one native CUDA child. The kind makes that
         * distinction structural; no null capture is interpreted as a fallback.
         */
        struct HostedDeviceGenerationFragment
        {
            enum class Kind : uint8_t
            {
                CapturedLocal = 0,
                MTPFullSidecarReplay,
                MTPChainedSidecarReplay,
                MTPGroupedVerifierReplay,
                /** Entry boundary owned by the hosted scheduler stream. */
                MoEOverlayEpochAcquire,
                /** Terminal boundary that closes the hosted parent's reader. */
                MoEOverlayEpochRelease,
            };

            const char *name = nullptr;
            Kind kind = Kind::CapturedLocal;
            const IGPUGraphCapture *capture = nullptr;
            const ComputeGraph *semantic_graph = nullptr;
            ForwardGraphSignature forward_signature{};
            DeviceControlledLoopFragmentExecution execution =
                DeviceControlledLoopFragmentExecution::Always;
            const uint32_t *condition_word_device = nullptr;

            /** @return Whether two plans embed the same immutable executable identity. */
            [[nodiscard]] bool hasSameExecutionIdentity(
                const HostedDeviceGenerationFragment &other) const noexcept
            {
                return kind == other.kind && capture == other.capture &&
                       semantic_graph == other.semantic_graph &&
                       forward_signature == other.forward_signature &&
                       execution == other.execution &&
                       condition_word_device == other.condition_word_device;
            }
        };

        /**
         * @brief Persistent owner for one policy-complete device generation loop.
         *
         * Fixed policy owns one immutable transaction body. Dynamic policy owns
         * one complete retained branch per legal draft depth, selected from the
         * resident controller's exact value. CUDA embeds those branches in a
         * native SWITCH-in-WHILE parent. HIP captures an isolated ticket
         * publisher and submits the selected retained branch on one explicit
         * scheduler stream. Mutable generation state never leaves the device.
         *
         * The stream is allocated with runner workspace setup. The graph object
         * is retained across request boundaries and rebuilt from exact child
         * captures only during the first transaction's graph-materialization
         * phase. Replay never creates a stream, graph owner, or fragment list.
         * Member order is intentional: C++ destroys `capture` before `stream`.
         */
        struct MTPDeviceGenerationLoopGraphCache
        {
            enum class ExecutionKind : uint8_t
            {
                Unmaterialized = 0,
                NativeConditionalParent,
                HostedDispatchTicketPublisher,
            };

            std::shared_ptr<void> stream;
            std::unique_ptr<IGPUGraphCapture> capture;
            /** Exact child capture, policy, and predicate identities in the executable. */
            std::vector<DeviceControlledLoopFragment> source_fragments;
            /** Exact semantic branch inventory for hosted heterogeneous replay. */
            std::vector<HostedDeviceGenerationFragment> hosted_fragments;
            /** Flat source-fragment span owned by each legal draft depth. */
            std::array<
                size_t,
                sampling_math::DeviceGenerationDepthPolicy::
                        kMaximumSupportedDraftDepth +
                    1>
                branch_offsets{};
            /** Number of source fragments in each legal draft-depth branch. */
            std::array<
                size_t,
                sampling_math::DeviceGenerationDepthPolicy::
                        kMaximumSupportedDraftDepth +
                    1>
                branch_fragment_counts{};
            uint64_t workspace_generation = 0;
            int request_count = 0;
            /** Maximum/capture draft width embedded in every verifier child. */
            int draft_depth = 0;
            int verifier_rows_per_request = 0;
            /** Inclusive native SWITCH range; equal for a fixed body. */
            int minimum_draft_depth = 0;
            int maximum_draft_depth = 0;
            /** DeviceGenerationDepthPolicyMode encoded into this executable. */
            int depth_policy_mode = -1;
            /** Greedy/stochastic compact-outcome topology in every branch. */
            std::optional<DeviceGenerationSamplingMode> sampling_mode;
            size_t fragment_count = 0;
            /** Number of fragments whose execution is selected by device state. */
            size_t conditional_fragment_count = 0;
            ExecutionKind execution_kind = ExecutionKind::Unmaterialized;
            bool valid = false;
            bool launched = false;
            /** Exact ordinal lifecycle for one authenticated hosted branch. */
            HostedDeviceGenerationAdvanceCursor hosted_advance{};

            /**
             * @brief Retire one successfully materialized request launch.
             *
             * The executable and its child identities remain valid across the
             * request boundary. Only one-shot launch ownership changes; the
             * next admission may replay this parent when every child still has
             * the same complete capture identity.
             *
             * @return true only when a live executable owned an in-flight
             *         request launch.
             */
            [[nodiscard]] bool retireCompletedLaunch() noexcept
            {
                if (!valid || !launched)
                    return false;
                launched = false;
                return true;
            }

            void invalidateGraph() noexcept
            {
                if (capture)
                    capture->reset();
                workspace_generation = 0;
                request_count = 0;
                draft_depth = 0;
                verifier_rows_per_request = 0;
                minimum_draft_depth = 0;
                maximum_draft_depth = 0;
                depth_policy_mode = -1;
                sampling_mode.reset();
                fragment_count = 0;
                conditional_fragment_count = 0;
                branch_offsets.fill(0);
                branch_fragment_counts.fill(0);
                execution_kind = ExecutionKind::Unmaterialized;
                valid = false;
                launched = false;
                hosted_advance.reset();
                source_fragments.clear();
                hosted_fragments.clear();
            }

            void release() noexcept
            {
                invalidateGraph();
                capture.reset();
                stream.reset();
            }
        };

        /**
         * @brief Capture owner for seeded stochastic row sampling and reduction.
         *
         * The cache owns one active fixed-depth/request-count geometry. Dynamic
         * depth materialization replaces this owner during setup; parent-loop
         * composition may borrow it only while the exact stage identity remains
         * current. Request seeds are capture identity and therefore can never be
         * changed underneath a native executable.
         */
        struct MTPStochasticSerialOutcomeGraphCache
        {
            std::unique_ptr<ComputeGraph> graph;
            DeviceGraphExecutor::GraphSegmentCache segment_cache;
            MTPStochasticSerialOutcomeStage *stage = nullptr;
            uint64_t workspace_generation = 0;
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
                    {
                        node->stage
                            ->resetSessionStatePreservingCapturedReplay();
                    }
                }
            }

            void invalidate()
            {
                segment_cache.reset(
                    DeviceGraphExecutor::GraphSegmentCache::
                        StreamResetPolicy::Destroy);
                graph.reset();
                stage = nullptr;
                workspace_generation = 0;
                valid = false;
            }
        };

        /**
         * @brief Capture owner for verifier penalties and compact target rows.
         *
         * One cache identity represents one immutable row geometry and sampler
         * policy. All dynamic logits, verifier tokens, and generated-history
         * counts remain in persistent device storage across parent-loop replay.
         */
        struct MTPStochasticTargetDistributionGraphCache
        {
            std::unique_ptr<ComputeGraph> graph;
            DeviceGraphExecutor::GraphSegmentCache segment_cache;
            MTPStochasticTargetDistributionStage *stage = nullptr;
            uint64_t workspace_generation = 0;
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
                    {
                        node->stage
                            ->resetSessionStatePreservingCapturedReplay();
                    }
                }
            }

            void invalidate()
            {
                segment_cache.reset(
                    DeviceGraphExecutor::GraphSegmentCache::
                        StreamResetPolicy::Destroy);
                graph.reset();
                stage = nullptr;
                workspace_generation = 0;
                valid = false;
            }
        };

        /**
         * @brief Capture owner for one ExpertOverlay reader lifecycle edge.
         *
         * Acquire and release each own one graph because the complete inference
         * transaction can contain an arbitrary number of main/MTP child graphs
         * between them. The arena and request slot are model-lifetime topology;
         * the workspace generation authenticates the capture/executor lifetime.
         */
        struct MoEOverlayEpochBoundaryGraphCache
        {
            std::unique_ptr<ComputeGraph> graph;
            DeviceGraphExecutor::GraphSegmentCache segment_cache;
            MoEOverlayEpochBoundaryStage *stage = nullptr;
            uint64_t workspace_generation = 0;
            bool valid = false;

            /** @brief Destroy capture state and every borrowed stage pointer. */
            void invalidate()
            {
                segment_cache.reset(
                    DeviceGraphExecutor::GraphSegmentCache::
                        StreamResetPolicy::Destroy);
                graph.reset();
                stage = nullptr;
                workspace_generation = 0;
                valid = false;
            }
        };

        MTPTerminalHiddenRowSelectGraphCache mtp_terminal_hidden_row_select_cache_;
        /// CPU/direct-fixture arbitrary-row selector. Production GPU paths use one of the typed caches below.
        MTPTerminalHiddenRowsSelectGraphCache mtp_terminal_hidden_rows_select_cache_;
        /// One immutable row-zero suffix graph per MTP catchup width, materialized before request execution.
        std::vector<std::unique_ptr<MTPTerminalHiddenRowsSelectGraphCache>>
            mtp_terminal_hidden_contiguous_rows_select_caches_;
        /**
         * @brief Exact-request-count device-indexed accepted-state publication graphs.
         *
         * Captured row-selection kernels have immutable launch geometry. Replaying
         * a maximum-capacity graph for a smaller request batch would consume
         * inactive accepted-row indices and publish terminal-hidden rows that do
         * not belong to the transaction. Keep one graph per legal request count
         * so capture geometry and publication ownership are identical by type.
         * None of these graphs owns pinned host row metadata.
         */
        std::vector<std::unique_ptr<MTPTerminalHiddenRowsSelectGraphCache>>
            mtp_terminal_hidden_device_accepted_rows_select_caches_;
        /**
         * @brief Device-geometry request-terminal graph for every legal count.
         *
         * Request count fixes launch/output geometry and therefore owns one
         * captured graph identity. Prompt width does not: every graph reads the
         * current padded stride from REQUEST_BATCH_GEOMETRY. Runtime execution
         * can consequently select by count without rebuilding for a new prompt.
         */
        std::vector<std::unique_ptr<MTPTerminalHiddenRowsSelectGraphCache>>
            mtp_terminal_hidden_request_rows_select_caches_;
        /**
         * @brief Shifted-prefill selectors indexed by request then row count.
         *
         * Every graph reads canonical main/shifted KV progress and resident
         * request geometry. Prompt width and current row are therefore replay
         * data, while request identity and bounded grouped width remain graph
         * topology. The vector is flattened as
         * `request * mtp_sidecar_condition_token_slot_width_ + rows - 1`.
         */
        std::vector<std::unique_ptr<MTPTerminalHiddenRowsSelectGraphCache>>
            mtp_shifted_prefill_hidden_rows_select_caches_;

        /// Active geometry/policy publication fragment used by resident MTP.
        MTPSpeculativeStatePublicationGraphCache
            mtp_speculative_state_publication_graph_;

        /// One immutable proposal publication graph for every legal draft slot.
        std::vector<std::unique_ptr<MTPDraftTokenPublicationGraphCache>>
            mtp_draft_token_publication_graphs_;

        /**
         * @brief Deterministic request-count/physical-width registry.
         *
         * Slot `((policy * request_capacity) + requests - 1) *
         * mtp_max_verifier_rows_ + (rows - 1)` owns exactly that key for the
         * current workspace generation. Scalar MTP populates only canonical
         * physical buckets; request-batched MTP uses its exact physical stride.
         */
        std::vector<std::unique_ptr<MTPVerifierPreparationGraphCache>>
            mtp_verifier_preparation_graphs_;

        /// CUDA conditional parent assembled from the exact active MTP fragments.
        MTPDeviceGenerationLoopGraphCache mtp_device_generation_loop_graph_;

        /**
         * @brief Allocation-free named child-capture transaction assembled in
         *        explicit producer-to-consumer order.
         *
         * Names travel into backend validation so a native conditional-body
         * restriction identifies the responsible graph role rather than an
         * opaque pointer or ordinal. Capacity is reserved with MTP workspace
         * setup and no element is added during graph replay.
         */
        std::vector<DeviceControlledLoopFragment>
            mtp_device_generation_loop_fragment_scratch_;

        /**
         * @brief Allocation-free hosted branch inventory assembled beside native children.
         *
         * This storage is used only by authenticated hosted scheduling. Its
         * semantic sidecar/verifier entries retain segmented production graphs;
         * every other entry names the same captured child used by native CUDA.
         */
        std::vector<HostedDeviceGenerationFragment>
            mtp_device_generation_loop_hosted_fragment_scratch_;

        /**
         * @brief Allocation-free native SWITCH branch descriptors.
         *
         * Entry `d` names the complete transaction for resident draft depth `d`.
         * Its span borrows the flattened fragment scratch above only during
         * synchronous parent composition. Workspace setup sizes this inventory
         * through the maximum supported depth so materialization never grows it.
         */
        std::vector<DeviceControlledLoopBranch>
            mtp_device_generation_loop_branch_scratch_;

        /**
         * @brief Allocation-free identity scratch for verifier graph selection.
         *
         * Capacity is reserved with the persistent MTP arena. The spans passed
         * to materializeMTPVerifierPreparationGraph() are valid for that call;
         * a newly constructed stage takes its own immutable copy, while cache
         * hits only compare against these rows.
         */
        std::vector<MTPVerifierPreparationStage::TokenRowBinding>
            mtp_verifier_preparation_token_row_scratch_;
        std::vector<MTPVerifierPreparationStage::MainKVCheckpointBinding>
            mtp_verifier_preparation_checkpoint_scratch_;

        /// Active seeded stochastic outcome fragment used by fixed-depth tuning.
        MTPStochasticSerialOutcomeGraphCache
            mtp_stochastic_serial_outcome_graph_;

        /// Active verifier penalty/distribution fragment for resident MTP.
        MTPStochasticTargetDistributionGraphCache
            mtp_stochastic_target_distribution_graph_;

        /// First unconditional fragment of every durable overlay transaction.
        MoEOverlayEpochBoundaryGraphCache
            moe_overlay_epoch_acquire_graph_;

        /// Last unconditional fragment before any placement maintenance.
        MoEOverlayEpochBoundaryGraphCache
            moe_overlay_epoch_release_graph_;

        /// Stable graph-builder-owned arena and immutable request-slot identity.
        DeviceMoEOverlayEpochExecutionBinding
            moe_overlay_epoch_execution_binding_;

        /** Shared finite relay epoch submitted before each local inference graph. */
        std::shared_ptr<MappedTransferProgressEpoch>
            moe_overlay_transfer_progress_epoch_;

        /**
         * Durable publication from the first acquire stream to every later child.
         * The event is model-runner lifetime and may be waited by multiple streams.
         */
        std::shared_ptr<void> moe_overlay_epoch_acquired_event_;

        /// Exact stream that most recently published the acquired-event.
        void *moe_overlay_epoch_acquire_producer_stream_ = nullptr;

        /// Completion edge preventing a later acquire from racing prior release.
        std::shared_ptr<void> moe_overlay_epoch_released_event_;

        /**
         * Single host authority consumed by inference and maintenance threads.
         * It owns the exact release producer stream together with the
         * `ReleasePublished` state so those values cannot be raced apart.
         */
        std::unique_ptr<MoEOverlayEpochLeaseLifecycle>
            moe_overlay_epoch_lease_lifecycle_ =
                std::make_unique<MoEOverlayEpochLeaseLifecycle>();

        /// Identity pinned while one complete direct MTP sequence owns the epoch.
        MoEOverlayExternalSequenceIdentity
            moe_overlay_external_sequence_identity_{};

        /// Exact stream admitted by the active forward phase of the lease.
        void *moe_overlay_epoch_forward_submission_stream_ = nullptr;

        /**
         * @brief Device-checkpoint bank written by the last successful hidden producer.
         *
         * This is diagnostic routing metadata only; it never supplies inference
         * values. Recording it immediately after executeForward() succeeds keeps
         * the failure path tied to the graph that actually produced terminal
         * hidden state. This includes grouped verifiers because accepted-state
         * publication can select their hidden rows into the terminal mailbox.
         * Inferring the bank later from mutable sidecar metadata can
         * accidentally select an older prefill after a grouped verifier.
         */
        std::string last_mirrored_layer_checkpoint_prefix_;

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
        std::shared_ptr<ForwardGraphExecutionRendezvous> forward_execution_rendezvous_;

        struct DeviceMoERebalanceMaintenanceGraphCache
        {
            /**
             * @brief Classifies reset boundaries by captured pointer identity.
             *
             * StableDeviceStateContents means that request-owned values may
             * change while every address embedded in the executable remains
             * owned by the same model-lifetime object. BindingIdentityChanged
             * is reserved for topology teardown or replacement of one of those
             * owners and therefore destroys the graph and capture stream.
             *
             * Keeping this distinction as a closed enum prevents callers from
             * expressing the old ambiguous "session reset" operation whose
             * behavior silently depended on which request path invoked it.
             */
            enum class ResetBoundary
            {
                StableDeviceStateContents,
                BindingIdentityChanged,
            };

            std::unique_ptr<ComputeGraph> graph;
            /**
             * @brief Instance-level collective nodes owned by @ref graph.
             *
             * This set is discovered exactly once when the maintenance graph
             * is built. Capture policy and executor classification must both
             * consume this same object so composite MoE stages can never be
             * capture-disabled by contradictory host bookkeeping.
             */
            std::unordered_set<std::string> collective_nodes;
            DeviceGraphExecutor::GraphSegmentCache segment_cache;
            /** Captured one-kernel serial/MTP boundary publisher. */
            std::unique_ptr<IGPUGraphCapture>
                cadence_boundary_publication_capture;
            /** Captured one-kernel skipped-boundary acknowledgement. */
            std::unique_ptr<IGPUGraphCapture>
                cadence_boundary_acknowledgement_capture;
            /** CUDA native `publish -> IF(due) maintenance -> ack` parent. */
            std::unique_ptr<IGPUGraphCapture> cadence_transaction_capture;
            /**
             * @brief Whether main HIP decode topology owns non-due cadence.
             *
             * This is immutable graph-construction evidence, not live device
             * state. It proves the rank's known-non-due branch can remain a
             * host no-op without skipping the device clock transaction.
             */
            bool hosted_decode_commit_boundary_embedded = false;
            /** True after the backend's native-transaction capability is known. */
            bool cadence_transaction_capability_checked = false;
            /** Immutable setup bounds used only to elide impossible HIP reads. */
            DeviceMoERebalanceHostedObservationSchedule
                hosted_observation_schedule;
            uint64_t workspace_generation = 0;
            uint64_t launch_count = 0;
            /**
             * @brief Newest launch whose device status reached the host.
             *
             * This is diagnostic bookkeeping, not a host mirror of live
             * placement. Steady decode may advance launch_count without
             * touching this value; an explicit epilogue then publishes only
             * the newest unseen status.
             */
            uint64_t last_status_publication_launch_count = 0;
            std::shared_ptr<void> completion_event;
            /**
             * @brief Whether device status/timing export still owns this event.
             *
             * This flag is host diagnostic bookkeeping. It may become false
             * when an epilogue observes completion even though the next
             * production graph still owes the event a device-side stream wait.
             */
            bool completion_event_in_flight = false;
            /**
             * @brief Whether the current persistent binding owns initialized request state.
             *
             * The maintenance graph can be materialized after the first request
             * reset has already crossed its boundary. In that case workspace
             * addresses exist only after family allocation, so family publication
             * must initialize the transaction once before any decode/verifier
             * graph may consume the controller. Subsequent request resets set this
             * flag through the same typed transaction owner.
             */
            bool request_transaction_initialized = false;

            /**
             * @brief Reset the sole request-owned transaction behind this graph.
             *
             * Graph reset and request reset are different lifetimes. The graph
             * executable, workspace addresses, streams, and events survive a
             * stable-content boundary, while controller epochs, active waves,
             * terminal poison, headers, cursors, and plan counts do not.
             * Discovering ownership from the typed stage makes a missing or
             * multiply-owned transaction a construction error instead of
             * silently preserving stale state.
             *
             * @param request_reset_stream Stream that already joined every old
             *        request producer and will later publish reset-ready.
             * @return true only when exactly one owning stage enqueued its reset.
             */
            bool resetRequestOwnedDeviceTransaction(void *request_reset_stream)
            {
                if (!graph || !request_reset_stream)
                    return false;

                std::vector<MoEDeviceRebalanceStage *> transaction_owners;
                for (const auto &node_name : graph->getExecutionOrder())
                {
                    ComputeNode *node = graph->getNode(node_name);
                    auto *stage =
                        node && node->stage
                            ? dynamic_cast<MoEDeviceRebalanceStage *>(
                                  node->stage.get())
                            : nullptr;
                    if (stage && stage->ownsRequestTransactionState())
                        transaction_owners.push_back(stage);
                }
                if (transaction_owners.size() != 1u)
                {
                    LOG_ERROR("[DeviceMoERebalanceMaintenanceGraphCache] Stable request reset requires exactly one transaction owner"
                              << " owners=" << transaction_owners.size());
                    return false;
                }
                const bool reset = transaction_owners.front()
                                       ->resetRequestTransactionStateOnStream(
                                           request_reset_stream);
                request_transaction_initialized = reset;
                return reset;
            }

            void resetReplayState()
            {
                segment_cache.reset(DeviceGraphExecutor::GraphSegmentCache::StreamResetPolicy::Preserve);
                cadence_transaction_capture.reset();
                cadence_boundary_acknowledgement_capture.reset();
                cadence_boundary_publication_capture.reset();
                cadence_transaction_capability_checked = false;
                hosted_observation_schedule = {};
            }

            /**
             * @brief Cross an explicit state boundary with deterministic cache semantics.
             *
             * A stable-content reset preserves the captured executable and its
             * lifecycle counters. It clears only graph completion bits,
             * transient stream aliases, and completed diagnostic ownership.
             * A binding-identity reset destroys every object that could retain
             * an obsolete pointer.
             */
            void reset(ResetBoundary boundary)
            {
                if (boundary == ResetBoundary::BindingIdentityChanged)
                {
                    invalidate();
                    return;
                }

                /*
                 * The exact boundary epilogue completes and exports maintenance
                 * diagnostics before reaching this replay-preserving reset.
                 * Retire only diagnostic in-flight state. The durable completion
                 * event and launch_count remain published, so every next-request
                 * production stream still queues its own manifest-validated
                 * event wait; diagnostic host observation is not execution
                 * ordering.
                 */
                completion_event_in_flight = false;
                if (!graph)
                    return;
                graph->reset();
                for (const auto &node_name : graph->getExecutionOrder())
                {
                    ComputeNode *node = graph->getNode(node_name);
                    if (node && node->stage)
                        node->stage->resetSessionStatePreservingCapturedReplay();
                }
            }

            void invalidate()
            {
                cadence_transaction_capture.reset();
                cadence_boundary_acknowledgement_capture.reset();
                cadence_boundary_publication_capture.reset();
                cadence_transaction_capability_checked = false;
                hosted_decode_commit_boundary_embedded = false;
                hosted_observation_schedule = {};
                segment_cache.reset(DeviceGraphExecutor::GraphSegmentCache::StreamResetPolicy::Destroy);
                graph.reset();
                collective_nodes.clear();
                workspace_generation = 0;
                launch_count = 0;
                last_status_publication_launch_count = 0;
                completion_event.reset();
                completion_event_in_flight = false;
                request_transaction_initialized = false;
            }
        };

        DeviceMoERebalanceMaintenanceGraphCache device_moe_rebalance_maintenance_graph_;

        /** One-way pinned observer and event for the HIP MoE ticket boundary. */
        std::unique_ptr<PinnedMoERebalanceDispatchTicketScratch>
            device_moe_rebalance_dispatch_ticket_host_scratch_;
        std::shared_ptr<void>
            device_moe_rebalance_dispatch_ticket_ready_event_;
        std::optional<DeviceMoERebalanceDispatchTicket>
            last_device_moe_rebalance_dispatch_ticket_;
        bool device_moe_rebalance_dispatch_ticket_copy_pending_ = false;
        uint32_t hosted_device_moe_rebalance_last_committed_round_ = 0;

        /// Padded sequence length from last forward_batch() call
        int padded_seq_len_ = 0;

        // =========================================================================
        // Snapshot Capture Members
        // =========================================================================

        /// Whether snapshot capture is enabled
        bool snapshot_enabled_ = false;

        /// Optional execution context prefix for disambiguating repeated stage keys.
        std::string snapshot_context_;

        /**
         * @brief Namespace to restore when the currently executing prefill chunk ends.
         *
         * ForwardExecutionEngine invokes the begin/end pair through an RAII
         * scope, so this optional value is populated for at most one chunk on
         * this single-threaded runner.
         */
        std::optional<std::string> prefill_chunk_snapshot_previous_context_;

        /**
         * @brief Ordered live-row chunks awaiting prompt-wide snapshot aggregation.
         *
         * The entries name host diagnostics only. Their row counts were already
         * enforced by the graph scheduler and do not duplicate a device-owned
         * execution cursor.
         */
        std::vector<SnapshotChunkSequencePart> prefill_chunk_snapshot_sequence_;

        /// Snapshot capture engine (owns storage + routing logic)
        SnapshotCapture snapshot_capture_;
        std::unordered_set<std::string> snapshot_capture_filter_;

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

        /// Whether the complete serial graph family has published its workspace.
        bool forward_workspace_family_materialized_ = false;

        /// Immutable GPU workspace generation certified by that family manifest.
        uint64_t forward_workspace_family_generation_ = 0;

        /// Unified buffer arena — owns and tracks coherence for all activation buffers
        std::unique_ptr<BufferArena> arena_;

        /// Cached device pointers for the arena-owned argmax partial-reduction
        /// scratch (two-pass GPU greedy sampling). Resolved once after arena
        /// allocation so the per-decode-step hot path avoids any arena lookups.
        void *argmax_partial_vals_dev_ = nullptr; ///< FP32 [1, argmax_partial_capacity_]
        void *argmax_partial_idxs_dev_ = nullptr; ///< INT32 [1, argmax_partial_capacity_]
        int argmax_partial_capacity_ = 0;         ///< Entries in the partial scratch (0 = unavailable)

        int stochastic_target_row_capacity_ = 0;
        int stochastic_draft_row_capacity_ = 0;
        int stochastic_batch_output_request_capacity_ = 1; ///< Configured request count owned by grouped MTP transactions on every backend.
        int mtp_max_draft_depth_ = 1; ///< Largest configured fixed/dynamic draft depth owned by this runner.
        int mtp_max_verifier_rows_ = 2; ///< Per-request draft rows plus the terminal bonus row.
        int stochastic_batch_output_token_stride_ = 2; ///< Per-request compact output capacity.

        /**
         * @brief Typed view over persistent device-owned generation rows.
         *
         * Response tokens and scalar controller words intentionally live in
         * separate arena buffers: response capacity scales with the configured
         * sequence limit, while the controller has a fixed ABI.  This view
         * binds them into one lifecycle and is the only code allowed to derive
         * per-request control-field addresses.
         */
        struct DeviceGenerationStorage
        {
            int request_capacity = 0;
            int response_token_stride = 0;
            int control_stride = 0;
            int32_t *response_tokens_device = nullptr;
            int *control_device = nullptr;
            sampling_math::DeviceGenerationDispatchTicket
                *dispatch_tickets_device = nullptr;
            int active_request_count = 0;

            bool bind(
                void *response_base,
                int response_rows,
                int response_stride,
                void *control_base,
                int control_rows,
                int controller_stride,
                void *dispatch_ticket_base,
                int dispatch_ticket_rows,
                int dispatch_ticket_words)
            {
                clear();
                constexpr int expected_dispatch_ticket_words =
                    static_cast<int>(
                        sizeof(sampling_math::DeviceGenerationDispatchTicket) /
                        sizeof(int32_t));
                if (!response_base || !control_base || !dispatch_ticket_base ||
                    response_rows <= 0 || response_rows != control_rows ||
                    response_rows != dispatch_ticket_rows ||
                    response_stride <= 0 ||
                    controller_stride <
                        sampling_math::kDeviceGenerationControlCount ||
                    dispatch_ticket_words != expected_dispatch_ticket_words)
                {
                    return false;
                }
                request_capacity = response_rows;
                response_token_stride = response_stride;
                control_stride = controller_stride;
                response_tokens_device =
                    static_cast<int32_t *>(response_base);
                control_device = static_cast<int *>(control_base);
                dispatch_tickets_device =
                    static_cast<
                        sampling_math::DeviceGenerationDispatchTicket *>(
                        dispatch_ticket_base);
                return true;
            }

            bool validFor(int request_count) const
            {
                return request_count > 0 &&
                       request_count <= request_capacity &&
                       response_tokens_device != nullptr &&
                       response_token_stride > 0 &&
                       control_device != nullptr &&
                       control_stride >=
                           sampling_math::kDeviceGenerationControlCount &&
                       dispatch_tickets_device != nullptr;
            }

            int *controlForRequest(int request_index) const
            {
                return request_index >= 0 &&
                               request_index < request_capacity &&
                               control_device
                           ? control_device +
                                 static_cast<size_t>(request_index) *
                                     static_cast<size_t>(control_stride)
                           : nullptr;
            }

            const uint32_t *transactionBudgetForRequest(
                int request_index) const
            {
                int *row = controlForRequest(request_index);
                return row
                           ? reinterpret_cast<const uint32_t *>(
                                 row + sampling_math::
                                           kDeviceGenerationControlTransactionCommitBudget)
                           : nullptr;
            }

            sampling_math::DeviceGenerationDispatchTicket *
            dispatchTicketForRequest(int request_index) const
            {
                return request_index >= 0 &&
                               request_index < request_capacity &&
                               dispatch_tickets_device
                           ? dispatch_tickets_device + request_index
                           : nullptr;
            }

            void clear()
            {
                request_capacity = 0;
                response_token_stride = 0;
                control_stride = 0;
                response_tokens_device = nullptr;
                control_device = nullptr;
                dispatch_tickets_device = nullptr;
                active_request_count = 0;
            }
        };

        DeviceGenerationStorage device_generation_storage_;
        void *stochastic_target_token_ids_dev_ = nullptr; ///< INT32 [target_rows, 256]
        void *stochastic_target_probs_dev_ = nullptr;     ///< FP32 [target_rows, 256]
        void *stochastic_draft_token_ids_dev_ = nullptr;  ///< INT32 [draft_rows, 256]
        void *stochastic_draft_probs_dev_ = nullptr;      ///< FP32 [draft_rows, 256]
        void *stochastic_processed_logits_dev_ = nullptr; ///< FP32 staging [target_rows, vocab]
        void *stochastic_inverse_rejection_samples_dev_ = nullptr; ///< FP32 [draft_rows, vocab]
        void *stochastic_target_sample_tokens_dev_ = nullptr; ///< INT32 [1, stochastic_target_row_capacity_]
        void *stochastic_draft_sample_tokens_dev_ = nullptr; ///< INT32 [1, stochastic_draft_row_capacity_]
        void *stochastic_draft_sample_probs_dev_ = nullptr; ///< FP32 [1, stochastic_draft_row_capacity_], p(sampled draft token)
        void *mtp_sidecar_condition_token_dev_ = nullptr; ///< INT32 [1, mtp_sidecar_condition_token_capacity_]
        int mtp_sidecar_condition_token_capacity_ = 0; ///< Total staged condition-token scalars across all sidecar slots.
        int mtp_sidecar_condition_token_slot_width_ = 2; ///< Flattened runtime row capacity reserved for each captured sidecar role.
        void *mtp_sidecar_position_ids_dev_ = nullptr; ///< INT32 [mtp_sidecar_condition_token_slot_width_], stable positions for chained device sidecars.
        void *mtp_verifier_position_ids_dev_ = nullptr; ///< INT32 [stochastic_target_row_capacity_], absolute grouped verifier rows derived from live device KV counts.
        void *mtp_verifier_request_lengths_dev_ = nullptr; ///< INT32 [stochastic_batch_output_request_capacity_], valid grouped-verifier width per request.
        void *request_token_ids_dev_ = nullptr; ///< INT32 [batch * max_seq_len], complete immutable external request after admission.
        void *request_position_ids_dev_ = nullptr; ///< INT32 [batch * max_seq_len], absolute positions paired with request_token_ids_dev_.
        void *prefill_chunk_token_ids_dev_ = nullptr; ///< INT32 [batch * activation_seq_len], captured current request window.
        void *prefill_chunk_position_ids_dev_ = nullptr; ///< INT32 [batch * activation_seq_len], captured absolute positions for the current window.
        void *prefill_chunk_geometry_dev_ = nullptr; ///< INT32 [2], current real row count followed by physical row stride.
        int32_t *prefill_chunk_real_rows_dev_ = nullptr; ///< First scalar in PREFILL_CHUNK_GEOMETRY.
        int32_t *prefill_chunk_row_stride_dev_ = nullptr; ///< Second scalar in PREFILL_CHUNK_GEOMETRY.
        void *mtp_shifted_prefill_token_ids_dev_ = nullptr; ///< INT32 [batch * activation_seq_len], graph-produced shifted condition tokens.
        void *mtp_shifted_prefill_position_ids_dev_ = nullptr; ///< INT32 [batch * activation_seq_len], graph-produced shifted absolute positions.
        void *mtp_shifted_prefill_append_lengths_dev_ = nullptr; ///< INT32 [request capacity], graph-produced shifted KV append widths.
        int request_input_row_capacity_ = 0; ///< Complete admitted-request row capacity, independent of activation graph width.
        int prefill_chunk_row_capacity_ = 0; ///< Stable captured chunk-view row capacity.
        int mtp_shifted_prefill_row_capacity_ = 0; ///< Stable shifted-MTP preparation row capacity.
        /**
         * @brief Immutable host source for graph-bucket padding admitted to the GPU.
         *
         * ForwardExecutionEngine may widen a real prefill row to the next captured
         * bucket after the request has crossed the API boundary.  The embedding
         * kernel still reads every bucket row even though attention and LM-head
         * selection mask the padding semantically.  This allocation is sized and
         * filled once during runner initialization, then used only as an immutable
         * source for the unused tail of REQUEST_TOKEN_IDS.  Its stable lifetime
         * makes the asynchronous H2D legal without a host wait or hot-path
         * allocation.
        */
        std::vector<int32_t> request_padding_token_ids_host_;
        /** Stable zero rows used only to initialize setup-capture positions. */
        std::vector<int32_t> request_setup_zero_rows_host_;
        DeviceRequestBatchGeometryLayout request_batch_geometry_layout_; ///< Typed model-lifetime layout of REQUEST_BATCH_GEOMETRY.
        void *request_batch_geometry_dev_ = nullptr; ///< Base of the arena-owned lengths-plus-stride record.
        void *request_sequence_lengths_dev_ = nullptr; ///< First INT32 request length in REQUEST_BATCH_GEOMETRY.
        const int32_t *request_row_stride_dev_ = nullptr; ///< Final INT32 scalar in REQUEST_BATCH_GEOMETRY.
        int request_sequence_lengths_capacity_ = 0; ///< Number of request rows reserved in the arena allocation.
        int request_sequence_lengths_active_count_ = 0; ///< Rows populated for the current request-batched prefill.
        /**
         * @brief Stable host source for one asynchronous geometry admission.
         *
         * `state_.sequence_lengths` advances as soon as graph execution is
         * submitted, so lending its storage to an asynchronous H2D copy races the
         * DMA reader. This fixed-capacity record contains every real length and
         * the padded row stride. It remains unchanged until the next admission,
         * whose single device writer is ordered after the prior graph's reuse
         * event. Lengths and stride therefore cannot acquire different epochs.
         */
        std::vector<int32_t> request_batch_geometry_host_;
        void *mtp_verifier_input_tokens_dev_ = nullptr; ///< INT32 stable compact verifier token row/matrix.
        sampling_math::MTPCommittedVerifierIdentityRecord *
            mtp_committed_verifier_identity_dev_ = nullptr; ///< Device-owned identity coupled to the last response/state commit.
        /**
         * @brief CPU-owned identities coupled to grouped verifier publication.
         *
         * CPU execution has no device generation-controller ledger. Its typed
         * speculative step publisher commits these records only after KV,
         * recurrent state, terminal hidden, and histogram publication all
         * succeed. One persistent slot per admitted request avoids allocating
         * diagnostic state in the inference path.
         */
        std::vector<sampling_math::MTPCommittedVerifierIdentityRecord>
            mtp_committed_verifier_identities_host_;
        void *mtp_verifier_stop_tokens_dev_ = nullptr; ///< INT32 fixed-width stop-token controls read inside captured reducers.
        void *mtp_greedy_penalty_policy_dev_ = nullptr; ///< Graph-stable MTPGreedyPenaltyPolicy written on the exact verifier stream.
        void *mtp_generated_token_counts_dev_ = nullptr; ///< INT32 [vocab], device-authoritative generated-token histogram.
        int mtp_generated_token_count_capacity_ = 0; ///< Vocabulary extent owned by mtp_generated_token_counts_dev_.
        void *stochastic_topk_partial_vals_dev_ = nullptr; ///< FP32 target/verifier top-k partial scratch.
        void *stochastic_topk_partial_idxs_dev_ = nullptr; ///< INT32 target/verifier top-k partial scratch.
        int stochastic_topk_partial_capacity_ = 0;
        void *stochastic_draft_topk_partial_vals_dev_ = nullptr; ///< FP32 MTP-draft top-k partial scratch.
        void *stochastic_draft_topk_partial_idxs_dev_ = nullptr; ///< INT32 MTP-draft top-k partial scratch.
        int stochastic_draft_topk_partial_capacity_ = 0;
        void *stochastic_verify_tokens_dev_ = nullptr;    ///< INT32 [1, stochastic_target_row_capacity_]
        void *stochastic_verify_accepted_dev_ = nullptr;  ///< INT32 [1, stochastic_target_row_capacity_]
        void *stochastic_verify_accept_probs_dev_ = nullptr; ///< FP32 [1, stochastic_target_row_capacity_]
        void *stochastic_verify_thresholds_dev_ = nullptr;   ///< FP32 [1, stochastic_target_row_capacity_]
        void *stochastic_batch_output_tokens_dev_ = nullptr; ///< INT32 [request, stochastic_batch_output_token_stride_]
        void *stochastic_batch_output_meta_dev_ = nullptr;   ///< INT32 [request, 10]
        sampling_math::MTPFirstTransactionDiagnosticRecord *
            mtp_first_transaction_diagnostic_dev_ = nullptr; ///< Optional transaction-zero evidence retained entirely on device until a fatal mirrored mismatch.
        std::unique_ptr<PinnedHostScratch> stochastic_batch_output_host_scratch_;
        std::unique_ptr<PinnedHostScratch> device_generation_terminal_host_scratch_;
        std::unique_ptr<PinnedDispatchTicketScratch>
            device_generation_dispatch_ticket_host_scratch_;
        std::shared_ptr<void> device_generation_dispatch_ticket_ready_event_;
        std::shared_ptr<void> device_generation_terminal_host_ready_event_;
        std::optional<sampling_math::DeviceGenerationDispatchTicket>
            last_device_generation_dispatch_ticket_;
        /** Typed scheduler/ticket/terminal lifecycle for hosted generation. */
        HostedDeviceGenerationCursor hosted_device_generation_cursor_{};

        /**
         * @brief Device representation stored in a stochastic verifier row slot.
         *
         * CompactDistribution rows own `(token_id, probability)` top-k tables.
         * ProcessedLogits rows own full-vocab logits after temperature/top-k/top-p
         * processing. Keeping this explicit prevents the verifier from silently
         * interpreting stale compact buffers as processed rows, or vice versa.
         */
        enum class StochasticRowFormat
        {
            Empty,
            CompactDistribution,
            ProcessedLogits,
        };

        std::vector<int> stochastic_target_top_k_;
        std::vector<int> stochastic_draft_top_k_;
        std::vector<StochasticRowFormat> stochastic_target_row_formats_;
        std::vector<StochasticRowFormat> stochastic_draft_row_formats_;

        /**
         * @brief Producer stream for each compact stochastic distribution slot.
         *
         * A distribution build can consume logits produced by captured replay on
         * a non-context stream. The later sample kernel must run on that same
         * explicit stream so it observes the finished token/probability arrays
         * without inserting a host synchronization. Slots are cleared when the
         * paired sampler consumes them or when request-scoped state is reset.
         */
        std::vector<void *> stochastic_target_distribution_streams_;

        /**
         * @brief Producer streams for MTP draft compact distributions.
         *
         * Draft distributions are normally built immediately after a sidecar
         * graph. Keeping the stream here lets stochastic sampling follow a
         * deferred sidecar graph replay on CUDA/ROCm without touching the device
         * null stream or synchronizing the whole context.
         */
        std::vector<void *> stochastic_draft_distribution_streams_;

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
            bool verifier_consumer_pending = false;
        };

        /**
         * @brief Event-backed ownership state for deferred shifted-MTP-KV writes.
         *
         * Unlike logits handoff, the consumer is not a single sampler call: the
         * next boundary may be another sidecar append, a cache truncate, or
         * accepted-state publication.  Keeping one explicit event-backed slot
         * makes that ownership transfer visible and testable.
         */
        struct PendingShiftedMTPKVReadyState
        {
            std::shared_ptr<void> event;
            void *producer_stream = nullptr;
            uint64_t mutation_generation = 0;
            bool valid = false;
        };

        /**
         * @brief Event-backed ownership state for deferred verifier row state.
         *
         * A multi-row verifier may finish asynchronously and hand logits to a
         * sampler first. Publication still has to wait for the same verifier
         * graph before restoring accepted row state. This event is deliberately
         * separate from PendingLogitsStreamHandoff so logits consumers cannot
         * erase publication's dependency.
         */
        struct PendingAllPositionVerifierStateReadyState
        {
            std::shared_ptr<void> event;
            void *producer_stream = nullptr;
            bool valid = false;
        };

        /**
         * @brief Event-backed ownership state for accepted verifier publication.
         *
         * Publication restores KV and hybrid recurrent state on the verifier
         * stream.  The next decode may replay on a fresh graph stream, so it
         * must wait on this event before reading live state.
         */
        struct PendingAcceptedSpecPublicationReadyState
        {
            std::shared_ptr<void> event;
            void *producer_stream = nullptr;
            bool valid = false;
            uint64_t live_state_epoch = 0;
        };
        /**
         * @brief Event-backed source ownership for async logical checkpoints.
         *
         * A checkpoint snapshot owns its payload destination.  This runner-side
         * state owns the matching dependency on the live source tensors: no
         * later graph may mutate KV/GDN state until this event has completed.
         */
        struct PendingLivePrefixCheckpointReadyState
        {
            std::shared_ptr<void> event;
            void *producer_stream = nullptr;
            bool valid = false;
            uint64_t live_state_epoch = 0;
        };
        /**
         * @brief Event-backed readiness state for async prefix restore/truncate.
         *
         * Restore and truncate mutate the live KV/GDN/terminal-hidden surface
         * on an explicit stream. Future graph replay may use a different stream,
         * so every consumer must first wait on this event.
         */
        struct PendingLivePrefixMutationReadyState
        {
            std::shared_ptr<void> event;
            void *producer_stream = nullptr;
            bool valid = false;
            uint64_t live_state_epoch = 0;
            std::vector<PrefixBlockHandle> retained_payload_sources;
            std::vector<std::shared_ptr<void>> retained_device_sources;
        };

        /**
         * @brief Restore payload owners awaiting their exact GPU completion event.
         *
         * Consuming the live-state handoff only queues a stream dependency; it
         * does not prove that the producer copies have completed on the host.
         * Moving the event and source owners here allows subsequent restores to
         * publish a fresh event while prior owners retire by nonblocking query.
         */
        struct PendingPrefixPayloadUse
        {
            std::shared_ptr<void> completion_event;
            std::vector<PrefixBlockHandle> retained_payload_sources;
            std::vector<std::shared_ptr<void>> retained_device_sources;

            bool valid() const
            {
                return completion_event &&
                       (!retained_payload_sources.empty() ||
                        !retained_device_sources.empty());
            }
        };

        /**
         * @brief Event-backed ownership for a grouped-prefill terminal-row archive.
         *
         * The event is allocated once during runner initialization and reused
         * only after the prior publication has been consumed. The producer
         * reads the main hidden tensor and writes the stable MTP terminal-hidden
         * buffer. Consumers wait entirely on device, allowing the host to
         * schedule the next shifted-prefill sidecar immediately.
         */
        struct PendingMTPPrefillTerminalArchiveReadyState
        {
            std::shared_ptr<void> event;
            void *producer_stream = nullptr;
            bool valid = false;
        };

        /**
         * @brief Event-backed completion of request-owned GPU state reset.
         *
         * GPU ring metadata, GDN recurrence, and short-conv state are reset on
         * the worker context's explicit state stream. Captured prefill/decode
         * graphs may replay on a different stream, so request reset publishes
         * one durable dependency after the final cache-owned reset launch.
         */
        struct PendingRequestStateResetReadyState
        {
            std::shared_ptr<void> event;
            void *producer_stream = nullptr;
            bool valid = false;
        };

        /**
         * @brief Completion of one cold graph-build device-state publication.
         *
         * Main and MTP graph builds own separate instances because a sidecar can
         * be materialized while the main graph cache remains live. Each state is
         * published after immutable descriptor/table writes and consumed exactly
         * once by the first execution stream for that newly built graph.
         */
        struct PendingGraphBuildDeviceStateReadyState
        {
            std::shared_ptr<void> event;
            void *producer_stream = nullptr;
            bool valid = false;
        };

        /**
         * @brief Event-backed ownership of one external GPU request admission.
         *
         * The event allocation is created during runner initialization, before
         * any request enters the hot path. Admission records it after the final
         * token/position/length H2D copy. The main graph consumes it with a
         * stream wait; no device or stream synchronization is permitted.
         */
        struct PendingRequestInputAdmissionReadyState
        {
            std::shared_ptr<void> event;
            void *producer_stream = nullptr;
            bool valid = false;
            int token_count = 0;
            int request_count = 0;
        };

        /**
         * @brief Exact event handoff for the reusable generation controller.
         *
         * Admission publishes initialized rows, verifier summary consumes them
         * before deriving its transaction budget, and fused accepted-state
         * publication republishes the mutated controller.  Reusing either
         * arena row without consuming this event is a fatal lifecycle error.
         */
        struct PendingDeviceGenerationStateReadyState
        {
            std::shared_ptr<void> event;
            void *producer_stream = nullptr;
            bool valid = false;
            int request_count = 0;
        };

        /**
         * @brief Semantic writer crossing the generation-controller event edge.
         *
         * Admission initializes only controller-owned scheduling storage. A
         * committed transaction and the composed-parent terminal also close
         * the verifier identity row written by their retained preparation
         * graph. Encoding that distinction here prevents admission from
         * falsely publishing an unmaterialized verifier row and prevents a
         * terminal parent from leaving the arena pointed at transaction zero.
         */
        enum class DeviceGenerationStatePublicationKind : std::uint8_t
        {
            /// Request admission initialized response, control, and ticket rows.
            Admission,
            /// The externally scheduled transaction committed state and identity.
            CommittedTransaction,
            /// The retained generation parent completed every admitted transaction.
            Terminal,
        };

        /**
         * @brief Device event that protects the reusable request-input arena bank.
         *
         * Admission readiness proves that H2D publication completed before the
         * first graph reader starts. It does not prove that the main graph and
         * shifted-MTP sidecars have stopped reading those addresses. The final
         * consumer chain records this second event on the main transaction stream;
         * the next admission transfer waits for it before overwriting any token,
         * position, or sequence-length row.
         *
         * `consumers_started` is host-side lifecycle metadata only. It never polls
         * device state or serializes GPU work; it makes an omitted release
         * publication a fatal state-machine error instead of a timing-dependent
         * overwrite.
         */
        struct PendingRequestInputReuseReadyState
        {
            std::shared_ptr<void> event;
            void *producer_stream = nullptr;
            bool valid = false;
            bool consumers_started = false;
        };

        /**
         * @brief Durable publication for the newest successful forward graph.
         *
         * Deliberately absent is a producer-stream field. Capture-stream
         * ownership belongs to `ForwardGraphCache`, whose invalidation may destroy
         * that stream before prefix harvest or another observer runs. The
         * preallocated event is the complete cross-lifetime dependency token.
         */
        struct ForwardGraphOutputReadyState
        {
            std::shared_ptr<void> event;
            DeviceId device = DeviceId::invalid();
            bool valid = false;
            uint64_t publication_sequence = 0;
            ForwardExecutionRole execution_role =
                ForwardExecutionRole::MainInference;
            bool is_decode = false;
            bool all_position_logits = false;
            ForwardCompletionScope completion_scope =
                ForwardCompletionScope::ModelForwardOnly;
            int graph_seq_len = 0;
            int graph_batch_size = 0;
        };

        /**
         * @brief Independent durable publications for semantic forward outputs.
         *
         * Main-model logits and grouped-verifier logits remain live at the same
         * time during an MTP transaction. They therefore cannot share one event
         * slot whose metadata is overwritten by whichever graph launched last.
         * Each semantic surface owns one persistent event, while the monotonic
         * sequence records which publication is newest for consumers that
         * explicitly accept either kind.
         */
        struct ForwardGraphOutputReadySet
        {
            ForwardGraphOutputReadyState main_forward;
            ForwardGraphOutputReadyState grouped_verifier;
            uint64_t next_publication_sequence = 1;
        };

        /** Select the publication slot owned by one concrete producer role. */
        ForwardGraphOutputReadyState *forwardGraphOutputPublicationFor(
            ForwardExecutionRole role,
            bool all_position_logits);

        /** Select one typed publication, or the newest slot for `Any`. */
        const ForwardGraphOutputReadyState *selectForwardGraphOutputPublication(
            ForwardGraphOutputKind required_kind) const;

        /** Invalidate semantic provenance while retaining persistent events. */
        void clearForwardGraphOutputPublications() noexcept;

        /**
         * @brief One fixed profiler-event pair owned for the runner lifetime.
         *
         * A pair is available when both owners have a reference count of one;
         * borrowers receive ordinary shared owners that travel with pending
         * measurements or compact outcome handles. No mutex is required because
         * one DeviceGraphOrchestrator is scheduled by one rank-control thread.
         */
        struct PersistentGpuTimingEventPair
        {
            std::shared_ptr<void> start_event;
            std::shared_ptr<void> stop_event;
        };

        /**
         * @brief Maximum number of GPU streams reading one logical-state publication.
         *
         * This bounds stream topology, not MTP depth or request count. Repeated
         * readers on one stream reuse that stream's completion lane, so
         * arbitrary work on the same stream still occupies one slot. Sixteen
         * lanes cover the main, sidecar, sampling, checkpoint, maintenance, and
         * result streams with growth room while keeping every event allocated
         * before graph capture. Exhaustion is a fatal topology error.
         */
        static constexpr size_t
            kDeviceResidentLogicalStateReaderStreamCapacity = 16;

        std::vector<StochasticSampleReadyState> stochastic_target_sample_ready_;
        std::vector<StochasticSampleReadyState> stochastic_draft_sample_ready_;
        PendingShiftedMTPKVReadyState shifted_mtp_kv_ready_;
        PendingAllPositionVerifierStateReadyState all_position_verifier_state_ready_;
        mutable PendingAcceptedSpecPublicationReadyState
            accepted_spec_publication_ready_;
        mutable PendingLivePrefixCheckpointReadyState live_prefix_checkpoint_ready_;
        mutable PendingLivePrefixMutationReadyState live_prefix_mutation_ready_;
        mutable std::vector<PendingPrefixPayloadUse>
            pending_prefix_payload_uses_;
        mutable PendingMTPPrefillTerminalArchiveReadyState
            mtp_prefill_terminal_archive_ready_;
        PendingRequestStateResetReadyState request_state_reset_ready_;
        PendingGraphBuildDeviceStateReadyState
            main_graph_build_device_state_ready_;
        PendingGraphBuildDeviceStateReadyState
            mtp_graph_build_device_state_ready_;
        PendingRequestInputAdmissionReadyState
            request_input_admission_ready_;
        PendingDeviceGenerationStateReadyState
            device_generation_state_ready_;
        PendingRequestInputReuseReadyState
            request_input_reuse_ready_;
        ForwardGraphOutputReadySet forward_graph_output_ready_;
        MainLogitsPublicationState current_main_logits_publication_;
        std::shared_ptr<void>
            device_resident_mtp_transaction_ready_event_;
        std::shared_ptr<void>
            device_resident_logical_sequence_state_ready_event_;
        std::array<
            std::shared_ptr<void>,
            kDeviceResidentLogicalStateReaderStreamCapacity>
            device_resident_logical_state_reader_completion_events_;
        std::vector<std::shared_ptr<void>>
            mtp_outcome_response_ready_event_pool_;
        std::vector<PersistentGpuTimingEventPair>
            mtp_gpu_timing_event_pool_;

        /**
         * @brief Retire cache payload owners whose restore event has completed.
         *
         * Runtime calls use nonblocking event queries. Destruction passes
         * `wait_for_all=true` so no asynchronous copy can outlive the cache
         * allocation that supplied its source bytes.
         */
        void retirePendingPrefixPayloadUses(bool wait_for_all) const;

        /**
         * @brief Retire every published GPU producer before arena-backed storage dies.
         *
         * C++ destroys members in reverse declaration order.  The arena is declared
         * after graph caches and the executor because those objects are configured
         * from arena bindings during initialization; consequently the arena would
         * otherwise release its allocations while cached graph streams can still
         * reference them.  The orchestrator destructor calls this method before
         * ordinary member destruction begins.
         *
         * The method waits only on exact, preallocated publication events.  It is
         * therefore a teardown ownership boundary rather than a device-wide or
         * stream-wide synchronization.  A missing event for a state marked valid,
         * or a backend failure while waiting, is fatal because releasing the arena
         * after either condition would permit use-after-release GPU work.
         */
        void retirePublishedDeviceWorkBeforeArenaRelease() noexcept;

        /**
         * @brief Persistent explicit stream for compact stochastic response copies.
         *
         * The vLLM-style MTP path flushes compact outcome tokens to host only
         * after device-resident state publication has already happened.  That
         * response bridge still needs an explicit non-default stream, but HIP
         * stream create/destroy is expensive enough to show up in fixed-depth
         * decode timings.  Keep one owned stream per orchestrator/device and
         * reuse it for each synchronized response flush.
         */
        std::shared_ptr<void> stochastic_outcome_response_bridge_stream_;

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
            bool all_tokens_from_host = false;
            int first_draft_slot = 0;
            int draft_token_count = 0;
            int total_verifier_input_tokens = 0;
            std::vector<int32_t> host_tokens;
        };
        std::optional<PendingMTPVerifierDeviceTokenPlan>
            pending_mtp_verifier_device_token_plan_;
        struct PendingMTPVerifierDeviceTokenBatchPlan
        {
            int request_count = 0;
            int logical_padded_seq_len = 0;
            std::vector<DeviceMTPVerifierInputBatchRequest> requests;
        };
        std::optional<PendingMTPVerifierDeviceTokenBatchPlan>
            pending_mtp_verifier_device_token_batch_plan_;
        struct MaterializedMTPVerifierDeviceTokenRow
        {
            bool valid = false;
            bool first_token_from_device = false;
            int first_target_sample_slot = -1;
            int total_verifier_input_tokens = 0;
            int draft_token_count = 0;
        };
        MaterializedMTPVerifierDeviceTokenRow
            materialized_mtp_verifier_device_token_row_;
        struct MaterializedMTPVerifierDeviceTokenBatch
        {
            bool valid = false;
            int request_count = 0;
            int logical_padded_seq_len = 0;
            int physical_padded_seq_len = 0;
            int total_token_capacity = 0;
        };
        MaterializedMTPVerifierDeviceTokenBatch
            materialized_mtp_verifier_device_token_batch_;

        /**
         * @brief Stable arena storage for request-lifetime logical sequence state.
         *
         * Verifier metadata scratch belongs to `WorkspaceAllocator`, whose
         * contiguous device block may be replaced when a different graph shape
         * is materialized. Published request state has a longer lifetime: the
         * next verifier, sidecar, scheduler, and KV publication all consume it.
         * This seven-row arena allocation therefore owns one initialization
         * scratch row plus the six canonical published values independently of
         * every graph-workspace generation. The scratch row exists because the
         * initialization primitive also emits a base-cache count, even when the
         * first scalar sidecar runs before verifier metadata workspace exists.
         *
         * The derive and prefill-initialization kernels write these rows
         * directly. There is deliberately no workspace-to-mailbox replay copy:
         * direct publication is both cheaper and makes stale workspace pointer
         * adoption impossible by construction.
         */
        struct DeviceResidentLogicalSequenceStateStorage
        {
            /// Number of distinct INT32 rows in the packed arena allocation.
            static constexpr size_t kFieldCount = 7;

            /// Field order in `BufferId::MTP_LOGICAL_SEQUENCE_STATE`.
            enum class Field : size_t
            {
                InitializationBaseCachedTokensScratch = 0,
                TargetCachedTokens,
                AcceptedStateCounts,
                NextConditionTokens,
                AllDraftsAcceptedFlags,
                StoppedFlags,
                PublicationOkFlags,
            };

            int request_capacity = 0;
            int32_t *initialization_base_cached_tokens_scratch_device = nullptr;
            int32_t *target_cached_tokens_device = nullptr;
            int32_t *accepted_state_counts_device = nullptr;
            int32_t *next_condition_tokens_device = nullptr;
            int32_t *all_drafts_accepted_flags_device = nullptr;
            int32_t *stopped_flags_device = nullptr;
            int32_t *publication_ok_flags_device = nullptr;

            /**
             * @brief Request count covered by the most recent live publication.
             *
             * This marker belongs to the durable arena owner rather than the
             * event-fenced mailbox view. Metadata preparation may clear that
             * view while an already-issued sidecar or publication handle still
             * addresses these rows across a graph-workspace replacement.
             */
            int published_request_count = 0;

            /// Live-state epoch associated with @ref published_request_count.
            uint64_t published_live_state_epoch = 0;

            /**
             * @brief Bind typed row pointers to one stable packed device block.
             *
             * @param base First INT32 element of the arena allocation.
             * @param capacity Number of request entries reserved in every row.
             * @return true when the complete storage layout is bound.
             */
            bool bind(void *base, int capacity)
            {
                clear();
                if (!base || capacity <= 0)
                    return false;

                request_capacity = capacity;
                auto *elements = static_cast<int32_t *>(base);
                auto row = [&](Field field)
                {
                    return elements +
                           static_cast<size_t>(field) *
                               static_cast<size_t>(request_capacity);
                };
                initialization_base_cached_tokens_scratch_device =
                    row(Field::InitializationBaseCachedTokensScratch);
                target_cached_tokens_device = row(Field::TargetCachedTokens);
                accepted_state_counts_device = row(Field::AcceptedStateCounts);
                next_condition_tokens_device = row(Field::NextConditionTokens);
                all_drafts_accepted_flags_device = row(Field::AllDraftsAcceptedFlags);
                stopped_flags_device = row(Field::StoppedFlags);
                publication_ok_flags_device = row(Field::PublicationOkFlags);
                return validFor(/*request_count=*/1);
            }

            /** @brief Return true when every durable row covers the request batch. */
            bool validFor(int request_count) const
            {
                return request_count > 0 &&
                       request_count <= request_capacity &&
                       initialization_base_cached_tokens_scratch_device != nullptr &&
                       target_cached_tokens_device != nullptr &&
                       accepted_state_counts_device != nullptr &&
                       next_condition_tokens_device != nullptr &&
                       all_drafts_accepted_flags_device != nullptr &&
                       stopped_flags_device != nullptr &&
                       publication_ok_flags_device != nullptr;
            }

            /**
             * @brief Redirect publication outputs from transient scratch to this owner.
             *
             * The remaining fields in @p pointers continue to address
             * graph-workspace scratch. Only values that survive the publication
             * transaction are rebound to arena storage.
             */
            bool bindPublicationOutputs(
                MTPSpecDecodeMetadataDevicePointers *pointers,
                int request_count) const
            {
                if (!pointers || !validFor(request_count))
                    return false;
                pointers->target_cached_tokens = target_cached_tokens_device;
                pointers->accepted_state_counts = accepted_state_counts_device;
                pointers->next_condition_tokens = next_condition_tokens_device;
                pointers->all_drafts_accepted_flags =
                    all_drafts_accepted_flags_device;
                pointers->stopped_flags = stopped_flags_device;
                pointers->publication_ok_flags = publication_ok_flags_device;
                return true;
            }

            /**
             * @brief Mark the arena rows as request-lifetime published state.
             *
             * @param request_count Number of request rows made visible.
             * @param live_state_epoch State epoch in which the rows were produced.
             * @return true when the publication fits the bound arena storage.
             */
            bool markPublished(int request_count, uint64_t live_state_epoch)
            {
                if (!validFor(request_count))
                    return false;
                published_request_count = request_count;
                published_live_state_epoch = live_state_epoch;
                return true;
            }

            /** @brief Return true when durable rows remain live for this epoch. */
            bool hasLivePublication(uint64_t live_state_epoch) const
            {
                return validFor(published_request_count) &&
                       published_live_state_epoch == live_state_epoch;
            }

            /**
             * @brief Compare the stable allocation identity of two snapshots.
             *
             * Workspace replacement is allowed to rebind transient metadata
             * consumers, but it must never alter any request-lifetime row or
             * its capacity.
             */
            bool aliasesSameRowsAs(
                const DeviceResidentLogicalSequenceStateStorage &other) const
            {
                return request_capacity == other.request_capacity &&
                       initialization_base_cached_tokens_scratch_device ==
                           other.initialization_base_cached_tokens_scratch_device &&
                       target_cached_tokens_device ==
                           other.target_cached_tokens_device &&
                       accepted_state_counts_device ==
                           other.accepted_state_counts_device &&
                       next_condition_tokens_device ==
                           other.next_condition_tokens_device &&
                       all_drafts_accepted_flags_device ==
                           other.all_drafts_accepted_flags_device &&
                       stopped_flags_device == other.stopped_flags_device &&
                       publication_ok_flags_device ==
                           other.publication_ok_flags_device;
            }

            /**
             * @brief Retire publication liveness without releasing arena rows.
             *
             * Arena storage persists for the orchestrator lifetime. Only true
             * request/session boundaries retire the logical values it contains.
             */
            void retirePublication()
            {
                published_request_count = 0;
                published_live_state_epoch = 0;
            }

            void clear()
            {
                *this = {};
            }
        };
        DeviceResidentLogicalSequenceStateStorage
            device_resident_logical_sequence_state_storage_;

        /**
         * @brief Operation that most recently published the resident logical row.
         *
         * The arena addresses are intentionally stable across the request lifetime,
         * so an address alone cannot identify which kernel transaction produced the
         * values currently protected by the mailbox event.  Keeping provenance as a
         * closed enum makes diagnostics and lifecycle assertions exhaustive whenever
         * a new publication path is introduced.
         */
        enum class DeviceResidentLogicalStatePublicationKind : uint8_t
        {
            RequestBatchConditionAdvance,
            TargetSampleInitialization,
            RestoredPrefixDecodeBridgeInitialization,
            AcceptedSpecState,
            MainBatchSampleInitialization,
        };

        /// Return the stable diagnostic name for one typed publication operation.
        static const char *deviceResidentLogicalStatePublicationKindName(
            DeviceResidentLogicalStatePublicationKind kind);

        /**
         * @brief Ordered boundaries preserved by the device-only MTP state tracer.
         *
         * The tracer is intentionally phase based rather than call-site based.
         * Each phase answers one ownership question in the publication
         * transaction: did derivation produce valid rows, did the mailbox expose
         * those rows, and did either asynchronous graph family change them? A
         * closed inventory keeps future publication phases from silently
         * disappearing from failure reports.
         */
        enum class DeviceResidentLogicalStateDiagnosticPhase : size_t
        {
            PrimaryDerive = 0,
            MailboxPublished,
            MaintenanceEntry,
            MaintenanceExit,
            SidecarEntry,
            SidecarExit,
            ObservationMailboxReady,
            ObservationPrefixCheckpointJoined,
            ObservationPublishedHandoffsJoined,
            ObservationGraphProducersJoined,
            ObservationMTPTransactionJoined,
            Count,
        };

        /** @brief Canonical logical-state rows copied at every diagnostic phase. */
        enum class DeviceResidentLogicalStateDiagnosticField : size_t
        {
            TargetCachedTokens = 0,
            AcceptedStateCounts,
            NextConditionTokens,
            AllDraftsAcceptedFlags,
            StoppedFlags,
            PublicationOkFlags,
            Count,
        };

        /// Return the stable diagnostic label for one device snapshot phase.
        static const char *deviceResidentLogicalStateDiagnosticPhaseName(
            DeviceResidentLogicalStateDiagnosticPhase phase);

        /// Return the stable diagnostic label for one copied logical-state row.
        static const char *deviceResidentLogicalStateDiagnosticFieldName(
            DeviceResidentLogicalStateDiagnosticField field);

        /**
         * @brief Persistent phase-major storage for device-only MTP diagnostics.
         *
         * Rows are laid out as `[phase][slot][field][request]`. The allocation is
         * optional and exists only when
         * `LLAMINAR_MTP_DEVICE_PHASE_SNAPSHOTS=1` was present at runner
         * initialization. Snapshotting uses only ordered D2D copies; this owner
         * never establishes a host mirror or participates in production
         * dispatch.
         */
        struct DeviceResidentLogicalStateDiagnosticStorage
        {
            static constexpr size_t kPhaseCount =
                static_cast<size_t>(
                    DeviceResidentLogicalStateDiagnosticPhase::Count);
            static constexpr size_t kFieldCount =
                static_cast<size_t>(
                    DeviceResidentLogicalStateDiagnosticField::Count);
            /**
             * @brief Unique destinations available to each phase.
             *
             * Reusing one destination across asynchronous producer streams
             * would require a stream wait and could hide the missing edge being
             * diagnosed. Four thousand monotonic slots cover the focused
             * transaction reproduction while keeping the optional allocation
             * small. Exhaustion fails the diagnostic instead of wrapping and
             * perturbing production ordering.
             */
            static constexpr size_t kSlotsPerPhase = 4096;
            static constexpr size_t kSnapshotRowCount =
                kPhaseCount * kSlotsPerPhase * kFieldCount;

            /**
             * @brief Rows reserved for one coherent deep-probe publication.
             *
             * Direct tiny D2H operations from the live arena owner are not a
             * supported observation contract. A deep probe first copies the
             * complete seven-row publication here on its ordered stream, then
             * exports this diagnostic allocation at an explicit result boundary.
             */
            static constexpr size_t kProbeStagingRowCount =
                DeviceResidentLogicalSequenceStateStorage::kFieldCount;

            static constexpr size_t kRowCount =
                kSnapshotRowCount + kProbeStagingRowCount;

            int request_capacity = 0;
            int32_t *base_device = nullptr;

            /** @brief Bind the phase table to one arena-owned device block. */
            bool bind(void *base, int capacity)
            {
                clear();
                if (!base || capacity <= 0)
                    return false;
                base_device = static_cast<int32_t *>(base);
                request_capacity = capacity;
                return true;
            }

            /** @brief Return true when the allocation covers a request batch. */
            bool validFor(int request_count) const
            {
                return base_device != nullptr &&
                       request_count > 0 &&
                       request_count <= request_capacity;
            }

            /** @brief Resolve one phase/field destination row. */
            int32_t *row(
                DeviceResidentLogicalStateDiagnosticPhase phase,
                size_t slot,
                DeviceResidentLogicalStateDiagnosticField field) const
            {
                if (!base_device)
                    return nullptr;
                const size_t phase_index = static_cast<size_t>(phase);
                const size_t field_index = static_cast<size_t>(field);
                if (phase_index >= kPhaseCount ||
                    slot >= kSlotsPerPhase ||
                    field_index >= kFieldCount)
                    return nullptr;
                const size_t row_index =
                    (phase_index * kSlotsPerPhase + slot) * kFieldCount +
                    field_index;
                return base_device +
                       row_index * static_cast<size_t>(request_capacity);
            }

            /** @brief First row of the deep-probe staging region. */
            int32_t *probeStagingBase() const
            {
                if (!base_device)
                    return nullptr;
                return base_device +
                       kSnapshotRowCount *
                           static_cast<size_t>(request_capacity);
            }

            /** @brief Retire the binding without releasing arena storage. */
            void clear()
            {
                *this = {};
            }
        };

        /**
         * @brief Event and provenance for one reusable diagnostic phase slot.
         *
         * Every record names a monotonic destination slot. The event can be
         * re-recorded without coupling producer streams because no destination
         * is reused. The latest event and slot are sufficient for the fatal
         * observer to materialize the newest value.
         */
        struct DeviceResidentLogicalStateDiagnosticRecord
        {
            std::shared_ptr<void> ready_event;
            void *producer_stream = nullptr;
            int request_count = 0;
            uint64_t live_state_epoch = 0;
            uint64_t publication_generation = 0;
            size_t slot = 0;
            size_t snapshot_count = 0;
            bool valid = false;
        };

        /**
         * @brief Event-fenced view of arena-owned logical sequence state.
         *
         * DGO treats target cached tokens as both the next logical position and
         * the next request sequence length. Every pointer below must alias
         * `device_resident_logical_sequence_state_storage_`; the mailbox adds
         * transaction epoch and stream ordering, but never owns or adopts a
         * graph-workspace address.
         */
        struct DeviceResidentLogicalSequenceStateMailbox
        {
            int request_count = 0;
            const int32_t *target_positions_device = nullptr;
            const int32_t *target_sequence_lengths_device = nullptr;
            const int32_t *accepted_state_counts_device = nullptr;
            const int32_t *next_condition_tokens_device = nullptr;
            const int32_t *all_drafts_accepted_flags_device = nullptr;
            const int32_t *stopped_flags_device = nullptr;
            const int32_t *publication_ok_flags_device = nullptr;
            void *producer_stream = nullptr;
            std::shared_ptr<void> ready_event;
            uint64_t live_state_epoch = 0;
            uint64_t publication_generation = 0;
            DeviceResidentLogicalStatePublicationKind publication_kind =
                DeviceResidentLogicalStatePublicationKind::MainBatchSampleInitialization;

            bool valid() const
            {
                return request_count > 0 &&
                       target_positions_device != nullptr &&
                       target_sequence_lengths_device != nullptr &&
                       accepted_state_counts_device != nullptr &&
                       next_condition_tokens_device != nullptr &&
                       all_drafts_accepted_flags_device != nullptr &&
                       stopped_flags_device != nullptr &&
                       publication_ok_flags_device != nullptr &&
                       producer_stream != nullptr &&
                       ready_event != nullptr;
            }

            /**
             * @brief True when @p handle is exactly this live mailbox.
             *
             * The resident logical-state handoff is a stream/event/pointer
             * ownership contract, not just a bag of device addresses.  Keeping
             * the comparison here prevents sidecar, forward, and future
             * scheduler consumers from drifting apart as Phase 10 moves more
             * state away from host-owned getters.
             */
            bool ownsHandle(
                const DeviceResidentLogicalSequenceStateHandle &handle,
                uint64_t current_live_state_epoch) const
            {
                return valid() &&
                       handle.valid() &&
                       live_state_epoch == current_live_state_epoch &&
                       handle.live_state_epoch == live_state_epoch &&
                       handle.request_count == request_count &&
                       handle.device.is_valid() &&
                       handle.target_positions_device == target_positions_device &&
                       handle.target_sequence_lengths_device == target_sequence_lengths_device &&
                       handle.accepted_state_counts_device == accepted_state_counts_device &&
                       handle.next_condition_tokens_device == next_condition_tokens_device &&
                       handle.all_drafts_accepted_flags_device == all_drafts_accepted_flags_device &&
                       handle.stopped_flags_device == stopped_flags_device &&
                       handle.publication_ok_flags_device == publication_ok_flags_device &&
                       handle.stream == producer_stream &&
                       handle.ready_event == ready_event.get() &&
                       handle.publication_generation == publication_generation;
            }

            void clear()
            {
                *this = {};
            }
        };
        DeviceResidentLogicalSequenceStateMailbox
            device_resident_logical_sequence_state_mailbox_;

        /**
         * @brief Host admission gate for one reusable device logical-state bank.
         *
         * The lock is held only while host code enqueues a reader and records
         * that reader's completion event. It never spans GPU completion. A
         * writer takes the exclusive side long enough to close admission and
         * queue fan-in waits, then the explicit pending-writer state keeps new
         * readers out until the replacement mailbox event is published.
         */
        struct DeviceResidentLogicalStateAdmissionSynchronization
        {
            std::shared_mutex admission_mutex;
            std::mutex reader_completion_mutex;
        };
        std::unique_ptr<DeviceResidentLogicalStateAdmissionSynchronization>
            device_resident_logical_state_admission_sync_ =
                std::make_unique<
                    DeviceResidentLogicalStateAdmissionSynchronization>();

        /**
         * @brief In-flight replacement of the reusable logical-state rows.
         *
         * This is a fail-closed transaction. Once active, every reader is
         * rejected until recordDeviceResidentLogicalSequenceStateMailbox()
         * publishes the exact replacement stream/event edge. Any abandoned
         * writer therefore stops inference instead of exposing partially
         * overwritten rows.
         */
        struct PendingDeviceResidentLogicalStateWriter
        {
            void *stream = nullptr;
            uint64_t replaced_publication_generation = 0;
            const char *writer_name = nullptr;

            bool valid() const
            {
                return stream != nullptr;
            }

            void begin(
                void *writer_stream,
                uint64_t replaced_generation,
                const char *name)
            {
                stream = writer_stream;
                replaced_publication_generation = replaced_generation;
                writer_name = name;
            }

            void clear()
            {
                *this = {};
            }
        } pending_device_resident_logical_state_writer_;

        /**
         * @brief One preallocated completion lane in a publication read epoch.
         *
         * Readers on one stream are already ordered by that stream, so its
         * newest event record subsumes every earlier read. Readers on different
         * streams own different events and never wait one another. The
         * replacement writer performs the only fan-in.
         */
        struct DeviceResidentLogicalStateReaderCompletion
        {
            void *stream = nullptr;
            uint64_t publication_generation = 0;
            uint64_t completion_sequence = 0;

            bool validFor(uint64_t generation) const
            {
                return stream != nullptr &&
                       publication_generation == generation &&
                       completion_sequence > 0;
            }

            void clear()
            {
                *this = {};
            }
        };

        /**
         * @brief Fan-out/fan-in epoch protecting reusable logical-state rows.
         *
         * The immutable publication event is the root of the epoch. Every
         * asynchronous reader waits only that root and publishes completion to
         * its own preallocated stream lane. A replacement writer waits the root
         * plus every occupied reader lane before its first store. The epoch
         * survives mailbox retirement because clearing a typed handle cannot
         * cancel device work that is already in flight.
         */
        struct DeviceResidentLogicalStateAccessEpoch
        {
            void *producer_stream = nullptr;
            std::shared_ptr<void> publication_ready_event;
            uint64_t publication_generation = 0;
            uint64_t access_sequence = 0;
            size_t reader_stream_count = 0;
            std::array<
                DeviceResidentLogicalStateReaderCompletion,
                kDeviceResidentLogicalStateReaderStreamCapacity>
                reader_completions{};

            bool valid() const
            {
                return producer_stream != nullptr &&
                       publication_ready_event != nullptr &&
                       publication_generation > 0 && access_sequence > 0 &&
                       reader_stream_count <= reader_completions.size();
            }

            void begin(
                void *stream,
                std::shared_ptr<void> ready_event,
                uint64_t generation)
            {
                producer_stream = stream;
                publication_ready_event = std::move(ready_event);
                publication_generation = generation;
                ++access_sequence;
                reader_stream_count = 0;
                for (auto &completion : reader_completions)
                    completion.clear();
            }

            void clear()
            {
                *this = {};
            }
        } device_resident_logical_state_access_epoch_;

        /**
         * @brief Fixed-storage transaction for one forward graph mailbox read.
         *
         * The forward prelude and graph launch are separated by engine-owned
         * dynamic-parameter work, so a lexical DGO read scope cannot span them.
         * This state carries only the exact stream and publication generation;
         * it performs no allocation and is cleared only after the engine's
         * mandatory post-launch completion hook records the stream's reader
         * completion lane.
         */
        struct PendingDeviceResidentLogicalStateForwardRead
        {
            void *stream = nullptr;
            uint64_t publication_generation = 0;
            std::shared_lock<std::shared_mutex> admission_lock;

            bool valid() const
            {
                return stream != nullptr && publication_generation > 0 &&
                       admission_lock.owns_lock();
            }

            void clear()
            {
                admission_lock = {};
                stream = nullptr;
                publication_generation = 0;
            }
        } pending_device_resident_logical_state_forward_read_;

        /**
         * @brief Scope one asynchronous reader of the reusable mailbox rows.
         *
         * Construction queues the reader behind the immutable publication-ready
         * event, never behind another reader. Destruction records completion in
         * the exact stream's preallocated epoch lane, including early-return
         * paths. The replacement writer then waits every occupied lane. A
         * completion-publication failure is fatal because allowing inference to
         * continue would permit a later writer to race an unknown in-flight
         * device read.
         */
        class DeviceResidentLogicalStateReadScope
        {
        public:
            DeviceResidentLogicalStateReadScope(
                DeviceGraphOrchestrator &owner,
                void *stream,
                const char *consumer_name);
            ~DeviceResidentLogicalStateReadScope() noexcept;

            DeviceResidentLogicalStateReadScope(
                const DeviceResidentLogicalStateReadScope &) = delete;
            DeviceResidentLogicalStateReadScope &operator=(
                const DeviceResidentLogicalStateReadScope &) = delete;
            DeviceResidentLogicalStateReadScope(
                DeviceResidentLogicalStateReadScope &&) = delete;
            DeviceResidentLogicalStateReadScope &operator=(
                DeviceResidentLogicalStateReadScope &&) = delete;

            bool ready() const { return ready_; }
            bool complete() noexcept;

        private:
            DeviceGraphOrchestrator *owner_ = nullptr;
            void *stream_ = nullptr;
            const char *consumer_name_ = nullptr;
            uint64_t publication_generation_ = 0;
            std::shared_lock<std::shared_mutex> admission_lock_;
            bool ready_ = false;
            bool active_ = false;
        };

        uint64_t device_resident_logical_state_publication_generation_ = 0;
        DeviceResidentLogicalStateDiagnosticStorage
            device_resident_logical_state_diagnostic_storage_;
        std::array<
            DeviceResidentLogicalStateDiagnosticRecord,
            DeviceResidentLogicalStateDiagnosticStorage::kPhaseCount>
            mutable device_resident_logical_state_diagnostic_records_;

        /**
         * @brief Persistent request-session owner for device-resident MTP KV state.
         *
         * The transaction points directly at each shifted cache's canonical
         * device count row. Its stream/event fence advances after every
         * mutation, but its identity remains stable until request reset. Compact
         * outcomes and logical-state handles hold child-local leases to this
         * shared object, so RankOrchestrator never has to rediscover ownership
         * through an ambient mailbox.
         */
        std::shared_ptr<DeviceResidentMTPTransactionState>
            device_resident_mtp_transaction_;

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
         * @brief Allocate graph workspace with an explicit family lifetime.
         *
         * Main prefill graphs reserve row-scaled scratch for the largest serial
         * participant before capture. Decode, verifier, sidecar, publication,
         * and catch-up graphs bind exact logical layouts into the same physical
         * family block. Explicit event handoffs make those layouts mutually
         * exclusive in time.
         */
        bool ensureDeviceWorkspaceAllocated(
            const ComputeGraph &graph,
            int workspace_seq_len,
            WorkspaceGraphFamilyPolicy graph_family_policy) override;

        /**
         * @brief Bind one exact graph under its explicit mathematical role.
         *
         * The role is supplied by ForwardExecutionEngine from ForwardInput and is
         * never reconstructed from M, current position, or output mode.
         */
        bool ensureDeviceWorkspaceAllocated(
            const ComputeGraph &graph,
            int workspace_seq_len,
            WorkspaceGraphFamilyPolicy graph_family_policy,
            WorkspaceGraphParticipantRole participant_role) override;

        /**
         * @brief Allocate the first workspace generation from a complete graph family.
         *
         * @param graph Primary forward graph.
         * @param workspace_seq_len Active primary-graph row count.
         * @param graph_family_policy Physical lifetime policy.
         * @param exact_serial_participants Materialized, role-typed prefill,
         *        MTP, publication, and helper graphs whose exact topology must
         *        be present before capture.
         * @return true when every graph consumer is bound to one stable layout.
         */
        bool ensureDeviceWorkspaceAllocated(
            const ComputeGraph &graph,
            int workspace_seq_len,
            WorkspaceGraphFamilyPolicy graph_family_policy,
            WorkspaceGraphParticipantRole primary_role,
            const std::vector<WorkspaceGraphParticipant> &
                exact_serial_participants);

        /**
         * @brief Materialize every backend MTP workspace topology without executing it.
         *
         * The returned graphs are short-lived declarations used by the first
         * workspace plan on CPU, CUDA, and ROCm. They are built through the same
         * model graph builder as production sidecars, so new stages and
         * workspace names automatically join the family instead of requiring
         * byte-count updates here. CPU stage objects retain raw workspace
         * addresses just as captured GPU graph nodes do, so CPU sidecars must
         * participate before the family allocation is published.
         *
         * @pre @p owned_mtp_graphs is empty. Primary-lane declarations have a
         *      separate owner, so this MTP contributor cannot erase, reorder,
         *      or replace them.
         */
        bool buildMTPWorkspaceFamilyManifest(
            std::vector<std::unique_ptr<ComputeGraph>> &
                owned_mtp_graphs);

        /**
         * @brief Return the current workspace generation for a device.
         */
        uint64_t workspaceGeneration(DeviceId device) const override;
        int residentGraphRows() const override
        {
            return state_.activation_seq_len;
        }

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

        /// Resolve the active GlobalTP/NodeTP domain context for scalar MTP coordination.
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

        /**
         * @brief Nonzero identity of the request-state session currently owned
         * by this orchestrator.
         *
         * Epoch zero is reserved by device dispatch tickets as their invalid,
         * unpublished value.  A newly constructed orchestrator already owns a
         * valid first session, before any caller has a reason to clear request
         * state, so its epoch starts at one.  Every request-state reset or
         * execution-cache invalidation advances the epoch and thereby makes a
         * ticket from an earlier session fail lifecycle authentication.
         */
        uint64_t session_epoch_ = 1;
        /**
         * Root-published generation consumed by graph-native sparse MoE stages.
         *
         * Zero means this runner has no admitted distributed overlay request.
         * It is assigned only through the typed IInferenceRunner contract at
         * initialization and request reset, never inferred from graph capture
         * or a stage-object execution count.
         */
        uint64_t moe_overlay_collective_request_generation_ = 0;
        /**
         * Rank-owned control authority for heterogeneous sparse graph launches.
         *
         * Single-domain native device-resident overlay execution deliberately leaves
         * this null: its device controller remains the sole authority and no
         * host ticket is introduced into the captured generation loop.
         */
        std::shared_ptr<MoEOverlayInferenceTransactionCoordinator>
            moe_overlay_inference_transaction_coordinator_;
        /** Stable LocalTP participant index authenticated by the coordinator. */
        int moe_overlay_inference_transaction_participant_index_ = -1;
        /**
         * Setup-owned GPU terminal receipt for this continuation participant.
         * Null only for CPU participants; every GPU child contributes its exact
         * event to the coordinator's rank-wide calibration fence.
         */
        std::shared_ptr<IMoEOverlayInferenceCompletionEvent>
            moe_overlay_inference_completion_fence_;
        uint64_t live_replay_state_epoch_ = 1;
        uint64_t live_state_mutation_count_ = 0;
        uint64_t live_state_accepted_publications_ = 0;
        uint64_t live_state_rejected_corrections_ = 0;
        uint64_t live_state_prefix_restores_ = 0;
        uint64_t live_state_prefix_truncates_ = 0;
        uint64_t live_state_session_resets_ = 0;
        /**
         * @brief Generation of auxiliary shifted-MTP KV publications.
         *
         * Shifted sidecar KV is not main live state: appending it changes no
         * main KV row, request position, recurrent state, terminal hidden row,
         * or captured graph binding. Its producer event and this generation
         * therefore form an independent lifecycle domain. Keeping it separate
         * prevents an auxiliary append from invalidating a still-current
         * device-resident logical-state mailbox.
         */
        uint64_t shifted_mtp_kv_mutation_generation_ = 0;
        LivePrefixMutationReason last_live_state_mutation_reason_ =
            LivePrefixMutationReason::Unknown;
        std::string last_live_state_mutation_operation_;
        uint64_t device_sampling_counter_ = 0;

        bool compute_all_position_logits_ = false;
        bool live_mtp_request_batch_condition_ = false;
        bool compute_row_indexed_all_position_logits_ = false;
        int row_indexed_all_position_logits_row_count_ = 0;
        int request_batched_prefill_logits_row_count_ = 0;

        enum class GreedyVerifierOutcomeGraphState
        {
            Idle,
            Armed,
            Produced,
        };

        /**
         * @brief Transaction-varying controls for one graph-owned greedy result.
         *
         * Request-constant stop tokens are intentionally absent. They are
         * configured through configureMTPRequestStopTokens() and published once
         * to `mtp_verifier_stop_tokens_dev_` by request admission. This object
         * owns only values that can legitimately change between verifier
         * transactions.
         */
        struct GreedyVerifierOutcomeGraphTransaction
        {
            GreedyVerifierOutcomeGraphState state =
                GreedyVerifierOutcomeGraphState::Idle;
            int verifier_token_count = 0;
            int stop_token_count = 0;
            MTPRequestPenaltyPolicy penalty_policy;
        };

        MTPVerifierOutcomeGraphMode mtp_verifier_outcome_graph_mode_ =
            MTPVerifierOutcomeGraphMode::Disabled;
        GreedyVerifierOutcomeGraphTransaction
            greedy_verifier_outcome_graph_transaction_;

        /**
         * @brief Host policy staged for the next request admission.
         *
         * This fixed-width array is not read by graph execution. It is the
         * bounded source for one admission-time H2D publication and for strict
         * API validation that later verifier calls name the same request
         * policy.
         */
        std::array<int32_t,
                   sampling_math::kSpeculativeBatchMaxStopTokens>
            mtp_request_stop_tokens_ = {
                -1, -1, -1, -1, -1, -1, -1, -1};
        int mtp_request_stop_token_count_ = 0;
        std::optional<uint64_t>
            mtp_request_stop_tokens_published_session_epoch_;

        /**
         * @brief Immutable host policy staged for one admission-time GPU write.
         *
         * This object never contains transaction state.  The corresponding
         * device buffer's history predicate is mutated only by captured
         * accepted-state publication after this session marker is established.
         */
        MTPRequestPenaltyPolicy mtp_request_penalty_policy_;
        MTPRequestPenaltyPolicy mtp_published_request_penalty_policy_;
        std::optional<uint64_t>
            mtp_request_penalty_policy_published_session_epoch_;

        /// Runner-owned graph metadata workspace for vLLM-style MTP verification.
        MTPSpecDecodeMetadataWorkspaceBinding mtp_spec_decode_metadata_binding_{
            MTPSpecDecodeMetadataShape{1, 3}};

        /// Host copy of the verifier row plan waiting to be uploaded before replay.
        std::optional<MTPSpecDecodeVerifierInputPlan>
            pending_mtp_spec_verifier_input_plan_;
        /**
         * @brief Whether BASE_CACHED_TOKENS holds a pre-verifier device snapshot.
         *
         * prepareAllPositionVerifierGraphMetadata() captures the live KV count
         * before verifier graph replay appends target rows.  Publication consumes
         * that snapshot later, after the scoped verifier row plan has been
         * cleared, so this flag is intentionally separate from
         * pending_mtp_spec_verifier_input_plan_.
         */
        bool mtp_publication_base_cache_snapshot_ready_ = false;
        int mtp_publication_base_cache_snapshot_request_count_ = 0;
        /**
         * @brief Immutable per-request main-KV metadata bases for publication.
         *
         * Each checkpoint is persistent VRAM allocated during runner setup.
         * prepareAllPositionVerifierGraphMetadata() overwrites the matching
         * checkpoint on the exact verifier stream before graph execution.
         * Accepted-state publication later derives every layer's head/count
         * pair from this base, making partially restored ring metadata
         * structurally impossible.
         */
        std::vector<DeviceKVSequenceStateCheckpoint>
            mtp_publication_main_kv_base_checkpoints_;
        bool mtp_publication_main_kv_base_checkpoints_ready_ = false;
        int mtp_publication_main_kv_base_checkpoint_request_count_ = 0;
        /**
         * @brief Scoped internal permission for grouped decode-equivalent publish.
         *
         * The normal batch publisher begins by checking
         * supportsMTPSpecStatePublication(), because external callers should not
         * be able to publish direct all-position verifier rows unless that strong
         * contract is advertised.  publishGroupedDecodeEquivalentMTPSpecStateBatch()
         * flips this flag only while it is on the stack after validating the
         * concrete grouped publication preconditions.
         */
        bool grouped_decode_equivalent_spec_publication_scope_ = false;

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

        /// Last applied expert replica assignment, replayed onto rebuilt graphs.
        ExpertReplicaSet current_expert_replica_set_;
        int current_expert_replica_participant_id_ = -1;
        uint64_t current_expert_replica_epoch_ = 0;

        /// Last applied routed expert ownership/cache masks, used to invalidate
        /// MoE-sensitive graph cache keys after non-replica ownership changes.
        std::vector<std::vector<bool>> current_expert_masks_;
        uint64_t current_expert_mask_epoch_ = 0;
        uint64_t moe_runtime_movement_epoch_ = 0;

        /** Exact active transient CPU LLEP transaction, if any. */
        CPUCurrentBatchLLEPTransactionState *active_cpu_llep_state_ = nullptr;
        ICPUCurrentBatchLLEPExpertConsumer *active_cpu_llep_expert_consumer_ = nullptr;
        const IGlobalTPContext *active_cpu_llep_tp_ctx_ = nullptr;

        /// Optional expert weight payload provider for metadata-based host retention (owned)
        std::unique_ptr<ExpertWeightPayloadProvider> expert_payload_provider_;

        /// Model-context-owned prepared weight store (Phase 4-5)
        std::shared_ptr<PreparedWeightStore> prepared_weight_store_;

        /// Frozen model weight set for audit/validation (Phase 6)
        std::unique_ptr<FrozenModelWeightSet> frozen_weight_set_;
        std::unique_ptr<FrozenModelWeightSet> decode_replicated_dense_weight_set_;

        /**
         * @brief Select the weight set that owns graph-native MTP decode work.
         *
         * Phase-split replicated decode deliberately has two frozen views: a
         * primary tensor-parallel prefill view and a replicated dense decode view.
         * The MTP sidecar verifies decode rows, so it must bind from the decode
         * view whenever dense_tp_decode_replicated is active.  Non phase-split
         * paths continue to bind from the primary frozen set.
         *
         * @param caller Short method name used in diagnostics.
         * @return Selected weight set, or nullptr after logging a configuration error.
         */
        const FrozenModelWeightSet *selectMTPDecodeWeightSet(const char *caller) const;

        /// Build FrozenModelWeightSet from pre-resolved layer weights (Phase 6)
        void buildFrozenWeightSet(
            const ModelWeights &weights,
            const std::unordered_map<int, LayerWeights> &resolved_layers,
            int first_layer, int last_layer);

        /**
         * @brief Validate the immutable terminal-hidden mailbox for MTP sidecar input.
         *
         * Most scalar callers consume row zero, while grouped verification and
         * shifted-cache catch-up consume several rows. Initialization reserves
         * the maximum configured row capacity in one arena-owned tensor on
         * every backend. This method only validates that owner and capacity;
         * it must never allocate, resize, or rebind because CPU executable
         * graphs and GPU captured graphs both retain the tensor address.
         *
         * @param min_rows Minimum logical row capacity required by the caller.
         * @return true when the initialization-time mailbox satisfies the
         *         requested shape and backend-storage contract.
         */
        bool ensureMTPTerminalHiddenBuffer(int min_rows = 1);

        /// Execute the cached graph-native row select used for MTP terminal hidden refresh.
        bool executeMTPTerminalHiddenRowSelect(int row_idx, int seq_len, void *stream = nullptr);

        /// Execute a cached graph-native hidden row select into an MTP buffer.
        bool executeMTPHiddenRowSelect(
            TensorBase *input,
            BufferId input_buffer_id,
            TensorBase *output,
            BufferId output_buffer_id,
            MTPTerminalHiddenRowSelectGraphCache &cache,
            const char *node_name,
            int row_idx,
            int seq_len,
            void *stream = nullptr);

        /// Execute a cached graph-native hidden rows select into an MTP buffer.
        bool executeMTPHiddenRowsSelect(
            TensorBase *input,
            BufferId input_buffer_id,
            TensorBase *output,
            BufferId output_buffer_id,
            MTPTerminalHiddenRowsSelectGraphCache &cache,
            const char *node_name,
            int row_start,
            int row_count,
            int seq_len,
            void *stream = nullptr);

        /// Execute a cached hidden-row select whose row indices are device-produced metadata.
        bool executeMTPHiddenRowsSelectFromDeviceMetadata(
            TensorBase *input,
            TensorBase *output,
            MTPTerminalHiddenRowsSelectGraphCache &cache,
            const char *node_name,
            const char *row_buffer_name,
            const int32_t *row_indices_device,
            int row_count,
            int seq_len,
            void *stream = nullptr);

        /**
         * @brief Execute one pre-materialized rows graph through full capture.
         *
         * The caller's stream is the producer before this graph and the
         * consumer after it. Two cache-owned event edges bridge that stream to
         * the graph's persistent capture stream without a host wait. Warmup,
         * capture, and replay all use the same strict monolithic policy, so a
         * successful cache can later be cloned into a device-controlled parent
         * graph without changing its computation.
         */
        bool executeMTPTerminalHiddenRowsCaptured(
            MTPTerminalHiddenRowsSelectGraphCache &cache,
            void *producer_consumer_stream,
            const char *perf_context);

        /**
         * @brief Materialize all GPU terminal-hidden publication graph objects.
         *
         * This runs after the largest-participant workspace family has fixed
         * arena and workspace addresses. It builds immutable contiguous and
         * accepted-state families plus one device-geometry graph for every
         * legal request count. Prompt width is resident data, not graph
         * identity. Decode execution treats a missing or stale cache as fatal
         * instead of allocating or rebuilding in the hot path.
         */
        bool materializeMTPTerminalHiddenPublicationGraphs();

        /**
         * @brief Build one typed GPU rows-select graph against current bindings.
         *
         * @param cache Destination cache whose previous graph is replaced during setup.
         * @param node_name Stable graph/node diagnostic name.
         * @param row_index_source Fixed contiguous or external-device source policy.
         * @param selected_row_count Immutable output row capacity.
         * @param fixed_contiguous_row_start Immutable first row for fixed-range mode.
         * @param row_buffer_name Stable producer-field name used only for diagnostics and cache identity.
         * @param external_device_row_indices Exact producer-owned row-index address for external mode.
         */
        bool materializeMTPTerminalHiddenRowsSelectGraph(
            MTPTerminalHiddenRowsSelectGraphCache &cache,
            const char *node_name,
            HiddenStateRowsSelectStage::DeviceRowIndexSource row_index_source,
            int selected_row_count,
            int fixed_contiguous_row_start = 0,
            const char *row_buffer_name = nullptr,
            const int32_t *external_device_row_indices = nullptr,
            int request_index = -1);

        /**
         * @brief Publish one terminal row per padded GPU request from resident lengths.
         *
         * The graph reads `REQUEST_BATCH_GEOMETRY` directly. Setup materializes
         * one graph per request count; current prompt width remains device data.
         * No host request-length vector, row-index upload, or runtime graph
         * construction participates in refresh.
         */
        bool selectMTPTerminalHiddenRowsFromDeviceRequestLengths(
            int request_count,
            int request_row_stride,
            int total_rows,
            void *stream);

        /**
         * @brief Publish the next shifted-prefill hidden range from device KV progress.
         *
         * The caller names only immutable graph geometry: request identity and
         * grouped row count. The captured stage derives its source range from
         * canonical main/shifted KV counters after consuming the prior shifted
         * mutation event. No host row cursor or runtime graph construction is
         * permitted by this contract.
         */
        bool selectMTPTerminalHiddenRowsFromShiftedPrefillProgress(
            int request_index,
            int request_count,
            int row_count,
            int total_rows,
            void *stream);

        /// Execute a cached graph-native arbitrary hidden-row select into an MTP buffer.
        bool executeMTPHiddenRowsSelect(
            TensorBase *input,
            BufferId input_buffer_id,
            TensorBase *output,
            BufferId output_buffer_id,
            MTPTerminalHiddenRowsSelectGraphCache &cache,
            const char *node_name,
            const std::vector<int> &row_indices,
            int seq_len,
            void *stream = nullptr);

        /// Copy a contiguous verifier/prefill row range into the stable MTP input buffer.
        bool selectMTPTerminalHiddenRows(
            int row_start,
            int row_count,
            int seq_len,
            void *stream = nullptr);

        /// Copy arbitrary verifier rows into the stable MTP input buffer.
        bool selectMTPTerminalHiddenRows(
            const std::vector<int> &row_indices,
            int seq_len,
            void *stream = nullptr);

        /// Copy accepted verifier rows named by the device-resident MTP metadata workspace.
        bool selectMTPTerminalHiddenRowsFromDeviceAcceptedState(
            int row_count,
            int seq_len,
            void *stream = nullptr);

        /**
         * @brief Derive Phase-10 publication metadata from a resident verifier outcome.
         *
         * This helper is the device-side half of
         * publishAcceptedMTPSpecStateBatchFromDeviceOutcome().  It does not
         * mutate KV, positions, or terminal hidden state; instead it validates
         * that @p request belongs to the last all-position verifier graph,
         * snapshots the canonical pre-verifier cache count device-to-device
         * into the runner metadata workspace, and asks the backend to derive
         * accepted restore rows,
         * target cache counts, accepted-state counts, and per-request validity
         * flags on the verifier stream.  Later publication slices consume those
         * workspace buffers directly, so no code should reconstruct these
         * counts from host-copied stochastic metadata.
         */
        bool prepareDeviceResidentMTPSpecPublicationMetadata(
            const DeviceSpeculativePublicationRequest &request,
            std::string *error = nullptr);

        /**
         * @brief Seal verifier geometry into a newly produced compact outcome.
         *
         * The retained all-position forward graph is the physical-stride
         * authority. The outcome producer supplies only the real logical width;
         * this helper proves the graph belongs to the same request batch and to
         * the canonical scalar-bucket policy before publishing both values in
         * @p handle. Publication requests cannot override either value.
         *
         * @param handle Compact outcome being completed by the producer.
         * @param request_count Number of logical requests represented by the graph.
         * @param logical_rows_per_request Number of real verifier rows per request.
         * @param producer_name Stable diagnostic name for the producing path.
         * @return true when the retained graph proves and seals the geometry.
         */
        bool sealDeviceSpeculativeOutcomeVerifierGeometry(
            DeviceSpeculativeOutcomeHandle *handle,
            int request_count,
            int logical_rows_per_request,
            const char *producer_name);

        /**
         * @brief Build or validate one fixed-depth proposal publication graph.
         *
         * Setup resolves the fresh sidecar logit row, branch-history pointers,
         * deterministic argmax workspace, and destination slot.  It does not
         * enqueue sampling work.
         */
        bool materializeMTPDraftTokenPublicationGraph(
            int row,
            int slot,
            const MTPRequestPenaltyPolicy &penalty_policy,
            std::string *error = nullptr);

        /**
         * @brief Replay one proposal publication after its exact sidecar producer.
         *
         * The private capture stream consumes the sidecar event and publishes
         * completion back to that same producer stream.  No synchronization or
         * host token bridge is permitted.
         */
        bool executeMTPDraftTokenPublicationCaptured(
            void *producer_stream,
            int row,
            int slot,
            std::string *error = nullptr);

        /**
         * @brief Find or capture the exact resident grouped-verifier prelude.
         *
         * The returned typed key deterministically addresses one preallocated
         * registry slot until the workspace generation is invalidated. No live
         * token or KV value is inspected on host while selecting the graph.
         */
        bool materializeMTPVerifierPreparationGraph(
            const MTPVerifierPreparationStage::Params &params,
            MTPVerifierPreparationGraphKey *key,
            std::string *error = nullptr);

        /**
         * @brief Replay one verifier-preparation graph between explicit events.
         *
         * @p producer_stream already carries every sample, shifted-KV, logical
         * state, and row-plan dependency. Completion is returned to that exact
         * stream before the main verifier graph may consume the prepared rows.
         */
        bool executeMTPVerifierPreparationCaptured(
            void *producer_stream,
            const MTPVerifierPreparationGraphKey &key,
            std::string *error = nullptr);

        /**
         * @brief Pair a just-completed verifier forward with its exact registry owner.
         *
         * Preparation and forward both receive the same immutable physical
         * geometry. After the forward succeeds, this method resolves that typed
         * key directly, validates the complete ForwardGraphSignature, and stores
         * it on that registry owner. Parent composition never consults mutable
         * recency state.
         *
         * @param key Exact request-count/physical-width registry address.
         * @param error Optional first violated pairing invariant.
         * @return true when the exact replay-ready pair is durably published.
         */
        bool bindMTPVerifierForwardGraphPair(
            const MTPVerifierPreparationGraphKey &key,
            std::string *error = nullptr);

        /**
         * @brief Validate one retained sidecar plan for hosted branch replay.
         *
         * The plan may contain explicit heterogeneous sparse boundaries but
         * must already be in steady replay state with stable device token and
         * position inputs. No warmup or capture is performed by validation.
         */
        bool validateHostedMTPSidecarReplay(
            MTPSidecarCaptureRole role,
            int row_count,
            const ComputeGraph **out_graph,
            std::string *error = nullptr) const;

        /**
         * @brief Enqueue an exact retained sidecar on its production replay stream.
         *
         * The scheduler and sidecar streams are connected with setup-owned
         * events. Every continuation participant enters the shared transaction
         * group before its sparse stages receive the root-assigned wire id.
         */
        bool replayHostedMTPSidecar(
            MTPSidecarCaptureRole role,
            int draft_depth,
            void **out_producer_stream,
            std::string *error = nullptr);

        /**
         * @brief Enqueue the exact retained grouped-verifier production plan.
         *
         * The preparation capture has already populated stable device rows.
         * This method stamps the transaction binding, replays the paired
         * ForwardExecutionEngine cache entry, and returns its exact stream.
         */
        bool replayHostedMTPGroupedVerifier(
            const ForwardGraphSignature &signature,
            int draft_depth,
            void **out_producer_stream,
            std::string *error = nullptr);

        /**
         * @brief Resolve one typed preparation key to its deterministic slot.
         *
         * Scalar keys must already be canonical power-of-two/final physical
         * buckets. Multi-request keys retain the exact padded stride. This
         * function is the sole registry-address calculation.
         *
         * @param key Structural graph key to validate.
         * @param error Optional first violated key/registry invariant.
         * @return Flat preallocated slot, or nullopt on invalid geometry.
         */
        [[nodiscard]] std::optional<size_t>
        mtpVerifierPreparationRegistryIndex(
            const MTPVerifierPreparationGraphKey &key,
            std::string *error = nullptr) const;

        /**
         * @brief Resolve the verifier-preparation policy owned by this request.
         *
         * A zero active controller count identifies standalone grouped
         * verification. Any positive count must exactly match the request and
         * the persistent storage geometry; partial or stale admission is a hard
         * lifecycle error rather than an implicit policy change.
         *
         * @param request_count Logical request rows entering the verifier.
         * @param error Optional first violated lifecycle invariant.
         * @return Exact capture policy, or nullopt for malformed admission.
         */
        [[nodiscard]] std::optional<
            MTPVerifierPreparationControlPolicy>
        mtpVerifierPreparationControlPolicyForRequest(
            int request_count,
            std::string *error = nullptr) const;

        /**
         * @brief Resolve the immutable physical-width policy for this request.
         *
         * Dynamic scalar generation changes verifier depth in the resident
         * controller after publication. It therefore owns one maximum-width
         * graph envelope. Fixed, observe-only, standalone, and request-batched
         * transactions retain their bounded logical geometry. Malformed
         * controller admission is rejected through the same lifecycle check as
         * verifier preparation; geometry may never guess controller ownership.
         *
         * @param request_count Logical requests entering the verifier.
         * @param error Optional first violated lifecycle invariant.
         * @return Exact graph-width policy, or nullopt for malformed admission.
         */
        [[nodiscard]] std::optional<MTPVerifierPhysicalWidthPolicy>
        mtpVerifierPhysicalWidthPolicyForRequest(
            int request_count,
            std::string *error = nullptr) const;

        /**
         * @brief Assemble the configured sampling policy into its backend executable.
         *
         * Fixed/observe policy clones one producer-ordered transaction into a
         * branch. Dynamic policy clones one complete transaction per legal
         * depth. CUDA composes native conditional nodes; HIP retains those
         * branches and instantiates an isolated dispatch-ticket publisher.
         * Every fragment must already be replay-ready at @p draft_depth capture
         * capacity and match the active request/workspace identity. This method
         * composes and instantiates only; it never launches generation or reads
         * mutable controller state on the host.
         *
         * @param request_count Exact admitted controller row count.
         * @param draft_depth Fixed transaction depth, or maximum capture depth
         *        for dynamic policy.
         * @param sampling_mode Exact compact-outcome topology retained by the
         *        verifier child and transaction tail.
         * @param error Optional first violated capture/composition invariant.
         * @return true when the complete backend policy is executable.
         */
        bool materializeMTPDeviceGenerationLoopGraph(
            int request_count,
            int draft_depth,
            DeviceGenerationSamplingMode sampling_mode,
            std::string *error = nullptr);

        /**
         * @brief Build or validate the fixed-geometry seeded stochastic fragment.
         *
         * Descriptor inspection is setup-only. The resulting stage captures no
         * host arrays: it retains immutable seeds/geometry and persistent device
         * addresses exclusively.
         */
        bool materializeMTPStochasticSerialOutcomeGraph(
            const DeviceStochasticBatchOutcomeRequest *requests,
            int request_count,
            std::string *error = nullptr);

        /**
         * @brief Execute the seeded stochastic fragment under mandatory capture.
         *
         * The producer stream owns completed target distributions on entry. The
         * same stream consumes compact output after an event-ordered graph replay.
         */
        bool executeMTPStochasticSerialOutcomeCaptured(
            void *producer_stream,
            int request_count,
            int comparison_rows_per_request,
            std::string *error = nullptr);

        /**
         * @brief Build or validate captured stochastic verifier row preparation.
         *
         * Setup resolves persistent arena addresses and freezes row geometry,
         * top-k/top-p policy, temperature, and optional penalty policy. It does
         * not launch work; execution is owned by the strict captured helper
         * below and exported parent-loop composition.
         */
        bool materializeMTPStochasticTargetDistributionGraph(
            int row_count,
            const SamplingParams &params,
            const MTPRequestPenaltyPolicy &penalty_policy,
            int vocab_size,
            std::string *error = nullptr);

        /**
         * @brief Replay captured verifier target preparation after its producer.
         *
         * The verifier stream remains the externally visible transaction
         * owner. Device events order the private capture stream after and then
         * back before that producer; no host or device synchronization occurs.
         */
        bool executeMTPStochasticTargetDistributionCaptured(
            void *producer_stream,
            int row_count,
            std::string *error = nullptr);

        /**
         * @brief Build or validate the immutable accepted-state publication graph.
         *
         * This is graph-setup work only. It resolves persistent device pointers
         * and verifier-stage identities but enqueues no publication kernel.
         */
        bool materializeMTPSpeculativeStatePublicationGraph(
            const DeviceSpeculativePublicationRequest &request,
            const MTPSpecDecodeMetadataDevicePointers &publication_metadata,
            ComputeGraph &verifier_graph,
            int verifier_rows_per_request,
            std::string *error = nullptr);

        /**
         * @brief Replay the publication graph between two explicit event edges.
         *
         * The caller stream owns compact verifier output on entry and consumes
         * all accepted state on return. Warmup/capture/replay occur on the
         * cache-owned stream; both crossings are device event waits.
         */
        bool executeMTPSpeculativeStatePublicationCaptured(
            const DeviceSpeculativePublicationRequest &request,
            std::string *error = nullptr);

        /**
         * @brief Publish host lifecycle metadata after one captured transaction.
         *
         * This method records readiness events and coherence ownership only. It
         * must not launch publication kernels or inspect device values. Parent
         * loop composition will call the equivalent lifecycle once after the
         * terminal loop launch rather than once per transaction.
         */
        bool finalizeMTPSpeculativeStatePublicationLaunch(
            const DeviceSpeculativePublicationRequest &request,
            std::string *error = nullptr);

        /// Drop any stale device logical-state mailbox after request/session mutation.
        void clearDeviceResidentLogicalSequenceStateMailbox();

        /**
         * @brief Retire a logical-state input consumed by one MTP condition graph.
         *
         * Every resident MTP-condition mailbox describes the token, position,
         * and sequence length at the graph's input boundary. Once that graph is
         * submitted, canonical device KV owns the post-forward token count and
         * the input mailbox must not masquerade as the resulting live state.
         * The transition is metadata-only: the access epoch retains the exact
         * reader-completion event so a later writer can safely reuse the arena
         * rows without blocking the host.
         *
         * @param completed_stream Exact stream that submitted the condition graph.
         * @param publication_generation Exact mailbox generation admitted by
         *        that graph's forward prelude.
         * @return true when the matching publication and reader completion were
         *         present and the consumed input handle was retired.
         */
        bool retireConsumedMTPConditionInputMailbox(
            void *completed_stream,
            uint64_t publication_generation);

        /// Retire the request-scoped device MTP transaction at a true session boundary.
        void retireDeviceResidentMTPTransaction();

        /**
         * @brief Advance the request-scoped MTP transaction fence after a mutation.
         *
         * The helper resolves canonical count rows directly from each IKVCache.
         * It creates a transaction only when the current request session has no
         * owner, then updates that owner's producer stream, event, and mutation
         * generation in place. No live-state epoch retargeting is involved.
         *
         * @param request_count Number of active request entries in each count row.
         * @param producer_stream Explicit stream that completed all cache publications.
         * @param producer_name Stable diagnostic label for perfstats.
         * @return true after the transaction fence has been advanced.
         */
        bool recordDeviceResidentMTPTransactionMutation(
            int request_count,
            void *producer_stream,
            const char *producer_name);

        /**
         * @brief Advance the persistent transaction after GPU state restore.
         *
         * Restore replaces cache contents but not the request-scoped IKVCache
         * allocations. The transaction therefore keeps its identity and records
         * a new fence after all restore writes. No host cache count is read or
         * adopted.
         *
         * @param producer_stream Explicit stream containing all restore writes.
         * @param producer_name Stable diagnostic label for perfstats and errors.
         * @return true when no GPU MTP caches exist or the fence was advanced.
         */
        bool recordRestoredDeviceResidentMTPTransaction(
            void *producer_stream,
            const char *producer_name);

        /// Return the current child-local lease, or an empty lease before admission.
        DeviceResidentMTPTransactionLease
        currentDeviceResidentMTPTransactionLease() const;

        /// Validate ownership and queue a wait for the transaction's latest fence.
        bool waitForDeviceResidentMTPTransaction(
            const DeviceResidentMTPTransactionLease &transaction,
            void *consumer_stream,
            const char *consumer_name) const;

        /// Read the device-owned main logical token count for diagnostics/checkpoints.
        std::optional<int> deviceResidentLogicalTokenCountForObservation(
            int request_index,
            void *consumer_stream,
            const char *consumer_name) const;

        /// Read the device-owned shifted-MTP KV token count for diagnostics/checkpoints.
        std::optional<int> deviceResidentShiftedMTPKVTokenCountForObservation(
            int depth,
            int request_index,
            void *consumer_stream,
            const char *consumer_name) const;

        /**
         * @brief Publish the first scalar-decode target as resident logical state.
         *
         * A normal main-model condition forward advances the canonical KV count
         * before its target sampler writes `STOCHASTIC_TARGET_SAMPLE_TOKENS`.
         * The first MTP sidecar already consumes those two device-owned inputs
         * directly, but every later resident consumer must observe the same
         * token/position pair through the durable logical-state mailbox. This
         * helper makes that ownership transition explicit.
         *
         * The publication kernel is appended to the exact target-sample
         * producer stream. It first orders reuse of the single-buffered mailbox
         * rows after the prior publication event, then initializes all durable
         * fields from the target slot and the canonical KV count, and finally
         * records a fresh mailbox event. No host position, allocation, transfer,
         * default stream, or synchronization is permitted.
         *
         * @param target_sample_slot Persistent target-sample row to publish.
         * @param live_position_device Canonical main-KV count on this device.
         * @return true after the complete publication has been enqueued.
         */
        bool publishDeviceResidentLogicalSequenceStateFromTargetSample(
            int target_sample_slot,
            const int32_t *live_position_device);

        /**
         * @brief Initialize the durable logical-state mailbox from device rows.
         *
         * The caller owns source readiness on @p producer_stream. This helper
         * orders reuse of the single arena-backed mailbox, derives the next
         * position/length and condition token with the canonical backend
         * kernel, and event-publishes the resulting rows. It performs no host
         * copy, allocation, synchronization, or graph launch.
         *
         * @param condition_token_device Device row containing one token per request.
         * @param live_position_device Canonical pre-forward KV count per request.
         * @param request_count Number of logical requests in the source rows.
         * @param producer_stream Exact stream on which both sources are ready.
         * @param publication_kind Typed provenance for the resulting mailbox.
         * @param source Stable diagnostic name for the source transaction.
         * @return true after initialization and mailbox publication are enqueued.
         */
        bool publishDeviceResidentLogicalSequenceStateFromDeviceRows(
            const int32_t *condition_token_device,
            const int32_t *live_position_device,
            int request_count,
            void *producer_stream,
            DeviceResidentLogicalStatePublicationKind publication_kind,
            const char *source);

        /// Record an event-fenced view of the arena-owned logical-state rows.
        bool recordDeviceResidentLogicalSequenceStateMailbox(
            int request_count,
            void *producer_stream,
            DeviceResidentLogicalStatePublicationKind publication_kind,
            std::string *error = nullptr);

        /**
         * @brief Preserve one event-ordered logical-state phase on the device.
         *
         * When the opt-in phase tracer is disabled this is a no-op. Otherwise
         * the method copies every canonical logical-state row D2D into the
         * phase table and records the phase's preallocated event. It performs
         * no allocation, H2D/D2H transfer, event wait on the host, stream
         * synchronization, or device synchronization.
         *
         * @param phase Typed transaction boundary being preserved.
         * @param producer_stream Exact stream that observes the source rows.
         * @param request_count Number of valid request entries in each row.
         * @param publication_generation Publication generation represented by
         *        the source rows; callers may name the next generation before
         *        the mailbox object itself is installed.
         */
        void snapshotDeviceResidentLogicalStatePhase(
            DeviceResidentLogicalStateDiagnosticPhase phase,
            void *producer_stream,
            int request_count,
            uint64_t publication_generation) const;

        /**
         * @brief Materialize device phase history at an already-fatal boundary.
         *
         * This is the only host-visible operation in the phase tracer. The
         * caller has already observed invalid production metadata and is about
         * to throw; the method joins each phase event onto @p observation_stream,
         * copies the tiny fixed table once, and returns a compact provenance
         * string for the fatal error. It is never called during successful
         * inference.
         */
        std::string materializeDeviceResidentLogicalStatePhaseDiagnostics(
            void *observation_stream) const;

        /**
         * @brief Admit one external request into persistent GPU-owned input rows.
         *
         * Token IDs, absolute positions, and real request lengths cross the
         * host/device boundary together exactly once. The copies are queued on
         * one explicit admission stream and publish a preallocated event; graph
         * execution waits for that event on its own stream without blocking the
         * host. Every prefill consumer, including shifted MTP publication, then
         * reads the same arena-owned rows.
         *
         * @param tokens Flattened request-major INT32 token rows.
         * @param position_ids Flattened request-major absolute position rows.
         * @param request_real_lengths Immutable logical token count for every
         *        physical request row.
         * @param total_tokens Physical number of token/position elements.
         * @param request_count Number of independent request rows.
         * @param padded_seq_len Physical width of each request row.
         * @return true after all copies and the readiness event are queued.
         */
        bool admitRequestInputsOnDevice(
            const int *tokens,
            const int *position_ids,
            std::span<const int> request_real_lengths,
            int total_tokens,
            int request_count,
            int padded_seq_len);

        /**
         * @brief Publish request-constant MTP stop controls on an ordered stream.
         *
         * The caller must already own the request boundary: normal prefill uses
         * RequestAdmissionTransfer after consuming RequestStateResetReady, while
         * a prefix hit uses PrefixRestoreMutation after the same dependency.
         * Repeated calls in one session are elided.
         *
         * @param producer_stream Explicit stream that owns request publication.
         * @param producer Human-readable lifecycle owner for diagnostics.
         * @return True when this session's device buffer is current.
         */
        bool publishMTPRequestStopTokensOnDevice(
            void *producer_stream,
            const char *producer);

        /**
         * @brief Publish immutable request penalty controls after request reset.
         *
         * Presence/frequency magnitudes are admitted once.  The publication
         * initializes the mutable history predicate to false; accepted-state
         * publication owns every subsequent device-side transition of that bit.
         * Repeated calls in one session only establish an event-ordered read and
         * never overwrite device-produced state.
         */
        bool publishMTPRequestPenaltyPolicyOnDevice(
            void *producer_stream,
            const char *producer);

        /**
         * @brief Order a consumer after the current resident penalty policy.
         *
         * The expected policy contains request constants only.  Dynamic history
         * state is never compared on host; BufferArena orders the consumer after
         * whichever device transaction most recently published that state.
         */
        bool consumeMTPRequestPenaltyPolicyOnDevice(
            const MTPRequestPenaltyPolicy &expected_policy,
            void *consumer_stream,
            const char *consumer);

        /**
         * @brief Queue the current graph stream behind external request admission.
         *
         * The event is consumed exactly once by the main prefill graph. Later
         * shifted-MTP sidecars inherit ordering from that graph's execution
         * provenance and terminal-hidden publication event.
         */
        bool waitForPendingRequestInputAdmission(
            void *consumer_stream,
            const char *consumer_name);

        /**
         * @brief Begin exactly one reader transaction for the reusable input bank.
         *
         * Both host-admitted prefill rows and device-generated serial-decode
         * positions write persistent request-input storage before a graph reads
         * it. This method is the single state transition from writer ownership
         * to active readers for both producer kinds. A second begin before the
         * transitive final reader publishes reuse readiness is a fatal lifecycle
         * violation.
         *
         * @param admission_kind Stable diagnostic name for the writer path.
         * @param token_count Physical input rows exposed to the graph.
         * @param request_count Number of logical request rows.
         */
        void beginRequestInputReaderTransaction(
            const char *admission_kind,
            int token_count,
            int request_count);

        /**
         * @brief Publish completion of every request-owned GPU state reset.
         *
         * Failure is fatal because replaying a graph without this dependency
         * can race GDN/short-conv zeroing and corrupt the new request.
         */
        void publishRequestStateResetReady(
            void *producer_stream,
            const char *producer_name);

        /**
         * @brief Join every old-request producer onto the cache reset stream.
         */
        bool joinPriorDeviceWorkForRequestStateReset(
            void *reset_stream,
            const char *consumer_name);

        /**
         * @brief Queue the first graph behind request-state reset completion.
         */
        bool waitForPendingRequestStateReset(
            void *consumer_stream,
            DeviceTimelineRole consumer_role,
            const char *consumer_name);

        /**
         * @brief Publish the newest complete generation-controller transition.
         * @param producer_stream Exact stream closing the typed transition.
         * @param request_count Number of request rows owned by the controller.
         * @param kind Typed payload surface completed by this producer.
         * @return true after arena and private controller events are coherent.
         */
        bool publishDeviceGenerationStateReady(
            void *producer_stream,
            int request_count,
            DeviceGenerationStatePublicationKind kind);

        /** Consume the current generation-controller transaction exactly once. */
        bool consumeDeviceGenerationStateReady(
            void *consumer_stream,
            DeviceTimelineRole consumer_role,
            int expected_request_count,
            const char *consumer_name);

        /**
         * @brief Publish one graph build's model-owned GPU descriptor writes.
         *
         * A second publication before the prior graph consumes its event is a
         * fatal state-machine error.
         */
        void publishGraphBuildDeviceStateReady(
            PendingGraphBuildDeviceStateReadyState &ready,
            void *producer_stream,
            const char *producer_name);

        /**
         * @brief Queue first graph execution after its cold-build publication.
         */
        bool waitForPendingGraphBuildDeviceStateReady(
            PendingGraphBuildDeviceStateReadyState &ready,
            void *consumer_stream,
            DeviceTimelineRole consumer_role,
            const char *consumer_name);

        /**
         * @brief Publish completion of every reader of the request-input bank.
         *
         * The caller supplies the exact main-transaction stream. If shifted-MTP
         * work still owns a side stream, this method first queues a non-consuming
         * wait for its latest completion event, making the supplied stream the
         * transitive last reader. It then records the preallocated reuse event.
         *
         * Event publication failure after GPU readers have launched is fatal:
         * without this event no later admission can safely establish ordering.
         */
        void publishRequestInputReuseReady(
            void *consumer_completion_stream,
            const char *producer_name);

        /**
         * @brief Publish the initial request-batched logical state after prefill.
         *
         * Terminal prefill sampling writes contiguous target-token slots on the
         * GPU. This helper waits for those slots on @p producer_stream, seeds the
         * persistent metadata workspace from the previously staged device position
         * row and sampled tokens, then records the first resident mailbox. No host
         * logical-state mirror participates in this publication.
         */
        bool initializeDeviceResidentLogicalSequenceStateFromMainBatchSamples(
            int request_count,
            void *producer_stream);

        /**
         * @brief Queue a reader behind the current publication producer.
         *
         * Reader admission waits only the mailbox's immutable ready event. It
         * must never wait another reader because read/read overlap is safe and
         * useful. DeviceResidentLogicalStateReadScope publishes eventual
         * completion to an independent per-stream lane after its work instead.
         */
        bool admitDeviceResidentLogicalSequenceStateRead(
            void *consumer_stream,
            const char *consumer_name,
            uint64_t *publication_generation,
            std::shared_lock<std::shared_mutex> *admission_lock);

        /**
         * @brief Queue a writer after every reader of the reusable state rows.
         *
         * Unlike the mailbox-reader wait, this consults the persistent access
         * fence even after the typed mailbox has been retired. Writers must
         * call it before their first store, not immediately before publication.
         */
        bool waitForDeviceResidentLogicalSequenceStateRowReuse(
            void *writer_stream,
            const char *writer_name);

        /** @brief Publish one reader's final device operation into its epoch lane. */
        bool recordDeviceResidentLogicalSequenceStateReadCompletion(
            void *consumer_stream,
            const char *consumer_name,
            uint64_t publication_generation);

        /**
         * @brief Queue an observation-only wait for resident logical-state metadata.
         *
         * Device-resident MTP publication leaves request positions and sequence
         * lengths in canonical GPU metadata. Host-visible diagnostics must wait
         * for the producer stream before reading paired live KV/GDN/MTP state,
         * but observation never adopts or replaces that device ownership.
         *
         * @param consumer_stream Exact non-null observation stream.
         * @param consumer_name Stable diagnostic identity.
         * @param replacement_writer_pending Optional typed deferral output;
         *        set only when a writer has closed admission but has not yet
         *        recorded the replacement publication event.
         * @return True after the immutable mailbox event wait is enqueued.
         */
        bool waitForDeviceResidentLogicalSequenceStateMailboxForObservation(
            void *consumer_stream,
            const char *consumer_name,
            bool *replacement_writer_pending = nullptr) const;

        /**
         * @brief Whether DGO can consume device-published logical sequence state.
         *
         * The cache can publish its device head/count rows independently, but
         * dynamic positions and sequence lengths must be published from the
         * same transaction metadata and consumed through DGO's resident
         * mailbox. Keeping this as a separate gate prevents partial cache-only
         * publication from being mistaken for a complete live-state handoff.
         */
        bool supportsDeviceResidentLogicalSequenceStatePublication() const;

        /// Copy the latest forward pass terminal hidden row into the stable MTP input buffer.
        bool refreshMTPTerminalHiddenState(
            int seq_len,
            int batch_size,
            void *hidden_producer_stream = nullptr);

        /// Reset input-dependent dynamic state on all cached kernels.
        ///
        /// This is a hard graph-replay invalidation helper. Do not call it from
        /// clear_cache(), because request-boundary reset preserves some captured
        /// GPU graph executables and their kernel-owned dynamic argument tables.
        /// Implemented in .cpp to avoid including KernelFactory.h in the header.
        void resetKernelDynamicState();

        /// Record the intentional preservation of kernel dynamic state.
        ///
        /// Keeping this visible in perf counters makes request-boundary capture
        /// reuse auditable and helps catch accidental hard resets in hot paths.
        void recordKernelDynamicStatePreservedForCapturedReplay(
            const char *reason) const;
    };

} // namespace llaminar2
