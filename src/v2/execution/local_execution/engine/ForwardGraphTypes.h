/**
 * @file ForwardGraphTypes.h
 * @brief Data types for forward graph caching (extracted from DeviceGraphOrchestrator)
 *
 * Contains ForwardGraphSignature, ForwardGraphSignatureHash, and ForwardGraphCache —
 * the key data structures for caching compiled forward graphs between decode steps.
 */

#pragma once

#include "../../../backends/DeviceId.h"
#include "../../../backends/IGPUGraphCapture.h"
#include "../../../backends/IWorkerGPUContext.h"
#include "../graph/DeviceGraphExecutor.h"
#include "../../compute_stages/IComputeStage.h"
#include "../graph/IGraphBuilder.h" // For ForwardOutput
#include "PrefillGraphCache.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace llaminar2
{

    /**
     * @brief Result of a graph build operation
     *
     * Holds either a successfully built ComputeGraph + ForwardOutput, or
     * an error string. Extracted from DeviceGraphOrchestrator for use by the
     * ForwardExecutionEngine.
     */
    class GraphBuildResult
    {
    public:
        GraphBuildResult() = default;
        GraphBuildResult(ComputeGraph graph, ForwardOutput output)
            : graph_(std::move(graph)), output_(output), success_(true) {}
        explicit GraphBuildResult(std::string error)
            : error_(std::move(error)), success_(false) {}

        [[nodiscard]] bool success() const { return success_; }
        [[nodiscard]] bool failed() const { return !success_; }
        [[nodiscard]] const std::string &error() const { return error_; }
        [[nodiscard]] ComputeGraph &graph() { return graph_; }
        [[nodiscard]] const ComputeGraph &graph() const { return graph_; }
        [[nodiscard]] const ForwardOutput &output() const { return output_; }
        [[nodiscard]] ComputeGraph takeGraph() { return std::move(graph_); }
        explicit operator bool() const { return success_; }

    private:
        ComputeGraph graph_;
        ForwardOutput output_{};
        std::string error_;
        bool success_ = false;
    };

    /**
     * @brief Configuration for graph caching behaviour
     */
    struct GraphCacheConfig
    {
        bool enabled = true;         ///< Enable graph caching (Phase 10)
        int decode_seq_len = 4;      ///< Max continuation length that can use decode caching.
        bool cache_attention = true; ///< Cache attention graphs
        bool cache_ffn = true;       ///< Cache FFN graphs
    };

    /**
     * @brief Signature for caching full forward graphs.
     *
     * Captures the execution shape so that graphs built for identical shapes
     * can be reused across decode steps without rebuilding stages/kernels.
     */
    struct ForwardGraphSignature
    {
        int seq_len = 0;
        int batch_size = 0;
        DeviceId device = DeviceId::cpu();
        bool decode = false;
        bool decode_has_history = false; ///< True for decode calls that already have KV/GDN history.
        bool all_position_logits = false;
        bool standard_path = true;
        bool pp_stage_enabled = false;
        int pp_first_layer = -1;
        int pp_last_layer = -1;
        bool pp_has_embedding = false;
        bool pp_has_lm_head = false;
        bool is_bucketed_prefill = false;
        int bucket_seq_len = 0;
        uint64_t moe_placement_epoch = 0;

        bool operator==(const ForwardGraphSignature &other) const
        {
            return seq_len == other.seq_len &&
                   batch_size == other.batch_size &&
                   device == other.device &&
                   decode == other.decode &&
                   decode_has_history == other.decode_has_history &&
                   all_position_logits == other.all_position_logits &&
                   standard_path == other.standard_path &&
                   pp_stage_enabled == other.pp_stage_enabled &&
                   pp_first_layer == other.pp_first_layer &&
                   pp_last_layer == other.pp_last_layer &&
                   pp_has_embedding == other.pp_has_embedding &&
                   pp_has_lm_head == other.pp_has_lm_head &&
                   is_bucketed_prefill == other.is_bucketed_prefill &&
                   bucket_seq_len == other.bucket_seq_len &&
                   moe_placement_epoch == other.moe_placement_epoch;
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
            h ^= (std::hash<bool>{}(sig.decode_has_history) + 0x9e3779b9 + (h << 6) + (h >> 2));
            h ^= (std::hash<bool>{}(sig.all_position_logits) + 0x9e3779b9 + (h << 6) + (h >> 2));
            h ^= (std::hash<bool>{}(sig.standard_path) + 0x9e3779b9 + (h << 6) + (h >> 2));
            h ^= (std::hash<bool>{}(sig.pp_stage_enabled) + 0x9e3779b9 + (h << 6) + (h >> 2));
            h ^= (std::hash<int>{}(sig.pp_first_layer) + 0x9e3779b9 + (h << 6) + (h >> 2));
            h ^= (std::hash<int>{}(sig.pp_last_layer) + 0x9e3779b9 + (h << 6) + (h >> 2));
            h ^= (std::hash<bool>{}(sig.pp_has_embedding) + 0x9e3779b9 + (h << 6) + (h >> 2));
            h ^= (std::hash<bool>{}(sig.pp_has_lm_head) + 0x9e3779b9 + (h << 6) + (h >> 2));
            h ^= (std::hash<bool>{}(sig.is_bucketed_prefill) + 0x9e3779b9 + (h << 6) + (h >> 2));
            h ^= (std::hash<int>{}(sig.bucket_seq_len) + 0x9e3779b9 + (h << 6) + (h >> 2));
            h ^= (std::hash<uint64_t>{}(sig.moe_placement_epoch) + 0x9e3779b9 + (h << 6) + (h >> 2));
            return h;
        }
    };

    /**
     * @brief Latest runtime metadata observed for a prefill graph-cache entry.
     *
     * The cache key remains bucket-shaped so one captured graph can safely serve
     * multiple real-token lengths in the same bucket. This observation records
     * the latest real chunk that flowed through that stable shape for diagnostics,
     * perf export, and Phase 12 graph-capture acceptance.
     */
    struct PrefillGraphExecutionObservation
    {
        bool valid = false;
        int chunk_index = 0;
        int bucket_seq_len = 0;
        int real_token_start = 0;
        int real_token_count = 0;
        int real_token_end = 0;
        std::string domain_id = "single";
        int participant_id = 0;
        uint64_t placement_epoch = 0;
        uint64_t topology_signature = 0;
        std::string capture_phase = "cold";
        std::string recapture_reason = "none";
    };

    /**
     * @brief RAII owner for an explicit GPU stream used by cached graph capture.
     *
     * The stream is created through the same worker GPU context that owns the
     * graph-capture backend. It is move-only so ForwardGraphCache entries remain
     * movable inside unordered_map storage without leaking backend stream handles.
     */
    struct CachedGraphStream
    {
        void *stream = nullptr;           ///< Backend stream handle (hipStream_t/cudaStream_t as void*)
        IWorkerGPUContext *ctx = nullptr; ///< Context that created the stream (not owned)

        CachedGraphStream() = default;
        ~CachedGraphStream() { reset(); }

        CachedGraphStream(const CachedGraphStream &) = delete;
        CachedGraphStream &operator=(const CachedGraphStream &) = delete;

        CachedGraphStream(CachedGraphStream &&other) noexcept
            : stream(other.stream), ctx(other.ctx)
        {
            other.stream = nullptr;
            other.ctx = nullptr;
        }

        CachedGraphStream &operator=(CachedGraphStream &&other) noexcept
        {
            if (this != &other)
            {
                reset();
                stream = other.stream;
                ctx = other.ctx;
                other.stream = nullptr;
                other.ctx = nullptr;
            }
            return *this;
        }

        /// @brief Ensure a stream exists for the provided GPU context.
        bool ensure(IWorkerGPUContext *new_ctx)
        {
            if (!new_ctx)
                return false;
            if (stream && ctx == new_ctx)
                return true;

            reset();
            stream = new_ctx->createStream();
            if (!stream)
            {
                ctx = nullptr;
                return false;
            }
            ctx = new_ctx;
            return true;
        }

        /// @brief Destroy the owned stream, if any.
        void reset()
        {
            if (stream && ctx)
            {
                ctx->synchronizeStream(stream);
                ctx->destroyStream(stream);
            }
            stream = nullptr;
            ctx = nullptr;
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
        ForwardOutput output;                ///< Cached output (logits pointer)
        bool valid = false;                  ///< Whether cache is usable

        /// Workspace generation recorded after the graph's stages were bound.
        uint64_t workspace_generation = 0;

        // Stable buffers — stages point to these, contents updated each step
        std::vector<int> token_ids;    ///< Persistent decode token storage
        std::vector<int> position_ids; ///< Persistent decode position IDs

        // PP hidden state copy — for non-embedding PP stages, the external
        // hidden state must be copied to the working buffer on every forward.
        // During graph build (cache MISS) this copy happens inline in
        // QwenStandardGraph::buildPartialForwardGraph(). On cache HIT we must redo
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

        // Pre-cached pointers to stages that override onGraphReplayed().
        // For prefill monolithic graph replay, these must be called after launch
        // to advance KV cache heads and other host-side bookkeeping.
        std::vector<IComputeStage *> replay_callback_stages;
        bool replay_callback_stages_cached = false;

        // Pre-cached pointers to stages that consume fixed-bucket prefill replay
        // metadata. These are updated before prefill capture/replay so callback
        // stages can advance host state by real tokens instead of padded rows.
        std::vector<IComputeStage *> prefill_replay_param_stages;
        bool prefill_replay_param_stages_cached = false;

        // Tracks whether setGPUStream has been applied to all stages.
        // Decode graph replay reapplies the capture stream before dynamic
        // params because manual/collective segments may temporarily bind a
        // different stream during execution.
        bool gpu_stream_applied = false;

        // The stream pointer that was last applied to all stages. Replay-state
        // resets preserve the capture stream, but workspace rebinds and full
        // invalidation can still force a new stream binding epoch. Track the
        // pointer so cached stages are rebound whenever that epoch changes.
        void *applied_stream = nullptr;

        // Tracks whether Phase 3 graph replay is active (no markCompleted calls),
        // allowing us to skip the 339-node graph.reset() since flags are already clear.
        bool phase3_active = false;

        /// Live replay-state epoch that the current segmented capture is safe for.
        /// Ordinary main-decode graphs are invalidated when speculative state
        /// publication advances the live state to a newer epoch. All-position
        /// verifier captures keep their own publication contract and do not use
        /// this stamp for preservation.
        uint64_t segmented_capture_live_state_epoch = 0;

        bool requiresLiveStateEpochRecapture(bool ordinary_decode_context,
                                             bool segmented_capture_allowed,
                                             uint64_t live_state_epoch) const
        {
            return ordinary_decode_context &&
                   segmented_capture_allowed &&
                   segment_cache.initialized &&
                   !segment_cache.needs_capture &&
                   segmented_capture_live_state_epoch != 0 &&
                   segmented_capture_live_state_epoch != live_state_epoch;
        }

        /// GPU graph capture/replay for eliminating per-kernel launch overhead
        std::unique_ptr<IGPUGraphCapture> gpu_graph;

        /// Segmented GPU graph cache. Capturable stages, including attention
        /// when its dynamic launch variant is stable, can live inside one
        /// segment; non-capturable stages and manual boundaries split it.
        DeviceGraphExecutor::GraphSegmentCache segment_cache;

        /// GPU stream (from IWorkerGPUContext::defaultStream()) for kernel dispatch
        /// Set when gpu_graph is created; used by stages to dispatch on correct stream
        void *gpu_stream = nullptr;

        /// GPU context for creating new graph captures (not owned)
        IWorkerGPUContext *gpu_ctx = nullptr;

        /// Prefill graph capture/replay cache (keyed by seq_len)
        std::unique_ptr<PrefillGraphCache> prefill_graph_cache;

        /// Latest diagnostic metadata for bucketed/chunked prefill graph execution.
        PrefillGraphExecutionObservation last_prefill_graph_observation;

        /// Monotonic engine-level LRU tick for bucketed prefill forward-cache eviction.
        uint64_t bucketed_prefill_last_access_tick = 0;

        /// Explicit stream for prefill warmup/capture/replay.
        CachedGraphStream prefill_capture_stream;

        /// Number of consecutive graph update failures (fallback heuristic)
        int gpu_graph_update_failures = 0;

        /// Maximum consecutive update failures before disabling graph capture
        static constexpr int kMaxGraphUpdateFailures = 4;

        /**
         * @brief Reset GPU graph replay/capture state while keeping the cached ComputeGraph.
         *
         * Session clears reset KV/GDN/conv recurrence but intentionally preserve graph
         * objects so host-resident weights do not need to be reloaded. Captured GPU
         * graphs are request-stateful: decode segment captures encode a specific
         * replay lifecycle, and monolithic prefill captures encode recurrent/KV
         * state mutation order from the previous request. Dropping executable
         * captures forces the next request to warm up/capture against the freshly
         * cleared model state while preserving the reusable ComputeGraph topology.
         *
         * The capture stream itself is retained because cached stages store that
         * stream pointer internally. Destroying it here would leave dynamic-param
         * updates (for example token-id preloads) with a dangling HIP stream before
         * the next warmup has a chance to rebind every stage.
         */
        void resetReplayState()
        {
            if (gpu_graph)
            {
                gpu_graph->reset();
                gpu_graph.reset();
            }
            segment_cache.reset(DeviceGraphExecutor::GraphSegmentCache::StreamResetPolicy::Preserve);
            gpu_graph_update_failures = 0;
            phase3_active = false;
            segmented_capture_live_state_epoch = 0;
        }

        /**
         * @brief Drop captured replay state after workspace buffers are rebound.
         *
         * The cached ComputeGraph remains valid, but HIP/CUDA graph captures
         * encode raw workspace addresses. Rebinding stages to a new workspace
         * manager therefore requires throwing away any captured decode segments
         * and monolithic prefill graph entries before the next launch.
         */
        void resetReplayStateAfterWorkspaceRebind()
        {
            resetReplayState();
            if (prefill_graph_cache)
                prefill_graph_cache->invalidateAll();
            gpu_stream_applied = false;
            applied_stream = nullptr;
        }

        /**
         * @brief Force cached stages to rebind their explicit GPU stream.
         *
         * KernelFactory::resetAllDynamicState() clears cached kernel stream
         * bindings. Some correction-replay paths deliberately preserve
         * verifier graph executables, but those stages still need to receive
         * the capture stream again before updateDynamicParams() or replay
         * callbacks can safely touch backend objects.
         */
        void markGPUStreamBindingsDirty()
        {
            gpu_stream_applied = false;
            applied_stream = nullptr;
        }

        /**
         * @brief Reset request-scoped state while preserving reusable graph objects.
         *
         * This is the request-boundary counterpart to invalidate(): it keeps the
         * cached ComputeGraph, stable token buffers, and workspace bindings, but
         * clears stage/kernels' dynamic metadata and all captured executable
         * state so the next prompt starts from cleared KV/GDN model state.
         */
        void resetSessionState()
        {
            resetReplayState();
            if (prefill_graph_cache)
                prefill_graph_cache->invalidateAll(PrefillGraphRejectReason::SessionReset);
            last_prefill_graph_observation = {};

            if (graph)
            {
                for (const auto &node_name : graph->getExecutionOrder())
                {
                    ComputeNode *node = graph->getNode(node_name);
                    if (node && node->stage)
                        node->stage->resetSessionState();
                }
            }

            gpu_stream_applied = false;
            applied_stream = nullptr;
            gpu_stream = nullptr;
            gpu_ctx = nullptr;
            segmented_capture_live_state_epoch = 0;
        }

        void invalidate()
        {
            resetReplayState();
            segment_cache.reset(DeviceGraphExecutor::GraphSegmentCache::StreamResetPolicy::Destroy);
            if (prefill_graph_cache)
                prefill_graph_cache->invalidateAll();
            last_prefill_graph_observation = {};
            prefill_capture_stream.reset();
            bucketed_prefill_last_access_tick = 0;
            graph.reset();
            valid = false;
            workspace_generation = 0;
            token_ids.clear();
            position_ids.clear();
            collective_nodes.clear();
            dynamic_param_stages.clear();
            dynamic_param_stages_cached = false;
            replay_callback_stages.clear();
            replay_callback_stages_cached = false;
            prefill_replay_param_stages.clear();
            prefill_replay_param_stages_cached = false;
            gpu_stream_applied = false;
            applied_stream = nullptr;
            gpu_stream = nullptr;
            gpu_ctx = nullptr;
            phase3_active = false;
            segmented_capture_live_state_epoch = 0;
            pp_external_hidden_state = nullptr;
            pp_working_buffer = nullptr;
            pp_copy_bytes = 0;
            pp_needs_copy = false;
        }
    };

} // namespace llaminar2
