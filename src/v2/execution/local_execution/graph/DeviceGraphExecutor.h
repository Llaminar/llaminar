/**
 * @file DeviceGraphExecutor.h
 * @brief Compute graph execution engine
 * @author David Sanftenberg
 * @date December 2025
 *
 * DeviceGraphExecutor coordinates the execution of ComputeStages within a compute graph,
 * managing:
 * - Device-aware stage scheduling
 * - Asynchronous execution with dependency tracking
 * - Multi-device work distribution via WorkDistributor
 *
 * Note: This was previously named LayerExecutor but renamed to better reflect
 * its purpose - it executes graphs, not layers specifically.
 *
 * Architecture:
 *   Pipeline / ModelExecutor
 *      |
 *      v
 *   DeviceGraphExecutor  <-- This component
 *      |
 *      v
 *   ComputeStage[] (device-specific implementations)
 *      |
 *      v
 *   DeviceContext (execution environment)
 */

#pragma once

#include "ComputeGraph.h"
#include "IGraphExecutor.h"
#include "../device/DeviceContext.h"
#include "StageTimeline.h"
#include "../../../utils/DebugEnv.h" // For LLAMINAR_ASSERTIONS_ACTIVE
#include "../../../interfaces/ICollectiveContext.h"
#include "../../../backends/IGPUGraphCapture.h"
#include "../../../backends/IWorkerGPUContext.h"
#include "../../../memory/BufferArena.h" // Phase 2: contract-based coherence
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <functional>

namespace llaminar2
{

    // Forward declarations
    class TensorBase;

    /**
     * @brief Policy controlling what happens during stage execution.
     *
     * All execution paths (full, fast-decode, graph-capture manual segments)
     * use a single `runStages()` loop parameterised by this policy. This
     * eliminates the class of bugs caused by divergent code paths where one
     * path does coherence / validation / profiling and another silently skips it.
     *
     * Factory methods provide canonical configurations:
     *   - full()       — first-time execution (prefill, cache miss)
     *   - fastDecode() — cached decode graphs where buffers are already on-device
     *   - debug()      — full + timeline (for diagnostics)
     */
    struct StageRunPolicy
    {
        bool coherence = true;            ///< Arena contract-based input/output coherence
        bool weight_coherence = true;     ///< Upload weights to device
        bool mark_dirty = true;           ///< Mark outputs device-authoritative after execute (ALWAYS ON — correctness, not overhead)
        bool validation = true;           ///< NaN/Inf output validation (Debug/Integration only)
        bool profiling = true;            ///< Per-stage timing breakdown
        bool collective_intercept = true; ///< Use CollectiveContext for allreduce/allgather
        bool timeline = false;            ///< GPU event-based per-stage profiling
        bool stage_dump = true;           ///< Stage dump framework (input/output snapshots)
        bool snapshot_callback = true;    ///< Invoke snapshot callback after execution
        bool pointer_validation = false;  ///< GPU pointer device validation
        bool preserve_gpu_streams = false; ///< Keep caller-assigned streams instead of rebinding normal passes to the worker stream

        /// Full execution — coherence, validation, profiling, everything.
        static StageRunPolicy full()
        {
            StageRunPolicy p{};
            p.timeline = true; // Enable GPU event timeline for prefill profiling
            return p;
        }

        /// Fast decode — minimal overhead for cached decode graphs.
        /// Buffers are already on-device, weights already uploaded.
        /// Profiling is gated by LLAMINAR_PROFILING env var so that decode
        /// stage timings are correctly reported when profiling is enabled.
        static StageRunPolicy fastDecode()
        {
            StageRunPolicy p;
            p.coherence = false;
            p.weight_coherence = false;
            p.mark_dirty = true;
            p.validation = false;
            p.profiling = debugEnv().profile.enabled;
            p.stage_dump = false;
            p.snapshot_callback = false;
            p.pointer_validation = false;
            p.timeline = true;
            return p;
        }

        /// Debug — everything on, including timeline.
        static StageRunPolicy debug()
        {
            StageRunPolicy p;
            p.timeline = true;
            p.pointer_validation = true;
            return p;
        }
    };

    /**
     * @brief Orchestrates execution of compute stages within a graph
     *
     * DeviceGraphExecutor manages the execution of compute graphs by handling
     * topological ordering, device contexts, and execution modes.
     *
     * Usage:
     * @code
     * DeviceGraphExecutor executor(config);
     *
     * // Build a graph
     * ComputeGraph graph;
     * graph.addNode("norm", createRMSNormStage(...));
     * graph.addNode("proj", createGEMMStage(...));
     * graph.addDependency("proj", "norm");
     *
     * // Execute
     * executor.execute(graph, ctx);
     * @endcode
     */
    class DeviceGraphExecutor : public IGraphExecutor
    {
    public:
        /**
         * @brief Construct with configuration
         * @param config Executor configuration
         */
        explicit DeviceGraphExecutor(const GraphExecutorConfig &config = GraphExecutorConfig());
        ~DeviceGraphExecutor() override;

        // Non-copyable, movable
        DeviceGraphExecutor(const DeviceGraphExecutor &) = delete;
        DeviceGraphExecutor &operator=(const DeviceGraphExecutor &) = delete;
        DeviceGraphExecutor(DeviceGraphExecutor &&) = default;
        DeviceGraphExecutor &operator=(DeviceGraphExecutor &&) = default;

        // =========================================================================
        // Configuration (IGraphExecutor interface)
        // =========================================================================

        /**
         * @brief Get current configuration
         */
        const GraphExecutorConfig &config() const override { return config_; }

        /**
         * @brief Set execution mode
         */
        void setExecutionMode(ExecutionMode mode) override { config_.mode = mode; }

        /**
         * @brief Get current execution mode
         */
        ExecutionMode executionMode() const override { return config_.mode; }

        /**
         * @brief Enable/disable profiling
         */
        void setProfilingEnabled(bool enabled) override { config_.enable_profiling = enabled; }

        /**
         * @brief Enable/disable output validation
         */
        void setValidationEnabled(bool enabled) override { config_.enable_validation = enabled; }

        /**
         * @brief Set snapshot callback for debugging
         */
        void setSnapshotCallback(StageSnapshotCallback callback) override { config_.snapshot_callback = std::move(callback); }

        void setStageFailureCallback(StageFailureCallback callback) { config_.stage_failure_callback = std::move(callback); }
        void setCancellationCallback(ExecutionCancellationCallback callback) { config_.cancellation_requested = std::move(callback); }

        /**
         * @brief Set the current layer context for stage dumping
         */
        void setCurrentLayerIdx(int layer_idx) override { config_.current_layer_idx = layer_idx; }

        /**
         * @brief Set the current iteration context for stage dumping
         */
        void setCurrentIteration(int iteration) override { config_.current_iteration = iteration; }

        /**
         * @brief Set the MPI rank for stage dumping
         */
        void setMPIRank(int rank) override { config_.mpi_rank = rank; }

        // =========================================================================
        // Execution (IGraphExecutor interface)
        // =========================================================================

        /**
         * @brief Execute a compute graph
         * @param graph The compute graph to execute
         * @param ctx Device context for execution
         * @return true on success
         */
        bool execute(ComputeGraph &graph, IDeviceContext *ctx) override;

        /**
         * @brief Execute a compute graph with multi-device support
         * @param graph The compute graph to execute
         * @param contexts Map of DeviceId -> DeviceContext
         * @return true on success
         */
        bool executeMultiDevice(
            ComputeGraph &graph,
            const std::unordered_map<DeviceId, IDeviceContext *> &contexts) override;

        // =========================================================================
        // Statistics (IGraphExecutor interface)
        // =========================================================================

        /**
         * @brief Get execution statistics
         */
        const GraphExecutorStats &stats() const override { return stats_; }

        /**
         * @brief Reset statistics
         */
        void resetStats() override { stats_.reset(); }

        // =========================================================================
        // Collective Context
        // =========================================================================

        /**
         * @brief Set the collective context for GPU-native collectives
         *
         * When a collective context is set, ALLREDUCE and ALLGATHER stages
         * will be intercepted and executed via the CollectiveContext
         * infrastructure (RCCL, NCCL, HOST) instead of the stage's
         * internal MPI fallback.
         *
         * @param ctx Collective context (not owned, must outlive executor)
         */
        void setCollectiveContext(ICollectiveContext *ctx) { collective_ctx_ = ctx; }

        /**
         * @brief Get the current collective context
         * @return Pointer to collective context (nullptr if not set)
         */
        ICollectiveContext *collectiveContext() const { return collective_ctx_; }

        // =========================================================================
        // Buffer Management
        // =========================================================================

        /**
         * @brief Set the BufferArena for contract-based coherence
         *
         * When both the arena is set and a stage returns a non-empty bufferContract(),
         * the executor uses the arena for coherence instead of getDumpInfo().
         *
         * @param arena BufferArena (not owned, must outlive executor)
         */
        void setArena(BufferArena *arena) { arena_ = arena; }

        /**
         * @brief Get the current BufferArena
         * @return Pointer to arena (nullptr if not set)
         */
        BufferArena *arena() const { return arena_; }

        /**
         * @brief Execute a cached decode graph with minimal overhead
         *
         * Stripped-down execution path for cached decode graphs where:
         * - All tensors are already on the target GPU device
         * - Output GPU buffers are already allocated
         * - No stage dumping, validation, or profiling is needed
         * - Stage objects are reused (not rebuilt)
         *
         * Skips: getDumpInfo, extractBuffers, cohereInputs/Outputs,
         *        shouldDump, printStageOutputs, profiling, assertions
         *
         * The collective_nodes set enables O(1) lookup for TP stages instead
         * of calling stage->type() (virtual dispatch) on every node.
         *
         * @param graph The cached compute graph (stages already configured)
         * @param ctx Device context for execution
         * @param collective_nodes Pre-computed set of node names that are collective stages (optional, for TP>1)
         * @return true on success
         */
        /**
         * @brief Get the stage timeline for external collection/printing
         *
         * The timeline is populated during executeFastDecode() when
         * LLAMINAR_GPU_STAGE_TIMING=1. The caller is responsible for calling
         * collect() and printSummary() after the forward pass completes.
         */
        StageTimeline &stageTimeline() { return stage_timeline_; }
        const StageTimeline &stageTimeline() const { return stage_timeline_; }

        bool executeFastDecode(ComputeGraph &graph, IDeviceContext *ctx,
                               const std::unordered_set<std::string> *collective_nodes = nullptr);

        /**
         * @brief Execute a cached decode graph with GPU graph capture/replay
         *
         * On first call: captures all kernel launches into a GPU graph, instantiates
         * and launches it. On subsequent calls: re-captures, updates the executable
         * in-place (hipGraphExecUpdate/cudaGraphExecUpdate), and launches.
         *
         * Falls back to executeFastDecode() if:
         * - Graph capture/instantiation fails
         * - Update fails more than max_failures times consecutively
         * - Collective nodes are present (TP>1 with cross-device communication)
         *
         * @param graph The cached compute graph
         * @param ctx Device context for execution
         * @param capture GPU graph capture object (created once, reused across calls)
         * @param collective_nodes Pre-computed collective node names (for TP>1)
         * @param gpu_stream Opaque GPU stream pointer to assign to stages for dispatch
         * @return true on success
         */
        bool executeWithGraphCapture(ComputeGraph &graph, IDeviceContext *ctx,
                                     IGPUGraphCapture *capture,
                                     const std::unordered_set<std::string> *collective_nodes = nullptr,
                                     void *gpu_stream = nullptr);

        // =========================================================================
        // Segmented GPU Graph Capture/Replay
        // =========================================================================

        /**
         * @brief A segment of the execution graph — either graph-capturable or manual
         *
         * The execution order is partitioned into contiguous segments based on
         * isGraphCapturable(). Capturable segments get their own GPU graph;
         * non-capturable segments (attention, KV cache) are executed manually
         * between graph launches.
         */
        struct GraphSegment
        {
            struct ArenaWriteBinding
            {
                BufferId id;
                DeviceId device;
            };

            std::vector<std::string> stage_names;          ///< Ordered stage names in this segment
            bool capturable = true;                        ///< Whether this segment can be graph-captured
            std::unique_ptr<IGPUGraphCapture> capture;     ///< GPU graph (only for capturable segments)
            std::vector<IComputeStage *> replay_callbacks; ///< Stages needing onGraphReplayed() (precomputed)
            uint64_t last_executed_step = 0;               ///< Last decode-step where this segment executed

            // Output coherence is marked through stable BufferArena ids after
            // replay. Avoid retaining raw TensorBase* caches here: prefix
            // restore, rollback, and request clears can legally mutate tensor
            // ownership while preserving graph topology.
            std::vector<ArenaWriteBinding> cached_arena_writes;
            bool arena_writes_cached = false;
        };

        /**
         * @brief Persistent cache of graph segments for segmented capture/replay
         *
         * Built once on the first decode step, reused across subsequent steps.
         * Capturable segments are replayed via GPU graph launch; non-capturable
         * segments are executed manually each step.
         */
        struct GraphSegmentCache
        {
            /**
             * @brief Controls whether reset() keeps the explicit capture stream alive.
             *
             * Preserve is used for recapture/retry state resets where cached stages
             * may still hold the stream pointer. Destroy is reserved for teardown or
             * cache invalidation where no stage will execute again before rebinding.
             */
            enum class StreamResetPolicy
            {
                Preserve,
                Destroy
            };

            std::vector<GraphSegment> segments;       ///< Ordered segments
            bool initialized = false;                 ///< Whether segments have been built
            bool needs_capture = false;               ///< True after warmup, before capture
            int consecutive_failures = 0;             ///< Segment-level failure counter
            uint64_t decode_step = 0;                 ///< Monotonic segmented-execution step counter
            uint64_t capture_variant_signature = 0;   ///< Stage-reported launch-topology variant for this cache
            uint64_t variant_recapture_count = 0;     ///< Resets caused by launch-topology variant changes
            std::string perf_context;                 ///< Optional structured stats tag for the replay caller
            void *capture_stream = nullptr;           ///< Locally-created blocking stream for capture/replay
            void *sync_event = nullptr;               ///< Cached event for GPU-side inter-stream sync
            IWorkerGPUContext *gpu_ctx_ref = nullptr; ///< GPU context for stream lifecycle (not owned)
            DeviceId capture_device = DeviceId::invalid(); ///< Device used to resolve the stream owner at teardown
            bool capture_context_from_pool = false; ///< True when the stream was created by the pool context
            static constexpr int kMaxFailures = 4;    ///< Disable after N failures

            GraphSegmentCache() = default;
            ~GraphSegmentCache()
            {
                reset(StreamResetPolicy::Destroy);
            }

            // Move-only (non-copyable due to stream/event ownership)
            GraphSegmentCache(GraphSegmentCache &&other) noexcept
                : segments(std::move(other.segments)),
                  initialized(other.initialized),
                  needs_capture(other.needs_capture),
                  consecutive_failures(other.consecutive_failures),
                  decode_step(other.decode_step),
                  capture_variant_signature(other.capture_variant_signature),
                  variant_recapture_count(other.variant_recapture_count),
                  perf_context(std::move(other.perf_context)),
                  capture_stream(other.capture_stream),
                  sync_event(other.sync_event),
                  gpu_ctx_ref(other.gpu_ctx_ref),
                  capture_device(other.capture_device),
                  capture_context_from_pool(other.capture_context_from_pool)
            {
                other.capture_stream = nullptr;
                other.sync_event = nullptr;
                other.gpu_ctx_ref = nullptr;
                other.capture_device = DeviceId::invalid();
                other.capture_context_from_pool = false;
                other.capture_variant_signature = 0;
                other.variant_recapture_count = 0;
            }
            GraphSegmentCache &operator=(GraphSegmentCache &&other) noexcept
            {
                if (this != &other)
                {
                    reset(StreamResetPolicy::Destroy);
                    segments = std::move(other.segments);
                    initialized = other.initialized;
                    needs_capture = other.needs_capture;
                    consecutive_failures = other.consecutive_failures;
                    decode_step = other.decode_step;
                    capture_variant_signature = other.capture_variant_signature;
                    variant_recapture_count = other.variant_recapture_count;
                    perf_context = std::move(other.perf_context);
                    capture_stream = other.capture_stream;
                    sync_event = other.sync_event;
                    gpu_ctx_ref = other.gpu_ctx_ref;
                    capture_device = other.capture_device;
                    capture_context_from_pool = other.capture_context_from_pool;
                    other.capture_stream = nullptr;
                    other.sync_event = nullptr;
                    other.gpu_ctx_ref = nullptr;
                    other.capture_device = DeviceId::invalid();
                    other.capture_context_from_pool = false;
                    other.capture_variant_signature = 0;
                    other.variant_recapture_count = 0;
                }
                return *this;
            }
            GraphSegmentCache(const GraphSegmentCache &) = delete;
            GraphSegmentCache &operator=(const GraphSegmentCache &) = delete;

            void reset(StreamResetPolicy stream_policy = StreamResetPolicy::Destroy)
            {
                synchronizeCaptureStream();
                segments.clear();
                initialized = false;
                needs_capture = false;
                consecutive_failures = 0;
                decode_step = 0;
                capture_variant_signature = 0;
                destroySyncEvent();
                if (stream_policy == StreamResetPolicy::Destroy)
                    destroyCaptureStream();
            }

            /// Create a local blocking stream for graph capture via the GPU context.
            /// @param ctx GPU context that creates the stream (stored for cleanup)
            bool ensureCaptureStream(IWorkerGPUContext *ctx,
                                     DeviceId device = DeviceId::invalid());

            /// Wait for queued replay/capture work before graph resources are torn down.
            void synchronizeCaptureStream();

            /// Destroy the capture stream if it exists
            void destroyCaptureStream();

            /// Create or get the cached sync event for inter-stream dependencies
            bool ensureSyncEvent(IWorkerGPUContext *ctx);

            /// Destroy the cached sync event if it exists
            void destroySyncEvent();

            /// Resolve the current lifecycle context for stream/event cleanup.
            IWorkerGPUContext *resolveLifecycleContext(const char *operation);
        };

        /**
         * @brief Execute with segmented GPU graph capture/replay
         *
         * Partitions the execution order into segments based on stage
         * isGraphCapturable(). Capturable segments (GEMMs, norms, SwiGLU, etc.)
         * are captured into separate GPU graphs and replayed. Non-capturable
         * segments (attention, KV cache append) are executed manually between
         * graph launches.
         *
         * For a 28-layer Qwen2.5 model this produces ~57 graph segments + 56
         * manual segments per decode step, with each hipGraphLaunch costing
         * ~5-10μs (total ~300-600μs host overhead vs ~43ms without graphs).
         *
         * On first call: builds segment list, captures all capturable segments.
         * On subsequent calls: replays captured graphs, re-executes manual stages.
         *
         * @param graph The cached compute graph
         * @param ctx Device context for execution
         * @param segment_cache Persistent segment cache (built once, reused)
         * @param gpu_stream Opaque GPU stream pointer for kernel dispatch
         * @param gpu_ctx GPU context for creating new graph captures
         * @return true on success
         */
        bool executeWithSegmentedGraphCapture(ComputeGraph &graph, IDeviceContext *ctx,
                                              GraphSegmentCache &segment_cache,
                                              void *gpu_stream,
                                              IWorkerGPUContext *gpu_ctx,
                                              const std::unordered_set<std::string> *collective_nodes = nullptr,
                                              bool collectives_graph_capturable = false,
                                              bool force_recapture = false,
                                              bool defer_final_sync = false);

        /**
         * @brief Policy object for decode capture/replay execution mode selection
         */
        struct DecodeCapturePolicy
        {
            bool allow_fast_decode = true;
            bool allow_segmented_capture = false;
            bool collective_segmented_enabled = false;
            bool collectives_graph_capturable = false; ///< True only for explicit future graph-captured collective paths
            bool force_recapture = false;              ///< Re-record graph segments on replay for callers with dynamic params not yet replay-safe
            bool defer_final_sync = false;             ///< Caller will synchronize the replay stream through a following operation.
            int max_segment_failures = 4;
        };

        /**
         * @brief Execute decode graph according to a single policy object
         *
         * Centralizes mode selection between segmented replay, fast decode, and
         * full executor fallback. When segmented replay fails, this method falls
         * back to fast decode automatically.
         */
        bool executeDecodeWithCapturePolicy(
            ComputeGraph &graph,
            IDeviceContext *ctx,
            GraphSegmentCache *segment_cache,
            void *gpu_stream,
            IWorkerGPUContext *gpu_ctx,
            const std::unordered_set<std::string> *collective_nodes,
            const DecodeCapturePolicy &policy,
            bool *used_segmented_capture = nullptr);

    private:
        GraphExecutorConfig config_;
        GraphExecutorStats stats_;
        ICollectiveContext *collective_ctx_ = nullptr; ///< Optional collective context (not owned)
        BufferArena *arena_ = nullptr;                 ///< Optional arena for contract coherence (not owned)
        StageTimeline stage_timeline_;                 ///< GPU event-based per-stage timeline profiler
        bool stage_timeline_info_populated_ = false;   ///< True after first setStageInfo pass (names never change)
        bool weights_session_cohered_ = false;         ///< True after first forward completes weight coherence for all nodes

        // =====================================================================
        // Unified stage runner (replaces divergent paths)
        // =====================================================================

        /**
         * @brief Execute all stages in a graph according to the given policy.
         *
         * This is the ONE execution loop used by every entry point:
         *   - execute()/executeSequential()  → full policy
         *   - executeFastDecode()             → fastDecode policy
         *   - graph capture non-captured segs → full policy
         *
         * @param graph            Compute graph with stages to run
         * @param ctx              Device context for execution
         * @param policy           Controls coherence, validation, profiling etc.
         * @param collective_nodes Pre-identified collective stage names (optional)
         * @return true if all stages executed successfully
         */
        bool runStages(ComputeGraph &graph,
                       IDeviceContext *ctx,
                       const StageRunPolicy &policy,
                       const std::unordered_set<std::string> *collective_nodes = nullptr);

        bool cancellationRequested(const std::string &node_name) const;
        void notifyStageFailure(const std::string &node_name, const std::string &reason) const;

        /**
         * @brief Execute a single stage according to the given policy.
         *
         * Implements every per-stage phase (coherence, execution, marking,
         * validation, profiling, dumps) gated by the policy flags.
         *
         * @param node          The compute node to execute
         * @param ctx           Device context
         * @param policy        Controls which phases are active
         * @param is_collective Pre-computed flag: true if stage is collective
         * @return true if stage executed successfully
         */
        bool runStage(ComputeNode &node,
                      IDeviceContext *ctx,
                      const StageRunPolicy &policy,
                      bool is_collective);

        // =====================================================================
        // Legacy internal helpers (now delegate to runStages/runStage)
        // =====================================================================
        bool executeSequential(ComputeGraph &graph, IDeviceContext *ctx);
        // executeParallel removed — PARALLEL mode falls through to executeSequential
        bool executeNode(ComputeNode &node, IDeviceContext *ctx);

        // Collective stage intercept helpers (GPU-native collectives)
        bool executeCollectiveAllreduce(ComputeNode &node, IDeviceContext *ctx);
        bool executeCollectiveAllgather(ComputeNode &node, IDeviceContext *ctx);

        /**
         * @brief Execute strided AllGather using NCCL + CUDA deinterleave
         *
         * Optimized for column-parallel LM head. Returns false if NCCL not
         * available or device is not CUDA, allowing fallback to MPI path.
         *
         * @param node The compute node containing AllGatherStage
         * @param ctx Device context (unused, CollectiveContext handles device)
         * @return true if executed successfully, false to fall back to stage execution
         */
        bool executeCollectiveStridedAllgather(ComputeNode &node, IDeviceContext *ctx);

        // Buffer validation (Debug/Integration builds only)
#if LLAMINAR_ASSERTIONS_ACTIVE
        // Stage verification delegated to free functions in StageVerifier.h
#endif

        // Workspace management
        std::vector<float> temp_buffer_;
        size_t temp_buffer_size_ = 0;

        float *getTemporaryBuffer(size_t elements);
    };

    // Backwards compatibility alias
    using LayerExecutor = DeviceGraphExecutor;

} // namespace llaminar2
