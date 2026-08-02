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
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <optional>
#include <span>

namespace llaminar2
{

    // Forward declarations
    class TensorBase;
    class FP32Tensor;

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
        /// Legacy per-stage profiling remains independently opt-in; production
        /// graph replay timing is collected by PerfStats events around replay.
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

        /// Capture phase — normal coherence and dirty marking, no host diagnostics.
        ///
        /// Phase-2 graph capture executes stages while HIP/CUDA stream capture may
        /// be active on sibling TP participants.  Diagnostics that materialize
        /// outputs on the host (validation, dumps, snapshot callbacks) are not
        /// part of inference semantics and can make HIP report illegal legacy
        /// stream dependencies during concurrent multi-device capture.
        static StageRunPolicy capturePhase()
        {
            StageRunPolicy p;
            p.coherence = true;
            p.weight_coherence = true;
            p.mark_dirty = true;
            p.validation = false;
            p.profiling = debugEnv().profile.enabled;
            p.collective_intercept = true;
            p.timeline = false;
            p.stage_dump = false;
            p.snapshot_callback = false;
            p.pointer_validation = false;
            p.preserve_gpu_streams = true;
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
         * @brief One immutable, graph-owned GPU snapshot output descriptor.
         *
         * The graph owner's warmup finalizes both this descriptor and its stable
         * destination storage. The same owner then consumes it during capture
         * and replay publication. Identically named stages in another graph
         * geometry therefore cannot overwrite it.
         */
        struct GraphSnapshotOutputCopy
        {
            std::string name;
            std::string dtype;
            size_t rows = 0;
            size_t cols = 0;
            size_t element_size = sizeof(float);
            size_t byte_size = 0;
            size_t storage_bytes = 0;
            DeviceId device = DeviceId::invalid();
            const void *source_ptr = nullptr;
            /**
             * True after a real stage execution recorded this descriptor and
             * its point-in-time D2D copy. Allocation-only pre-capture passes
             * may inspect stage metadata, but must not replace a warmed source
             * with a pre-execution fallback view.
             */
            bool descriptor_finalized = false;
            std::unique_ptr<FP32Tensor> storage;
        };

        /** @brief Snapshot slots owned by one stage in one forward graph. */
        struct GraphSnapshotStageCopies
        {
            std::vector<GraphSnapshotOutputCopy> outputs;
        };

        /**
         * @brief Complete GPU snapshot manifest owned by one forward graph.
         *
         * Stage names are unique only inside a ComputeGraph; they are not
         * process-wide identities. ForwardGraphCache stores this object beside
         * the executable whose captured D2D nodes write these slots.
         */
        struct GraphSnapshotManifest
        {
            std::unordered_map<std::string, GraphSnapshotStageCopies> stage_copies;
            std::unordered_set<std::string> outputless_stages;

            void clear()
            {
                stage_copies.clear();
                outputless_stages.clear();
            }
        };

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
        void setSnapshotCallback(StageSnapshotCallback callback) override
        {
            config_.snapshot_callback = std::move(callback);
            advanceSnapshotConfigurationEpoch();
        }
        void setSnapshotStageFilter(StageSnapshotFilter filter)
        {
            config_.snapshot_stage_filter = std::move(filter);
            advanceSnapshotConfigurationEpoch();
        }
        /**
         * @brief Current graph identity epoch for diagnostic snapshot nodes.
         *
         * Snapshot callback/filter changes alter captured graph topology because
         * selected GPU stages gain or lose D2D copy nodes. Cached executables
         * compare this epoch before launch and re-enter their normal warmup and
         * capture lifecycle when it changes.
         */
        uint64_t snapshotConfigurationEpoch() const noexcept
        {
            return snapshot_configuration_epoch_;
        }

        void setStageFailureCallback(StageFailureCallback callback) { config_.stage_failure_callback = std::move(callback); }
        void setCancellationCallback(ExecutionCancellationCallback callback) { config_.cancellation_requested = std::move(callback); }
        /** @brief Override worker GPU context resolution for this executor. */
        void setWorkerGPUContextResolver(
            std::function<IWorkerGPUContext *(DeviceId)> resolver,
            bool uses_process_pool = false)
        {
            config_.worker_gpu_context_resolver = std::move(resolver);
            config_.worker_gpu_context_uses_process_pool = uses_process_pool;
        }

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
         * Skips: getDumpInfo and ordinary arena preparation,
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
                               const std::unordered_set<std::string> *collective_nodes = nullptr,
                               GraphSnapshotManifest *snapshot_manifest = nullptr);

        /**
         * @brief Execute an eager graph against its explicit snapshot owner.
         *
         * Cached graph construction uses this boundary so warmup descriptors
         * remain attached to the exact graph that will later capture them.
         */
        bool executeWithSnapshotManifest(
            ComputeGraph &graph,
            IDeviceContext *ctx,
            GraphSnapshotManifest &snapshot_manifest);

        /**
         * @brief Publish snapshot callbacks from an already-executed graph.
         *
         * GPU graph capture/replay intentionally disables per-stage callbacks
         * while kernels are being recorded or launched. Parity diagnostics still
         * need the resulting stage tensors. GPU stages are published exclusively
         * from the immutable graph-snapshot manifest whose D2D copy nodes ran at
         * each stage boundary; the executor never re-queries mutable stage dump
         * metadata after launch. CPU stages retain their ordinary synchronous
         * dump callback because they do not participate in GPU graph capture.
         *
         * @param graph Graph whose stages have just executed.
         * @param producer_stream_override Explicit producer stream for the graph
         *                                 execution. When non-null it is treated
         *                                 as authoritative, because captured graph
         *                                 launches may run on a stream different
         *                                 from stale per-stage stream pointers.
         *                                 Null is allowed only for stages whose
         *                                 outputs are already host-readable or
         *                                 whose stage stream is current.
         * @param context Human-readable context for diagnostics.
         * @return true if all requested snapshots were published.
         */
        bool publishSnapshotsAfterGraphExecution(
            ComputeGraph &graph,
            void *producer_stream_override = nullptr,
            const char *context = nullptr,
            GraphSnapshotManifest *snapshot_manifest = nullptr);

        /**
         * @brief Publish the terminal captured mutable state row from a graph.
         *
         * GDN/short-conv stages can execute against speculative state slots when
         * snapshot/verifier capture is enabled. The graph may then have correct
         * tensor outputs while the backend-owned live recurrent state still
         * points at the pre-graph value. This helper restores the final real row
         * into stage-owned live state on the graph stream before callers observe,
         * snapshot, or prefix-cache that live state.
         *
         * This is intentionally graph-generic: prefix cache, parity snapshots,
         * and future runtime-state transactions should all depend on the same
         * publication primitive instead of each feature inventing its own GDN
         * state side path.
         *
         * @param graph Graph whose mutable-state stages have just completed.
         * @param terminal_row Flat terminal row for a scalar request. Ignored
         *        for a logical request batch, where terminal rows are derived
         *        from the metadata owner appropriate to each stage backend.
         * @param producer_stream_override Stream that produced graph outputs,
         *        or null to use each stage's bound stream.
         * @param context Optional diagnostic label.
         * @param device_request_seq_lens Stable device metadata allocation.
         *        This pointer may remain non-null when request capacity exceeds
         *        the active scalar request count; `request_count`, not pointer
         *        presence, selects grouped publication.
         * @param request_count Active logical request count for this execution.
         * @param request_row_width Padded rows per request for grouped state.
         * @param host_request_seq_lens Host-owned real lengths for CPU request
         *        batches. GPU stages never consume this vector; they derive
         *        terminal rows from @p device_request_seq_lens without a host
         *        synchronization. At least one length owner is required for a
         *        logical request batch, and each stage requires the owner that
         *        matches its backend.
         * @return true when every captured state owner is committed.
         */
        bool publishCapturedTerminalStateAfterGraphExecution(
            ComputeGraph &graph,
            int terminal_row,
            void *producer_stream_override = nullptr,
            const char *context = nullptr,
            const int *device_request_seq_lens = nullptr,
            int request_count = 1,
            int request_row_width = 0,
            const std::vector<int> *host_request_seq_lens = nullptr);

        /**
         * @brief Pre-register graph-stable snapshot buffers before GPU graph capture.
         *
         * Captured parity diagnostics record device-to-device snapshot copies as
         * graph nodes immediately after producing stages execute. The destination
         * buffers and per-stage output descriptors must therefore exist before
         * beginCapture(); discovering either while capture is active is a hard
         * graph-capture contract violation.
         *
         * This method performs no payload copies and does not execute graph
         * stages. Callers must pass the explicit graph stream that will be used
         * for capture.
         */
        bool prepareSnapshotsForGraphCapture(
            ComputeGraph &graph,
            IDeviceContext *ctx,
            void *producer_stream,
            const char *context = nullptr,
            GraphSnapshotManifest *snapshot_manifest = nullptr);

        /**
         * @brief Join every arena input to the exact stream before graph capture.
         *
         * Arena residency and producer ordering are separate contracts. Warmup
         * can leave valid activation storage whose completion event belongs to a
         * different eager stream. CUDA/HIP graph capture cannot import that
         * external event after `beginCapture()`, so this method walks all graph
         * reads, deduplicates their backing tensors, and enqueues nonblocking
         * event waits on @p capture_stream before the capture transaction starts.
         *
         * `TransferEngine::requireDeviceInput()` records each exact
         * `{completion event, capture stream}` pair. Stage-level prepared-input
         * guards can then prove the dependency was established and must hard-fail
         * instead of attempting a late capture-time event wait.
         *
         * @param graph Graph whose complete native body will be captured.
         * @param ctx GPU context owning the graph and stream.
         * @param capture_stream Exact non-null stream passed to beginCapture().
         * @param context Optional diagnostic label.
         * @return true after every unique arena read dependency is joined.
         */
        bool prepareInputsForGraphCapture(
            ComputeGraph &graph,
            IDeviceContext *ctx,
            void *capture_stream,
            const char *context = nullptr);

        /**
         * @brief Execute a cached decode graph with GPU graph capture/replay
         *
         * On first call: captures all kernel launches into a GPU graph, instantiates
         * and launches it. On subsequent calls: re-captures, updates the executable
         * in-place (hipGraphExecUpdate/cudaGraphExecUpdate), and launches.
         *
         * Capture, instantiation, update, and launch failures are hard failures.
         * This API never changes execution modes after work begins. Callers
         * selecting eager execution must do so explicitly before invoking it.
         *
         * @param graph The cached compute graph
         * @param ctx Device context for execution
         * @param capture GPU graph capture object (created once, reused across calls)
         * @param externally_visible_outputs Tensor outputs whose device writes
         *        become observable outside this graph call. The executor records
         *        their completion events on capture.executionStream() after the
         *        graph launch. An empty list is an invalid publication contract.
         * @param collective_nodes Pre-computed collective node names (for TP>1)
         * @return true on success
         */
        bool executeWithGraphCapture(ComputeGraph &graph, IDeviceContext *ctx,
                                     IGPUGraphCapture *capture,
                                     std::span<ITensor *const> externally_visible_outputs,
                                     const std::unordered_set<std::string> *collective_nodes = nullptr);

        // =========================================================================
        // Cached GPU Graph Replay
        // =========================================================================

        /**
         * @brief One contiguous replay unit in a GPU execution graph.
         *
         * A production homogeneous CUDA/ROCm graph should normally contain one
         * capturable unit spanning compute, state mutation, and NCCL/RCCL
         * collectives. Multiple units exist for explicit diagnostics and for a
         * topology containing a genuinely uncapturable stage. They are not the
         * target architecture and must never be created merely because a
         * specialized collective was omitted from orchestration classification.
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
            uint64_t last_executed_step = 0;               ///< Last decode-step where this segment executed

            // Output coherence is marked through stable BufferArena ids after
            // replay. Avoid retaining raw TensorBase* caches here: prefix
            // restore, rollback, and request clears can legally mutate tensor
            // ownership while preserving graph topology.
            std::vector<ArenaWriteBinding> cached_arena_writes;
            bool arena_writes_cached = false;
        };

        using GraphCaptureBoundaryHook =
            std::function<bool(const std::string &, void *capture_stream)>;

        /**
         * @brief Persistent cache of GPU graph replay units.
         *
         * Built on the first decode step and reused across subsequent steps.
         * The ordinary production plan has one fully captured replay unit.
         * Manual units are retained only for explicitly supported non-capturable
         * topologies and are structurally visible through replay-mode telemetry.
         */
        struct GraphSegmentCache
        {
            /**
             * @brief Immutable view of one complete replay-ready graph template.
             *
             * The cache retains ownership. A parent graph may clone the native
             * executable through @ref capture, but must not launch, reset, or
             * mutate it through this borrowed view.
             */
            struct DeviceLoopGraphTemplateView
            {
                const IGPUGraphCapture *capture = nullptr;
                void *stream = nullptr;
                size_t stage_count = 0;
                size_t captured_node_count = 0;
            };

            /**
             * @brief Stable execution geometry represented by this graph cache.
             *
             * Deferred GPU timing events can complete several host iterations
             * after their graph launch.  Reading mutable runner state when an
             * event is reclaimed would consequently attach the wrong M, batch,
             * or verifier mode to an otherwise accurate GPU interval.  The
             * descriptor instead travels with the cached executable whose event
             * ring produced the sample.
             *
             * The fields are deliberately scalar and backend-neutral.  They add
             * no allocation, transfer, synchronization, or graph node to replay;
             * PerfStats converts them to strings only when structured metrics
             * are already enabled.  `m` is the flattened physical row count and
             * is therefore the exact grouped-verifier work size for the common
             * batch-size-one MTP transaction.
             */
            struct ReplayWorkloadGeometry
            {
                int seq_len = 0;                 ///< Rows per logical request.
                int batch_size = 0;              ///< Logical request count.
                int m = 0;                       ///< Flattened physical graph rows.
                int all_position_rows = 0;       ///< Compact verifier/logit rows, or zero.
                uint8_t verifier_outcome_mode = 0; ///< Typed MTP outcome enum value.
                uint8_t position_policy = 0;     ///< Typed ForwardPositionPolicy value.
                uint64_t moe_placement_epoch = 0; ///< Expert-placement identity in the cache key.
                bool decode = false;             ///< Decode-equivalent graph family.
                bool all_position_logits = false; ///< Graph projects verifier rows.
                bool live_mtp_request_batch_condition = false; ///< One live condition row per request.

                [[nodiscard]] bool valid() const noexcept
                {
                    return seq_len > 0 && batch_size > 0 && m > 0;
                }
            };

            /**
             * @brief One preallocated asynchronous GPU replay timing interval.
             *
             * Timing events are backend resources, so they are created during
             * graph warmup and retained for the cache lifetime. Replay only
             * records the two existing events and changes this small host-side
             * state record; it never allocates, destroys, or synchronizes an
             * event. A completed interval is reclaimed with a nonblocking event
             * query before the slot is reused.
             */
            struct ReplayGpuTimingSlot
            {
                enum class Scope
                {
                    TotalReplay,
                    ReplayUnit
                };

                void *start_event = nullptr;
                void *stop_event = nullptr;
                Scope scope = Scope::TotalReplay;
                size_t replay_unit_index = 0;
                bool deferred_final_fence = false;
                bool started = false;
                bool pending = false;
            };

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
            /**
             * Snapshot descriptors and destinations captured by these exact
             * replay units. Cached decode callers cannot omit this owner because
             * it is an intrinsic part of the segment cache lifecycle.
             */
            GraphSnapshotManifest snapshot_manifest;
            bool initialized = false;                 ///< Whether segments have been built
            bool needs_capture = false;               ///< True after warmup, before capture
            uint64_t decode_step = 0;                 ///< Monotonic segmented-execution step counter
            uint64_t capture_variant_signature = 0;   ///< Stage-reported launch-topology variant for this cache
            uint64_t variant_recapture_count = 0;     ///< Resets caused by launch-topology variant changes
            uint64_t snapshot_configuration_epoch = 0; ///< Executor snapshot topology represented by this cache
            std::string perf_context;                 ///< Optional structured stats tag for the replay caller
            ReplayWorkloadGeometry replay_workload;   ///< Exact cache-key geometry for deferred replay metrics
            void *capture_stream = nullptr;           ///< Locally-created blocking stream for capture/replay
            void *sync_event = nullptr;               ///< Cached event for GPU-side inter-stream sync
            IWorkerGPUContext *gpu_ctx_ref = nullptr; ///< GPU context for stream lifecycle (not owned)
            DeviceId capture_device = DeviceId::invalid(); ///< Device used to resolve the stream owner at teardown
            bool capture_context_from_pool = false; ///< True when the stream was created by the pool context
            std::vector<ReplayGpuTimingSlot> replay_gpu_timing_slots; ///< Fixed event ring allocated before capture
            std::string replay_gpu_timing_device_name; ///< Stable PerfStats device label for completed slots
            uint64_t replay_gpu_timing_busy_samples = 0; ///< Replays intentionally not sampled while every slot is in flight

            GraphSegmentCache() = default;
            ~GraphSegmentCache()
            {
                reset(StreamResetPolicy::Destroy);
            }

            // Move-only (non-copyable due to stream/event ownership)
            GraphSegmentCache(GraphSegmentCache &&other) noexcept
                : segments(std::move(other.segments)),
                  snapshot_manifest(std::move(other.snapshot_manifest)),
                  initialized(other.initialized),
                  needs_capture(other.needs_capture),
                  decode_step(other.decode_step),
                  capture_variant_signature(other.capture_variant_signature),
                  variant_recapture_count(other.variant_recapture_count),
                  snapshot_configuration_epoch(other.snapshot_configuration_epoch),
                  perf_context(std::move(other.perf_context)),
                  replay_workload(other.replay_workload),
                  capture_stream(other.capture_stream),
                  sync_event(other.sync_event),
                  gpu_ctx_ref(other.gpu_ctx_ref),
                  capture_device(other.capture_device),
                  capture_context_from_pool(other.capture_context_from_pool),
                  replay_gpu_timing_slots(std::move(other.replay_gpu_timing_slots)),
                  replay_gpu_timing_device_name(std::move(other.replay_gpu_timing_device_name)),
                  replay_gpu_timing_busy_samples(other.replay_gpu_timing_busy_samples)
            {
                other.capture_stream = nullptr;
                other.sync_event = nullptr;
                other.gpu_ctx_ref = nullptr;
                other.capture_device = DeviceId::invalid();
                other.capture_context_from_pool = false;
                other.capture_variant_signature = 0;
                other.variant_recapture_count = 0;
                other.snapshot_configuration_epoch = 0;
                other.replay_gpu_timing_slots.clear();
                other.replay_gpu_timing_device_name.clear();
                other.replay_gpu_timing_busy_samples = 0;
            }
            GraphSegmentCache &operator=(GraphSegmentCache &&other) noexcept
            {
                if (this != &other)
                {
                    reset(StreamResetPolicy::Destroy);
                    segments = std::move(other.segments);
                    snapshot_manifest = std::move(other.snapshot_manifest);
                    initialized = other.initialized;
                    needs_capture = other.needs_capture;
                    decode_step = other.decode_step;
                    capture_variant_signature = other.capture_variant_signature;
                    variant_recapture_count = other.variant_recapture_count;
                    snapshot_configuration_epoch = other.snapshot_configuration_epoch;
                    perf_context = std::move(other.perf_context);
                    replay_workload = other.replay_workload;
                    capture_stream = other.capture_stream;
                    sync_event = other.sync_event;
                    gpu_ctx_ref = other.gpu_ctx_ref;
                    capture_device = other.capture_device;
                    capture_context_from_pool = other.capture_context_from_pool;
                    replay_gpu_timing_slots = std::move(other.replay_gpu_timing_slots);
                    replay_gpu_timing_device_name = std::move(other.replay_gpu_timing_device_name);
                    replay_gpu_timing_busy_samples = other.replay_gpu_timing_busy_samples;
                    other.capture_stream = nullptr;
                    other.sync_event = nullptr;
                    other.gpu_ctx_ref = nullptr;
                    other.capture_device = DeviceId::invalid();
                    other.capture_context_from_pool = false;
                    other.capture_variant_signature = 0;
                    other.variant_recapture_count = 0;
                    other.snapshot_configuration_epoch = 0;
                    other.replay_gpu_timing_slots.clear();
                    other.replay_gpu_timing_device_name.clear();
                    other.replay_gpu_timing_busy_samples = 0;
                }
                return *this;
            }
            GraphSegmentCache(const GraphSegmentCache &) = delete;
            GraphSegmentCache &operator=(const GraphSegmentCache &) = delete;

            /**
             * @brief Export a strict monolithic template for device-loop composition.
             *
             * Segmentation, incomplete capture, ambiguous stream ownership, or
             * incomplete stage coverage are hard failures. This method never
             * launches eagerly and never recaptures as a substitute.
             *
             * @param graph Complete source graph whose stage lifecycle
             *        contracts must be self-contained at replay time.
             * @param error Optional first violated contract.
             * @return Borrowed replay template when every invariant holds.
             */
            [[nodiscard]] std::optional<DeviceLoopGraphTemplateView>
            deviceLoopGraphTemplate(
                const ComputeGraph &graph,
                std::string *error = nullptr) const;

            void reset(StreamResetPolicy stream_policy = StreamResetPolicy::Destroy)
            {
                waitForCaptureStreamFence();
                segments.clear();
                initialized = false;
                needs_capture = false;
                decode_step = 0;
                capture_variant_signature = 0;
                destroyReplayGpuTimingEvents();
                destroySyncEvent();
                if (stream_policy == StreamResetPolicy::Destroy)
                {
                    snapshot_manifest.clear();
                    snapshot_configuration_epoch = 0;
                    destroyCaptureStream();
                }
            }

            /// Create a local blocking stream for graph capture via the GPU context.
            /// @param ctx GPU context that creates the stream (stored for cleanup)
            bool ensureCaptureStream(
                IWorkerGPUContext *ctx,
                DeviceId device = DeviceId::invalid(),
                bool context_from_process_pool = false);

            /**
             * @brief Wait on one event representing all prior capture-stream work.
             *
             * This is a host ownership fence, not an ordering primitive between
             * GPU streams. Normal producer/consumer ordering must use
             * orderCaptureStreamAfter() or another device-side event wait. The
             * fence is reserved for native graph-capture entry and resource
             * teardown, where the host must know that prior stream work has
             * completed before changing stream or graph lifetime.
             */
            void waitForCaptureStreamFence();

            /// Destroy the capture stream if it exists
            void destroyCaptureStream();

            /// Create or get the cached sync event for inter-stream dependencies
            bool ensureSyncEvent(IWorkerGPUContext *ctx);

            /**
             * @brief Order the capture stream after work queued on another stream.
             *
             * The method records the cache-owned event on `producer_stream` and
             * queues a wait on `capture_stream`. Both operations stay on the
             * device; no host wait or device-wide synchronization is permitted.
             *
             * @param ctx GPU context owning both streams and the event.
             * @param producer_stream Explicit stream whose prior work must finish.
             * @return true when the complete GPU-side dependency was queued.
             */
            bool orderCaptureStreamAfter(
                IWorkerGPUContext *ctx,
                void *producer_stream);

            /**
             * @brief Order an external consumer after the capture stream.
             *
             * This is the reverse half of @ref orderCaptureStreamAfter. It
             * records the cache-owned handoff event on the exact stream that
             * launched the captured graph, then queues a wait on the caller's
             * explicit consumer stream. The operation is entirely device-side;
             * it never synchronizes the host or changes graph ownership.
             *
             * @param ctx GPU context that owns both streams and the handoff event.
             * @param consumer_stream Explicit stream that will consume graph output.
             * @return true after the stream wait has been enqueued.
             */
            bool orderStreamAfterCapture(
                IWorkerGPUContext *ctx,
                void *consumer_stream);

            /// Destroy the cached sync event if it exists
            void destroySyncEvent();

            /**
             * @brief Destroy every cache-owned replay timing event after its stream fence.
             *
             * reset() calls this only after waitForCaptureStreamFence(), so no
             * recorded event can still be in flight. Steady-state replay must
             * never call this method; event ownership remains fixed from warmup
             * until reset or teardown.
             */
            void destroyReplayGpuTimingEvents();

            /**
             * @brief Resolve the mandatory owner of a live stream or event.
             *
             * Calling this method means a backend resource already exists.
             * Failure is therefore process-fatal: silently dropping the
             * resource would make queued-work and graph lifetime unknowable.
             *
             * @param operation Human-readable lifecycle operation for diagnostics.
             * @return Non-null context that owns the live backend resource.
             */
            IWorkerGPUContext *requireLifecycleContext(const char *operation);
        };

        /**
         * @brief Topology contract governing whether a replay plan may contain
         *        more than one graph unit or any host/manual execution unit.
         *
         * Homogeneous CUDA-only and ROCm-only execution domains use
         * @ref RequireFullGraph. Every stage, including NCCL/RCCL collectives,
         * must then belong to one captured graph.
         *
         * @ref AllowHeterogeneousCollectiveSegmentation exists only for a
         * genuinely mixed device-type domain whose graph contains collectives
         * crossing those device types. Callers must prove both properties
         * before selecting it. The planner independently requires collective
         * nodes, preventing this policy from becoming a generic escape hatch
         * for an uncapturable stage.
         */
        enum class GraphReplayPlanPolicy
        {
            RequireFullGraph,
            AllowHeterogeneousCollectiveSegmentation
        };

        /**
         * @brief Execute a cached GPU graph replay plan.
         *
         * The planner consumes each stage's graph-capture and collective
         * contracts. Fully capturable homogeneous CUDA/ROCm graphs, including
         * NCCL/RCCL LocalTP collectives, become one captured replay unit.
         * Segmentation is reserved for explicitly admitted heterogeneous
         * collective boundaries; it is not an automatic stage fallback.
         *
         * On first call the method builds and captures the replay plan. On
         * subsequent calls it launches captured units and executes any
         * explicitly admitted manual units.
         *
         * @param graph The cached compute graph
         * @param ctx Device context for execution
         * @param segment_cache Persistent segment cache (built once, reused)
         * @param gpu_stream Opaque GPU stream pointer for kernel dispatch
         * @param gpu_ctx GPU context for creating new graph captures
         * @return true on success
         */
        bool executeWithCachedGraphReplay(ComputeGraph &graph, IDeviceContext *ctx,
                                              GraphSegmentCache &segment_cache,
                                              void *gpu_stream,
                                              IWorkerGPUContext *gpu_ctx,
                                              const std::unordered_set<std::string> *collective_nodes = nullptr,
                                              bool collectives_graph_capturable = false,
                                              bool force_recapture = false,
                                              bool defer_final_sync = false,
                                              GraphCaptureBoundaryHook capture_boundary = {},
                                              GraphReplayPlanPolicy plan_policy =
                                                  GraphReplayPlanPolicy::RequireFullGraph);

        /**
         * @brief Policy object for decode capture/replay execution mode selection
         */
        struct DecodeCapturePolicy
        {
            bool allow_fast_decode = true;
            bool allow_cached_graph_replay = false;
            bool collective_segmented_enabled = false;
            bool collectives_graph_capturable = false; ///< True when LocalTP NCCL/RCCL collectives are captured in the replay graph
            GraphReplayPlanPolicy graph_replay_plan_policy =
                GraphReplayPlanPolicy::RequireFullGraph; ///< Full graph unless a proven heterogeneous collective domain explicitly admits segmentation.
            bool force_recapture = false;              ///< Re-record graph segments on replay for callers with dynamic params not yet replay-safe
            bool defer_final_sync = false;             ///< Caller will synchronize the replay stream through a following operation.
            /**
             * Optional domain-level capture lifecycle rendezvous.
             *
             * Capture invokes this hook before beginCapture() and immediately
             * after endCapture(). LocalTP participants install one shared cyclic
             * barrier so every device enters and exits native capture as a
             * domain. The exit rendezvous remains required for a single full
             * graph because sibling graph instances are recorded independently.
             */
            GraphCaptureBoundaryHook capture_boundary;
        };

        /**
         * @brief Execute decode graph according to a single policy object
         *
         * Centralizes the configuration-time selection between cached GPU graph
         * replay, fast decode, and the general executor. Once cached replay is
         * selected, any capture or launch failure is returned as a hard
         * execution failure; this method never changes paths after an attempt.
         */
        bool executeDecodeWithCapturePolicy(
            ComputeGraph &graph,
            IDeviceContext *ctx,
            GraphSegmentCache *segment_cache,
            void *gpu_stream,
            IWorkerGPUContext *gpu_ctx,
            const std::unordered_set<std::string> *collective_nodes,
            const DecodeCapturePolicy &policy,
            bool *used_graph_replay = nullptr);

    private:
        GraphExecutorConfig config_;
        GraphExecutorStats stats_;
        ICollectiveContext *collective_ctx_ = nullptr; ///< Optional collective context (not owned)
        BufferArena *arena_ = nullptr;                 ///< Optional arena for contract coherence (not owned)
        StageTimeline stage_timeline_;                 ///< GPU event-based per-stage timeline profiler
        bool stage_timeline_info_populated_ = false;   ///< True after first setStageInfo pass (names never change)
        bool weights_session_cohered_ = false;         ///< True after first forward completes weight coherence for all nodes
        uint64_t snapshot_configuration_epoch_ = 1;   ///< Monotonic snapshot graph-topology identity

        void advanceSnapshotConfigurationEpoch() noexcept
        {
            ++snapshot_configuration_epoch_;
            if (snapshot_configuration_epoch_ == 0)
                snapshot_configuration_epoch_ = 1;
        }

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
                       const std::unordered_set<std::string> *collective_nodes = nullptr,
                       GraphSnapshotManifest *snapshot_manifest = nullptr);

        bool cancellationRequested(const std::string &node_name) const;
        void notifyStageFailure(const std::string &node_name, const std::string &reason) const;

        /**
         * @brief Validate every device-owned prepared weight declared by a stage.
         *
         * Prepared weight identity is the tuple
         * `{model, binding, kind, device}` held by PreparedWeightRef. Validation
         * happens before raw tensor coherence so a host-only source tensor can
         * never be mistaken for the allocation consumed by a GPU kernel.
         */
        bool validatePreparedWeightBindings(
            const ComputeNode &node,
            const StageBufferContract &contract,
            DeviceId target_device) const;

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
                      bool is_collective,
                      GraphSnapshotManifest *snapshot_manifest = nullptr);

        bool prepareOrRecordGraphSnapshotCopies(ComputeNode &node,
                                                DeviceId target_device,
                                                void *producer_stream,
                                                bool record_device_copy,
                                                GraphSnapshotManifest &snapshot_manifest);
        bool shouldCaptureSnapshotStage(const std::string &node_name) const;

        /**
         * @brief Validate graph-stable snapshot storage finalized by warmup.
         *
         * GPU graph capture cannot discover or allocate snapshot outputs while
         * capture is active. Warmup must therefore execute every selected
         * producer and finalize its exact descriptor/storage first. This method
         * hard-fails when that manifest is absent; it never substitutes a
         * pre-execution stage tensor. The captured point-in-time copy is still
         * recorded after each producer executes.
         */
        bool prepareGraphSnapshotCopies(ComputeNode &node,
                                        DeviceId target_device,
                                        void *producer_stream,
                                        GraphSnapshotManifest &snapshot_manifest);

        /**
         * @brief Enqueue point-in-time device copies for graph-captured snapshots.
         *
         * Fast decode, segmented graph capture, and prefill graph capture disable
         * host callbacks while device work is being recorded/launched. Draining
         * live stage outputs after the full graph finishes is incorrect when the
         * arena reuses an activation slot later in the graph. This helper records
         * a device-to-device copy immediately after the producing stage into
         * a unique device-visible mapped-host slot. Graph replay therefore
         * preserves the stage-boundary value without re-entering eager
         * execution or reserving a second activation graph in device memory.
         */
        bool captureGraphSnapshotCopies(ComputeNode &node,
                                        DeviceId target_device,
                                        void *producer_stream,
                                        GraphSnapshotManifest &snapshot_manifest);

        /**
         * @brief Publish one GPU stage from its immutable graph snapshot slots.
         *
         * Graph replay updates a mapped slot through its device-visible pointer
         * without executing stage C++. This method records a real post-launch
         * completion event, waits only for that publication point, builds a
         * callback descriptor solely from the frozen slot manifest, and invokes
         * the configured snapshot callback over the host-visible mapping.
         * It deliberately never calls getDumpInfo() on the live stage.
         */
        bool publishGraphSnapshotCopies(const std::string &stage_name,
                                        void *producer_stream,
                                        GraphSnapshotManifest &snapshot_manifest);

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
        /**
         * Snapshot owner for uncached eager diagnostics only.
         *
         * Every cached forward graph passes its ForwardGraphCache-owned manifest
         * explicitly. This transient state is never a cached executable owner.
         */
        GraphSnapshotManifest transient_snapshot_manifest_;

        float *getTemporaryBuffer(size_t elements);
    };

    // Backwards compatibility alias
    using LayerExecutor = DeviceGraphExecutor;

} // namespace llaminar2
