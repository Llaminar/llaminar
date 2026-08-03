#pragma once

#include "DeviceGraphExecutor.h"

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_set>

namespace llaminar2
{

    /**
     * @brief Stateless helper that owns GPU graph capture/replay phase orchestration logic.
     *
     * This controller extracts the warmup/capture/replay state machine from `DeviceGraphExecutor`
     * so executor code stays focused on mandatory execution policy and
     * node-level primitives.
     *
     * Design notes for junior developers:
     * - All methods are static to keep this utility side-effect free outside passed-in state.
     * - Mutable execution state lives in `DeviceGraphExecutor::GraphSegmentCache`.
     *   A cache with one capturable unit and no manual units is a full-graph
     *   replay plan. Segmented replay is valid only for a proven mixed-device
     *   graph that contains heterogeneous collectives.
     * - Executor-provided hooks let this controller call back into execution/coherence behavior
     *   without creating circular ownership.
     */
    class DeviceGraphCaptureController
    {
    public:
        /**
         * @brief Result for running one capturable segment during replay.
         */
        struct ReplayCapturableResult
        {
            /// True when segment replay path succeeded.
            bool success = false;
            /// True when verify mode intentionally skipped non-idempotent replay comparison.
            bool skipped_non_idempotent = false;
        };

        /**
         * @brief Result for running one segment during replay (capturable or manual).
         */
        struct ReplaySegmentResult
        {
            /// True when the selected segment path succeeded.
            bool success = false;
            /// Propagated verify-mode skip marker.
            bool skipped_non_idempotent = false;
        };

        /**
         * @brief Result for the full Phase-2 capture pass.
         */
        struct CapturePhaseResult
        {
            /// True when all segments in capture phase completed successfully.
            bool success = false;
            /// True when caller should reset `GraphSegmentCache` resources.
            bool reset_cache = false;
        };

        /**
         * @brief Decide whether Phase-2 capture also executes the new graph.
         *
         * `ExecuteCapturedWork` is used when capture itself is the logical
         * invocation. `MaterializeOnly` is used immediately after a successful
         * warmup in the same initialization transaction: warmup already produced
         * the user-visible result, so Phase 2 may record and instantiate the
         * future replay executable but must not launch or manually re-execute any
         * stage. This closed policy prevents first-use graph setup from mutating
         * KV or recurrent state twice.
         */
        enum class CapturePublicationPolicy : uint8_t
        {
            ExecuteCapturedWork,
            MaterializeOnly,
        };

        /**
         * @brief Result for the full replay phase.
         */
        struct ReplayPhaseResult
        {
            /// True when replay phase completed successfully.
            bool success = false;
        };

        /**
         * @brief Executor-provided hooks used by capture/replay orchestration.
         *
         * Keeping these together reduces repeated lambda plumbing and makes call-sites clearer.
         */
        struct ReplayHooks
        {
            /**
             * @brief Joins every external producer event to the exact graph stream.
             *
             * This hook is mandatory for native capture and diagnostic
             * recapture. It runs after stage-owned launch metadata has been
             * prepared and before `beginCapture()`, because metadata preparation
             * may itself publish a new device-write event. A capture transaction
             * must never discover or import that external event from inside its
             * recorded body.
             */
            std::function<bool(const DeviceGraphExecutor::GraphSegment &)> cohere_inputs;
            /// Executes one stage through executor's canonical node path.
            std::function<bool(ComputeNode &)> execute_node;
            /// Preallocates point-in-time snapshot copy descriptors/storage before capture.
            std::function<bool(ComputeNode &, void *)> prepare_snapshot_copies;
            /// Records point-in-time snapshot copies after direct stage execution.
            std::function<bool(ComputeNode &, void *)> record_snapshot_copies;
            /// Runs post-launch lifecycle hooks (dirty marking, callbacks, step bookkeeping).
            std::function<void(DeviceGraphExecutor::GraphSegment &, void *)> post_launch;
            /**
             * Optional domain-level fence at both capture entry and capture exit.
             *
             * One callback owns both transitions so callers cannot configure a
             * begin-only LocalTP lifecycle accidentally.
             */
            DeviceGraphExecutor::GraphCaptureBoundaryHook capture_boundary;
        };

        /**
         * @brief Result for verify-mode replay of one capturable segment.
         */
        struct VerifyReplayResult
        {
            /// True when verify flow completed (including skip path).
            bool success = false;
            /// True when skipped due to non-idempotent stage semantics.
            bool skipped_non_idempotent = false;
        };

        /**
         * @brief Cached graph-replay phase selector.
         */
        enum class Phase
        {
            /// First graph-replay pass: execute normally and build replay metadata.
            Warmup,
            /// Explicit capture of a caller-supplied initialized cache state.
            Capture,
            /// Steady state: replay captured graph units.
            Replay
        };

        /**
         * @brief Return value from phase transition logic.
         */
        struct Transition
        {
            /// Selected phase for this decode step.
            Phase phase = Phase::Replay;
            /// Monotonic segmented decode step index.
            uint64_t decode_step = 0;
        };

        /**
         * @brief Return the canonical PerfStats label for a replay phase.
         * @param phase Typed warmup, capture, or replay phase.
         * @return Stable label consumed by graph-lifecycle validators.
         *
         * Every graph launch path, including optimized direct launches, must
         * publish one of these labels. Keeping the vocabulary here prevents a
         * launch optimization from creating a private phase name that silently
         * disappears from lifecycle validation.
         */
        static const char *phaseName(Phase phase);

        /**
         * @brief Advance decode replay step and select warmup/capture/replay phase.
         * @param initialized Whether graph-replay state has completed warmup.
         * @param needs_capture In/out flag requesting Phase-2 capture.
         * @param decode_step In/out monotonic step counter.
         * @return Phase transition decision with updated step.
         */
        static Transition beginStep(bool initialized, bool &needs_capture, uint64_t &decode_step);

        /**
         * @brief Mark warmup completion and arm capture phase for next step.
         */
        static void markWarmupComplete(bool &initialized, bool &needs_capture);

        /**
         * @brief Apply backend-specific device preparation for cached graph capture.
         *
         * Currently ensures ROCm device binding is correct on the active thread.
         */
        static void prepareDeviceForGraphCapture(IDeviceContext *ctx);

        /**
         * @brief Compute a graph-wide launch-topology variant signature.
         *
         * The signature is zero when every stage reports the default variant.
         * Nonzero stage signatures are folded with stage identity so the
         * replay cache can recapture before replaying a graph whose baked
         * launch topology no longer matches current dynamic parameters.
         */
        static uint64_t computeCaptureVariantSignature(ComputeGraph &graph);

        /**
         * @brief Return a stable mode label for the built replay plan.
         *
         * A cache with exactly one capturable unit and no manual units is a
         * full-graph replay plan. Plans with manual boundaries or multiple
         * captured units are segmented replay plans.
         */
        static const char *replayModeName(const DeviceGraphExecutor::GraphSegmentCache &segment_cache);

        /**
         * @brief Allocate the bounded replay-timing event ring before graph capture.
         *
         * When GPU stage timing is disabled this is a no-op. When it is enabled,
         * the complete bounded sampling ring is created here during warmup;
         * allocation failure is fatal to the selected profiling execution mode.
         * A burst larger than the ring is counted as unsampled while all slots
         * are busy; it never changes inference ordering. No replay invocation
         * may lazily create an event.
         */
        static bool prepareReplayGpuTiming(
            DeviceGraphExecutor::GraphSegmentCache &segment_cache,
            IWorkerGPUContext *gpu_ctx,
            const std::string &device_name);

        /**
         * @brief Emit every completed replay interval without waiting on the GPU.
         *
         * The method uses queryEventChecked() and leaves incomplete slots owned
         * by the cache. It never synchronizes an event, stream, or device. A
         * backend query or elapsed-time failure is a hard profiling failure.
         */
        static bool reclaimReplayGpuTimingNonblocking(
            DeviceGraphExecutor::GraphSegmentCache &segment_cache,
            IWorkerGPUContext *gpu_ctx);

        /**
         * @brief Execute warmup-phase bookkeeping (segment build + state transition).
         */
        static void executeWarmupPhase(
            ComputeGraph &graph,
            DeviceGraphExecutor::GraphSegmentCache &segment_cache,
            const std::unordered_set<std::string> *collective_nodes,
            bool has_collective_nodes,
            bool collectives_graph_capturable = false,
            DeviceGraphExecutor::GraphReplayPlanPolicy plan_policy =
                DeviceGraphExecutor::GraphReplayPlanPolicy::RequireFullGraph);

        /**
         * @brief Build and validate the execution plan for warmup/capture/replay.
         *
         * Homogeneous domains must produce one full graph. A caller-proven
         * heterogeneous collective domain may produce alternating captured and
         * manual units where cross-device execution prevents monolithic capture.
         */
        static void buildWarmupSegments(
            ComputeGraph &graph,
            DeviceGraphExecutor::GraphSegmentCache &segment_cache,
            const std::unordered_set<std::string> *collective_nodes,
            bool has_collective_nodes,
            bool collectives_graph_capturable = false,
            DeviceGraphExecutor::GraphReplayPlanPolicy plan_policy =
                DeviceGraphExecutor::GraphReplayPlanPolicy::RequireFullGraph);

        /**
         * @brief Execute replay in stream-only diagnostic mode (no graph launch).
         */
        static bool executeStreamOnlyReplay(
            ComputeGraph &graph,
            DeviceGraphExecutor::GraphSegmentCache &segment_cache,
            IDeviceContext *ctx,
            IWorkerGPUContext *gpu_ctx,
            void *capture_stream,
            bool use_default_stream);

        /**
         * @brief Detect stages that are non-idempotent for verify-mode comparison.
         */
        static bool segmentHasNonIdempotentStage(
            ComputeGraph &graph,
            const DeviceGraphExecutor::GraphSegment &segment);

        /**
         * @brief Execute one manual (non-capturable) segment during replay.
         */
        static bool executeManualReplaySegment(
            ComputeGraph &graph,
            DeviceGraphExecutor::GraphSegment &segment,
            IDeviceContext *ctx,
            IWorkerGPUContext *gpu_ctx,
            void *capture_stream,
            bool has_collective_nodes,
            bool needs_segment_sync,
            uint64_t current_step,
            const std::function<bool(ComputeNode &)> &execute_node_cb,
            const std::function<bool(ComputeNode &, void *)> &record_snapshot_copies_cb);

        /**
         * @brief Launch one capturable segment in normal replay mode.
         */
        static bool executeCapturedReplaySegmentNormal(
            DeviceGraphExecutor::GraphSegment &segment,
            IWorkerGPUContext *gpu_ctx,
            void *capture_stream,
            bool needs_segment_sync,
            bool full_graph_replay,
            const std::string &perf_context,
            const std::string &device_name,
            const std::function<void(DeviceGraphExecutor::GraphSegment &, void *)> &post_launch_cb);

        /**
         * @brief Re-capture and replay one capturable segment in diagnostics mode.
         */
        static bool executeCapturedReplaySegmentRecapture(
            ComputeGraph &graph,
            DeviceGraphExecutor::GraphSegment &segment,
            IDeviceContext *ctx,
            IWorkerGPUContext *gpu_ctx,
            void *capture_stream,
            int segment_index,
            uint64_t current_step,
            const std::string &perf_context,
            const std::function<bool(const DeviceGraphExecutor::GraphSegment &)> &cohere_inputs_cb,
            const DeviceGraphExecutor::GraphCaptureBoundaryHook &capture_boundary_cb,
            const std::function<bool(ComputeNode &, void *)> &record_snapshot_copies_cb,
            const std::function<void(DeviceGraphExecutor::GraphSegment &, void *)> &post_launch_cb);

        /**
         * @brief Verify captured replay against direct execution for one segment.
         */
        static VerifyReplayResult executeCapturedReplaySegmentVerify(
            ComputeGraph &graph,
            DeviceGraphExecutor::GraphSegment &segment,
            IDeviceContext *ctx,
            IWorkerGPUContext *gpu_ctx,
            void *capture_stream,
            bool needs_segment_sync,
            int segment_index,
            const std::function<void(DeviceGraphExecutor::GraphSegment &, void *)> &post_launch_cb);

        /**
         * @brief Finalize one captured segment during Phase-2 capture.
         *
         * Handles instantiate/launch path for non-collective graphs and Phase-2
         * execute-node semantics for collective graphs. A segment advertised as
         * capturable must produce at least one native graph node; zero-node
         * capture is a violated stage/stream ownership contract and fails.
         *
         * @param full_graph_capture True when this segment is the complete
         *        homogeneous graph rather than one unit of an explicitly
         *        admitted heterogeneous collective plan.
         * @param perf_context Stable graph-owner context attached to the
         *        executable-node PerfStats evidence.
         * @param publication_policy Whether the captured executable represents
         *        this invocation or is only being materialized for future replay.
         */
        static bool finalizeCapturePhaseCapturableSegment(
            ComputeGraph &graph,
            DeviceGraphExecutor::GraphSegment &segment,
            IDeviceContext *ctx,
            IWorkerGPUContext *gpu_ctx,
            void *capture_stream,
            bool has_collective_nodes,
            bool full_graph_capture,
            const std::string &perf_context,
            uint64_t current_step,
            CapturePublicationPolicy publication_policy,
            const std::function<bool(ComputeNode &)> &execute_node_cb,
            const std::function<bool(ComputeNode &, void *)> &record_snapshot_copies_cb,
            const std::function<void(DeviceGraphExecutor::GraphSegment &, void *)> &post_launch_cb);

        /**
         * @brief Execute one manual segment during Phase-2 capture.
         */
        static bool executeCapturePhaseManualSegment(
            ComputeGraph &graph,
            DeviceGraphExecutor::GraphSegment &segment,
            IDeviceContext *ctx,
            IWorkerGPUContext *gpu_ctx,
            void *capture_stream,
            bool has_collective_nodes,
            uint64_t current_step,
            const std::function<bool(ComputeNode &)> &execute_node_cb,
            const std::function<bool(ComputeNode &, void *)> &record_snapshot_copies_cb);

        /**
         * @brief Run stage-owned preparation required at one graph lifecycle boundary.
         */
        static bool prepareGraphLaunchMetadata(
            ComputeGraph &graph,
            const DeviceGraphExecutor::GraphSegment &segment,
            IDeviceContext *ctx,
            void *stream,
            GraphLaunchPreparationPhase phase);

        /**
         * @brief Execute one capturable replay segment under selected diagnostics mode.
         */
        static ReplayCapturableResult executeReplayCapturableSegment(
            ComputeGraph &graph,
            DeviceGraphExecutor::GraphSegment &segment,
            IDeviceContext *ctx,
            IWorkerGPUContext *gpu_ctx,
            void *capture_stream,
            bool needs_segment_sync,
            bool verify_mode,
            bool recapture_mode,
            bool full_graph_replay,
            int segment_index,
            uint64_t current_step,
            const std::string &perf_context,
            const std::function<bool(const DeviceGraphExecutor::GraphSegment &)> &cohere_inputs_cb,
            const DeviceGraphExecutor::GraphCaptureBoundaryHook &capture_boundary_cb,
            const std::function<bool(ComputeNode &, void *)> &record_snapshot_copies_cb,
            const std::function<void(DeviceGraphExecutor::GraphSegment &, void *)> &post_launch_cb);

        /**
         * @brief Execute one replay segment (capturable or manual).
         */
        static ReplaySegmentResult executeReplaySegment(
            ComputeGraph &graph,
            DeviceGraphExecutor::GraphSegment &segment,
            IDeviceContext *ctx,
            IWorkerGPUContext *gpu_ctx,
            void *capture_stream,
            bool has_collective_nodes,
            bool needs_segment_sync,
            bool verify_mode,
            bool recapture_mode,
            bool full_graph_replay,
            uint64_t current_step,
            int segment_index,
            const std::string &perf_context,
            const std::function<bool(const DeviceGraphExecutor::GraphSegment &)> &cohere_inputs_cb,
            const std::function<bool(ComputeNode &)> &execute_node_cb,
            const DeviceGraphExecutor::GraphCaptureBoundaryHook &capture_boundary_cb,
            const std::function<bool(ComputeNode &, void *)> &record_snapshot_copies_cb,
            const std::function<void(DeviceGraphExecutor::GraphSegment &, void *)> &post_launch_cb);

        /**
         * @brief Execute full Phase-2 capture over all segments.
         *
         * Materialization-only capture requires one complete capturable graph;
         * a manual unit would necessarily execute host-orchestrated work and is
         * therefore rejected rather than silently changing semantics.
         */
        static CapturePhaseResult executeCapturePhase(
            ComputeGraph &graph,
            DeviceGraphExecutor::GraphSegmentCache &segment_cache,
            IDeviceContext *ctx,
            IWorkerGPUContext *gpu_ctx,
            bool has_collective_nodes,
            uint64_t current_step,
            const ReplayHooks &hooks,
            CapturePublicationPolicy publication_policy =
                CapturePublicationPolicy::ExecuteCapturedWork);

        /**
         * @brief Execute full replay phase over all segments.
         */
        static ReplayPhaseResult executeReplayPhase(
            ComputeGraph &graph,
            DeviceGraphExecutor::GraphSegmentCache &segment_cache,
            IDeviceContext *ctx,
            IWorkerGPUContext *gpu_ctx,
            bool has_collective_nodes,
            bool collectives_graph_capturable,
            uint64_t current_step,
            const ReplayHooks &hooks,
            bool force_recapture = false,
            bool defer_final_sync = false);

        /**
         * @brief Coherence helper for all stages in one replay segment.
         */
        static bool cohereReplaySegmentInputs(
            ComputeGraph &graph,
            const DeviceGraphExecutor::GraphSegment &segment,
            const std::function<bool(ComputeNode &)> &cohere_stage_cb);

        /**
         * @brief Post-launch lifecycle for one captured segment.
         *
         * Applies output-dirty marking and step bookkeeping. Device state
         * publication is part of the captured graph; host replay callbacks are
         * deliberately not an execution primitive.
         */
        static void postCapturedSegmentLaunch(
            ComputeGraph &graph,
            DeviceGraphExecutor::GraphSegment &segment,
            uint64_t current_step,
            void *stream,
            const std::function<void(BufferId, DeviceId)> &mark_arena_write_dirty_cb);
    };

} // namespace llaminar2
