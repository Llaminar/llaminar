#pragma once

#include "DeviceGraphExecutor.h"

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_set>

namespace llaminar2
{

    /**
     * @brief Stateless helper that owns segmented graph-capture phase orchestration logic.
     *
     * This controller extracts the warmup/capture/replay state machine from `DeviceGraphExecutor`
     * so executor code stays focused on fallback policy and node-level primitives.
     *
     * Design notes for junior developers:
     * - All methods are static to keep this utility side-effect free outside passed-in state.
     * - Mutable execution state lives in `DeviceGraphExecutor::GraphSegmentCache`.
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
            /// True when normal launch failed and caller should use fast-decode fallback.
            bool launch_failure_fallback = false;
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
            /// Propagated fallback hint for normal launch failure.
            bool launch_failure_fallback = false;
        };

        /**
         * @brief Result for the full Phase-2 capture pass.
         */
        struct CapturePhaseResult
        {
            /// True when all segments in capture phase completed successfully.
            bool success = false;
            /// True when caller should abandon segmented path and execute fast decode.
            bool fallback_to_fast_decode = false;
            /// True when caller should reset `GraphSegmentCache` resources.
            bool reset_cache = false;
        };

        /**
         * @brief Result for the full replay phase.
         */
        struct ReplayPhaseResult
        {
            /// True when replay phase completed successfully.
            bool success = false;
            /// True when caller should trigger fast-decode fallback.
            bool launch_failure_fallback = false;
        };

        /**
         * @brief Executor-provided hooks used by capture/replay orchestration.
         *
         * Keeping these together reduces repeated lambda plumbing and makes call-sites clearer.
         */
        struct ReplayHooks
        {
            /// Ensures replay inputs/weights/outputs are coherent before launch.
            std::function<bool(const DeviceGraphExecutor::GraphSegment &)> cohere_inputs;
            /// Executes one stage through executor's canonical node path.
            std::function<bool(ComputeNode &)> execute_node;
            /// Runs post-launch lifecycle hooks (dirty marking, callbacks, step bookkeeping).
            std::function<void(DeviceGraphExecutor::GraphSegment &, void *)> post_launch;
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
         * @brief Segmented execution phase selector.
         */
        enum class Phase
        {
            /// First segmented pass: execute normally and build segment metadata.
            Warmup,
            /// Second segmented pass: capture graph segments.
            Capture,
            /// Steady state: replay captured graph segments.
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
         * @brief Advance segmented decode step and select warmup/capture/replay phase.
         * @param initialized Whether segmented state has completed warmup.
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
         * @brief Apply backend-specific device preparation for segmented capture.
         *
         * Currently ensures ROCm device binding is correct on the active thread.
         */
        static void prepareDeviceForSegmentedCapture(IDeviceContext *ctx);

        /**
         * @brief Execute warmup-phase bookkeeping (segment build + state transition).
         */
        static void executeWarmupPhase(
            ComputeGraph &graph,
            DeviceGraphExecutor::GraphSegmentCache &segment_cache,
            const std::unordered_set<std::string> *collective_nodes,
            bool has_collective_nodes,
            bool collectives_graph_capturable = false);

        /**
         * @brief Build segmented execution plan for warmup/capture/replay phases.
         *
         * Splits graph order into alternating capturable/manual segments and applies
         * collective safeguards plus optional max-stage partitioning.
         */
        static void buildWarmupSegments(
            ComputeGraph &graph,
            DeviceGraphExecutor::GraphSegmentCache &segment_cache,
            const std::unordered_set<std::string> *collective_nodes,
            bool has_collective_nodes,
            bool collectives_graph_capturable = false);

        /**
         * @brief Precompute `onGraphReplayed()` callback lists for capturable segments.
         */
        static void initializeReplayCallbacks(
            ComputeGraph &graph,
            DeviceGraphExecutor::GraphSegmentCache &segment_cache);

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
            const std::function<bool(ComputeNode &)> &execute_node_cb);

        /**
         * @brief Launch one capturable segment in normal replay mode.
         */
        static bool executeCapturedReplaySegmentNormal(
            DeviceGraphExecutor::GraphSegment &segment,
            IWorkerGPUContext *gpu_ctx,
            void *capture_stream,
            bool needs_segment_sync,
            const std::string &perf_context,
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
         * execute-node semantics for collective graphs.
         */
        static bool finalizeCapturePhaseCapturableSegment(
            ComputeGraph &graph,
            DeviceGraphExecutor::GraphSegment &segment,
            IDeviceContext *ctx,
            IWorkerGPUContext *gpu_ctx,
            void *capture_stream,
            bool has_collective_nodes,
            uint64_t current_step,
            const std::function<bool(ComputeNode &)> &execute_node_cb,
            const std::function<void(DeviceGraphExecutor::GraphSegment &, void *)> &post_launch_cb);

        /**
         * @brief Execute one manual segment during Phase-2 capture.
         */
        static bool executeCapturePhaseManualSegment(
            ComputeGraph &graph,
            DeviceGraphExecutor::GraphSegment &segment,
            IDeviceContext *ctx,
            IWorkerGPUContext *gpu_ctx,
            bool has_collective_nodes,
            uint64_t current_step,
            const std::function<bool(ComputeNode &)> &execute_node_cb);

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
            int segment_index,
            const std::string &perf_context,
            const std::function<bool(const DeviceGraphExecutor::GraphSegment &)> &cohere_inputs_cb,
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
            uint64_t current_step,
            int segment_index,
            const std::string &perf_context,
            const std::function<bool(const DeviceGraphExecutor::GraphSegment &)> &cohere_inputs_cb,
            const std::function<bool(ComputeNode &)> &execute_node_cb,
            const std::function<void(DeviceGraphExecutor::GraphSegment &, void *)> &post_launch_cb);

        /**
         * @brief Execute full Phase-2 capture over all segments.
         */
        static CapturePhaseResult executeCapturePhase(
            ComputeGraph &graph,
            DeviceGraphExecutor::GraphSegmentCache &segment_cache,
            IDeviceContext *ctx,
            IWorkerGPUContext *gpu_ctx,
            bool has_collective_nodes,
            uint64_t current_step,
            const ReplayHooks &hooks);

        /**
         * @brief Execute full replay phase over all segments.
         */
        static ReplayPhaseResult executeReplayPhase(
            ComputeGraph &graph,
            DeviceGraphExecutor::GraphSegmentCache &segment_cache,
            IDeviceContext *ctx,
            IWorkerGPUContext *gpu_ctx,
            bool has_collective_nodes,
            uint64_t current_step,
            const ReplayHooks &hooks,
            bool force_recapture = false);

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
         * Applies output-dirty marking, replay callbacks, and step bookkeeping.
         *
         * @param skip_replay_callbacks When true, skips onGraphReplayed() callbacks.
         *        Used during the capture phase where execute() already ran host-side
         *        bookkeeping (e.g., KV cache head advancement). Calling onGraphReplayed()
         *        during capture would double-advance host state.
         */
        static void postCapturedSegmentLaunch(
            ComputeGraph &graph,
            DeviceGraphExecutor::GraphSegment &segment,
            uint64_t current_step,
            void *stream,
            const std::function<void(ComputeNode &, void *)> &mark_stage_outputs_dirty_cb,
            bool skip_replay_callbacks = false);
    };

} // namespace llaminar2
