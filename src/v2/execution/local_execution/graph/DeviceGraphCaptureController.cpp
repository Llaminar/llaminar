/**
 * @file DeviceGraphCaptureController.cpp
 * @brief Native graph capture lifecycle planning and execution.
 *
 * This implementation validates graph-owned lifecycle envelopes, materializes
 * their replay units, and preserves every producer/consumer edge with streams,
 * events, or an explicitly declared immutable host ticket.
 */

#include "DeviceGraphCaptureController.h"
#include "GraphCaptureGuard.h"

#include "../coherence/CoherencePolicy.h"
#include "../../../tensors/TensorClasses.h"
#include "../../../utils/DebugEnv.h"
#include "../../../utils/ForwardPassProfiler.h"
#include "../../../utils/KernelProfiler.h"
#include "../../../utils/Logger.h"
#include "../../../utils/PerfStatsCollector.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace llaminar2
{
    namespace
    {
        enum class GraphReplayCaptureMode
        {
            Segmented,
            FullGraph
        };

        const char *graphExecutableLaunchPhaseName(
            DeviceGraphExecutor::GraphExecutableLaunchPhase phase)
        {
            using Phase = DeviceGraphExecutor::GraphExecutableLaunchPhase;
            switch (phase)
            {
            case Phase::InitialTransaction:
                return "initial_transaction";
            case Phase::SteadyReplay:
                return "steady_replay";
            case Phase::DiagnosticRecapture:
                return "diagnostic_recapture";
            case Phase::DiagnosticVerification:
                return "diagnostic_verification";
            }
            return "unknown";
        }

        bool applyGraphLaunchDependency(
            const DeviceGraphExecutor::GraphLaunchDependencyHook &hook,
            DeviceGraphExecutor::GraphExecutableLaunchPhase phase,
            void *execution_stream)
        {
            if (!hook)
                return true;
            if (!execution_stream)
            {
                LOG_ERROR(
                    "[DeviceGraphCaptureController] Captured executable launch "
                    "dependency requires the exact non-null graph stream"
                    << " phase=" << graphExecutableLaunchPhaseName(phase));
                return false;
            }
            if (!hook(phase, execution_stream))
            {
                LOG_ERROR(
                    "[DeviceGraphCaptureController] Captured executable launch "
                    "dependency was rejected"
                    << " phase=" << graphExecutableLaunchPhaseName(phase)
                    << " stream=" << execution_stream);
                return false;
            }
            return true;
        }

        /**
         * @brief Pair one graph-owned auxiliary fork and join structurally.
         *
         * The owner lives wholly inside an already-open native capture. A stage
         * failure still calls @ref finish before the backend capture closes, so
         * the auxiliary stream can never remain an unjoined member of the
         * capture transaction. Destruction with an active branch is fatal
         * because ending the primary capture without its terminal edge would
         * leave backend stream ownership unknowable.
         */
        class ScopedCapturedAuxiliaryBranch final
        {
        public:
            /** Bind an optional cache-owned branch and exact primary stream. */
            ScopedCapturedAuxiliaryBranch(
                IGraphCaptureAuxiliaryBranch *branch,
                void *primary_stream) noexcept
                : branch_(branch), primary_stream_(primary_stream)
            {
            }

            /** Join an abandoned active branch or terminate on lost ordering. */
            ~ScopedCapturedAuxiliaryBranch()
            {
                if (!active_)
                    return;
                if (!branch_->recordJoin(primary_stream_))
                {
                    LOG_ERROR(
                        "[DeviceGraphCaptureController] Fatal failure joining an abandoned captured auxiliary branch '"
                        << branch_->name() << "'");
                    std::terminate();
                }
            }

            ScopedCapturedAuxiliaryBranch(
                const ScopedCapturedAuxiliaryBranch &) = delete;
            ScopedCapturedAuxiliaryBranch &operator=(
                const ScopedCapturedAuxiliaryBranch &) = delete;

            /** @return true after an absent branch or a complete recorded fork. */
            [[nodiscard]] bool begin() noexcept
            {
                if (!branch_)
                    return true;
                if (!primary_stream_ || !branch_->recordFork(primary_stream_))
                    return false;
                active_ = true;
                return true;
            }

            /** @return true after an absent branch or a complete recorded join. */
            [[nodiscard]] bool finish() noexcept
            {
                if (!active_)
                    return branch_ == nullptr;
                if (!branch_->recordJoin(primary_stream_))
                    return false;
                active_ = false;
                return true;
            }

        private:
            IGraphCaptureAuxiliaryBranch *branch_ = nullptr;
            void *primary_stream_ = nullptr;
            bool active_ = false;
        };

        const char *captureModeTag(GraphReplayCaptureMode mode)
        {
            return mode == GraphReplayCaptureMode::FullGraph ? "full_graph" : "segmented";
        }

        const char *captureModeSource(GraphReplayCaptureMode mode)
        {
            return mode == GraphReplayCaptureMode::FullGraph
                       ? "full_graph_capture"
                       : "segmented_graph_capture";
        }

        const char *captureModeHostScope(GraphReplayCaptureMode mode)
        {
            return mode == GraphReplayCaptureMode::FullGraph
                       ? "full_graph_replay_host"
                       : "segmented_replay_host";
        }

        const char *captureModeEventScope(GraphReplayCaptureMode mode)
        {
            return mode == GraphReplayCaptureMode::FullGraph
                       ? "full_graph_replay_events"
                       : "segmented_replay_events";
        }

        const char *captureModePlanScope(GraphReplayCaptureMode mode)
        {
            return mode == GraphReplayCaptureMode::FullGraph
                       ? "full_graph_capture_plan"
                       : "segmented_capture_plan";
        }

        std::string replayMetricName(GraphReplayCaptureMode mode, const char *suffix)
        {
            return std::string(mode == GraphReplayCaptureMode::FullGraph
                                   ? "full_graph_replay_"
                                   : "segmented_replay_") +
                   suffix;
        }

        std::string planMetricName(GraphReplayCaptureMode mode, const char *suffix)
        {
            return std::string(mode == GraphReplayCaptureMode::FullGraph
                                   ? "full_graph_plan_"
                                   : "segmented_plan_") +
                   suffix;
        }

        std::string planUnitMetricName(GraphReplayCaptureMode mode)
        {
            return planMetricName(mode, mode == GraphReplayCaptureMode::FullGraph
                                            ? "graphs"
                                            : "segments");
        }

        std::string planMaxUnitStagesMetricName(GraphReplayCaptureMode mode)
        {
            return planMetricName(mode, mode == GraphReplayCaptureMode::FullGraph
                                            ? "max_graph_stages"
                                            : "max_segment_stages");
        }

        const char *stageGpuPlanUnitMetricName(GraphReplayCaptureMode mode)
        {
            return mode == GraphReplayCaptureMode::FullGraph
                       ? "graph_replay_plan_graphs"
                       : "graph_replay_plan_segments";
        }

        const char *planUnitTypeTagName(GraphReplayCaptureMode mode)
        {
            return mode == GraphReplayCaptureMode::FullGraph
                       ? "graph_type"
                       : "segment_type";
        }

        const char *replayUnitCounterName(GraphReplayCaptureMode mode)
        {
            return mode == GraphReplayCaptureMode::FullGraph
                       ? "full_graph_replay_calls"
                       : "segmented_replay_segments";
        }

        std::string replayUnitTimingName(GraphReplayCaptureMode mode)
        {
            return replayMetricName(mode, mode == GraphReplayCaptureMode::FullGraph
                                              ? "graph"
                                              : "segment");
        }

        const char *replayUnitGpuEventName(GraphReplayCaptureMode mode)
        {
            return mode == GraphReplayCaptureMode::FullGraph
                       ? "graph_replay.graph"
                       : "graph_replay.segment";
        }

        const char *replayUnitIndexTagName(GraphReplayCaptureMode mode)
        {
            return mode == GraphReplayCaptureMode::FullGraph
                       ? "graph_index"
                       : "segment_index";
        }

        const char *replayUnitTimingScope(GraphReplayCaptureMode mode)
        {
            return mode == GraphReplayCaptureMode::FullGraph
                       ? "graph_replay_gpu_event"
                       : "segment_replay_gpu_event";
        }

        GraphReplayCaptureMode captureModeForPlan(size_t total_segments,
                                                  size_t capturable_segments,
                                                  size_t manual_segments)
        {
            return total_segments == 1 && capturable_segments == 1 && manual_segments == 0
                       ? GraphReplayCaptureMode::FullGraph
                       : GraphReplayCaptureMode::Segmented;
        }

        GraphReplayCaptureMode captureModeForCache(const DeviceGraphExecutor::GraphSegmentCache &cache)
        {
            size_t capturable_segments = 0;
            size_t manual_segments = 0;
            for (const auto &segment : cache.segments)
            {
                if (segment.capturable)
                    ++capturable_segments;
                else
                    ++manual_segments;
            }
            return captureModeForPlan(cache.segments.size(), capturable_segments, manual_segments);
        }

        const char *segmentTypeName(const DeviceGraphExecutor::GraphSegment &segment)
        {
            return segment.capturable ? "capturable" : "manual";
        }

        std::string describeSegmentStages(const DeviceGraphExecutor::GraphSegment &segment)
        {
            std::ostringstream out;
            out << '[';
            for (size_t i = 0; i < segment.stage_names.size(); ++i)
            {
                if (i > 0)
                    out << ", ";
                out << segment.stage_names[i];
            }
            out << ']';
            return out.str();
        }

        bool manualSegmentRequiresHostTicketFence(
            ComputeGraph &graph,
            const DeviceGraphExecutor::GraphSegment &segment,
            std::string *consumer_stage)
        {
            if (consumer_stage)
                consumer_stage->clear();
            if (segment.capturable)
                return false;
            for (const auto &stage_name : segment.stage_names)
            {
                const auto *node = graph.getNode(stage_name);
                if (!node || !node->stage ||
                    !node->stage->requiresHostGraphTicketFence())
                {
                    continue;
                }
                if (!node->stage->isManualGraphBoundary())
                {
                    throw std::logic_error(
                        "Stage '" + stage_name +
                        "' requests a host graph-ticket fence without declaring a manual graph boundary");
                }
                if (consumer_stage)
                    *consumer_stage = stage_name;
                return true;
            }
            return false;
        }

        void awaitManualHostTicketBoundary(
            ComputeGraph &graph,
            DeviceGraphExecutor::GraphSegmentCache &segment_cache,
            size_t segment_index,
            const char *phase,
            const DeviceId &device)
        {
            if (segment_index >= segment_cache.segments.size())
                throw std::out_of_range("Host ticket boundary segment index is invalid");

            std::string consumer_stage;
            const auto &segment = segment_cache.segments[segment_index];
            if (!manualSegmentRequiresHostTicketFence(
                    graph,
                    segment,
                    &consumer_stage))
            {
                return;
            }
            if (segment_index == 0 ||
                !segment_cache.segments[segment_index - 1].capturable)
            {
                throw std::runtime_error(
                    "Manual host ticket consumer '" + consumer_stage +
                    "' is not immediately preceded by a captured producer segment");
            }

            segment_cache.waitForManualHostTicketFence();
            PerfStatsCollector::addCounter(
                "forward_graph",
                "heterogeneous_host_ticket_fences",
                1.0,
                phase,
                device.toString(),
                {{"consumer_stage", consumer_stage},
                 {"context", segment_cache.perf_context},
                 {"authority", "captured_pinned_ticket"}});
        }

        /**
         * @brief Materialize one frozen stage-order ledger before beginCapture().
         *
         * The production executor supplies arena-derived internal edges.  Pure
         * controller tests have no BufferArena and receive an empty plan.  If
         * such a test invokes the production runStage() path, stage entry fails
         * immediately because there is no planned stage; simple state-machine
         * callback tests can still exercise begin/end orchestration without
         * constructing tensors.
         */
        std::unique_ptr<GraphCaptureDependencyLedger> planCaptureDependencies(
            ComputeGraph &graph,
            const DeviceGraphExecutor::GraphSegment &segment,
            DeviceId device,
            void *stream,
            const std::function<std::unique_ptr<GraphCaptureDependencyLedger>(
                const DeviceGraphExecutor::GraphSegment &)> &planner,
            const std::string &context)
        {
            if (planner)
                return planner(segment);

            (void)graph;
            (void)segment;
            std::vector<GraphCaptureDependencyLedger::StagePlan> stages;

            try
            {
                return std::make_unique<GraphCaptureDependencyLedger>(
                    device, stream, std::move(stages), context);
            }
            catch (const std::exception &e)
            {
                LOG_ERROR(
                    "[DeviceGraphCaptureController] Strict capture dependency "
                    "planning failed ("
                    << context << "): " << e.what());
                return nullptr;
            }
        }

        /**
         * @brief Return the strict cross-participant name for one capture wave.
         *
         * Raw replay-segment indexes are intentionally absent. A heterogeneous
         * root owns manual sparse segments that its sibling device graphs do not
         * contain. Capture-wave ordinals count only native-capture lifecycle
         * participation and therefore remain symmetric. Explicit identities
         * admit deliberately different stage ranges; implicit waves retain the
         * old first/last-stage structural check.
         */
        std::string graphCaptureBoundaryName(
            const char *phase,
            uint64_t current_step,
            size_t capture_wave_ordinal,
            const std::string &capture_wave_identity,
            const DeviceGraphExecutor::GraphSegment *active_segment,
            const std::string &perf_context)
        {
            std::ostringstream out;
            out << "before_begin:"
                << "phase=" << phase
                << ":step=" << current_step
                << ":wave=" << capture_wave_ordinal;
            if (!capture_wave_identity.empty())
            {
                out << ":identity=explicit:" << capture_wave_identity;
            }
            else if (active_segment)
            {
                const std::string first_stage =
                    active_segment->stage_names.empty()
                        ? std::string("<empty>")
                        : active_segment->stage_names.front();
                const std::string last_stage =
                    active_segment->stage_names.empty()
                        ? std::string("<empty>")
                        : active_segment->stage_names.back();
                out << ":identity=implicit:"
                    << first_stage << ".." << last_stage;
            }
            else
            {
                throw std::logic_error(
                    "Passive capture wave requires an explicit identity");
            }
            if (!perf_context.empty())
                out << ":context=" << perf_context;
            return out.str();
        }

        /**
         * @brief Join sibling-only capture waves after the anchor segment completes.
         *
         * The anchor may be a launched graph or a completed manual collective.
         * The begin rendezvous holds this participant while a sibling performs
         * intervening heterogeneous work. The immediate end rendezvous then
         * holds it until that sibling has finished recording and launching its
         * native graph. No empty graph, dummy transfer, device allocation, or
         * kernel is introduced on the passive participant.
         */
        bool joinPassiveCaptureWaves(
            ComputeGraph &graph,
            const DeviceGraphExecutor::GraphSegment &segment,
            const char *phase_prefix,
            uint64_t current_step,
            const std::string &perf_context,
            const DeviceGraphExecutor::GraphCaptureBoundaryHook &capture_boundary,
            void *capture_stream,
            const DeviceId &device)
        {
            if (segment.passive_capture_waves_after.empty())
                return true;
            if (!capture_boundary || !capture_stream)
            {
                LOG_ERROR(
                    "[DeviceGraphCaptureController] Passive capture waves require "
                    "a domain boundary hook and exact non-null stream");
                return false;
            }

            for (const auto &passive_wave :
                 segment.passive_capture_waves_after)
            {
                const std::string begin_phase =
                    std::string(phase_prefix) + "_begin";
                const std::string begin_boundary =
                    graphCaptureBoundaryName(
                        begin_phase.c_str(),
                        current_step,
                        passive_wave.ordinal,
                        passive_wave.identity,
                        nullptr,
                        perf_context);
                if (!capture_boundary(begin_boundary, capture_stream))
                {
                    LOG_ERROR(
                        "[DeviceGraphCaptureController] Passive capture-wave begin "
                        "rendezvous failed boundary="
                        << begin_boundary);
                    return false;
                }

                const std::string end_phase =
                    std::string(phase_prefix) + "_end";
                const std::string end_boundary =
                    graphCaptureBoundaryName(
                        end_phase.c_str(),
                        current_step,
                        passive_wave.ordinal,
                        passive_wave.identity,
                        nullptr,
                        perf_context);
                if (!capture_boundary(end_boundary, capture_stream))
                {
                    LOG_ERROR(
                        "[DeviceGraphCaptureController] Passive capture-wave end "
                        "rendezvous failed boundary="
                        << end_boundary);
                    return false;
                }

                PerfStatsCollector::addCounter(
                    "forward_graph",
                    "passive_capture_wave_joins",
                    1.0,
                    phase_prefix,
                    device.toString(),
                    {{"identity", passive_wave.identity},
                     {"wave", std::to_string(passive_wave.ordinal)},
                     {"context", perf_context}});

                /*
                 * These graph nodes are immutable no-ops on this participant.
                 * Mark them complete only after the sibling's active wave has
                 * both launched and left capture, which is their declarative
                 * position in the shared execution schedule.
                 */
                for (const auto &stage_name :
                     passive_wave.declarative_noop_stage_names)
                {
                    graph.markCompleted(stage_name);
                }
            }
            return true;
        }

        void addContextTag(PerfStatsCollector::Tags &tags, const std::string &perf_context)
        {
            if (!perf_context.empty())
                tags.emplace("context", perf_context);
        }

        /**
         * @brief Attach immutable cache-key geometry to one replay metric.
         *
         * The descriptor belongs to the same GraphSegmentCache as the timing
         * event ring.  This is essential for deferred collection: by the time a
         * stop event is queried, another MTP transaction may already be active
         * on the host.  No runner state is sampled here.
         */
        void addReplayWorkloadTags(
            PerfStatsCollector::Tags &tags,
            const DeviceGraphExecutor::GraphSegmentCache::ReplayWorkloadGeometry &workload)
        {
            if (!workload.valid())
                return;

            tags.emplace("seq_len", std::to_string(workload.seq_len));
            tags.emplace("batch_size", std::to_string(workload.batch_size));
            tags.emplace("m", std::to_string(workload.m));
            tags.emplace(
                "all_position_rows",
                std::to_string(workload.all_position_rows));
            tags.emplace(
                "verifier_outcome_mode",
                std::to_string(workload.verifier_outcome_mode));
            tags.emplace(
                "position_policy",
                std::to_string(workload.position_policy));
            tags.emplace(
                "moe_placement_epoch",
                std::to_string(workload.moe_placement_epoch));
            tags.emplace("decode", workload.decode ? "true" : "false");
            tags.emplace(
                "all_position_logits",
                workload.all_position_logits ? "true" : "false");
            tags.emplace(
                "live_mtp_request_batch_condition",
                workload.live_mtp_request_batch_condition ? "true" : "false");
        }

        PerfStatsCollector::Tags replaySegmentTags(const DeviceGraphExecutor::GraphSegment &segment,
                                                   const std::string &perf_context,
                                                   const DeviceGraphExecutor::GraphSegmentCache::ReplayWorkloadGeometry *workload = nullptr)
        {
            PerfStatsCollector::Tags tags{
                {"type", segmentTypeName(segment)},
                {"stage_count", std::to_string(segment.stage_names.size())}};
            if (!segment.stage_names.empty())
            {
                tags.emplace("first_stage", segment.stage_names.front());
                tags.emplace("last_stage", segment.stage_names.back());
            }
            addContextTag(tags, perf_context);
            if (workload)
                addReplayWorkloadTags(tags, *workload);
            return tags;
        }

        PerfStatsCollector::Tags replayCacheTags(const DeviceGraphExecutor::GraphSegmentCache &cache)
        {
            size_t stage_count = 0;
            bool has_capturable = false;
            bool has_manual = false;
            for (const auto &segment : cache.segments)
            {
                stage_count += segment.stage_names.size();
                has_capturable = has_capturable || segment.capturable;
                has_manual = has_manual || !segment.capturable;
            }

            const char *type = "empty";
            if (has_capturable && has_manual)
                type = "mixed";
            else if (has_capturable)
                type = "capturable";
            else if (has_manual)
                type = "manual";

            PerfStatsCollector::Tags tags{
                {"type", type},
                {"segment_count", std::to_string(cache.segments.size())},
                {"stage_count", std::to_string(stage_count)}};
            addContextTag(tags, cache.perf_context);
            addReplayWorkloadTags(tags, cache.replay_workload);
            return tags;
        }

        PerfStatsCollector::Tags replayCacheTags(const DeviceGraphExecutor::GraphSegmentCache &cache,
                                                 GraphReplayCaptureMode mode)
        {
            auto tags = replayCacheTags(cache);
            if (mode == GraphReplayCaptureMode::FullGraph)
            {
                tags.erase("segment_count");
                tags.emplace("graph_count", std::to_string(cache.segments.size()));
            }
            return tags;
        }

        PerfStatsCollector::Tags graphReplayHostTimingTags(PerfStatsCollector::Tags tags,
                                                           const char *timing_scope,
                                                           GraphReplayCaptureMode mode)
        {
            tags.emplace("attribution", "host_wall");
            tags.emplace("source", captureModeSource(mode));
            tags.emplace("graph_capture_scope", captureModeHostScope(mode));
            if (timing_scope && timing_scope[0] != '\0')
            {
                tags.emplace("timing_scope", timing_scope);
            }
            return tags;
        }

        PerfStatsCollector::Tags graphReplayGpuEventTags(PerfStatsCollector::Tags tags,
                                                         const char *timing_scope,
                                                         GraphReplayCaptureMode mode)
        {
            tags.emplace("attribution", "gpu_event");
            tags.emplace("source", captureModeSource(mode));
            tags.emplace("graph_capture_scope", captureModeEventScope(mode));
            if (timing_scope && timing_scope[0] != '\0')
            {
                tags.emplace("timing_scope", timing_scope);
            }
            return tags;
        }

        PerfStatsCollector::Tags graphReplayMetadataTags(PerfStatsCollector::Tags tags,
                                                         const std::string &perf_context,
                                                         GraphReplayCaptureMode mode)
        {
            tags.emplace("attribution", "graph_replay_metadata");
            tags.emplace("source", captureModeSource(mode));
            tags.emplace("graph_capture_scope", captureModePlanScope(mode));
            addContextTag(tags, perf_context);
            return tags;
        }

        constexpr size_t kInvalidReplayGpuTimingSlot =
            std::numeric_limits<size_t>::max();

        /*
         * Sixteen independent intervals provide a useful sample of a backlogged
         * replay burst without turning diagnostics into an event-per-token memory
         * commitment. When all slots are in flight, replay remains untouched and
         * the cache increments an explicit unsampled count. This is a bounded
         * profiler sampling policy, never a growth or synchronization policy.
         */
        constexpr size_t kDeferredReplayGpuTimingSlotCapacity = 16;

        void emitReplayGpuTiming(
            DeviceGraphExecutor::GraphSegmentCache &segment_cache,
            const DeviceGraphExecutor::GraphSegmentCache::ReplayGpuTimingSlot &slot,
            float elapsed_ms)
        {
            const GraphReplayCaptureMode replay_mode =
                captureModeForCache(segment_cache);
            const uint64_t elapsed_ns = static_cast<uint64_t>(
                static_cast<double>(std::max(0.0f, elapsed_ms)) * 1.0e6);
            const char *sync_scope = slot.deferred_final_fence
                                         ? "asynchronous_event_reclaimed"
                                         : "stream_synchronized";

            if (slot.scope ==
                DeviceGraphExecutor::GraphSegmentCache::ReplayGpuTimingSlot::Scope::TotalReplay)
            {
                auto total_tags = replayCacheTags(segment_cache, replay_mode);
                total_tags.emplace("sync_scope", sync_scope);
                PerfStatsCollector::recordTimingNs(
                    "stage_gpu",
                    "graph_replay.total",
                    elapsed_ns,
                    "decode",
                    segment_cache.replay_gpu_timing_device_name,
                    graphReplayGpuEventTags(
                        std::move(total_tags),
                        "total_replay_gpu_event",
                        replay_mode));

                /*
                 * A monolithic replay has one graph unit, so its aggregate and
                 * unit interval are exactly the same pair of stream events. Do
                 * not enqueue a duplicate pair merely to emit the second view.
                 */
                if (replay_mode == GraphReplayCaptureMode::FullGraph &&
                    segment_cache.segments.size() == 1)
                {
                    auto unit_tags = replaySegmentTags(
                        segment_cache.segments.front(),
                        segment_cache.perf_context,
                        &segment_cache.replay_workload);
                    unit_tags.emplace("graph_index", "0");
                    unit_tags.emplace("sync_scope", sync_scope);
                    PerfStatsCollector::recordTimingNs(
                        "stage_gpu",
                        replayUnitGpuEventName(replay_mode),
                        elapsed_ns,
                        "decode",
                        segment_cache.replay_gpu_timing_device_name,
                        graphReplayGpuEventTags(
                            std::move(unit_tags),
                            replayUnitTimingScope(replay_mode),
                            replay_mode));
                }
                return;
            }

            if (slot.replay_unit_index >= segment_cache.segments.size())
            {
                throw std::logic_error(
                    "Replay GPU timing slot refers to a missing replay unit");
            }

            auto unit_tags = replaySegmentTags(
                segment_cache.segments[slot.replay_unit_index],
                segment_cache.perf_context,
                &segment_cache.replay_workload);
            unit_tags.emplace(
                replayUnitIndexTagName(replay_mode),
                std::to_string(slot.replay_unit_index));
            unit_tags.emplace("sync_scope", sync_scope);
            PerfStatsCollector::recordTimingNs(
                "stage_gpu",
                replayUnitGpuEventName(replay_mode),
                elapsed_ns,
                "decode",
                segment_cache.replay_gpu_timing_device_name,
                graphReplayGpuEventTags(
                    std::move(unit_tags),
                    replayUnitTimingScope(replay_mode),
                    replay_mode));
        }

        bool reclaimReplayGpuTimingNonblockingImpl(
            DeviceGraphExecutor::GraphSegmentCache &segment_cache,
            IWorkerGPUContext *gpu_ctx)
        {
            bool reclaimed_any = false;
            for (auto &slot : segment_cache.replay_gpu_timing_slots)
            {
                if (!slot.pending)
                    continue;

                bool ready = false;
                if (!gpu_ctx->queryEventChecked(slot.stop_event, ready))
                {
                    LOG_ERROR(
                        "[DeviceGraphCaptureController] Failed to query a "
                        "pending replay GPU timing event");
                    return false;
                }
                if (!ready)
                    continue;

                const float elapsed_ms = gpu_ctx->eventElapsedTime(
                    slot.start_event,
                    slot.stop_event);
                if (elapsed_ms < 0.0f)
                {
                    LOG_ERROR(
                        "[DeviceGraphCaptureController] Failed to read a "
                        "completed replay GPU timing interval");
                    return false;
                }

                emitReplayGpuTiming(segment_cache, slot, elapsed_ms);
                slot.started = false;
                slot.pending = false;
                reclaimed_any = true;
            }

            if (reclaimed_any &&
                segment_cache.replay_gpu_timing_busy_samples > 0)
            {
                PerfStatsCollector::addCounter(
                    "forward_graph",
                    "replay_gpu_timing_busy_samples",
                    static_cast<double>(
                        segment_cache.replay_gpu_timing_busy_samples),
                    "decode",
                    segment_cache.replay_gpu_timing_device_name,
                    {{"context", segment_cache.perf_context},
                     {"sampling_policy", "bounded_nonblocking"}});
                segment_cache.replay_gpu_timing_busy_samples = 0;
            }
            return true;
        }

        bool beginReplayGpuTiming(
            DeviceGraphExecutor::GraphSegmentCache &segment_cache,
            IWorkerGPUContext *gpu_ctx,
            DeviceGraphExecutor::GraphSegmentCache::ReplayGpuTimingSlot::Scope scope,
            size_t replay_unit_index,
            bool deferred_final_fence,
            size_t &out_slot_index)
        {
            out_slot_index = kInvalidReplayGpuTimingSlot;
            if (!PerfStatsCollector::gpuStageEventTimingEnabled())
                return true;
            if (!gpu_ctx || !segment_cache.capture_stream ||
                segment_cache.replay_gpu_timing_slots.empty())
            {
                LOG_ERROR(
                    "[DeviceGraphCaptureController] GPU replay timing was "
                    "selected without its capture-owned event ring");
                return false;
            }
            if (!reclaimReplayGpuTimingNonblockingImpl(segment_cache, gpu_ctx))
                return false;

            for (size_t index = 0;
                 index < segment_cache.replay_gpu_timing_slots.size();
                 ++index)
            {
                auto &slot = segment_cache.replay_gpu_timing_slots[index];
                if (slot.started || slot.pending)
                    continue;

                slot.scope = scope;
                slot.replay_unit_index = replay_unit_index;
                slot.deferred_final_fence = deferred_final_fence;
                if (!gpu_ctx->recordEventChecked(
                        slot.start_event,
                        segment_cache.capture_stream))
                {
                    LOG_ERROR(
                        "[DeviceGraphCaptureController] Failed to publish a "
                        "replay GPU timing start event");
                    return false;
                }
                slot.started = true;
                out_slot_index = index;
                return true;
            }

            ++segment_cache.replay_gpu_timing_busy_samples;
            return true;
        }

        bool finishReplayGpuTiming(
            DeviceGraphExecutor::GraphSegmentCache &segment_cache,
            IWorkerGPUContext *gpu_ctx,
            size_t slot_index)
        {
            if (slot_index == kInvalidReplayGpuTimingSlot)
                return true;
            if (!gpu_ctx || slot_index >= segment_cache.replay_gpu_timing_slots.size())
                return false;

            auto &slot = segment_cache.replay_gpu_timing_slots[slot_index];
            if (!slot.started || slot.pending)
                return false;
            if (!gpu_ctx->recordEventChecked(
                    slot.stop_event,
                    segment_cache.capture_stream))
            {
                LOG_ERROR(
                    "[DeviceGraphCaptureController] Failed to publish a replay "
                    "GPU timing stop event");
                return false;
            }
            slot.pending = true;
            return true;
        }
    }


    const char *DeviceGraphCaptureController::phaseName(Phase phase)
    {
        switch (phase)
        {
        case Phase::Capture:
            return "capture";
        case Phase::Replay:
            return "replay";
        }

        throw std::logic_error("Unknown GPU graph replay phase");
    }

    bool DeviceGraphCaptureController::publishKernelInventory(
        const IGPUGraphCapture &capture,
        std::string_view fragment,
        std::optional<int> transaction_depth,
        DeviceId device,
        std::string &error)
    {
        if (!debugEnv().runtime_debug.gpu_graph_kernel_inventory)
            return true;

        const std::string fragment_name(fragment);
        const std::string depth_name = transaction_depth
                                           ? std::to_string(*transaction_depth)
                                           : "not_applicable";
        std::vector<GPUGraphKernelNodeInfo> kernel_nodes;
        std::string inspection_error;
        if (!capture.inspectKernelNodes(kernel_nodes, &inspection_error))
        {
            error = "kernel inventory failed for fragment '" +
                    fragment_name + "': " + inspection_error;
            return false;
        }

        size_t unresolved_names = 0u;
        for (const GPUGraphKernelNodeInfo &node : kernel_nodes)
        {
            if (!node.valid())
            {
                error =
                    "kernel inventory returned invalid launch geometry for fragment '" +
                    fragment_name + "' at " + node.graph_path;
                return false;
            }
            unresolved_names += node.name_resolved ? 0u : 1u;

            const std::string grid =
                std::to_string(node.grid_x) + "x" +
                std::to_string(node.grid_y) + "x" +
                std::to_string(node.grid_z);
            const std::string block =
                std::to_string(node.block_x) + "x" +
                std::to_string(node.block_y) + "x" +
                std::to_string(node.block_z);
            PerfStatsCollector::Tags tags{
                {"backend", capture.backendName()},
                {"fragment", fragment_name},
                {"transaction_depth", depth_name},
                {"kernel", node.name},
                {"grid", grid},
                {"block", block},
                {"dynamic_smem_bytes",
                 std::to_string(node.dynamic_shared_memory_bytes)},
                {"static_smem_bytes",
                 std::to_string(node.static_shared_memory_bytes)},
                {"local_bytes_per_thread",
                 std::to_string(node.local_memory_bytes_per_thread)},
                {"registers_per_thread",
                 std::to_string(node.registers_per_thread)},
                {"max_threads_per_block",
                 std::to_string(node.max_threads_per_block)},
                {"max_active_blocks_per_sm",
                 std::to_string(node.max_active_blocks_per_sm)},
                {"nesting_depth", std::to_string(node.nesting_depth)},
                {"name_resolved", node.name_resolved ? "true" : "false"},
            };
            if (!node.name_resolved)
            {
                tags.emplace(
                    "function_identity",
                    std::to_string(node.function_identity));
            }
            PerfStatsCollector::addCounter(
                "gpu_graph_inventory",
                "kernel_nodes",
                1.0,
                "graph_setup",
                device.toString(),
                std::move(tags));
        }

        const PerfStatsCollector::Tags summary_tags{
            {"backend", capture.backendName()},
            {"fragment", fragment_name},
            {"transaction_depth", depth_name},
        };
        auto kernel_summary_tags = summary_tags;
        kernel_summary_tags.emplace(
            "unresolved_names", std::to_string(unresolved_names));
        PerfStatsCollector::addCounter(
            "gpu_graph_inventory",
            "fragment_kernel_nodes",
            static_cast<double>(kernel_nodes.size()),
            "graph_setup",
            device.toString(),
            std::move(kernel_summary_tags));
        PerfStatsCollector::addCounter(
            "gpu_graph_inventory",
            "fragment_native_nodes",
            static_cast<double>(capture.nodeCount()),
            "graph_setup",
            device.toString(),
            summary_tags);
        return true;
    }

    DeviceGraphCaptureController::Transition DeviceGraphCaptureController::beginStep(
        bool initialized,
        bool &needs_capture,
        uint64_t &decode_step)
    {
        // Cached graph replay uses a monotonic step so we can reason about which
        // segments were executed in each phase.
        ++decode_step;

        if (!initialized)
        {
            return {Phase::Capture, decode_step};
        }

        if (needs_capture)
        {
            throw std::logic_error(
                "GPU graph cache exposed initialized-without-executable state");
        }

        return {Phase::Replay, decode_step};
    }

    void DeviceGraphCaptureController::prepareDeviceForGraphCapture(IDeviceContext *ctx)
    {
        if (ctx)
            ctx->activateDevice();
    }

    uint64_t DeviceGraphCaptureController::computeCaptureVariantSignature(ComputeGraph &graph)
    {
        constexpr uint64_t kFnvOffset = 1469598103934665603ull;
        constexpr uint64_t kFnvPrime = 1099511628211ull;

        uint64_t h = kFnvOffset;
        bool has_variant = false;
        const auto &order = graph.getExecutionOrder();
        for (const auto &name : order)
        {
            auto *node = graph.getNode(name);
            if (!node || !node->stage)
            {
                continue;
            }

            const uint64_t stage_variant = node->stage->graphCaptureVariantSignature();
            if (stage_variant == 0)
            {
                continue;
            }

            has_variant = true;
            const uint64_t name_hash = static_cast<uint64_t>(std::hash<std::string>{}(name));
            const uint64_t type_hash = static_cast<uint64_t>(static_cast<int>(node->stage->type()));
            for (uint64_t part : {name_hash, type_hash, stage_variant})
            {
                h ^= part;
                h *= kFnvPrime;
            }
        }

        return has_variant ? h : 0;
    }

    const char *DeviceGraphCaptureController::replayModeName(
        const DeviceGraphExecutor::GraphSegmentCache &segment_cache)
    {
        return captureModeTag(captureModeForCache(segment_cache));
    }

    bool DeviceGraphCaptureController::constrainReplayPolicyToNativeEnvelope(
        GraphNativeCaptureEnvelope envelope,
        DeviceGraphExecutor::DecodeCapturePolicy &policy,
        std::string *error)
    {
        if (error)
            error->clear();

        const auto reject = [&](std::string reason)
        {
            if (error)
                *error = std::move(reason);
            return false;
        };

        switch (envelope)
        {
        case GraphNativeCaptureEnvelope::Ordinary:
            return true;
        case GraphNativeCaptureEnvelope::DeviceOwnedTimelineTransaction:
            if (!policy.allow_cached_graph_replay)
            {
                return reject(
                    "device-owned timeline requires mandatory native graph replay");
            }
            policy.graph_replay_plan_policy =
                DeviceGraphExecutor::GraphReplayPlanPolicy::RequireFullGraph;
            policy.retained_parent_composer = {};
            policy.defer_final_sync = true;
            return true;
        case GraphNativeCaptureEnvelope::HeterogeneousTicketTransaction:
            if (!policy.allow_cached_graph_replay ||
                !policy.heterogeneous_segmented_enabled)
            {
                return reject(
                    "heterogeneous ticket transaction requires topology-admitted segmented native replay");
            }
            policy.graph_replay_plan_policy =
                DeviceGraphExecutor::GraphReplayPlanPolicy::
                    AllowHeterogeneousBoundarySegmentation;
            policy.retained_parent_composer = {};
            policy.defer_final_sync = true;
            return true;
        }

        return reject("unknown graph-native capture envelope");
    }

    bool DeviceGraphCaptureController::prepareReplayGpuTiming(
        DeviceGraphExecutor::GraphSegmentCache &segment_cache,
        IWorkerGPUContext *gpu_ctx,
        const std::string &device_name)
    {
        if (!PerfStatsCollector::gpuStageEventTimingEnabled())
            return true;
        if (!gpu_ctx || !segment_cache.capture_stream ||
            segment_cache.segments.empty())
        {
            LOG_ERROR(
                "[DeviceGraphCaptureController] Replay GPU timing preparation "
                "requires a complete replay plan, context, and capture stream");
            return false;
        }

        if (!segment_cache.replay_gpu_timing_slots.empty())
        {
            if (segment_cache.replay_gpu_timing_device_name != device_name)
            {
                LOG_ERROR(
                    "[DeviceGraphCaptureController] Replay GPU timing ring was "
                    "prepared for a different device label");
                return false;
            }
            return true;
        }

        const GraphReplayCaptureMode replay_mode =
            captureModeForCache(segment_cache);
        const size_t simultaneous_intervals =
            replay_mode == GraphReplayCaptureMode::FullGraph
                ? 1
                : segment_cache.segments.size() + 1;
        const size_t slot_capacity = std::max(
            simultaneous_intervals,
            replay_mode == GraphReplayCaptureMode::FullGraph
                ? kDeferredReplayGpuTimingSlotCapacity
                : simultaneous_intervals);

        segment_cache.replay_gpu_timing_device_name = device_name;
        segment_cache.replay_gpu_timing_slots.resize(slot_capacity);
        for (auto &slot : segment_cache.replay_gpu_timing_slots)
        {
            slot.start_event = gpu_ctx->createEvent();
            slot.stop_event = gpu_ctx->createEvent();
            if (!slot.start_event || !slot.stop_event)
            {
                LOG_ERROR(
                    "[DeviceGraphCaptureController] Failed to preallocate the "
                    "replay GPU timing event ring");
                segment_cache.destroyReplayGpuTimingEvents();
                return false;
            }
        }

        PerfStatsCollector::addCounter(
            "forward_graph",
            "replay_timing_event_slots_preallocated",
            static_cast<double>(slot_capacity),
            "capture",
            device_name,
            {{"context", segment_cache.perf_context},
             {"replay_mode", captureModeTag(replay_mode)}});
        return true;
    }

    bool DeviceGraphCaptureController::reclaimReplayGpuTimingNonblocking(
        DeviceGraphExecutor::GraphSegmentCache &segment_cache,
        IWorkerGPUContext *gpu_ctx)
    {
        if (!PerfStatsCollector::gpuStageEventTimingEnabled() ||
            segment_cache.replay_gpu_timing_slots.empty())
        {
            return true;
        }
        if (!gpu_ctx)
            return false;
        return reclaimReplayGpuTimingNonblockingImpl(segment_cache, gpu_ctx);
    }

    void DeviceGraphCaptureController::buildCapturePlan(
        ComputeGraph &graph,
        DeviceGraphExecutor::GraphSegmentCache &segment_cache,
        const std::unordered_set<std::string> *collective_nodes,
        bool has_collective_nodes,
        bool collectives_graph_capturable,
        DeviceGraphExecutor::GraphReplayPlanPolicy plan_policy)
    {
        segment_cache.segments.clear();
        segment_cache.graph_replay_plan_policy = plan_policy;

        const auto &order = graph.getExecutionOrder();
        const GraphNativeCaptureEnvelope native_capture_envelope =
            graph.nativeCaptureEnvelope();
        const bool device_owned_timeline_transaction =
            requiresSingleNativeExecutable(native_capture_envelope);
        const bool heterogeneous_ticket_transaction =
            requiresHeterogeneousTicketSegmentation(
                native_capture_envelope);
        if (device_owned_timeline_transaction &&
            plan_policy !=
                DeviceGraphExecutor::GraphReplayPlanPolicy::RequireFullGraph)
        {
            throw std::logic_error(
                "A device-owned timeline transaction requires the full-graph replay policy");
        }
        if (heterogeneous_ticket_transaction &&
            plan_policy !=
                DeviceGraphExecutor::GraphReplayPlanPolicy::
                    AllowHeterogeneousBoundarySegmentation)
        {
            throw std::logic_error(
                "A heterogeneous ticket transaction requires the explicit heterogeneous segmentation policy");
        }
        const auto &segmented_collective_capture_allow =
            debugEnv().execution.gpu_graph_collective_segmented_capture_allow;

        auto stage_in_collective_allowlist = [&](const std::string &stage_name) -> bool
        {
            if (segmented_collective_capture_allow.empty())
            {
                return false;
            }
            for (const auto &needle : segmented_collective_capture_allow)
            {
                if (stage_name.find(needle) != std::string::npos)
                {
                    return true;
                }
            }
            return false;
        };

        bool current_capturable = false;
        bool first = true;
        bool force_new_segment = false;

        for (const auto &name : order)
        {
            auto *node = graph.getNode(name);
            if (!node || !node->stage)
            {
                continue;
            }

            const auto *const wave_contract =
                native_capture_envelope ==
                            GraphNativeCaptureEnvelope::Ordinary &&
                        node->graph_capture_wave
                    ? &*node->graph_capture_wave
                    : nullptr;
            if (wave_contract &&
                wave_contract->participation ==
                    GraphCaptureWaveParticipation::Passive)
            {
                if (!node->stage->isPassiveGraphCaptureNoOp())
                {
                    throw std::logic_error(
                        "Graph node '" + name + "' declares passive capture "
                        "participation, but its stage no longer certifies an "
                        "immutable no-op role");
                }
                if (first || segment_cache.segments.empty() ||
                    segment_cache.segments.back().stage_names.empty())
                {
                    throw std::logic_error(
                        "Passive capture node '" + name + "' has no preceding "
                        "execution segment to anchor its rendezvous");
                }

                auto &anchor = segment_cache.segments.back();
                const auto append_passive_wave =
                    [&](const std::string &identity,
                        const std::string *declarative_noop_stage)
                {
                    const auto duplicate = std::find_if(
                        anchor.passive_capture_waves_after.begin(),
                        anchor.passive_capture_waves_after.end(),
                        [&](const DeviceGraphExecutor::GraphSegment::
                                PassiveCaptureWave &wave)
                        {
                            return wave.identity == identity;
                        });
                    if (duplicate !=
                        anchor.passive_capture_waves_after.end())
                    {
                        throw std::logic_error(
                            "Captured segment repeats passive wave identity '" +
                            identity + "'");
                    }

                    DeviceGraphExecutor::GraphSegment::PassiveCaptureWave wave{
                        .ordinal = 0,
                        .identity = identity,
                    };
                    if (declarative_noop_stage)
                    {
                        wave.declarative_noop_stage_names.push_back(
                            *declarative_noop_stage);
                    }
                    anchor.passive_capture_waves_after.push_back(
                        std::move(wave));
                };

                append_passive_wave(wave_contract->identity, &name);
                for (const auto &following_identity :
                     wave_contract->passive_following_identities)
                {
                    append_passive_wave(following_identity, nullptr);
                }

                /*
                 * A passive wave is an intervening domain lifecycle boundary.
                 * The next local device operation must start a new executable;
                 * fusing it into the anchor would move that work ahead of the
                 * sibling wave this participant just joined.
                 */
                force_new_segment = true;
                continue;
            }

            // Start from stage capability, then layer on safety gates.
            bool stage_capturable = node->stage->isGraphCapturable();
            const bool preparation_dependent_capture =
                !stage_capturable && node->stage->supportsGraphCaptureAfterLaunchPreparation();
            if (preparation_dependent_capture)
            {
                stage_capturable = true;
            }
            const bool stage_capture_capable = stage_capturable;
            const bool collective_by_stage = node->stage->isCollectiveStage();
            const bool collective_by_name = (collective_nodes && collective_nodes->count(name));

            /*
             * A collective belongs in the graph only when both contracts hold:
             * the concrete stage instance is capture-ready and the LocalTP
             * backend supports stream-ordered graph-captured collectives.
             * Backend capability must never override a stage-level rejection
             * caused by invalid geometry, missing workspace, or stale state.
             */
            if (collective_by_stage || collective_by_name)
            {
                stage_capturable =
                    stage_capturable && collectives_graph_capturable;
            }
            const bool capture_wave_dormant_by_collective_policy =
                wave_contract &&
                wave_contract->participation ==
                    GraphCaptureWaveParticipation::Active &&
                stage_capture_capable &&
                (collective_by_stage || collective_by_name) &&
                !collectives_graph_capturable;

            if (has_collective_nodes && !segmented_collective_capture_allow.empty())
            {
                // Explicit allowlist mode: only allowlisted stages are capturable
                stage_capturable = stage_in_collective_allowlist(name);
            }
            if (node->stage->requiresHostGraphTicketFence())
            {
                if (!node->stage->isManualGraphBoundary())
                {
                    throw std::logic_error(
                        "Stage '" + name +
                        "' requests a host graph-ticket fence without declaring a manual graph boundary");
                }
                if (stage_capturable)
                {
                    throw std::logic_error(
                        "Stage '" + name +
                        "' requests a host graph-ticket fence but was classified as capturable");
                }
            }
            // Otherwise: trust each stage's isGraphCapturable() declaration.
            // Stages that need per-step updates either return false or report
            // segment boundaries around themselves when they can be captured
            // alone but cannot safely be fused with adjacent captured work.
            // A collective remains inside the surrounding captured segment
            // when both capture contracts pass. Only an explicitly enabled
            // compatibility policy may leave it as a manual segment.
            const bool boundary_before =
                stage_capturable &&
                node->stage->requiresGraphCaptureSegmentBoundaryBefore() &&
                !segment_cache.segments.empty() &&
                !segment_cache.segments.back().stage_names.empty();
            force_new_segment = force_new_segment || boundary_before;

            /*
             * An explicit identity is a logical capture-unit boundary. Two
             * adjacent captured nodes with different identities cannot share a
             * native executable because sibling graphs may participate in only
             * one of those waves. An unannotated prefix may join the first
             * explicit identity in its segment; that is how the continuation
             * compute and root-only ticket publication form one active wave.
             */
            if (stage_capturable && wave_contract &&
                !segment_cache.segments.empty() &&
                segment_cache.segments.back().capturable &&
                !segment_cache.segments.back().stage_names.empty() &&
                !segment_cache.segments.back().capture_wave_identity.empty() &&
                segment_cache.segments.back().capture_wave_identity !=
                    wave_contract->identity)
            {
                force_new_segment = true;
            }

            if (first || force_new_segment || stage_capturable != current_capturable)
            {
                // Create a new segment whenever capturable/manual mode changes
                // so each segment has uniform execution semantics.
                segment_cache.segments.emplace_back();
                segment_cache.segments.back().capturable = stage_capturable;
                current_capturable = stage_capturable;
                first = false;
                force_new_segment = false;
            }

            segment_cache.segments.back().stage_names.push_back(name);
            if (wave_contract)
            {
                if (!stage_capturable)
                {
                    /*
                     * An active collective wave can be declared once in the
                     * model graph even when an explicit diagnostic policy runs
                     * collectives as manual heterogeneous segments.  In that
                     * mode the wave is dormant: the manual collective remains
                     * in the plan and consumes no native-capture ordinal. A
                     * stage that is intrinsically incapable of capture still
                     * fails, so this cannot hide stale preparation or geometry.
                     */
                    if (capture_wave_dormant_by_collective_policy)
                    {
                        force_new_segment = false;
                        continue;
                    }
                    throw std::logic_error(
                        "Graph node '" + name + "' declares capture wave '" +
                        wave_contract->identity +
                        "' but is not graph-capturable");
                }

                auto &segment = segment_cache.segments.back();
                if (segment.capture_wave_identity.empty())
                {
                    segment.capture_wave_identity =
                        wave_contract->identity;
                }
                else if (segment.capture_wave_identity !=
                         wave_contract->identity)
                {
                    throw std::logic_error(
                        "Captured segment combines incompatible wave identities '" +
                        segment.capture_wave_identity + "' and '" +
                        wave_contract->identity + "'");
                }

                for (const auto &passive_identity :
                     wave_contract->passive_following_identities)
                {
                    const auto duplicate = std::find_if(
                        segment.passive_capture_waves_after.begin(),
                        segment.passive_capture_waves_after.end(),
                        [&](const DeviceGraphExecutor::GraphSegment::PassiveCaptureWave &wave)
                        {
                            return wave.identity == passive_identity;
                        });
                    if (duplicate != segment.passive_capture_waves_after.end())
                    {
                        throw std::logic_error(
                            "Captured segment repeats passive wave identity '" +
                            passive_identity + "'");
                    }
                    segment.passive_capture_waves_after.push_back(
                        {.ordinal = 0, .identity = passive_identity});
                }
            }
            force_new_segment =
                stage_capturable &&
                node->stage->requiresGraphCaptureSegmentBoundaryAfter();
        }

        const int max_stages = debugEnv().execution.gpu_graph_max_stages;
        if (max_stages > 0)
        {
            std::vector<DeviceGraphExecutor::GraphSegment> split_segments;
            for (auto &seg : segment_cache.segments)
            {
                if (seg.capturable && static_cast<int>(seg.stage_names.size()) > max_stages)
                {
                    if (device_owned_timeline_transaction)
                    {
                        throw std::runtime_error(
                            "gpu_graph_max_stages cannot split a device-owned timeline transaction");
                    }
                    if (!seg.capture_wave_identity.empty() ||
                        !seg.passive_capture_waves_after.empty())
                    {
                        throw std::runtime_error(
                            "gpu_graph_max_stages would split explicit capture wave '" +
                            (seg.capture_wave_identity.empty()
                                 ? std::string("<implicit>")
                                 : seg.capture_wave_identity) +
                            "'; synchronized role-asymmetric LocalTP waves may not be split independently");
                    }
                    for (size_t i = 0; i < seg.stage_names.size(); i += max_stages)
                    {
                        DeviceGraphExecutor::GraphSegment sub;
                        sub.capturable = true;
                        size_t end = std::min(i + static_cast<size_t>(max_stages), seg.stage_names.size());
                        for (size_t j = i; j < end; j++)
                        {
                            sub.stage_names.push_back(seg.stage_names[j]);
                        }
                        split_segments.push_back(std::move(sub));
                    }
                }
                else
                {
                    split_segments.push_back(std::move(seg));
                }
            }
            segment_cache.segments = std::move(split_segments);
        }

        if (plan_policy ==
            DeviceGraphExecutor::GraphReplayPlanPolicy::
                RequireRetainedParentComposition)
        {
            /*
             * Child captures share stable arena storage but do not execute
             * while they are recorded. A read crossing a child boundary is
             * therefore neither globally authoritative nor graph-internal to
             * the child. Freeze the exact BufferId proof here, while the whole
             * retained-parent topology is still visible. The composer later
             * clones these children in this same order and supplies the actual
             * device-side producer/consumer edge.
             *
             * Writes become eligible only after the complete current segment,
             * so an ordinary producer within this child remains an internal
             * dependency rather than masquerading as a parent import.
             */
            std::unordered_set<BufferId> prior_child_writes;
            for (auto &segment : segment_cache.segments)
            {
                if (!segment.capturable)
                {
                    throw std::runtime_error(
                        "Retained parent composition forbids manual replay units");
                }

                std::unordered_set<BufferId> segment_parent_inputs;
                std::unordered_set<BufferId> segment_writes;
                for (const std::string &stage_name : segment.stage_names)
                {
                    const ComputeNode *const node = graph.getNode(stage_name);
                    if (!node || !node->stage)
                    {
                        throw std::runtime_error(
                            "Retained parent dependency planning cannot resolve stage '" +
                            stage_name + "'");
                    }

                    const StageBufferContract contract =
                        node->stage->bufferContract();
                    for (const auto &binding : contract.allArenaReads())
                    {
                        if (prior_child_writes.contains(binding.id))
                            segment_parent_inputs.insert(binding.id);
                    }
                    for (const auto &binding : contract.allWrites())
                        segment_writes.insert(binding.id);
                }

                segment.retained_parent_input_ids.assign(
                    segment_parent_inputs.begin(),
                    segment_parent_inputs.end());
                std::sort(
                    segment.retained_parent_input_ids.begin(),
                    segment.retained_parent_input_ids.end());
                prior_child_writes.insert(
                    segment_writes.begin(), segment_writes.end());
            }
        }

        /*
         * Ordinals count domain lifecycle waves, not local replay segments.
         * Manual sparse work exists only on the continuation root and therefore
         * must not perturb the number used by the strict LocalTP rendezvous.
         * A passive follower consumes one ordinal exactly as an active sibling
         * segment does, keeping every later common capture wave aligned.
         */
        size_t next_capture_wave_ordinal = 0;
        for (auto &segment : segment_cache.segments)
        {
            if (segment.capturable)
                segment.capture_wave_ordinal = next_capture_wave_ordinal++;
            for (auto &passive_wave : segment.passive_capture_waves_after)
                passive_wave.ordinal = next_capture_wave_ordinal++;
        }

        size_t host_ticket_boundary_segments = 0u;
        for (size_t segment_index = 0;
             segment_index < segment_cache.segments.size();
             ++segment_index)
        {
            std::string consumer_stage;
            if (!manualSegmentRequiresHostTicketFence(
                    graph,
                    segment_cache.segments[segment_index],
                    &consumer_stage))
            {
                continue;
            }
            if (segment_index == 0 ||
                !segment_cache.segments[segment_index - 1].capturable)
            {
                throw std::runtime_error(
                    "Manual host ticket consumer '" + consumer_stage +
                    "' is not immediately preceded by a captured producer segment");
            }
            ++host_ticket_boundary_segments;
        }

        size_t capturable_segments = 0, manual_segments = 0;
        size_t capturable_stages = 0, manual_stages = 0;
        size_t max_capturable_segment_stages = 0;
        size_t max_manual_segment_stages = 0;
        std::map<std::string, size_t> capturable_stage_types;
        std::map<std::string, size_t> manual_stage_types;
        for (const auto &seg : segment_cache.segments)
        {
            if (seg.capturable)
            {
                capturable_segments++;
                capturable_stages += seg.stage_names.size();
                max_capturable_segment_stages =
                    std::max(max_capturable_segment_stages, seg.stage_names.size());
            }
            else
            {
                manual_segments++;
                manual_stages += seg.stage_names.size();
                max_manual_segment_stages =
                    std::max(max_manual_segment_stages, seg.stage_names.size());
            }

            auto &stage_types = seg.capturable ? capturable_stage_types : manual_stage_types;
            for (const auto &stage_name : seg.stage_names)
            {
                auto *node = graph.getNode(stage_name);
                if (!node || !node->stage)
                {
                    continue;
                }
                stage_types[computeStageTypeName(node->stage->type())]++;
            }
        }

        if (device_owned_timeline_transaction &&
            (capturable_segments != 1u || manual_segments != 0u ||
             segment_cache.segments.size() != 1u ||
             !segment_cache.segments.front().passive_capture_waves_after.empty()))
        {
            std::ostringstream diagnostic;
            diagnostic
                << "Device-owned timeline capture did not lower to exactly "
                   "one native executable"
                << ": segments=" << segment_cache.segments.size()
                << " capturable_segments=" << capturable_segments
                << " manual_segments=" << manual_segments
                << " capturable_stages=" << capturable_stages
                << " manual_stages=" << manual_stages;
            for (std::size_t segment_index = 0u;
                 segment_index < segment_cache.segments.size();
                 ++segment_index)
            {
                const auto &segment =
                    segment_cache.segments[segment_index];
                diagnostic << " segment[" << segment_index
                           << "]={capturable="
                           << (segment.capturable ? "true" : "false")
                           << ",stages=" << segment.stage_names.size()
                           << ",passive_waves="
                           << segment.passive_capture_waves_after.size();
                if (!segment.capturable)
                {
                    diagnostic << ",manual_frontier=[";
                    bool first_stage = true;
                    for (const auto &stage_name : segment.stage_names)
                    {
                        auto *const node = graph.getNode(stage_name);
                        if (!first_stage)
                            diagnostic << ';';
                        first_stage = false;
                        diagnostic << stage_name;
                        if (node && node->stage)
                        {
                            diagnostic << "(type="
                                       << computeStageTypeName(
                                              node->stage->type())
                                       << ",readiness="
                                       << node->stage
                                              ->graphCaptureReadinessDebugString()
                                       << ')';
                        }
                    }
                    diagnostic << ']';
                }
                diagnostic << '}';
            }
            throw std::runtime_error(diagnostic.str());
        }

        if (heterogeneous_ticket_transaction &&
            (capturable_segments < 2u || manual_segments == 0u ||
             host_ticket_boundary_segments == 0u ||
             segment_cache.segments.empty() ||
             !segment_cache.segments.front().capturable ||
             !segment_cache.segments.back().capturable))
        {
            throw std::runtime_error(
                "Heterogeneous ticket capture must lower to captured producer and consumer units separated by an authenticated manual ticket boundary");
        }

        if (heterogeneous_ticket_transaction)
        {
            PerfStatsCollector::addCounter(
                "forward_graph",
                "heterogeneous_ticket_transactions",
                1.0,
                "capture",
                "",
                {{"capturable_segments",
                  std::to_string(capturable_segments)},
                 {"manual_segments", std::to_string(manual_segments)},
                 {"ticket_boundaries",
                  std::to_string(host_ticket_boundary_segments)}});
        }

        if (plan_policy ==
            DeviceGraphExecutor::GraphReplayPlanPolicy::
                RequireRetainedParentComposition)
        {
            if (manual_segments != 0u || capturable_segments == 0u)
            {
                throw std::runtime_error(
                    "Retained parent composition requires one or more native child graphs and forbids manual replay units");
            }
            for (const auto &segment : segment_cache.segments)
            {
                for (const std::string &stage_name : segment.stage_names)
                {
                    const ComputeNode *const node = graph.getNode(stage_name);
                    if (!node || !node->stage ||
                        node->stage->graphLaunchPreparationPolicy() ==
                            GraphLaunchPreparationPolicy::CaptureAndReplay)
                    {
                        throw std::runtime_error(
                            "Retained parent composition requires immutable replay metadata for stage '" +
                            stage_name + "'");
                    }
                }
            }
        }

        const GraphReplayCaptureMode plan_mode =
            captureModeForPlan(segment_cache.segments.size(), capturable_segments, manual_segments);

        if (plan_mode == GraphReplayCaptureMode::FullGraph)
        {
            LOG_DEBUG("[DeviceGraphExecutor] Full graph capture plan: "
                      << capturable_stages << " capturable stages, no manual segments");
        }
        else
        {
            LOG_DEBUG("[DeviceGraphExecutor] Segmented graph capture plan: "
                      << capturable_segments << " capturable segments ("
                      << capturable_stages << " stages) + "
                      << manual_segments << " manual segments ("
                      << manual_stages << " stages)");
        }

        if (PerfStatsCollector::isDomainEnabled("forward_graph"))
        {
            PerfStatsCollector::addCounter(
                "forward_graph",
                planUnitMetricName(plan_mode),
                static_cast<double>(segment_cache.segments.size()),
                "decode",
                "",
                {{"type", "total"}});
            PerfStatsCollector::addCounter(
                "forward_graph",
                planUnitMetricName(plan_mode),
                static_cast<double>(capturable_segments),
                "decode",
                "",
                {{"type", "capturable"}});
            PerfStatsCollector::addCounter(
                "forward_graph",
                planUnitMetricName(plan_mode),
                static_cast<double>(manual_segments),
                "decode",
                "",
                {{"type", "manual"}});
            PerfStatsCollector::addCounter(
                "forward_graph",
                planMetricName(plan_mode, "stages"),
                static_cast<double>(capturable_stages),
                "decode",
                "",
                {{"type", "capturable"}});
            PerfStatsCollector::addCounter(
                "forward_graph",
                planMetricName(plan_mode, "stages"),
                static_cast<double>(manual_stages),
                "decode",
                "",
                {{"type", "manual"}});
            PerfStatsCollector::addCounter(
                "forward_graph",
                planMaxUnitStagesMetricName(plan_mode),
                static_cast<double>(max_capturable_segment_stages),
                "decode",
                "",
                {{"type", "capturable"}});
            PerfStatsCollector::addCounter(
                "forward_graph",
                planMaxUnitStagesMetricName(plan_mode),
                static_cast<double>(max_manual_segment_stages),
                "decode",
                "",
                {{"type", "manual"}});

            for (const auto &[type_name, count] : capturable_stage_types)
            {
                PerfStatsCollector::addCounter(
                    "forward_graph",
                    planMetricName(plan_mode, "stage_types"),
                    static_cast<double>(count),
                    "decode",
                    "",
                    {{planUnitTypeTagName(plan_mode), "capturable"}, {"stage_type", type_name}});
            }
            for (const auto &[type_name, count] : manual_stage_types)
            {
                PerfStatsCollector::addCounter(
                    "forward_graph",
                    planMetricName(plan_mode, "stage_types"),
                    static_cast<double>(count),
                    "decode",
                    "",
                    {{planUnitTypeTagName(plan_mode), "manual"}, {"stage_type", type_name}});
            }

            auto record_stage_gpu_plan = [&](const char *name,
                                             double value,
                                             PerfStatsCollector::Tags tags)
            {
                addReplayWorkloadTags(
                    tags,
                    segment_cache.replay_workload);
                PerfStatsCollector::addCounter(
                    "stage_gpu",
                    name,
                    value,
                    "decode",
                    "",
                    graphReplayMetadataTags(std::move(tags), segment_cache.perf_context, plan_mode));
            };

            record_stage_gpu_plan(
                stageGpuPlanUnitMetricName(plan_mode),
                static_cast<double>(segment_cache.segments.size()),
                {{"type", "total"}});
            record_stage_gpu_plan(
                stageGpuPlanUnitMetricName(plan_mode),
                static_cast<double>(capturable_segments),
                {{"type", "capturable"}});
            record_stage_gpu_plan(
                stageGpuPlanUnitMetricName(plan_mode),
                static_cast<double>(manual_segments),
                {{"type", "manual"}});
            record_stage_gpu_plan(
                "graph_replay_plan_stages",
                static_cast<double>(capturable_stages),
                {{"type", "capturable"}});
            record_stage_gpu_plan(
                "graph_replay_plan_stages",
                static_cast<double>(manual_stages),
                {{"type", "manual"}});

            for (const auto &[type_name, count] : capturable_stage_types)
            {
                record_stage_gpu_plan(
                    "graph_replay_plan_stage_types",
                    static_cast<double>(count),
                    {{planUnitTypeTagName(plan_mode), "capturable"}, {"stage_type", type_name}});
            }
            for (const auto &[type_name, count] : manual_stage_types)
            {
                record_stage_gpu_plan(
                    "graph_replay_plan_stage_types",
                    static_cast<double>(count),
                    {{planUnitTypeTagName(plan_mode), "manual"}, {"stage_type", type_name}});
            }
        }

        if (plan_mode == GraphReplayCaptureMode::Segmented)
        {
            std::vector<std::string> undeclared_manual_boundaries;
            for (const auto &segment : segment_cache.segments)
            {
                if (segment.capturable)
                    continue;
                for (const auto &stage_name : segment.stage_names)
                {
                    const auto *node = graph.getNode(stage_name);
                    if (!node || !node->stage ||
                        (!node->stage->isCollectiveStage() &&
                         !node->stage->isManualGraphBoundary()))
                    {
                        undeclared_manual_boundaries.push_back(stage_name);
                    }
                }
            }

            const bool heterogeneous_boundary_segmentation_admitted =
                plan_policy ==
                    DeviceGraphExecutor::GraphReplayPlanPolicy::
                        AllowHeterogeneousBoundarySegmentation &&
                undeclared_manual_boundaries.empty();
            const bool retained_parent_composition_admitted =
                plan_policy ==
                    DeviceGraphExecutor::GraphReplayPlanPolicy::
                        RequireRetainedParentComposition &&
                manual_segments == 0u;

            if (!heterogeneous_boundary_segmentation_admitted &&
                !retained_parent_composition_admitted)
            {
                std::ostringstream detail;
                detail
                    << "GPU replay planning produced "
                    << capturable_segments << " capturable segment(s) and "
                    << manual_segments << " manual segment(s), but this "
                       "execution domain requires one fully captured graph";
                if (plan_policy ==
                    DeviceGraphExecutor::GraphReplayPlanPolicy::RequireFullGraph)
                {
                    detail
                        << "; the execution topology did not admit a "
                           "heterogeneous manual boundary";
                }
                else if (plan_policy ==
                         DeviceGraphExecutor::GraphReplayPlanPolicy::
                             RequireRetainedParentComposition)
                {
                    detail
                        << "; retained parent composition admits native child "
                           "graphs only and cannot contain a manual unit";
                }
                if (!undeclared_manual_boundaries.empty())
                {
                    detail << "; uncapturable stages without an explicit "
                              "collective/manual-boundary contract:";
                    for (const auto &stage_name : undeclared_manual_boundaries)
                        detail << ' ' << stage_name;
                }
                detail << ". Manual stage types:";
                if (manual_stage_types.empty())
                {
                    detail << " <none>";
                }
                else
                {
                    for (const auto &[type_name, count] : manual_stage_types)
                    {
                        detail << ' ' << type_name << '=' << count;
                    }
                }
                detail << ". Manual stages:";
                bool described_manual_stage = false;
                for (const auto &segment : segment_cache.segments)
                {
                    if (segment.capturable)
                        continue;

                    for (const auto &stage_name : segment.stage_names)
                    {
                        auto *node = graph.getNode(stage_name);
                        if (!node || !node->stage)
                        {
                            detail << " [" << stage_name << ": missing-node]";
                            described_manual_stage = true;
                            continue;
                        }

                        const std::string readiness =
                            node->stage->graphCaptureReadinessDebugString();
                        detail
                            << " [" << stage_name
                            << ": type="
                            << computeStageTypeName(node->stage->type())
                            << ", launch_preparation="
                            << (node->stage->supportsGraphCaptureAfterLaunchPreparation()
                                    ? "true"
                                    : "false");
                        if (!readiness.empty())
                            detail << ", " << readiness;
                        detail << ']';
                        described_manual_stage = true;
                    }
                }
                if (!described_manual_stage)
                    detail << " <none>";

                LOG_ERROR("[DeviceGraphCaptureController] " << detail.str());
                throw std::runtime_error(detail.str());
            }
        }

        for (auto &seg : segment_cache.segments)
        {
            seg.last_executed_step = 0;
        }
    }

    bool DeviceGraphCaptureController::executeStreamOnlyReplay(
        ComputeGraph &graph,
        DeviceGraphExecutor::GraphSegmentCache &segment_cache,
        IDeviceContext *ctx,
        IWorkerGPUContext *gpu_ctx,
        void *capture_stream,
        bool use_default_stream)
    {
        if (!ctx || !gpu_ctx)
        {
            LOG_ERROR("[DeviceGraphCaptureController] Stream-only replay missing context");
            return false;
        }

        // Always use an explicit stream object — never nullptr (hipStream_t(0)).
        // AMDDeviceContext creates a real hipStream_t via hipStreamCreateWithFlags,
        // so defaultStream() is a distinct object from hipStream_t(0). Using nullptr
        // would target the HIP legacy null stream, causing stream identity mismatches
        // with event-based synchronization.
        void *use_stream = use_default_stream ? gpu_ctx->defaultStream() : capture_stream;
        for (auto &seg : segment_cache.segments)
        {
            for (const auto &stage_name : seg.stage_names)
            {
                auto *node = graph.getNode(stage_name);
                if (!node || !node->stage)
                {
                    LOG_ERROR("[DeviceGraphCaptureController] Stream-only replay missing stage: " << stage_name);
                    return false;
                }

                node->stage->setGPUStream(use_stream);
                if (!node->stage->execute(ctx))
                {
                    LOG_ERROR("[DeviceGraphCaptureController] Stream-only stage failed: " << stage_name);
                    return false;
                }
                if (node->stage->isManualGraphBoundary() &&
                    !node->stage->manualGraphBoundaryComplete())
                {
                    LOG_ERROR("[DeviceGraphCaptureController] Stream-only manual boundary incomplete: "
                              << stage_name);
                    return false;
                }
                graph.markCompleted(stage_name);
            }
        }

        if (!gpu_ctx->synchronizeStreamChecked(use_stream))
        {
            LOG_ERROR("[DeviceGraphCaptureController] Stream-only replay sync failed");
            return false;
        }
        return true;
    }

    bool DeviceGraphCaptureController::segmentHasNonIdempotentStage(
        ComputeGraph &graph,
        const DeviceGraphExecutor::GraphSegment &segment)
    {
        for (const auto &stage_name : segment.stage_names)
        {
            auto *node = graph.getNode(stage_name);
            if (!node || !node->stage)
            {
                continue;
            }

            if (node->stage->type() == ComputeStageType::ADD_RESIDUAL)
            {
                return true;
            }
        }
        return false;
    }

    bool DeviceGraphCaptureController::executeManualReplaySegment(
        ComputeGraph &graph,
        DeviceGraphExecutor::GraphSegment &segment,
        IDeviceContext *ctx,
        IWorkerGPUContext *gpu_ctx,
        void *capture_stream,
        bool has_collective_nodes,
        bool needs_segment_sync,
        uint64_t current_step,
        const std::function<bool(ComputeNode &)> &execute_node_cb,
        const std::function<bool(ComputeNode &, void *)> &record_snapshot_copies_cb)
    {
        if (!ctx || !gpu_ctx)
        {
            LOG_ERROR("[DeviceGraphCaptureController] Manual replay missing context");
            return false;
        }

        const bool trace_replay = debugEnv().execution.gpu_graph_trace_replay;
        const auto &device_id = ctx->deviceId();

        bool manual_had_collective = false;
        for (const auto &stage_name : segment.stage_names)
        {
            auto *node = graph.getNode(stage_name);
            if (!node || !node->stage)
            {
                LOG_ERROR("[DeviceGraphCaptureController] Manual segment missing stage: " << stage_name);
                return false;
            }

            const auto stage_type = node->stage->type();
            const bool is_collective = node->stage->isCollectiveStage();

            if (is_collective)
            {
                // Collective stages (allreduce, allgather) run on their own
                // internal streams (RCCL/NCCL). The preceding captured graph
                // segment was replayed on capture_stream, so we MUST ensure
                // graph compute completes before the collective reads GPU data.
                //
                // We use GPU-side event dependencies instead of host-blocking
                // synchronizeStream() + hipDeviceSynchronize().  Host-level
                // blocking can deadlock multi-device TP: device A blocks in
                // hipDeviceSynchronize (waiting for RCCL, which needs device B)
                // while device B blocks in synchronizeStream or its own
                // hipDeviceSynchronize from a different segment.
                //
                // GPU-side events let both device threads queue ALL segments
                // without blocking, so all RCCL calls are enqueued promptly
                // and the only host sync is the final stream-level sync
                // at the end of executeReplayPhase().
                manual_had_collective = true;

                // Use the ACTUAL default stream, not nullptr (hipStream_t(0)).
                // AMDDeviceContext creates a real hipStream_t via
                // hipStreamCreateWithFlags, so defaultStream() != nullptr.
                // The RCCL coordinator's pre/post sync events are recorded
                // on this same stream (registered via setComputeStreams()),
                // so the event chain MUST target the same stream object.
                // Using nullptr would target the HIP null stream which is
                // a DIFFERENT stream — event dependencies wouldn't chain
                // with the allreduce at all, causing intermittent stale
                // reads and deadlocks.
                void *compute_stream = gpu_ctx->defaultStream();

                if (trace_replay)
                {
                    LOG_DEBUG("[ReplayTrace] " << device_id.toString()
                                               << " COLLECTIVE enter: " << stage_name
                                               << " insertStreamDependency(compute←capture)");
                }

                if (capture_stream)
                {
                    // GPU-side: compute_stream waits for capture_stream.
                    // The allreduce's pre-sync event (hipEventRecord on
                    // compute_stream) will then chain after graph
                    // completion, ensuring RCCL reads committed data.
                    if (!gpu_ctx->insertStreamDependency(
                            compute_stream,
                            capture_stream))
                    {
                        LOG_ERROR("[DeviceGraphCaptureController] Failed to publish "
                                  "capture-to-collective stream dependency for "
                                  << stage_name);
                        return false;
                    }
                }

                if (trace_replay)
                {
                    LOG_DEBUG("[ReplayTrace] " << device_id.toString()
                                               << " COLLECTIVE execute: " << stage_name);
                }

                node->stage->setGPUStream(compute_stream);
                if (!execute_node_cb(*node))
                {
                    LOG_ERROR("[DeviceGraphCaptureController] Manual collective stage failed on replay: " << stage_name);
                    return false;
                }

                if (capture_stream)
                {
                    // GPU-side: capture_stream waits for compute_stream.
                    // compute_stream already has a dependency on RCCL
                    // completion (via the allreduce's internal post-sync
                    // event), so capture_stream chains after RCCL finishes.
                    if (!gpu_ctx->insertStreamDependency(
                            capture_stream,
                            compute_stream))
                    {
                        LOG_ERROR("[DeviceGraphCaptureController] Failed to publish "
                                  "collective-to-capture stream dependency for "
                                  << stage_name);
                        return false;
                    }
                }

                if (trace_replay)
                {
                    LOG_DEBUG("[ReplayTrace] " << device_id.toString()
                                               << " COLLECTIVE done: " << stage_name);
                }
            }
            else
            {
                node->stage->setGPUStream(capture_stream);
                const bool stage_ok = execute_node_cb
                                          ? execute_node_cb(*node)
                                          : node->stage->execute(ctx);
                if (!stage_ok)
                {
                    LOG_ERROR("[DeviceGraphCaptureController] Manual stage failed on replay: " << stage_name);
                    return false;
                }
            }

            if (record_snapshot_copies_cb &&
                !record_snapshot_copies_cb(*node, capture_stream))
            {
                LOG_ERROR("[DeviceGraphCaptureController] Manual replay snapshot copy failed: "
                          << stage_name);
                return false;
            }

            if (node->stage->isManualGraphBoundary() &&
                !node->stage->manualGraphBoundaryComplete())
            {
                LOG_ERROR("[DeviceGraphCaptureController] Manual boundary incomplete on replay: "
                          << stage_name);
                return false;
            }
        }

        segment.last_executed_step = current_step;
        if (needs_segment_sync)
        {
            if (manual_had_collective)
            {
                // Collective stages ran on compute_stream (defaultStream),
                // non-collective stages ran on capture_stream. Sync both
                // instead of using device-wide hipDeviceSynchronize.
                if (!gpu_ctx->synchronizeStreamChecked(gpu_ctx->defaultStream()))
                {
                    LOG_ERROR("[DeviceGraphCaptureController] Manual replay default-stream sync failed after segment starting at "
                              << (segment.stage_names.empty() ? std::string("<empty>") : segment.stage_names.front()));
                    return false;
                }
                if (!gpu_ctx->synchronizeStreamChecked(capture_stream))
                {
                    LOG_ERROR("[DeviceGraphCaptureController] Manual replay capture-stream sync failed after segment starting at "
                              << (segment.stage_names.empty() ? std::string("<empty>") : segment.stage_names.front()));
                    return false;
                }
            }
            else
            {
                if (!gpu_ctx->synchronizeStreamChecked(capture_stream))
                {
                    LOG_ERROR("[DeviceGraphCaptureController] Manual replay capture-stream sync failed after segment starting at "
                              << (segment.stage_names.empty() ? std::string("<empty>") : segment.stage_names.front()));
                    return false;
                }
            }
        }

        return true;
    }

    bool DeviceGraphCaptureController::executeCapturedReplaySegmentNormal(
        DeviceGraphExecutor::GraphSegment &segment,
        IWorkerGPUContext *gpu_ctx,
        void *capture_stream,
        bool needs_segment_sync,
        bool full_graph_replay,
        const std::string &perf_context,
        const std::string &device_name,
        const DeviceGraphExecutor::GraphLaunchDependencyHook &launch_dependency_cb,
        const std::function<void(DeviceGraphExecutor::GraphSegment &, void *)> &post_launch_cb)
    {
        if (!gpu_ctx)
        {
            LOG_ERROR("[DeviceGraphCaptureController] Normal replay missing GPU context");
            return false;
        }
        if (!segment.capture)
        {
            LOG_ERROR("[DeviceGraphCaptureController] Normal replay missing segment capture");
            return false;
        }
        if (!capture_stream ||
            segment.capture->executionStream() != capture_stream)
        {
            LOG_ERROR(
                "[DeviceGraphCaptureController] Replay executable does not own "
                "the declared graph-cache producer stream"
                << " declared_stream=" << capture_stream
                << " executable_stream="
                << segment.capture->executionStream());
            return false;
        }

        const bool profiling =
            KernelProfiler::isEnabled() ||
            PerfStatsCollector::isDomainEnabled("forward_pass");
        const GraphReplayCaptureMode replay_mode = full_graph_replay
                                                       ? GraphReplayCaptureMode::FullGraph
                                                       : GraphReplayCaptureMode::Segmented;

        if (!applyGraphLaunchDependency(
                launch_dependency_cb,
                DeviceGraphExecutor::GraphExecutableLaunchPhase::SteadyReplay,
                capture_stream))
        {
            return false;
        }

        // Time the graph launch itself
        auto launch_t0 = std::chrono::high_resolution_clock::now();
        if (!segment.capture->launch())
        {
            LOG_ERROR("[DeviceGraphCaptureController] Segment graph launch failed on replay");
            return false;
        }
        if (profiling)
        {
            auto launch_t1 = std::chrono::high_resolution_clock::now();
            ForwardPassProfiler::addReplayLaunchNs(
                static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(launch_t1 - launch_t0).count()));
        }
        if (PerfStatsCollector::isDomainEnabled("forward_graph"))
        {
            auto launch_t1 = std::chrono::high_resolution_clock::now();
            PerfStatsCollector::recordTimingNs(
                "forward_graph",
                replayMetricName(replay_mode, "graph_launch"),
                static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(launch_t1 - launch_t0).count()),
                "decode",
                device_name,
                replaySegmentTags(segment, perf_context));
        }

        // Time post-launch output-coherence publication.
        auto post_t0 = std::chrono::high_resolution_clock::now();
        post_launch_cb(segment, capture_stream);
        if (profiling)
        {
            auto post_t1 = std::chrono::high_resolution_clock::now();
            ForwardPassProfiler::addReplayPostLaunchNs(
                static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(post_t1 - post_t0).count()));
        }
        if (PerfStatsCollector::isDomainEnabled("forward_graph"))
        {
            auto post_t1 = std::chrono::high_resolution_clock::now();
            PerfStatsCollector::recordTimingNs(
                "forward_graph",
                replayMetricName(replay_mode, "post_launch"),
                static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(post_t1 - post_t0).count()),
                "decode",
                device_name,
                replaySegmentTags(segment, perf_context));
        }

        if (needs_segment_sync)
        {
            if (!gpu_ctx->synchronizeStreamChecked(capture_stream))
            {
                LOG_ERROR("[DeviceGraphCaptureController] Captured replay stream sync failed after segment starting at "
                          << (segment.stage_names.empty() ? std::string("<empty>") : segment.stage_names.front()));
                return false;
            }
        }

        return true;
    }

    bool DeviceGraphCaptureController::executeCapturedReplaySegmentRecapture(
        ComputeGraph &graph,
        DeviceGraphExecutor::GraphSegment &segment,
        IDeviceContext *ctx,
        IWorkerGPUContext *gpu_ctx,
        void *capture_stream,
        int segment_index,
        uint64_t current_step,
        const std::string &perf_context,
        const std::function<bool(const DeviceGraphExecutor::GraphSegment &)> &cohere_inputs_cb,
        const std::function<std::unique_ptr<GraphCaptureDependencyLedger>(
            const DeviceGraphExecutor::GraphSegment &)> &plan_capture_dependencies_cb,
        const std::function<bool(ComputeNode &)> &execute_node_cb,
        const DeviceGraphExecutor::GraphCaptureBoundaryHook &capture_boundary_cb,
        IGraphCaptureAuxiliaryBranch *auxiliary_branch,
        const std::function<bool(ComputeNode &, void *)> &record_snapshot_copies_cb,
        const DeviceGraphExecutor::GraphLaunchDependencyHook &launch_dependency_cb,
        const std::function<void(DeviceGraphExecutor::GraphSegment &, void *)> &post_launch_cb)
    {
        if (!ctx || !gpu_ctx)
        {
            LOG_ERROR("[DeviceGraphCaptureController] Re-capture missing context");
            return false;
        }
        if (!segment.capture)
        {
            LOG_ERROR("[DeviceGraphCaptureController] Re-capture missing segment capture");
            return false;
        }
        if (!execute_node_cb)
        {
            LOG_ERROR(
                "[DeviceGraphCaptureController] Re-capture requires the canonical "
                "executor stage hook");
            return false;
        }

        for (const auto &stage_name : segment.stage_names)
        {
            auto *node = graph.getNode(stage_name);
            if (node && node->stage)
            {
                node->stage->setGPUStream(capture_stream);
            }
        }

        if (!prepareGraphLaunchMetadata(
                graph,
                segment,
                ctx,
                capture_stream,
                GraphLaunchPreparationPhase::Capture))
        {
            LOG_ERROR("[DeviceGraphCaptureController] Re-capture metadata preparation failed, seg "
                      << segment_index);
            return false;
        }
        /*
         * Launch metadata preparation can publish new device writes. Join those
         * exact event generations to the capture stream only after preparation,
         * but before the domain rendezvous and beginCapture(). Stage-level
         * requirePreparedInput() calls inside capture then prove this prejoin
         * instead of attempting an illegal external-event import.
         */
        if (!cohere_inputs_cb)
        {
            LOG_ERROR("[DeviceGraphCaptureController] Re-capture requires a pre-capture input coherence hook, seg "
                      << segment_index);
            return false;
        }
        if (!cohere_inputs_cb(segment))
        {
            LOG_ERROR("[DeviceGraphCaptureController] Re-capture input preflight failed, seg "
                      << segment_index);
            return false;
        }

        bool exec_ok = true;
        std::string failed_stage_name;
        auto dependency_ledger = planCaptureDependencies(
            graph,
            segment,
            ctx->deviceId(),
            capture_stream,
            plan_capture_dependencies_cb,
            "segment_recapture_" + std::to_string(segment_index));
        if (!dependency_ledger)
            return false;
        {
            if (capture_boundary_cb)
            {
                const std::string boundary_name =
                    graphCaptureBoundaryName(
                        "recapture_begin",
                        current_step,
                        segment.capture_wave_ordinal,
                        segment.capture_wave_identity,
                        &segment,
                        perf_context);
                if (!capture_boundary_cb(boundary_name, capture_stream))
                {
                    LOG_ERROR("[DeviceGraphCaptureController] Re-capture begin boundary rendezvous failed, seg "
                              << segment_index << " boundary=" << boundary_name);
                    return false;
                }
            }

            ScopedBackendGraphCapture capture_transaction(
                *gpu_ctx,
                *segment.capture,
                "segment recapture index=" +
                    std::to_string(segment_index),
                dependency_ledger.get());
            if (!capture_transaction.begin())
            {
                LOG_ERROR("[DeviceGraphCaptureController] Re-capture beginCapture failed, seg " << segment_index);
                return false;
            }

            ScopedCapturedAuxiliaryBranch captured_auxiliary(
                auxiliary_branch,
                capture_stream);
            if (!captured_auxiliary.begin())
            {
                exec_ok = false;
                failed_stage_name = "<auxiliary_branch_fork>";
            }

            for (const auto &stage_name : segment.stage_names)
            {
                if (!exec_ok)
                    break;
                auto *node = graph.getNode(stage_name);
                if (!node || !node->stage || !execute_node_cb(*node))
                {
                    exec_ok = false;
                    failed_stage_name = stage_name;
                    break;
                }
                if (record_snapshot_copies_cb &&
                    !record_snapshot_copies_cb(*node, capture_stream))
                {
                    LOG_ERROR("[DeviceGraphCaptureController] Re-capture snapshot copy failed: "
                              << stage_name);
                    exec_ok = false;
                    failed_stage_name = stage_name;
                    break;
                }
            }

            if (!captured_auxiliary.finish() && exec_ok)
            {
                exec_ok = false;
                failed_stage_name = "<auxiliary_branch_join>";
            }

            capture_transaction.finish();
        }

        /*
         * endCapture() is participant-local. A faster LocalTP participant must
         * not proceed to a manual NCCL/RCCL segment while a sibling remains in
         * global stream-capture mode, because CUDA/HIP then rejects the
         * collective stream dependency. Pair every begin rendezvous with this
         * exit rendezvous before graph instantiation or manual execution.
         */
        if (capture_boundary_cb)
        {
            const std::string boundary_name =
                graphCaptureBoundaryName(
                    exec_ok ? "recapture_end" : "recapture_end_failed",
                    current_step,
                    segment.capture_wave_ordinal,
                    segment.capture_wave_identity,
                    &segment,
                    perf_context);
            if (!capture_boundary_cb(boundary_name, capture_stream))
            {
                LOG_ERROR("[DeviceGraphCaptureController] Re-capture end boundary rendezvous failed, seg "
                          << segment_index << " boundary=" << boundary_name);
                return false;
            }
        }

        if (!exec_ok)
        {
            LOG_ERROR(
                "[DeviceGraphCaptureController] Re-capture stage execution "
                "failed; refusing eager recovery, seg "
                << segment_index
                << " stage="
                << (failed_stage_name.empty()
                        ? std::string("<unknown>")
                        : failed_stage_name));
            return false;
        }

        /*
         * Executable publication is selected by the graph implementation's
         * typed capability, not by device-name knowledge in the orchestrator.
         * A backend without in-place update support replaces its executable
         * directly. Runtime update failures are therefore genuine failures,
         * never a feature-probe or an invitation to clear errors and retry.
         */
        GraphUpdateResult update_result = GraphUpdateResult::NeedsReinstantiate;
        if (segment.capture->supportsExecutableUpdate())
        {
            update_result = segment.capture->tryUpdate();
            if (update_result == GraphUpdateResult::Failed)
            {
                LOG_ERROR(
                    "[DeviceGraphCaptureController] Re-capture graph update failed; "
                    "refusing to discard the backend error and instantiate a "
                    "replacement executable");
                return false;
            }
        }
        if (update_result == GraphUpdateResult::NeedsReinstantiate)
        {
            if (!segment.capture->instantiate())
            {
                LOG_ERROR("[DeviceGraphCaptureController] Re-capture instantiate failed");
                return false;
            }
        }

        if (!applyGraphLaunchDependency(
                launch_dependency_cb,
                DeviceGraphExecutor::GraphExecutableLaunchPhase::DiagnosticRecapture,
                capture_stream))
        {
            return false;
        }

        if (!segment.capture->launch())
        {
            LOG_ERROR("[DeviceGraphCaptureController] Re-capture launch failed");
            return false;
        }

        post_launch_cb(segment, capture_stream);

        for (const auto &stage_name : segment.stage_names)
        {
            graph.markCompleted(stage_name);
        }

        return true;
    }

    DeviceGraphCaptureController::VerifyReplayResult DeviceGraphCaptureController::executeCapturedReplaySegmentVerify(
        ComputeGraph &graph,
        DeviceGraphExecutor::GraphSegment &segment,
        IDeviceContext *ctx,
        IWorkerGPUContext *gpu_ctx,
        void *capture_stream,
        bool needs_segment_sync,
        int segment_index,
        const DeviceGraphExecutor::GraphLaunchDependencyHook &launch_dependency_cb,
        const std::function<void(DeviceGraphExecutor::GraphSegment &, void *)> &post_launch_cb)
    {
        VerifyReplayResult result{false, false};

        if (!ctx || !gpu_ctx)
        {
            LOG_ERROR("[DeviceGraphCaptureController] Verify replay missing context");
            return result;
        }
        if (!segment.capture)
        {
            LOG_ERROR("[DeviceGraphCaptureController] Verify replay missing segment capture");
            return result;
        }

        if (segmentHasNonIdempotentStage(graph, segment))
        {
            // Verify mode compares "captured replay" against "direct execute".
            // Non-idempotent stages (e.g. residual accumulation) can legitimately
            // diverge on re-execution, so we skip comparison but still run replay.
            if (!applyGraphLaunchDependency(
                    launch_dependency_cb,
                    DeviceGraphExecutor::GraphExecutableLaunchPhase::DiagnosticVerification,
                    capture_stream))
            {
                return result;
            }
            if (!segment.capture->launch())
            {
                LOG_ERROR("[DeviceGraphCaptureController] Verify-skip: graph launch failed, seg " << segment_index);
                return result;
            }

            post_launch_cb(segment, capture_stream);

            if (needs_segment_sync)
            {
                if (!gpu_ctx->synchronizeStreamChecked(capture_stream))
                {
                    LOG_ERROR("[DeviceGraphCaptureController] Verify-skip stream sync failed after seg " << segment_index);
                    return result;
                }
            }

            fprintf(stderr,
                    "[GRAPH_VERIFY] seg %d skipped (non-idempotent stage detected)\n",
                    segment_index);

            for (const auto &stage_name : segment.stage_names)
            {
                graph.markCompleted(stage_name);
            }

            result.success = true;
            result.skipped_non_idempotent = true;
            return result;
        }

        if (!applyGraphLaunchDependency(
                launch_dependency_cb,
                DeviceGraphExecutor::GraphExecutableLaunchPhase::DiagnosticVerification,
                capture_stream))
        {
            return result;
        }
        if (!segment.capture->launch())
        {
            LOG_ERROR("[DeviceGraphCaptureController] Verify: graph launch failed, seg " << segment_index);
            return result;
        }
        post_launch_cb(segment, capture_stream);
        if (!gpu_ctx->synchronizeStreamChecked(capture_stream))
        {
            LOG_ERROR("[DeviceGraphCaptureController] Verify stream sync failed after seg " << segment_index);
            return result;
        }

        void *default_stream = gpu_ctx->defaultStream();

        struct StageOutput
        {
            std::string name;
            float values[8] = {};
            size_t count = 0;
            bool has_gpu_ptr = false;
        };

        std::vector<StageOutput> graph_outputs(segment.stage_names.size());
        // Pass 1: collect outputs from captured replay launch.
        for (size_t s = 0; s < segment.stage_names.size(); s++)
        {
            auto *node = graph.getNode(segment.stage_names[s]);
            graph_outputs[s].name = segment.stage_names[s];
            if (node && node->stage)
            {
                StageDumpInfo dump_info = node->stage->getDumpInfoSnapshot();
                if (!dump_info.outputs.empty())
                {
                    const auto &out = dump_info.outputs[0];
                    if (out.tensor)
                    {
                        auto *base = dynamic_cast<TensorBase *>(out.tensor);
                        if (base)
                        {
                            const void *gpu_ptr = base->gpu_data_ptr();
                            if (gpu_ptr)
                            {
                                graph_outputs[s].count = std::min<size_t>(8, out.rows * out.cols);
                                graph_outputs[s].has_gpu_ptr =
                                    ctx->copyToHost(graph_outputs[s].values, gpu_ptr,
                                                    graph_outputs[s].count * sizeof(float),
                                                    capture_stream);
                            }
                        }
                    }
                }
            }
        }

        for (const auto &stage_name : segment.stage_names)
        {
            // Pass 2: execute the same stages directly (no graph launch) to
            // produce a reference output for comparison.
            auto *node = graph.getNode(stage_name);
            if (!node || !node->stage)
            {
                LOG_ERROR("[DeviceGraphCaptureController] Verify: missing stage during direct exec: " << stage_name);
                return result;
            }
            node->stage->setGPUStream(default_stream);
            if (!node->stage->execute(ctx))
            {
                LOG_ERROR("[DeviceGraphCaptureController] Verify: direct exec failed: " << stage_name);
                return result;
            }
        }
        if (!gpu_ctx->synchronizeStreamChecked(default_stream))
        {
            LOG_ERROR("[DeviceGraphCaptureController] Verify direct-exec stream sync failed");
            return result;
        }

        std::vector<StageOutput> direct_outputs(segment.stage_names.size());
        for (size_t s = 0; s < segment.stage_names.size(); s++)
        {
            auto *node = graph.getNode(segment.stage_names[s]);
            direct_outputs[s].name = segment.stage_names[s];
            if (node && node->stage)
            {
                StageDumpInfo dump_info = node->stage->getDumpInfoSnapshot();
                if (!dump_info.outputs.empty())
                {
                    const auto &out = dump_info.outputs[0];
                    if (out.tensor)
                    {
                        auto *base = dynamic_cast<TensorBase *>(out.tensor);
                        if (base)
                        {
                            const void *gpu_ptr = base->gpu_data_ptr();
                            if (gpu_ptr)
                            {
                                direct_outputs[s].count = std::min<size_t>(8, out.rows * out.cols);
                                direct_outputs[s].has_gpu_ptr =
                                    ctx->copyToHost(direct_outputs[s].values, gpu_ptr,
                                                    direct_outputs[s].count * sizeof(float),
                                                    default_stream);
                            }
                        }
                    }
                }
            }
        }

        FILE *f = fopen("/tmp/graph_phase3.log", "a");
        float seg_max_diff = 0;

        for (size_t s = 0; s < segment.stage_names.size(); s++)
        {
            if (!graph_outputs[s].has_gpu_ptr || !direct_outputs[s].has_gpu_ptr)
            {
                continue;
            }

            float max_diff = 0;
            size_t n = std::min(graph_outputs[s].count, direct_outputs[s].count);
            for (size_t i = 0; i < n; i++)
            {
                max_diff = std::max(max_diff, std::abs(graph_outputs[s].values[i] - direct_outputs[s].values[i]));
            }
            seg_max_diff = std::max(seg_max_diff, max_diff);

            char buf[1024];
            snprintf(buf, sizeof(buf),
                     "[STAGE_VERIFY] seg %d stage %zu/%zu (%s) max_diff=%.6e",
                     segment_index, s, segment.stage_names.size(), segment.stage_names[s].c_str(), max_diff);

            if (f)
            {
                fprintf(f, "%s\n", buf);
                if (max_diff > 1e-6f)
                {
                    fprintf(f, "  GRAPH : ");
                    for (size_t i = 0; i < n; i++)
                    {
                        fprintf(f, "%.6f%s", graph_outputs[s].values[i], i < n - 1 ? ", " : "");
                    }
                    fprintf(f, "\n  DIRECT: ");
                    for (size_t i = 0; i < n; i++)
                    {
                        fprintf(f, "%.6f%s", direct_outputs[s].values[i], i < n - 1 ? ", " : "");
                    }
                    fprintf(f, "\n");
                }
                fflush(f);
            }
        }

        {
            char buf[512];
            snprintf(buf, sizeof(buf),
                     "[GRAPH_VERIFY] seg %d (%zu stages, %zu nodes, last=%s) seg_max_diff=%.6e",
                     segment_index, segment.stage_names.size(), segment.capture->nodeCount(),
                     segment.stage_names.back().c_str(), seg_max_diff);
            fprintf(stderr, "%s\n", buf);
            if (f)
            {
                fprintf(f, "%s\n\n", buf);
                fflush(f);
            }
        }

        if (f)
        {
            fclose(f);
        }

        for (const auto &stage_name : segment.stage_names)
        {
            graph.markCompleted(stage_name);
        }

        result.success = true;
        return result;
    }

    bool DeviceGraphCaptureController::finalizeCapturePhaseCapturableSegment(
        ComputeGraph &graph,
        DeviceGraphExecutor::GraphSegment &segment,
        IDeviceContext *ctx,
        IWorkerGPUContext *gpu_ctx,
        void *capture_stream,
        bool full_graph_capture,
        const std::string &perf_context,
        uint64_t current_step,
        const DeviceGraphExecutor::GraphLaunchDependencyHook &launch_dependency_cb,
        const std::function<void(DeviceGraphExecutor::GraphSegment &, void *)> &post_launch_cb,
        CapturedUnitFinalization finalization)
    {
        if (!ctx || !gpu_ctx)
        {
            LOG_ERROR("[DeviceGraphCaptureController] Capture finalize missing context");
            return false;
        }
        if (!segment.capture)
        {
            LOG_ERROR("[DeviceGraphCaptureController] Capture finalize missing segment capture");
            return false;
        }
        if (!capture_stream ||
            segment.capture->executionStream() != capture_stream)
        {
            LOG_ERROR(
                "[DeviceGraphCaptureController] Captured executable does not own "
                "the declared graph-cache producer stream"
                << " declared_stream=" << capture_stream
                << " executable_stream="
                << segment.capture->executionStream());
            return false;
        }

        if (segment.capture->nodeCount() == 0)
        {
            LOG_ERROR(
                "[DeviceGraphCaptureController] A mandatory capturable graph "
                "unit produced zero native nodes; refusing manual execution"
                << " mode="
                << (full_graph_capture ? "full_graph" : "segmented")
                << " stages=" << describeSegmentStages(segment));
            return false;
        }

        /*
         * Inspect the exact graph that will be retained, rather than a logical
         * stage list or a capture-time launch counter. This opt-in boundary is
         * shared with MTP composition and therefore exposes missing native
         * nodes in either lifecycle with one diagnostic vocabulary.
         */
        const std::string inventory_fragment =
            perf_context +
            (full_graph_capture ? "/full/" : "/segment/") +
            (segment.stage_names.empty()
                 ? std::string("empty")
                 : segment.stage_names.front());
        std::string inventory_error;
        if (!publishKernelInventory(
                *segment.capture,
                inventory_fragment,
                std::nullopt,
                ctx->deviceId(),
                inventory_error))
        {
            LOG_ERROR(
                "[DeviceGraphCaptureController] " << inventory_error);
            return false;
        }

        const bool graph_only_child =
            finalization == CapturedUnitFinalization::RetainGraphOnly;
        const bool instantiate_without_launch =
            finalization ==
            CapturedUnitFinalization::InstantiateWithoutLaunch;
        if (graph_only_child && segment.capture->hasExecutable())
        {
            LOG_ERROR(
                "[DeviceGraphCaptureController] Retained parent child unexpectedly owns an executable before composition");
            return false;
        }
        if (!graph_only_child && !segment.capture->instantiate())
        {
            LOG_ERROR("[DeviceGraphCaptureController] Segment instantiation failed ("
                      << segment.capture->nodeCount() << " nodes)");
            return false;
        }

        const GraphReplayCaptureMode capture_mode =
            full_graph_capture
                ? GraphReplayCaptureMode::FullGraph
                : GraphReplayCaptureMode::Segmented;
        auto executable_tags = replaySegmentTags(segment, perf_context);
        executable_tags["backend"] = segment.capture->backendName();
        executable_tags["type"] =
            graph_only_child
                ? "captured_child_template"
                : instantiate_without_launch
                      ? "materialized_unlaunched_executable"
                      : "captured_executable";
        PerfStatsCollector::addCounter(
            "forward_graph",
            graph_only_child
                ? "retained_parent_child_graph_nodes"
                : (full_graph_capture
                       ? "full_graph_capture_executable_nodes"
                       : "segmented_graph_capture_executable_nodes"),
            static_cast<double>(segment.capture->nodeCount()),
            "decode",
            ctx->deviceId().toString(),
            graphReplayMetadataTags(
                std::move(executable_tags),
                perf_context,
                capture_mode));

        /*
         * Native node count alone cannot explain a changed graph topology: one
         * logical stage can lower to several CUDA/HIP graph nodes. Publish the
         * logical stage inventory at the same durable capture boundary so a
         * performance artifact can identify exactly which stage family grew
         * without enabling verbose logging or rebuilding under a debugger.
         */
        std::map<std::string, size_t> executable_stage_types;
        for (const auto &stage_name : segment.stage_names)
        {
            const auto *node = graph.getNode(stage_name);
            if (!node || !node->stage)
            {
                LOG_ERROR(
                    "[DeviceGraphCaptureController] Captured executable is "
                    "missing a logical stage while publishing its topology: "
                    << stage_name);
                return false;
            }
            executable_stage_types[
                computeStageTypeName(node->stage->type())]++;
        }
        for (const auto &[stage_type, count] : executable_stage_types)
        {
            PerfStatsCollector::addCounter(
                "forward_graph",
                full_graph_capture
                    ? "full_graph_capture_stage_types"
                    : "segmented_graph_capture_stage_types",
                static_cast<double>(count),
                "decode",
                ctx->deviceId().toString(),
                graphReplayMetadataTags(
                    {{"stage_type", stage_type},
                     {"type", graph_only_child
                                  ? "captured_child_template"
                                  : "captured_executable"}},
                    perf_context,
                capture_mode));
        }

        if (graph_only_child)
        {
            LOG_DEBUG(
                "[DeviceGraphCaptureController] Retained graph-only child captured: "
                << segment.capture->nodeCount() << " nodes, "
                << segment.stage_names.size() << " stages");
            return true;
        }

        if (instantiate_without_launch)
        {
            /*
             * Native capture and instantiation are complete, but setup has not
             * acquired an inference epoch and therefore cannot publish writes or
             * submit arithmetic. Freeze only the host-side publication manifest;
             * the first authenticated launch will apply it exactly once.
             */
            cacheCapturedSegmentArenaWrites(graph, segment);
            LOG_DEBUG(
                "[DeviceGraphCaptureController] Segment materialized without launch: "
                << segment.capture->nodeCount() << " nodes, "
                << segment.stage_names.size() << " stages");
            return true;
        }

        /*
         * Capture records launch topology but does not execute it. Transaction
         * zero is therefore always the first launch of the new executable,
         * including graphs containing NCCL/RCCL nodes. Re-running stage methods
         * manually here would bypass the captured collective topology and create
         * a second, host-orchestrated execution contract.
         */
        if (!applyGraphLaunchDependency(
                launch_dependency_cb,
                DeviceGraphExecutor::GraphExecutableLaunchPhase::InitialTransaction,
                capture_stream))
        {
            return false;
        }
        if (!segment.capture->launch())
        {
            LOG_ERROR("[DeviceGraphCaptureController] Segment initial launch failed");
            return false;
        }
        // Capture and replay use the same post-launch coherence lifecycle;
        // mutable execution state remains device-owned in both phases.
        post_launch_cb(segment, capture_stream);
        LOG_DEBUG("[DeviceGraphCaptureController] Segment captured+launched: "
                  << segment.capture->nodeCount() << " nodes, " << segment.stage_names.size() << " stages");
        return true;
    }

    bool DeviceGraphCaptureController::finalizeRetainedParentTransaction(
        ComputeGraph &graph,
        DeviceGraphExecutor::GraphSegmentCache &segment_cache,
        IDeviceContext *ctx,
        IWorkerGPUContext *gpu_ctx,
        uint64_t current_step,
        const ReplayHooks &hooks,
        DeviceGraphExecutor::GraphInitialSubmissionPolicy initial_submission)
    {
        if (!ctx || !gpu_ctx || !segment_cache.capture_stream ||
            !hooks.retained_parent_composer ||
            segment_cache.retained_parent_capture)
        {
            LOG_ERROR(
                "[DeviceGraphCaptureController] Retained parent finalization has incomplete context, composer, stream, or ownership");
            return false;
        }

        std::string child_error;
        auto child_units =
            segment_cache.retainedCaptureUnitTemplatesForParentComposition(
                graph, &child_error);
        if (!child_units)
        {
            LOG_ERROR(
                "[DeviceGraphCaptureController] Retained parent child export failed: "
                << child_error);
            return false;
        }

        std::unique_ptr<IGPUGraphCapture> parent =
            gpu_ctx->createGraphCapture(segment_cache.capture_stream);
        if (!parent ||
            parent->executionStream() != segment_cache.capture_stream ||
            parent->nodeCount() != 0u || parent->hasExecutable())
        {
            LOG_ERROR(
                "[DeviceGraphCaptureController] Retained parent factory did not return one empty graph owner on the exact cache stream");
            return false;
        }

        bool composition_ok = false;
        try
        {
            composition_ok = hooks.retained_parent_composer(
                *parent, graph, *child_units);
        }
        catch (const std::exception &ex)
        {
            LOG_ERROR(
                "[DeviceGraphCaptureController] Retained parent composition rejected its typed topology: "
                << ex.what());
            return false;
        }
        if (!composition_ok || parent->nodeCount() == 0u ||
            parent->hasExecutable() ||
            parent->executionStream() != segment_cache.capture_stream)
        {
            LOG_ERROR(
                "[DeviceGraphCaptureController] Retained parent composer did not produce one non-empty, uninstantiated graph on the exact stream");
            return false;
        }
        if (!parent->instantiate())
        {
            LOG_ERROR(
                "[DeviceGraphCaptureController] Retained parent instantiation failed");
            return false;
        }

        const bool materialize_without_launch =
            initial_submission ==
            DeviceGraphExecutor::GraphInitialSubmissionPolicy::
                MaterializeWithoutLaunch;
        if (materialize_without_launch)
        {
            /*
             * The child contracts already describe every write performed by the
             * composed parent. Cache those manifests now without advancing arena
             * authority: no device node has run yet. Ownership moves into the
             * cache only after the executable is fully instantiated.
             */
            for (auto &segment : segment_cache.segments)
                cacheCapturedSegmentArenaWrites(graph, segment);
            segment_cache.retained_parent_capture = std::move(parent);
            PerfStatsCollector::addCounter(
                "forward_graph",
                "retained_parent_materialized_without_launch",
                1.0,
                "setup",
                ctx->deviceId().toString(),
                {{"context", segment_cache.perf_context},
                 {"child_units", std::to_string(child_units->size())},
                 {"parent_nodes",
                  std::to_string(
                      segment_cache.retained_parent_capture->nodeCount())}});
            LOG_DEBUG(
                "[DeviceGraphCaptureController] Retained parent composed and "
                "instantiated without launch: "
                << segment_cache.retained_parent_capture->nodeCount()
                << " nodes from " << child_units->size()
                << " graph-only children");
            return true;
        }
        if (!applyGraphLaunchDependency(
                hooks.launch_dependency,
                DeviceGraphExecutor::GraphExecutableLaunchPhase::
                    InitialTransaction,
                segment_cache.capture_stream))
        {
            return false;
        }
        if (!parent->launch())
        {
            LOG_ERROR(
                "[DeviceGraphCaptureController] Retained parent transaction-zero launch failed");
            return false;
        }

        /*
         * Child graphs were never submitted. Their buffer contracts still form
         * the complete parent publication manifest, so cache each segment's
         * writes only after the one parent launch has been accepted by the
         * backend. This is host-side authority bookkeeping, not child replay.
         */
        for (auto &segment : segment_cache.segments)
            hooks.post_launch(segment, segment_cache.capture_stream);

        PerfStatsCollector::addCounter(
            "forward_graph",
            "retained_parent_transaction_zero_launches",
            1.0,
            "decode",
            ctx->deviceId().toString(),
            {{"context", segment_cache.perf_context},
             {"child_units", std::to_string(child_units->size())},
             {"parent_nodes", std::to_string(parent->nodeCount())}});
        segment_cache.retained_parent_capture = std::move(parent);
        LOG_DEBUG(
            "[DeviceGraphCaptureController] Retained parent composed+launched: "
            << segment_cache.retained_parent_capture->nodeCount()
            << " nodes from " << child_units->size() << " graph-only children"
            << " step=" << current_step);
        return true;
    }

    bool DeviceGraphCaptureController::executeCapturePhaseManualSegment(
        ComputeGraph &graph,
        DeviceGraphExecutor::GraphSegment &segment,
        IDeviceContext *ctx,
        IWorkerGPUContext *gpu_ctx,
        void *capture_stream,
        bool has_collective_nodes,
        uint64_t current_step,
        const std::function<bool(ComputeNode &)> &execute_node_cb,
        const std::function<bool(ComputeNode &, void *)> &record_snapshot_copies_cb)
    {
        if (!ctx || !gpu_ctx || !capture_stream)
        {
            LOG_ERROR("[DeviceGraphCaptureController] Capture manual segment missing context");
            return false;
        }

        /*
         * Phase-2 capture runs after cached dynamic parameters have already
         * uploaded token/position metadata to the capture stream. Non-collective
         * manual stages must therefore execute on that same stream; rebinding
         * them to the worker stream can make CUDA read stale per-step metadata.
         *
         * Collective stages are the exception. NCCL/RCCL stages use the worker
         * stream registered with the collective coordinator, so we bridge with
         * GPU-side stream dependencies instead of falling back to the null stream
         * or a host-wide device synchronize.
        */
        void *compute_stream = gpu_ctx->defaultStream();

        for (const auto &stage_name : segment.stage_names)
        {
            auto *node = graph.getNode(stage_name);
            if (!node || !node->stage)
            {
                LOG_ERROR("[DeviceGraphCaptureController] Capture manual segment missing stage: " << stage_name);
                return false;
            }

            const bool is_collective = node->stage->isCollectiveStage();
            void *stage_stream = capture_stream;
            if (is_collective)
            {
                stage_stream = compute_stream;
                if (!gpu_ctx->insertStreamDependency(
                        compute_stream,
                        capture_stream))
                {
                    LOG_ERROR("[DeviceGraphCaptureController] Failed to publish "
                              "capture-to-collective dependency while recording "
                              "manual segment stage "
                              << stage_name);
                    return false;
                }
            }

            node->stage->setGPUStream(stage_stream);

            if (execute_node_cb)
            {
                if (!execute_node_cb(*node))
                {
                    LOG_ERROR("[DeviceGraphCaptureController] Capture manual stage failed: " << stage_name);
                    return false;
                }
            }
            else if (is_collective)
            {
                LOG_ERROR("[DeviceGraphCaptureController] Capture manual collective stage has no canonical executor hook: "
                          << stage_name);
                return false;
            }
            else if (!node->stage->execute(ctx))
            {
                LOG_ERROR("[DeviceGraphCaptureController] Capture manual stage failed: " << stage_name);
                return false;
            }

            if (is_collective)
            {
                if (!gpu_ctx->insertStreamDependency(
                        capture_stream,
                        compute_stream))
                {
                    LOG_ERROR("[DeviceGraphCaptureController] Failed to publish "
                              "collective-to-capture dependency while recording "
                              "manual segment stage "
                              << stage_name);
                    return false;
                }
                stage_stream = capture_stream;
            }

            if (record_snapshot_copies_cb &&
                !record_snapshot_copies_cb(*node, stage_stream))
            {
                LOG_ERROR("[DeviceGraphCaptureController] Capture manual snapshot copy failed: "
                          << stage_name);
                return false;
            }

            if (node->stage->isManualGraphBoundary() &&
                !node->stage->manualGraphBoundaryComplete())
            {
                LOG_ERROR("[DeviceGraphCaptureController] Capture manual boundary incomplete: "
                          << stage_name);
                return false;
            }

            graph.markCompleted(stage_name);
        }

        segment.last_executed_step = current_step;
        return true;
    }

    bool DeviceGraphCaptureController::prepareGraphLaunchMetadata(
        ComputeGraph &graph,
        const DeviceGraphExecutor::GraphSegment &segment,
        IDeviceContext *ctx,
        void *stream,
        GraphLaunchPreparationPhase phase)
    {
        for (const auto &stage_name : segment.stage_names)
        {
            auto *node = graph.getNode(stage_name);
            if (!node || !node->stage ||
                !requiresGraphLaunchPreparation(
                    node->stage->graphLaunchPreparationPolicy(), phase))
                continue;

            if (stream)
                node->stage->setGPUStream(stream);
            if (!node->stage->prepareGraphLaunch(ctx, stream))
            {
                LOG_ERROR("[DeviceGraphCaptureController] Graph launch metadata preparation failed for stage: "
                          << stage_name);
                return false;
            }
        }
        return true;
    }

    DeviceGraphCaptureController::ReplayCapturableResult DeviceGraphCaptureController::executeReplayCapturableSegment(
        ComputeGraph &graph,
        DeviceGraphExecutor::GraphSegment &segment,
        IDeviceContext *ctx,
        IWorkerGPUContext *gpu_ctx,
        void *capture_stream,
        bool needs_segment_sync,
        bool verify_mode,
        bool recapture_mode,
        bool require_input_preflight,
        bool full_graph_replay,
        int segment_index,
        uint64_t current_step,
        const std::string &perf_context,
        const std::function<bool(const DeviceGraphExecutor::GraphSegment &)> &cohere_inputs_cb,
        const std::function<bool(ComputeNode &)> &execute_node_cb,
        const std::function<std::unique_ptr<GraphCaptureDependencyLedger>(
            const DeviceGraphExecutor::GraphSegment &)> &plan_capture_dependencies_cb,
        const DeviceGraphExecutor::GraphCaptureBoundaryHook &capture_boundary_cb,
        IGraphCaptureAuxiliaryBranch *auxiliary_branch,
        const std::function<bool(ComputeNode &, void *)> &record_snapshot_copies_cb,
        const DeviceGraphExecutor::GraphLaunchDependencyHook &launch_dependency_cb,
        const std::function<void(DeviceGraphExecutor::GraphSegment &, void *)> &post_launch_cb)
    {
        ReplayCapturableResult result{};

        if (!ctx || !gpu_ctx)
        {
            LOG_ERROR("[DeviceGraphCaptureController] Capturable replay missing context");
            return result;
        }

        // OPTIMIZATION: Skip coherence for normal replay of capturable segments.
        // All buffers (inputs, weights, outputs) were ensured on device during
        // the capture phase and haven't moved off GPU since. The graph replay
        // writes to the same GPU buffers, so re-checking is_on_device() for every
        // tensor of every stage (338 stages × ~4 buffers = 1352 checks with
        // dynamic_cast + virtual getDumpInfo) is pure CPU overhead.
        //
        // Coherence IS needed for verify/recapture modes since they may re-execute
        // stages in a different order or on different streams. It is also
        // mandatory for transaction zero of a setup-materialized executable:
        // setup bound addresses without publishing request bytes. Recapture owns
        // the preflight internally so it runs after launch-metadata preparation.
        const bool prepare_replay_inputs =
            (verify_mode || require_input_preflight) && !recapture_mode;
        const std::string device_name = ctx->deviceId().toString();

        if (prepare_replay_inputs &&
            (!cohere_inputs_cb || !cohere_inputs_cb(segment)))
        {
            return result;
        }

        if (!recapture_mode &&
            !prepareGraphLaunchMetadata(
                graph,
                segment,
                ctx,
                capture_stream,
                GraphLaunchPreparationPhase::Replay))
        {
            return result;
        }

        if (recapture_mode)
        {
            const bool recapture_ok = executeCapturedReplaySegmentRecapture(
                graph,
                segment,
                ctx,
                gpu_ctx,
                capture_stream,
                segment_index,
                current_step,
                perf_context,
                cohere_inputs_cb,
                plan_capture_dependencies_cb,
                execute_node_cb,
                capture_boundary_cb,
                auxiliary_branch,
                record_snapshot_copies_cb,
                launch_dependency_cb,
                post_launch_cb);
            result.success =
                recapture_ok &&
                joinPassiveCaptureWaves(
                    graph,
                    segment,
                    "recapture",
                    current_step,
                    perf_context,
                    capture_boundary_cb,
                    capture_stream,
                    ctx->deviceId());
            return result;
        }

        if (verify_mode)
        {
            const auto verify_result = executeCapturedReplaySegmentVerify(
                graph,
                segment,
                ctx,
                gpu_ctx,
                capture_stream,
                needs_segment_sync,
                segment_index,
                launch_dependency_cb,
                post_launch_cb);
            result.success = verify_result.success;
            result.skipped_non_idempotent = verify_result.skipped_non_idempotent;
            return result;
        }

        const bool launch_ok = executeCapturedReplaySegmentNormal(
            segment,
            gpu_ctx,
            capture_stream,
            needs_segment_sync,
            full_graph_replay,
            perf_context,
            device_name,
            launch_dependency_cb,
            post_launch_cb);
        result.success = launch_ok;
        return result;
    }

    DeviceGraphCaptureController::ReplaySegmentResult DeviceGraphCaptureController::executeReplaySegment(
        ComputeGraph &graph,
        DeviceGraphExecutor::GraphSegment &segment,
        IDeviceContext *ctx,
        IWorkerGPUContext *gpu_ctx,
        void *capture_stream,
        bool has_collective_nodes,
        bool needs_segment_sync,
        bool verify_mode,
        bool recapture_mode,
        bool require_input_preflight,
        bool full_graph_replay,
        uint64_t current_step,
        int segment_index,
        const std::string &perf_context,
        const std::function<bool(const DeviceGraphExecutor::GraphSegment &)> &cohere_inputs_cb,
        const std::function<bool(ComputeNode &)> &execute_node_cb,
        const std::function<std::unique_ptr<GraphCaptureDependencyLedger>(
            const DeviceGraphExecutor::GraphSegment &)> &plan_capture_dependencies_cb,
        const DeviceGraphExecutor::GraphCaptureBoundaryHook &capture_boundary_cb,
        IGraphCaptureAuxiliaryBranch *auxiliary_branch,
        const std::function<bool(ComputeNode &, void *)> &record_snapshot_copies_cb,
        const DeviceGraphExecutor::GraphLaunchDependencyHook &launch_dependency_cb,
        const std::function<void(DeviceGraphExecutor::GraphSegment &, void *)> &post_launch_cb)
    {
        ReplaySegmentResult result{};

        if (segment.capturable && segment.capture && segment.capture->hasExecutable())
        {
            const auto capturable_result = executeReplayCapturableSegment(
                graph,
                segment,
                ctx,
                gpu_ctx,
                capture_stream,
                needs_segment_sync,
                verify_mode,
                recapture_mode,
                require_input_preflight,
                full_graph_replay,
                segment_index,
                current_step,
                perf_context,
                cohere_inputs_cb,
                execute_node_cb,
                plan_capture_dependencies_cb,
                capture_boundary_cb,
                auxiliary_branch,
                record_snapshot_copies_cb,
                launch_dependency_cb,
                post_launch_cb);
            result.success = capturable_result.success;
            result.skipped_non_idempotent = capturable_result.skipped_non_idempotent;
            return result;
        }

        const bool manual_ok = executeManualReplaySegment(
            graph,
            segment,
            ctx,
            gpu_ctx,
            capture_stream,
            has_collective_nodes,
            needs_segment_sync,
            current_step,
            execute_node_cb,
            record_snapshot_copies_cb);
        result.success = manual_ok;
        return result;
    }

    DeviceGraphCaptureController::CapturePhaseResult DeviceGraphCaptureController::executeCapturePhase(
        ComputeGraph &graph,
        DeviceGraphExecutor::GraphSegmentCache &segment_cache,
        IDeviceContext *ctx,
        IWorkerGPUContext *gpu_ctx,
        bool has_collective_nodes,
        uint64_t current_step,
        const ReplayHooks &hooks,
        DeviceGraphExecutor::GraphInitialSubmissionPolicy initial_submission)
    {
        CapturePhaseResult result{};

        if (!ctx || !gpu_ctx)
        {
            LOG_ERROR("[DeviceGraphCaptureController] Capture phase missing context");
            return result;
        }

        if (!segment_cache.ensureCaptureStream(gpu_ctx, ctx->deviceId()))
        {
            LOG_ERROR(
                "[DeviceGraphCaptureController] Failed to create the mandatory "
                "capture stream");
            return result;
        }
        void *capture_stream = segment_cache.capture_stream;

        const bool full_graph_capture =
            captureModeForCache(segment_cache) == GraphReplayCaptureMode::FullGraph;
        const bool retained_parent_composition =
            segment_cache.graph_replay_plan_policy ==
            DeviceGraphExecutor::GraphReplayPlanPolicy::
                RequireRetainedParentComposition;
        const bool materialize_without_launch =
            initial_submission ==
            DeviceGraphExecutor::GraphInitialSubmissionPolicy::
                MaterializeWithoutLaunch;

        if (retained_parent_composition !=
            static_cast<bool>(hooks.retained_parent_composer))
        {
            LOG_ERROR(
                "[DeviceGraphCaptureController] Retained parent policy and topology composer must be selected together");
            result.reset_cache = true;
            return result;
        }
        if (retained_parent_composition &&
            std::any_of(
                segment_cache.segments.begin(),
                segment_cache.segments.end(),
                [](const DeviceGraphExecutor::GraphSegment &segment)
                {
                    return !segment.capturable;
                }))
        {
            LOG_ERROR(
                "[DeviceGraphCaptureController] Retained parent capture plan contains a manual replay unit");
            result.reset_cache = true;
            return result;
        }

        const bool heterogeneous_ticket_transaction =
            requiresHeterogeneousTicketSegmentation(
                graph.nativeCaptureEnvelope());

        /*
         * A cross-lifetime dependency has exactly one legal placement. For an
         * ordinary graph it decorates the sole complete executable. For the
         * typed heterogeneous ticket envelope it decorates only the leading
         * captured producer unit: that callback publishes the remote ticket
         * before any producer work launches, while later captured consumers
         * remain ordered through the declared manual boundary and stream
         * events. Applying it to every segment would publish one logical
         * transaction more than once.
         */
        if (hooks.launch_dependency &&
            (!retained_parent_composition &&
             !((full_graph_capture &&
                segment_cache.segments.size() == 1u &&
                segment_cache.segments.front().capturable) ||
               (heterogeneous_ticket_transaction &&
                !segment_cache.segments.empty() &&
                segment_cache.segments.front().capturable))))
        {
            LOG_ERROR(
                "[DeviceGraphCaptureController] A captured executable launch "
                "dependency requires one complete graph or the leading "
                "producer of a typed heterogeneous ticket transaction");
            result.reset_cache = true;
            return result;
        }
        if (hooks.auxiliary_branch &&
            (retained_parent_composition || !full_graph_capture ||
             segment_cache.segments.size() != 1u ||
             !segment_cache.segments.front().capturable ||
             hooks.auxiliary_branch->device() != ctx->deviceId()))
        {
            LOG_ERROR(
                "[DeviceGraphCaptureController] A graph-owned auxiliary branch requires exactly one complete capturable graph on the same device");
            result.reset_cache = true;
            return result;
        }

        /*
         * Establish all arena addresses before any stage builds persistent
         * launch descriptors. This pass is deliberately allocation-only: a
         * later segment may consume bytes produced by an intervening manual
         * heterogeneous boundary, so importing input data or producer events
         * here would violate topological execution order.
         */
        const bool has_capturable_segment = std::any_of(
            segment_cache.segments.begin(),
            segment_cache.segments.end(),
            [](const DeviceGraphExecutor::GraphSegment &segment)
            {
                return segment.capturable;
            });
        if (has_capturable_segment && !hooks.prebind_storage)
        {
            LOG_ERROR(
                "[DeviceGraphCaptureController] Native graph capture requires "
                "an allocation-only arena prebind hook");
            result.reset_cache = true;
            return result;
        }
        for (const auto &seg : segment_cache.segments)
        {
            if (seg.capturable && !hooks.prebind_storage(seg))
            {
                LOG_ERROR(
                    "[DeviceGraphCaptureController] Arena storage prebind failed "
                    "for segment starting at "
                    << (seg.stage_names.empty()
                            ? std::string("<empty>")
                            : seg.stage_names.front()));
                result.reset_cache = true;
                return result;
            }
        }

        /*
         * Launch preparation owns immutable topology setup and must complete for
         * every provisional capture unit before any native capture begins. This
         * replaces eager model execution as the source of kernel wrappers,
         * descriptor tables, pointer arrays, and transfer events. Only after
         * preparation may the stricter capture-readiness predicates be trusted.
         */
        for (const auto &seg : segment_cache.segments)
        {
            if (!seg.capturable)
                continue;
            for (const auto &stage_name : seg.stage_names)
            {
                auto *node = graph.getNode(stage_name);
                if (node && node->stage)
                    node->stage->setGPUStream(capture_stream);
            }
            if (!prepareGraphLaunchMetadata(
                    graph,
                    seg,
                    ctx,
                    capture_stream,
                    GraphLaunchPreparationPhase::Capture))
            {
                result.reset_cache = true;
                return result;
            }
        }

        for (const auto &seg : segment_cache.segments)
        {
            if (!seg.capturable)
                continue;
            for (const auto &stage_name : seg.stage_names)
            {
                auto *node = graph.getNode(stage_name);
                if (!node || !node->stage)
                    continue;
                if (!node->stage->isGraphCapturable())
                {
                    const std::string readiness =
                        node->stage->graphCaptureReadinessDebugString();
                    LOG_ERROR(
                        "[DeviceGraphCaptureController] Prepared stage '"
                        << stage_name
                        << "' is not graph-capturable after launch preparation"
                        << (readiness.empty() ? "" : "; ")
                        << readiness
                        << "; mandatory capture cannot proceed");
                    PerfStatsCollector::addCounter(
                        "forward_graph",
                        "launch_preparation_capture_not_ready",
                        1.0,
                        "decode",
                        ctx ? ctx->deviceId().toString() : std::string{},
                        {{"stage", stage_name},
                         {"stage_type", computeStageTypeName(node->stage->type())},
                         {"preparation_dependent",
                          node->stage->supportsGraphCaptureAfterLaunchPreparation()
                              ? "true"
                              : "false"},
                         {"context", segment_cache.perf_context},
                         {"readiness", readiness.empty() ? std::string("unspecified") : readiness}});
                    result.reset_cache = true;
                    return result;
                }
            }
        }

        for (size_t segment_index = 0; segment_index < segment_cache.segments.size(); ++segment_index)
        {
            auto &seg = segment_cache.segments[segment_index];
            if (seg.capturable)
            {
                // Capturable path: prepared stream -> begin capture -> record
                // nodes -> end capture -> instantiate and launch transaction zero.

                /*
                 * Establish the complete arena frontier before asking stages for
                 * snapshot descriptors. A descriptor embeds the producer's final
                 * device pointer; preparing it while an arena output is still
                 * unbound records either a null pointer or a stale allocation.
                 * The same frontier pass joins every launch-preparation producer
                 * event to the exact capture stream, so recording cannot discover
                 * an external dependency inside beginCapture().
                 */
                if (!hooks.cohere_inputs)
                {
                    LOG_ERROR("[DeviceGraphCaptureController] Native graph capture requires a pre-capture input coherence hook for segment starting at "
                              << (seg.stage_names.empty()
                                      ? std::string("<empty>")
                                      : seg.stage_names.front()));
                    result.reset_cache = true;
                    result.success = false;
                    return result;
                }
                if (!hooks.cohere_inputs(seg))
                {
                    LOG_ERROR("[DeviceGraphCaptureController] Native graph capture input preflight failed for segment starting at "
                              << (seg.stage_names.empty()
                                      ? std::string("<empty>")
                                      : seg.stage_names.front()));
                    result.reset_cache = true;
                    result.success = false;
                    return result;
                }

                /*
                 * Snapshot copy nodes need their per-stage output descriptors and
                 * destination storage allocated before stream capture begins.
                 * Preparation is descriptor-only: it must not copy payload bytes,
                 * enqueue stream work, or publish a producer event. The exact
                 * point-in-time D2D copy is recorded only after its producing
                 * stage executes inside the graph transaction.
                 */
                if (hooks.prepare_snapshot_copies)
                {
                    for (const auto &stage_name : seg.stage_names)
                    {
                        auto *node = graph.getNode(stage_name);
                        if (!node || !node->stage)
                            continue;
                        if (!hooks.prepare_snapshot_copies(*node, capture_stream))
                        {
                            LOG_ERROR("[DeviceGraphCaptureController] Snapshot descriptor preparation failed before cached graph capture: "
                                      << stage_name);
                            result.reset_cache = true;
                            result.success = false;
                            return result;
                        }
                    }
                }

                /*
                 * Launch preparation and native capture share one exact stream.
                 * The stream queue itself orders setup writes before the capture
                 * boundary; waiting for a host-visible event here would insert
                 * an intermediate CPU rendezvous into transaction zero. CUDA
                 * and HIP begin capture after the stream's previously submitted
                 * work without requiring a host synchronization.
                 */

                if (hooks.capture_boundary)
                {
                    const std::string boundary_name =
                        graphCaptureBoundaryName(
                            "capture_begin",
                            current_step,
                            seg.capture_wave_ordinal,
                            seg.capture_wave_identity,
                            &seg,
                            segment_cache.perf_context);
                    if (!hooks.capture_boundary(boundary_name, capture_stream))
                    {
                        LOG_ERROR("[DeviceGraphCaptureController] Capture begin boundary rendezvous failed before segment starting at "
                                  << (seg.stage_names.empty() ? std::string("<empty>") : seg.stage_names.front())
                                  << " boundary=" << boundary_name);
                        result.reset_cache = true;
                        result.success = false;
                        return result;
                    }
                }

                seg.capture = gpu_ctx->createGraphCapture(capture_stream);
                if (!seg.capture)
                {
                    LOG_ERROR("[DeviceGraphCaptureController] Failed to create graph capture for segment");
                    result.reset_cache = true;
                    return result;
                }

                if (!hooks.execute_node)
                {
                    LOG_ERROR(
                        "[DeviceGraphCaptureController] Native graph capture "
                        "requires the canonical executor stage hook");
                    result.reset_cache = true;
                    result.success = false;
                    return result;
                }

                auto dependency_ledger = planCaptureDependencies(
                    graph,
                    seg,
                    ctx->deviceId(),
                    capture_stream,
                    hooks.plan_capture_dependencies,
                    "capture_phase_segment_" + std::to_string(segment_index));
                if (!dependency_ledger)
                {
                    result.reset_cache = true;
                    result.success = false;
                    return result;
                }

                bool exec_ok = true;
                std::string failed_stage_name;
                {
                    ScopedBackendGraphCapture capture_transaction(
                        *gpu_ctx,
                        *seg.capture,
                        "capture phase segment=" +
                            std::to_string(segment_index) +
                            " stages=" + describeSegmentStages(seg),
                        dependency_ledger.get());
                    if (!capture_transaction.begin())
                    {
                        LOG_ERROR("[DeviceGraphCaptureController] beginCapture failed for segment");
                        result.reset_cache = true;
                        return result;
                    }

                    ScopedCapturedAuxiliaryBranch auxiliary_branch(
                        hooks.auxiliary_branch,
                        capture_stream);
                    if (!auxiliary_branch.begin())
                    {
                        LOG_ERROR(
                            "[DeviceGraphCaptureController] Could not record the graph-owned auxiliary fork for segment starting at "
                            << (seg.stage_names.empty()
                                    ? std::string("<empty>")
                                    : seg.stage_names.front()));
                        exec_ok = false;
                        failed_stage_name = "<auxiliary_branch_fork>";
                    }

                    for (const auto &stage_name : seg.stage_names)
                    {
                        if (!exec_ok)
                            break;
                        auto *node = graph.getNode(stage_name);
                        if (!node || !node->stage || !hooks.execute_node(*node))
                        {
                            LOG_ERROR("[DeviceGraphCaptureController] Stage failed during cached graph capture: " << stage_name);
                            exec_ok = false;
                            failed_stage_name = stage_name;
                            break;
                        }
                        if (hooks.record_snapshot_copies &&
                            !hooks.record_snapshot_copies(*node, capture_stream))
                        {
                            LOG_ERROR("[DeviceGraphCaptureController] Snapshot copy failed during cached graph capture: "
                                      << stage_name);
                            exec_ok = false;
                            failed_stage_name = stage_name;
                            break;
                        }
                        graph.markCompleted(stage_name);
                    }

                    if (!auxiliary_branch.finish())
                    {
                        LOG_ERROR(
                            "[DeviceGraphCaptureController] Could not record the graph-owned auxiliary join for segment starting at "
                            << (seg.stage_names.empty()
                                    ? std::string("<empty>")
                                    : seg.stage_names.front()));
                        if (exec_ok)
                        {
                            exec_ok = false;
                            failed_stage_name =
                                "<auxiliary_branch_join>";
                        }
                    }

                    capture_transaction.finish();
                }

                /*
                 * A begin-only barrier is insufficient: participants can take
                 * different amounts of time to record a segment. Keep every
                 * participant out of following manual collectives until all
                 * siblings have left native stream-capture mode.
                 */
                if (hooks.capture_boundary)
                {
                    const std::string boundary_name =
                        graphCaptureBoundaryName(
                            exec_ok ? "capture_end" : "capture_end_failed",
                            current_step,
                            seg.capture_wave_ordinal,
                            seg.capture_wave_identity,
                            &seg,
                            segment_cache.perf_context);
                    if (!hooks.capture_boundary(boundary_name, capture_stream))
                    {
                        LOG_ERROR("[DeviceGraphCaptureController] Capture end boundary rendezvous failed after segment starting at "
                                  << (seg.stage_names.empty() ? std::string("<empty>") : seg.stage_names.front())
                                  << " boundary=" << boundary_name);
                        result.reset_cache = true;
                        result.success = false;
                        return result;
                    }
                }

                if (!exec_ok)
                {
                    LOG_ERROR(
                        "[DeviceGraphCaptureController] Mandatory cached graph "
                        "capture failed; refusing eager recovery"
                        << " segment=" << segment_index
                        << " stage="
                        << (failed_stage_name.empty()
                                ? std::string("<unknown>")
                                : failed_stage_name));
                    result.reset_cache = true;
                    result.success = false;
                    return result;
                }

                const CapturedUnitFinalization finalization =
                    retained_parent_composition
                        ? CapturedUnitFinalization::RetainGraphOnly
                        : (materialize_without_launch
                               ? CapturedUnitFinalization::
                                     InstantiateWithoutLaunch
                               : CapturedUnitFinalization::
                                     InstantiateAndLaunch);
                const bool capture_finalize_ok = finalizeCapturePhaseCapturableSegment(
                    graph,
                    seg,
                    ctx,
                    gpu_ctx,
                    capture_stream,
                    full_graph_capture,
                    segment_cache.perf_context,
                    current_step,
                    retained_parent_composition ||
                            (heterogeneous_ticket_transaction &&
                             segment_index != 0u)
                        ? DeviceGraphExecutor::GraphLaunchDependencyHook{}
                        : hooks.launch_dependency,
                    hooks.post_launch,
                    finalization);
                if (!capture_finalize_ok)
                {
                    result.reset_cache = true;
                    return result;
                }
                /*
                 * Ordinary segmented execution launches its unit before joining
                 * following passive waves. Parent composition deliberately does
                 * not: every sibling first finishes the same graph-only capture
                 * wave, and only the fully composed endpoint graph may launch.
                 */
                if (!joinPassiveCaptureWaves(
                        graph,
                        seg,
                        "capture",
                        current_step,
                        segment_cache.perf_context,
                        hooks.capture_boundary,
                        capture_stream,
                        ctx->deviceId()))
                {
                    result.reset_cache = true;
                    return result;
                }
            }
            else
            {
                /*
                 * Manual segments are declarative members of the segmented
                 * plan, not a recovery path. They execute exactly once in the
                 * same position during capture and replay. Setup-only native
                 * materialization is the one intentional exception: it records
                 * every neighbouring executable but cannot submit manual model
                 * arithmetic before request admission. The first ordinary replay
                 * executes this unit in its declared position.
                 */
                if (!materialize_without_launch)
                {
                    awaitManualHostTicketBoundary(
                        graph,
                        segment_cache,
                        segment_index,
                        "capture",
                        ctx->deviceId());
                    const bool manual_capture_ok =
                        executeCapturePhaseManualSegment(
                            graph,
                            seg,
                            ctx,
                            gpu_ctx,
                            capture_stream,
                            has_collective_nodes,
                            current_step,
                            hooks.execute_node,
                            hooks.record_snapshot_copies);
                    if (!manual_capture_ok)
                    {
                        result.reset_cache = true;
                        return result;
                    }
                }
                if (!joinPassiveCaptureWaves(
                        graph,
                        seg,
                        "capture",
                        current_step,
                        segment_cache.perf_context,
                        hooks.capture_boundary,
                        capture_stream,
                        ctx->deviceId()))
                {
                    result.reset_cache = true;
                    return result;
                }
            }
        }

        if (retained_parent_composition &&
            !finalizeRetainedParentTransaction(
                graph,
                segment_cache,
                ctx,
                gpu_ctx,
                current_step,
                hooks,
                initial_submission))
        {
            result.reset_cache = true;
            return result;
        }

        result.success = true;
        return result;
    }

    DeviceGraphCaptureController::ReplayPhaseResult DeviceGraphCaptureController::executeReplayPhase(
        ComputeGraph &graph,
        DeviceGraphExecutor::GraphSegmentCache &segment_cache,
        IDeviceContext *ctx,
        IWorkerGPUContext *gpu_ctx,
        bool has_collective_nodes,
        bool collectives_graph_capturable,
        uint64_t current_step,
        const ReplayHooks &hooks,
        bool force_recapture,
        bool defer_final_sync)
    {
        ReplayPhaseResult result{};

        if (!ctx || !gpu_ctx)
        {
            LOG_ERROR("[DeviceGraphCaptureController] Replay phase missing context");
            return result;
        }

        // Reset thread-local replay profiling state for this iteration
        const bool profiling =
            KernelProfiler::isEnabled() ||
            PerfStatsCollector::isDomainEnabled("forward_pass");
        if (profiling)
        {
            ForwardPassProfiler::resetReplayTimings();
        }

        void *capture_stream = segment_cache.capture_stream;
        const auto &exec_cfg = debugEnv().execution;
        const bool verify_mode = exec_cfg.gpu_graph_verify;
        const bool recapture_mode = exec_cfg.gpu_graph_recapture || force_recapture;
        const bool stream_only_mode = exec_cfg.gpu_graph_stream_only;
        const bool stream_only_default = exec_cfg.gpu_graph_stream_only_default;
        const GraphReplayCaptureMode replay_mode = captureModeForCache(segment_cache);
        const bool full_graph_replay = replay_mode == GraphReplayCaptureMode::FullGraph;
        const bool heterogeneous_ticket_transaction =
            requiresHeterogeneousTicketSegmentation(
                graph.nativeCaptureEnvelope());
        if (hooks.launch_dependency &&
            !((full_graph_replay &&
               segment_cache.segments.size() == 1u &&
               segment_cache.segments.front().capturable) ||
              (heterogeneous_ticket_transaction &&
               !segment_cache.segments.empty() &&
               segment_cache.segments.front().capturable)))
        {
            LOG_ERROR(
                "[DeviceGraphCaptureController] A captured executable launch "
                "dependency cannot decorate this replay envelope");
            return result;
        }
        if (stream_only_mode)
        {
            if (hooks.launch_dependency)
            {
                LOG_ERROR(
                    "[DeviceGraphCaptureController] Stream-only diagnostics "
                    "cannot satisfy a captured executable launch dependency");
                return result;
            }
            // Diagnostics mode: avoid graph launch and just run stages on the
            // selected stream to isolate stream-related issues.
            result.success = executeStreamOnlyReplay(
                graph,
                segment_cache,
                ctx,
                gpu_ctx,
                capture_stream,
                stream_only_default);
            return result;
        }
        // Production segments are ordered entirely on-device. Verification is
        // the sole mode that needs intermediate host visibility for comparison.
        const bool needs_segment_sync = verify_mode;

        const bool captured_collectives_can_defer_final_sync =
            has_collective_nodes &&
            collectives_graph_capturable;
        const bool collective_sync_requires_eager_wait =
            has_collective_nodes && !captured_collectives_can_defer_final_sync;
        const bool can_defer_final_sync =
            defer_final_sync &&
            !collective_sync_requires_eager_wait &&
            std::all_of(segment_cache.segments.begin(),
                        segment_cache.segments.end(),
                        [](const DeviceGraphExecutor::GraphSegment &segment)
                        {
                            return segment.capturable;
                        });
        const bool trace_replay = exec_cfg.gpu_graph_trace_replay;
        const bool forward_graph_stats_enabled =
            PerfStatsCollector::isDomainEnabled("forward_graph");
        const auto &device_id = ctx->deviceId();
        const std::string device_name =
            forward_graph_stats_enabled || trace_replay
                ? device_id.toString()
                : std::string{};
        const int total_segments = static_cast<int>(segment_cache.segments.size());
        std::optional<PerfStatsCollector::ScopedTimer> replay_timer;
        if (forward_graph_stats_enabled)
        {
            auto replay_total_tags = replayCacheTags(
                segment_cache,
                replay_mode);
            replay_total_tags.emplace(
                "sync_scope",
                can_defer_final_sync
                    ? "launch_only_deferred"
                    : "stream_synchronized");
            replay_timer.emplace(
                "forward_graph",
                replayMetricName(replay_mode, "total"),
                "decode",
                device_name,
                graphReplayHostTimingTags(
                    std::move(replay_total_tags),
                    "total_replay_host_wall",
                    replay_mode));
        }

        size_t total_replay_timing_slot = kInvalidReplayGpuTimingSlot;
        if (!beginReplayGpuTiming(
                segment_cache,
                gpu_ctx,
                DeviceGraphExecutor::GraphSegmentCache::ReplayGpuTimingSlot::Scope::TotalReplay,
                /*replay_unit_index=*/0,
                can_defer_final_sync,
                total_replay_timing_slot))
        {
            return result;
        }

        int seg_idx = 0;
        for (auto &seg : segment_cache.segments)
        {
            PerfStatsCollector::Tags seg_tags;
            if (forward_graph_stats_enabled)
            {
                seg_tags = replaySegmentTags(
                    seg,
                    segment_cache.perf_context,
                    &segment_cache.replay_workload);
            }
            if (trace_replay)
            {
                const char *seg_display_type = seg.capturable ? "GRAPH" : "MANUAL";
                const auto &first_name = seg.stage_names.empty() ? std::string("<empty>") : seg.stage_names.front();
                LOG_DEBUG("[ReplayTrace] " << device_id.toString()
                                           << " step=" << current_step
                                           << " seg=" << seg_idx << "/" << total_segments
                                           << " [" << seg_display_type << "]"
                                           << " stages=" << seg.stage_names.size()
                                           << " first=" << first_name);
            }

            if (forward_graph_stats_enabled)
            {
                PerfStatsCollector::addCounter(
                    "forward_graph",
                    replayUnitCounterName(replay_mode),
                    1.0,
                    "decode",
                    device_name,
                    seg_tags);
            }

            size_t replay_unit_timing_slot = kInvalidReplayGpuTimingSlot;
            if (!full_graph_replay &&
                !beginReplayGpuTiming(
                    segment_cache,
                    gpu_ctx,
                    DeviceGraphExecutor::GraphSegmentCache::ReplayGpuTimingSlot::Scope::ReplayUnit,
                    static_cast<size_t>(seg_idx),
                    can_defer_final_sync,
                    replay_unit_timing_slot))
            {
                (void)finishReplayGpuTiming(
                    segment_cache,
                    gpu_ctx,
                    total_replay_timing_slot);
                return result;
            }

            const auto segment_t0 = std::chrono::high_resolution_clock::now();
            uint64_t manual_ticket_wait_ns = 0;
            if (!seg.capturable)
            {
                const auto ticket_wait_t0 =
                    std::chrono::high_resolution_clock::now();
                awaitManualHostTicketBoundary(
                    graph,
                    segment_cache,
                    static_cast<size_t>(seg_idx),
                    "replay",
                    ctx->deviceId());
                if (profiling)
                {
                    const auto ticket_wait_t1 =
                        std::chrono::high_resolution_clock::now();
                    manual_ticket_wait_ns = static_cast<uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            ticket_wait_t1 - ticket_wait_t0)
                            .count());
                    ForwardPassProfiler::addReplayManualHostTicketWaitNs(
                        manual_ticket_wait_ns);
                }
            }
            // Segment execution picks capturable or manual behavior based on
            // segment metadata prepared during capture setup.
            const DeviceGraphExecutor::GraphLaunchDependencyHook
                segment_launch_dependency =
                    heterogeneous_ticket_transaction && seg_idx != 0
                        ? DeviceGraphExecutor::GraphLaunchDependencyHook{}
                        : hooks.launch_dependency;
            const auto replay_result = executeReplaySegment(
                graph,
                seg,
                ctx,
                gpu_ctx,
                capture_stream,
                has_collective_nodes,
                needs_segment_sync,
                verify_mode,
                recapture_mode,
                hooks.require_replay_input_preflight,
                full_graph_replay,
                current_step,
                seg_idx,
                segment_cache.perf_context,
                hooks.cohere_inputs,
                hooks.execute_node,
                hooks.plan_capture_dependencies,
                hooks.capture_boundary,
                hooks.auxiliary_branch,
                hooks.record_snapshot_copies,
                segment_launch_dependency,
                hooks.post_launch);
            if (profiling && !seg.capturable)
            {
                const auto manual_segment_t1 =
                    std::chrono::high_resolution_clock::now();
                const uint64_t manual_segment_total_ns =
                    static_cast<uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            manual_segment_t1 - segment_t0)
                            .count());
                const uint64_t manual_execution_ns =
                    manual_segment_total_ns >= manual_ticket_wait_ns
                        ? manual_segment_total_ns - manual_ticket_wait_ns
                        : 0;

                bool has_dispatch_descriptor = false;
                bool has_sparse_rank_protocol = false;
                for (const auto &stage_name : seg.stage_names)
                {
                    const auto *node = graph.getNode(stage_name);
                    if (!node || !node->stage)
                        continue;
                    switch (node->stage->type())
                    {
                    case ComputeStageType::MOE_EXPERT_DISPATCH:
                        has_dispatch_descriptor = true;
                        break;
                    case ComputeStageType::MOE_RANK_BATCH_DISPATCH:
                    case ComputeStageType::MOE_RANK_BATCH_RETURN_REDUCE:
                        has_sparse_rank_protocol = true;
                        break;
                    default:
                        break;
                    }
                }

                if (has_sparse_rank_protocol)
                {
                    ForwardPassProfiler::addReplayManualSparseProtocolNs(
                        manual_execution_ns);
                }
                else if (has_dispatch_descriptor)
                {
                    ForwardPassProfiler::addReplayManualDispatchNs(
                        manual_execution_ns);
                }
                else
                {
                    ForwardPassProfiler::addReplayManualOtherNs(
                        manual_execution_ns);
                }
            }
            if (forward_graph_stats_enabled)
            {
                const auto segment_t1 = std::chrono::high_resolution_clock::now();
                const auto segment_ns =
                    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(segment_t1 - segment_t0).count());
                PerfStatsCollector::recordTimingNs(
                    "forward_graph",
                    replayUnitTimingName(replay_mode),
                    segment_ns,
                    "decode",
                    device_name,
                    graphReplayHostTimingTags(seg_tags,
                                               full_graph_replay
                                                   ? "graph_replay_host_wall"
                                                   : "segment_replay_host_wall",
                                               replay_mode));
            }

            if (!replay_result.success)
            {
                (void)finishReplayGpuTiming(
                    segment_cache,
                    gpu_ctx,
                    replay_unit_timing_slot);
                (void)finishReplayGpuTiming(
                    segment_cache,
                    gpu_ctx,
                    total_replay_timing_slot);
                return result;
            }

            if (recapture_mode && !seg.capturable &&
                !joinPassiveCaptureWaves(
                    graph,
                    seg,
                    "recapture",
                    current_step,
                    segment_cache.perf_context,
                    hooks.capture_boundary,
                    capture_stream,
                    ctx->deviceId()))
            {
                (void)finishReplayGpuTiming(
                    segment_cache,
                    gpu_ctx,
                    replay_unit_timing_slot);
                (void)finishReplayGpuTiming(
                    segment_cache,
                    gpu_ctx,
                    total_replay_timing_slot);
                return result;
            }

            if (!finishReplayGpuTiming(
                    segment_cache,
                    gpu_ctx,
                    replay_unit_timing_slot))
            {
                (void)finishReplayGpuTiming(
                    segment_cache,
                    gpu_ctx,
                    total_replay_timing_slot);
                return result;
            }

            if (trace_replay)
            {
                LOG_DEBUG("[ReplayTrace] " << device_id.toString()
                                           << " step=" << current_step
                                           << " seg=" << seg_idx << "/" << total_segments << " DONE");
            }

            seg_idx++;
        }

        if (!finishReplayGpuTiming(
                segment_cache,
                gpu_ctx,
                total_replay_timing_slot))
        {
            return result;
        }

        if (trace_replay)
        {
            LOG_DEBUG("[ReplayTrace] " << device_id.toString()
                                       << " step=" << current_step
                                       << " ALL " << total_segments
                                       << " segments done, entering final capture-event ownership fence");
        }
        // A caller may explicitly defer this fence when it immediately enqueues
        // a dependent GPU operation. Every manual collective publishes an event
        // back to capture_stream, so one capture-stream completion event covers
        // the complete replay DAG, including work submitted on collective streams.
        if (can_defer_final_sync)
        {
            if (forward_graph_stats_enabled)
            {
                PerfStatsCollector::addCounter(
                    "forward_graph",
                    replayMetricName(replay_mode, "final_sync_deferred"),
                    1.0,
                    "decode",
                    device_name,
                    replayCacheTags(segment_cache, replay_mode));
            }
            /*
             * Timing must preserve the exact same asynchronous graph topology as
             * an unprofiled launch. Opportunistically retire older completed
             * slots, but never wait for this replay's stop event on the host.
             */
            if (!reclaimReplayGpuTimingNonblocking(segment_cache, gpu_ctx))
                return result;
            result.success = true;
            return result;
        }
        {
            auto sync_t0 = std::chrono::high_resolution_clock::now();
            segment_cache.waitForCaptureStreamFence();
            auto sync_t1 = std::chrono::high_resolution_clock::now();
            if (profiling)
            {
                ForwardPassProfiler::addReplayStreamSyncNs(
                    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(sync_t1 - sync_t0).count()));
            }
            if (forward_graph_stats_enabled)
            {
                auto capture_tags = replayCacheTags(segment_cache, replay_mode);
                capture_tags.emplace("stream", "capture_event");
                PerfStatsCollector::recordTimingNs(
                    "forward_graph",
                    replayMetricName(replay_mode, "stream_sync"),
                    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(sync_t1 - sync_t0).count()),
                    "decode",
                    device_name,
                    std::move(capture_tags));
                PerfStatsCollector::recordTimingNs(
                    "forward_graph",
                    replayMetricName(replay_mode, "final_sync"),
                    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(sync_t1 - sync_t0).count()),
                    "decode",
                    device_name,
                    graphReplayHostTimingTags(replayCacheTags(segment_cache, replay_mode),
                                               "final_stream_sync_host_wall",
                                               replay_mode));
            }
        }
        if (!reclaimReplayGpuTimingNonblocking(segment_cache, gpu_ctx))
            return result;
        if (PerfStatsCollector::gpuStageEventTimingEnabled() &&
            std::any_of(
                segment_cache.replay_gpu_timing_slots.begin(),
                segment_cache.replay_gpu_timing_slots.end(),
                [](const auto &slot)
                {
                    return slot.pending;
                }))
        {
            LOG_ERROR(
                "[DeviceGraphCaptureController] Capture-stream ownership fence "
                "completed but a replay GPU timing event remained pending");
            return result;
        }
        if (trace_replay)
        {
            LOG_DEBUG("[ReplayTrace] " << device_id.toString()
                                       << " step=" << current_step
                                       << " final capture-event ownership fence complete");
        }
        result.success = true;
        return result;
    }

    bool DeviceGraphCaptureController::cohereReplaySegmentInputs(
        ComputeGraph &graph,
        const DeviceGraphExecutor::GraphSegment &segment,
        const std::function<bool(
            ComputeNode &,
            const std::vector<BufferBinding> &)> &cohere_stage_cb)
    {
        GraphArenaDependencyTracker dependency_tracker;
        for (const auto &stage_name : segment.stage_names)
        {
            auto *node = graph.getNode(stage_name);
            if (!node || !node->stage)
            {
                continue;
            }

            const StageBufferContract contract = node->stage->bufferContract();
            const auto external_reads =
                dependency_tracker.observeStage(contract);
            if (!cohere_stage_cb(*node, external_reads))
            {
                return false;
            }
        }

        return true;
    }

    void DeviceGraphCaptureController::cacheCapturedSegmentArenaWrites(
        ComputeGraph &graph,
        DeviceGraphExecutor::GraphSegment &segment)
    {
        if (segment.arena_writes_cached)
            return;

        segment.cached_arena_writes.clear();

        auto add_write = [&](BufferId id, DeviceId device)
        {
            for (const auto &existing : segment.cached_arena_writes)
            {
                if (existing.id == id && existing.device == device)
                {
                    return;
                }
            }
            segment.cached_arena_writes.push_back({id, device});
        };

        for (const auto &stage_name : segment.stage_names)
        {
            auto *node = graph.getNode(stage_name);
            if (!node || !node->stage)
            {
                continue;
            }

            const StageBufferContract contract = node->stage->bufferContract();
            if (contract.empty())
            {
                continue;
            }

            const DeviceId target_device =
                node->device.is_valid() ? node->device : node->stage->device();
            for (const auto &binding : contract.outputs)
            {
                add_write(binding.id, target_device);
            }
            for (const auto &binding : contract.inouts)
            {
                add_write(binding.id, target_device);
            }
        }

        segment.arena_writes_cached = true;
        LOG_DEBUG("[DeviceGraphCaptureController] Cached "
                  << segment.cached_arena_writes.size()
                  << " unique arena writes for " << segment.stage_names.size()
                  << " replay stages");
    }

    void DeviceGraphCaptureController::postCapturedSegmentLaunch(
        ComputeGraph &graph,
        DeviceGraphExecutor::GraphSegment &segment,
        uint64_t current_step,
        void * /*stream*/,
        const std::function<void(BufferId, DeviceId)> &mark_arena_write_dirty_cb)
    {
        cacheCapturedSegmentArenaWrites(graph, segment);

        // Use arena ids instead of retaining raw TensorBase* pointers. Prefix
        // restore, rollback, and request clears may preserve graph topology
        // while replacing or rewinding tensor-backed state.
        for (const auto &write : segment.cached_arena_writes)
        {
            mark_arena_write_dirty_cb(write.id, write.device);
        }

        segment.last_executed_step = current_step;
    }

} // namespace llaminar2
