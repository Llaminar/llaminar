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
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
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

        std::string graphCaptureBoundaryName(
            const char *phase,
            uint64_t current_step,
            int segment_index,
            const DeviceGraphExecutor::GraphSegment &segment,
            const std::string &perf_context)
        {
            const std::string first_stage =
                segment.stage_names.empty() ? std::string("<empty>") : segment.stage_names.front();
            const std::string last_stage =
                segment.stage_names.empty() ? std::string("<empty>") : segment.stage_names.back();
            std::ostringstream out;
            out << "before_begin:"
                << "phase=" << phase
                << ":step=" << current_step
                << ":segment=" << segment_index
                << ":type=" << segmentTypeName(segment)
                << ":first=" << first_stage
                << ":last=" << last_stage;
            if (!perf_context.empty())
                out << ":context=" << perf_context;
            return out.str();
        }

        void addContextTag(PerfStatsCollector::Tags &tags, const std::string &perf_context)
        {
            if (!perf_context.empty())
                tags.emplace("context", perf_context);
        }

        PerfStatsCollector::Tags replaySegmentTags(const DeviceGraphExecutor::GraphSegment &segment,
                                                   const std::string &perf_context)
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

        class ReplayGpuEventTimer
        {
        public:
            ReplayGpuEventTimer(IWorkerGPUContext *gpu_ctx,
                                void *stream,
                                std::string name,
                                std::string phase,
                                std::string device,
                                PerfStatsCollector::Tags tags)
                : gpu_ctx_(gpu_ctx),
                  stream_(stream),
                  name_(std::move(name)),
                  phase_(std::move(phase)),
                  device_(std::move(device)),
                  tags_(std::move(tags))
            {
                if (!PerfStatsCollector::gpuStageEventTimingEnabled() ||
                    !gpu_ctx_ ||
                    !stream_)
                {
                    return;
                }

                start_event_ = gpu_ctx_->createEvent();
                stop_event_ = gpu_ctx_->createEvent();
                if (!start_event_ || !stop_event_)
                {
                    destroyEvents();
                    return;
                }

                gpu_ctx_->recordEvent(start_event_, stream_);
                active_ = true;
            }

            ~ReplayGpuEventTimer()
            {
                destroyEvents();
            }

            ReplayGpuEventTimer(const ReplayGpuEventTimer &) = delete;
            ReplayGpuEventTimer &operator=(const ReplayGpuEventTimer &) = delete;

            ReplayGpuEventTimer(ReplayGpuEventTimer &&other) noexcept
                : gpu_ctx_(other.gpu_ctx_),
                  stream_(other.stream_),
                  start_event_(other.start_event_),
                  stop_event_(other.stop_event_),
                  active_(other.active_),
                  stopped_(other.stopped_),
                  recorded_(other.recorded_),
                  name_(std::move(other.name_)),
                  phase_(std::move(other.phase_)),
                  device_(std::move(other.device_)),
                  tags_(std::move(other.tags_))
            {
                other.gpu_ctx_ = nullptr;
                other.stream_ = nullptr;
                other.start_event_ = nullptr;
                other.stop_event_ = nullptr;
                other.active_ = false;
                other.stopped_ = false;
                other.recorded_ = true;
            }

            ReplayGpuEventTimer &operator=(ReplayGpuEventTimer &&other) noexcept
            {
                if (this != &other)
                {
                    destroyEvents();
                    gpu_ctx_ = other.gpu_ctx_;
                    stream_ = other.stream_;
                    start_event_ = other.start_event_;
                    stop_event_ = other.stop_event_;
                    active_ = other.active_;
                    stopped_ = other.stopped_;
                    recorded_ = other.recorded_;
                    name_ = std::move(other.name_);
                    phase_ = std::move(other.phase_);
                    device_ = std::move(other.device_);
                    tags_ = std::move(other.tags_);

                    other.gpu_ctx_ = nullptr;
                    other.stream_ = nullptr;
                    other.start_event_ = nullptr;
                    other.stop_event_ = nullptr;
                    other.active_ = false;
                    other.stopped_ = false;
                    other.recorded_ = true;
                }
                return *this;
            }

            bool active() const { return active_; }

            void stop()
            {
                if (!active_ || stopped_)
                    return;
                gpu_ctx_->recordEvent(stop_event_, stream_);
                stopped_ = true;
            }

            void record(bool synchronize_stop_event)
            {
                if (!active_ || !stopped_ || recorded_)
                    return;

                if (synchronize_stop_event)
                    gpu_ctx_->synchronizeEvent(stop_event_);

                const float elapsed_ms = gpu_ctx_->eventElapsedTime(start_event_, stop_event_);
                if (elapsed_ms >= 0.0f)
                {
                    PerfStatsCollector::recordTimingNs(
                        "stage_gpu",
                        name_,
                        static_cast<uint64_t>(static_cast<double>(elapsed_ms) * 1.0e6),
                        phase_,
                        device_,
                        tags_);
                }
                recorded_ = true;
            }

        private:
            void destroyEvents()
            {
                if (gpu_ctx_)
                {
                    if (start_event_)
                        gpu_ctx_->destroyEvent(start_event_);
                    if (stop_event_)
                        gpu_ctx_->destroyEvent(stop_event_);
                }
                start_event_ = nullptr;
                stop_event_ = nullptr;
                active_ = false;
            }

            IWorkerGPUContext *gpu_ctx_ = nullptr;
            void *stream_ = nullptr;
            void *start_event_ = nullptr;
            void *stop_event_ = nullptr;
            bool active_ = false;
            bool stopped_ = false;
            bool recorded_ = false;
            std::string name_;
            std::string phase_;
            std::string device_;
            PerfStatsCollector::Tags tags_;
        };
    }


    const char *DeviceGraphCaptureController::phaseName(Phase phase)
    {
        switch (phase)
        {
        case Phase::Warmup:
            return "warmup";
        case Phase::Capture:
            return "capture";
        case Phase::Replay:
            return "replay";
        }

        throw std::logic_error("Unknown GPU graph replay phase");
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
            return {Phase::Warmup, decode_step};
        }

        if (needs_capture)
        {
            needs_capture = false;
            return {Phase::Capture, decode_step};
        }

        return {Phase::Replay, decode_step};
    }

    void DeviceGraphCaptureController::markWarmupComplete(bool &initialized, bool &needs_capture)
    {
        initialized = true;
        needs_capture = true;
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

    void DeviceGraphCaptureController::executeWarmupPhase(
        ComputeGraph &graph,
        DeviceGraphExecutor::GraphSegmentCache &segment_cache,
        const std::unordered_set<std::string> *collective_nodes,
        bool has_collective_nodes,
        bool collectives_graph_capturable,
        DeviceGraphExecutor::GraphReplayPlanPolicy plan_policy)
    {
        buildWarmupSegments(
            graph,
            segment_cache,
            collective_nodes,
            has_collective_nodes,
            collectives_graph_capturable,
            plan_policy);
        markWarmupComplete(
            segment_cache.initialized,
            segment_cache.needs_capture);
    }

    void DeviceGraphCaptureController::buildWarmupSegments(
        ComputeGraph &graph,
        DeviceGraphExecutor::GraphSegmentCache &segment_cache,
        const std::unordered_set<std::string> *collective_nodes,
        bool has_collective_nodes,
        bool collectives_graph_capturable,
        DeviceGraphExecutor::GraphReplayPlanPolicy plan_policy)
    {
        segment_cache.segments.clear();

        const auto &order = graph.getExecutionOrder();
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

            // Start from stage capability, then layer on safety gates.
            bool stage_capturable = node->stage->isGraphCapturable();
            const bool warmup_dependent_capture =
                !stage_capturable && node->stage->supportsWarmupDependentGraphCapture();
            if (warmup_dependent_capture)
            {
                stage_capturable = true;
            }
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

            if (has_collective_nodes && !segmented_collective_capture_allow.empty())
            {
                // Explicit allowlist mode: only allowlisted stages are capturable
                stage_capturable = stage_in_collective_allowlist(name);
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

        if (PerfStatsCollector::isEnabled())
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
            const bool heterogeneous_collective_segmentation_admitted =
                plan_policy ==
                    DeviceGraphExecutor::GraphReplayPlanPolicy::
                        AllowHeterogeneousCollectiveSegmentation &&
                has_collective_nodes;

            if (!heterogeneous_collective_segmentation_admitted)
            {
                std::ostringstream detail;
                detail
                    << "GPU replay planning produced "
                    << capturable_segments << " capturable segment(s) and "
                    << manual_segments << " manual segment(s), but this "
                       "execution domain requires one fully captured graph";
                if (!has_collective_nodes)
                {
                    detail
                        << "; segmented execution is never admitted for a "
                           "graph without collectives";
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

                LOG_ERROR("[DeviceGraphCaptureController] " << detail.str());
                throw std::runtime_error(detail.str());
            }
        }

        for (auto &seg : segment_cache.segments)
        {
            seg.last_executed_step = 0;
        }
    }

    void DeviceGraphCaptureController::initializeReplayCallbacks(
        ComputeGraph &graph,
        DeviceGraphExecutor::GraphSegmentCache &segment_cache)
    {
        for (auto &seg : segment_cache.segments)
        {
            seg.replay_callbacks.clear();
            if (!seg.capturable)
            {
                continue;
            }
            for (const auto &stage_name : seg.stage_names)
            {
                auto *node = graph.getNode(stage_name);
                if (node && node->stage && node->stage->needsOnGraphReplayed())
                {
                    seg.replay_callbacks.push_back(node->stage.get());
                }
            }
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
            else if (has_collective_nodes)
            {
                // Non-collective stages in a collective graph's manual segment
                // (e.g., embedding, attention, KV cache, RoPE). These run on
                // the capture stream for GPU-side ordering. No host sync needed
                // between non-collective stages — the GPU stream provides ordering.
                node->stage->setGPUStream(capture_stream);
                if (!execute_node_cb(*node))
                {
                    LOG_ERROR("[DeviceGraphCaptureController] Manual stage failed on replay (collective graph): " << stage_name);
                    return false;
                }
            }
            else
            {
                node->stage->setGPUStream(capture_stream);
                if (!node->stage->execute(ctx))
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

        const bool profiling = KernelProfiler::isEnabled();
        const GraphReplayCaptureMode replay_mode = full_graph_replay
                                                       ? GraphReplayCaptureMode::FullGraph
                                                       : GraphReplayCaptureMode::Segmented;

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
        if (PerfStatsCollector::isEnabled())
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

        // Time post-launch callbacks (markOutputsDirty + onGraphReplayed)
        auto post_t0 = std::chrono::high_resolution_clock::now();
        post_launch_cb(segment, capture_stream);
        if (profiling)
        {
            auto post_t1 = std::chrono::high_resolution_clock::now();
            ForwardPassProfiler::addReplayPostLaunchNs(
                static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(post_t1 - post_t0).count()));
        }
        if (PerfStatsCollector::isEnabled())
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
        const DeviceGraphExecutor::GraphCaptureBoundaryHook &capture_boundary_cb,
        const std::function<bool(ComputeNode &, void *)> &record_snapshot_copies_cb,
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

        for (const auto &stage_name : segment.stage_names)
        {
            auto *node = graph.getNode(stage_name);
            if (node && node->stage)
            {
                node->stage->setGPUStream(capture_stream);
            }
        }

        if (!prepareGraphLaunchMetadata(graph, segment, ctx, capture_stream))
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
        {
            if (capture_boundary_cb)
            {
                const std::string boundary_name =
                    graphCaptureBoundaryName(
                        "recapture_begin",
                        current_step,
                        segment_index,
                        segment,
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
                    std::to_string(segment_index));
            if (!capture_transaction.begin())
            {
                LOG_ERROR("[DeviceGraphCaptureController] Re-capture beginCapture failed, seg " << segment_index);
                return false;
            }

            for (const auto &stage_name : segment.stage_names)
            {
                auto *node = graph.getNode(stage_name);
                if (!node || !node->stage || !node->stage->execute(ctx))
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
                    segment_index,
                    segment,
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
                                                    graph_outputs[s].count * sizeof(float));
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
                                                    direct_outputs[s].count * sizeof(float));
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
        bool has_collective_nodes,
        bool full_graph_capture,
        const std::string &perf_context,
        uint64_t current_step,
        const std::function<bool(ComputeNode &)> &execute_node_cb,
        const std::function<bool(ComputeNode &, void *)> &record_snapshot_copies_cb,
        const std::function<void(DeviceGraphExecutor::GraphSegment &, void *)> &post_launch_cb)
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

        if (!segment.capture->instantiate())
        {
            LOG_ERROR("[DeviceGraphCaptureController] Segment instantiation failed ("
                      << segment.capture->nodeCount() << " nodes)");
            return false;
        }

        const GraphReplayCaptureMode capture_mode =
            full_graph_capture
                ? GraphReplayCaptureMode::FullGraph
                : GraphReplayCaptureMode::Segmented;
        PerfStatsCollector::addCounter(
            "forward_graph",
            full_graph_capture
                ? "full_graph_capture_executable_nodes"
                : "segmented_graph_capture_executable_nodes",
            static_cast<double>(segment.capture->nodeCount()),
            "decode",
            ctx->deviceId().toString(),
            graphReplayMetadataTags(
                {{"backend", segment.capture->backendName()},
                 {"type", "captured_executable"}},
                perf_context,
                capture_mode));

        if (has_collective_nodes)
        {
            bool phase2_exec_ok = true;
            for (const auto &stage_name : segment.stage_names)
            {
                auto *node = graph.getNode(stage_name);
                if (!node || !node->stage)
                {
                    LOG_ERROR("[DeviceGraphCaptureController] Capturable segment missing stage during Phase-2 execution: " << stage_name);
                    phase2_exec_ok = false;
                    break;
                }

                if (!execute_node_cb(*node))
                {
                    LOG_ERROR("[DeviceGraphCaptureController] Capturable segment stage failed during Phase-2 execution: " << stage_name);
                    phase2_exec_ok = false;
                    break;
                }
                if (record_snapshot_copies_cb &&
                    !record_snapshot_copies_cb(*node, capture_stream))
                {
                    LOG_ERROR("[DeviceGraphCaptureController] Capturable segment snapshot copy failed during Phase-2 execution: "
                              << stage_name);
                    phase2_exec_ok = false;
                    break;
                }
                graph.markCompleted(stage_name);
            }

            if (!phase2_exec_ok)
            {
                return false;
            }

            // NOTE: Do NOT call onGraphReplayed() here. During capture phase,
            // execute() already ran host-side bookkeeping (e.g., KV cache head
            // advancement). Calling onGraphReplayed() would double-advance.
            segment.last_executed_step = current_step;
            LOG_DEBUG("[DeviceGraphCaptureController] Segment captured+executed (Phase-2 semantics): "
                      << segment.capture->nodeCount() << " nodes, " << segment.stage_names.size() << " stages");
            return true;
        }

        if (!segment.capture->launch())
        {
            LOG_ERROR("[DeviceGraphCaptureController] Segment initial launch failed");
            return false;
        }
        // Capture phase: pass skip_replay_callbacks=true because segmented
        // capture runs logical host-side bookkeeping while recording. This keeps
        // later stages in the same capture (e.g. attention after KV append)
        // seeing normal-execution cache metadata. onGraphReplayed() must only
        // run during replay (Phase 3) or host state would double-advance.
        post_launch_cb(segment, capture_stream);
        LOG_DEBUG("[DeviceGraphCaptureController] Segment captured+launched: "
                  << segment.capture->nodeCount() << " nodes, " << segment.stage_names.size() << " stages");
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

            const bool needs_execute_node = has_collective_nodes || is_collective;
            if (needs_execute_node)
            {
                if (!execute_node_cb(*node))
                {
                    LOG_ERROR("[DeviceGraphCaptureController] Capture manual stage failed: " << stage_name);
                    return false;
                }
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
        void *stream)
    {
        for (const auto &stage_name : segment.stage_names)
        {
            auto *node = graph.getNode(stage_name);
            if (!node || !node->stage || !node->stage->needsGraphLaunchPreparation())
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
        bool full_graph_replay,
        int segment_index,
        uint64_t current_step,
        const std::string &perf_context,
        const std::function<bool(const DeviceGraphExecutor::GraphSegment &)> &cohere_inputs_cb,
        const DeviceGraphExecutor::GraphCaptureBoundaryHook &capture_boundary_cb,
        const std::function<bool(ComputeNode &, void *)> &record_snapshot_copies_cb,
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
        // stages in a different order or on different streams. Recapture owns
        // the preflight internally so it runs after launch-metadata preparation.
        const bool prepare_verify_inputs = verify_mode && !recapture_mode;
        const std::string device_name = ctx->deviceId().toString();

        if (prepare_verify_inputs &&
            (!cohere_inputs_cb || !cohere_inputs_cb(segment)))
        {
            return result;
        }

        if (!prepareGraphLaunchMetadata(graph, segment, ctx, capture_stream))
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
                capture_boundary_cb,
                record_snapshot_copies_cb,
                post_launch_cb);
            result.success = recapture_ok;
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
        bool full_graph_replay,
        uint64_t current_step,
        int segment_index,
        const std::string &perf_context,
        const std::function<bool(const DeviceGraphExecutor::GraphSegment &)> &cohere_inputs_cb,
        const std::function<bool(ComputeNode &)> &execute_node_cb,
        const DeviceGraphExecutor::GraphCaptureBoundaryHook &capture_boundary_cb,
        const std::function<bool(ComputeNode &, void *)> &record_snapshot_copies_cb,
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
                full_graph_replay,
                segment_index,
                current_step,
                perf_context,
                cohere_inputs_cb,
                capture_boundary_cb,
                record_snapshot_copies_cb,
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
        const ReplayHooks &hooks)
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

        initializeReplayCallbacks(graph, segment_cache);
        const bool full_graph_capture =
            captureModeForCache(segment_cache) == GraphReplayCaptureMode::FullGraph;

        /*
         * Some MoE stages only know whether their graph-capturable fast path is
         * available after the warmup pass has prepared backend resources.  Make
         * that decision before recording any Phase-2 segment.  The old behavior
         * discovered this inside the segment loop after earlier captured
         * segments had already mutated activation buffers, then restarted the
         * whole decode graph from those dirty buffers. That is not a valid
         * replay/restore boundary. A failed warmup-dependent preflight is now a
         * hard capture-contract failure before any Phase-2 segment executes.
         */
        for (const auto &seg : segment_cache.segments)
        {
            if (!seg.capturable)
                continue;
            for (const auto &stage_name : seg.stage_names)
            {
                auto *node = graph.getNode(stage_name);
                if (!node || !node->stage)
                    continue;
                if (node->stage->supportsWarmupDependentGraphCapture() &&
                    !node->stage->isGraphCapturable())
                {
                    const std::string readiness =
                        node->stage->graphCaptureReadinessDebugString();
                    LOG_ERROR(
                        "[DeviceGraphCaptureController] Warmup-dependent stage '"
                        << stage_name
                        << "' is not graph-capturable after warmup"
                        << (readiness.empty() ? "" : "; ")
                        << readiness
                        << "; mandatory capture cannot proceed");
                    PerfStatsCollector::addCounter(
                        "forward_graph",
                        "warmup_dependent_capture_not_ready",
                        1.0,
                        "decode",
                        ctx ? ctx->deviceId().toString() : std::string{},
                        {{"stage", stage_name},
                         {"stage_type", computeStageTypeName(node->stage->type())},
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
                // Capturable path: set stream -> begin capture -> execute nodes
                // into graph -> end capture -> finalize for Phase-2 semantics.
                for (const auto &stage_name : seg.stage_names)
                {
                    auto *node = graph.getNode(stage_name);
                    if (node && node->stage)
                    {
                        node->stage->setGPUStream(capture_stream);
                    }
                }
                for (const auto &stage_name : seg.stage_names)
                {
                    auto *node = graph.getNode(stage_name);
                    if (!node || !node->stage)
                    {
                        continue;
                    }
                    if (node->stage->supportsWarmupDependentGraphCapture() &&
                        !node->stage->isGraphCapturable())
                    {
                        LOG_ERROR("[DeviceGraphCaptureController] Warmup-dependent stage '"
                                  << stage_name
                                  << "' is still not graph-capturable after warmup");
                        result.reset_cache = true;
                        return result;
                    }
                }

                if (!prepareGraphLaunchMetadata(graph, seg, ctx, capture_stream))
                {
                    result.reset_cache = true;
                    return result;
                }

                /*
                 * Snapshot copy nodes need their per-stage output descriptors and
                 * storage allocated before stream capture begins. Some outputs,
                 * especially in-place collective outputs such as TP allreduce
                 * tensors, first appear in snapshot publication at the collective
                 * stage itself. If we discover them while capture is active we
                 * cannot allocate the destination tensor safely. This prepare
                 * hook must not copy payload bytes: the point-in-time copy is
                 * recorded only after the producing stage executes.
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
                            LOG_ERROR("[DeviceGraphCaptureController] Snapshot prewarm failed before cached graph capture: "
                                      << stage_name);
                            result.reset_cache = true;
                            result.success = false;
                            return result;
                        }
                    }
                }

                /*
                 * This is the mandatory external-event import boundary for
                 * Phase-2 native capture. It deliberately follows every
                 * stage-owned metadata/snapshot preparation call: either may
                 * publish a fresh completion event on an eager stream. The
                 * complete segment is prejoined before the LocalTP rendezvous,
                 * making late capture-time event discovery impossible.
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

                // Native capture requires an idle stream. Fence exactly the
                // capture stream's prior warmup work with the cache-owned event;
                // no device-wide or stream-wide synchronization participates.
                segment_cache.waitForCaptureStreamFence();

                if (hooks.capture_boundary)
                {
                    const std::string boundary_name =
                        graphCaptureBoundaryName(
                            "capture_begin",
                            current_step,
                            static_cast<int>(segment_index),
                            seg,
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

                bool exec_ok = true;
                std::string failed_stage_name;
                {
                    ScopedBackendGraphCapture capture_transaction(
                        *gpu_ctx,
                        *seg.capture,
                        "capture phase segment=" +
                            std::to_string(segment_index) +
                            " stages=" + describeSegmentStages(seg));
                    if (!capture_transaction.begin())
                    {
                        LOG_ERROR("[DeviceGraphCaptureController] beginCapture failed for segment");
                        result.reset_cache = true;
                        return result;
                    }

                    for (const auto &stage_name : seg.stage_names)
                    {
                        auto *node = graph.getNode(stage_name);
                        if (!node || !node->stage || !node->stage->execute(ctx))
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
                            static_cast<int>(segment_index),
                            seg,
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

                const bool capture_finalize_ok = finalizeCapturePhaseCapturableSegment(
                    graph,
                    seg,
                    ctx,
                    gpu_ctx,
                    capture_stream,
                    has_collective_nodes,
                    full_graph_capture,
                    segment_cache.perf_context,
                    current_step,
                    hooks.execute_node,
                    hooks.record_snapshot_copies,
                    hooks.post_launch);
                if (!capture_finalize_ok)
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
                 * same position during warmup, capture phase, and replay.
                 */
                const bool manual_capture_ok = executeCapturePhaseManualSegment(
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
        const bool profiling = KernelProfiler::isEnabled();
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
        if (stream_only_mode)
        {
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
        const auto &device_id = ctx->deviceId();
        const std::string device_name = device_id.toString();
        const int total_segments = static_cast<int>(segment_cache.segments.size());
        const GraphReplayCaptureMode replay_mode = captureModeForCache(segment_cache);
        const bool full_graph_replay = replay_mode == GraphReplayCaptureMode::FullGraph;
        auto replay_total_tags = replayCacheTags(segment_cache, replay_mode);
        replay_total_tags.emplace(
            "sync_scope",
            can_defer_final_sync ? "launch_only_deferred" : "stream_synchronized");
        PerfStatsCollector::ScopedTimer replay_timer(
            "forward_graph",
            replayMetricName(replay_mode, "total"),
            "decode",
            device_name,
            graphReplayHostTimingTags(std::move(replay_total_tags), "total_replay_host_wall", replay_mode));

        std::vector<ReplayGpuEventTimer> replay_event_timers;
        replay_event_timers.reserve(segment_cache.segments.size());
        auto total_replay_event_tags = replayCacheTags(segment_cache, replay_mode);
        total_replay_event_tags.emplace(
            "sync_scope",
            can_defer_final_sync ? "profiling_event_synchronized" : "stream_synchronized");
        ReplayGpuEventTimer total_replay_event_timer(
            gpu_ctx,
            capture_stream,
            "graph_replay.total",
            "decode",
            device_name,
            graphReplayGpuEventTags(std::move(total_replay_event_tags), "total_replay_gpu_event", replay_mode));

        auto collect_replay_gpu_events = [&](bool synchronize_total_event)
        {
            if (total_replay_event_timer.active())
            {
                total_replay_event_timer.stop();
                total_replay_event_timer.record(synchronize_total_event);
            }
            for (auto &timer : replay_event_timers)
            {
                timer.record(false);
            }
        };

        int seg_idx = 0;
        for (auto &seg : segment_cache.segments)
        {
            const auto seg_tags = replaySegmentTags(seg, segment_cache.perf_context);
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

            PerfStatsCollector::addCounter(
                "forward_graph",
                replayUnitCounterName(replay_mode),
                1.0,
                "decode",
                device_name,
                seg_tags);

            auto event_tags = seg_tags;
            event_tags.emplace(replayUnitIndexTagName(replay_mode), std::to_string(seg_idx));
            event_tags.emplace(
                "sync_scope",
                can_defer_final_sync ? "profiling_event_synchronized" : "stream_synchronized");
            replay_event_timers.emplace_back(
                gpu_ctx,
                capture_stream,
                replayUnitGpuEventName(replay_mode),
                "decode",
                device_name,
                graphReplayGpuEventTags(std::move(event_tags), replayUnitTimingScope(replay_mode), replay_mode));

            const auto segment_t0 = std::chrono::high_resolution_clock::now();
            // Segment execution picks capturable or manual behavior based on
            // segment metadata prepared during warmup.
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
                full_graph_replay,
                current_step,
                seg_idx,
                segment_cache.perf_context,
                hooks.cohere_inputs,
                hooks.execute_node,
                hooks.capture_boundary,
                hooks.record_snapshot_copies,
                hooks.post_launch);
            if (PerfStatsCollector::isEnabled())
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
                return result;
            }

            replay_event_timers.back().stop();

            if (trace_replay)
            {
                LOG_DEBUG("[ReplayTrace] " << device_id.toString()
                                           << " step=" << current_step
                                           << " seg=" << seg_idx << "/" << total_segments << " DONE");
            }

            seg_idx++;
        }

        if (total_replay_event_timer.active())
            total_replay_event_timer.stop();

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
            if (PerfStatsCollector::isEnabled())
            {
                PerfStatsCollector::addCounter(
                    "forward_graph",
                    replayMetricName(replay_mode, "final_sync_deferred"),
                    1.0,
                    "decode",
                    device_name,
                    replayCacheTags(segment_cache, replay_mode));
            }
            // Production MTP sidecar replay can deliberately defer the final
            // ownership fence. When GPU stage timing is requested, wait only on
            // the replay stop event here so exported stage_gpu graph-replay
            // rows are true GPU elapsed time, not host enqueue duration.
            collect_replay_gpu_events(/*synchronize_total_event=*/true);
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
            if (PerfStatsCollector::isEnabled())
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
        collect_replay_gpu_events(/*synchronize_total_event=*/false);
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
        const std::function<bool(ComputeNode &)> &cohere_stage_cb)
    {
        for (const auto &stage_name : segment.stage_names)
        {
            auto *node = graph.getNode(stage_name);
            if (!node || !node->stage)
            {
                continue;
            }

            if (!cohere_stage_cb(*node))
            {
                return false;
            }
        }

        return true;
    }

    void DeviceGraphCaptureController::postCapturedSegmentLaunch(
        ComputeGraph &graph,
        DeviceGraphExecutor::GraphSegment &segment,
        uint64_t current_step,
        void * /*stream*/,
        const std::function<void(BufferId, DeviceId)> &mark_arena_write_dirty_cb,
        bool skip_replay_callbacks)
    {
        if (!segment.arena_writes_cached)
        {
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

                const DeviceId target_device = node->device.is_valid() ? node->device : node->stage->device();
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

        // Use arena ids instead of retaining raw TensorBase* pointers. Prefix
        // restore, rollback, and request clears may preserve graph topology
        // while replacing or rewinding tensor-backed state.
        for (const auto &write : segment.cached_arena_writes)
        {
            mark_arena_write_dirty_cb(write.id, write.device);
        }

        // Skip replay callbacks during the capture phase: execute() already ran
        // all host-side bookkeeping (e.g., KV cache head/count advancement).
        // Calling onGraphReplayed() here would double-advance host state, causing
        // decode steps to write to wrong KV cache positions and produce garbage.
        if (!skip_replay_callbacks)
        {
            for (auto *stage : segment.replay_callbacks)
            {
                stage->onGraphReplayed();
            }
            LOG_DEBUG("[DeviceGraphCaptureController] Ran " << segment.replay_callbacks.size()
                                                            << " onGraphReplayed() callbacks");
        }
        else
        {
            LOG_DEBUG("[DeviceGraphCaptureController] SKIPPED " << segment.replay_callbacks.size()
                                                                << " onGraphReplayed() callbacks (capture phase)");
        }

        segment.last_executed_step = current_step;
    }

} // namespace llaminar2
