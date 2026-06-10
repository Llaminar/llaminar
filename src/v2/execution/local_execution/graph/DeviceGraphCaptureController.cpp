#include "DeviceGraphCaptureController.h"
#include "GraphCaptureGuard.h"

#include "../coherence/StageCoherence.h"
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
#include <vector>

namespace llaminar2
{
    namespace
    {
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

        PerfStatsCollector::Tags graphReplayHostTimingTags(PerfStatsCollector::Tags tags,
                                                           const char *timing_scope)
        {
            tags.emplace("attribution", "host_wall");
            tags.emplace("source", "segmented_graph_capture");
            tags.emplace("graph_capture_scope", "segmented_replay_host");
            if (timing_scope && timing_scope[0] != '\0')
            {
                tags.emplace("timing_scope", timing_scope);
            }
            return tags;
        }

        PerfStatsCollector::Tags graphReplayGpuEventTags(PerfStatsCollector::Tags tags,
                                                         const char *timing_scope)
        {
            tags.emplace("attribution", "gpu_event");
            tags.emplace("source", "segmented_graph_capture");
            tags.emplace("graph_capture_scope", "segmented_replay_events");
            if (timing_scope && timing_scope[0] != '\0')
            {
                tags.emplace("timing_scope", timing_scope);
            }
            return tags;
        }

        PerfStatsCollector::Tags graphReplayMetadataTags(PerfStatsCollector::Tags tags,
                                                         const std::string &perf_context)
        {
            tags.emplace("attribution", "graph_replay_metadata");
            tags.emplace("source", "segmented_graph_capture");
            tags.emplace("graph_capture_scope", "segmented_capture_plan");
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


    DeviceGraphCaptureController::Transition DeviceGraphCaptureController::beginStep(
        bool initialized,
        bool &needs_capture,
        uint64_t &decode_step)
    {
        // Segmented decode uses a monotonic step so we can reason about which
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

    void DeviceGraphCaptureController::prepareDeviceForSegmentedCapture(IDeviceContext *ctx)
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

    void DeviceGraphCaptureController::executeWarmupPhase(
        ComputeGraph &graph,
        DeviceGraphExecutor::GraphSegmentCache &segment_cache,
        const std::unordered_set<std::string> *collective_nodes,
        bool has_collective_nodes,
        bool collectives_graph_capturable)
    {
        buildWarmupSegments(
            graph,
            segment_cache,
            collective_nodes,
            has_collective_nodes,
            collectives_graph_capturable);
        markWarmupComplete(
            segment_cache.initialized,
            segment_cache.needs_capture);
    }

    void DeviceGraphCaptureController::buildWarmupSegments(
        ComputeGraph &graph,
        DeviceGraphExecutor::GraphSegmentCache &segment_cache,
        const std::unordered_set<std::string> *collective_nodes,
        bool has_collective_nodes,
        bool collectives_graph_capturable)
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

        auto is_collective_stage = [](ComputeStageType t)
        {
            return t == ComputeStageType::ALLREDUCE ||
                   t == ComputeStageType::ALLGATHER ||
                   t == ComputeStageType::ALLGATHER_V;
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
            const bool collective_by_type = is_collective_stage(node->stage->type());
            const bool collective_by_name = (collective_nodes && collective_nodes->count(name));

            // Collective stages: capturable only if the backend supports on-stream
            // allreduce (graph-capturable collectives). Otherwise force manual.
            if (collective_by_type || collective_by_name)
            {
                stage_capturable = collectives_graph_capturable;
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
            // Collective stages are forced manual above. Compute-only stages
            // (GEMM, norms, SwiGLU, residual add, lm_head) can be safely
            // captured in graph segments between the manual collective segments.
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

        LOG_DEBUG("[DeviceGraphExecutor] Segmented graph: " << capturable_segments << " capturable segments ("
                                                            << capturable_stages << " stages) + " << manual_segments << " manual segments ("
                                                            << manual_stages << " stages)");

        if (PerfStatsCollector::isEnabled())
        {
            PerfStatsCollector::addCounter(
                "forward_graph",
                "segmented_plan_segments",
                static_cast<double>(segment_cache.segments.size()),
                "decode",
                "",
                {{"type", "total"}});
            PerfStatsCollector::addCounter(
                "forward_graph",
                "segmented_plan_segments",
                static_cast<double>(capturable_segments),
                "decode",
                "",
                {{"type", "capturable"}});
            PerfStatsCollector::addCounter(
                "forward_graph",
                "segmented_plan_segments",
                static_cast<double>(manual_segments),
                "decode",
                "",
                {{"type", "manual"}});
            PerfStatsCollector::addCounter(
                "forward_graph",
                "segmented_plan_stages",
                static_cast<double>(capturable_stages),
                "decode",
                "",
                {{"type", "capturable"}});
            PerfStatsCollector::addCounter(
                "forward_graph",
                "segmented_plan_stages",
                static_cast<double>(manual_stages),
                "decode",
                "",
                {{"type", "manual"}});
            PerfStatsCollector::addCounter(
                "forward_graph",
                "segmented_plan_max_segment_stages",
                static_cast<double>(max_capturable_segment_stages),
                "decode",
                "",
                {{"type", "capturable"}});
            PerfStatsCollector::addCounter(
                "forward_graph",
                "segmented_plan_max_segment_stages",
                static_cast<double>(max_manual_segment_stages),
                "decode",
                "",
                {{"type", "manual"}});

            for (const auto &[type_name, count] : capturable_stage_types)
            {
                PerfStatsCollector::addCounter(
                    "forward_graph",
                    "segmented_plan_stage_types",
                    static_cast<double>(count),
                    "decode",
                    "",
                    {{"segment_type", "capturable"}, {"stage_type", type_name}});
            }
            for (const auto &[type_name, count] : manual_stage_types)
            {
                PerfStatsCollector::addCounter(
                    "forward_graph",
                    "segmented_plan_stage_types",
                    static_cast<double>(count),
                    "decode",
                    "",
                    {{"segment_type", "manual"}, {"stage_type", type_name}});
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
                    graphReplayMetadataTags(std::move(tags), segment_cache.perf_context));
            };

            record_stage_gpu_plan(
                "graph_replay_plan_segments",
                static_cast<double>(segment_cache.segments.size()),
                {{"type", "total"}});
            record_stage_gpu_plan(
                "graph_replay_plan_segments",
                static_cast<double>(capturable_segments),
                {{"type", "capturable"}});
            record_stage_gpu_plan(
                "graph_replay_plan_segments",
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
                    {{"segment_type", "capturable"}, {"stage_type", type_name}});
            }
            for (const auto &[type_name, count] : manual_stage_types)
            {
                record_stage_gpu_plan(
                    "graph_replay_plan_stage_types",
                    static_cast<double>(count),
                    {{"segment_type", "manual"}, {"stage_type", type_name}});
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
        const std::function<bool(ComputeNode &)> &execute_node_cb)
    {
        if (!ctx || !gpu_ctx)
        {
            LOG_ERROR("[DeviceGraphCaptureController] Manual replay missing context");
            return false;
        }

        auto is_collective_stage = [](ComputeStageType t)
        {
            return t == ComputeStageType::ALLREDUCE ||
                   t == ComputeStageType::ALLGATHER ||
                   t == ComputeStageType::ALLGATHER_V;
        };

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
            const bool is_collective = is_collective_stage(stage_type);

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
                    gpu_ctx->insertStreamDependency(compute_stream, capture_stream);
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
                    gpu_ctx->insertStreamDependency(capture_stream, compute_stream);
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
                "segmented_replay_graph_launch",
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
                "segmented_replay_post_launch",
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

        bool exec_ok = true;
        bool end_capture_ok = true;
        {
            gpu_ctx->setGraphCaptureActive(true);
            GraphCaptureGuard capture_guard(false);
            gpu_ctx->clearLastError();

            if (!segment.capture->beginCapture())
            {
                gpu_ctx->setGraphCaptureActive(false);
                LOG_ERROR("[DeviceGraphCaptureController] Re-capture beginCapture failed, seg " << segment_index);
                return false;
            }

            for (const auto &stage_name : segment.stage_names)
            {
                auto *node = graph.getNode(stage_name);
                if (!node || !node->stage || !node->stage->execute(ctx))
                {
                    exec_ok = false;
                    break;
                }
            }

            gpu_ctx->setGraphCaptureActive(false);
            if (!exec_ok)
            {
                segment.capture->endCapture();
                end_capture_ok = false;
            }
            else
            {
                end_capture_ok = segment.capture->endCapture();
            }
        }

        if (!exec_ok || !end_capture_ok)
        {
            if (capture_stream)
            {
                (void)gpu_ctx->synchronizeStreamChecked(capture_stream);
            }
            gpu_ctx->clearLastError();
            if (!exec_ok)
            {
                LOG_ERROR("[DeviceGraphCaptureController] Re-capture stage execution failed, seg " << segment_index);
            }
            else
            {
                LOG_ERROR("[DeviceGraphCaptureController] Re-capture endCapture failed, seg " << segment_index);
            }
            return false;
        }

        const bool skip_in_place_update = ctx->deviceId().is_rocm();
        auto update_result = GraphUpdateResult::NeedsReinstantiate;
        if (!skip_in_place_update)
        {
            update_result = segment.capture->tryUpdate();
        }
        if (skip_in_place_update ||
            update_result == GraphUpdateResult::NeedsReinstantiate ||
            update_result == GraphUpdateResult::Failed)
        {
            gpu_ctx->clearLastError();
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
        if (!gpu_ctx->synchronizeStreamChecked(capture_stream))
        {
            LOG_ERROR("[DeviceGraphCaptureController] Re-capture stream sync failed after seg " << segment_index);
            return false;
        }

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
                const auto &dump_info = node->stage->getDumpInfo();
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
                const auto &dump_info = node->stage->getDumpInfo();
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
        uint64_t current_step,
        const std::function<bool(ComputeNode &)> &execute_node_cb,
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
            LOG_DEBUG("[DeviceGraphCaptureController] Segment captured 0 nodes (CPU-only), will execute manually");
            segment.capture.reset();
            return true;
        }

        if (!segment.capture->instantiate())
        {
            LOG_WARN("[DeviceGraphCaptureController] Segment instantiation failed ("
                     << segment.capture->nodeCount() << " nodes)");
            return false;
        }

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
            // Use stream-level sync (not device-wide) to avoid conflict with
            // hipStreamCaptureModeGlobal on other TP devices.
            if (!gpu_ctx->synchronizeStreamChecked(capture_stream))
            {
                LOG_ERROR("[DeviceGraphCaptureController] Capture-phase stream sync failed after segment starting at "
                          << (segment.stage_names.empty() ? std::string("<empty>") : segment.stage_names.front())
                          << " stages=" << describeSegmentStages(segment));
                return false;
            }
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
        // Use stream-level sync (not device-wide) — capture_stream is where
        // the graph was launched, and hipDeviceSynchronize is illegal during
        // global capture mode if another TP device is still capturing.
        if (!gpu_ctx->synchronizeStreamChecked(capture_stream))
        {
            LOG_ERROR("[DeviceGraphCaptureController] Initial captured launch stream sync failed after segment starting at "
                      << (segment.stage_names.empty() ? std::string("<empty>") : segment.stage_names.front())
                      << " stages=" << describeSegmentStages(segment));
            return false;
        }
        LOG_DEBUG("[DeviceGraphCaptureController] Segment captured+launched: "
                  << segment.capture->nodeCount() << " nodes, " << segment.stage_names.size() << " stages");
        return true;
    }

    bool DeviceGraphCaptureController::executeCapturePhaseManualSegment(
        ComputeGraph &graph,
        DeviceGraphExecutor::GraphSegment &segment,
        IDeviceContext *ctx,
        IWorkerGPUContext *gpu_ctx,
        bool has_collective_nodes,
        uint64_t current_step,
        const std::function<bool(ComputeNode &)> &execute_node_cb)
    {
        if (!ctx || !gpu_ctx)
        {
            LOG_ERROR("[DeviceGraphCaptureController] Capture manual segment missing context");
            return false;
        }

        auto is_collective_stage = [](ComputeStageType t)
        {
            return t == ComputeStageType::ALLREDUCE ||
                   t == ComputeStageType::ALLGATHER ||
                   t == ComputeStageType::ALLGATHER_V;
        };

        // Use the real default stream, not nullptr. AMDDeviceContext creates a
        // real hipStream_t via hipStreamCreateWithFlags, so defaultStream() is
        // a distinct object from hipStream_t(0). The RCCL coordinator's
        // compute_events are recorded on this same stream (registered via
        // setComputeStreams()), so stages must target the same stream identity
        // for event-based allreduce synchronization to chain correctly.
        void *compute_stream = gpu_ctx->defaultStream();

        // Use stream-level sync (not device-wide hipDeviceSynchronize) to avoid
        // conflict with hipStreamCaptureModeGlobal. In multi-device TP, the other
        // device may have already started graph capture by the time we reach this
        // sync. hipDeviceSynchronize() is always illegal during global capture mode
        // (returns hipErrorStreamCaptureUnsupported), but hipStreamSynchronize()
        // on a non-captured stream is safe.
        if (!gpu_ctx->synchronizeStreamChecked(compute_stream))
        {
            LOG_ERROR("[DeviceGraphCaptureController] Capture manual pre-sync failed");
            return false;
        }

        for (const auto &stage_name : segment.stage_names)
        {
            auto *node = graph.getNode(stage_name);
            if (!node || !node->stage)
            {
                LOG_ERROR("[DeviceGraphCaptureController] Capture manual segment missing stage: " << stage_name);
                return false;
            }

            node->stage->setGPUStream(compute_stream);

            const bool needs_execute_node = has_collective_nodes || is_collective_stage(node->stage->type());
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
        if (!gpu_ctx->synchronizeStreamChecked(compute_stream))
        {
            LOG_ERROR("[DeviceGraphCaptureController] Capture manual post-sync failed after segment starting at "
                      << (segment.stage_names.empty() ? std::string("<empty>") : segment.stage_names.front()));
            return false;
        }
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
        int segment_index,
        const std::string &perf_context,
        const std::function<bool(const DeviceGraphExecutor::GraphSegment &)> &cohere_inputs_cb,
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
        // stages in a different order or on different streams.
        const bool skip_coherence = !recapture_mode && !verify_mode;
        const std::string device_name = ctx->deviceId().toString();

        if (!skip_coherence && !cohere_inputs_cb(segment))
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
            perf_context,
            device_name,
            post_launch_cb);
        result.success = launch_ok;
        result.launch_failure_fallback = !launch_ok;
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
        uint64_t current_step,
        int segment_index,
        const std::string &perf_context,
        const std::function<bool(const DeviceGraphExecutor::GraphSegment &)> &cohere_inputs_cb,
        const std::function<bool(ComputeNode &)> &execute_node_cb,
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
                segment_index,
                perf_context,
                cohere_inputs_cb,
                post_launch_cb);
            result.success = capturable_result.success;
            result.skipped_non_idempotent = capturable_result.skipped_non_idempotent;
            result.launch_failure_fallback = capturable_result.launch_failure_fallback;
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
            execute_node_cb);
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
            // No capture stream means segmented capture cannot proceed safely;
            // caller should fall back to fast decode for this step.
            LOG_WARN("[DeviceGraphCaptureController] Failed to create capture stream, falling back");
            result.fallback_to_fast_decode = true;
            return result;
        }
        void *capture_stream = segment_cache.capture_stream;

        initializeReplayCallbacks(graph, segment_cache);

        // When a capturable segment fails mid-capture, we CANNOT fall back to
        // executeFastDecode (full re-execution) because prior manual segments
        // already executed allreduces. Re-executing them would cause a collective
        // count mismatch between devices → deadlock. Instead, we abandon capture
        // and continue executing the remaining segments normally.
        bool capture_abandoned = false;

        for (auto &seg : segment_cache.segments)
        {
            if (seg.capturable && !capture_abandoned)
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
                    result.fallback_to_fast_decode = true;
                    return result;
                }

                // Drain any pending warmup work on the capture stream before
                // starting capture. During warmup (step 0), all stages —
                // including RCCL allreduce — execute on this stream. If that
                // work hasn't completed, beginCapture() encounters pending ops
                // and either includes them in the graph (wrong) or fails.
                // Use stream-level sync (not device-wide) because
                // hipDeviceSynchronize() is illegal under global capture mode
                // if another TP device has already started capturing.
                if (!gpu_ctx->synchronizeStreamChecked(capture_stream))
                {
                    LOG_ERROR("[DeviceGraphCaptureController] Capture warmup stream sync failed before segment starting at "
                              << (seg.stage_names.empty() ? std::string("<empty>") : seg.stage_names.front()));
                    result.reset_cache = true;
                    result.fallback_to_fast_decode = true;
                    return result;
                }

                seg.capture = gpu_ctx->createGraphCapture(capture_stream);
                if (!seg.capture)
                {
                    LOG_ERROR("[DeviceGraphCaptureController] Failed to create graph capture for segment");
                    result.reset_cache = true;
                    result.fallback_to_fast_decode = true;
                    return result;
                }

                bool exec_ok = true;
                bool end_capture_ok = true;
                {
                    // Set capture-active flags only for the actual stream-capture
                    // interval. Segmented non-collective capture skips replay
                    // callbacks on the immediate launch, so stateful stages must
                    // apply logical host bookkeeping while recording. Collective
                    // Phase-2 capture re-executes stages after capture, and
                    // prefill capture uses replay callbacks, so both leave this
                    // bookkeeping flag disabled.
                    const bool capture_exec_updates_host_state = !has_collective_nodes;

                    // Set capture-active flag BEFORE beginCapture() to minimize
                    // the race window where another device's thread might call
                    // synchronize() on this context after HIP starts capturing
                    // but before the flag is set.
                    gpu_ctx->setGraphCaptureActive(true);
                    GraphCaptureGuard capture_guard(capture_exec_updates_host_state);

                    // Clear any sticky HIP error left over from warmup or prior
                    // operations on this stream. Without this, the first kernel
                    // launch after beginCapture would see the stale error and fail
                    // with "operation failed due to a previous error during capture".
                    gpu_ctx->clearLastError();

                    if (!seg.capture->beginCapture())
                    {
                        gpu_ctx->setGraphCaptureActive(false);
                        LOG_ERROR("[DeviceGraphCaptureController] beginCapture failed for segment");
                        result.reset_cache = true;
                        result.fallback_to_fast_decode = true;
                        return result;
                    }

                    for (const auto &stage_name : seg.stage_names)
                    {
                        auto *node = graph.getNode(stage_name);
                        if (!node || !node->stage || !node->stage->execute(ctx))
                        {
                            LOG_ERROR("[DeviceGraphCaptureController] Stage failed during segmented capture: " << stage_name);
                            exec_ok = false;
                            break;
                        }
                        graph.markCompleted(stage_name);
                    }

                    // Clear capture-active flag BEFORE endCapture so that
                    // post-capture operations can sync normally.
                    gpu_ctx->setGraphCaptureActive(false);

                    // If a stage failed mid-capture, we MUST still call endCapture()
                    // to exit capture mode on the stream. Otherwise the stream
                    // remains in capture state and any subsequent
                    // synchronizeStream()/clearLastError()/kernel launch will fail
                    // with "operation not permitted when stream is capturing",
                    // cascading into a SIGSEGV inside libcudart when it tries to
                    // cuMemcpyHtoDAsync on a still-capturing stream.
                    if (!exec_ok)
                    {
                        // Call endCapture() for its side-effect (exit capture mode);
                        // the resulting graph is unusable because the stage failed.
                        seg.capture->endCapture();
                        end_capture_ok = false;
                    }
                    else
                    {
                        end_capture_ok = seg.capture->endCapture();
                    }
                }

                if (!exec_ok || !end_capture_ok)
                {
                    LOG_WARN("[DeviceGraphCaptureController] Segmented capture failed, "
                             "continuing without capture to preserve collective sync");

                    // After a failed capture, the HIP/CUDA error state is sticky —
                    // subsequent kernel launches on this device will fail with
                    // "operation failed due to a previous error during capture" or
                    // "invalid argument" until the error is consumed.
                    // Synchronize both streams and clear the last error so that
                    // the remaining stages can execute normally.
                    // Use stream-level sync (not device-wide) to avoid conflict
                    // with hipStreamCaptureModeGlobal on other TP devices.
                    (void)gpu_ctx->synchronizeStreamChecked(capture_stream);
                    void *default_stream = gpu_ctx->defaultStream();
                    (void)gpu_ctx->synchronizeStreamChecked(default_stream);
                    gpu_ctx->clearLastError();
                    // Re-execute the remaining un-completed stages of this segment
                    // without capture. Stages that completed before the failure already
                    // ran (capture mode executes AND records), so skip those.
                    for (const auto &stage_name : seg.stage_names)
                    {
                        auto *node = graph.getNode(stage_name);
                        if (!node || !node->stage)
                            continue;
                        if (node->completed)
                            continue; // Already ran during the partial capture

                        node->stage->setGPUStream(default_stream);
                        if (!node->stage->execute(ctx))
                        {
                            LOG_ERROR("[DeviceGraphCaptureController] Recovery execution "
                                      "failed for stage: "
                                      << stage_name);
                            result.reset_cache = true;
                            result.success = false;
                            return result;
                        }
                        graph.markCompleted(stage_name);
                    }

                    capture_abandoned = true;
                    seg.capture.reset(); // Discard the failed capture object
                    continue;            // Process remaining segments normally
                }

                const bool capture_finalize_ok = finalizeCapturePhaseCapturableSegment(
                    graph,
                    seg,
                    ctx,
                    gpu_ctx,
                    capture_stream,
                    has_collective_nodes,
                    current_step,
                    hooks.execute_node,
                    hooks.post_launch);
                if (!capture_finalize_ok)
                {
                    result.reset_cache = true;
                    return result;
                }
            }
            else
            {
                // Manual path (or abandoned-capture fallback): execute with
                // normal stage semantics. When capture_abandoned is set, even
                // capturable segments go through this path to preserve the
                // 1:1 collective call balance between devices.
                const bool manual_capture_ok = executeCapturePhaseManualSegment(
                    graph,
                    seg,
                    ctx,
                    gpu_ctx,
                    has_collective_nodes,
                    current_step,
                    hooks.execute_node);
                if (!manual_capture_ok)
                {
                    result.reset_cache = true;
                    return result;
                }
            }
        }

        if (capture_abandoned)
        {
            // Execution completed but capture was abandoned. Reset cache so the
            // next decode step tries capturing again from scratch.
            result.reset_cache = true;
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
        const bool needs_segment_sync = ctx->deviceId().is_cuda();

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

        const bool can_defer_final_sync =
            defer_final_sync &&
            !has_collective_nodes &&
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
        auto replay_total_tags = replayCacheTags(segment_cache);
        replay_total_tags.emplace(
            "sync_scope",
            can_defer_final_sync ? "launch_only_deferred" : "stream_synchronized");
        PerfStatsCollector::ScopedTimer replay_timer(
            "forward_graph",
            "segmented_replay_total",
            "decode",
            device_name,
            graphReplayHostTimingTags(std::move(replay_total_tags), "total_replay_host_wall"));

        std::vector<ReplayGpuEventTimer> replay_event_timers;
        replay_event_timers.reserve(segment_cache.segments.size());
        auto total_replay_event_tags = replayCacheTags(segment_cache);
        total_replay_event_tags.emplace(
            "sync_scope",
            can_defer_final_sync ? "profiling_event_synchronized" : "stream_synchronized");
        ReplayGpuEventTimer total_replay_event_timer(
            gpu_ctx,
            capture_stream,
            "graph_replay.total",
            "decode",
            device_name,
            graphReplayGpuEventTags(std::move(total_replay_event_tags), "total_replay_gpu_event"));

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
            const char *seg_type = segmentTypeName(seg);
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
                "segmented_replay_segments",
                1.0,
                "decode",
                device_name,
                seg_tags);

            auto event_tags = seg_tags;
            event_tags.emplace("segment_index", std::to_string(seg_idx));
            event_tags.emplace(
                "sync_scope",
                can_defer_final_sync ? "profiling_event_synchronized" : "stream_synchronized");
            replay_event_timers.emplace_back(
                gpu_ctx,
                capture_stream,
                "graph_replay.segment",
                "decode",
                device_name,
                graphReplayGpuEventTags(std::move(event_tags), "segment_replay_gpu_event"));

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
                current_step,
                seg_idx,
                segment_cache.perf_context,
                hooks.cohere_inputs,
                hooks.execute_node,
                hooks.post_launch);
            if (PerfStatsCollector::isEnabled())
            {
                const auto segment_t1 = std::chrono::high_resolution_clock::now();
                const auto segment_ns =
                    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(segment_t1 - segment_t0).count());
                PerfStatsCollector::recordTimingNs(
                    "forward_graph",
                    "segmented_replay_segment",
                    segment_ns,
                    "decode",
                    device_name,
                    graphReplayHostTimingTags(seg_tags, "segment_replay_host_wall"));
            }

            if (!replay_result.success)
            {
                result.launch_failure_fallback = replay_result.launch_failure_fallback;
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
                                       << " ALL " << total_segments << " segments done, entering final synchronize()");
        }
        // Sync both known streams instead of device-wide hipDeviceSynchronize.
        // Graph segments replayed on capture_stream; manual segments (embedding)
        // ran on defaultStream. Syncing only these two is cheaper than a
        // device-wide barrier and avoids interference with global capture mode.
        //
        // A caller may explicitly defer this sync when it will immediately enqueue
        // a dependent GPU operation on the same capture stream and synchronize
        // through that operation instead. This is used for MTP sidecar replay plus
        // greedy argmax sampling.
        if (can_defer_final_sync)
        {
            if (PerfStatsCollector::isEnabled())
            {
                PerfStatsCollector::addCounter(
                    "forward_graph",
                    "segmented_replay_final_sync_deferred",
                    1.0,
                    "decode",
                    device_name,
                    replayCacheTags(segment_cache));
            }
            // Production MTP sidecar replay can deliberately defer the final
            // stream sync. When GPU stage timing is requested, synchronize only
            // the replay stop event here so exported stage_gpu graph-replay
            // rows are true GPU elapsed time, not host enqueue duration.
            collect_replay_gpu_events(/*synchronize_total_event=*/true);
            result.success = true;
            return result;
        }
        {
            auto sync_t0 = std::chrono::high_resolution_clock::now();
            if (!gpu_ctx->synchronizeStreamChecked(capture_stream))
            {
                LOG_ERROR("[DeviceGraphCaptureController] Final replay capture-stream sync failed");
                return result;
            }
            auto sync_capture_t1 = std::chrono::high_resolution_clock::now();
            if (!gpu_ctx->synchronizeStreamChecked(gpu_ctx->defaultStream()))
            {
                LOG_ERROR("[DeviceGraphCaptureController] Final replay default-stream sync failed");
                return result;
            }
            auto sync_t1 = std::chrono::high_resolution_clock::now();
            if (profiling)
            {
                ForwardPassProfiler::addReplayStreamSyncNs(
                    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(sync_t1 - sync_t0).count()));
            }
            if (PerfStatsCollector::isEnabled())
            {
                auto capture_tags = replayCacheTags(segment_cache);
                capture_tags.emplace("stream", "capture");
                auto default_tags = replayCacheTags(segment_cache);
                default_tags.emplace("stream", "default");
                PerfStatsCollector::recordTimingNs(
                    "forward_graph",
                    "segmented_replay_stream_sync",
                    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(sync_capture_t1 - sync_t0).count()),
                    "decode",
                    device_name,
                    std::move(capture_tags));
                PerfStatsCollector::recordTimingNs(
                    "forward_graph",
                    "segmented_replay_stream_sync",
                    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(sync_t1 - sync_capture_t1).count()),
                    "decode",
                    device_name,
                    std::move(default_tags));
                PerfStatsCollector::recordTimingNs(
                    "forward_graph",
                    "segmented_replay_final_sync",
                    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(sync_t1 - sync_t0).count()),
                    "decode",
                    device_name,
                    graphReplayHostTimingTags(replayCacheTags(segment_cache), "final_stream_sync_host_wall"));
            }
        }
        collect_replay_gpu_events(/*synchronize_total_event=*/false);
        if (trace_replay)
        {
            LOG_DEBUG("[ReplayTrace] " << device_id.toString()
                                       << " step=" << current_step << " final synchronize() complete");
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
