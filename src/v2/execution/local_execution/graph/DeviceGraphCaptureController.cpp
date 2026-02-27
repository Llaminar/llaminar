#include "DeviceGraphCaptureController.h"

#include "../coherence/StageCoherence.h"
#include "../../../tensors/TensorClasses.h"
#include "../../../utils/DebugEnv.h"
#include "../../../utils/Logger.h"
#ifdef HAVE_ROCM
#include "../../../backends/rocm/HipDeviceGuard.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace llaminar2
{

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
#ifdef HAVE_ROCM
        if (ctx && ctx->deviceId().type == DeviceType::ROCm)
        {
            HipDeviceGuard::forceSetDevice(ctx->deviceId().toKernelDeviceIndex());
        }
#else
        (void)ctx;
#endif
    }

    void DeviceGraphCaptureController::executeWarmupPhase(
        ComputeGraph &graph,
        DeviceGraphExecutor::GraphSegmentCache &segment_cache,
        const std::unordered_set<std::string> *collective_nodes,
        bool has_collective_nodes)
    {
        buildWarmupSegments(
            graph,
            segment_cache,
            collective_nodes,
            has_collective_nodes);
        markWarmupComplete(
            segment_cache.initialized,
            segment_cache.needs_capture);
    }

    void DeviceGraphCaptureController::buildWarmupSegments(
        ComputeGraph &graph,
        DeviceGraphExecutor::GraphSegmentCache &segment_cache,
        const std::unordered_set<std::string> *collective_nodes,
        bool has_collective_nodes)
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

        for (const auto &name : order)
        {
            auto *node = graph.getNode(name);
            if (!node || !node->stage)
            {
                continue;
            }

            // Start from stage capability, then layer on safety gates.
            bool stage_capturable = node->stage->isGraphCapturable();
            const bool collective_by_type = is_collective_stage(node->stage->type());
            const bool collective_by_name = (collective_nodes && collective_nodes->count(name));
            if (collective_by_type || collective_by_name)
            {
                stage_capturable = false;
            }

            if (has_collective_nodes && !segmented_collective_capture_allow.empty())
            {
                stage_capturable = stage_in_collective_allowlist(name);
            }

            if (has_collective_nodes && node->stage->hasDynamicParams())
            {
                stage_capturable = false;
            }

            if (has_collective_nodes && node->stage->type() == ComputeStageType::ADD_RESIDUAL)
            {
                stage_capturable = false;
            }

            if (has_collective_nodes && (name == "final_norm" || name == "lm_head"))
            {
                stage_capturable = false;
            }

            if (first || stage_capturable != current_capturable)
            {
                // Create a new segment whenever capturable/manual mode changes
                // so each segment has uniform execution semantics.
                segment_cache.segments.emplace_back();
                segment_cache.segments.back().capturable = stage_capturable;
                current_capturable = stage_capturable;
                first = false;
            }

            segment_cache.segments.back().stage_names.push_back(name);
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
        for (const auto &seg : segment_cache.segments)
        {
            if (seg.capturable)
            {
                capturable_segments++;
                capturable_stages += seg.stage_names.size();
            }
            else
            {
                manual_segments++;
                manual_stages += seg.stage_names.size();
            }
        }

        LOG_INFO("[DeviceGraphExecutor] Segmented graph: " << capturable_segments << " capturable segments ("
                                                           << capturable_stages << " stages) + " << manual_segments << " manual segments ("
                                                           << manual_stages << " stages)");

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

        void *use_stream = use_default_stream ? nullptr : capture_stream;
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
                graph.markCompleted(stage_name);
            }
        }

        gpu_ctx->synchronize();
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
            if (has_collective_nodes)
            {
                manual_had_collective = manual_had_collective || is_collective_stage(stage_type);

                if (needs_segment_sync)
                {
                    gpu_ctx->synchronizeStream(capture_stream);
                }

                node->stage->setGPUStream(nullptr);
                if (!execute_node_cb(*node))
                {
                    LOG_ERROR("[DeviceGraphCaptureController] Manual stage failed on replay (collective graph): " << stage_name);
                    return false;
                }

                gpu_ctx->synchronize();
            }
            else if (is_collective_stage(stage_type))
            {
                manual_had_collective = true;

                if (needs_segment_sync)
                {
                    gpu_ctx->synchronizeStream(capture_stream);
                }

                node->stage->setGPUStream(nullptr);
                if (!execute_node_cb(*node))
                {
                    LOG_ERROR("[DeviceGraphCaptureController] Manual collective stage failed on replay: " << stage_name);
                    return false;
                }

                gpu_ctx->synchronize();
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
        }

        segment.last_executed_step = current_step;
        if (needs_segment_sync)
        {
            if (manual_had_collective)
            {
                gpu_ctx->synchronize();
            }
            else
            {
                gpu_ctx->synchronizeStream(capture_stream);
            }
        }

        return true;
    }

    bool DeviceGraphCaptureController::executeCapturedReplaySegmentNormal(
        DeviceGraphExecutor::GraphSegment &segment,
        IWorkerGPUContext *gpu_ctx,
        void *capture_stream,
        bool needs_segment_sync,
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

        if (!segment.capture->launch())
        {
            LOG_ERROR("[DeviceGraphCaptureController] Segment graph launch failed on replay");
            return false;
        }

        post_launch_cb(segment, capture_stream);

        if (needs_segment_sync)
        {
            gpu_ctx->synchronizeStream(capture_stream);
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

        if (!segment.capture->beginCapture())
        {
            LOG_ERROR("[DeviceGraphCaptureController] Re-capture beginCapture failed, seg " << segment_index);
            return false;
        }

        bool exec_ok = true;
        for (const auto &stage_name : segment.stage_names)
        {
            auto *node = graph.getNode(stage_name);
            if (!node || !node->stage || !node->stage->execute(ctx))
            {
                exec_ok = false;
                break;
            }
        }

        if (!exec_ok || !segment.capture->endCapture())
        {
            LOG_ERROR("[DeviceGraphCaptureController] Re-capture failed, seg " << segment_index);
            return false;
        }

        auto update_result = segment.capture->tryUpdate();
        if (update_result == GraphUpdateResult::NeedsReinstantiate ||
            update_result == GraphUpdateResult::Failed)
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
        gpu_ctx->synchronize();

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
                gpu_ctx->synchronizeStream(capture_stream);
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
        gpu_ctx->synchronize();

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
            node->stage->setGPUStream(nullptr);
            if (!node->stage->execute(ctx))
            {
                LOG_ERROR("[DeviceGraphCaptureController] Verify: direct exec failed: " << stage_name);
                return result;
            }
        }
        gpu_ctx->synchronize();

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
            gpu_ctx->synchronize();
            LOG_DEBUG("[DeviceGraphCaptureController] Segment captured+executed (Phase-2 semantics): "
                      << segment.capture->nodeCount() << " nodes, " << segment.stage_names.size() << " stages");
            return true;
        }

        if (!segment.capture->launch())
        {
            LOG_ERROR("[DeviceGraphCaptureController] Segment initial launch failed");
            return false;
        }
        // Capture phase: pass skip_replay_callbacks=true because execute() already
        // ran host-side bookkeeping during capture recording. onGraphReplayed()
        // must only run during the replay phase (Phase 3).
        post_launch_cb(segment, capture_stream);
        gpu_ctx->synchronize();
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

        gpu_ctx->synchronize();

        for (const auto &stage_name : segment.stage_names)
        {
            auto *node = graph.getNode(stage_name);
            if (!node || !node->stage)
            {
                LOG_ERROR("[DeviceGraphCaptureController] Capture manual segment missing stage: " << stage_name);
                return false;
            }

            node->stage->setGPUStream(nullptr);

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

            graph.markCompleted(stage_name);
        }

        segment.last_executed_step = current_step;
        gpu_ctx->synchronize();
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

        if (!skip_coherence && !cohere_inputs_cb(segment))
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

        if (!segment_cache.ensureCaptureStream(gpu_ctx))
        {
            // No capture stream means segmented capture cannot proceed safely;
            // caller should fall back to fast decode for this step.
            LOG_WARN("[DeviceGraphCaptureController] Failed to create capture stream, falling back");
            result.fallback_to_fast_decode = true;
            return result;
        }
        void *capture_stream = segment_cache.capture_stream;

        initializeReplayCallbacks(graph, segment_cache);

        for (auto &seg : segment_cache.segments)
        {
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

                seg.capture = gpu_ctx->createGraphCapture(capture_stream);
                if (!seg.capture)
                {
                    LOG_ERROR("[DeviceGraphCaptureController] Failed to create graph capture for segment");
                    result.reset_cache = true;
                    result.fallback_to_fast_decode = true;
                    return result;
                }

                if (!seg.capture->beginCapture())
                {
                    LOG_ERROR("[DeviceGraphCaptureController] beginCapture failed for segment");
                    result.reset_cache = true;
                    result.fallback_to_fast_decode = true;
                    return result;
                }

                bool exec_ok = true;
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

                if (!exec_ok || !seg.capture->endCapture())
                {
                    LOG_ERROR("[DeviceGraphCaptureController] Segmented capture failed");
                    result.reset_cache = true;
                    result.success = exec_ok;
                    return result;
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
                // Manual path: execute with normal stage semantics.
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
        const ReplayHooks &hooks)
    {
        ReplayPhaseResult result{};

        if (!ctx || !gpu_ctx)
        {
            LOG_ERROR("[DeviceGraphCaptureController] Replay phase missing context");
            return result;
        }

        void *capture_stream = segment_cache.capture_stream;
        const auto &exec_cfg = debugEnv().execution;
        const bool verify_mode = exec_cfg.gpu_graph_verify;
        const bool recapture_mode = exec_cfg.gpu_graph_recapture;
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

        int seg_idx = 0;
        for (auto &seg : segment_cache.segments)
        {
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
                hooks.cohere_inputs,
                hooks.execute_node,
                hooks.post_launch);

            if (!replay_result.success)
            {
                result.launch_failure_fallback = replay_result.launch_failure_fallback;
                return result;
            }

            seg_idx++;
        }

        gpu_ctx->synchronize();
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
        void *stream,
        const std::function<void(ComputeNode &, void *)> &mark_stage_outputs_dirty_cb,
        bool skip_replay_callbacks)
    {
        // OPTIMIZATION: Cache output buffers on first replay to avoid per-step
        // getDumpInfo() + extractOutputBuffers() + dynamic_cast overhead.
        // For Qwen2.5-7B (338 stages), this eliminates ~1352 vector allocations
        // and ~676 virtual calls per decode step.
        if (!segment.replay_buffers_cached)
        {
            segment.cached_all_output_buffers.clear();
            for (const auto &stage_name : segment.stage_names)
            {
                auto *node = graph.getNode(stage_name);
                if (!node || !node->stage)
                {
                    continue;
                }

                const auto &dump_info = node->stage->getDumpInfo();
                for (const auto &output : dump_info.outputs)
                {
                    CoherenceBuffer buf;
                    buf.tensor = output.tensor;
                    buf.name = output.name;
                    buf.data = output.data;
                    buf.rows = output.rows;
                    buf.cols = output.cols;
                    buf.dtype = output.dtype;
                    buf.is_inout = false;
                    segment.cached_all_output_buffers.push_back(buf);
                }
            }
            segment.replay_buffers_cached = true;

            LOG_DEBUG("[DeviceGraphCaptureController] Cached " << segment.cached_all_output_buffers.size()
                                                               << " output buffers for " << segment.stage_names.size() << " stages");
        }

        // Use flags-only dirty marking — no hipEventRecord per output tensor.
        // The final gpu_ctx->synchronize() in executeReplayPhase ensures all
        // GPU work completes before the caller reads the output.
        markOutputsDirtyFlagsOnly(segment.cached_all_output_buffers);

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
