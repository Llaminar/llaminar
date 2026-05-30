/**
 * @file DeviceGraphExecutor_GraphCapture.cpp
 * @brief GPU graph capture/replay implementation for DeviceGraphExecutor
 *
 * Split from DeviceGraphExecutor.cpp to isolate the GPU graph capture subsystem.
 * Contains:
 * - GraphSegmentCache resource management (streams, events)
 * - executeWithGraphCapture (single-graph capture/replay)
 * - executeDecodeWithCapturePolicy (policy-based mode selection)
 * - executeWithSegmentedGraphCapture (segmented capture/replay)
 */

#include "DeviceGraphExecutor.h"
#include "DeviceGraphCaptureController.h"
#include "../coherence/StageCoherence.h"
#include "../../../tensors/TensorClasses.h"
#include "../../../utils/Logger.h"
#include "../../../memory/BufferArena.h"
#include "../../../backends/IGPUGraphCapture.h"
#include "../../../backends/IWorkerGPUContext.h"

namespace llaminar2
{

    // =========================================================================
    // GraphSegmentCache — capture stream management
    // =========================================================================

    bool DeviceGraphExecutor::GraphSegmentCache::ensureCaptureStream(IWorkerGPUContext *ctx)
    {
        if (capture_stream)
            return true;
        if (!ctx)
        {
            LOG_ERROR("[GraphSegmentCache] No GPU context for stream creation");
            return false;
        }
        capture_stream = ctx->createStream();
        if (!capture_stream)
        {
            LOG_ERROR("[GraphSegmentCache] Failed to create capture stream");
            return false;
        }
        gpu_ctx_ref = ctx;
        LOG_DEBUG("[GraphSegmentCache] Created local capture stream");
        return true;
    }

    void DeviceGraphExecutor::GraphSegmentCache::destroyCaptureStream()
    {
        if (!capture_stream)
            return;
        if (gpu_ctx_ref)
            gpu_ctx_ref->destroyStream(capture_stream);
        capture_stream = nullptr;
        gpu_ctx_ref = nullptr;
    }

    bool DeviceGraphExecutor::GraphSegmentCache::ensureSyncEvent(IWorkerGPUContext *ctx)
    {
        if (sync_event)
            return true;
        if (!ctx)
            return false;
        sync_event = ctx->createEvent();
        if (!sync_event)
        {
            LOG_ERROR("[GraphSegmentCache] Failed to create sync event");
            return false;
        }
        return true;
    }

    void DeviceGraphExecutor::GraphSegmentCache::destroySyncEvent()
    {
        if (!sync_event)
            return;
        if (gpu_ctx_ref)
            gpu_ctx_ref->destroyEvent(sync_event);
        sync_event = nullptr;
    }

    // =========================================================================
    // Single-Graph Capture/Replay
    // =========================================================================

    bool DeviceGraphExecutor::executeWithGraphCapture(ComputeGraph &graph, IDeviceContext *ctx,
                                                      IGPUGraphCapture *capture,
                                                      const std::unordered_set<std::string> *collective_nodes,
                                                      void *gpu_stream)
    {
        if (!capture)
        {
            LOG_WARN("[DeviceGraphExecutor] GPU graph capture is null, falling back to fast decode");
            return executeFastDecode(graph, ctx, collective_nodes);
        }

        // TP>1 with collectives cannot be captured in a single-device graph
        if (collective_nodes && !collective_nodes->empty())
        {
            LOG_DEBUG("[DeviceGraphExecutor] Collective nodes present, skipping graph capture for TP>1");
            return executeFastDecode(graph, ctx, collective_nodes);
        }

        // Step 1: Begin capture
        if (!capture->beginCapture())
        {
            LOG_WARN("[DeviceGraphExecutor] GPU graph beginCapture failed, falling back to fast decode");
            return executeFastDecode(graph, ctx, collective_nodes);
        }

        // Step 2: Execute all stages into the captured stream
        // Set GPU device once before the loop (same as executeFastDecode)
        DeviceGraphCaptureController::prepareDeviceForSegmentedCapture(ctx);

        const auto &order = graph.getExecutionOrder();
        bool exec_success = true;

        // Set GPU stream on all stages so kernels dispatch to the capture stream
        if (gpu_stream)
        {
            for (const auto &name : order)
            {
                auto *node = graph.getNode(name);
                if (node && node->stage)
                    node->stage->setGPUStream(gpu_stream);
            }
        }

        for (const auto &name : order)
        {
            auto *node = graph.getNode(name);
            if (!node->stage->execute(ctx))
            {
                LOG_ERROR("[DeviceGraphExecutor] Stage failed during graph capture: " << name);
                exec_success = false;
                break;
            }
            graph.markCompleted(name);
        }

        // Step 3: End capture
        if (!exec_success || !capture->endCapture())
        {
            LOG_WARN("[DeviceGraphExecutor] GPU graph capture failed (exec_success=" << exec_success
                                                                                     << "), stream may be in bad state");
            // If capture was started but execute failed, we still need to end capture
            // to restore the stream to a usable state
            if (exec_success)
            {
                // endCapture failed
                capture->reset();
            }
            // Fall through — the stream should be usable again after endCapture
            // The kernels were recorded during capture but NOT executed
            return exec_success;
        }

        // If graph captured 0 nodes, all work was CPU-side (already executed
        // during capture).  Skip instantiate/update/launch — there is nothing
        // to replay.
        if (capture->nodeCount() == 0)
        {
            LOG_WARN("[DeviceGraphExecutor] GPU graph captured 0 nodes — kernels NOT on capture stream! Skipping graph replay");
            return true;
        }

        // Step 4: Instantiate or update + launch
        LOG_WARN("[DeviceGraphExecutor] GPU graph captured " << capture->nodeCount()
                                                             << " nodes, hasExecutable=" << capture->hasExecutable());
        if (capture->hasExecutable())
        {
            // Try in-place update
            GraphUpdateResult result = capture->tryUpdate();
            LOG_WARN("[DeviceGraphExecutor] tryUpdate result=" << static_cast<int>(result));
            if (result == GraphUpdateResult::Success)
            {
                // In-place update succeeded — launch the updated executable
                if (!capture->launch())
                {
                    LOG_ERROR("[DeviceGraphExecutor] GPU graph launch failed after update");
                    return false;
                }
                LOG_TRACE("[DeviceGraphExecutor] GPU graph updated and launched ("
                          << capture->nodeCount() << " nodes)");
                return true;
            }
            else if (result == GraphUpdateResult::NeedsReinstantiate)
            {
                // Topology changed — reinstantiate
                LOG_WARN("[DeviceGraphExecutor] NeedsReinstantiate — calling instantiate()");
                if (!capture->instantiate())
                {
                    LOG_WARN("[DeviceGraphExecutor] GPU graph reinstantiation failed");
                    return false;
                }
            }
            else
            {
                // Update failed
                LOG_WARN("[DeviceGraphExecutor] GPU graph update failed (result="
                         << static_cast<int>(result) << ")");
                return false;
            }
        }
        else
        {
            // First time — instantiate from captured graph
            LOG_WARN("[DeviceGraphExecutor] First instantiation attempt (" << capture->nodeCount() << " nodes)");
            if (!capture->instantiate())
            {
                LOG_WARN("[DeviceGraphExecutor] GPU graph instantiation failed");
                return false;
            }
            LOG_WARN("[DeviceGraphExecutor] GPU graph instantiated with " << capture->nodeCount()
                                                                          << " nodes (" << capture->backendName() << ")");
        }

        // Launch the (newly instantiated) executable
        if (!capture->launch())
        {
            LOG_ERROR("[DeviceGraphExecutor] GPU graph launch failed");
            return false;
        }

        return true;
    }

    // =========================================================================
    // Decode Capture Policy
    // =========================================================================

    bool DeviceGraphExecutor::executeDecodeWithCapturePolicy(
        ComputeGraph &graph,
        IDeviceContext *ctx,
        GraphSegmentCache *segment_cache,
        void *gpu_stream,
        IWorkerGPUContext *gpu_ctx,
        const std::unordered_set<std::string> *collective_nodes,
        const DecodeCapturePolicy &policy,
        bool *used_segmented_capture)
    {
        if (used_segmented_capture)
        {
            *used_segmented_capture = false;
        }

        if (!policy.allow_fast_decode)
        {
            return execute(graph, ctx);
        }

        const bool segmented_ready =
            policy.allow_segmented_capture &&
            segment_cache &&
            gpu_stream &&
            gpu_ctx &&
            segment_cache->consecutive_failures < policy.max_segment_failures;

        if (segmented_ready)
        {
            bool success = executeWithSegmentedGraphCapture(
                graph,
                ctx,
                *segment_cache,
                gpu_stream,
                gpu_ctx,
                collective_nodes,
                policy.collectives_graph_capturable);

            if (success)
            {
                if (used_segmented_capture)
                {
                    *used_segmented_capture = true;
                }
                return true;
            }

            LOG_WARN("[DeviceGraphExecutor] Segmented replay failed under policy, falling back to fast decode");
            graph.reset();
            return executeFastDecode(graph, ctx, collective_nodes);
        }

        return executeFastDecode(graph, ctx, collective_nodes);
    }

    // =========================================================================
    // Segmented GPU Graph Capture/Replay
    // =========================================================================

    bool DeviceGraphExecutor::executeWithSegmentedGraphCapture(ComputeGraph &graph, IDeviceContext *ctx,
                                                               GraphSegmentCache &segment_cache,
                                                               void *gpu_stream,
                                                               IWorkerGPUContext *gpu_ctx,
                                                               const std::unordered_set<std::string> *collective_nodes,
                                                               bool collectives_graph_capturable)
    {
        if (!gpu_stream || !gpu_ctx)
        {
            LOG_WARN("[DeviceGraphExecutor] Segmented graph capture: missing stream or gpu_ctx, falling back");
            return executeFastDecode(graph, ctx);
        }

        const bool has_collective_nodes = (collective_nodes && !collective_nodes->empty());

        // Monotonic step counter + phase transition selection for segmented mode.
        const auto phase_transition = DeviceGraphCaptureController::beginStep(
            segment_cache.initialized,
            segment_cache.needs_capture,
            segment_cache.decode_step);
        const uint64_t current_step = phase_transition.decode_step;

        // ===== FAST PATH: Phase 3 (Replay) =====
        // During steady-state replay, only the post_launch hook is invoked
        // (cohere_inputs is skipped for capturable segments, execute_node is
        // unused). Avoid constructing unused lambdas and skip phase 1/2 checks
        // to minimize host overhead on the hot decode path.
        if (phase_transition.phase == DeviceGraphCaptureController::Phase::Replay)
        {
            DeviceGraphCaptureController::prepareDeviceForSegmentedCapture(ctx);

            DeviceGraphCaptureController::ReplayHooks fast_hooks{
                nullptr, // cohere_inputs — skipped during normal replay (skip_coherence=true)
                [&](ComputeNode &node) -> bool
                {
                    return executeNode(node, ctx);
                },
                [&](DeviceGraphExecutor::GraphSegment &seg, void *stream)
                {
                    // postCapturedSegmentLaunch uses its internal cached dirty
                    // marking (cached_dirty_tensor_bases), so the per-node
                    // mark_stage_outputs_dirty_cb is never called. Pass a no-op.
                    DeviceGraphCaptureController::postCapturedSegmentLaunch(
                        graph, seg, current_step, stream,
                        [](ComputeNode &, void *) {});
                }};

            const auto replay_result = DeviceGraphCaptureController::executeReplayPhase(
                graph, segment_cache, ctx, gpu_ctx,
                has_collective_nodes, current_step, fast_hooks);

            if (!replay_result.success)
            {
                if (replay_result.launch_failure_fallback)
                {
                    segment_cache.consecutive_failures++;
                    if (segment_cache.consecutive_failures >= GraphSegmentCache::kMaxFailures)
                    {
                        LOG_WARN("[DeviceGraphExecutor] Too many segmented graph failures, disabling");
                        segment_cache.reset(GraphSegmentCache::StreamResetPolicy::Preserve);
                    }
                    graph.reset();
                    return executeFastDecode(graph, ctx, collective_nodes);
                }
                return false;
            }

            segment_cache.consecutive_failures = 0;
            return true;
        }

        // ===== SLOW PATH: Phase 1 (Warmup) or Phase 2 (Capture) =====
        auto mark_stage_outputs_dirty = [&](ComputeNode &node, void *stream)
        {
            if (!arena_)
                return;
            const StageBufferContract contract = node.stage->bufferContract();
            if (contract.empty())
                return;
            DeviceId target_device = node.device.is_valid() ? node.device : node.stage->device();
            for (const auto &binding : contract.allWrites())
            {
                arena_->markWritten(binding.id, target_device, stream);
            }
        };

        auto post_captured_segment_launch = [&](GraphSegment &seg, void *stream)
        {
            DeviceGraphCaptureController::postCapturedSegmentLaunch(
                graph,
                seg,
                current_step,
                stream,
                [&](ComputeNode &node, void *node_stream)
                {
                    mark_stage_outputs_dirty(node, node_stream);
                });
        };

        auto cohere_replay_stage = [&](ComputeNode &node) -> bool
        {
            const auto policy = node.stage->coherencePolicy();
            if (policy != CoherencePolicy::INPUT && policy != CoherencePolicy::FULL)
            {
                return true;
            }

            if (!arena_)
            {
                LOG_ERROR("[DeviceGraphExecutor] No arena for replay coherence on stage: " << node.name);
                return false;
            }

            DeviceId target_device = node.device.is_valid() ? node.device : node.stage->device();
            const StageBufferContract contract = node.stage->bufferContract();

            // Cohere arena-managed reads (inputs + inouts)
            for (const auto &binding : contract.allArenaReads())
            {
                if (!arena_->prepareForRead(binding.id, target_device))
                {
                    LOG_ERROR("[DeviceGraphExecutor] Arena prepareForRead failed for replay stage: " << node.name);
                    return false;
                }
            }

            // Cohere weights (not arena-managed)
            if (!node.weights_cohered)
            {
                for (auto *weight : contract.weight_tensors)
                {
                    if (auto *tb = dynamic_cast<TensorBase *>(weight))
                    {
                        tb->ensureOnDevice(target_device);
                    }
                }
                node.weights_cohered = true;
            }

            // Cohere arena-managed writes (outputs + inouts)
            for (const auto &binding : contract.allWrites())
            {
                if (!arena_->prepareForWrite(binding.id, target_device))
                {
                    LOG_ERROR("[DeviceGraphExecutor] Arena prepareForWrite failed for replay stage: " << node.name);
                    return false;
                }
            }

            return true;
        };

        auto cohere_segment_inputs = [&](const GraphSegment &seg) -> bool
        {
            return DeviceGraphCaptureController::cohereReplaySegmentInputs(
                graph,
                seg,
                [&](ComputeNode &node)
                {
                    return cohere_replay_stage(node);
                });
        };

        DeviceGraphCaptureController::prepareDeviceForSegmentedCapture(ctx);

        DeviceGraphCaptureController::ReplayHooks replay_hooks{
            [&](const GraphSegment &segment)
            {
                return cohere_segment_inputs(segment);
            },
            [&](ComputeNode &node)
            {
                return executeNode(node, ctx);
            },
            [&](GraphSegment &segment, void *stream)
            {
                post_captured_segment_launch(segment, stream);
            }};

        // Capture-phase hooks: same as replay hooks except post_launch skips
        // onGraphReplayed() callbacks. During capture, execute() already ran
        // host-side bookkeeping; calling onGraphReplayed() would double-advance
        // KV cache head positions and corrupt subsequent decode steps.
        DeviceGraphCaptureController::ReplayHooks capture_hooks{
            replay_hooks.cohere_inputs,
            replay_hooks.execute_node,
            [&](GraphSegment &segment, void *stream)
            {
                DeviceGraphCaptureController::postCapturedSegmentLaunch(
                    graph,
                    segment,
                    current_step,
                    stream,
                    [&](ComputeNode &node, void *node_stream)
                    {
                        mark_stage_outputs_dirty(node, node_stream);
                    },
                    /*skip_replay_callbacks=*/true);
            }};

        // ===== Phase 1: Warmup (first call) — build segments, execute normally =====
        // We do NOT capture on the first call. Some kernels lazily initialize workspace
        // buffers (hipMalloc), which isn't compatible with stream capture.
        // First call builds the segment list and runs via executeFastDecode.
        //
        // CRITICAL: Run warmup on the CAPTURE stream (not default stream). CK and
        // ROCm kernel dispatch may cache per-stream state (dispatch tables, workspace
        // allocations). If warmup runs on the default stream, the capture stream sees
        // "fresh" kernel contexts that trigger capture-unsafe lazy initialization,
        // causing intermittent "operation failed due to a previous error during capture".
        if (phase_transition.phase == DeviceGraphCaptureController::Phase::Warmup)
        {
            DeviceGraphCaptureController::executeWarmupPhase(
                graph,
                segment_cache,
                collective_nodes,
                has_collective_nodes,
                collectives_graph_capturable);

            // Create the capture stream early so warmup runs on it.
            if (segment_cache.ensureCaptureStream(gpu_ctx))
            {
                void *warmup_stream = segment_cache.capture_stream;

                // Synchronize prior work on the old stream before switching.
                // Previous decode steps (e.g., KV cache writes) may still be
                // in-flight on gpu_stream. Without this dependency, the warmup
                // on capture_stream could read stale KV data, causing the model
                // to repeat the previous token (token duplication bug).
                if (gpu_stream && gpu_stream != warmup_stream)
                {
                    gpu_ctx->synchronize();
                }

                // Point all stages at the capture stream for warmup execution.
                for (const auto &name : graph.getExecutionOrder())
                {
                    auto *node = graph.getNode(name);
                    if (node && node->stage)
                    {
                        node->stage->setGPUStream(warmup_stream);
                    }
                }
            }

            // Warmup executes all stages normally (no capture) to ensure
            // lazy kernel initialization and workspace allocation complete
            // on the capture stream.
            const bool warmup_ok = executeFastDecode(graph, ctx, collective_nodes);
            if (warmup_ok)
            {
                // Some stages become capturable only after warmup seeds their
                // persistent runtime state. MoE decode is the important case:
                // its first execution uploads runtime placement banks and grouped
                // descriptor tables, after which routing/expert stages can be
                // captured safely. Rebuild the segment plan before Phase 2 so
                // capture sees the warmed capabilities instead of the cold ones.
                DeviceGraphCaptureController::buildWarmupSegments(
                    graph,
                    segment_cache,
                    collective_nodes,
                    has_collective_nodes,
                    collectives_graph_capturable);
            }
            return warmup_ok;
        }

        // ===== Phase 2: Capture (second call) — record capturable segments =====
        if (phase_transition.phase == DeviceGraphCaptureController::Phase::Capture)
        {
            const auto capture_result = DeviceGraphCaptureController::executeCapturePhase(
                graph,
                segment_cache,
                ctx,
                gpu_ctx,
                has_collective_nodes,
                current_step,
                capture_hooks);

            if (capture_result.reset_cache)
            {
                segment_cache.reset(GraphSegmentCache::StreamResetPolicy::Preserve);
            }

            if (!capture_result.success)
            {
                if (capture_result.fallback_to_fast_decode)
                {
                    // The partial capture marked some nodes completed —
                    // reset so fast decode re-executes all stages.
                    graph.reset();
                    return executeFastDecode(graph, ctx, collective_nodes);
                }
                return false;
            }

            return true;
        }

        // Phase 3 is handled by the fast path above; this point is unreachable
        // after Phase 1 and Phase 2 both early-return.
        LOG_ERROR("[DeviceGraphExecutor] Unexpected phase in segmented graph capture");
        return false;
    }

} // namespace llaminar2
