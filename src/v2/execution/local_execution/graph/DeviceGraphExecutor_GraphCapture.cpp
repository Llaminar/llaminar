/**
 * @file DeviceGraphExecutor_GraphCapture.cpp
 * @brief GPU graph capture/replay implementation for DeviceGraphExecutor
 *
 * Split from DeviceGraphExecutor.cpp to isolate the GPU graph capture subsystem.
 * Contains:
 * - GraphSegmentCache resource management (streams, events)
 * - executeWithGraphCapture (single-graph capture/replay)
 * - executeDecodeWithCapturePolicy (policy-based mode selection)
 * - executeWithCachedGraphReplay (cached graph replay)
 */

#include "DeviceGraphExecutor.h"
#include "DeviceGraphCaptureController.h"
#include "GraphCaptureGuard.h"
#include "../coherence/CoherencePolicy.h"
#include "../../../tensors/TensorClasses.h"
#include "../../../utils/Logger.h"
#include "../../../utils/PerfStatsCollector.h"
#include "../../../memory/BufferArena.h"
#include "../../../backends/GPUDeviceContextPool.h"
#include "../../../backends/IGPUGraphCapture.h"
#include "../../../backends/IWorkerGPUContext.h"
#include "../../../transfer/TransferEngine.h"

#include <sstream>
#include <unordered_set>

namespace llaminar2
{
    namespace
    {
        /**
         * @brief Rebind every GPU stage for an explicitly selected eager pass.
         *
         * Capture and replay phases deliberately point stages at a graph-owned
         * stream. A caller that explicitly disables cached graph replay must
         * therefore establish one worker-owned stream for the complete eager
         * graph before execution. This helper is not an error-recovery path:
         * once replay is selected, replay failure is a hard execution failure.
         *
         * @param graph Graph whose stages will execute eagerly by policy.
         * @param gpu_stream Worker-owned stream supplied by the capture policy.
         * CPU-only graphs legitimately have no GPU stream. The helper first
         * validates the complete graph and determines whether any stage is
         * device-backed; only then does it require and publish the explicit
         * stream. This keeps CPU decode usable without weakening the GPU
         * ownership contract.
         *
         * @return true when every GPU graph stage was rebound, or when the
         *         graph is entirely CPU-owned.
         */
        bool bindGraphStagesForEagerExecution(ComputeGraph &graph, void *gpu_stream)
        {
            bool has_gpu_stage = false;

            for (const auto &name : graph.getExecutionOrder())
            {
                ComputeNode *node = graph.getNode(name);
                if (!node || !node->stage)
                {
                    LOG_ERROR("[DeviceGraphExecutor] Capture-policy eager execution cannot bind missing stage '"
                              << name << "'");
                    return false;
                }
                const DeviceId stage_device =
                    node->stage->device().is_valid()
                        ? node->stage->device()
                        : node->device;
                has_gpu_stage = has_gpu_stage || stage_device.is_gpu();
            }

            if (has_gpu_stage && !gpu_stream)
            {
                LOG_ERROR(
                    "[DeviceGraphExecutor] Capture-policy eager GPU execution "
                    "requires an explicit non-null stream");
                return false;
            }

            if (has_gpu_stage)
            {
                for (const auto &name : graph.getExecutionOrder())
                {
                    ComputeNode *node = graph.getNode(name);
                    node->stage->setGPUStream(gpu_stream);
                }
            }
            return true;
        }

        /**
         * @brief Terminate after a graph-cache resource loses a valid lifecycle.
         *
         * A non-null stream or event is an owned backend resource. Once its
         * owner cannot be resolved, synchronization fails, or destruction
         * throws, neither dropping the handle nor continuing inference is
         * correct: queued work may still reference graph state that the host is
         * about to release. Centralizing the fatal policy here prevents future
         * lifecycle call sites from quietly reintroducing log-and-continue
         * behavior.
         *
         * @param detail Exact ownership or backend operation that failed.
         */
        [[noreturn]] void terminateGraphSegmentCacheLifecycle(
            const std::string &detail) noexcept
        {
            LOG_ERROR(
                "[GraphSegmentCache] Fatal GPU graph resource lifecycle "
                "violation: "
                << detail);
            std::terminate();
        }
    }

    // =========================================================================
    // GraphSegmentCache — capture stream management
    // =========================================================================

    IWorkerGPUContext *DeviceGraphExecutor::GraphSegmentCache::requireLifecycleContext(
        const char *operation)
    {
        if (capture_device.is_gpu() && capture_context_from_pool)
        {
            try
            {
                IWorkerGPUContext &resolved =
                    GPUDeviceContextPool::instance().getContext(capture_device);
                gpu_ctx_ref = &resolved;
                return &resolved;
            }
            catch (const std::exception &e)
            {
                terminateGraphSegmentCacheLifecycle(
                    std::string("failed to resolve the owning GPU context for ") +
                    operation + " on " + capture_device.toString() + ": " +
                    e.what());
            }
        }

        if (!gpu_ctx_ref)
        {
            terminateGraphSegmentCacheLifecycle(
                std::string("owned resource has no live GPU context during ") +
                operation);
        }
        return gpu_ctx_ref;
    }

    bool DeviceGraphExecutor::GraphSegmentCache::ensureCaptureStream(
        IWorkerGPUContext *ctx,
        DeviceId device,
        bool context_from_process_pool)
    {
        if (capture_stream)
        {
            if (capture_device.is_valid() &&
                device.is_valid() &&
                !(capture_device == device))
            {
                LOG_ERROR("[GraphSegmentCache] Capture stream already belongs to "
                          << capture_device.toString() << ", cannot reuse for "
                          << device.toString());
                return false;
            }
            if (!capture_device.is_valid() && device.is_gpu())
            {
                capture_device = device;
            }
            capture_context_from_pool =
                capture_context_from_pool || context_from_process_pool;
            if (!gpu_ctx_ref)
                gpu_ctx_ref = ctx;
            if (!gpu_ctx_ref &&
                !(capture_device.is_gpu() && capture_context_from_pool))
            {
                terminateGraphSegmentCacheLifecycle(
                    "existing capture stream has no resolvable owner");
            }
            return true;
        }
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
        capture_device = device.is_gpu() ? device : DeviceId::invalid();
        capture_context_from_pool = context_from_process_pool;
        LOG_DEBUG("[GraphSegmentCache] Created local capture stream"
                  << (capture_device.is_valid()
                          ? std::string(" for ") + capture_device.toString()
                          : std::string{}));
        return true;
    }

    void DeviceGraphExecutor::GraphSegmentCache::waitForCaptureStreamFence()
    {
        if (!capture_stream)
            return;

        IWorkerGPUContext *ctx =
            requireLifecycleContext("capture stream event fence");

        try
        {
            if (!ensureSyncEvent(ctx))
            {
                terminateGraphSegmentCacheLifecycle(
                    "capture stream event fence allocation was rejected by the backend");
            }
            if (!ctx->recordEventChecked(sync_event, capture_stream))
            {
                terminateGraphSegmentCacheLifecycle(
                    "capture stream event fence publication was rejected by the backend");
            }
            if (!ctx->synchronizeEventChecked(sync_event))
            {
                terminateGraphSegmentCacheLifecycle(
                    "capture stream event fence wait was rejected by the backend");
            }
        }
        catch (const std::exception &e)
        {
            terminateGraphSegmentCacheLifecycle(
                std::string("capture stream event fence threw: ") +
                e.what());
        }
        catch (...)
        {
            terminateGraphSegmentCacheLifecycle(
                "capture stream event fence threw an unknown exception");
        }
    }

    void DeviceGraphExecutor::GraphSegmentCache::destroyCaptureStream()
    {
        if (!capture_stream)
        {
            gpu_ctx_ref = nullptr;
            capture_device = DeviceId::invalid();
            capture_context_from_pool = false;
            return;
        }

        IWorkerGPUContext *ctx =
            requireLifecycleContext("capture stream destruction");
        try
        {
            ctx->destroyStream(capture_stream);
        }
        catch (const std::exception &e)
        {
            terminateGraphSegmentCacheLifecycle(
                std::string("capture stream destruction threw: ") + e.what());
        }
        catch (...)
        {
            terminateGraphSegmentCacheLifecycle(
                "capture stream destruction threw an unknown exception");
        }
        capture_stream = nullptr;
        gpu_ctx_ref = nullptr;
        capture_device = DeviceId::invalid();
        capture_context_from_pool = false;
    }

    bool DeviceGraphExecutor::GraphSegmentCache::ensureSyncEvent(IWorkerGPUContext *ctx)
    {
        if (sync_event)
        {
            if (!gpu_ctx_ref &&
                !(capture_device.is_gpu() && capture_context_from_pool))
            {
                terminateGraphSegmentCacheLifecycle(
                    "existing capture-stream handoff event has no resolvable owner");
            }
            return true;
        }
        if (!ctx)
            return false;
        if (gpu_ctx_ref && gpu_ctx_ref != ctx && !capture_context_from_pool)
        {
            terminateGraphSegmentCacheLifecycle(
                "capture-stream handoff event context differs from the capture "
                "stream owner");
        }
        sync_event = ctx->createEvent();
        if (!sync_event)
        {
            LOG_ERROR("[GraphSegmentCache] Failed to create sync event");
            return false;
        }
        if (!gpu_ctx_ref)
            gpu_ctx_ref = ctx;
        return true;
    }

    bool DeviceGraphExecutor::GraphSegmentCache::orderCaptureStreamAfter(
        IWorkerGPUContext *ctx,
        void *producer_stream)
    {
        if (!ctx || !producer_stream || !capture_stream)
        {
            LOG_ERROR(
                "[GraphSegmentCache] Capture-stream handoff requires one "
                "context and two explicit streams");
            return false;
        }
        if (producer_stream == capture_stream)
            return true;
        if (!ensureSyncEvent(ctx))
            return false;
        if (!ctx->recordEventChecked(sync_event, producer_stream))
        {
            LOG_ERROR(
                "[GraphSegmentCache] Failed to record capture-stream handoff "
                "event on the producer stream");
            return false;
        }
        if (!ctx->waitEventChecked(sync_event, capture_stream))
        {
            LOG_ERROR(
                "[GraphSegmentCache] Failed to queue capture-stream wait for "
                "the producer handoff event");
            return false;
        }
        return true;
    }

    void DeviceGraphExecutor::GraphSegmentCache::destroySyncEvent()
    {
        if (!sync_event)
            return;
        IWorkerGPUContext *ctx =
            requireLifecycleContext("sync event destruction");
        try
        {
            ctx->destroyEvent(sync_event);
        }
        catch (const std::exception &e)
        {
            terminateGraphSegmentCacheLifecycle(
                std::string("capture-stream handoff event destruction threw: ") +
                e.what());
        }
        catch (...)
        {
            terminateGraphSegmentCacheLifecycle(
                "capture-stream handoff event destruction threw an unknown "
                "exception");
        }
        sync_event = nullptr;
    }

    // =========================================================================
    // Single-Graph Capture/Replay
    // =========================================================================

    bool DeviceGraphExecutor::executeWithGraphCapture(ComputeGraph &graph, IDeviceContext *ctx,
                                                      IGPUGraphCapture *capture,
                                                      std::span<ITensor *const> externally_visible_outputs,
                                                      const std::unordered_set<std::string> *collective_nodes)
    {
        if (!capture)
        {
            LOG_ERROR("[DeviceGraphExecutor] GPU graph capture requires a non-null capture owner");
            return false;
        }

        // TP>1 with collectives cannot be captured in a single-device graph
        if (collective_nodes && !collective_nodes->empty())
        {
            LOG_ERROR("[DeviceGraphExecutor] Single-device graph capture cannot own collective nodes");
            return false;
        }

        if (!ctx || !ctx->deviceId().is_gpu())
        {
            LOG_ERROR(
                "[DeviceGraphExecutor] Single-graph capture requires an explicit "
                "GPU execution context");
            return false;
        }

        void *const gpu_stream = capture->executionStream();
        if (!gpu_stream)
        {
            LOG_ERROR(
                "[DeviceGraphExecutor] Single-graph capture owner has no explicit "
                "execution stream");
            return false;
        }

        /*
         * Capture-time tensor writes intentionally cannot publish host-waitable
         * completion events: those event records would become graph nodes and
         * describe recording rather than replay completion. The graph boundary
         * must therefore declare every tensor that may escape this call.
         * Validate the complete declaration before capture begins so an invalid
         * output cannot leave a partially published graph launch behind.
         */
        if (externally_visible_outputs.empty())
        {
            LOG_ERROR(
                "[DeviceGraphExecutor] Single-graph capture requires at least "
                "one externally visible output publication");
            return false;
        }

        std::unordered_set<ITensor *> unique_outputs;
        unique_outputs.reserve(externally_visible_outputs.size());
        for (ITensor *output : externally_visible_outputs)
        {
            if (!output)
            {
                LOG_ERROR(
                    "[DeviceGraphExecutor] Single-graph capture output "
                    "publication contains a null tensor");
                return false;
            }
            if (!unique_outputs.insert(output).second)
            {
                LOG_ERROR(
                    "[DeviceGraphExecutor] Single-graph capture output "
                    "publication contains a duplicate tensor");
                return false;
            }

            auto *base = dynamic_cast<TensorBase *>(output);
            if (!base || !base->gpu_data_ptr() ||
                !base->current_device().has_value() ||
                *base->current_device() != ctx->deviceId())
            {
                LOG_ERROR(
                    "[DeviceGraphExecutor] Single-graph capture output is not "
                    "resident on the execution device "
                    << ctx->deviceId().toString());
                return false;
            }
        }

        const auto &order = graph.getExecutionOrder();

        // Set GPU stream on all stages before any capture starts. Some stages
        // own tiny graph metadata buffers and must upload them on this explicit
        // stream before beginCapture(); doing so from execute() would record an
        // illegal H2D operation into the graph.
        for (const auto &name : order)
        {
            auto *node = graph.getNode(name);
            if (node && node->stage)
                node->stage->setGPUStream(gpu_stream);
        }

        for (const auto &name : order)
        {
            auto *node = graph.getNode(name);
            if (!node || !node->stage || !node->stage->needsGraphLaunchPreparation())
                continue;
            if (!node->stage->prepareGraphLaunch(ctx, gpu_stream))
            {
                LOG_ERROR("[DeviceGraphExecutor] Graph launch metadata preparation failed before capture: "
                          << name);
                return false;
            }
        }

        /*
         * Stage-owned launch preparation may publish metadata on an eager
         * stream. Import every resulting arena-read event onto the exact
         * capture stream before beginCapture(); stage execution inside the
         * transaction is validation-only and must never add an external wait.
         */
        if (!prepareInputsForGraphCapture(
                graph,
                ctx,
                gpu_stream,
                "single_graph_capture"))
        {
            LOG_ERROR("[DeviceGraphExecutor] Single-graph capture input preflight failed");
            return false;
        }

        // Step 1: Begin one structurally owned capture transaction.
        ScopedBackendGraphCapture capture_transaction(
            *capture,
            "single-graph capture");
        if (!capture_transaction.begin())
        {
            LOG_ERROR("[DeviceGraphExecutor] GPU graph beginCapture failed");
            return false;
        }

        // Step 2: Execute all stages into the captured stream
        // Set GPU device once before the loop (same as executeFastDecode)
        DeviceGraphCaptureController::prepareDeviceForGraphCapture(ctx);

        bool exec_success = true;

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

        // Step 3: Always leave native stream-capture mode before branching.
        capture_transaction.finish();
        if (!exec_success)
        {
            LOG_ERROR(
                "[DeviceGraphExecutor] A stage failed during mandatory "
                "single-graph capture; refusing eager recovery");
            capture->reset();
            return false;
        }

        /*
         * Selecting this API is an architectural assertion that every stage
         * in the graph emitted replayable device work onto the owned capture
         * stream. A zero-node result therefore means the stage binding,
         * residency preflight, or stream propagation contract is broken.
         * Treating the capture-time stage execution as a successful eager run
         * would let production silently leave graph mode.
         */
        if (capture->nodeCount() == 0)
        {
            LOG_ERROR(
                "[DeviceGraphExecutor] Mandatory GPU graph capture produced "
                "zero nodes; refusing capture-time eager execution");
            capture->reset();
            return false;
        }

        // Step 4: Select the one executable-publication operation supported by
        // the capture backend, then launch. This capability branch is static:
        // a backend that cannot update in place directly replaces its old
        // executable, without issuing a doomed update call and attempting to
        // recover from the resulting runtime error.
        LOG_DEBUG("[DeviceGraphExecutor] GPU graph captured " << capture->nodeCount()
                                                              << " nodes, hasExecutable=" << capture->hasExecutable());
        const bool can_update_existing_executable =
            capture->hasExecutable() && capture->supportsExecutableUpdate();
        if (can_update_existing_executable)
        {
            const GraphUpdateResult result = capture->tryUpdate();
            LOG_DEBUG("[DeviceGraphExecutor] tryUpdate result=" << static_cast<int>(result));
            if (result == GraphUpdateResult::Success)
            {
                LOG_TRACE("[DeviceGraphExecutor] GPU graph executable updated in place ("
                          << capture->nodeCount() << " nodes)");
            }
            else if (result == GraphUpdateResult::NeedsReinstantiate)
            {
                LOG_DEBUG("[DeviceGraphExecutor] NeedsReinstantiate; instantiating the captured topology");
                if (!capture->instantiate())
                {
                    LOG_ERROR("[DeviceGraphExecutor] GPU graph reinstantiation failed");
                    return false;
                }
            }
            else
            {
                // Update failed
                LOG_ERROR("[DeviceGraphExecutor] GPU graph update failed (result="
                          << static_cast<int>(result) << ")");
                return false;
            }
        }
        else
        {
            LOG_DEBUG("[DeviceGraphExecutor] Instantiating captured graph "
                      << (capture->hasExecutable() ? "as an executable replacement" : "for the first time")
                      << " (" << capture->nodeCount() << " nodes)");
            if (!capture->instantiate())
            {
                LOG_ERROR("[DeviceGraphExecutor] GPU graph instantiation failed");
                return false;
            }
            LOG_DEBUG("[DeviceGraphExecutor] GPU graph instantiated with " << capture->nodeCount()
                                                                           << " nodes (" << capture->backendName() << ")");
        }

        // Launch exactly once whether publication updated or replaced the
        // executable. A failed publication never reaches this point.
        if (!capture->launch())
        {
            LOG_ERROR("[DeviceGraphExecutor] GPU graph launch failed");
            return false;
        }

        /*
         * This is the sole externally visible publication boundary for the
         * direct single-graph API. Every event is recorded after launch on the
         * capture-owned stream, so a host read or another device stream consumes
         * the exact replay that produced the declared output. Event
         * creation/record failures throw from TransferEngine and intentionally
         * stop inference; there is no stream-synchronization fallback.
         */
        for (ITensor *output : externally_visible_outputs)
        {
            TransferEngine::publishDeviceWrite(
                output,
                ctx->deviceId(),
                gpu_stream);
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
        bool *used_graph_replay)
    {
        if (used_graph_replay)
        {
            *used_graph_replay = false;
        }

        if (!policy.allow_fast_decode)
        {
            return execute(graph, ctx);
        }

        if (policy.allow_cached_graph_replay)
        {
            if (!segment_cache || !gpu_stream || !gpu_ctx)
            {
                LOG_ERROR(
                    "[DeviceGraphExecutor] Cached graph replay policy requires "
                    "a segment cache, explicit stream, and live GPU context");
                return false;
            }
            bool success = executeWithCachedGraphReplay(
                graph,
                ctx,
                *segment_cache,
                gpu_stream,
                gpu_ctx,
                collective_nodes,
                policy.collectives_graph_capturable,
                policy.force_recapture,
                policy.defer_final_sync,
                policy.capture_boundary,
                policy.graph_replay_plan_policy);

            if (success)
            {
                if (used_graph_replay)
                {
                    *used_graph_replay = true;
                }
                return publishSnapshotsAfterGraphExecution(
                    graph,
                    segment_cache->capture_stream,
                    "decode_graph_replay");
            }

            const char *mode_name =
                segment_cache->initialized
                    ? DeviceGraphCaptureController::replayModeName(*segment_cache)
                    : "graph_replay";
            LOG_ERROR("[DeviceGraphExecutor] " << mode_name
                                               << " failed under mandatory cached graph replay policy");
            return false;
        }

        if (!bindGraphStagesForEagerExecution(graph, gpu_stream))
            return false;
        if (!executeFastDecode(graph, ctx, collective_nodes))
            return false;
        /*
         * executeFastDecode() publishes eager snapshots at each producer.  A
         * second post-graph pass is both redundant and incorrect for aliased
         * CPU buffers.  Post-graph publication remains mandatory above for a
         * real captured replay because only its recorded D2D copies preserve
         * every intermediate until host materialization.
         */
        return true;
    }

    // =========================================================================
    // Cached GPU Graph Replay
    // =========================================================================

    bool DeviceGraphExecutor::executeWithCachedGraphReplay(ComputeGraph &graph, IDeviceContext *ctx,
                                                               GraphSegmentCache &segment_cache,
                                                               void *gpu_stream,
                                                               IWorkerGPUContext *gpu_ctx,
                                                               const std::unordered_set<std::string> *collective_nodes,
                                                               bool collectives_graph_capturable,
                                                               bool force_recapture,
                                                               bool defer_final_sync,
                                                               GraphCaptureBoundaryHook capture_boundary,
                                                               GraphReplayPlanPolicy plan_policy)
    {
        if (!gpu_stream || !gpu_ctx)
        {
            LOG_ERROR("[DeviceGraphExecutor] GPU graph capture/replay requires an explicit stream and live GPU context");
            return false;
        }

        const bool has_collective_nodes = (collective_nodes && !collective_nodes->empty());

        const uint64_t current_variant_signature =
            DeviceGraphCaptureController::computeCaptureVariantSignature(graph);
        if (segment_cache.initialized &&
            segment_cache.capture_variant_signature != current_variant_signature)
        {
            LOG_DEBUG("[DeviceGraphExecutor] GPU graph launch-topology variant changed from "
                      << segment_cache.capture_variant_signature << " to "
                      << current_variant_signature << "; recapturing");
            PerfStatsCollector::addCounter(
                "forward_graph",
                "decode_graph_variant_recapture",
                1.0,
                "decode",
                ctx ? ctx->deviceId().toString() : std::string{},
                {{"old_signature", std::to_string(segment_cache.capture_variant_signature)},
                 {"new_signature", std::to_string(current_variant_signature)},
                 {"context", segment_cache.perf_context}});
            segment_cache.reset(GraphSegmentCache::StreamResetPolicy::Preserve);
            segment_cache.capture_variant_signature = current_variant_signature;
            segment_cache.variant_recapture_count++;
        }
        else if (!segment_cache.initialized)
        {
            segment_cache.capture_variant_signature = current_variant_signature;
        }

        // Monotonic step counter + phase transition selection for segmented mode.
        const auto phase_transition = DeviceGraphCaptureController::beginStep(
            segment_cache.initialized,
            segment_cache.needs_capture,
            segment_cache.decode_step);
        const uint64_t current_step = phase_transition.decode_step;
        PerfStatsCollector::addCounter(
            "forward_graph",
            "decode_graph_phase",
            1.0,
            "decode",
            ctx ? ctx->deviceId().toString() : std::string{},
            {{"context", segment_cache.perf_context},
             {"phase", DeviceGraphCaptureController::phaseName(phase_transition.phase)}});

        auto mark_arena_write_dirty = [&](BufferId id, DeviceId device)
        {
            if (!arena_)
                return;

            /*
             * Steady-state graph replay normally stays fully device-owned, so a
             * flags-only transition is enough and avoids an event per stage.
             * Parity snapshot callbacks are different: they immediately read
             * intermediate outputs on the host after replay.  Record the same
             * stream event used by the non-captured execution path so
             * ensureOnHost() waits for the captured kernels that produced the
             * tensor instead of racing a stale host copy.
             */
            if (config_.snapshot_callback && segment_cache.capture_stream)
                arena_->markWritten(id, device, segment_cache.capture_stream);
            else
                arena_->markWrittenFlagsOnly(id, device);
        };

        auto record_graph_snapshot_copies = [&](ComputeNode &node, void *producer_stream) -> bool
        {
            if (!config_.snapshot_callback || !node.stage)
                return true;

            DeviceId snapshot_device = node.device.is_valid() ? node.device : node.stage->device();
            if (!snapshot_device.is_valid() && ctx)
                snapshot_device = ctx->deviceId();

            void *stream = producer_stream ? producer_stream : node.stage->gpuStream();
            return captureGraphSnapshotCopies(node, snapshot_device, stream);
        };

        auto prepare_graph_snapshot_copies = [&](ComputeNode &node, void *producer_stream) -> bool
        {
            if (!config_.snapshot_callback || !node.stage)
                return true;

            DeviceId snapshot_device = node.device.is_valid() ? node.device : node.stage->device();
            if (!snapshot_device.is_valid() && ctx)
                snapshot_device = ctx->deviceId();

            void *stream = producer_stream ? producer_stream : node.stage->gpuStream();
            return prepareGraphSnapshotCopies(node, snapshot_device, stream);
        };

        // ===== FAST PATH: Phase 3 (Replay) =====
        // During steady-state replay, only the post_launch hook is invoked
        // (cohere_inputs is skipped for capturable segments, execute_node is
        // unused). Avoid constructing unused lambdas and skip phase 1/2 checks
        // to minimize host overhead on the hot decode path.
        const auto &exec_env = debugEnv().execution;
        const bool replay_diagnostics_enabled =
            exec_env.gpu_graph_verify || exec_env.gpu_graph_recapture || force_recapture;
        if (phase_transition.phase == DeviceGraphCaptureController::Phase::Replay &&
            !replay_diagnostics_enabled)
        {
            DeviceGraphCaptureController::prepareDeviceForGraphCapture(ctx);

            DeviceGraphCaptureController::ReplayHooks fast_hooks{
                nullptr, // cohere_inputs — skipped during normal replay (skip_coherence=true)
                [&](ComputeNode &node) -> bool
                {
                    return executeNode(node, ctx);
                },
                prepare_graph_snapshot_copies,
                record_graph_snapshot_copies,
                [&](DeviceGraphExecutor::GraphSegment &seg, void *stream)
                {
                    DeviceGraphCaptureController::postCapturedSegmentLaunch(
                        graph, seg, current_step, stream,
                        [&](BufferId id, DeviceId device)
                        {
                            mark_arena_write_dirty(id, device);
                        });
                },
                nullptr};

            const auto replay_result = DeviceGraphCaptureController::executeReplayPhase(
                graph, segment_cache, ctx, gpu_ctx,
                has_collective_nodes, collectives_graph_capturable, current_step, fast_hooks,
                /*force_recapture=*/false,
                defer_final_sync);

            if (!replay_result.success)
            {
                return false;
            }

            return true;
        }

        // ===== SLOW PATH: Phase 1 (Warmup) or Phase 2 (Capture) =====
        auto post_captured_segment_launch = [&](GraphSegment &seg, void *stream)
        {
            DeviceGraphCaptureController::postCapturedSegmentLaunch(
                graph,
                seg,
                current_step,
                stream,
                [&](BufferId id, DeviceId device)
                {
                    mark_arena_write_dirty(id, device);
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
            void *const coherence_stream = segment_cache.capture_stream;
            if (!coherence_stream)
            {
                LOG_ERROR("[DeviceGraphExecutor] Replay coherence for stage '"
                          << node.name
                          << "' requires the graph cache's exact non-null capture stream");
                return false;
            }

            std::string prepared_error;
            if (!node.stage->validatePreparedWeights(&prepared_error))
            {
                LOG_ERROR("[DeviceGraphExecutor] Prepared weight validation failed for replay stage '"
                          << node.name << "': " << prepared_error);
                return false;
            }
            if (!validatePreparedWeightBindings(node, contract, target_device))
                return false;

            // Cohere arena-managed reads (inputs + inouts)
            for (const auto &binding : contract.allArenaReads())
            {
                if (!arena_->prepareForRead(
                        binding.id, target_device, coherence_stream))
                {
                    LOG_ERROR("[DeviceGraphExecutor] Arena prepareForRead failed for replay stage: " << node.name);
                    return false;
                }
            }

            // Validate exact weight residency and join producer ordering.
            // Graph capture is never an allocation or lazy-upload boundary.
            if (!node.weights_cohered)
            {
                for (auto *weight : contract.weight_tensors)
                {
                    if (auto *tb = dynamic_cast<TensorBase *>(weight))
                    {
                        TransferEngine::requireDeviceInput(
                            tb, target_device, coherence_stream);
                    }
                }
                node.weights_cohered = true;
            }

            // Cohere arena-managed writes that require fresh storage.
            for (const auto &binding : contract.writesRequiringPrepare())
            {
                if (!arena_->prepareForWrite(
                        binding.id, target_device, coherence_stream))
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

        DeviceGraphCaptureController::prepareDeviceForGraphCapture(ctx);

        auto is_collective_node = [&](const ComputeNode &node) -> bool
        {
            if (collective_nodes && collective_nodes->find(node.name) != collective_nodes->end())
                return true;
            return node.stage && node.stage->isCollectiveStage();
        };

        DeviceGraphCaptureController::ReplayHooks replay_hooks{
            [&](const GraphSegment &segment)
            {
                return cohere_segment_inputs(segment);
            },
            [&](ComputeNode &node)
            {
                return executeNode(node, ctx);
            },
            prepare_graph_snapshot_copies,
            record_graph_snapshot_copies,
            [&](GraphSegment &segment, void *stream)
            {
                post_captured_segment_launch(segment, stream);
            },
            capture_boundary};

        // Capture-phase hooks: same as replay hooks except post_launch skips
        // onGraphReplayed() callbacks. During capture, execute() already ran
        // host-side bookkeeping; calling onGraphReplayed() would double-advance
        // KV cache head positions and corrupt subsequent decode steps.
        const StageRunPolicy capture_phase_policy = StageRunPolicy::capturePhase();
        DeviceGraphCaptureController::ReplayHooks capture_hooks{
            replay_hooks.cohere_inputs,
            [&](ComputeNode &node)
            {
                return runStage(node, ctx, capture_phase_policy, is_collective_node(node));
            },
            replay_hooks.prepare_snapshot_copies,
            record_graph_snapshot_copies,
            [&](GraphSegment &segment, void *stream)
            {
                DeviceGraphCaptureController::postCapturedSegmentLaunch(
                    graph,
                    segment,
                    current_step,
                    stream,
                    [&](BufferId id, DeviceId device)
                    {
                        mark_arena_write_dirty(id, device);
                    },
                    /*skip_replay_callbacks=*/true);
            },
            capture_boundary};

        if (phase_transition.phase == DeviceGraphCaptureController::Phase::Replay)
        {
            const auto replay_result = DeviceGraphCaptureController::executeReplayPhase(
                graph, segment_cache, ctx, gpu_ctx,
                has_collective_nodes, collectives_graph_capturable, current_step, replay_hooks, force_recapture,
                defer_final_sync);

            if (!replay_result.success)
            {
                return false;
            }

            return true;
        }

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
                collectives_graph_capturable,
                plan_policy);

            // Create the capture stream early so warmup runs on it.
            if (!segment_cache.ensureCaptureStream(
                    gpu_ctx,
                    ctx ? ctx->deviceId() : DeviceId::invalid(),
                    config_.worker_gpu_context_resolver
                        ? config_.worker_gpu_context_uses_process_pool
                        : true))
            {
                LOG_ERROR("[DeviceGraphExecutor] GPU graph warmup could not establish its explicit capture stream");
                return false;
            }
            void *warmup_stream = segment_cache.capture_stream;

            /*
             * Preserve stream order entirely on device. Previous decode work,
             * including KV/GDN writes, was queued on the worker stream. Warmup
             * consumes those tensors on the capture stream, so record one
             * worker-stream event and make the capture stream wait for it.
             * A device synchronize here both destroys overlap and lets one
             * LocalTP rank race ahead into collective warmup while another rank
             * is still changing streams.
             */
            if (gpu_stream && gpu_stream != warmup_stream)
            {
                if (!segment_cache.orderCaptureStreamAfter(
                        gpu_ctx,
                        gpu_stream))
                {
                    LOG_ERROR(
                        "[DeviceGraphExecutor] GPU graph warmup could not "
                        "publish its worker-to-capture stream handoff");
                    return false;
                }
                PerfStatsCollector::addCounter(
                    "forward_graph",
                    "warmup_stream_event_handoffs",
                    1.0,
                    "decode",
                    ctx ? ctx->deviceId().toString() : std::string{},
                    {{"context", segment_cache.perf_context}});
            }

            /*
             * LocalTP participants must enter warmup as one lifecycle. Actual
             * capture already invokes this hook before each beginCapture(); the
             * first warmup execution is equally collective-bearing and must not
             * be allowed to drift independently across ranks.
             */
            if (capture_boundary)
            {
                std::ostringstream boundary;
                boundary << "before_warmup:"
                         << "phase=warmup"
                         << ":step=" << current_step
                         << ":scope=full_graph";
                if (!segment_cache.perf_context.empty())
                    boundary << ":context=" << segment_cache.perf_context;
                if (!capture_boundary(boundary.str(), gpu_stream))
                {
                    LOG_ERROR(
                        "[DeviceGraphExecutor] GPU graph warmup boundary "
                        "rendezvous failed: "
                        << boundary.str());
                    return false;
                }
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

            // Warmup executes all stages normally (no capture) to ensure lazy
            // kernel initialization and workspace allocation complete on the
            // capture stream.  Preserve the capture-stream assignment above:
            // the generic fast-decode entry point intentionally rebinds eager
            // fallback passes to the worker stream to avoid stale stream
            // ownership of live device state.
            auto warmup_policy = StageRunPolicy::fastDecode();
            warmup_policy.preserve_gpu_streams = true;
            return runStages(graph, ctx, warmup_policy, collective_nodes);
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
                return false;
            }

            return true;
        }

        // Phase 3 is handled by the fast path above; this point is unreachable
        // after Phase 1 and Phase 2 both early-return.
        LOG_ERROR("[DeviceGraphExecutor] Unexpected phase in GPU graph capture/replay");
        return false;
    }

} // namespace llaminar2
