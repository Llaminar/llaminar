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

#include <algorithm>
#include <sstream>
#include <unordered_set>

namespace llaminar2
{
    std::optional<
        DeviceGraphExecutor::GraphSegmentCache::DeviceLoopGraphTemplateView>
    DeviceGraphExecutor::GraphSegmentCache::deviceLoopGraphTemplate(
        const ComputeGraph &graph,
        std::string *error) const
    {
        auto reject = [&](const std::string &reason)
            -> std::optional<DeviceLoopGraphTemplateView>
        {
            if (error)
                *error = reason;
            return std::nullopt;
        };
        if (error)
            error->clear();
        const auto &execution_order = graph.getExecutionOrder();
        if (execution_order.empty())
            return reject("source graph has no stages");
        if (!initialized || needs_capture)
            return reject("graph cache is not replay-ready");
        if (segments.size() != 1)
        {
            return reject(
                "graph is segmented: replay_units=" +
                std::to_string(segments.size()));
        }

        const GraphSegment &segment = segments.front();
        if (!segment.capturable || !segment.capture ||
            !segment.capture->hasExecutable() ||
            segment.capture->nodeCount() == 0)
        {
            return reject("monolithic replay unit has no executable capture");
        }
        if (!capture_stream ||
            segment.capture->executionStream() != capture_stream)
        {
            return reject(
                "captured replay unit has ambiguous producer-stream ownership");
        }
        if (segment.stage_names != execution_order)
        {
            return reject(
                "captured replay unit does not cover the complete source graph");
        }
        for (const auto &stage_name : execution_order)
        {
            const ComputeNode *node = graph.getNode(stage_name);
            if (!node || !node->stage)
            {
                return reject(
                    "source graph contains an unresolved stage: " + stage_name);
            }
            if (node->stage->graphLaunchPreparationPolicy() ==
                GraphLaunchPreparationPolicy::CaptureAndReplay)
            {
                return reject(
                    "stage requires external preparation before every replay: " +
                    stage_name);
            }
        }

        return DeviceLoopGraphTemplateView{
            .capture = segment.capture.get(),
            .stream = capture_stream,
            .stage_count = segment.stage_names.size(),
            .captured_node_count = segment.capture->nodeCount(),
        };
    }

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

    bool DeviceGraphExecutor::GraphSegmentCache::orderStreamAfterCapture(
        IWorkerGPUContext *ctx,
        void *consumer_stream)
    {
        if (!ctx || !consumer_stream || !capture_stream)
        {
            LOG_ERROR(
                "[GraphSegmentCache] Capture-stream publication requires one "
                "context and two explicit streams");
            return false;
        }
        if (consumer_stream == capture_stream)
            return true;
        if (!ensureSyncEvent(ctx))
            return false;
        if (!ctx->recordEventChecked(sync_event, capture_stream))
        {
            LOG_ERROR(
                "[GraphSegmentCache] Failed to record capture-stream output "
                "handoff event");
            return false;
        }
        if (!ctx->waitEventChecked(sync_event, consumer_stream))
        {
            LOG_ERROR(
                "[GraphSegmentCache] Failed to queue captured-output wait on "
                "the consumer stream");
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

    void DeviceGraphExecutor::GraphSegmentCache::destroyReplayGpuTimingEvents()
    {
        if (replay_gpu_timing_slots.empty())
        {
            replay_gpu_timing_device_name.clear();
            replay_gpu_timing_busy_samples = 0;
            return;
        }

        IWorkerGPUContext *ctx =
            requireLifecycleContext("replay GPU timing event destruction");
        try
        {
            for (auto &slot : replay_gpu_timing_slots)
            {
                if (slot.start_event)
                    ctx->destroyEvent(slot.start_event);
                if (slot.stop_event)
                    ctx->destroyEvent(slot.stop_event);
                slot = {};
            }
        }
        catch (const std::exception &e)
        {
            terminateGraphSegmentCacheLifecycle(
                std::string("replay GPU timing event destruction threw: ") +
                e.what());
        }
        catch (...)
        {
            terminateGraphSegmentCacheLifecycle(
                "replay GPU timing event destruction threw an unknown exception");
        }

        replay_gpu_timing_slots.clear();
        replay_gpu_timing_device_name.clear();
        replay_gpu_timing_busy_samples = 0;
    }

    // =========================================================================
    // Single-Graph Capture/Replay
    // =========================================================================

    std::unique_ptr<GraphCaptureDependencyLedger>
    DeviceGraphExecutor::planGraphCaptureDependencies(
        ComputeGraph &graph,
        std::span<const std::string> stage_names,
        DeviceId capture_device,
        void *capture_stream,
        const char *context)
    {
        if (!capture_device.is_gpu() || !capture_stream)
        {
            LOG_ERROR(
                "[DeviceGraphExecutor] Capture dependency planning requires an "
                "exact GPU and non-null stream");
            return nullptr;
        }

        const std::string capture_context =
            context ? context : "graph_capture_dependencies";
        std::vector<GraphCaptureDependencyLedger::StagePlan> stage_plans;
        stage_plans.reserve(stage_names.size());

        /*
         * BufferId, rather than a tensor's current coherence flag, defines the
         * topological producer.  A tensor may still be HOST_AUTHORITATIVE while
         * its producer kernel is merely being recorded.  The canonical storage
         * owner is retained only after that BufferId decision has been made.
         */
        std::unordered_map<BufferId, size_t> latest_producer;

        auto canonical_owner = [&](BufferId id,
                                   const std::string &stage_name,
                                   const char *access) -> const TensorBase *
        {
            if (!arena_)
            {
                LOG_ERROR("[DeviceGraphExecutor] Capture dependency stage '"
                          << stage_name << "' declares arena " << access
                          << " " << bufferIdName(id) << " without a BufferArena ("
                          << capture_context << ")");
                return nullptr;
            }
            ITensor *tensor = arena_->getTensor(id);
            auto *base = dynamic_cast<TensorBase *>(tensor);
            const TensorBase *owner = base ? base->transferStorageOwner() : nullptr;
            if (!owner)
            {
                LOG_ERROR("[DeviceGraphExecutor] Capture dependency stage '"
                          << stage_name << "' has no canonical tensor owner for "
                          << access << " " << bufferIdName(id) << " ("
                          << capture_context << ")");
            }
            return owner;
        };

        for (size_t stage_index = 0; stage_index < stage_names.size(); ++stage_index)
        {
            const std::string &stage_name = stage_names[stage_index];
            ComputeNode *node = graph.getNode(stage_name);
            if (!node || !node->stage)
            {
                LOG_ERROR("[DeviceGraphExecutor] Capture dependency plan is missing stage '"
                          << stage_name << "' (" << capture_context << ")");
                return nullptr;
            }

            DeviceId stage_device =
                node->device.is_valid() ? node->device : node->stage->device();
            if (!stage_device.is_valid())
                stage_device = capture_device;
            if (stage_device != capture_device)
            {
                LOG_ERROR("[DeviceGraphExecutor] Capture dependency stage '"
                          << stage_name << "' belongs to "
                          << stage_device.toString() << " instead of "
                          << capture_device.toString() << " ("
                          << capture_context << ")");
                return nullptr;
            }

            GraphCaptureDependencyLedger::StagePlan plan;
            plan.stage_identity = node->stage.get();
            plan.stage_name = stage_name;
            const StageBufferContract contract = node->stage->bufferContract();

            std::unordered_map<const TensorBase *, size_t> internal_owners;
            std::unordered_set<const TensorBase *> external_owners;
            for (const auto &binding : contract.allArenaReads())
            {
                const TensorBase *owner =
                    canonical_owner(binding.id, stage_name, "read");
                if (!owner)
                    return nullptr;

                const auto producer = latest_producer.find(binding.id);
                if (producer == latest_producer.end())
                {
                    if (internal_owners.count(owner) == 0)
                        external_owners.insert(owner);
                    continue;
                }

                external_owners.erase(owner);
                auto [it, inserted] =
                    internal_owners.emplace(owner, producer->second);
                if (!inserted)
                    it->second = std::max(it->second, producer->second);
            }

            plan.external_inputs.assign(
                external_owners.begin(), external_owners.end());
            plan.internal_inputs.reserve(internal_owners.size());
            for (const auto &[owner, producer_stage_index] : internal_owners)
            {
                plan.internal_inputs.push_back({
                    .tensor = owner,
                    .producer_stage_index = producer_stage_index,
                });
            }

            std::unordered_set<const TensorBase *> output_owners;
            for (const auto &binding : contract.allWrites())
            {
                const TensorBase *owner =
                    canonical_owner(binding.id, stage_name, "write");
                if (!owner)
                    return nullptr;
                if (output_owners.insert(owner).second)
                    plan.outputs.push_back(owner);
                latest_producer[binding.id] = stage_index;
            }

            stage_plans.push_back(std::move(plan));
        }

        try
        {
            return std::make_unique<GraphCaptureDependencyLedger>(
                capture_device,
                capture_stream,
                std::move(stage_plans),
                capture_context);
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("[DeviceGraphExecutor] Capture dependency planning failed ("
                      << capture_context << "): " << e.what());
            return nullptr;
        }
    }

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
            if (!node || !node->stage ||
                !requiresGraphLaunchPreparation(
                    node->stage->graphLaunchPreparationPolicy(),
                    GraphLaunchPreparationPhase::Capture))
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
        if (!prepareGraphStorageForCapture(
                graph,
                ctx,
                gpu_stream,
                "single_graph_capture"))
        {
            LOG_ERROR("[DeviceGraphExecutor] Single-graph capture storage preflight failed");
            return false;
        }

        auto dependency_ledger = planGraphCaptureDependencies(
            graph,
            std::span<const std::string>(order.data(), order.size()),
            ctx->deviceId(),
            gpu_stream,
            "single_graph_capture");
        if (!dependency_ledger)
        {
            LOG_ERROR(
                "[DeviceGraphExecutor] Single-graph capture dependency planning failed");
            return false;
        }

        // Step 1: Begin one structurally owned capture transaction.
        ScopedBackendGraphCapture capture_transaction(
            *capture,
            "single-graph capture",
            dependency_ledger.get());
        if (!capture_transaction.begin())
        {
            LOG_ERROR("[DeviceGraphExecutor] GPU graph beginCapture failed");
            return false;
        }

        // Step 2: Execute all stages into the captured stream
        // Set GPU device once before the loop (same as executeFastDecode)
        DeviceGraphCaptureController::prepareDeviceForGraphCapture(ctx);

        bool exec_success = true;

        const StageRunPolicy capture_policy = StageRunPolicy::capturePhase();
        for (const auto &name : order)
        {
            auto *node = graph.getNode(name);
            if (!node || !node->stage ||
                !runStage(
                    *node,
                    ctx,
                    capture_policy,
                    /*is_collective=*/false))
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
                policy.graph_replay_plan_policy,
                policy.launch_dependency);

            if (success)
            {
                if (used_graph_replay)
                {
                    *used_graph_replay = true;
                }
                return publishSnapshotsAfterGraphExecution(
                    graph,
                    segment_cache->capture_stream,
                    "decode_graph_replay",
                    &segment_cache->snapshot_manifest);
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
        if (!executeFastDecode(
                graph,
                ctx,
                collective_nodes))
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
                                                               GraphReplayPlanPolicy plan_policy,
                                                               GraphLaunchDependencyHook launch_dependency)
    {
        if (!gpu_stream || !gpu_ctx)
        {
            LOG_ERROR("[DeviceGraphExecutor] GPU graph capture/replay requires an explicit stream and live GPU context");
            return false;
        }

        const bool has_collective_nodes = (collective_nodes && !collective_nodes->empty());

        /*
         * Snapshot diagnostics change graph topology: every selected GPU stage
         * gains a D2D copy node into graph-owned mapped storage. A graph warmed
         * or captured under another callback/filter configuration cannot be
         * reused merely because its compute-stage launch signature is unchanged.
         *
         * Re-enter Phase 1 explicitly. This is the ordinary capture lifecycle,
         * not eager recovery after a failed graph launch: the old executable is
         * retired before any work is submitted under the new topology. One
         * initialization transaction warms the new manifest and materializes its
         * replacement executable before returning to the caller.
         */
        if (segment_cache.snapshot_configuration_epoch !=
            snapshot_configuration_epoch_)
        {
            segment_cache.reset(GraphSegmentCache::StreamResetPolicy::Preserve);
            segment_cache.snapshot_manifest.clear();
            segment_cache.snapshot_configuration_epoch =
                snapshot_configuration_epoch_;
            PerfStatsCollector::addCounter(
                "forward_graph",
                "decode_snapshot_configuration_rewarm",
                1.0,
                "decode",
                ctx ? ctx->deviceId().toString() : std::string{},
                {{"snapshot_epoch",
                  std::to_string(snapshot_configuration_epoch_)},
                 {"context", segment_cache.perf_context}});
        }

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
            return captureGraphSnapshotCopies(
                node,
                snapshot_device,
                stream,
                segment_cache.snapshot_manifest);
        };

        auto prepare_graph_snapshot_copies = [&](ComputeNode &node, void *producer_stream) -> bool
        {
            if (!config_.snapshot_callback || !node.stage)
                return true;

            DeviceId snapshot_device = node.device.is_valid() ? node.device : node.stage->device();
            if (!snapshot_device.is_valid() && ctx)
                snapshot_device = ctx->deviceId();

            void *stream = producer_stream ? producer_stream : node.stage->gpuStream();
            return prepareGraphSnapshotCopies(
                node,
                snapshot_device,
                stream,
                segment_cache.snapshot_manifest);
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
                .cohere_inputs = nullptr,
                .execute_node = [&](ComputeNode &node) -> bool
                {
                    return executeNode(node, ctx);
                },
                .prepare_snapshot_copies = prepare_graph_snapshot_copies,
                .record_snapshot_copies = record_graph_snapshot_copies,
                .post_launch = [&](DeviceGraphExecutor::GraphSegment &seg, void *stream)
                {
                    DeviceGraphCaptureController::postCapturedSegmentLaunch(
                        graph, seg, current_step, stream,
                        [&](BufferId id, DeviceId device)
                        {
                            mark_arena_write_dirty(id, device);
                        });
                },
                .capture_boundary = nullptr,
                .launch_dependency = launch_dependency,
            };

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

        // ===== FIRST-CAPTURE PATH =====
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

        auto cohere_replay_stage = [&](
                                       ComputeNode &node,
                                       const std::vector<BufferBinding> &external_reads,
                                       std::unordered_set<ITensor *> &prepared_inputs,
                                       std::unordered_set<ITensor *> &prepared_outputs) -> bool
        {
            DeviceId target_device = node.device.is_valid() ? node.device : node.stage->device();
            const StageBufferContract contract = node.stage->bufferContract();
            void *const coherence_stream = segment_cache.capture_stream;
            if (!prepareStageArenaFrontierForCapture(
                    node,
                    external_reads,
                    ctx->deviceId(),
                    coherence_stream,
                    prepared_inputs,
                    prepared_outputs,
                    "cached_graph_segment"))
            {
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

            return true;
        };

        auto cohere_segment_inputs = [&](const GraphSegment &seg) -> bool
        {
            std::unordered_set<ITensor *> prepared_inputs;
            std::unordered_set<ITensor *> prepared_outputs;
            return DeviceGraphCaptureController::cohereReplaySegmentInputs(
                graph,
                seg,
                [&](ComputeNode &node,
                    const std::vector<BufferBinding> &external_reads)
                {
                    return cohere_replay_stage(
                        node,
                        external_reads,
                        prepared_inputs,
                        prepared_outputs);
                });
        };

        auto plan_capture_dependencies = [&](const GraphSegment &seg)
            -> std::unique_ptr<GraphCaptureDependencyLedger>
        {
            return planGraphCaptureDependencies(
                graph,
                std::span<const std::string>(
                    seg.stage_names.data(), seg.stage_names.size()),
                ctx->deviceId(),
                segment_cache.capture_stream,
                "cached_graph_segment");
        };

        DeviceGraphCaptureController::prepareDeviceForGraphCapture(ctx);

        auto is_collective_node = [&](const ComputeNode &node) -> bool
        {
            if (collective_nodes && collective_nodes->find(node.name) != collective_nodes->end())
                return true;
            return node.stage && node.stage->isCollectiveStage();
        };

        const StageRunPolicy capture_phase_policy =
            StageRunPolicy::capturePhase();

        DeviceGraphCaptureController::ReplayHooks replay_hooks{
            .cohere_inputs = [&](const GraphSegment &segment)
            {
                return cohere_segment_inputs(segment);
            },
            .execute_node = [&](ComputeNode &node)
            {
                if (isGraphCaptureActive())
                {
                    return runStage(
                        node,
                        ctx,
                        capture_phase_policy,
                        is_collective_node(node),
                        &segment_cache.snapshot_manifest);
                }
                return executeNode(node, ctx);
            },
            .prepare_snapshot_copies = prepare_graph_snapshot_copies,
            .record_snapshot_copies = record_graph_snapshot_copies,
            .post_launch = [&](GraphSegment &segment, void *stream)
            {
                post_captured_segment_launch(segment, stream);
            },
            .capture_boundary = capture_boundary,
            .launch_dependency = launch_dependency,
            .plan_capture_dependencies = plan_capture_dependencies,
        };

        // Capture and replay share the same post-launch lifecycle. All mutable
        // GPU state is published by captured kernels and explicit event edges.
        DeviceGraphCaptureController::ReplayHooks capture_hooks{
            .cohere_inputs = replay_hooks.cohere_inputs,
            .execute_node = [&](ComputeNode &node)
            {
                return runStage(
                    node,
                    ctx,
                    capture_phase_policy,
                    is_collective_node(node),
                    &segment_cache.snapshot_manifest);
            },
            .prepare_snapshot_copies = replay_hooks.prepare_snapshot_copies,
            .record_snapshot_copies = record_graph_snapshot_copies,
            .post_launch = [&](GraphSegment &segment, void *stream)
            {
                DeviceGraphCaptureController::postCapturedSegmentLaunch(
                    graph,
                    segment,
                    current_step,
                    stream,
                    [&](BufferId id, DeviceId device)
                    {
                        mark_arena_write_dirty(id, device);
                    });
            },
            .capture_boundary = capture_boundary,
            .launch_dependency = launch_dependency,
            .plan_capture_dependencies = plan_capture_dependencies,
        };

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

        // ===== First use: prepare, capture, instantiate, and launch =====
        // Launch preparation binds immutable topology only. No stage arithmetic
        // runs outside the graph, so transaction zero is the first launch of the
        // captured executable and follows the exact same path as every replay.
        if (phase_transition.phase == DeviceGraphCaptureController::Phase::Capture)
        {
            DeviceGraphCaptureController::buildCapturePlan(
                graph,
                segment_cache,
                collective_nodes,
                has_collective_nodes,
                collectives_graph_capturable,
                plan_policy);

            // The graph and every preparation operation own this exact stream.
            if (!segment_cache.ensureCaptureStream(
                    gpu_ctx,
                    ctx ? ctx->deviceId() : DeviceId::invalid(),
                    config_.worker_gpu_context_resolver
                        ? config_.worker_gpu_context_uses_process_pool
                        : true))
            {
                LOG_ERROR("[DeviceGraphExecutor] GPU graph capture could not establish its explicit stream");
                return false;
            }
            void *capture_stream = segment_cache.capture_stream;

            if (!DeviceGraphCaptureController::prepareReplayGpuTiming(
                    segment_cache,
                    gpu_ctx,
                    ctx ? ctx->deviceId().toString() : std::string{}))
            {
                LOG_ERROR(
                    "[DeviceGraphExecutor] GPU graph capture could not establish "
                    "its fixed asynchronous replay timing ring");
                return false;
            }

            /*
             * Preserve stream order entirely on device. Previous decode work,
             * including KV/GDN writes, was queued on the worker stream. Capture
             * consumes those tensors on its own stream, so record one
             * worker-stream event and make the capture stream wait for it.
             * A device synchronization here would destroy overlap and obscure
             * the producer/consumer edge represented by this event.
             */
            if (gpu_stream && gpu_stream != capture_stream)
            {
                if (!segment_cache.orderCaptureStreamAfter(
                        gpu_ctx,
                        gpu_stream))
                {
                    LOG_ERROR(
                        "[DeviceGraphExecutor] GPU graph capture could not "
                        "publish its worker-to-capture stream handoff");
                    return false;
                }
                PerfStatsCollector::addCounter(
                    "forward_graph",
                    "capture_stream_event_handoffs",
                    1.0,
                    "decode",
                    ctx ? ctx->deviceId().toString() : std::string{},
                    {{"context", segment_cache.perf_context}});
            }

            graph.reset();
            const auto capture_result =
                DeviceGraphCaptureController::executeCapturePhase(
                    graph,
                    segment_cache,
                    ctx,
                    gpu_ctx,
                    has_collective_nodes,
                    current_step,
                    capture_hooks);
            if (capture_result.reset_cache)
            {
                segment_cache.reset(
                    GraphSegmentCache::StreamResetPolicy::Preserve);
            }
            if (!capture_result.success)
            {
                return false;
            }
            segment_cache.initialized = true;
            segment_cache.needs_capture = false;
            return true;
        }

        // Replay is handled by the fast path above; this point is unreachable.
        LOG_ERROR("[DeviceGraphExecutor] Unexpected phase in GPU graph capture/replay");
        return false;
    }

} // namespace llaminar2
