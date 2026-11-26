/**
 * @file DeviceGraphCaptureControllerPhases.cpp
 * @brief Capture phase finalization, retained parent transactions, and replay phase orchestration.
 */
#include "DeviceGraphCaptureControllerInternal.h"

namespace llaminar2
{
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
        std::string memory_contract_error;
        if (!graph_only_child && !segment.capture->validateExecutableMemoryClass(
                graph.executableMemoryClass(), &memory_contract_error))
        {
            LOG_ERROR("[DeviceGraphCaptureController] Native executable memory contract: "
                      << memory_contract_error << " context=" << perf_context);
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
        DeviceGraphExecutor::GraphInitialSubmissionPolicy initial_submission,
        std::unique_ptr<IGPUGraphCapture> parent)
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
        // Only this parent is executable. Its branch must span all child and
        // CPU-ticket waits, not stop in the gaps between graph-only children.
        if (hooks.attach_original_parent ? !hooks.attach_original_parent(*parent) :
            (hooks.auxiliary_branch && !hooks.auxiliary_branch->attach(*parent)))
        {
            LOG_ERROR("[DeviceGraphCaptureController] Retained parent auxiliary attachment failed");
            return false;
        }
        std::string memory_contract_error;
        if (!parent->validateExecutableMemoryClass(
                graph.executableMemoryClass(), &memory_contract_error))
        {
            LOG_ERROR("[DeviceGraphCaptureController] Retained parent memory contract: "
                      << memory_contract_error);
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
        std::vector<size_t> concurrent_service_segments;
        if (DeviceGraphExecutor::hasConcurrentTicketService(
                segment_cache.graph_replay_plan_policy))
        {
            for (size_t index = 0u; index < segment_cache.segments.size(); ++index)
            {
                if (!segment_cache.segments[index].capturable)
                    concurrent_service_segments.push_back(index);
            }
        }
        if (materialize_without_launch)
        {
            /*
             * The child contracts already describe every write performed by the
             * composed parent. Cache those manifests now without advancing arena
             * authority: no device node has run yet. Ownership moves into the
             * cache only after the executable is fully instantiated.
             */
            for (auto &segment : segment_cache.segments)
            {
                if (segment.capturable)
                    cacheCapturedSegmentArenaWrites(graph, segment);
            }
            segment_cache.retained_parent_capture = std::move(parent);
            // Setup is a physical lifecycle edge too: publish the same sealed
            // child/service inventory as transaction zero and later replay.
            // Missing service metadata must never look like implicit zero.
            auto boundary_tags = retainedParentBoundaryTags(
                segment_cache.perf_context, child_units->size(),
                concurrent_service_segments);
            boundary_tags.emplace("parent_nodes", std::to_string(
                segment_cache.retained_parent_capture->nodeCount()));
            PerfStatsCollector::addCounter(
                "forward_graph",
                "retained_parent_materialized_without_launch",
                1.0,
                "setup",
                ctx->deviceId().toString(),
                std::move(boundary_tags));
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
        const bool parent_submission_ok =
            concurrent_service_segments.empty()
                ? parent->launch()
                : hooks.submit_parent_with_concurrent_ticket_service &&
                      hooks.submit_parent_with_concurrent_ticket_service(
                          *parent,
                          concurrent_service_segments,
                          current_step);
        if (!parent_submission_ok)
        {
            LOG_ERROR(
                "[DeviceGraphCaptureController] Retained parent transaction-zero submission failed");
            return false;
        }

        /*
         * Child graphs were never submitted. Their buffer contracts still form
         * the complete parent publication manifest, so cache each segment's
         * writes only after the one parent launch has been accepted by the
         * backend. This is host-side authority bookkeeping, not child replay.
         */
        for (auto &segment : segment_cache.segments)
        {
            if (segment.capturable)
                hooks.post_launch(segment, segment_cache.capture_stream);
        }

        // First-use MTP may submit here instead of through the executor's
        // setup-only path. Publish the same physical boundary only after both
        // the parent launch and its concurrent CPU service have succeeded.
        auto boundary_tags = retainedParentBoundaryTags(
            segment_cache.perf_context, child_units->size(),
            concurrent_service_segments);
        boundary_tags.emplace("parent_nodes", std::to_string(parent->nodeCount()));
        PerfStatsCollector::addCounter(
            "forward_graph",
            "retained_parent_transaction_zero_launches",
            1.0,
            "decode",
            ctx->deviceId().toString(),
            std::move(boundary_tags));
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
        if(graph.readAheadBranch())throw std::logic_error("Read-ahead requires an unchanged complete captured parent; eager/manual/recapture verification is forbidden");
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

            if(node->stage->isParticipantLocalCapturedTransfer()){
                LOG_ERROR("[DeviceGraphCaptureController] Native captured transfer cannot execute as a manual service: "<<stage_name);
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

    bool DeviceGraphCaptureController::executeConcurrentTicketService(
            ComputeGraph &graph,
            DeviceGraphExecutor::GraphSegmentCache &segment_cache,
            IDeviceContext *ctx,
            IWorkerGPUContext *gpu_ctx,
            uint64_t current_step,
            std::span<const size_t> segment_indices,
            const std::function<bool(ComputeNode &)> &execute_node_cb)
    {
        if (!ctx || !gpu_ctx || !segment_cache.capture_stream ||
            !DeviceGraphExecutor::hasConcurrentTicketService(
                segment_cache.graph_replay_plan_policy))
        {
            LOG_ERROR(
                "[DeviceGraphCaptureController] Concurrent ticket service has incomplete context or the wrong replay policy");
            return false;
        }

        const auto fatal_abort_failure = [](const std::string &detail) noexcept
        {
            LOG_ERROR(
                "[DeviceGraphCaptureController] Fatal retained-parent ticket-service abort-publication failure: "
                << detail);
            std::terminate();
        };

        const auto publish_abort = [&]() noexcept
        {
            bool found_publisher = false;
            for (const size_t segment_index : segment_indices)
            {
                if (segment_index >= segment_cache.segments.size())
                {
                    fatal_abort_failure(
                        "sealed manual segment index is outside the graph plan");
                }
                const auto &segment = segment_cache.segments[segment_index];
                for (const std::string &stage_name : segment.stage_names)
                {
                    ComputeNode *const node = graph.getNode(stage_name);
                    if (!node || !node->stage)
                    {
                        fatal_abort_failure(
                            "cannot resolve abort publisher stage '" +
                            stage_name + "'");
                    }
                    if (node->stage->concurrentManualFailureRole() !=
                        ConcurrentManualFailureRole::DeviceIngressPublisher)
                    {
                        continue;
                    }
                    found_publisher = true;
                    if (!node->stage->publishConcurrentManualFailure())
                    {
                        fatal_abort_failure(
                            "device-ingress publisher rejected abort for stage '" +
                            stage_name + "'");
                    }
                }
            }
            if (!found_publisher)
            {
                fatal_abort_failure(
                    "failed transaction has no device-ingress publisher");
            }
        };

        for (const size_t segment_index : segment_indices)
        {
            if (segment_index >= segment_cache.segments.size())
            {
                fatal_abort_failure(
                    "sealed manual segment index is outside the graph plan");
            }
            auto &segment = segment_cache.segments[segment_index];
            if (segment.capturable)
            {
                fatal_abort_failure(
                    "sealed concurrent-service index names a captured unit");
            }

            bool service_ok = false;
            try
            {
                service_ok = executeManualReplaySegment(
                    graph,
                    segment,
                    ctx,
                    gpu_ctx,
                    segment_cache.capture_stream,
                    /*has_collective_nodes=*/false,
                    /*needs_segment_sync=*/false,
                    current_step,
                    execute_node_cb,
                    /*record_snapshot_copies_cb=*/{});
            }
            catch (const std::exception &error)
            {
                LOG_ERROR(
                    "[DeviceGraphCaptureController] Concurrent ticket service threw while executing segment "
                    << segment_index << ": " << error.what());
            }
            catch (...)
            {
                LOG_ERROR(
                    "[DeviceGraphCaptureController] Concurrent ticket service threw a non-standard exception while executing segment "
                    << segment_index);
            }
            if (!service_ok)
            {
                /*
                 * Release every captured device wait before returning failure.
                 * The parent-submission owner joins this worker and only then
                 * performs the exceptional stream fence, so backend submission
                 * and stream retirement can never race across host threads.
                 */
                publish_abort();
                return false;
            }
        }
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
                node->stage->setGPUStream(graph.executionStreamForNode(node->name,stream));
            if (!node->stage->prepareGraphLaunch(ctx, node->stage->gpuStream()))
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

        if (!gpu_ctx->ownsCurrentThread())
        {
            // The control-thread caller retains this whole cold transaction
            // until its original worker returns. Dispatching individual leaves
            // would split preparation from recording and lose the worker's
            // capture scope. TP participants are already fanned out by Rank;
            // each enters its own worker and the existing capture rendezvous.
            gpu_ctx->submitAndWait([&]
            {
                if (!gpu_ctx->ownsCurrentThread())
                    throw std::logic_error(
                        "Cold graph capture dispatch did not enter its original GPU worker");
                result = executeCapturePhase(
                    graph, segment_cache, ctx, gpu_ctx, has_collective_nodes,
                    current_step, hooks, initial_submission);
            });
            return result;
        }

        // This whole cold transaction is now on its original GPU worker.
        // Guarded scopes deliberately hide any exact admission of an outer graph.
        const ScopedCapturedVerifierExtent verifier_extent_scope(segment_cache.captured_verifier_extent);
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
            DeviceGraphExecutor::isRetainedParentPlanPolicy(
                segment_cache.graph_replay_plan_policy);
        const bool concurrent_ticket_service =
            DeviceGraphExecutor::hasConcurrentTicketService(
                segment_cache.graph_replay_plan_policy);
        const bool materialize_without_launch =
            initial_submission ==
            DeviceGraphExecutor::GraphInitialSubmissionPolicy::
                MaterializeWithoutLaunch;
        std::array<
            CaptureMaterializationAggregate,
            static_cast<std::size_t>(
                CaptureMaterializationPhase::Count)>
            materialization_timings{};
        std::map<std::string, CaptureStageRecordAggregate>
            stage_record_aggregates;
        const auto record_materialization_timing = [&] (
            CaptureMaterializationPhase phase,
            PerfStatsCollector::Clock::time_point begin,
            const DeviceGraphExecutor::GraphSegment *segment,
            size_t segment_index)
        {
            const auto measured_ns =
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    PerfStatsCollector::Clock::now() - begin)
                    .count();
            const auto elapsed_ns = static_cast<std::uint64_t>(
                std::max<std::int64_t>(0, measured_ns));
            const size_t stage_count =
                segment ? segment->stage_names.size() : 0u;
            const size_t node_count =
                segment && segment->capture
                    ? segment->capture->nodeCount()
                    : 0u;
            const char *const phase_name =
                captureMaterializationPhaseName(phase);
            materialization_timings[static_cast<std::size_t>(phase)]
                .observe(elapsed_ns);
            PerfStatsCollector::recordTimingNs(
                "forward_graph",
                std::string("capture_materialization_") + phase_name,
                elapsed_ns,
                materialize_without_launch ? "setup" : "capture",
                ctx->deviceId().toString(),
                {{"capture_mode", replayTopologyTag(segment_cache)},
                 {"context", segment_cache.perf_context},
                 {"initial_submission",
                  materialize_without_launch
                      ? "materialize_without_launch"
                      : "instantiate_and_launch"},
                 {"segment_index", std::to_string(segment_index)},
                 {"stages", std::to_string(stage_count)},
                 {"nodes", std::to_string(node_count)}});
        };

        if (retained_parent_composition !=
            static_cast<bool>(hooks.retained_parent_composer))
        {
            LOG_ERROR(
                "[DeviceGraphCaptureController] Retained parent policy and topology composer must be selected together");
            result.reset_cache = true;
            return result;
        }
        if (retained_parent_composition && !concurrent_ticket_service &&
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
            (hooks.auxiliary_branch->device() != ctx->deviceId() ||
             (!retained_parent_composition &&
              (!full_graph_capture || segment_cache.segments.size() != 1u ||
               !segment_cache.segments.front().capturable))))
        {
            LOG_ERROR(
                "[DeviceGraphCaptureController] An auxiliary branch requires one final native executable on the same device");
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
        const auto prebind_begin = PerfStatsCollector::Clock::now();
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
        record_materialization_timing(
            CaptureMaterializationPhase::ArenaPrebind,
            prebind_begin,
            /*segment=*/nullptr,
            /*segment_index=*/0u);

        /*
         * Launch preparation owns immutable topology setup and must complete for
         * every provisional capture unit before any native capture begins. This
         * replaces eager model execution as the source of kernel wrappers,
         * descriptor tables, pointer arrays, and transfer events. Only after
         * preparation may the stricter capture-readiness predicates be trusted.
         */
        const auto launch_preparation_begin =
            PerfStatsCollector::Clock::now();
        for (const auto &seg : segment_cache.segments)
        {
            if (!seg.capturable)
                continue;
            for (const auto &stage_name : seg.stage_names)
            {
                auto *node = graph.getNode(stage_name);
                if (node && node->stage)
                    node->stage->setGPUStream(graph.executionStreamForNode(node->name,capture_stream));
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
        record_materialization_timing(
            CaptureMaterializationPhase::LaunchPreparation,
            launch_preparation_begin,
            /*segment=*/nullptr,
            /*segment_index=*/0u);

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

        /*
         * Snapshot destinations form one graph-owned allocation, so descriptor
         * discovery is a graph-wide setup phase rather than per-segment work.
         * Every stage has now frozen its persistent launch metadata and output
         * pointers. Bind the complete arena once before the first beginCapture();
         * no later segment may allocate or extend snapshot storage.
         */
        const auto snapshot_manifest_begin =
            PerfStatsCollector::Clock::now();
        if (hooks.prepare_snapshot_manifest &&
            !hooks.prepare_snapshot_manifest())
        {
            LOG_ERROR(
                "[DeviceGraphCaptureController] Complete snapshot manifest preparation failed before native graph capture");
            result.reset_cache = true;
            result.success = false;
            return result;
        }
        record_materialization_timing(
            CaptureMaterializationPhase::SnapshotManifest,
            snapshot_manifest_begin,
            /*segment=*/nullptr,
            /*segment_index=*/0u);

        // The native owner must exist before recording: CUDA conditional
        // handles cannot be transplanted from independent captured graphs.
        // A local RAII owner also keeps failed construction out of the live cache.
        std::unique_ptr<IGPUGraphCapture> recording_parent;
        if (retained_parent_composition)
        {
            recording_parent = gpu_ctx->createGraphCapture(capture_stream);
            if (!recording_parent || recording_parent->nodeCount() != 0u ||
                recording_parent->hasExecutable() ||
                recording_parent->executionStream() != capture_stream)
            {
                result.reset_cache = true;
                return result;
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
                const auto input_coherence_begin =
                    PerfStatsCollector::Clock::now();
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
                record_materialization_timing(
                    CaptureMaterializationPhase::InputCoherence,
                    input_coherence_begin,
                    &seg,
                    segment_index);

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

                const auto native_record_begin =
                    PerfStatsCollector::Clock::now();

                // Source-definition ownership is a compile-time program policy,
                // not a retry after native composition fails. Shared-timeline
                // views preserve embedded conditional handles in their final
                // graph. A cloning parent instead needs independent definitions
                // that its lowerer can validate and import before instantiation.
                seg.capture = recording_parent &&
                    segment_cache.graph_replay_plan_policy !=
                        DeviceGraphExecutor::GraphReplayPlanPolicy::RequireCloneableParentComposition
                    ? recording_parent->createOrderedTimelineFragment()
                    : gpu_ctx->createGraphCapture(capture_stream);
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

                graph.sealReadAheadRecordingStream(seg.capture->executionStream());

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

                std::vector<IComputeStage *> capture_stages;
                capture_stages.reserve(seg.stage_names.size());
                for (const auto &stage_name : seg.stage_names)
                {
                    ComputeNode *node = graph.getNode(stage_name);
                    capture_stages.push_back(
                        node && node->stage ? node->stage.get() : nullptr);
                }

                bool exec_ok = true;
                std::string failed_stage_name;
                std::vector<CaptureStageRecordObservation>
                    stage_record_observations;
                stage_record_observations.reserve(seg.stage_names.size());
                std::uint64_t capture_begin_ns = 0u;
                std::uint64_t capture_close_ns = 0u;
                {
                    ScopedGraphCaptureStageActivity stage_capture_activity(
                        capture_stages,
                        ctx,
                        capture_stream,
                        "capture phase segment=" +
                            std::to_string(segment_index));
                    ScopedBackendGraphCapture capture_transaction(
                        *gpu_ctx,
                        *seg.capture,
                        "capture phase segment=" +
                            std::to_string(segment_index) +
                            " stages=" + describeSegmentStages(seg),
                        dependency_ledger.get());
                    if (!stage_capture_activity.begin())
                    {
                        LOG_ERROR(
                            "[DeviceGraphCaptureController] Stage capture "
                            "activity admission failed for segment");
                        result.reset_cache = true;
                        result.success = false;
                        return result;
                    }
                    const auto capture_begin_call =
                        PerfStatsCollector::Clock::now();
                    if (!capture_transaction.begin())
                    {
                        (void)stage_capture_activity.finish(
                            GraphCaptureActivityTransition::Aborted);
                        LOG_ERROR("[DeviceGraphCaptureController] beginCapture failed for segment");
                        result.reset_cache = true;
                        return result;
                    }
#if defined(HAVE_CUDA) && CUDART_VERSION >= 12030
                    const ScopedCUDAFinalTimelineRecording final_timeline_recording(*seg.capture);
#endif
                    capture_begin_ns = static_cast<std::uint64_t>(
                        std::max<std::int64_t>(
                            0,
                            std::chrono::duration_cast<std::chrono::nanoseconds>(
                                PerfStatsCollector::Clock::now() -
                                capture_begin_call)
                                .count()));

                    for (const auto &stage_name : seg.stage_names)
                    {
                        if (!exec_ok)
                            break;
                        auto *node = graph.getNode(stage_name);
                        CaptureStageRecordObservation observation{
                            .stage_name = stage_name,
                            .stage_type =
                                node && node->stage
                                    ? computeStageTypeName(node->stage->type())
                                    : "invalid",
                        };
                        const auto stage_record_begin =
                            PerfStatsCollector::Clock::now();
                        bool stage_recorded = false;
#ifdef HAVE_ROCM
                        int rocm_probe_idx = -1;
                        if (ctx->deviceId() == DeviceId::rocm(0) && segment_cache.perf_context == "main_decode") {
                            auto &probe = RocmDecodeTimelineProbe::instance();
                            probe.initIfNeeded();
                            if (probe.enabled && probe.buf) {
                                rocm_probe_idx = probe.slotFor(stage_name, observation.stage_type);
                                if (rocm_probe_idx >= 0) {
                                    rocmDebugRecordWallClock(probe.buf, static_cast<uint32_t>(2 * rocm_probe_idx), 0, capture_stream);
                                }
                            }
                        }
#endif
                        try
                        {
                            stage_recorded =
                                node && node->stage && hooks.execute_node(*node);
                        }
                        catch (const std::exception &exception)
                        {
                            // Preserve the exact throwing stage before RAII closes
                            // native capture and withdraws stage/worker activity.
                            LOG_ERROR("[DeviceGraphCaptureController] Stage threw during cached graph capture"
                                      << " stage=" << stage_name
                                      << " type=" << observation.stage_type
                                      << " participant_device=" << ctx->deviceId().toString()
                                      << " segment=" << segment_index
                                      << " wave=" << seg.capture_wave_ordinal
                                      << " wave_identity=" << seg.capture_wave_identity
                                      << " context=" << segment_cache.perf_context
                                      << " exception=" << exception.what());
                            throw;
                        }
                        catch (...)
                        {
                            LOG_ERROR("[DeviceGraphCaptureController] Stage threw during cached graph capture"
                                      << " stage=" << stage_name
                                      << " type=" << observation.stage_type
                                      << " participant_device=" << ctx->deviceId().toString()
                                      << " segment=" << segment_index
                                      << " wave=" << seg.capture_wave_ordinal
                                      << " wave_identity=" << seg.capture_wave_identity
                                      << " context=" << segment_cache.perf_context
                                      << " exception=<non-standard exception>");
                            throw;
                        }
                        if (stage_recorded)
                        {
#ifdef HAVE_ROCM
                            if (rocm_probe_idx >= 0) {
                                rocmDebugRecordWallClock(
                                    RocmDecodeTimelineProbe::instance().buf,
                                    static_cast<uint32_t>(2 * rocm_probe_idx + 1),
                                    0,
                                    capture_stream);
                            }
#endif
                        }
                        observation.execute_ns = static_cast<std::uint64_t>(
                            std::max<std::int64_t>(
                                0,
                                std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    PerfStatsCollector::Clock::now() -
                                    stage_record_begin)
                                    .count()));
                        if (!stage_recorded)
                        {
                            LOG_ERROR("[DeviceGraphCaptureController] Stage failed during cached graph capture: " << stage_name);
                            stage_record_aggregates[observation.stage_type]
                                .observe(observation);
                            stage_record_observations.push_back(
                                std::move(observation));
                            exec_ok = false;
                            failed_stage_name = stage_name;
                            break;
                        }
                        bool snapshots_recorded = true;
                        if (hooks.record_snapshot_copies)
                        {
                            const auto snapshot_record_begin =
                                PerfStatsCollector::Clock::now();
                            snapshots_recorded =
                                hooks.record_snapshot_copies(
                                    *node, capture_stream);
                            observation.snapshot_ns =
                                static_cast<std::uint64_t>(
                                    std::max<std::int64_t>(
                                        0,
                                        std::chrono::duration_cast<
                                            std::chrono::nanoseconds>(
                                            PerfStatsCollector::Clock::now() -
                                            snapshot_record_begin)
                                            .count()));
                        }
                        stage_record_aggregates[observation.stage_type]
                            .observe(observation);
                        stage_record_observations.push_back(
                            std::move(observation));
                        if (!snapshots_recorded)
                        {
                            LOG_ERROR("[DeviceGraphCaptureController] Snapshot copy failed during cached graph capture: "
                                      << stage_name);
                            exec_ok = false;
                            failed_stage_name = stage_name;
                            break;
                        }
                        graph.markCompleted(stage_name);
                    }

                    const auto capture_close_call =
                        PerfStatsCollector::Clock::now();
                    capture_transaction.finish();
                    if (!stage_capture_activity.finish(
                            exec_ok
                                ? GraphCaptureActivityTransition::Completed
                                : GraphCaptureActivityTransition::Aborted))
                    {
                        if (exec_ok)
                        {
                            failed_stage_name =
                                "<stage_capture_activity>";
                        }
                        exec_ok = false;
                    }
                    capture_close_ns = static_cast<std::uint64_t>(
                        std::max<std::int64_t>(
                            0,
                            std::chrono::duration_cast<std::chrono::nanoseconds>(
                                PerfStatsCollector::Clock::now() -
                                capture_close_call)
                                .count()));
                }
                const auto native_record_ns = static_cast<std::uint64_t>(
                    std::max<std::int64_t>(
                        0,
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            PerfStatsCollector::Clock::now() -
                            native_record_begin)
                            .count()));
                logSlowCaptureStageAttribution(
                    ctx->deviceId(),
                    segment_cache.perf_context,
                    stage_record_observations,
                    capture_begin_ns,
                    capture_close_ns,
                    native_record_ns);
                record_materialization_timing(
                    CaptureMaterializationPhase::NativeRecord,
                    native_record_begin,
                    &seg,
                    segment_index);

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
                // Direct capture and parent composition use the same single
                // attachment operation, always on the final execution owner.
                if (!retained_parent_composition &&
                    (hooks.attach_original_parent ? !hooks.attach_original_parent(*seg.capture) :
                     (hooks.auxiliary_branch && !hooks.auxiliary_branch->attach(*seg.capture))))
                {
                    LOG_ERROR("[DeviceGraphCaptureController] Native executable auxiliary attachment failed");
                    result.reset_cache = true;
                    return result;
                }
                const auto native_finalize_begin =
                    PerfStatsCollector::Clock::now();
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
                record_materialization_timing(
                    CaptureMaterializationPhase::NativeFinalize,
                    native_finalize_begin,
                    &seg,
                    segment_index);
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
                if (!materialize_without_launch &&
                    !retained_parent_composition)
                {
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
                initial_submission,
                std::move(recording_parent)))
        {
            result.reset_cache = true;
            return result;
        }

        const auto milliseconds = [](
            std::uint64_t nanoseconds) noexcept
        {
            return static_cast<double>(nanoseconds) / 1.0e6;
        };
        const auto &prebind = materialization_timings[
            static_cast<std::size_t>(
                CaptureMaterializationPhase::ArenaPrebind)];
        const auto &launch_preparation = materialization_timings[
            static_cast<std::size_t>(
                CaptureMaterializationPhase::LaunchPreparation)];
        const auto &snapshot_manifest = materialization_timings[
            static_cast<std::size_t>(
                CaptureMaterializationPhase::SnapshotManifest)];
        const auto &input_coherence = materialization_timings[
            static_cast<std::size_t>(
                CaptureMaterializationPhase::InputCoherence)];
        const auto &native_record = materialization_timings[
            static_cast<std::size_t>(
                CaptureMaterializationPhase::NativeRecord)];
        const auto &native_finalize = materialization_timings[
            static_cast<std::size_t>(
                CaptureMaterializationPhase::NativeFinalize)];
        const std::size_t capture_compilation_units =
            static_cast<std::size_t>(std::count_if(
                segment_cache.segments.begin(),
                segment_cache.segments.end(),
                [](const DeviceGraphExecutor::GraphSegment &segment)
                {
                    return segment.capturable;
                }));
        const std::size_t service_programs =
            segment_cache.segments.size() - capture_compilation_units;
        std::size_t captured_stage_objects = 0u;
        std::size_t service_stage_objects = 0u;
        std::size_t captured_native_nodes = 0u;
        for (const auto &segment : segment_cache.segments)
        {
            if (segment.capturable)
            {
                captured_stage_objects += segment.stage_names.size();
                if (segment.capture)
                    captured_native_nodes += segment.capture->nodeCount();
            }
            else
            {
                service_stage_objects += segment.stage_names.size();
            }
        }
        /*
         * A retained parent imports every graph-only child but exposes one
         * executable to inference. Keep that distinct from compilation-unit
         * count so setup diagnostics never imply serial runtime replay.
         */
        const std::size_t runtime_native_executables =
            retained_parent_composition ? 1u : capture_compilation_units;
        const std::size_t runtime_native_nodes =
            retained_parent_composition &&
                    segment_cache.retained_parent_capture
                ? segment_cache.retained_parent_capture->nodeCount()
                : captured_native_nodes;
        logCaptureStageTypeAggregate(
            ctx->deviceId(),
            segment_cache.perf_context,
            stage_record_aggregates);
        LOG_INFO(
            "[GraphCaptureMaterialization] complete"
            << " device=" << ctx->deviceId().toString()
            << " context=" << segment_cache.perf_context
            << " capture_mode="
            << replayTopologyTag(segment_cache)
            << " initial_submission="
            << (materialize_without_launch
                    ? "materialize_without_launch"
                    : "instantiate_and_launch")
            << " capture_compilation_units="
            << capture_compilation_units
            << " service_programs=" << service_programs
            << " captured_stage_objects=" << captured_stage_objects
            << " service_stage_objects=" << service_stage_objects
            << " captured_native_nodes=" << captured_native_nodes
            << " runtime_native_executables="
            << runtime_native_executables
            << " runtime_native_nodes=" << runtime_native_nodes
            << " prebind_ms=" << milliseconds(prebind.total_ns)
            << " launch_preparation_ms="
            << milliseconds(launch_preparation.total_ns)
            << " snapshot_manifest_ms="
            << milliseconds(snapshot_manifest.total_ns)
            << " input_coherence_ms="
            << milliseconds(input_coherence.total_ns)
            << " input_coherence_max_ms="
            << milliseconds(input_coherence.max_ns)
            << " input_coherence_samples=" << input_coherence.samples
            << " native_record_ms="
            << milliseconds(native_record.total_ns)
            << " native_record_max_ms="
            << milliseconds(native_record.max_ns)
            << " native_record_samples=" << native_record.samples
            << " native_finalize_ms="
            << milliseconds(native_finalize.total_ns)
            << " native_finalize_max_ms="
            << milliseconds(native_finalize.max_ns)
            << " native_finalize_samples=" << native_finalize.samples);

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

        const auto &diagnostic_config = debugEnv().execution;
        const bool executes_diagnostic_stages = diagnostic_config.gpu_graph_verify ||
            diagnostic_config.gpu_graph_recapture || diagnostic_config.gpu_graph_stream_only || force_recapture;
        if (executes_diagnostic_stages && !gpu_ctx->ownsCurrentThread())
        {
            // Diagnostic replay may execute stages and record new native nodes.
            // Preserve the original worker and immutable admission for that whole
            // transaction just as cold capture does; ordinary replay submits only.
            gpu_ctx->submitAndWait([&] {
                if (!gpu_ctx->ownsCurrentThread())
                    throw std::logic_error("Diagnostic graph recording did not enter its original GPU worker");
                result = executeReplayPhase(graph, segment_cache, ctx, gpu_ctx,
                    has_collective_nodes, collectives_graph_capturable, current_step,
                    hooks, force_recapture, defer_final_sync);
            });
            return result;
        }
        std::optional<ScopedCapturedVerifierExtent> diagnostic_extent;
        if (executes_diagnostic_stages)
            diagnostic_extent.emplace(segment_cache.captured_verifier_extent);

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

        const ReplayTerminalFencePlan terminal_fence_plan =
            resolveReplayTerminalFencePlan(
                graph,
                segment_cache,
                has_collective_nodes,
                collectives_graph_capturable,
                defer_final_sync);
        const bool can_defer_final_sync =
            terminal_fence_plan.defersHostObservation();
        const bool forward_graph_stats_enabled =
            PerfStatsCollector::isDomainEnabled("forward_graph");
        const auto &device_id = ctx->deviceId();
        const std::string device_name =
            forward_graph_stats_enabled
                ? device_id.toString()
                : std::string{};
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
                    manual_segment_total_ns;

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

            seg_idx++;
        }

        if (!finishReplayGpuTiming(
                segment_cache,
                gpu_ctx,
                total_replay_timing_slot))
        {
            return result;
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
