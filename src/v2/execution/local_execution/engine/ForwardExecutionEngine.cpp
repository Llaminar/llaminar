/**
 * @file ForwardExecutionEngine.cpp
 * @brief Prefill graph-cache state machine, cache-miss graph execution, and GPU timeline collection.
 */
#include "ForwardExecutionEngineInternal.h"

namespace llaminar2
{
    // =========================================================================
    // executePrefillWithGraphCache() — Prefill Graph Capture/Replay State Machine
    // =========================================================================

    bool ForwardExecutionEngine::executePrefillWithGraphCache(
        const ForwardInput &input,
        ForwardGraphCache &forward_cache,
        IDeviceContext *ctx,
        IForwardExecutionHost &host,
        bool *used_graph_replay,
        void **out_producer_stream,
        DeviceGraphExecutor::GraphInitialSubmissionPolicy
            initial_submission)
    {
        if (used_graph_replay)
            *used_graph_replay = false;
        if (out_producer_stream)
            *out_producer_stream = nullptr;
        const bool materialize_without_launch =
            initial_submission ==
            DeviceGraphExecutor::GraphInitialSubmissionPolicy::
                MaterializeWithoutLaunch;
        // Initialize prefill graph cache on first use
        if (!forward_cache.prefill_graph_cache)
        {
            forward_cache.prefill_graph_cache =
                std::make_unique<PrefillGraphCache>(
                    makePrefillGraphConfigFromEnv(host));
        }

        auto &cache = *forward_cache.prefill_graph_cache;
        const uint64_t snapshot_configuration_epoch =
            executor_.snapshotConfigurationEpoch();
        if (forward_cache.snapshot_configuration_epoch !=
            snapshot_configuration_epoch)
        {
            /*
             * Selected GPU snapshot stages become captured D2D nodes. Retire
             * prefill executables recorded under another callback/filter
             * topology before choosing the current key's phase. The stable
             * ComputeGraph and prepared weights remain reusable; the cache
             * simply performs its normal warmup/capture lifecycle again.
             */
            cache.invalidateAll();
            forward_cache.snapshot_manifest.clear();
            forward_cache.snapshot_configuration_epoch =
                snapshot_configuration_epoch;
            PerfStatsCollector::addCounter(
                "forward_graph",
                "prefill_snapshot_configuration_rewarm",
                1.0,
                "prefill",
                input.device.toString(),
                {{"snapshot_epoch",
                  std::to_string(snapshot_configuration_epoch)}});
        }

        PrefillGraphCacheKey key = makePrefillGraphKey(input, host);

        auto phase = cache.phase(key);
        const int real_seq_len = effectiveRealSeqLen(input);
        const int bucket_seq_len = effectiveBucketSeqLen(input);
        const bool padded_bucket = isPaddedBucketExecution(input);
        const bool snapshots_active = (executor_.config().snapshot_callback != nullptr);
        DeviceGraphExecutor::GraphSnapshotLogicalRows snapshot_logical_rows;
        if (snapshots_active)
        {
            if (input.batch_size <= 0 || bucket_seq_len <= 0 ||
                real_seq_len <= 0 || real_seq_len > bucket_seq_len)
            {
                LOG_ERROR("[ForwardExecutionEngine] Padded prefill snapshot publication has invalid row geometry: batch_size="
                          << input.batch_size
                          << " real_seq_len=" << real_seq_len
                          << " bucket_seq_len=" << bucket_seq_len);
                return false;
            }

            snapshot_logical_rows.physical_rows_per_sequence =
                static_cast<size_t>(bucket_seq_len);
            snapshot_logical_rows.logical_rows_per_sequence.reserve(
                static_cast<size_t>(input.batch_size));
            for (int batch = 0; batch < input.batch_size; ++batch)
            {
                const int logical_rows =
                    input.sequence_lengths &&
                            static_cast<size_t>(batch) <
                                input.sequence_lengths->size()
                        ? (*input.sequence_lengths)[static_cast<size_t>(batch)]
                        : real_seq_len;
                if (logical_rows <= 0 || logical_rows > bucket_seq_len)
                {
                    LOG_ERROR("[ForwardExecutionEngine] Padded prefill snapshot publication has an invalid logical row count for batch "
                              << batch << ": logical_rows=" << logical_rows
                              << " bucket_seq_len=" << bucket_seq_len);
                    return false;
                }
                snapshot_logical_rows.logical_rows_per_sequence.push_back(
                    static_cast<size_t>(logical_rows));
            }
        }
        const auto *snapshot_logical_rows_ptr =
            snapshot_logical_rows.active() ? &snapshot_logical_rows : nullptr;
        if (snapshots_active)
        {
            /*
             * This log is deliberately confined to snapshot-enabled runs.
             * Fixed-width graph execution is allowed to retain padded device
             * storage, whereas the diagnostic callback must publish only the
             * request-owned prefix.  Reporting both descriptors here makes a
             * malformed handoff distinguishable from a later publisher bug
             * without adding work to ordinary inference.
             */
            LOG_DEBUG(
                "[ForwardExecutionEngine] Prefill snapshot row contract"
                << " chunk=" << input.prefill_chunk_index
                << " real_seq_len=" << real_seq_len
                << " bucket_seq_len=" << bucket_seq_len
                << " padded=" << padded_bucket
                << " batch_size=" << input.batch_size
                << " projection_active="
                << (snapshot_logical_rows_ptr != nullptr)
                << " logical_rows="
                << (snapshot_logical_rows_ptr &&
                            !snapshot_logical_rows.logical_rows_per_sequence.empty()
                        ? std::to_string(
                              snapshot_logical_rows
                                  .logical_rows_per_sequence.front())
                        : std::string("none")));
        }
        const bool moe_rebalancing_active = host.isMoeRebalancingActive();
        const bool moe_rebalancing_graph_stable =
            host.isMoeRebalancingGraphStableForPrefillCapture();
        const bool host_prefill_graph_disabled = host.prefillGraphCaptureDisabledByHost();
        auto prefill_capture_policy =
            host.buildDecodeCapturePolicy(
                !forward_cache.collective_nodes.empty(),
                ctx);
        if (input.moe_overlay_graph_launch_dependency)
        {
            if (prefill_capture_policy.launch_dependency)
            {
                const auto existing =
                    std::move(prefill_capture_policy.launch_dependency);
                const auto overlay =
                    input.moe_overlay_graph_launch_dependency;
                prefill_capture_policy.launch_dependency =
                    [existing, overlay](
                        DeviceGraphExecutor::GraphExecutableLaunchPhase phase,
                        void *stream)
                {
                    return existing(phase, stream) &&
                           overlay(phase, stream);
                };
            }
            else
            {
                prefill_capture_policy.launch_dependency =
                    input.moe_overlay_graph_launch_dependency;
            }
            if (!prefill_capture_policy.allow_cached_graph_replay)
            {
                LOG_ERROR(
                    "[ForwardExecutionEngine] GPU ExpertOverlay prefill "
                    "requires retained graph execution so its remote ticket "
                    "can be armed at the exact executable launch edge");
                return false;
            }
        }
        const GraphNativeCaptureEnvelope native_capture_envelope =
            forward_cache.graph->nativeCaptureEnvelope();
        const bool uses_device_timeline_transaction =
            requiresSingleNativeExecutable(native_capture_envelope);
        bool uses_retained_sparse_parent = false;
        try
        {
            prefill_capture_policy.auxiliary_branch_factory =
                sourceDefinitionBoundaryFactory(host,input,ctx->deviceId(),false,forward_cache);
            std::string envelope_error;
            if (!DeviceGraphCaptureController::
                    constrainReplayPolicyToNativeEnvelope(
                        native_capture_envelope,
                        prefill_capture_policy,
                        &envelope_error))
            {
                LOG_ERROR(
                    "[ForwardExecutionEngine] Prefill rejected its graph-native capture envelope: "
                    << envelope_error);
                return false;
            }
            if (auto sparse_parent_plan =
                    makeMoEOverlayRetainedParentPlan(
                        *forward_cache.graph);
                sparse_parent_plan)
            {
                if (!prefill_capture_policy.allow_cached_graph_replay)
                {
                    LOG_ERROR(
                        "[ForwardExecutionEngine] Typed ExpertOverlay prefill packet graph requires mandatory retained-parent capture/replay");
                    return false;
                }
                prefill_capture_policy.graph_replay_plan_policy =
                    sparse_parent_plan->replay_policy;
                prefill_capture_policy.retained_parent_composer =
                    std::move(sparse_parent_plan->composer);
                /*
                 * The retained parent is a single asynchronous device
                 * transaction. Terminal-state and snapshot publication are
                 * enqueued on its exact producer stream below; no host fence is
                 * needed between the sparse parent and those consumers.
                 */
                prefill_capture_policy.defer_final_sync = true;
                uses_retained_sparse_parent = true;
            }
        }
        catch (const std::exception &ex)
        {
            LOG_ERROR(
                "[ForwardExecutionEngine] Could not derive retained ExpertOverlay prefill endpoint authority: "
                << ex.what());
            return false;
        }
        const bool prefill_collectives_graph_capturable =
            !forward_cache.collective_nodes.empty() &&
            prefill_capture_policy.collectives_graph_capturable;
        const bool prefill_heterogeneous_segmentation_admitted =
            prefillUsesCommonGraphCache(native_capture_envelope, prefill_capture_policy);
        bool padded_preflight_checked = false;
        PrefillGraphRejectReason padded_preflight_reason = PrefillGraphRejectReason::None;
        std::string padded_reject_stage_name;
        std::string padded_reject_stage_type;

        auto launchPrefillGraph = [&](IWorkerGPUContext *gpu_ctx,
                                      void *stream,
                                      const char *capture_phase,
                                      PrefillGraphPhase cache_phase,
                                      const char *launch_kind,
                                      const std::string &recapture_reason) -> bool
        {
            void *gpu_start_event = nullptr;
            void *gpu_stop_event = nullptr;
            const bool should_time_gpu =
                PerfStatsCollector::gpuStageEventTimingEnabled() &&
                gpu_ctx != nullptr &&
                stream != nullptr;
            bool gpu_event_timing_active = false;
            if (should_time_gpu)
            {
                gpu_start_event = gpu_ctx->createEvent(GPUEventPurpose::Timing);
                gpu_stop_event = gpu_ctx->createEvent(GPUEventPurpose::Timing);
                if (gpu_start_event && gpu_stop_event)
                {
                    gpu_ctx->recordEvent(gpu_start_event, stream);
                    gpu_event_timing_active = true;
                }
            }

            const auto launch_t0 = std::chrono::high_resolution_clock::now();
            const bool ok = cache.launch(key);
            const auto launch_t1 = std::chrono::high_resolution_clock::now();
            const auto ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(launch_t1 - launch_t0).count());

            auto destroy_gpu_events = [&]()
            {
                if (gpu_ctx)
                {
                    if (gpu_start_event)
                        gpu_ctx->destroyEvent(gpu_start_event);
                    if (gpu_stop_event)
                        gpu_ctx->destroyEvent(gpu_stop_event);
                }
                gpu_start_event = nullptr;
                gpu_stop_event = nullptr;
            };

            if (KernelProfiler::isEnabled())
            {
                ForwardPassProfiler::addReplayLaunchNs(ns);
            }

            if (PerfStatsCollector::isDomainEnabled("forward_graph"))
            {
                auto observation = makePrefillGraphObservation(
                    input,
                    key,
                    capture_phase,
                    recapture_reason);
                auto tags = prefillGraphObservationTags(
                    observation,
                    prefillGraphPhaseName(cache_phase));
                tags.emplace("launch_kind", launch_kind ? launch_kind : "unknown");
                PerfStatsCollector::recordTimingNsWithSequence(
                    "forward_graph",
                    "prefill_graph_launch",
                    ns,
                    {static_cast<uint64_t>(observation.chunk_index),
                 static_cast<uint64_t>(observation.real_token_start),
                 static_cast<uint64_t>(observation.real_token_end)},
                    "prefill",
                    input.device.toString(),
                    std::move(tags));
            }

            if (gpu_event_timing_active && ok)
            {
                gpu_ctx->recordEvent(gpu_stop_event, stream);
                gpu_ctx->synchronizeEvent(gpu_stop_event);
                const float elapsed_ms = gpu_ctx->eventElapsedTime(gpu_start_event, gpu_stop_event);
                if (elapsed_ms >= 0.0f)
                {
                    auto observation = makePrefillGraphObservation(
                        input,
                        key,
                        capture_phase,
                        recapture_reason);
                    auto tags = prefillGraphObservationTags(
                        observation,
                        prefillGraphPhaseName(cache_phase));
                    tags.emplace("attribution", "gpu_event");
                    tags.emplace("source", "prefill_graph_cache");
                    tags.emplace("graph_capture_scope", "prefill_graph_replay");
                    tags.emplace("timing_scope", "total_replay_gpu_event");
                    tags.emplace("sync_scope", "profiling_event_synchronized");
                    tags.emplace("launch_kind", launch_kind ? launch_kind : "unknown");
                    PerfStatsCollector::recordTimingNsWithSequence(
                        "stage_gpu",
                        "prefill_graph.replay",
                        static_cast<uint64_t>(static_cast<double>(elapsed_ms) * 1.0e6),
                    {static_cast<uint64_t>(observation.chunk_index),
                 static_cast<uint64_t>(observation.real_token_start),
                 static_cast<uint64_t>(observation.real_token_end)},
                        "prefill",
                        input.device.toString(),
                        std::move(tags));
                }
            }
            destroy_gpu_events();

            return ok;
        };

        if (padded_bucket)
        {
            padded_preflight_reason = preflightPrefillGraph(
                cache,
                *forward_cache.graph,
                key,
                forward_cache.collective_nodes,
                input,
                snapshots_active,
                moe_rebalancing_active,
                PrefillGraphPreflightMode::ColdPaddedSupport,
                prefill_collectives_graph_capturable,
                prefill_heterogeneous_segmentation_admitted,
                moe_rebalancing_graph_stable,
                host_prefill_graph_disabled,
                &padded_reject_stage_name,
                &padded_reject_stage_type);
            padded_preflight_checked = true;

            if (padded_preflight_reason != PrefillGraphRejectReason::None)
            {
                LOG_ERROR("[ForwardExecutionEngine] Padded prefill graph rejected by preflight: "
                          << toString(padded_preflight_reason)
                          << (padded_reject_stage_name.empty() ? "" : " stage=")
                          << padded_reject_stage_name
                          << (padded_reject_stage_type.empty() ? "" : " type=")
                          << padded_reject_stage_type
                          << " real_seq_len=" << real_seq_len
                          << " bucket_seq_len=" << bucket_seq_len);
                return false;
            }
        }

        auto gpuContextForPrefill = [&]() -> IWorkerGPUContext *
        {
            if (!ctx || !ctx->deviceId().is_gpu())
                return nullptr;
            return host.getWorkerGPUContext(ctx->deviceId());
        };

        auto bindPrefillStreamToStages = [&](void *stream)
        {
            const auto &order = forward_cache.graph->getExecutionOrder();
            for (const auto &node_name : order)
            {
                ComputeNode *node = forward_cache.graph->getNode(node_name);
                if (node && node->stage)
                    node->stage->setGPUStream(stream);
            }
        };

        auto preparePrefillGraphLaunchMetadata = [&](void *stream,
                                                     GraphLaunchPreparationPhase phase,
                                                     const char *phase_name) -> bool
        {
            const auto &order = forward_cache.graph->getExecutionOrder();
            for (const auto &node_name : order)
            {
                ComputeNode *node = forward_cache.graph->getNode(node_name);
                if (!node || !node->stage ||
                    !requiresGraphLaunchPreparation(
                        node->stage->graphLaunchPreparationPolicy(), phase))
                    continue;

                /*
                 * Monolithic prefill graphs have the same mutable-metadata
                 * contract as segmented decode graphs: tiny device-side
                 * scalars such as row-select indices must be uploaded on the
                 * explicit capture/replay stream before the graph is recorded
                 * or launched. The stage hook owns that upload; recording an
                 * H2D inside capture is a correctness bug because future graph
                 * replays would bake the first request's metadata.
                 */
                if (!node->stage->prepareGraphLaunch(ctx, stream))
                {
                    LOG_ERROR("[ForwardExecutionEngine] Prefill graph "
                              << (phase_name ? phase_name : "launch")
                              << " metadata preparation failed for stage '"
                              << node_name << "'");
                    return false;
                }
            }
            return true;
        };

        auto publishPrefillCapturedTerminalState = [&](void *stream,
                                                       const char *context) -> bool
        {
            const int terminal_row = real_seq_len - 1;
            if (!executor_.publishCapturedTerminalStateAfterGraphExecution(
                    *forward_cache.graph,
                    terminal_row,
                    stream,
                    context,
                    input.sequence_lengths_device,
                    input.batch_size,
                    input.seq_len,
                    input.sequence_lengths))
            {
                LOG_ERROR("[ForwardExecutionEngine] Prefill graph terminal state publication failed"
                          << (context ? std::string(" during ") + context : std::string{})
                          << " real_seq_len=" << real_seq_len
                          << " bucket_seq_len=" << bucket_seq_len);
                return false;
            }
            return true;
        };

        auto ensurePrefillCaptureStream = [&]() -> std::pair<IWorkerGPUContext *, void *>
        {
            IWorkerGPUContext *gpu_ctx = gpuContextForPrefill();
            if (!gpu_ctx)
                return {nullptr, nullptr};
            if (!forward_cache.prefill_capture_stream.ensure(gpu_ctx))
                return {gpu_ctx, nullptr};
            return {gpu_ctx, forward_cache.prefill_capture_stream.stream};
        };

        if (prefill_heterogeneous_segmentation_admitted)
        {
            /*
             * A heterogeneous prefill graph is never routed through the
             * monolithic PrefillGraphCache. Its explicit CPU/cross-device
             * boundaries are stable members of the replay topology, so the
             * generic segment cache captures every GPU region and executes each
             * declared manual boundary in the same order on transaction zero
             * and every replay. Only exact row geometry is admitted here.
             */
            std::string reject_stage_name;
            std::string reject_stage_type;
            const auto segmented_preflight = preflightPrefillGraph(
                cache,
                *forward_cache.graph,
                key,
                forward_cache.collective_nodes,
                input,
                snapshots_active,
                moe_rebalancing_active,
                PrefillGraphPreflightMode::Default,
                prefill_collectives_graph_capturable,
                /*heterogeneous_segmentation_admitted=*/true,
                moe_rebalancing_graph_stable,
                host_prefill_graph_disabled,
                &reject_stage_name,
                &reject_stage_type);
            if (segmented_preflight != PrefillGraphRejectReason::None)
            {
                LOG_ERROR(
                    "[ForwardExecutionEngine] Heterogeneous segmented prefill "
                    "rejected by typed preflight: "
                    << toString(segmented_preflight)
                    << (reject_stage_name.empty() ? "" : " stage=")
                    << reject_stage_name
                    << (reject_stage_type.empty() ? "" : " type=")
                    << reject_stage_type
                    << " real_seq_len=" << real_seq_len
                    << " bucket_seq_len=" << bucket_seq_len);
                return false;
            }

            IWorkerGPUContext *gpu_ctx = gpuContextForPrefill();
            if (!gpu_ctx ||
                !forward_cache.segment_cache.ensureCaptureStream(
                    gpu_ctx,
                    ctx->deviceId()) ||
                !forward_cache.segment_cache.ensureCaptureInputEvent(gpu_ctx) ||
                !forward_cache.segment_cache.ensureCaptureOutputEvent(gpu_ctx) ||
                !forward_cache.segment_cache.capture_stream)
            {
                LOG_ERROR(
                    "[ForwardExecutionEngine] Heterogeneous segmented prefill "
                    "could not establish its explicit capture stream");
                return false;
            }

            void *const stream = forward_cache.segment_cache.capture_stream;
            bindPrefillStreamToStages(stream);
            forward_cache.segment_cache.perf_context =
                input.bucket_seq_len > 0 ? "prefill_bucket" : "prefill";
            const bool was_initialized =
                forward_cache.segment_cache.initialized &&
                !forward_cache.segment_cache.needs_capture;

            if (materialize_without_launch && !was_initialized &&
                !host.prepareGraphBuildStateForMaterialization(
                    input,
                    stream,
                    ctx->deviceId()))
            {
                LOG_ERROR(
                    "[ForwardExecutionEngine] Setup segmented prefill "
                    "materialization could not consume graph-build device-state "
                    "readiness on "
                    << input.device.toString());
                return false;
            }

            auto capture_boundary =
                [&host, &input](
                    const std::string &boundary_name,
                    void *capture_stream) -> bool
            {
                return host.waitAtPrefillGraphCaptureBoundary(
                    input,
                    input.device,
                    boundary_name,
                    capture_stream);
            };

            const bool replay_ok = executor_.executeWithCachedGraphReplay(
                *forward_cache.graph,
                ctx,
                forward_cache.segment_cache,
                stream,
                gpu_ctx,
                &forward_cache.collective_nodes,
                prefill_capture_policy.collectives_graph_capturable,
                /*force_recapture=*/false,
                prefill_capture_policy.defer_final_sync,
                std::move(capture_boundary),
                prefill_capture_policy.graph_replay_plan_policy,
                prefill_capture_policy.launch_dependency,
                std::span<const BufferId>{},
                prefill_capture_policy.retained_parent_composer,
                initial_submission,
                prefill_capture_policy.auxiliary_branch_factory);
            if (!replay_ok)
            {
                LOG_ERROR(
                    "[ForwardExecutionEngine] Mandatory heterogeneous segmented "
                    "prefill capture/replay failed");
                return false;
            }

            if (materialize_without_launch)
            {
                PerfStatsCollector::addCounter(
                    "forward_graph",
                    "prefill_graph_materialized_without_launch",
                    1.0,
                    "setup",
                    input.device.toString(),
                    {{"bucket_seq_len", std::to_string(bucket_seq_len)},
                     {"replay_plan",
                      uses_device_timeline_transaction
                          ? "device_timeline_full_graph"
                          : uses_retained_sparse_parent
                          ? "retained_sparse_parent"
                          : "heterogeneous_segments"}});
                publishPrefillGraphObservation(
                    forward_cache,
                    input,
                    key,
                    PrefillGraphPhase::Ready,
                    "materialized_without_launch",
                    uses_retained_sparse_parent
                        ? "retained_sparse_parent"
                        : "heterogeneous_boundary_segmentation");
                return true;
            }

            if (!publishPrefillCapturedTerminalState(
                    stream,
                    was_initialized
                        ? "heterogeneous_prefill_segmented_replay"
                        : "heterogeneous_prefill_segmented_capture"))
            {
                return false;
            }
            if (!executor_.publishSnapshotsAfterGraphExecution(
                    *forward_cache.graph,
                    stream,
                    was_initialized
                        ? "heterogeneous_prefill_segmented_replay"
                        : "heterogeneous_prefill_segmented_capture",
                    &forward_cache.segment_cache.snapshot_manifest,
                    snapshot_logical_rows_ptr))
            {
                return false;
            }

            if (used_graph_replay)
                *used_graph_replay = true;
            if (out_producer_stream)
                *out_producer_stream = stream;
            PerfStatsCollector::addCounter(
                "forward_graph",
                "prefill_heterogeneous_segmented_graph",
                1.0,
                "prefill",
                input.device.toString(),
                {{"transaction", was_initialized ? "replay" : "capture"},
                 {"real_seq_len", std::to_string(real_seq_len)},
                     {"bucket_seq_len", std::to_string(bucket_seq_len)},
                     {"replay_plan",
                      uses_retained_sparse_parent
                          ? "retained_sparse_parent"
                          : "heterogeneous_segments"}});
            publishPrefillGraphObservation(
                forward_cache,
                input,
                key,
                PrefillGraphPhase::Ready,
                was_initialized ? "replay" : "capture",
                uses_retained_sparse_parent
                    ? "retained_sparse_parent"
                    : "heterogeneous_boundary_segmentation");
            return true;
        }

        /*
         * Setup owns a direct Cold -> Initialized transition. It has already
         * built the exact graph topology and permanent storage family, so an
         * eager synthetic request would add payload side effects without
         * establishing any additional capture invariant.
         */
        if (materialize_without_launch && phase == PrefillGraphPhase::Cold)
        {
            cache.markInitializedForSetupMaterialization(key);
            phase = PrefillGraphPhase::Initialized;
        }

        if (materialize_without_launch &&
            phase == PrefillGraphPhase::Ready && cache.hasGraph(key))
        {
            PerfStatsCollector::addCounter(
                "forward_graph",
                "setup_materialized_graph_reuses",
                1.0,
                "setup",
                input.device.toString(),
                {{"context", "monolithic_prefill"},
                 {"bucket_seq_len", std::to_string(bucket_seq_len)},
                 {"transaction_zero_pending",
                  cache.materializedTransactionZeroPending(key) ? "true"
                                                                : "false"}});
            publishPrefillGraphObservation(
                forward_cache,
                input,
                key,
                PrefillGraphPhase::Ready,
                "materialized_without_launch",
                "monolithic_prefill_reuse");
            return true;
        }

        if (phase == PrefillGraphPhase::Ready && !host_prefill_graph_disabled)
        {
            // === REPLAY PATH ===
            // Dynamic params already updated by executeCacheHit caller.
            auto [gpu_ctx, stream] = ensurePrefillCaptureStream();
            if (!gpu_ctx || !stream)
            {
                LOG_ERROR("[ForwardExecutionEngine] Prefill graph replay missing explicit capture stream");
                return false;
            }
            bindPrefillStreamToStages(stream);

            const bool transaction_zero_after_setup =
                cache.materializedTransactionZeroPending(key);
            if (transaction_zero_after_setup &&
                !executor_.prepareGraphStorageForCapture(
                    *forward_cache.graph,
                    ctx,
                    stream,
                    "prefill_graph_transaction_zero_frontier",
                    GraphCaptureDependencyLedger::ExternalInputAuthority::
                        RequireReadyBytes))
            {
                LOG_ERROR(
                    "[ForwardExecutionEngine] Setup-materialized prefill graph "
                    "could not validate its live transaction-zero input frontier");
                return false;
            }

            if (!preparePrefillGraphLaunchMetadata(
                    stream,
                    GraphLaunchPreparationPhase::Replay,
                    "replay"))
                return false;

            const char *const launch_kind =
                transaction_zero_after_setup
                    ? "transaction_zero_after_setup"
                    : "replay";
            const std::string replay_reason =
                transaction_zero_after_setup
                    ? "setup_materialized_transaction_zero"
                    : "none";
            if (!launchPrefillGraph(
                    gpu_ctx,
                    stream,
                    "replay",
                    PrefillGraphPhase::Ready,
                    launch_kind,
                    replay_reason))
            {
                LOG_ERROR("[ForwardExecutionEngine] Prefill graph replay FAILED for seq_len=" << input.seq_len);
                return false;
            }

            if (!publishPrefillCapturedTerminalState(
                    stream,
                    "prefill_graph_replay"))
            {
                return false;
            }

            if (!executor_.publishSnapshotsAfterGraphExecution(
                    *forward_cache.graph,
                    stream,
                    "prefill_graph_replay",
                    &forward_cache.snapshot_manifest,
                    snapshot_logical_rows_ptr))
            {
                return false;
            }

            if (cache.config().trace)
                LOG_TRACE("[ForwardExecutionEngine] Prefill graph REPLAY seq_len=" << input.seq_len
                                                                                    << " replay_count=" << cache.replayCount(key));
            publishPrefillGraphObservation(
                forward_cache,
                input,
                key,
                PrefillGraphPhase::Ready,
                "replay",
                replay_reason);
            if (used_graph_replay)
                *used_graph_replay = true;
            if (out_producer_stream)
                *out_producer_stream = stream;
            return true;
        }

        const bool can_attempt_capture =
            phase == PrefillGraphPhase::Warmup ||
            phase == PrefillGraphPhase::Initialized;
        PrefillGraphRejectReason capture_ready_reason = PrefillGraphRejectReason::None;
        std::string capture_ready_reject_stage_name;
        std::string capture_ready_reject_stage_type;
        if (can_attempt_capture)
        {
            capture_ready_reason = preflightPrefillGraph(
                cache,
                *forward_cache.graph,
                key,
                forward_cache.collective_nodes,
                input,
                snapshots_active,
                moe_rebalancing_active,
                materialize_without_launch
                    ? PrefillGraphPreflightMode::Default
                    : PrefillGraphPreflightMode::CaptureReady,
                prefill_collectives_graph_capturable,
                prefill_heterogeneous_segmentation_admitted,
                moe_rebalancing_graph_stable,
                host_prefill_graph_disabled,
                &capture_ready_reject_stage_name,
                &capture_ready_reject_stage_type);
        }

        if (can_attempt_capture && capture_ready_reason == PrefillGraphRejectReason::None)
        {
            // === CAPTURE PATH ===
            auto [gpu_ctx, stream] = ensurePrefillCaptureStream();
            if (!gpu_ctx || !stream)
            {
                LOG_ERROR("[ForwardExecutionEngine] Prefill graph capture missing explicit capture stream");
                return false;
            }

            /*
             * Apply the dedicated prefill stream to every stage before capture.
             * The warmup, metadata preparation, snapshot preparation, and
             * captured launch all use this same explicit stream. CUDA/HIP
             * stream order therefore supplies the complete producer-to-capture
             * dependency; a host stream drain here would only serialize the
             * Warmup -> Capture transition.
            */
            bindPrefillStreamToStages(stream);
            if (materialize_without_launch &&
                !host.prepareGraphBuildStateForMaterialization(
                    input,
                    stream,
                    ctx->deviceId()))
            {
                LOG_ERROR(
                    "[ForwardExecutionEngine] Setup monolithic prefill "
                    "materialization could not consume graph-build device-state "
                    "readiness on "
                    << input.device.toString());
                return false;
            }
            if (!executor_.allocateGraphStorageForCapture(
                    *forward_cache.graph,
                    ctx,
                    "prefill_graph_capture_prebind"))
            {
                LOG_ERROR("[ForwardExecutionEngine] Prefill graph storage prebind failed for seq_len="
                          << input.seq_len);
                return false;
            }
            if (!preparePrefillGraphLaunchMetadata(
                    stream,
                    GraphLaunchPreparationPhase::Capture,
                    "capture"))
                return false;

            if (materialize_without_launch)
            {
                capture_ready_reason = preflightPrefillGraph(
                    cache,
                    *forward_cache.graph,
                    key,
                    forward_cache.collective_nodes,
                    input,
                    snapshots_active,
                    moe_rebalancing_active,
                    PrefillGraphPreflightMode::CaptureReady,
                    prefill_collectives_graph_capturable,
                    prefill_heterogeneous_segmentation_admitted,
                    moe_rebalancing_graph_stable,
                    host_prefill_graph_disabled,
                    &capture_ready_reject_stage_name,
                    &capture_ready_reject_stage_type);
                if (capture_ready_reason != PrefillGraphRejectReason::None)
                {
                    LOG_ERROR(
                        "[ForwardExecutionEngine] Setup monolithic prefill "
                        "capture readiness rejected after launch preparation: "
                        << toString(capture_ready_reason)
                        << (capture_ready_reject_stage_name.empty()
                                ? ""
                                : " stage=")
                        << capture_ready_reject_stage_name
                        << (capture_ready_reject_stage_type.empty()
                                ? ""
                                : " type=")
                        << capture_ready_reject_stage_type);
                    return false;
                }
            }
            if (!executor_.prepareSnapshotsForGraphCapture(
                    *forward_cache.graph,
                    ctx,
                    stream,
                    "prefill_graph_capture",
                    &forward_cache.snapshot_manifest))
            {
                LOG_ERROR("[ForwardExecutionEngine] Prefill graph snapshot preparation failed for seq_len="
                          << input.seq_len);
                return false;
            }

            /*
             * Warmup may publish arena inputs from eager streams other than this
             * graph's dedicated stream. Join every such producer here, while
             * external event waits are legal, so the capture body contains only
             * graph-owned ordering. This is a nonblocking device-side fence; it
             * performs no stream or device synchronization.
             */
            if (!executor_.prepareGraphStorageForCapture(
                    *forward_cache.graph,
                    ctx,
                    stream,
                    "prefill_graph_capture",
                    materialize_without_launch
                        ? GraphCaptureDependencyLedger::ExternalInputAuthority::
                              BindDeclaredAddressesOnly
                        : GraphCaptureDependencyLedger::ExternalInputAuthority::
                              RequireReadyBytes))
            {
                LOG_ERROR("[ForwardExecutionEngine] Prefill graph storage preparation failed for seq_len="
                          << input.seq_len);
                return false;
            }

            const auto &prefill_capture_order =
                forward_cache.graph->getExecutionOrder();
            auto dependency_ledger = executor_.planGraphCaptureDependencies(
                *forward_cache.graph,
                std::span<const std::string>(
                    prefill_capture_order.data(), prefill_capture_order.size()),
                input.device,
                stream,
                "prefill_graph_capture",
                std::span<const BufferId>{},
                materialize_without_launch
                    ? GraphCaptureDependencyLedger::ExternalInputAuthority::
                          BindDeclaredAddressesOnly
                    : GraphCaptureDependencyLedger::ExternalInputAuthority::
                          RequireReadyBytes);
            if (!dependency_ledger)
            {
                LOG_ERROR(
                    "[ForwardExecutionEngine] Prefill graph dependency planning "
                    "failed for seq_len="
                    << input.seq_len);
                return false;
            }

            const std::string capture_begin_boundary =
                "before_begin:seq=" + std::to_string(input.seq_len) +
                ":real=" + std::to_string(real_seq_len) +
                ":bucket=" + std::to_string(bucket_seq_len) +
                ":offset=" + std::to_string(input.position_offset);
            if (!host.waitAtPrefillGraphCaptureBoundary(
                    input,
                    input.device,
                    capture_begin_boundary,
                    stream))
            {
                LOG_ERROR("[ForwardExecutionEngine] Prefill graph capture begin rendezvous failed for seq_len="
                          << input.seq_len << " on " << input.device.toString());
                return false;
            }

            /*
             * The cache owns begin/body/end as one scoped transaction. The
             * callback cannot strand the native stream in capture mode, and a
             * failed graph body cannot enter eager recovery.
             */
            // Match decode's stage-owned capture lifecycle. Completed is emitted
            // only after this exact prefill unit has successfully instantiated;
            // the RAII owner aborts every entered stage on failures/exceptions.
            std::vector<IComputeStage *> capture_stages;
            for (const auto &name : forward_cache.graph->getExecutionOrder())
            {
                const auto *node = forward_cache.graph->getNode(name);
                if (!node || !node->stage) return false;
                capture_stages.push_back(node->stage.get());
            }
            ScopedGraphCaptureStageActivity stage_capture_activity(
                capture_stages, ctx, stream, "monolithic prefill capture");
            if (!stage_capture_activity.begin()) return false;
            if (!cache.captureAndInstantiate(
                    key,
                    gpu_ctx,
                    stream,
                    [&]() -> bool
                    {
                        if (materialize_without_launch)
                        {
                            return executor_.recordSetupGraphMaterialization(
                                *forward_cache.graph,
                                ctx,
                                &forward_cache.collective_nodes,
                                &forward_cache.snapshot_manifest);
                        }
                        return executor_.executeFastDecode(
                            *forward_cache.graph,
                            ctx,
                            &forward_cache.collective_nodes,
                            &forward_cache.snapshot_manifest);
                    },
                    dependency_ledger.get()))
            {
                LOG_ERROR(
                    "[ForwardExecutionEngine] Mandatory prefill graph capture transaction failed for seq_len="
                    << input.seq_len);
                return false;
            }

            if (!stage_capture_activity.finish(GraphCaptureActivityTransition::Completed))
            {
                LOG_ERROR("[ForwardExecutionEngine] Prefill stage capture terminal publication failed");
                return false;
            }

            if (materialize_without_launch)
            {
                PerfStatsCollector::addCounter(
                    "forward_graph",
                    "prefill_graph_materialized_without_launch",
                    1.0,
                    "setup",
                    input.device.toString(),
                    {{"bucket_seq_len", std::to_string(bucket_seq_len)},
                     {"replay_plan", "monolithic_prefill"},
                     {"transaction_zero_pending", "true"}});
                publishPrefillGraphObservation(
                    forward_cache,
                    input,
                    key,
                    PrefillGraphPhase::Ready,
                    "materialized_without_launch",
                    "monolithic_prefill");
                return true;
            }

            const std::string capture_launch_boundary =
                "before_launch_after_capture:seq=" + std::to_string(input.seq_len) +
                ":real=" + std::to_string(real_seq_len) +
                ":bucket=" + std::to_string(bucket_seq_len) +
                ":offset=" + std::to_string(input.position_offset);
            if (!host.waitAtPrefillGraphCaptureBoundary(
                    input,
                    input.device,
                    capture_launch_boundary,
                    stream))
            {
                LOG_ERROR("[ForwardExecutionEngine] Prefill graph capture launch rendezvous failed for seq_len="
                          << input.seq_len << " on " << input.device.toString());
                return false;
            }

            // Kernels recorded during HIP/CUDA stream capture are not executed
            // until the executable graph is launched. Launch once immediately so
            // the capture request produces logits and advances device state.
            const std::string recapture_reason =
                phase == PrefillGraphPhase::Initialized
                    ? "lazy_initialized_after_request_reset"
                    : "armed_warmup";
            if (!launchPrefillGraph(gpu_ctx, stream, "capture", PrefillGraphPhase::Ready, "launch_after_capture",
                                    recapture_reason))
            {
                LOG_ERROR("[ForwardExecutionEngine] Prefill graph launch-after-capture failed for seq_len=" << input.seq_len);
                return false;
            }

            if (!publishPrefillCapturedTerminalState(
                    stream,
                    "prefill_graph_capture_launch"))
            {
                return false;
            }

            if (!executor_.publishSnapshotsAfterGraphExecution(
                    *forward_cache.graph,
                    stream,
                    "prefill_graph_capture_launch",
                    &forward_cache.snapshot_manifest,
                    snapshot_logical_rows_ptr))
            {
                return false;
            }

            LOG_INFO("[ForwardExecutionEngine] Prefill graph CAPTURED seq_len=" << input.seq_len
                                                                                << " nodes=" << cache.nodeCount(key));
            publishPrefillGraphObservation(
                forward_cache,
                input,
                key,
                PrefillGraphPhase::Ready,
                "capture",
                recapture_reason);
            if (out_producer_stream)
                *out_producer_stream = stream;
            return true;
        }

        if (can_attempt_capture && capture_ready_reason != PrefillGraphRejectReason::None)
        {
            LOG_ERROR("[ForwardExecutionEngine] Prefill graph capture readiness rejected after "
                      << prefillGraphPhaseName(phase)
                      << ": " << toString(capture_ready_reason)
                      << (capture_ready_reject_stage_name.empty() ? "" : " stage=")
                      << capture_ready_reject_stage_name
                      << (capture_ready_reject_stage_type.empty() ? "" : " type=")
                      << capture_ready_reject_stage_type
                      << " seq_len=" << input.seq_len
                      << "; refusing eager prefill fallback");
            publishPrefillGraphObservation(
                forward_cache,
                input,
                key,
                phase,
                "rejected",
                toString(capture_ready_reason),
                capture_ready_reject_stage_name,
                capture_ready_reject_stage_type);
            return false;
        }

        // === WARMUP/COLD PATH ===
        bool cold_capture_candidate = false;
        PrefillGraphRejectReason cold_reject_reason = PrefillGraphRejectReason::None;
        std::string cold_reject_stage_name;
        std::string cold_reject_stage_type;
        const bool cold_phase =
            phase == PrefillGraphPhase::Cold ||
            phase == PrefillGraphPhase::Initialized ||
            phase == PrefillGraphPhase::Warmup;
        if (cold_phase)
        {
            if (padded_preflight_checked)
            {
                cold_reject_reason = padded_preflight_reason;
                cold_reject_stage_name = padded_reject_stage_name;
                cold_reject_stage_type = padded_reject_stage_type;
            }
            else
            {
                cold_reject_reason = preflightPrefillGraph(
                    cache,
                    *forward_cache.graph,
                    key,
                    forward_cache.collective_nodes,
                    input,
                    snapshots_active,
                    moe_rebalancing_active,
                    padded_bucket
                        ? PrefillGraphPreflightMode::ColdPaddedSupport
                        : PrefillGraphPreflightMode::Default,
                    prefill_collectives_graph_capturable,
                    prefill_heterogeneous_segmentation_admitted,
                    moe_rebalancing_graph_stable,
                    host_prefill_graph_disabled,
                    &cold_reject_stage_name,
                    &cold_reject_stage_type);
            }
            cold_capture_candidate = (cold_reject_reason == PrefillGraphRejectReason::None);
        }

        bool cold_stream_ready = false;
        if (cold_capture_candidate)
        {
            auto [gpu_ctx, stream] = ensurePrefillCaptureStream();
            if (gpu_ctx && stream)
            {
                // Warm up lazy allocations on the same explicit stream that the
                // next request will capture on. This mirrors decode segmented
                // capture and avoids capture-unsafe first-use work.
                bindPrefillStreamToStages(stream);
                cold_stream_ready = true;
            }
            else
            {
                cold_reject_reason = PrefillGraphRejectReason::NoGPUContext;
                cold_capture_candidate = false;
            }
        }

        if (cold_phase && (!cold_capture_candidate || !cold_stream_ready))
        {
            LOG_ERROR("[ForwardExecutionEngine] Prefill graph capture rejected: "
                      << toString(cold_reject_reason)
                      << (cold_reject_stage_name.empty() ? "" : " stage=")
                      << cold_reject_stage_name
                      << (cold_reject_stage_type.empty() ? "" : " type=")
                      << cold_reject_stage_type
                      << " seq_len=" << input.seq_len
                      << "; refusing eager prefill fallback");
            publishPrefillGraphObservation(
                forward_cache,
                input,
                key,
                PrefillGraphPhase::Cold,
                "rejected",
                toString(cold_reject_reason),
                cold_reject_stage_name,
                cold_reject_stage_type);
            return false;
        }

        // Execute normally to warm up lazy allocations.
        bool exec_success = executor_.executeFastDecode(
            *forward_cache.graph,
            ctx,
            &forward_cache.collective_nodes,
            &forward_cache.snapshot_manifest);

        if (!exec_success)
            return false;

        if (!publishPrefillCapturedTerminalState(
                nullptr,
                "prefill_graph_warmup"))
        {
            return false;
        }

        if (!executor_.publishSnapshotsAfterGraphExecution(
                *forward_cache.graph,
                nullptr,
                "prefill_graph_warmup",
                &forward_cache.snapshot_manifest,
                snapshot_logical_rows_ptr))
        {
            return false;
        }

        if (out_producer_stream)
            *out_producer_stream =
                forward_cache.prefill_capture_stream.stream;

        // After successful warmup, check if graph capture is eligible
        if (cold_phase)
        {
            cache.markWarmedUp(key);
            publishPrefillGraphObservation(
                forward_cache,
                input,
                key,
                PrefillGraphPhase::Warmup,
                "warmup",
                "none");
            if (cache.config().trace)
                LOG_INFO("[ForwardExecutionEngine] Prefill graph ARMED for capture: seq_len=" << input.seq_len);
        }

        return true;
    }

    // =========================================================================
    // executeCacheMiss() — Build and Execute New Forward Graph
    // =========================================================================

    /**
     * @brief Materialize and submit one participant-local forward transaction.
     * @param input_in Logical request and declared device/stream ownership.
     * @param output Receives outputs with exact producer provenance on success.
     * @param signature Complete identity used for any retained graph entry.
     * @param build_cache Optional persistent storage for captured graph inputs.
     * @param should_cache Whether this transaction installs a retained entry.
     * @param host Participant's graph, resource and publication authority.
     * @param is_decode Selects decode-specific capture and completion policy.
     * @param start Start timestamp used only for passive timing evidence.
     * @return False on rejected setup or execution; no alternate path is tried.
     * @throws std::runtime_error if output ownership escapes this participant.
     *
     * Cache storage and stream provenance survive together. Pipeline stages
     * use their own participant's cache; cross-device composition is never an
     * uncached exception to this engine's lifecycle.
     */
    bool ForwardExecutionEngine::executeCacheMiss(
        const ForwardInput &input_in,
        ForwardOutput &output,
        const ForwardGraphSignature &signature,
        ForwardGraphCache *build_cache,
        bool should_cache,
        IForwardExecutionHost &host,
        bool is_decode,
        std::chrono::high_resolution_clock::time_point start)
    {
        // ===== CACHE MISS: Build new graph =====

        // For cache misses on standard path: redirect token_ids and
        // position_ids to stable buffers so that cached stages' pointers survive.
        ForwardInput effective_input = input_in;

        if (should_cache && build_cache)
        {
            // Retire every previous capture/auxiliary/source borrower BEFORE
            // replacing its graph or stable inputs. An invalid cache entry can
            // still own a failed cold attempt's native objects. Reapply only
            // immutable signature metadata after this original-owner teardown.
            build_cache->invalidate();
            bindGraphSnapshotArenaReusePolicy(*build_cache, signature);
            build_cache->segment_cache.replay_workload = replayWorkloadGeometryForSignature(signature);
            const int total_tokens = effective_input.batch_size * effective_input.seq_len;
            if (effective_input.token_ids)
            {
                build_cache->token_ids.assign(
                    effective_input.token_ids,
                    effective_input.token_ids + total_tokens);
                effective_input.token_ids = build_cache->token_ids.data();
            }
            if (effective_input.position_ids)
            {
                build_cache->position_ids.assign(
                    effective_input.position_ids,
                    effective_input.position_ids + total_tokens);
                effective_input.position_ids = build_cache->position_ids.data();
            }
        }

        /*
         * A non-embedding pipeline stage consumes an activation produced by the
         * preceding stage. Perform that handoff before graph construction so
         * model graph builders remain declarative and never materialize GPU
         * activations through data()/mutable_data(). TransferEngine selects D2D,
         * peer transport, or the explicitly designed heterogeneous host-staged
         * boundary and publishes destination authority as one operation.
         */
        consumeHostPipelineActivationBoundary(
            effective_input.graph_submission_intent == ForwardGraphSubmissionIntent::MaterializeExecutableWithoutLaunch
                ? HostPipelineBoundaryExecution::MaterializeOnly : HostPipelineBoundaryExecution::Execute,
            HostPipelineBoundarySubmission::Current,
            [&] { return host.resolvePPCopyInfo(effective_input); },
            [&](const auto &pp_copy) {
                auto result = TransferEngine::instance().copyActivation(
                    pp_copy.external_hidden, pp_copy.working_buffer,
                    pp_copy.device, pp_copy.copy_bytes);
                if (!result.success)
                    throw std::runtime_error("Initial PP hidden-state copy failed: " + result.error);
            });

        // Build forward graph via host callback
        GraphBuildResult build_result = host.buildForwardGraph(effective_input);

        if (!build_result)
        {
            LOG_ERROR("[ForwardExecutionEngine] Graph build failed: " << build_result.error());
            return false;
        }

        output = build_result.output();
        // Capture seals retain the ComputeGraph address, not just its stages.
        // The existing INVALID entry owns its graph at the final heap address
        // before workspace/capture callbacks. Only the original success boundary
        // publishes validity. Failure erasure retires segment/source borrowers
        // before graph destruction (ForwardGraphCache declaration order).
        // Uncached execution retains its original stack allocation/lifetime.
        std::optional<ComputeGraph> uncached_graph;
        if (should_cache && build_cache)
            build_cache->graph = std::make_unique<ComputeGraph>(build_result.takeGraph());
        else
            uncached_graph.emplace(build_result.takeGraph());
        ComputeGraph &graph = should_cache && build_cache ? *build_cache->graph : *uncached_graph;

        // Native pipeline transport is an explicit graph operation. The
        // surrounding chunk/verifier prelude still owns local positions/real lengths;
        // graph capture never copies a previous stage's TensorBase or changes
        // its current-device authority.
        if (config_.pipeline_forward_edges && config_.pipeline_forward_edges->encloses(effective_input))
            config_.pipeline_forward_edges->append(graph, effective_input);

        if (effective_input.device_decode_position)
        {
            const auto model_roots = graph.getRootNodes();
            if (model_roots.empty())
            {
                LOG_ERROR("[ForwardExecutionEngine] Cannot bind a decode position producer to an empty model graph");
                return false;
            }
            constexpr const char *node = "decode_position_snapshot";
            graph.addNode(node, std::make_unique<DecodePositionSnapshotStage>(
                DecodePositionSnapshotStage::Params{
                    .device_id = effective_input.device,
                    .binding = *effective_input.device_decode_position}), effective_input.device);
            // The cache count may be changed by attention inside any model
            // branch. Freeze it before every root, not merely before RoPE.
            for (const auto &root : model_roots)
                graph.addDependency(root, node);
        }

        if (effective_input.device_prefill_chunk)
        {
            const DevicePrefillChunkGraphBinding &binding =
                *effective_input.device_prefill_chunk;
            if (!devicePrefillChunkOwnsInput(effective_input))
            {
                LOG_ERROR("[ForwardExecutionEngine] Device prefill chunk binding does not exactly own the forward graph inputs");
                return false;
            }

            const std::vector<std::string> model_roots = graph.getRootNodes();
            if (model_roots.empty())
            {
                LOG_ERROR("[ForwardExecutionEngine] Cannot prepend a device chunk materializer to an empty model graph");
                return false;
            }

            constexpr const char *kChunkMaterializationNode =
                "prefill_chunk_materialization";
            graph.addNode(
                kChunkMaterializationNode,
                ComputeStageFactory::createPrefillChunkMaterialization({
                    .device_id = effective_input.device,
                    .backend = binding.backend,
                    .request_token_ids_device =
                        binding.request_token_ids_device,
                    .request_position_ids_device =
                        binding.request_position_ids_device,
                    .request_total_rows_device =
                        binding.request_total_rows_device,
                    .cached_tokens_device = binding.cached_tokens_device,
                    .chunk_token_ids_device = binding.chunk_token_ids_device,
                    .chunk_position_ids_device =
                        binding.chunk_position_ids_device,
                    .chunk_real_rows_device = binding.chunk_real_rows_device,
                    .chunk_row_stride_device =
                        binding.chunk_row_stride_device,
                    .request_row_capacity = binding.request_row_capacity,
                    .bucket_seq_len = binding.bucket_seq_len,
                    .pad_token_id = binding.pad_token_id,
                    .require_exact_rows = binding.require_exact_rows,
                    .capture_identity = binding.capture_identity,
                    .stage_name = kChunkMaterializationNode,
                }),
                effective_input.device);
            for (const std::string &root : model_roots)
                graph.addDependency(root, kChunkMaterializationNode);
        }
        auto collective_nodes = graph.collectiveNodeNames();

        if (signature.live_mtp_request_batch_condition &&
            debugEnv().runtime_debug.mtp_condition_graph_contract_trace)
        {
            std::ostringstream manifest;
            bool first = true;
            for (const auto &node_name : graph.getExecutionOrder())
            {
                if (!collective_nodes.contains(node_name))
                    continue;
                if (!first)
                    manifest << ',';
                first = false;
                manifest << node_name;
            }
            LOG_INFO(
                "[MTPConditionGraphContract] event=graph_manifest"
                << " device=" << effective_input.device.toString()
                << " stages=" << graph.size()
                << " collective_count=" << collective_nodes.size()
                << " collectives=[" << manifest.str() << ']');
        }

        LOG_DEBUG("[ForwardExecutionEngine] Forward graph built with " << graph.size() << " stages");

        if (graph.size() == 0)
        {
            LOG_ERROR("[ForwardExecutionEngine] Empty forward graph");
            return false;
        }

        const bool bucketed_prefill_miss =
            should_cache && build_cache && signature.is_bucketed_prefill && !is_decode;
        bool bucketed_prefill_capture_candidate = false;
        bool bucketed_prefill_heterogeneous_segmented_candidate = false;
        PrefillGraphRejectReason bucketed_prefill_reject_reason = PrefillGraphRejectReason::None;
        std::string bucketed_prefill_reject_stage_name;
        std::string bucketed_prefill_reject_stage_type;

        if (bucketed_prefill_miss)
        {
            PrefillGraphCache preflight_cache(
                makePrefillGraphConfigFromEnv(host));
            PrefillGraphCacheKey key = makePrefillGraphKey(effective_input, host);
            /*
             * Reject graph-intrinsic violations before asking the host for a
             * launch context.  The optimistic policy flags make this first
             * pass ignore only the two capabilities that genuinely depend on
             * that context; recurrent padded stages, invalid geometry, and
             * host/rebalancing policy remain fatal here.  Besides preserving
             * fail-fast behavior, this keeps rejected graphs from touching a
             * backend whose stream/context lifecycle they can never use.
             */
            bucketed_prefill_reject_reason = preflightPrefillGraph(
                preflight_cache,
                graph,
                key,
                collective_nodes,
                effective_input,
                executor_.config().snapshot_callback != nullptr,
                host.isMoeRebalancingActive(),
                PrefillGraphPreflightMode::Default,
                /*collectives_graph_capturable=*/true,
                /*heterogeneous_segmentation_admitted=*/true,
                host.isMoeRebalancingGraphStableForPrefillCapture(),
                host.prefillGraphCaptureDisabledByHost(),
                &bucketed_prefill_reject_stage_name,
                &bucketed_prefill_reject_stage_type);

            if (bucketed_prefill_reject_reason ==
                PrefillGraphRejectReason::None)
            {
                IDeviceContext *prefill_preflight_ctx =
                    host.getDeviceContext(effective_input.device);
                const auto prefill_capture_policy =
                    host.buildDecodeCapturePolicy(
                        !collective_nodes.empty(),
                        prefill_preflight_ctx);
                const bool prefill_collectives_graph_capturable =
                    !collective_nodes.empty() &&
                    prefill_capture_policy.collectives_graph_capturable;
                const bool prefill_heterogeneous_segmentation_admitted =
                    prefill_capture_policy.heterogeneous_segmented_enabled;

                bucketed_prefill_reject_reason = preflightPrefillGraph(
                    preflight_cache,
                    graph,
                    key,
                    collective_nodes,
                    effective_input,
                    executor_.config().snapshot_callback != nullptr,
                    host.isMoeRebalancingActive(),
                    PrefillGraphPreflightMode::Default,
                    prefill_collectives_graph_capturable,
                    prefill_heterogeneous_segmentation_admitted,
                    host.isMoeRebalancingGraphStableForPrefillCapture(),
                    host.prefillGraphCaptureDisabledByHost(),
                    &bucketed_prefill_reject_stage_name,
                    &bucketed_prefill_reject_stage_type);
                bucketed_prefill_heterogeneous_segmented_candidate =
                    bucketed_prefill_reject_reason ==
                        PrefillGraphRejectReason::None &&
                    prefill_heterogeneous_segmentation_admitted;
            }
            bucketed_prefill_capture_candidate =
                (bucketed_prefill_reject_reason == PrefillGraphRejectReason::None);

            if (!bucketed_prefill_capture_candidate)
            {
                if (build_cache && !build_cache->prefill_graph_cache)
                {
                    build_cache->prefill_graph_cache =
                        std::make_unique<PrefillGraphCache>(
                            makePrefillGraphConfigFromEnv(host));
                }
                if (build_cache && build_cache->prefill_graph_cache)
                {
                    publishPrefillGraphObservation(
                        *build_cache,
                        effective_input,
                        key,
                        PrefillGraphPhase::Cold,
                        "rejected",
                        toString(bucketed_prefill_reject_reason),
                        bucketed_prefill_reject_stage_name,
                        bucketed_prefill_reject_stage_type);
                }

                std::string rejected_stage_readiness;
                if (!bucketed_prefill_reject_stage_name.empty())
                {
                    const auto *rejected_node =
                        graph.getNode(bucketed_prefill_reject_stage_name);
                    if (rejected_node && rejected_node->stage)
                    {
                        rejected_stage_readiness =
                            rejected_node->stage
                                ->graphCaptureReadinessDebugString();
                    }
                }

                LOG_ERROR("[ForwardExecutionEngine] Bucketed prefill graph rejected by preflight before execution: "
                          << toString(bucketed_prefill_reject_reason)
                          << (bucketed_prefill_reject_stage_name.empty() ? "" : " stage=")
                          << bucketed_prefill_reject_stage_name
                          << (bucketed_prefill_reject_stage_type.empty() ? "" : " type=")
                          << bucketed_prefill_reject_stage_type
                          << " real_seq_len=" << effectiveRealSeqLen(effective_input)
                          << " bucket_seq_len=" << effectiveBucketSeqLen(effective_input)
                          << (rejected_stage_readiness.empty()
                                  ? ""
                                  : " readiness={")
                          << rejected_stage_readiness
                          << (rejected_stage_readiness.empty() ? "" : "}")
                          << "; refusing eager prefill fallback");
                return false;
            }
        }

        // Cache hits already refresh this small stage list before replay.
        // Cache misses must do the same before the first graph execution; otherwise
        // padded bucket stages can append/select the bucket tail during warmup.
        std::vector<IComputeStage *> prefill_replay_param_stages;
        if (!is_decode)
        {
            prefill_replay_param_stages = collectPrefillReplayParamStages(graph);
            updatePrefillReplayParamStages(effective_input, prefill_replay_param_stages);
        }

        // Sparse manual boundaries must receive the same root-published
        // identity before the very first graph execution and all later cache
        // replays. A local stage-object execution count is not stable across
        // independent rank graph construction or capture lifecycle changes.
        std::vector<IComputeStage *> moe_overlay_collective_runtime_stages =
            collectMoEOverlayCollectiveRuntimeStages(graph);
        updateMoEOverlayCollectiveRuntimeStages(
            effective_input,
            moe_overlay_collective_runtime_stages);

        // Ensure declared CPU/GPU workspace is allocated for this graph. Cache
        // hits repeat this step because another bucket can grow the shared
        // per-device workspace and leave cached stages bound to old pointers.
        const WorkspaceGraphParticipantRole workspace_role =
            workspaceParticipantRole(
                effective_input.execution_role,
                is_decode);
        if (!host.ensureDeviceWorkspaceAllocated(
                graph,
                effective_input.seq_len,
                workspaceFamilyPolicy(workspace_role),
                workspace_role))
        {
            LOG_ERROR("[ForwardExecutionEngine] Failed to allocate workspace for forward graph on "
                      << effective_input.device.toString() << " seq_len=" << effective_input.seq_len);
            return false;
        }
        const uint64_t workspace_generation = host.workspaceGeneration(effective_input.device);

        // Notify host that graph is ready — allows releasing transient resources
        // (e.g., mmap pages) before execution allocates large activation buffers.
        if (!first_graph_ready_fired_)
        {
            first_graph_ready_fired_ = true;
            host.onFirstGraphReady();
        }

        /*
         * A cacheable GPU decode graph and an exact heterogeneous segmented
         * prefill graph each have one lifecycle, including their first
         * invocation. Install the fully built graph into its stable cache
         * before submitting any kernels, then enter executeCacheHit(), whose
         * capture transaction executes the logical payload once and returns
         * with complete native graph executable units.
         *
         * Keeping a separate eager cache-miss execution here used to expose a
         * valid ComputeGraph with no replay-ready GraphSegmentCache. The MTP
         * parent graph could observe that half-state immediately after the
         * grouped verifier produced its rows. Delegating first use through the
         * ordinary cached path makes that state structurally impossible and
         * also guarantees that cache misses and hits share identical stream,
         * dynamic-parameter, collective, snapshot, and publication ordering.
         */
        const bool atomic_gpu_graph_first_use =
            is_decode ||
            effective_input.graph_submission_intent ==
                ForwardGraphSubmissionIntent::
                    MaterializeExecutableWithoutLaunch ||
            bucketed_prefill_heterogeneous_segmented_candidate;
        if (should_cache && build_cache && atomic_gpu_graph_first_use &&
            effective_input.device.is_gpu())
        {
            // The original graph already occupies its final cache-owned address.
            build_cache->output = output;
            build_cache->workspace_generation = workspace_generation;
            build_cache->snapshot_configuration_epoch =
                executor_.snapshotConfigurationEpoch();
            build_cache->collective_nodes = std::move(collective_nodes);
            build_cache->prefill_replay_param_stages =
                std::move(prefill_replay_param_stages);
            build_cache->prefill_replay_param_stages_cached = !is_decode;
            build_cache->moe_overlay_collective_runtime_stages =
                std::move(moe_overlay_collective_runtime_stages);
            build_cache->moe_overlay_collective_runtime_stages_cached = true;
            build_cache->valid = true;

            // The current boundary was already consumed above, including the
            // setup-only no-copy decision. First-use routing must not repeat it.
            const bool first_use_success = executeCacheHit(
                effective_input,
                output,
                *build_cache,
                host,
                signature,
                start, HostPipelineBoundarySubmission::AlreadySubmitted);
            if (!first_use_success)
            {
                build_cache->invalidate();
                return false;
            }

            LOG_DEBUG(
                "[ForwardExecutionEngine] Atomically materialized first-use "
                "GPU graph [kind="
                << (is_decode
                        ? "decode"
                        : effective_input.graph_submission_intent ==
                                  ForwardGraphSubmissionIntent::
                                      MaterializeExecutableWithoutLaunch
                              ? "setup_prefill"
                              : "heterogeneous_prefill")
                << ", seq_len="
                << signature.seq_len
                << ", batch_size=" << signature.batch_size
                << ", device=" << signature.device.to_string()
                << ", all_position_logits="
                << signature.all_position_logits << "]");
            return true;
        }

        if (effective_input.graph_submission_intent ==
            ForwardGraphSubmissionIntent::
                MaterializeExecutableWithoutLaunch)
        {
            LOG_ERROR(
                "[ForwardExecutionEngine] Setup-owned GPU graph family was not "
                "admitted to the stable forward cache; refusing to execute its "
                "model payload eagerly");
            return false;
        }

        bool success = false;

        /*
         * Retain the concrete producer stream independently of graph caching.
         * Short GPU prefills can be intentionally ineligible for caching, but
         * their hidden-state consumers need the same exact ordering contract as
         * Ready graph replays.
         */
        void *execution_stream_used = nullptr;

        /* A CPU decode cache miss executes its first step eagerly here rather
         * than through executeCacheHit(). It owns the same per-step fork cell,
         * so the first decode step is not silently missing from the summary
         * that cache-hit steps publish. GPU first use returned above. */
        const llaminar::v2::OpenMPTeamForkScope openmp_forks_at_exec_entry;

        // The graph and every producer stream belong to this participant.
        {
            LOG_DEBUG("[ForwardExecutionEngine] Getting device context for " << effective_input.device << "...");
            IDeviceContext *ctx = host.getDeviceContext(effective_input.device);
            if (!ctx)
            {
                LOG_ERROR("[ForwardExecutionEngine] Failed to get device context");
                return false;
            }
            LOG_DEBUG("[ForwardExecutionEngine] Got device context, starting execution...");

            void *execution_stream = nullptr;
            IWorkerGPUContext *execution_gpu_ctx = nullptr;
            if (ctx->deviceId().is_gpu())
            {
                try
                {
                    execution_gpu_ctx = host.getWorkerGPUContext(ctx->deviceId());
                    if (!execution_gpu_ctx)
                    {
                        LOG_ERROR("[ForwardExecutionEngine] Host did not provide a worker GPU context for "
                                  << ctx->deviceId().toString());
                        return false;
                    }
                    execution_stream = execution_gpu_ctx->defaultStream();
                }
                catch (const std::exception &e)
                {
                    LOG_ERROR("[ForwardExecutionEngine] Could not resolve forward graph stream for "
                              << ctx->deviceId().toString() << ": " << e.what());
                    return false;
                }

                if (bucketed_prefill_capture_candidate)
                {
                    /*
                     * The first bucketed-prefill cache miss is the warmup pass
                     * for the later monolithic graph capture.  Run it on the
                     * same explicit stream that the Warmup->Capture transition
                     * will record on; otherwise lazy first-use work may be
                     * ordered on one stream while capture later observes another.
                     */
                    if (!build_cache->prefill_capture_stream.ensure(execution_gpu_ctx) ||
                        !build_cache->prefill_capture_stream.stream)
                    {
                        LOG_ERROR("[ForwardExecutionEngine] Bucketed prefill cache miss could not create an explicit capture stream for "
                                  << ctx->deviceId().toString());
                        return false;
                    }
                    execution_stream = build_cache->prefill_capture_stream.stream;
                }
                execution_stream_used = execution_stream;
            }

            /*
             * Cache-hit replay runs a stream/dynamic-parameter prelude before
             * launching the cached graph. Cache misses need the same contract
             * for their first execution: graph capture may start inside
             * DeviceGraphExecutor, so stages such as EmbeddingStage must upload
             * token IDs to their workspace before executor-side capture begins.
             *
             * The stream is Llaminar's owned worker stream, never the legacy
             * CUDA/HIP null stream. Stages that do not override
             * updateDynamicParams() simply ignore this setup.
             */
            std::vector<IComputeStage *> cache_miss_dynamic_param_stages;
            if (ctx->deviceId().is_gpu())
            {
                if (!execution_stream)
                {
                    LOG_ERROR("[ForwardExecutionEngine] GPU cache-miss graph execution requires an explicit stream for "
                              << ctx->deviceId().toString());
                    return false;
                }

                cache_miss_dynamic_param_stages =
                    collectDynamicParamStages(graph);
                for (const auto &node_name : graph.getExecutionOrder())
                {
                    ComputeNode *node = graph.getNode(node_name);
                    if (!node || !node->stage)
                        continue;
                    node->stage->setGPUStream(execution_stream);
                }

                if (!is_decode)
                {
                    /*
                     * Padded prefill replay params were pushed once before
                     * workspace allocation so host-only stages can see the
                     * metadata early.  GPU GDN/short-conv stages also upload a
                     * workspace-backed effective-length scalar, so repeat the
                     * update after the workspace has been bound and after every
                     * stage owns this explicit stream.  Otherwise a warmup run
                     * can leave the scalar at the previous real length and the
                     * following capture mutates live GDN state through one or
                     * more padding rows.
                     */
                    updatePrefillReplayParamStages(
                        effective_input,
                        prefill_replay_param_stages);
                }
            }

            if (!host.prepareLiveStateForForwardGraphExecution(
                    effective_input,
                    execution_stream,
                    ctx->deviceId()))
            {
                LOG_ERROR("[ForwardExecutionEngine] Failed to prepare live state for forward graph execution");
                return false;
            }
            ScopedForwardLiveStateRead live_state_read(
                host,
                effective_input,
                execution_stream,
                ctx->deviceId());

            if (!host.prepareDeviceTokenInputsForForwardGraphExecution(
                    effective_input,
                    execution_stream,
                    ctx->deviceId()))
            {
                LOG_ERROR("[ForwardExecutionEngine] Failed to prepare forward graph device-token inputs");
                return false;
            }

            if (host.computeAllPositionLogitsEnabled() &&
                host.allPositionLogitRows() > 0)
            {
                if (!host.prepareAllPositionVerifierGraphMetadata(
                        effective_input,
                        execution_stream,
                        ctx->deviceId()))
                {
                    LOG_ERROR("[ForwardExecutionEngine] Failed to prepare all-position verifier graph metadata");
                    return false;
                }
            }

            if (!updateDynamicParamStages(
                    effective_input,
                    cache_miss_dynamic_param_stages,
                    effective_input.position_ids,
                    "cache-miss graph execution"))
            {
                return false;
            }

            if (!host.waitBeforeForwardGraphExecution(
                    effective_input,
                    ctx->deviceId(),
                    /*cache_miss=*/true))
            {
                LOG_ERROR("[ForwardExecutionEngine] Forward graph pre-execution rendezvous failed for "
                          << ctx->deviceId().toString());
                return false;
            }

            success =
                build_cache
                    ? executor_.executeWithSnapshotManifest(
                          graph,
                          ctx,
                          build_cache->snapshot_manifest)
                    : executor_.execute(graph, ctx);

            /*
             * Match the pre-execution live-state admission on every outcome.
             * Even a failed executor may have queued kernels, so skipping this
             * event edge would let the next mailbox writer race partial work.
             */
            live_state_read.complete(execution_stream);
        }

        DeviceId producer_device = effective_input.device;
        if (success)
        {
            producer_device = resolveForwardOutputProducerDevice(output, effective_input.device);
            if (producer_device != effective_input.device)
                throw std::runtime_error("Participant-local forward published an output on another device");

            publishForwardOutputProvenance(
                output,
                producer_device,
                execution_stream_used,
                effective_input.execution_role,
                is_decode,
                host.computeAllPositionLogitsEnabled(),
                forwardCompletionScopeForInput(effective_input),
                effective_input.seq_len,
                effective_input.batch_size);

            if (is_decode && producer_device.is_gpu())
            {
                const bool all_position_verifier =
                    host.computeAllPositionLogitsEnabled();
                const bool defer_main_decode =
                    !all_position_verifier &&
                    host.shouldDeferMainDecodeFinalSync();

                if ((all_position_verifier || defer_main_decode) &&
                    !execution_stream_used)
                {
                    LOG_ERROR("[ForwardExecutionEngine] GPU cache-miss decode requires an explicit producer stream for its device-state handoff on "
                              << producer_device.toString());
                    return false;
                }

                if (all_position_verifier)
                {
                    host.setPendingAllPositionVerifierStream(
                        execution_stream_used);
                }
                if (defer_main_decode)
                {
                    host.setPendingMainDecodeStream(execution_stream_used);
                }
            }
        }

        /*
         * A result's public tensor event and a sampler's private stream handoff
         * serve different consumers. Publish both on cache misses just as on
         * replay. Publication records an event; it does not materialize logits
         * on the host or wait for device completion.
         */
        if (success)
        {
            if (!is_decode && effectiveRealSeqLen(effective_input) > 0)
            {
                const int terminal_row = effectiveRealSeqLen(effective_input) - 1;
                if (!executor_.publishCapturedTerminalStateAfterGraphExecution(
                        graph,
                        terminal_row,
                        nullptr,
                        should_cache
                            ? "prefill_cache_miss"
                            : "prefill_eager",
                        effective_input.sequence_lengths_device,
                        effective_input.batch_size,
                        effective_input.seq_len,
                        effective_input.sequence_lengths))
                {
                    LOG_ERROR("[ForwardExecutionEngine] Failed to publish captured terminal prefill state after cache miss");
                    return false;
                }
            }

            IDeviceContext *publication_ctx =
                host.getDeviceContext(producer_device);
            if (!publication_ctx || !host.publishForwardResultAtBoundary(
                    output, publication_ctx))
            {
                LOG_ERROR("[ForwardExecutionEngine] Failed to publish cache-miss forward result at the device boundary");
                return false;
            }

            const std::string stage_context =
                should_cache
                    ? forwardGraphPerfContext(signature)
                    : (is_decode ? "main_decode" : "prefill");
            collectTimeline(
                host,
                host.getDeviceContext(effective_input.device),
                is_decode, effective_input, start, stage_context);

            // Same publisher, key and gate as the cache-hit step; a CPU cache
            // miss never replays a captured graph.
            if (is_decode && !effective_input.device.is_gpu())
            {
                llaminar::v2::publishOpenMPTeamForkSummary(
                    openmp_forks_at_exec_entry,
                    effective_input.device,
                    [this, &signature] { return forwardGraphPerfContext(signature); },
                    /*used_graph_replay=*/false);
            }
        }

        if (success)
        {
            recordMoEOverlayCollectiveTransaction(
                effective_input,
                moe_overlay_collective_runtime_stages,
                "cache_miss");
        }

        // Cache the graph for future matching forward signatures
        if (should_cache && build_cache && success)
        {
            // The original graph already occupies its final cache-owned address.
            build_cache->output = output;
            build_cache->workspace_generation = workspace_generation;
            build_cache->snapshot_configuration_epoch =
                executor_.snapshotConfigurationEpoch();

            // Pre-compute collective node set for fast decode intercept and
            // padded-prefill safety checks on later same-bucket cache hits.
            build_cache->collective_nodes = std::move(collective_nodes);

            build_cache->prefill_replay_param_stages = std::move(prefill_replay_param_stages);
            build_cache->prefill_replay_param_stages_cached = !is_decode;
            build_cache->moe_overlay_collective_runtime_stages =
                std::move(moe_overlay_collective_runtime_stages);
            build_cache->moe_overlay_collective_runtime_stages_cached = true;

            build_cache->valid = true;
            touchPrefillForwardCache(signature, *build_cache);

            // PP live rows belong to each subsequent host boundary, not this cache.

            LOG_DEBUG("[ForwardExecutionEngine] Cached forward graph for signature "
                      << "[seq_len=" << signature.seq_len
                      << ", batch_size=" << signature.batch_size
                      << ", device=" << signature.device.to_string()
                      << ", decode=" << signature.decode
                      << ", decode_has_history=" << signature.decode_has_history
                      << "] (" << build_cache->graph->size() << " stages)");

            if (signature.is_bucketed_prefill)
            {
                if (!build_cache->prefill_graph_cache)
                {
                    build_cache->prefill_graph_cache =
                        std::make_unique<PrefillGraphCache>(
                            makePrefillGraphConfigFromEnv(host));
                }

                PrefillGraphCacheKey key = makePrefillGraphKey(effective_input, host);
                if (bucketed_prefill_capture_candidate)
                {
                    build_cache->prefill_graph_cache->markWarmedUp(key);
                    publishPrefillGraphObservation(
                        *build_cache,
                        effective_input,
                        key,
                        PrefillGraphPhase::Warmup,
                        "warmup",
                        "none");
                    if (build_cache->prefill_graph_cache->config().trace)
                    {
                        LOG_INFO("[ForwardExecutionEngine] Prefill graph ARMED during cache miss warmup: seq_len="
                                 << effective_input.seq_len);
                    }
                }
                else
                {
                    publishPrefillGraphObservation(
                        *build_cache,
                        effective_input,
                        key,
                        PrefillGraphPhase::Cold,
                        "rejected",
                        toString(bucketed_prefill_reject_reason),
                        bucketed_prefill_reject_stage_name,
                        bucketed_prefill_reject_stage_type);
                }
            }
            if (!signature.decode)
                enforcePrefillForwardCapacity(host, &signature);
        }

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;

        LOG_DEBUG("[ForwardExecutionEngine] Forward completed in " << ms << "ms, success=" << success);

        return success;
    }

    // =========================================================================
    // collectTimeline() — GPU Stage Timing
    // =========================================================================

    void ForwardExecutionEngine::collectTimeline(
        IForwardExecutionHost &host,
        IDeviceContext *ctx,
        bool is_decode,
        const ForwardInput &input,
        std::chrono::high_resolution_clock::time_point start,
        std::string stage_context)
    {
        if (!PerfStatsCollector::gpuStageEventTimingEnabled() ||
            !ctx || !ctx->deviceId().is_gpu())
        {
            return;
        }

        auto &timeline = executor_.stageTimeline();
        if (!timeline.isInitialized())
        {
            return;
        }

        // When suppressed (warmup), discard any stale events but don't
        // collect or accumulate. This prevents stale events from the Cold
        // phase persisting after graph capture invalidates the stream state.
        if (suppress_timeline_)
        {
            timeline.resetTimings();
            return;
        }

        IWorkerGPUContext *gpu_ctx = host.getWorkerGPUContext(ctx->deviceId());
        if (!gpu_ctx)
        {
            LOG_ERROR("[ForwardExecutionEngine] Host did not provide a worker GPU context for timeline collection on "
                      << ctx->deviceId().toString());
            return;
        }
        timeline.collect(gpu_ctx);

        double wall_ms = std::chrono::duration<double, std::milli>(
                             std::chrono::high_resolution_clock::now() - start)
                             .count();
        std::string dev_str = ctx->deviceId().toString();
        const char *dev_name = dev_str.c_str();
        const char *phase_name = is_decode ? "decode" : "prefill";
        PerfStatsCollector::Tags stage_tags;
        if (!stage_context.empty())
            stage_tags.emplace("context", std::move(stage_context));
        timeline.recordPerfStats(phase_name, dev_name, "stage_gpu", std::move(stage_tags));

        if (is_decode)
        {
            // Accumulate decode iterations — print once via flushStageTimeline()
            timeline.accumulateIteration(wall_ms);
        }
        else if (accumulate_prefill_)
        {
            // Accumulate prefill iterations — print once via flushStageTimeline() (benchmark mode)
            int tokens = input.batch_size * input.seq_len;
            timeline.accumulatePrefillIteration(wall_ms, tokens);
        }
        else
        {
            // Flush any pending decode data before printing prefill
            timeline.printAccumulatedSummary("DECODE", dev_name);

            int tokens = input.batch_size * input.seq_len;
            timeline.printSummary("PREFILL", tokens, wall_ms, dev_name);
            if (debugEnv().gpu_stage_timing_detail)
                timeline.printDetailedTimeline("PREFILL", dev_name);
        }
        timeline.resetTimings();
    }

    void ForwardExecutionEngine::forEachCachedStage(
        ComputeStageType type,
        const std::function<void(IComputeStage *)> &visitor) const
    {
        for (const auto &[sig, cache] : cache_)
        {
            if (!cache.valid || !cache.graph)
                continue;
            for (const auto &node_name : cache.graph->getExecutionOrder())
            {
                auto *node = cache.graph->getNode(node_name);
                if (node && node->stage && node->stage->type() == type)
                    visitor(node->stage.get());
            }
        }
    }

    void ForwardExecutionEngine::forEachCachedStage(
        const std::function<void(IComputeStage *)> &visitor) const
    {
        for (const auto &[sig, cache] : cache_)
        {
            (void)sig;
            if (!cache.valid || !cache.graph)
                continue;
            for (const auto &node_name : cache.graph->getExecutionOrder())
            {
                auto *node = cache.graph->getNode(node_name);
                if (node && node->stage)
                    visitor(node->stage.get());
            }
        }
    }

    std::optional<ForwardExecutionEngine::PrefillGraphCacheSnapshot> ForwardExecutionEngine::prefillGraphCacheSnapshot(
        const ForwardGraphSignature &signature,
        const PrefillGraphCacheKey &key) const
    {
        auto it = cache_.find(signature);
        if (it == cache_.end())
            return std::nullopt;

        PrefillGraphCacheSnapshot snapshot;
        const ForwardGraphCache &forward_cache = it->second;
        snapshot.forward_cache_valid = forward_cache.valid;
        snapshot.eviction_count = prefill_forward_eviction_count_;
        snapshot.bucket_seq_len = key.seq_len;
        snapshot.domain_id = key.domain_id;
        snapshot.participant_id = key.participant_id;
        snapshot.placement_epoch = key.placement_epoch;
        snapshot.topology_signature = key.topology_signature;
        snapshot.capture_phase = "unknown";
        snapshot.recapture_reason = "none";
        snapshot.reject_stage_name.clear();
        snapshot.reject_stage_type.clear();

        if (forward_cache.last_prefill_graph_observation.valid)
        {
            const auto &observation = forward_cache.last_prefill_graph_observation;
            snapshot.observation_valid = true;
            snapshot.chunk_index = observation.chunk_index;
            snapshot.bucket_seq_len = observation.bucket_seq_len;
            snapshot.real_token_start = observation.real_token_start;
            snapshot.real_token_count = observation.real_token_count;
            snapshot.real_token_end = observation.real_token_end;
            snapshot.domain_id = observation.domain_id;
            snapshot.participant_id = observation.participant_id;
            snapshot.placement_epoch = observation.placement_epoch;
            snapshot.topology_signature = observation.topology_signature;
            snapshot.capture_phase = observation.capture_phase;
            snapshot.recapture_reason = observation.recapture_reason;
            snapshot.reject_stage_name = observation.reject_stage_name;
            snapshot.reject_stage_type = observation.reject_stage_type;
        }

        if (forward_cache.segment_cache.initialized ||
            forward_cache.segment_cache.successful_capture_count != 0u)
        {
            applySegmentedPrefillGraphSnapshot(
                forward_cache.segment_cache, &snapshot);
            return snapshot;
        }

        if (!forward_cache.prefill_graph_cache)
            return snapshot;

        const PrefillGraphCache &prefill_cache = *forward_cache.prefill_graph_cache;
        snapshot.prefill_cache_initialized = true;
        snapshot.phase = prefill_cache.phase(key);
        snapshot.cache_size = prefill_cache.size();
        snapshot.node_count = prefill_cache.nodeCount(key);
        snapshot.replay_count = prefill_cache.replayCount(key);
        snapshot.warmup_count = prefill_cache.warmupCount(key);
        snapshot.initialized_count = prefill_cache.initializedCount(key);
        snapshot.capture_count = prefill_cache.captureCount(key);
        snapshot.eviction_count += prefill_cache.evictionCount();
        return snapshot;
    }

} // namespace llaminar2
