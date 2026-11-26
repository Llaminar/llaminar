/**
 * @file DeviceGraphOrchestratorRebalance.cpp
 * @brief DeviceGraphOrchestrator method shard (13852..17630).
 */
#include "DeviceGraphOrchestratorInternal.h"

namespace llaminar2
{

    bool DeviceGraphOrchestrator::maybeRunDeviceMoERebalanceMaintenanceGraph(
        MoEOverlayEpochMaintenanceBoundarySource boundary_source)
    {
        const auto &env = debugEnv();
        if (!env.moe_rebalance.device_rebalance_maintenance_graph)
        {
            if (state_.device_id.is_gpu() &&
                usesParticipantLocalDeviceMoERebalanceController())
            {
                throw std::runtime_error(
                    "Single-domain native device-resident ExpertOverlay maintenance cannot run because its captured graph is disabled");
            }
            return true;
        }

        /*
         * Static placement has no maintenance transaction. Keep this typed
         * policy check ahead of device/context inspection and timing so Static
         * pays no per-token controller tax even when maintenance support is
         * enabled globally.
         */
        if (!isMoeRebalancingActive())
            return true;

        if (!state_.device_id.is_gpu() ||
            compute_all_position_logits_ ||
            !graph_builder_)
        {
            return true;
        }

        if (!usesParticipantLocalDeviceMoERebalanceController())
            return true;

        const std::string device_key = state_.device_id.toString();
        const auto fatal_maintenance =
            [&](const std::string &reason) -> bool
        {
            /*
             * A device-owned maintenance failure is never a recoverable
             * scheduling miss. The graph may own newer routing or expert
             * publication state than the main decode stream, so returning a
             * soft error would let a future caller accidentally continue with
             * an ambiguous ownership timeline.
             */
            throw std::runtime_error(
                "Fatal device MoE rebalance maintenance failure on " +
                device_key + ": " + reason);
        };
        const PerfStatsCollector::Tags scheduler_tags = {
            {"scheduling_authority", "device_commit_boundary"},
            {"host_position_authority", "false"},
            {"boundary_source",
             moeOverlayEpochMaintenanceBoundarySourceName(
                 boundary_source)}};
        PerfStatsCollector::addCounter(
            "moe_rebalance",
            "device_maintenance_scheduler_ticks",
            1.0,
            "decode",
            device_key,
            scheduler_tags);

        auto &cache = device_moe_rebalance_maintenance_graph_;
        const PerfStatsCollector::Tags maintenance_tags = {
            {"scheduling_authority", "device_commit_boundary"},
            {"scheduled_plan_launch", "device_gated"}};

        PerfStatsCollector::ScopedTimer maintenance_timer(
            "moe_rebalance",
            "device_maintenance_graph_boundary_enqueue",
            "decode",
            device_key,
            maintenance_tags);

        IDeviceContext *ctx = getDeviceContext(state_.device_id);
        if (!ctx)
        {
            return fatal_maintenance(
                "could not resolve the device context");
        }

        IWorkerGPUContext *gpu_ctx = nullptr;
        try
        {
            gpu_ctx = &GPUDeviceContextPool::instance().getContext(state_.device_id);
        }
        catch (const std::exception &e)
        {
            return fatal_maintenance(
                "could not resolve the GPU worker context: " +
                std::string(e.what()));
        }
        if (!gpu_ctx)
        {
            return fatal_maintenance(
                "resolved a null GPU worker context");
        }

        /*
         * The maintenance graph owns the complete device transaction. Launches
         * are serialized by its dedicated stream, and production consumers
         * observe the newest publication through the terminal event below.
         * No event query, D2H status read, or host rendezvous is permitted in
         * this scheduling path.
         */
        auto &active_cache = cache;
        constexpr const char *active_kind_name = "atomic";

        if (!active_cache.graph)
        {
            return fatal_maintenance(
                "the production maintenance graph was not retained by eager "
                "graph-family declaration");
        }
        if (active_cache.collective_nodes.empty())
        {
            return fatal_maintenance(
                "the retained maintenance graph has no declared collective nodes");
        }

        const uint64_t current_generation =
            workspaceGeneration(state_.device_id);
        if (active_cache.workspace_generation == 0 ||
            active_cache.workspace_generation != current_generation)
        {
            return fatal_maintenance(
                "the retained maintenance graph does not own the published "
                "workspace generation");
        }

        if (!active_cache.segment_cache.ensureCaptureStream(
                gpu_ctx,
                state_.device_id,
                /*context_from_process_pool=*/true))
        {
            return fatal_maintenance(
                "could not create the explicit capture stream");
        }
        void *maintenance_stream = active_cache.segment_cache.capture_stream;
        if (!maintenance_stream)
        {
            return fatal_maintenance(
                "capture-stream creation returned a null stream");
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
        {
            return fatal_maintenance(
                "could not resolve the GPU backend");
        }
        const int device_ordinal = state_.device_id.gpu_ordinal();

        std::string cadence_error;
        if (!prepareDeviceMoERebalanceCadenceTransactionSources(
                gpu_ctx,
                backend,
                maintenance_stream,
                &cadence_error) ||
            !composeDeviceMoERebalanceCadenceTransaction(
                &cadence_error))
        {
            return fatal_maintenance(
                "could not materialize cadence-gated scheduling: " +
                cadence_error);
        }

        /*
         * The maintenance graph reads the canonical device-resident MoE
         * histograms, so every launch consumes the complete committed
         * inference-state timeline. Waiting only on MainDecode is insufficient
         * for MTP because the verifier can consume its one-shot logits handoff
         * before accepted-state publication reaches this scheduler.
         *
         * This dependency must remain outside the reusable maintenance graph.
         * On transaction zero the executor invokes this hook only after native
         * capture and instantiation; on replay it invokes the same hook directly
         * before launch. The callback queues event waits on the exact graph
         * stream and performs no host synchronization or state transfer.
         */
        DeviceGraphExecutor::GraphLaunchDependencyHook maintenance_launch_dependency =
            [&](DeviceGraphExecutor::GraphExecutableLaunchPhase phase,
                void *execution_stream) -> bool
        {
            if (!execution_stream || execution_stream != maintenance_stream)
            {
                LOG_ERROR(
                    "[DeviceGraphOrchestrator] Device MoE maintenance launch "
                    "dependency received a stream other than the cache-owned "
                    "execution stream on "
                    << device_key
                    << " expected=" << maintenance_stream
                    << " actual=" << execution_stream);
                return false;
            }

            auto dependency_tags = maintenance_tags;
            dependency_tags["launch_phase"] =
                graphExecutableLaunchPhaseTag(phase);
            PerfStatsCollector::ScopedTimer dependency_timer(
                "moe_rebalance",
                "device_maintenance_graph_stream_dependency",
                "decode",
                device_key,
                dependency_tags);
            if (boundary_source ==
                MoEOverlayEpochMaintenanceBoundarySource::
                    GraphLaunchDependency)
            {
                if (moe_overlay_epoch_execution_binding_ &&
                    moe_overlay_epoch_lease_lifecycle_->load() ==
                        MoEOverlayEpochLeaseState::ExternalSequence)
                {
                    LOG_ERROR(
                        "[DeviceGraphOrchestrator] Device MoE maintenance "
                        "reached executable launch while a complete MTP "
                        "execution sequence still owned the residency epoch on "
                        << device_key);
                    return false;
                }
                if (!waitForLiveInferenceStateReadyForObservation(
                        execution_stream,
                        "moe_device_rebalance_maintenance_before_executable_launch",
                        DeviceTimelineRole::MoERebalanceMaintenance))
                {
                    return false;
                }
                /*
                 * The observation join above is the first stream edge known to
                 * be after main inference, every MTP sidecar/verifier, and
                 * accepted-state publication. Close the external reader before
                 * maintenance can reserve or publish its peer bank.
                 */
                if (moe_overlay_epoch_execution_binding_ &&
                    moe_overlay_epoch_lease_lifecycle_->load() ==
                        MoEOverlayEpochLeaseState::ExternalReader &&
                    !releaseMoEOverlayEpochForExternalTransaction(
                        execution_stream,
                        "device_moe_maintenance_commit_boundary"))
                {
                    return false;
                }
            }
            else if (boundary_source ==
                         MoEOverlayEpochMaintenanceBoundarySource::
                             AuthenticatedDispatchTicket &&
                     moe_overlay_epoch_execution_binding_ &&
                     moe_overlay_epoch_lease_lifecycle_->load() !=
                         MoEOverlayEpochLeaseState::ReleasePublished)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Authenticated HIP maintenance ticket did not publish the required ExpertOverlay release edge on "
                          << device_key);
                return false;
            }
            else if (boundary_source !=
                     MoEOverlayEpochMaintenanceBoundarySource::
                         AuthenticatedDispatchTicket)
            {
                LOG_ERROR(
                    "[DeviceGraphOrchestrator] Device MoE maintenance received an invalid typed inference-boundary source on "
                    << device_key);
                return false;
            }
            if (device_resident_logical_sequence_state_mailbox_.valid() &&
                device_resident_logical_sequence_state_mailbox_.live_state_epoch ==
                    live_replay_state_epoch_)
            {
                snapshotDeviceResidentLogicalStatePhase(
                    DeviceResidentLogicalStateDiagnosticPhase::MaintenanceEntry,
                    execution_stream,
                    device_resident_logical_sequence_state_mailbox_.request_count,
                    device_resident_logical_sequence_state_mailbox_
                        .publication_generation);
            }
            PerfStatsCollector::addCounter(
                "moe_rebalance",
                "device_maintenance_graph_launch_dependencies",
                1.0,
                "decode",
                device_key,
                std::move(dependency_tags));
            return true;
        };

        recordMoERebalanceTransferStreamPaths(
            active_cache.graph.get(),
            device_key,
            maintenance_tags);

        /*
         * Preflight the ownership event before capture or replay can submit
         * any maintenance work. Once the graph is accepted by the backend,
         * this event is the only durable publication edge that protects the
         * next main/MTP consumer from partially visible routing or payload
         * state. Event allocation after launch would leave an unorderable
         * graph in flight if allocation failed.
         */
        if (!active_cache.completion_event)
        {
            void *raw_event = backend->createEvent(device_ordinal);
            if (!raw_event)
            {
                return fatal_maintenance(
                    "could not preflight the maintenance completion event");
            }

            active_cache.completion_event.reset(
                raw_event,
                [backend, device_ordinal](void *event)
                {
                    if (event)
                        backend->destroyEvent(event, device_ordinal);
                });
        }

        DeviceGraphExecutor::DecodeCapturePolicy policy =
            buildDecodeCapturePolicy(
                !active_cache.collective_nodes.empty(),
                ctx);
        policy.defer_final_sync = true;
        policy.launch_dependency = maintenance_launch_dependency;
        dgo_detail::bindMoEMaintenanceReplayContext(active_cache.segment_cache);

        bool used_graph_replay = false;
        bool used_cadence_parent = false;
        bool ok = false;
        {
            auto replay_tags = maintenance_tags;
            replay_tags["maintenance_graph_kind"] = active_kind_name;
            PerfStatsCollector::ScopedTimer replay_timer(
                "moe_rebalance",
                "device_maintenance_graph_replay_enqueue",
                "decode",
                device_key,
                std::move(replay_tags));
            if (active_cache.cadence_transaction_capture &&
                active_cache.cadence_transaction_capture->hasExecutable())
            {
                DeviceGraphCaptureController::prepareDeviceForGraphCapture(ctx);
                if (!policy.launch_dependency(
                        DeviceGraphExecutor::GraphExecutableLaunchPhase::
                            SteadyReplay,
                        maintenance_stream))
                {
                    return fatal_maintenance(
                        "cadence parent launch dependency failed");
                }
                if (!active_cache.cadence_transaction_capture
                         ->launchOnStream(maintenance_stream))
                {
                    return fatal_maintenance(
                        "cadence parent graph launch failed");
                }
                ok = true;
                used_graph_replay = true;
                used_cadence_parent = true;
                PerfStatsCollector::addCounter(
                    "moe_rebalance",
                    "device_maintenance_cadence_parent_launches",
                    1.0,
                    "decode",
                    device_key,
                    {{"backend",
                      active_cache.cadence_transaction_capture
                          ->backendName()},
                     {"predicate_authority", "device"},
                     {"host_predicate_read", "false"}});
            }
            else
            {
                const DirectMoEMaintenanceReplayResult direct_replay =
                    tryLaunchCapturedMoERebalanceMaintenanceGraphDirect(
                        active_cache.segment_cache,
                        ctx,
                        gpu_ctx,
                        device_key,
                        maintenance_tags,
                        policy.launch_dependency);
                if (direct_replay ==
                    DirectMoEMaintenanceReplayResult::Launched)
                {
                    ok = true;
                    used_graph_replay = true;
                }
                else
                {
                    ok = executor_.executeDecodeWithCapturePolicy(
                        *active_cache.graph,
                        ctx,
                        &active_cache.segment_cache,
                        maintenance_stream,
                        gpu_ctx,
                        &active_cache.collective_nodes,
                        policy,
                        &used_graph_replay);
                }
            }
        }
        if (!ok)
        {
            return fatal_maintenance(
                "captured graph launch or capture failed");
        }

        /*
         * Initial source capture belongs to the first real maintenance
         * transaction. Compose immediately after that capture publishes its
         * executable, so the next boundary cannot submit another unconditional
         * heavy graph merely to discover that the cadence is not due.
         */
        if (!used_cadence_parent &&
            !composeDeviceMoERebalanceCadenceTransaction(&cadence_error))
        {
            return fatal_maintenance(
                "could not publish the cadence-gated transaction after source capture: " +
                cadence_error);
        }

        if (device_resident_logical_sequence_state_mailbox_.valid() &&
            device_resident_logical_sequence_state_mailbox_.live_state_epoch ==
                live_replay_state_epoch_)
        {
            snapshotDeviceResidentLogicalStatePhase(
                DeviceResidentLogicalStateDiagnosticPhase::MaintenanceExit,
                maintenance_stream,
                device_resident_logical_sequence_state_mailbox_.request_count,
                device_resident_logical_sequence_state_mailbox_
                    .publication_generation);
        }

        {
            PerfStatsCollector::ScopedTimer record_timer(
                "moe_rebalance",
                "device_maintenance_graph_completion_record",
                "decode",
                device_key,
                maintenance_tags);
            if (!gpu_ctx->recordEventChecked(active_cache.completion_event.get(), maintenance_stream))
            {
                LOG_ERROR("[DGO] Device MoE rebalance maintenance graph failed to record completion event for "
                          << device_key);
                /*
                 * The graph may already have mutated the active ownership
                 * bank. Without a recorded completion event there is no legal
                 * stream dependency that can publish those writes. Throwing
                 * would unwind through objects whose device work is still
                 * unordered, so terminate the process at this backend contract
                 * violation.
                 */
                std::terminate();
            }
        }

        ++active_cache.launch_count;
        /*
         * Publish every maintenance wave through its recorded device event,
         * including its shared-domain NCCL/RCCL collectives. The maintenance
         * graph uses the enclosing graph family's communicator by construction.
         * The next main or MTP graph calls
         * waitForPendingDeviceMoERebalanceMaintenance() before replay and
         * inserts a streamWaitEvent() on its exact execution stream. That
         * device-side dependency serializes shared-communicator collectives
         * without blocking the host, synchronizing the whole device, or
         * reading live placement state back to the CPU.
         *
         * The old shared-communicator branch synchronized the maintenance
         * stream here and then cleared completion_event_in_flight. ROCm
         * therefore launched graph-owned maintenance without ever exercising
         * the production consumer-event handoff. Besides adding a host fence,
         * that made request-boundary correctness depend on backend timing.
         * Keep the event as a durable publication after diagnostics observe
         * completion. Every sidecar and main-forward stream independently
         * queues the manifest-validated wait, so one consumer can never retire
         * another consumer's ordering edge.
         */
        active_cache.completion_event_in_flight = true;
        PerfStatsCollector::Tags launch_tags;
        launch_tags["used_graph_replay"] = boolTag(used_graph_replay);
        launch_tags["scheduling_authority"] = "device_commit_boundary";
        launch_tags["maintenance_graph_kind"] = active_kind_name;
        launch_tags["collective_lane_policy"] =
            "shared_domain_event_ordered";
        launch_tags["completion_handoff"] = "device_event";
        launch_tags["live_state_observation_barrier"] =
            "queued_immediately_before_executable_launch";
        launch_tags["cadence_parent"] = boolTag(used_cadence_parent);
        if (used_cadence_parent)
        {
            PerfStatsCollector::addCounterWithSequence(
                "moe_rebalance",
                "device_maintenance_scheduler_transactions",
                1.0,
                {active_cache.launch_count},
                "decode",
                device_key,
                std::move(launch_tags));
        }
        else
        {
            PerfStatsCollector::addCounterWithSequence(
                "moe_rebalance",
                "device_maintenance_graph_launches",
                1.0,
                {active_cache.launch_count},
                "decode",
                device_key,
                std::move(launch_tags));
        }
        return true;
    }

    // =====================================================================
    // IForwardExecutionHost interface implementations
    // =====================================================================

    std::optional<std::size_t> DeviceGraphOrchestrator::retainedExactPrefillCacheEntries() const
    {
        const auto placement = graph_builder_ ? graph_builder_->config().ngram_table_placement : nullptr;
        if (!placement || !placement->fixedHeadPipeline() || !serving_graph_family_plan_)
            return std::nullopt;
        return serving_graph_family_plan_->prefill_bucket_rows.size() + kExpertOverlayPrefillGraphIdentityReserve;
    }

    void DeviceGraphOrchestrator::ensureForwardEngine()
    {
        if (forward_engine_)
            return;

        ForwardExecutionEngine::Config engine_config;
        engine_config.cache_config = cache_config_;
        engine_config.sparse_query_program_count = sparseQueryProgramCount();
        engine_config.pp_stage_config = pp_stage_config_;
        engine_config.pipeline_forward_edges = pipeline_forward_edges_;

        forward_engine_ = std::make_unique<ForwardExecutionEngine>(
            std::move(engine_config), executor_);

        // Forward current timeline flags
        forward_engine_->setSuppressTimeline(suppress_timeline_);
        forward_engine_->setAccumulatePrefill(accumulate_prefill_);
    }

    GraphBuildResult DeviceGraphOrchestrator::buildForwardGraph(
        const ForwardInput &input)
    {
        if(input.graph_submission_intent!=ForwardGraphSubmissionIntent::WorkspaceDeclaration &&
            !executableForwardSubmissionIntent(input.graph_submission_intent))
            throw std::invalid_argument("DGO has an unknown forward graph construction intent");
        ForwardInput build_input = input;
        if(build_input.graph_submission_intent!=ForwardGraphSubmissionIntent::WorkspaceDeclaration &&
            build_input.device.is_gpu() && build_input.batch_size==1 &&
            build_input.execution_role==ForwardExecutionRole::MainInference &&
            build_input.execution_phase==ForwardExecutionPhase::Prefill && !build_input.device_prefill_chunk &&
            (build_input.device_request_query || (build_input.token_ids_device==request_token_ids_dev_ &&
                build_input.position_ids_device==request_position_ids_dev_ && build_input.sequence_lengths_device==request_sequence_lengths_dev_)))
        {
            const auto original=DeviceRequestQueryBinding::fromArena(*arena_,build_input.device);
            if(build_input.device_request_query && build_input.device_request_query->captureIdentity()!=original.captureIdentity())
                throw std::invalid_argument("DGO prefill received a foreign original-request query binding");
            build_input.device_request_query=original;
            requireDeviceRequestQueryOwnsInput(build_input);
        }
        if (build_input.device.is_gpu())
        {
            if (!build_input.device_state_publication_stream)
            {
                build_input.device_state_publication_stream =
                    explicitGPUStreamForOperation(
                        "forward_graph_device_state_publication");
            }
            if (!build_input.device_state_publication_stream)
            {
                throw std::runtime_error(
                    "GPU forward graph construction requires an explicit "
                    "non-null device-state publication stream");
            }
        }
        if (pipeline_forward_edges_ && pipeline_forward_edges_->fixedHeadPLEConsumer())
            build_input.fixed_head_pipeline_port = pipeline_forward_edges_->fixedHeadPLEConsumer();
        if (graph_builder_ && (graph_builder_->config().ngram_replicated_ple_cohort || build_input.fixed_head_pipeline_port))
            build_input.replicated_ple_query = fixedHeadPLEQuery(build_input);
        auto session = buildGraph().forInput(build_input);

        auto finalize = [&](GraphBuildResult result) -> GraphBuildResult
        {
            if (!result)
                return result;
            applyForwardGraphConstructionIntent(result.graph(),build_input);

            std::string cadence_error;
            if (!appendHostedDeviceMoEDecodeCommitBoundary(
                    result.graph(),
                    build_input,
                    &cadence_error))
            {
                return GraphBuildResult(
                    "Failed to append the HIP device MoE decode commit "
                    "boundary: " +
                    cadence_error);
            }

            std::string epoch_error;
            if (!appendCapturedMainForwardMoEOverlayEpochTransaction(
                    result.graph(),
                    build_input,
                    &epoch_error))
            {
                return GraphBuildResult(
                    "Failed to enclose the main forward in its "
                    "captured ExpertOverlay epoch transaction: " +
                    epoch_error);
            }

            applyCurrentExpertPlacementToGraph(
                result.graph(),
                build_input.device_state_publication_stream);

            if (build_input.device.is_gpu())
            {
                publishGraphBuildDeviceStateReady(
                    main_graph_build_device_state_ready_,
                    build_input.device_state_publication_stream,
                    "forward_graph_build");
            }

            if (raw_expert_weights_released_after_graph_build_ || !graph_builder_)
                return result;

            const auto &cfg = graph_builder_->config();
            const bool release_enabled = cfg.moe.rebalance_config.release_raw_expert_weights ||
                                         debugEnv().moe_rebalance.release_raw_weights;
            if (!release_enabled || cfg.default_device.is_gpu())
                return result;

            size_t total_freed = 0;
            int stage_count = 0;
            PreparedWeightStore *summary_store = prepared_weight_store_.get();
            for (const auto &node_name : result.graph().getExecutionOrder())
            {
                ComputeNode *node = result.graph().getNode(node_name);
                if (!node || !node->stage || node->stage->type() != ComputeStageType::MOE_EXPERT_FFN)
                    continue;
                auto *moe = dynamic_cast<MoEExpertComputeStage *>(node->stage.get());
                if (!moe)
                    continue;
                if (!summary_store)
                    summary_store = moe->buildWeightContext().prepared_store;
                total_freed += moe->releaseRawExpertWeights();
                ++stage_count;
            }

            raw_expert_weights_released_after_graph_build_ = true;

            size_t cached_freed = 0;
            if (auto concrete_weight_manager = std::dynamic_pointer_cast<WeightManager>(weight_manager_))
            {
                cached_freed = concrete_weight_manager->releaseMoEExpertHostWeightData();
                concrete_weight_manager->logHostMemorySummary("after eager graph build raw release");
            }

            if (summary_store)
            {
                LOG_DEBUG("[DGO] Released raw expert weights after eager graph build across "
                          << stage_count << " MoE stages: " << (total_freed >> 20)
                          << " MB stage-owned + " << (cached_freed >> 20)
                          << " MB WeightManager-cached freed");
            }
            return result;
        };

        if (pp_stage_config_.has_value())
        {
            const auto &pp = pp_stage_config_.value();
            LOG_DEBUG("[DeviceGraphOrchestrator] Building PARTIAL forward graph: "
                      << "layers=[" << pp.first_layer << ", " << pp.last_layer << ") "
                      << "has_embedding=" << pp.has_embedding
                      << " has_lm_head=" << pp.has_lm_head);

            return finalize(session
                                .forPPStage(pp.first_layer, pp.last_layer,
                                            pp.has_embedding, pp.has_lm_head)
                                .buildPartial());
        }
        else
        {
            LOG_DEBUG("[DeviceGraphOrchestrator] Building FULL forward graph...");
            return finalize(session.buildForward());
        }
    }

    void DeviceGraphOrchestrator::applyCurrentExpertPlacementToGraph(
        ComputeGraph &graph,
        void *publication_stream)
    {
        const bool has_current_masks = !current_expert_masks_.empty();
        const bool has_current_replicas = current_expert_replica_participant_id_ >= 0;
        if (!has_current_masks && !has_current_replicas)
            return;

        int mask_count = 0;
        int replica_count = 0;
        auto ensure_graph_placement_publication_stream = [&]() -> void *
        {
            if (!publication_stream)
            {
                throw std::runtime_error(
                    "MoE current placement graph publication requires an explicit non-null GPU stream");
            }
            return publication_stream;
        };

        for (const auto &node_name : graph.getExecutionOrder())
        {
            ComputeNode *node = graph.getNode(node_name);
            if (!node || !node->stage || node->stage->type() != ComputeStageType::MOE_EXPERT_FFN)
                continue;

            auto *moe = dynamic_cast<MoEExpertComputeStage *>(node->stage.get());
            if (!moe)
                continue;

            if (moe->device().is_gpu())
            {
                moe->setGPUStream(ensure_graph_placement_publication_stream());
            }

            if (has_current_masks)
            {
                const int layer_idx = moe->layerIndex();
                if (layer_idx < 0 ||
                    layer_idx >= static_cast<int>(current_expert_masks_.size()))
                {
                    throw std::runtime_error(
                        "New MoE graph stage has layer " +
                        std::to_string(layer_idx) +
                        " outside the persisted expert-mask table");
                }

                const auto &mask =
                    current_expert_masks_[static_cast<size_t>(layer_idx)];

                /*
                 * Adopt current arrivals before releasing static-placement
                 * engines. PreparedWeightStore owns the packed lifetime, so
                 * this is handle publication rather than hot-path repacking.
                 * The order mirrors graph-stable rebalance publication and
                 * prevents a fresh stage from transiently dropping an engine
                 * that it is about to adopt from the shared store.
                 */
                if (!moe->registerAndPrepareNewExperts(mask, nullptr))
                {
                    throw std::runtime_error(
                        "New MoE graph stage failed to adopt prepared expert "
                        "engines for layer " + std::to_string(layer_idx));
                }
                (void)moe->releaseDepartedExperts(mask);
                moe->applyExpertMask(mask);
                ++mask_count;
            }

            if (has_current_replicas)
            {
                /*
                 * Replica dispatch becomes visible only after the matching
                 * residency mask and prepared engine handles are in place.
                 */
                moe->setReplicaSet(current_expert_replica_set_,
                                   current_expert_replica_participant_id_);
                ++replica_count;
            }
        }

        const std::string device =
            graph_builder_ ? graph_builder_->config().default_device.toString() : std::string{};
        if (mask_count > 0)
        {
            PerfStatsCollector::addCounter(
                "moe_rebalance",
                "expert_mask_graph_stage_applications",
                static_cast<double>(mask_count),
                "rebalance",
                device,
                {{"mask_epoch", std::to_string(current_expert_mask_epoch_)}});
        }
        if (replica_count > 0)
        {
            PerfStatsCollector::addCounter(
                "moe_rebalance",
                "replica_set_graph_stage_applications",
                static_cast<double>(replica_count),
                "rebalance",
                device,
                {{"domain_id", current_expert_replica_set_.domain_id},
                 {"participant", std::to_string(current_expert_replica_participant_id_)},
                 {"replica_epoch", std::to_string(current_expert_replica_epoch_)},
                 {"replicas", std::to_string(current_expert_replica_set_.num_replicated)}});
        }

        if (mask_count > 0 || replica_count > 0)
        {
            LOG_DEBUG("[DGO] Restored persisted MoE placement on "
                      << std::max(mask_count, replica_count)
                      << " newly built stage(s): masks=" << mask_count
                      << " replicas=" << replica_count
                      << " participant=" << current_expert_replica_participant_id_);
        }
    }

    IForwardExecutionHost::PPCopyInfo
    DeviceGraphOrchestrator::resolvePPCopyInfo(
        const ForwardInput &input) const
    {
        PPCopyInfo info;
        if (!input.external_hidden_state || !graph_builder_)
            return info;

        const auto &cfg = graph_builder_->config();
        const auto &bufs = graph_builder_->buffers();
        InferenceMode mode(cfg.activation_precision);

        TensorBase *working_buffer =
            bufs.layer_buffers.residual && mode.isHybridQ16()
                ? bufs.layer_buffers.residual
                : bufs.current_hidden;

        if (!working_buffer ||
            input.external_hidden_state == working_buffer)
            return info;

        if (input.batch_size <= 0 || input.seq_len <= 0 ||
            input.batch_size > std::numeric_limits<int>::max() / input.seq_len)
            throw std::invalid_argument("PP residual copy requires positive representable request rows");
        const auto rows = requireHostPipelineActivationRows(
            input.host_pipeline_activation_rows, input.external_hidden_state,
            cfg.default_device, input.batch_size, input.seq_len,
            input.graph_submission_intent == ForwardGraphSubmissionIntent::MaterializeExecutableWithoutLaunch
                ? HostPipelineBoundaryExecution::MaterializeOnly : HostPipelineBoundaryExecution::Execute);
        const ResidualRowGeometry residual_rows(cfg.residualWidth());
        const auto live_bytes = residual_rows.bytesForRows(input.external_hidden_state->shape(), rows);
        (void)residual_rows.bytesForRows(working_buffer->shape(), rows);
        const auto copy_elems = live_bytes / sizeof(float);

        if (mode.isHybridQ16())
        {
            size_t num_blocks = (copy_elems + 31) / 32;
            info.copy_bytes = num_blocks * sizeof(Q16_1Block);
        }
        else
        {
            info.copy_bytes = live_bytes;
        }

        info.external_hidden = input.external_hidden_state;
        info.working_buffer = working_buffer;
        info.device = cfg.default_device;
        info.needs_copy = true;

        LOG_DEBUG("[DeviceGraphOrchestrator] Resolved PP copy info: "
                  << info.copy_bytes << " bytes on "
                  << cfg.default_device.toString());
        return info;
    }

    bool DeviceGraphOrchestrator::publishForwardResultAtBoundary(
        const ForwardOutput &output,
        IDeviceContext *ctx)
    {
        if (!ctx)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Forward-result publication requires its execution context");
            return false;
        }

        const bool publishes_hidden =
            pp_stage_config_.has_value() &&
            !pp_stage_config_->has_lm_head;
        TensorBase *const result =
            publishes_hidden ? output.hidden : output.logits;
        if (!result)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Forward graph omitted its declared "
                      << (publishes_hidden ? "pipeline hidden-state" : "logits")
                      << " result on " << ctx->deviceId().toString());
            return false;
        }

        if (!ctx->isGPU())
            return true; // CPU execution is synchronous.

        if (!output.execution.valid ||
            output.execution.device != ctx->deviceId() ||
            !output.execution.stream)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] GPU forward-result publication requires exact producer provenance"
                      << " context_device=" << ctx->deviceId().toString()
                      << " producer_valid=" << output.execution.valid
                      << " producer_device=" << output.execution.device.toString()
                      << " producer_stream=" << output.execution.stream);
            return false;
        }

        /*
         * Re-record the graph output tensor's persistent completion event after
         * launch. Event records made while capture is active become graph nodes
         * and cannot represent replay completion. This post-launch publication
         * is a normal runtime event on the real producer stream. It deliberately
         * does not call ensureOnHost(): GPU inference keeps logits device-owned,
         * and logits() is the explicit host-observation boundary.
         */
        TransferEngine::publishDeviceWrite(
            result,
            ctx->deviceId(),
            output.execution.stream);
        return true;
    }

    DeviceGraphExecutor::DecodeCapturePolicy DeviceGraphOrchestrator::buildDecodeCapturePolicy(
        bool has_collective_nodes,
        IDeviceContext *ctx) const
    {
        DeviceGraphExecutor::DecodeCapturePolicy policy;

        const auto &env = debugEnv();
        policy.allow_fast_decode = env.execution.fast_decode;

        if (!policy.allow_fast_decode)
        {
            return policy;
        }

        const bool heterogeneous_execution_domain =
            hasHeterogeneousCollectiveExecutionDomain();
        const bool heterogeneous_segmented_backend_supported =
            heterogeneous_execution_domain &&
            supportsHeterogeneousSegmentedReplay();

        bool captured_collectives_backend_supported = false;
        std::string captured_collectives_reason =
            has_collective_nodes ? "capture_collectives_not_requested" : "no_collective_nodes";
        if (has_collective_nodes && env.execution.gpu_graph_capture_collectives)
        {
            captured_collectives_backend_supported =
                collectivesSupportCapturedGraph(&captured_collectives_reason);
        }

        /*
         * Collective capture is a property of the stage's LocalTP domain, not
         * of every participant reachable through an ExpertOverlay boundary.
         * A homogeneous CUDA continuation pair therefore keeps its NCCL
         * collectives inside captured segments even when explicit sparse
         * protocol stages cross to ROCm or CPU.  The heterogeneous topology
         * still selects segmented replay below; only those typed cross-domain
         * boundaries remain manual.
         */
        policy.collectives_graph_capturable =
            has_collective_nodes &&
            env.execution.gpu_graph_capture_collectives &&
            captured_collectives_backend_supported;

        policy.heterogeneous_segmented_enabled =
            heterogeneous_execution_domain &&
            heterogeneous_segmented_backend_supported;
        policy.graph_replay_plan_policy =
            policy.heterogeneous_segmented_enabled
                ? DeviceGraphExecutor::GraphReplayPlanPolicy::
                      AllowHeterogeneousBoundarySegmentation
                : DeviceGraphExecutor::GraphReplayPlanPolicy::
                      RequireFullGraph;

        /*
         * A homogeneous GPU collective graph has exactly one legal production
         * architecture: capture the collective inside the full graph. Do not
         * convert a missing capability, disabled collective capture switch, or
         * stale stage contract into eager execution. Diagnostics measure this
         * production topology through PerfStats GPU events and never weaken
         * the capture requirement.
         */
        if (has_collective_nodes &&
            env.execution.gpu_graphs &&
            ctx &&
            ctx->isGPU() &&
            !heterogeneous_execution_domain &&
            !policy.collectives_graph_capturable)
        {
            std::ostringstream detail;
            detail
                << "Homogeneous GPU collective decode requires one fully "
                   "captured graph, but collective capture was rejected: "
                << captured_collectives_reason;
            LOG_ERROR("[DeviceGraphOrchestrator] " << detail.str());
            throw std::runtime_error(detail.str());
        }

        const bool can_use_segmented_graph =
            !has_collective_nodes ||
            policy.heterogeneous_segmented_enabled ||
            policy.collectives_graph_capturable;

        /*
         * Profiling is deliberately absent from this policy. PerfStats records
         * explicit GPU events around graph replay; selecting eager execution to
         * obtain per-stage timings would measure a different engine and can
         * violate graph-owned MTP publication contracts.
         */
        policy.allow_cached_graph_replay =
            env.execution.gpu_graphs &&
            ctx && ctx->isGPU() &&
            can_use_segmented_graph;

        if (has_collective_nodes && env.execution.gpu_graph_capture_collectives)
        {
            PerfStatsCollector::addCounter(
                "forward_graph",
                "decode_collective_graph_capture_policy",
                1.0,
                "decode",
                ctx ? ctx->deviceId().toString() : std::string{},
	                {{"requested", "true"},
	                 {"allowed", boolTag(policy.collectives_graph_capturable)},
	                 {"backend_supported", boolTag(captured_collectives_backend_supported)},
	                 {"heterogeneous_execution_domain", boolTag(heterogeneous_execution_domain)},
	                 {"heterogeneous_segmentation_admitted", boolTag(policy.heterogeneous_segmented_enabled)},
	                 {"reason", captured_collectives_reason}});
        }
        return policy;
    }

    std::optional<DeviceGraphExecutor::DecodeCapturePolicy>
    DeviceGraphOrchestrator::buildDecodeCapturePolicyForGraph(
        const ComputeGraph &graph,
        bool has_collective_nodes,
        IDeviceContext *ctx,
        const char *consumer) const
    {
        auto policy = buildDecodeCapturePolicy(has_collective_nodes, ctx);
        std::string policy_error;
        if (!DeviceGraphCaptureController::
                constrainReplayPolicyToNativeEnvelope(
                    graph.nativeCaptureEnvelope(),
                    policy,
                    &policy_error))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] "
                      << (consumer ? consumer : "device-owned graph")
                      << " rejected graph-native capture envelope on "
                      << state_.device_id.toString() << ": "
                      << policy_error);
            return std::nullopt;
        }

        try
        {
            if (auto retained_parent =
                    makeMoEOverlayRetainedParentPlan(graph);
                retained_parent)
            {
                if (!policy.allow_cached_graph_replay)
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] "
                              << (consumer ? consumer
                                           : "ExpertOverlay graph")
                              << " requires its topology-composed parent, but "
                                 "native graph replay is unavailable on "
                              << state_.device_id.toString());
                    return std::nullopt;
                }
                policy.graph_replay_plan_policy =
                    retained_parent->replay_policy;
                policy.retained_parent_composer =
                    std::move(retained_parent->composer);
                policy.defer_final_sync = true;
            }
        }
        catch (const std::exception &error)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] "
                      << (consumer ? consumer : "ExpertOverlay graph")
                      << " could not derive its retained parent on "
                      << state_.device_id.toString() << ": "
                      << error.what());
            return std::nullopt;
        }
        return policy;
    }

    bool DeviceGraphOrchestrator::hasHeterogeneousCollectiveExecutionDomain() const
    {
        if (pipeline_forward_edges_ && pipeline_forward_edges_->hasHeterogeneousBoundary())
            return true;
        const auto &graph_cfg = graph_builder_->config();
        std::optional<DeviceType> first_type;
        bool mixed_device_types = false;

        const auto observe_type =
            [&](DeviceType type)
        {
            if (!first_type)
            {
                first_type = type;
            }
            else if (*first_type != type)
            {
                mixed_device_types = true;
            }
        };

        const ITPContext *tp_ctx = graph_cfg.tp_ctx;
        if (tp_ctx && tp_ctx->isLocal() && tp_ctx->degree() > 1)
        {
            if (const auto *local_tp =
                    dynamic_cast<const ILocalTPContext *>(tp_ctx))
            {
                for (const auto &device : local_tp->devices())
                {
                    observe_type(device.device_type);
                }
            }
        }

        /*
         * ExpertOverlay may keep its continuation TP domain homogeneous while
         * adding CPU or cross-vendor routed-expert domains. Those participants
         * are part of the same forward graph and therefore part of the
         * segmentation topology proof.
         */
        std::shared_ptr<const MoERoutedExpertPlacementPlan> overlay_plan =
            graph_cfg.moe.routed_expert_plan;
        if (!overlay_plan && graph_cfg.moe.expert_overlay_runtime_plan)
        {
            overlay_plan =
                graph_cfg.moe.expert_overlay_runtime_plan->sourcePlanPtr();
        }
        if (overlay_plan)
        {
            for (const auto &domain :
                 overlay_plan->dense_domains)
            {
                for (const auto &participant : domain.participants)
                {
                    observe_type(participant.device_type);
                }
            }
            for (const auto &domain :
                 overlay_plan->domains)
            {
                for (const auto &participant : domain.participants)
                {
                    observe_type(participant.device_type);
                }
            }
        }

        if (domain_config_)
        {
            for (const auto &domain : domain_config_->domains())
            {
                for (const auto &device : domain.devices)
                {
                    observe_type(device.type);
                }
            }
        }

        return mixed_device_types;
    }

    bool DeviceGraphOrchestrator::supportsHeterogeneousSegmentedReplay() const
    {
        const auto &graph_cfg = graph_builder_->config();
        const bool has_local_tp =
            graph_cfg.tp_ctx &&
            graph_cfg.tp_ctx->isLocal() &&
            graph_cfg.tp_ctx->degree() > 1;

        /*
         * Backend capability is deliberately subordinate to the topology
         * proof. Homogeneous LocalTP must capture NCCL/RCCL inside one graph;
         * it may never select this manual-boundary path merely because the
         * backend happens to support stream-ordered collectives.
         */
        if (!hasHeterogeneousCollectiveExecutionDomain())
        {
            LOG_DEBUG(
                "[DeviceGraphOrchestrator] Disabling segmented GPU-graph replay "
                "because the LocalTP collective domain is not heterogeneous");
            return false;
        }

        // For Local TP (multi-device, single MPI rank), the per-device
        // DeviceGraphOrchestrator may not have an injected_collective_ctx_
        // because RankOrchestrator owns the LocalTPContext.
        const bool single_rank_collectives =
            (injected_collective_ctx_ && injected_collective_ctx_->worldSize() == 1) ||
            has_local_tp;

        if (has_local_tp && single_rank_collectives)
        {
            const auto backend = graph_cfg.tp_ctx->backend();
            const bool supported =
                (backend == CollectiveBackendType::NCCL ||
                 backend == CollectiveBackendType::RCCL ||
                 backend == CollectiveBackendType::HETEROGENEOUS ||
                 backend == CollectiveBackendType::HOST);
            if (supported)
                return true;

            LOG_DEBUG("[DeviceGraphOrchestrator] Disabling heterogeneous segmented GPU-graph replay for unsupported LocalTP backend");
        }

        /*
         * A routed ExpertOverlay plan is itself the typed heterogeneous
         * boundary declaration. Its host descriptor, sparse dispatch/return,
         * and local-expert stages independently opt into manual replay; the
         * capture planner still rejects every other uncapturable stage.
         */
        if (graph_cfg.moe.routed_expert_plan ||
            graph_cfg.moe.expert_overlay_runtime_plan)
            return true;

        LOG_DEBUG("[DeviceGraphOrchestrator] Disabling heterogeneous segmented GPU-graph replay because no supported LocalTP or ExpertOverlay boundary exists");
        return false;
    }

    bool DeviceGraphOrchestrator::collectivesSupportCapturedGraph(std::string *reason_out) const
    {
        auto reject = [&](const char *reason, const std::string &message) -> bool
        {
            if (reason_out)
            {
                *reason_out = reason;
            }
            LOG_DEBUG(message);
            return false;
        };

        auto accept = [&]() -> bool
        {
            if (reason_out)
            {
                *reason_out = "supported";
            }
            return true;
        };

        const auto &graph_cfg = graph_builder_->config();
        const ITPContext *tp_ctx = pipeline_forward_edges_
            ? pipeline_forward_edges_->context() : graph_cfg.tp_ctx;
        if (!tp_ctx && pipeline_forward_edges_ && pipeline_forward_edges_->hasDomainBoundary())
            return accept(); // A single member records only the explicit captured channel, not a host collective.
        if (!tp_ctx || !tp_ctx->isLocal() || tp_ctx->degree() <= 1)
        {
            return reject(
                "non_local_or_single_device_tp",
                "[DeviceGraphOrchestrator] Disabling captured collective GPU graphs for non-local or single-device TP");
        }

        const auto backend = tp_ctx->backend();
        if (backend != CollectiveBackendType::NCCL &&
            backend != CollectiveBackendType::RCCL)
        {
            return reject(
                "unsupported_backend",
                std::string("[DeviceGraphOrchestrator] Disabling captured collective GPU graphs for backend ") +
                    collectiveBackendTypeToString(backend));
        }
        const auto *local_tp = dynamic_cast<const ILocalTPContext *>(tp_ctx);
        if (!local_tp)
        {
            return reject(
                "missing_local_device_list",
                "[DeviceGraphOrchestrator] Disabling captured collective GPU graphs because LOCAL TP context lacks local device list");
        }

        const auto &devices = local_tp->devices();
        if (static_cast<int>(devices.size()) != tp_ctx->degree() || devices.empty())
        {
            return reject(
                "device_list_degree_mismatch",
                "[DeviceGraphOrchestrator] Disabling captured collective GPU graphs because device list does not match TP degree");
        }

        const DeviceType expected_type =
            backend == CollectiveBackendType::NCCL ? DeviceType::CUDA : DeviceType::ROCm;
        const std::string &local_tp_hostname = devices.front().hostname;
        for (const auto &device : devices)
        {
            /*
             * GlobalDeviceAddress::isLocal() recognizes only the shorthand
             * hostname "localhost". Runtime topology discovery canonicalizes
             * that shorthand to the node's real hostname, so using isLocal()
             * here falsely rejected a valid rank-local NCCL domain. The typed
             * LocalTP context plus one shared hostname is the correct locality
             * proof; backend/type agreement proves homogeneity.
             */
            if (device.hostname != local_tp_hostname || !device.isGPU() ||
                device.device_type != expected_type)
            {
                return reject(
                    "non_homogeneous_local_tp_device",
                    std::string("[DeviceGraphOrchestrator] Disabling captured collective GPU graphs for non-homogeneous LocalTP device ") +
                        device.toString());
            }
        }

        /*
         * Other overlay domains may use another backend or device type. They
         * are separated by explicit manual sparse boundaries and do not alter
         * the capture capability of this homogeneous LocalTP subdomain.
         */
        return accept();
    }

    bool DeviceGraphOrchestrator::execute(ComputeGraph &graph, IDeviceContext *ctx)
    {
        // Ensure declared CPU/GPU workspace is allocated for kernels and stages.
        if (!ensureDeviceWorkspaceAllocated(graph))
            return false;
        return executor_.execute(graph, ctx);
    }

    bool DeviceGraphOrchestrator::ensureDeviceWorkspaceAllocated(
        const ComputeGraph &graph,
        int workspace_seq_len)
    {
        return ensureDeviceWorkspaceAllocated(
            graph,
            workspace_seq_len,
            WorkspaceGraphFamilyPolicy::SerialDeviceFamilyExactParticipant);
    }

    bool DeviceGraphOrchestrator::ensureDeviceWorkspaceAllocated(
        const ComputeGraph &graph,
        int workspace_seq_len,
        WorkspaceGraphFamilyPolicy graph_family_policy)
    {
        return ensureDeviceWorkspaceAllocated(
            graph,
            workspace_seq_len,
            graph_family_policy,
            graph_family_policy ==
                    WorkspaceGraphFamilyPolicy::
                        SerialDeviceFamilyLargestParticipant
                ? WorkspaceGraphParticipantRole::Prefill
                : WorkspaceGraphParticipantRole::Decode,
            {});
    }

    bool DeviceGraphOrchestrator::ensureDeviceWorkspaceAllocated(
        const ComputeGraph &graph,
        int workspace_seq_len,
        WorkspaceGraphFamilyPolicy graph_family_policy,
        WorkspaceGraphParticipantRole participant_role)
    {
        return ensureDeviceWorkspaceAllocated(
            graph,
            workspace_seq_len,
            graph_family_policy,
            participant_role,
            {});
    }

    bool DeviceGraphOrchestrator::ensureDeviceWorkspaceAllocated(
        const ComputeGraph &graph,
        int workspace_seq_len,
        WorkspaceGraphFamilyPolicy graph_family_policy,
        WorkspaceGraphParticipantRole primary_role,
        const std::vector<WorkspaceGraphParticipant> &
            exact_serial_participants)
    {
        const auto &config = graph_builder_->config();
        /*
         * Execution enablement and retained setup capacity are deliberately
         * different policies. A runner may keep MTP execution disabled while
         * reserving a wider graph family for a later model-context lease. Use
         * the canonical retained-capacity predicate for every workspace
         * contributor; consulting `enabled` here would make the first runner
         * under-allocate the model-lifetime backing block even though memory
         * admission already charged the complete declared envelope.
         */
        const bool retains_mtp_graph_capacity =
            retainsMTPGraphCapacity(config.mtp);
        if (!workspace_allocator_)
        {
            if (reusable_execution_workspaces_ && state_.device_id.is_gpu())
            {
                if (!physical_memory_authority_)
                {
                    LOG_ERROR(
                        "[DeviceGraphOrchestrator] Reusable production workspace has no rank-local physical-memory authority for "
                        << state_.device_id.toString());
                    return false;
                }
                const int first_layer = pp_stage_config_.has_value()
                                            ? pp_stage_config_->first_layer
                                            : std::max(0, config.pp_layer_offset);
                const int last_layer = pp_stage_config_.has_value()
                                           ? pp_stage_config_->last_layer - 1
                                           : first_layer +
                                                 std::max(1, config.n_layers) - 1;
                const int tp_degree = config.tp_ctx
                                          ? std::max(1, config.tp_ctx->degree())
                                          : 1;
                const ReusableExecutionWorkspaceKey key{
                    .device = state_.device_id,
                    .first_layer = first_layer,
                    .last_layer = last_layer,
                    .tensor_parallel_participant =
                        std::max(0, config.tp_device_idx),
                    .tensor_parallel_degree = tp_degree,
                    .owns_embedding = !pp_stage_config_.has_value() ||
                                      pp_stage_config_->has_embedding,
                    .owns_terminal_head = !pp_stage_config_.has_value() ||
                                          pp_stage_config_->has_lm_head,
                };
                std::string acquire_error;
                reusable_workspace_lease_ =
                    reusable_execution_workspaces_->acquire(
                        key,
                        physical_memory_authority_,
                        &acquire_error);
                if (!reusable_workspace_lease_)
                {
                    LOG_ERROR(
                        "[DeviceGraphOrchestrator] Could not acquire reusable execution workspace for "
                        << state_.device_id.toString() << ": "
                        << (acquire_error.empty()
                                ? std::string("unknown workspace ownership failure")
                                : acquire_error));
                    return false;
                }
                workspace_allocator_ =
                    reusable_workspace_lease_->allocator();
            }
            else
            {
                workspace_allocator_ =
                    physical_memory_authority_
                        ? std::make_shared<WorkspaceAllocator>(
                              physical_memory_authority_)
                        : std::make_shared<WorkspaceAllocator>();
            }
        }

        const uint64_t workspace_generation_before =
            workspace_allocator_->deviceGeneration(state_.device_id);
        /*
         * Snapshot the durable owner, not the current mailbox view. Metadata
         * preparation deliberately clears the view before it may grow the
         * graph workspace, while previously issued handles remain valid and
         * continue to address these arena rows. This is exactly the lifetime
         * transition that exposed the old stale-workspace-pointer crash.
         */
        const auto logical_state_storage_before =
            device_resident_logical_sequence_state_storage_;
        const bool had_live_logical_state_publication =
            logical_state_storage_before.hasLivePublication(
                live_replay_state_epoch_);

        WorkspaceSizingHints hints;
        const int default_workspace_seq_len = config.max_seq_len > 0 ? config.max_seq_len : 4096;

        // Every production graph on one device belongs to an event-ordered
        // serial family. Ordinary prefill additionally sizes row-scaled scratch
        // for the largest resident bucket; compact decode/verifier demand is
        // queried independently below because it is not monotonic in M.
        hints.max_seq_len = workspace_seq_len > 0 ? workspace_seq_len : default_workspace_seq_len;
        hints.graph_family_policy = graph_family_policy;
        // Both serial-family participant policies publish the family's
        // largest prefill rows: an exact decode participant may create
        // generation one (prefix-hit warmup) and must not omit them.
        if (graph_family_policy !=
            WorkspaceGraphFamilyPolicy::ExclusiveLifetime)
        {
            const int largest_bucket_rows =
                state_.activation_seq_len > 0
                    ? state_.activation_seq_len
                    : resolveActivationBufferSeqLen(
                          default_workspace_seq_len,
                          state_.device_id);
            hints.serial_family_max_rows =
                std::max(1, largest_bucket_rows) *
                std::max(1, state_.batch_size);
        }
        if (graph_family_policy !=
                WorkspaceGraphFamilyPolicy::ExclusiveLifetime &&
            retains_mtp_graph_capacity)
        {
            hints.serial_family_max_compact_rows =
                std::max(1, mtp_max_verifier_rows_) *
                std::max(1, state_.batch_size);
        }
        hints.n_heads = config.n_heads > 0 ? config.n_heads : 128;
        hints.head_dim = config.d_model > 0 && config.n_heads > 0
                             ? config.d_model / config.n_heads
                             : 128;
        hints.d_model = config.d_model > 0 ? config.d_model : 896;
        hints.batch_size = state_.batch_size > 0 ? state_.batch_size : 1;
        hints.vocab_size = config.vocab_size > 0 ? config.vocab_size : 151936;
        if (graph_family_policy !=
                WorkspaceGraphFamilyPolicy::ExclusiveLifetime &&
            ((config.dense_tp_enabled &&
              config.dense_tp_decode_replicated) ||
             config.mtpUsesMirroredTerminalHeadBinding()))
        {
            /*
             * Column-parallel prefill may be the first graph captured, but
             * phase-split dense decode and mirrored LocalTP MTP verification
             * bind a full-vocabulary terminal projection later. Their
             * K-partition/scatter workspaces use stable names, so declare the
             * full terminal width while planning the first serial-family
             * participant. WorkspaceAllocator will still reject any capacity
             * omitted from this preflight instead of moving a captured
             * address.
             */
            hints.serial_family_max_terminal_projection_columns =
                hints.vocab_size;
        }

        std::vector<WorkspaceConsumerRequest> extras;
        if (state_.device_id.is_gpu() && config.ordinary_terminal_projection.hasExplicitRows() &&
            config.hasNativeLocalVocabularyGather())
        {
            const int ordinary_rows = config.ordinary_terminal_projection.resolve(
                state_.batch_size, config.max_activation_rows);
            // Preview the existing native terminal stage's exact named bank.
            // The graph keeps its frozen live rows; this request only publishes
            // admitted family storage before any captured pointer can escape.
            std::vector<NativeVocabularyAllGatherStage *> declared;
            const auto preview = [&](const ComputeGraph &participant)
            {
                for (const auto &name : participant.getExecutionOrder())
                {
                    const auto *node = participant.getNode(name);
                    auto *vocabulary = node && node->stage && node->device == state_.device_id
                        ? dynamic_cast<NativeVocabularyAllGatherStage *>(node->stage.get()) : nullptr;
                    if (vocabulary && vocabulary->params().rows.capacity() < ordinary_rows &&
                        std::find(declared.begin(), declared.end(), vocabulary) == declared.end())
                    {
                        declared.push_back(vocabulary);
                        extras.push_back({vocabulary, state_.device_id, ordinary_rows, 0, 0,
                                          WorkspaceConsumerShapePolicy::FixedDeclaredShape});
                    }
                }
            };
            preview(graph);
            for (const auto &participant : exact_serial_participants)
                if (participant.graph) preview(*participant.graph);
        }
        if (state_.kv_cache)
        {
            auto *kv_consumer = dynamic_cast<IWorkspaceConsumer *>(state_.kv_cache.get());
            if (kv_consumer && state_.device_id.is_gpu())
            {
                extras.push_back(WorkspaceConsumerRequest{
                    kv_consumer,
                    state_.device_id,
                    std::max(1, hints.max_seq_len),
                    std::max(1, hints.batch_size),
                    0,
                });
            }
        }
        for (auto &mtp_kv_cache : state_.mtp_kv_caches)
        {
            if (!mtp_kv_cache)
                continue;

            auto *kv_consumer = dynamic_cast<IWorkspaceConsumer *>(mtp_kv_cache.get());
            if (kv_consumer && state_.device_id.is_gpu())
            {
                extras.push_back(WorkspaceConsumerRequest{
                    kv_consumer,
                    state_.device_id,
                    std::max(1, hints.max_seq_len),
                    std::max(1, hints.batch_size),
                    0,
                });
            }
        }
        if (state_.device_id.is_gpu() && retains_mtp_graph_capacity)
        {
            /*
             * Verifier metadata is a configured graph-family participant, not a
             * demand discovered by the first verifier request. Publishing its
             * maximum request/depth geometry here prevents the first MTP step
             * from introducing new workspace names after prefill capture.
             * Request-lifetime logical state remains arena-owned and is
             * intentionally absent from this transient workspace declaration.
             */
            MTPSpecDecodeMetadataShape metadata_shape =
                mtp_spec_decode_metadata_binding_.capacity();
            metadata_shape.max_requests = std::max(
                metadata_shape.max_requests,
                std::max(1, config.mtp.max_request_batch));
            metadata_shape.max_draft_tokens = std::max(
                metadata_shape.max_draft_tokens,
                std::max(1, mtp_max_verifier_rows_));
            mtp_spec_decode_metadata_binding_.ensureCapacity(metadata_shape);
            extras.push_back(WorkspaceConsumerRequest{
                &mtp_spec_decode_metadata_binding_,
                state_.device_id,
                std::max(1, metadata_shape.max_requests),
                std::max(3, metadata_shape.max_draft_tokens),
                /*k=*/0,
                WorkspaceConsumerShapePolicy::FixedDeclaredShape,
            });
        }

        WorkspaceBudgetConfig workspace_budget;
        if (!workspace_allocator_->allocateForGraphFamily(
                graph,
                primary_role,
                exact_serial_participants,
                hints,
                extras,
                workspace_budget))
        {
            return false;
        }

        const uint64_t workspace_generation_after =
            workspace_allocator_->deviceGeneration(state_.device_id);
        if (had_live_logical_state_publication &&
            workspace_generation_after != workspace_generation_before)
        {
            const bool arena_ownership_intact =
                device_resident_logical_sequence_state_storage_
                    .hasLivePublication(live_replay_state_epoch_) &&
                device_resident_logical_sequence_state_storage_
                    .aliasesSameRowsAs(logical_state_storage_before);
            if (!arena_ownership_intact)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Graph workspace replacement invalidated arena-owned MTP logical sequence state");
                return false;
            }
            PerfStatsCollector::addCounter(
                "mtp",
                "device_resident_logical_state_workspace_rebind_survivals",
                1.0,
                perfPhaseName(),
                state_.device_id.toString(),
                {{"generation_before", std::to_string(workspace_generation_before)},
                 {"generation_after", std::to_string(workspace_generation_after)},
                 {"storage_owner", "arena_persistent"}});
        }
        return true;
    }

    uint64_t DeviceGraphOrchestrator::workspaceGeneration(DeviceId device) const
    {
        return workspace_allocator_ ? workspace_allocator_->deviceGeneration(device) : 0;
    }

    void DeviceGraphOrchestrator::onFirstGraphReady()
    {
        // Intentionally no-op for mmap reclaim. This callback fires when this
        // participant-local graph has a workspace, which is too early for TP/MoE
        // domains whose sibling participants may still be resolving captured
        // transfers. Reclaim happens after the first successful prefill instead.
    }

    void DeviceGraphOrchestrator::scheduleMmapReclaimAfterFirstPrefill()
    {
        if (!weight_manager_)
            return;

        /*
         * WeightManager reaches inference only after DeviceLoadPipeline has
         * drained every H2D and repack stream. Runtime graph execution consumes
         * the resulting device allocations, never the mmap source pages.
         * Publish reclaim to its prestarted owner worker; this authority thread
         * performs no host unregistration, page advice, or synchronization.
         */
        const auto submission = weight_manager_->scheduleMmapReclaim();
        if (debugEnv().vram_trace)
        {
            LOG_TRACE("[VRAM_TRACE] mmap_reclaim phase=after_first_prefill submission="
                      << MmapReclaimLifecycle::toString(submission));
        }
    }

    bool DeviceGraphOrchestrator::executeAttention(
        const LayerWeights &layer,
        ActivationBuffers &buffers,
        int layer_idx,
        int seq_len,
        IKVCache *kv_cache,
        const int *position_ids,
        DeviceId device)
    {
        // Debug: dump input to attention (for layer 0 only)
        if (layer_idx == 0)
        {
            const float *input = buffers.current_hidden->fp32_data();
            LOG_TRACE("[ORCH_ATTN_INPUT] layer=" << layer_idx << " seq_len=" << seq_len
                                                 << " input[0:4]=" << input[0] << "," << input[1]
                                                 << "," << input[2] << "," << input[3]);
        }

        // Get device context
        IDeviceContext *ctx = getDeviceContext(device);
        if (!ctx)
        {
            return false;
        }

        int pos_offset = position_ids ? position_ids[0] : 0;

        // =============================================================================
        // Graph Caching for Decode Mode (Phase 10)
        // =============================================================================
        if (cache_config_.enabled && cache_config_.cache_attention &&
            seq_len <= std::max(1, cache_config_.decode_seq_len) &&
            layer_idx >= 0 && static_cast<size_t>(layer_idx) < layer_graph_cache_.size())
        {
            auto &cache = layer_graph_cache_[layer_idx];

            // Check if we have a valid cached graph
            if (cache.attention_decode &&
                cache.attention_cached_seq_len == seq_len &&
                cache.attention_cached_all_position_logits == compute_all_position_logits_ &&
                cache.valid)
            {
                LOG_DEBUG("[DeviceGraphOrchestrator] Reusing cached attention graph for layer "
                          << layer_idx << " (pos_offset=" << pos_offset << ")");

                // Update dynamic parameters (position offset)
                updateCachedGraphParams(*cache.attention_decode, pos_offset, seq_len);

                // Execute cached graph
                bool success = executor_.execute(*cache.attention_decode, ctx);
                if (!success)
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Cached attention graph failed at layer " << layer_idx);
                }

                cache.attention_decode->reset();
                cache_stats_.attention_cache_hits++;
                return success;
            }

            // Build and cache the graph using fluent API
            LOG_DEBUG("[DeviceGraphOrchestrator] Building and caching attention graph for layer "
                      << layer_idx << " (decode mode)");

            auto result = buildAttentionGraph()
                              .forLayer(layer, layer_idx)
                              .withBuffers(buffers)
                              .withSequence(seq_len, 1)
                              .onDevice(device)
                              .withKVCache(kv_cache)
                              .withPositionIds(position_ids)
                              .build();

            if (!result)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Attention graph build failed: " << result.error());
                return false;
            }

            cache.attention_decode = std::make_unique<ComputeGraph>(result.takeGraph());
            cache.cached_seq_len = seq_len;
            cache.attention_cached_seq_len = seq_len;
            cache.attention_cached_all_position_logits = compute_all_position_logits_;
            cache.valid = true;
            cache_stats_.attention_cache_misses++;

            // Execute the newly built graph
            ensureDeviceWorkspaceAllocated(*cache.attention_decode);
            bool success = executor_.execute(*cache.attention_decode, ctx);
            if (!success)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Attention block failed at layer " << layer_idx);
            }

            cache.attention_decode->reset();
            return success;
        }

        // =============================================================================
        // Non-cached path (prefill or caching disabled) using fluent API
        // =============================================================================
        cache_stats_.attention_cache_misses++;

        auto result = buildAttentionGraph()
                          .forLayer(layer, layer_idx)
                          .withBuffers(buffers)
                          .withSequence(seq_len, 1)
                          .onDevice(device)
                          .withKVCache(kv_cache)
                          .withPositionIds(position_ids)
                          .build();

        if (!result)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Attention graph build failed: " << result.error());
            return false;
        }

        ComputeGraph graph = result.takeGraph();

        // Debug: log graph structure
        if (layer_idx == 0)
        {
            auto order = graph.getExecutionOrder();
            LOG_TRACE("[ORCH_ATTN] Graph has " << graph.size() << " nodes, execution order:");
            for (const auto &name : order)
            {
                LOG_TRACE("[ORCH_ATTN]   - " << name);
            }
        }

        ensureDeviceWorkspaceAllocated(graph);
        bool success = executor_.execute(graph, ctx);

        // Debug: dump intermediate buffers (for layer 0 only)
        if (layer_idx == 0 && buffers.normalized && buffers.Q &&
            buffers.attn_output && buffers.attn_proj)
        {
            const float *output = buffers.current_hidden->fp32_data();
            LOG_TRACE("[ORCH_ATTN_OUTPUT] layer=" << layer_idx << " seq_len=" << seq_len
                                                  << " output[0:4]=" << output[0] << "," << output[1]
                                                  << "," << output[2] << "," << output[3]);
        }

        if (!success)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Attention block failed at layer " << layer_idx);
        }

        return success;
    }

    bool DeviceGraphOrchestrator::executeFFN(
        const LayerWeights &layer,
        ActivationBuffers &buffers,
        int layer_idx,
        int seq_len,
        DeviceId device)
    {
        // Get device context
        IDeviceContext *ctx = getDeviceContext(device);
        if (!ctx)
        {
            return false;
        }

        // =============================================================================
        // Graph Caching for Decode Mode (Phase 10)
        // =============================================================================
        if (cache_config_.enabled && cache_config_.cache_ffn &&
            seq_len <= std::max(1, cache_config_.decode_seq_len) &&
            layer_idx >= 0 && static_cast<size_t>(layer_idx) < layer_graph_cache_.size())
        {
            auto &cache = layer_graph_cache_[layer_idx];

            // Check if we have a valid cached FFN graph
            if (cache.ffn_decode &&
                cache.ffn_cached_seq_len == seq_len &&
                cache.ffn_cached_all_position_logits == compute_all_position_logits_ &&
                cache.valid)
            {
                LOG_DEBUG("[DeviceGraphOrchestrator] Reusing cached FFN graph for layer " << layer_idx);

                // Execute cached graph (no params to update for FFN)
                bool success = executor_.execute(*cache.ffn_decode, ctx);
                if (!success)
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Cached FFN graph failed at layer " << layer_idx);
                }

                cache.ffn_decode->reset();
                cache_stats_.ffn_cache_hits++;
                return success;
            }

            // Build and cache the graph using fluent API
            LOG_DEBUG("[DeviceGraphOrchestrator] Building and caching FFN graph for layer "
                      << layer_idx << " (decode mode)");

            auto result = buildFFNGraph()
                              .forLayer(layer, layer_idx)
                              .withBuffers(buffers)
                              .withSequence(seq_len, 1)
                              .onDevice(device)
                              .withDeviceStatePublicationStream(
                                  device.is_gpu()
                                      ? explicitGPUStreamForOperation(
                                            "cached_ffn_graph_build")
                                      : nullptr)
                              .build();

            if (!result)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] FFN graph build failed: " << result.error());
                return false;
            }

            cache.ffn_decode = std::make_unique<ComputeGraph>(result.takeGraph());
            cache.cached_seq_len = seq_len;
            cache.ffn_cached_seq_len = seq_len;
            cache.ffn_cached_all_position_logits = compute_all_position_logits_;
            cache.valid = true;
            cache_stats_.ffn_cache_misses++;

            // Execute the newly built graph
            ensureDeviceWorkspaceAllocated(*cache.ffn_decode);
            bool success = executor_.execute(*cache.ffn_decode, ctx);
            if (!success)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] FFN block failed at layer " << layer_idx);
            }

            cache.ffn_decode->reset();
            return success;
        }

        // =============================================================================
        // Non-cached path (prefill or caching disabled) using fluent API
        // =============================================================================
        cache_stats_.ffn_cache_misses++;

        auto result = buildFFNGraph()
                          .forLayer(layer, layer_idx)
                          .withBuffers(buffers)
                          .withSequence(seq_len, 1)
                          .onDevice(device)
                          .withDeviceStatePublicationStream(
                              device.is_gpu()
                                  ? explicitGPUStreamForOperation(
                                        "ffn_graph_build")
                                  : nullptr)
                          .build();

        if (!result)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] FFN graph build failed: " << result.error());
            return false;
        }

        ComputeGraph graph = result.takeGraph();

        ensureDeviceWorkspaceAllocated(graph);
        bool success = executor_.execute(graph, ctx);

        if (!success)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] FFN block failed at layer " << layer_idx);
        }

        return success;
    }

    bool DeviceGraphOrchestrator::executeLayer(
        const LayerWeights &layer,
        ActivationBuffers &buffers,
        int layer_idx,
        int seq_len,
        IKVCache *kv_cache,
        const int *position_ids,
        DeviceId device)
    {
        LOG_TRACE("[DeviceGraphOrchestrator::executeLayer] LAYER_EXEC_ENTERED layer_idx="
                  << layer_idx << " seq_len=" << seq_len);

        // =====================================================================
        // Weight Streaming Hooks (Option B)
        // =====================================================================
        // Before layer execution: Ensure weights are on device
        // After layer execution: Release layer and prefetch next
        // =====================================================================
        if (weight_streamer_)
        {
            // Ensure this layer's weights are on the target device
            if (!weight_streamer_->ensureLayerOnDevice(layer_idx, device))
            {
                LOG_ERROR("[DeviceGraphOrchestrator::executeLayer] Failed to stream layer "
                          << layer_idx << " to device " << device.toString());
                return false;
            }

            // Prefetch next layer(s) asynchronously to overlap with compute
            int n_layers = graph_builder_ ? graph_builder_->config().n_layers : 0;
            if (layer_idx + 1 < n_layers)
            {
                weight_streamer_->prefetchLayer(layer_idx + 1, device);
                LOG_TRACE("[DeviceGraphOrchestrator::executeLayer] Prefetching layer " << (layer_idx + 1));
            }
        }

        // Execute attention block
        if (!executeAttention(layer, buffers, layer_idx, seq_len, kv_cache, position_ids, device))
        {
            // Release layer on failure if streaming
            if (weight_streamer_)
            {
                weight_streamer_->releaseLayer(layer_idx);
            }
            return false;
        }

        // Execute FFN block
        if (!executeFFN(layer, buffers, layer_idx, seq_len, device))
        {
            // Release layer on failure if streaming
            if (weight_streamer_)
            {
                weight_streamer_->releaseLayer(layer_idx);
            }
            return false;
        }

        // After layer execution: Release layer (marks as eligible for eviction)
        if (weight_streamer_)
        {
            weight_streamer_->releaseLayer(layer_idx);
            LOG_TRACE("[DeviceGraphOrchestrator::executeLayer] Released layer " << layer_idx);
        }

        return true;
    }

    // =========================================================================
    // Cache Management
    // =========================================================================

    void DeviceGraphOrchestrator::invalidateExecutionCaches()
    {
        /*
         * Use the same complete destructive transition as public buffer release
         * and object teardown.  Keeping one inventory prevents a newly added
         * MTP/controller graph from silently outliving the arena merely because
         * one reset path forgot to name it.
         */
        destroyCapturedExecutionTopologyBeforeArenaRelease();

        // Reset state
        last_pos_offset_ = -1;

        // Reset stats
        cache_stats_ = CacheStats{};

        // Reset input-dependent cached state on all kernels
        resetKernelDynamicState();
        clearCurrentMainLogitsPublication("invalidate_execution_caches");
        ++session_epoch_;

        LOG_DEBUG("[DeviceGraphOrchestrator] Execution caches invalidated");
    }

    void DeviceGraphOrchestrator::invalidateGraphCache(int layer_idx)
    {
        if (layer_idx < 0)
        {
            // Invalidate all layers
            for (auto &cache : layer_graph_cache_)
            {
                cache.invalidate();
            }
            cache_stats_.cached_layers = 0;
            LOG_DEBUG("[DeviceGraphOrchestrator] All layer graph caches invalidated");
        }
        else if (static_cast<size_t>(layer_idx) < layer_graph_cache_.size())
        {
            layer_graph_cache_[layer_idx].invalidate();
            LOG_DEBUG("[DeviceGraphOrchestrator] Layer " << layer_idx << " graph cache invalidated");
        }
    }

    void DeviceGraphOrchestrator::resetKernelDynamicState()
    {
        /*
         * KernelFactory and PreparedWeightStore own backend-side dynamic
         * launch metadata such as descriptor tables, pointer arrays, and
         * per-kernel parameter buffers. Cached ComputeGraphs may intentionally
         * survive this boundary, so any stage-local handles into those backend
         * tables must be invalidated explicitly instead of leaving stale IDs
         * behind for the next prompt or prefix-restore prefill.
         */
        llaminar::v2::kernels::KernelFactory::resetAllDynamicState();
        if (prepared_weight_store_)
        {
            prepared_weight_store_->resetDynamicState();
        }

        auto invalidate_graph_stage_handles = [](ComputeGraph *graph)
        {
            if (!graph)
                return;
            for (const auto &node_name : graph->getExecutionOrder())
            {
                ComputeNode *node = graph->getNode(node_name);
                if (node && node->stage)
                    node->stage->invalidateKernelDynamicState();
            }
        };

        if (forward_engine_)
        {
            forward_engine_->forEachCachedStage(
                [](IComputeStage *stage)
                {
                    if (stage)
                        stage->invalidateKernelDynamicState();
                });
        }

        for (auto &cache : layer_graph_cache_)
        {
            invalidate_graph_stage_handles(cache.attention_decode.get());
            invalidate_graph_stage_handles(cache.ffn_decode.get());
        }

        invalidate_graph_stage_handles(mtp_sidecar_depth0_cache_.graph.get());
        invalidate_graph_stage_handles(mtp_sidecar_depth0_device_token_cache_.graph.get());
        invalidate_graph_stage_handles(mtp_sidecar_depth0_chained_cache_.graph.get());
        invalidate_graph_stage_handles(mtp_sidecar_depth0_chained_device_token_cache_.graph.get());
        invalidate_graph_stage_handles(mtp_sidecar_depth0_kv_only_cache_.graph.get());
        invalidate_graph_stage_handles(mtp_sidecar_depth0_kv_only_device_token_cache_.graph.get());
        for (auto &cache : mtp_sidecar_depth0_kv_only_batch_caches_)
            invalidate_graph_stage_handles(cache->graph.get());

        invalidate_graph_stage_handles(mtp_terminal_hidden_row_select_cache_.graph.get());
        invalidate_graph_stage_handles(mtp_terminal_hidden_rows_select_cache_.graph.get());
        for (auto &cache : mtp_terminal_hidden_contiguous_rows_select_caches_)
        {
            if (cache)
                invalidate_graph_stage_handles(cache->graph.get());
        }
        for (auto &cache :
             mtp_terminal_hidden_device_accepted_rows_select_caches_)
        {
            if (cache)
                invalidate_graph_stage_handles(cache->graph.get());
        }
        for (auto &cache :
             mtp_terminal_hidden_request_rows_select_caches_)
        {
            if (cache)
                invalidate_graph_stage_handles(cache->graph.get());
        }
        for (auto &cache :
             mtp_shifted_prefill_hidden_rows_select_caches_)
        {
            if (cache)
                invalidate_graph_stage_handles(cache->graph.get());
        }
        invalidate_graph_stage_handles(device_moe_rebalance_maintenance_graph_.graph.get());
    }

    void DeviceGraphOrchestrator::recordKernelDynamicStatePreservedForCapturedReplay(
        const char *reason) const
    {
        PerfStatsCollector::addCounter(
            "mtp",
            "kernel_dynamic_state_preserved_for_captured_replay",
            1.0,
            "decode",
            state_.device_id.toString(),
            PerfStatsCollector::Tags{
                {"reason", reason ? reason : "unknown"},
                {"scope", "request_boundary"}});
    }

    bool DeviceGraphOrchestrator::hasValidCachedGraph(int layer_idx, bool is_attention) const
    {
        if (!cache_config_.enabled)
            return false;
        if (layer_idx < 0 || static_cast<size_t>(layer_idx) >= layer_graph_cache_.size())
            return false;

        const auto &cache = layer_graph_cache_[layer_idx];
        if (!cache.valid)
            return false;

        return is_attention ? (cache.attention_decode != nullptr) : (cache.ffn_decode != nullptr);
    }

    void DeviceGraphOrchestrator::setGraphCachingEnabled(bool enabled)
    {
        if (cache_config_.enabled != enabled)
        {
            cache_config_.enabled = enabled;
            if (!enabled)
            {
                invalidateGraphCache(-1);
            }
            LOG_DEBUG("[DeviceGraphOrchestrator] Graph caching "
                      << (enabled ? "enabled" : "disabled"));
        }
    }

    void DeviceGraphOrchestrator::initializeGraphCache(int n_layers)
    {
        layer_graph_cache_.resize(n_layers);
        cache_stats_.cached_layers = n_layers;
        LOG_DEBUG("[DeviceGraphOrchestrator] Graph cache initialized for " << n_layers << " layers");
    }

    // =========================================================================
    // Inference State Management
    // =========================================================================

    bool DeviceGraphOrchestrator::initializeInferenceStateFromArena(
        int batch_size,
        int max_seq_len,
        DeviceId device,
        const InferenceStateInitConfig &init_config)
    {
        if (!graph_builder_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Cannot initialize state: no graph builder");
            return false;
        }

        const auto &config = graph_builder_->config();
        const int activation_seq_len =
            init_config.activation_seq_len > 0
                ? std::min(max_seq_len, init_config.activation_seq_len)
                : max_seq_len;
        if (batch_size <= 0 || max_seq_len <= 0 || activation_seq_len <= 0)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Invalid inference state shape: batch_size="
                      << batch_size << " max_seq_len=" << max_seq_len
                      << " activation_seq_len=" << activation_seq_len);
            return false;
        }

        /*
         * Arena registration consumes immutable shape information.  Publish
         * it before `initializeBuffers()` so every persistent allocation sees
         * the configured capacity rather than InferenceState's zero-valued
         * construction defaults.  Runtime positions and sequence lengths are
         * still initialized only after their backing objects are available.
         */
        state_.batch_size = batch_size;
        state_.max_seq_len = max_seq_len;
        state_.activation_seq_len = activation_seq_len;
        state_.device_id = device;

        /*
         * Native CUDA/HIP graphs consume opaque slabs from a context-owned
         * driver pool. There is no per-executable allocation handle to lease,
         * and a free-memory delta around one instantiation can represent pool
         * capacity later consumed by several sibling graphs. Commit the exact
         * family BOM once, before any graph can be materialized. The retained
         * reservation is therefore the one PhysicalMemoryAuthority claim for
         * this runner's complete graph family; backend observations remain
         * certification evidence and never become a parallel live ledger.
         */
        if (device.is_gpu() && !native_graph_memory_reservation_)
        {
            if (!physical_memory_authority_)
            {
                if (memory_authority_mode_ ==
                    MemoryAuthorityMode::ProductionRequired)
                {
                    LOG_ERROR(
                        "[DeviceGraphOrchestrator] Production GPU graph family has no physical-memory authority for "
                        << device.toString());
                    return false;
                }
            }
            else
            {
                try
                {
                    const std::size_t admitted_graph_bytes =
                        physical_memory_authority_->plannedBytes(
                            device,
                            PhysicalMemoryOwner::NativeGraphExecutable);
                    if (admitted_graph_bytes != 0u)
                    {
                        native_graph_memory_reservation_ = std::make_shared<
                            PhysicalMemoryOwnerReservation>(
                            physical_memory_authority_->reserveNewAllocations(
                                device,
                                PhysicalMemoryOwner::NativeGraphExecutable,
                                admitted_graph_bytes));
                    }
                }
                catch (const std::exception &error)
                {
                    LOG_ERROR(
                        "[DeviceGraphOrchestrator] Could not commit native graph-pool admission on "
                        << device.toString() << ": " << error.what());
                    return false;
                }
            }
        }

        /*
         * Diagnostic D2D destinations are real, independently releasable GPU
         * allocations. Commit their complete caller-declared pool beside the
         * opaque native-graph pool, then let DeviceGraphExecutor issue one
         * child lease per arena. This occurs before snapshot enablement and
         * graph capture, so no graph can embed an unaccounted address.
         */
        if (device.is_gpu() && !graph_snapshot_memory_reservation_ &&
            physical_memory_authority_)
        {
            try
            {
                const std::size_t admitted_snapshot_bytes =
                    physical_memory_authority_->plannedBytes(
                        device,
                        PhysicalMemoryOwner::GraphSnapshotArena);
                if (admitted_snapshot_bytes != 0u)
                {
                    graph_snapshot_memory_reservation_ = std::make_shared<
                        PhysicalMemoryOwnerReservation>(
                        physical_memory_authority_->reserveNewAllocations(
                            device,
                            PhysicalMemoryOwner::GraphSnapshotArena,
                            admitted_snapshot_bytes));
                    executor_.setGraphSnapshotMemoryReservation(
                        graph_snapshot_memory_reservation_);
                }
            }
            catch (const std::exception &error)
            {
                LOG_ERROR(
                    "[DeviceGraphOrchestrator] Could not commit graph snapshot admission on "
                    << device.toString() << ": " << error.what());
                return false;
            }
        }
        if(config.retained_host_services && physical_memory_authority_){
            const RetainedHostServiceInstance role{physical_memory_authority_->worldRank(),device,RetainedHostServiceRole::ExecutorParentTicket};
            if(config.retained_host_services->contains(role)){
                const auto& program=config.retained_host_services->program(role);
                if(!executor_.hasRetainedParentTicketServiceWorker()){
                    auto worker=injected_model_ctx_->borrowRetainedHostService({role.worldRank,device},role.role,physical_memory_authority_);
                    executor_.setRetainedParentTicketServiceWorker(std::move(worker),physical_memory_authority_);
                }
                executor_.requireRetainedParentTicketServiceProgram(program,physical_memory_authority_);
            }
        }
        if (config.mtp.enabled && !device.is_gpu())
        {
            mtp_committed_verifier_identities_host_.assign(
                static_cast<size_t>(batch_size),
                sampling_math::MTPCommittedVerifierIdentityRecord{});
        }
        else
        {
            mtp_committed_verifier_identities_host_.clear();
        }

        // =====================================================================
        // Ensure TensorFactory exists (needed for arena allocation + snapshots)
        // =====================================================================
        if (!tensor_factory_)
        {
            std::shared_ptr<IMPIContext> local_mpi_ctx = mpi_ctx_;
            if (!local_mpi_ctx)
            {
                local_mpi_ctx = std::make_shared<MPIContext>(0, 1, MPI_COMM_NULL);
            }
            owned_tensor_factory_ = std::make_unique<TensorFactory>(*local_mpi_ctx);
            tensor_factory_ = owned_tensor_factory_.get();

        }

        // =====================================================================
        // Create arena if not already set up via initializeBuffers()
        // =====================================================================
        if (!arena_)
        {
            if (!initializeBuffers(activation_seq_len))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to initialize buffers via arena");
                return false;
            }
        }

        // =====================================================================
        // Pull activation buffers from arena (schema-driven allocation)
        // =====================================================================
        state_.hidden = arena_->getSharedTensor(BufferId::HIDDEN_STATE);
        state_.logits = arena_->getSharedTensor(BufferId::LOGITS);
        state_.logits_local = arena_->getSharedTensor(BufferId::LOGITS_LOCAL); // nullptr if not TP
        state_.normalized = arena_->getSharedTensor(BufferId::NORMALIZED);
        state_.residual = arena_->getSharedTensor(BufferId::RESIDUAL);
        state_.Q = arena_->getSharedTensor(BufferId::Q_PROJ);
        state_.K = arena_->getSharedTensor(BufferId::K_PROJ);
        state_.V = arena_->getSharedTensor(BufferId::V_PROJ);
        state_.attn_output = arena_->getSharedTensor(BufferId::ATTN_OUTPUT);
        state_.attn_proj = arena_->getSharedTensor(BufferId::ATTN_PROJ);
        state_.gate = arena_->getSharedTensor(BufferId::GATE_PROJ);
        state_.up = arena_->getSharedTensor(BufferId::UP_PROJ);
        state_.ffn_output = arena_->getSharedTensor(BufferId::FFN_OUTPUT);
        state_.workspace_scores = arena_->getSharedTensor(BufferId::ATTN_SCORES_WORKSPACE);
        state_.workspace_context = arena_->getSharedTensor(BufferId::ATTN_CONTEXT_WORKSPACE);
        state_.workspace_mask = arena_->getSharedTensor(BufferId::GEMM_WORKSPACE);

        // Conditional buffers (Hybrid/HybridQ16 mode only — nullptr if not in schema)
        state_.Q_rope = arena_->getSharedTensor(BufferId::Q_ROPE);
        state_.K_rope = arena_->getSharedTensor(BufferId::K_ROPE);
        state_.V_dequant = arena_->getSharedTensor(BufferId::V_DEQUANT);

        // Auto-discover extension buffers (model-specific BufferIds registered
        // by the schema, e.g. GDN, MoE). Any BufferId that doesn't map to a
        // named InferenceState field is stored in extension_buffers and flows
        // through toModelBuffers() → ActivationBuffers::extensions automatically.
        static const std::unordered_set<BufferId> core_ids = {
            BufferId::HIDDEN_STATE,
            BufferId::LOGITS,
            BufferId::LOGITS_LOCAL,
            BufferId::NORMALIZED,
            BufferId::RESIDUAL,
            BufferId::Q_PROJ,
            BufferId::K_PROJ,
            BufferId::V_PROJ,
            BufferId::ATTN_OUTPUT,
            BufferId::ATTN_PROJ,
            BufferId::GATE_PROJ,
            BufferId::UP_PROJ,
            BufferId::FFN_OUTPUT,
            BufferId::ATTN_SCORES_WORKSPACE,
            BufferId::ATTN_CONTEXT_WORKSPACE,
            BufferId::GEMM_WORKSPACE,
            BufferId::Q_ROPE,
            BufferId::K_ROPE,
            BufferId::V_DEQUANT,
        };
        state_.extension_buffers.clear();
        arena_->forEachRegistered([&](BufferId id)
                                  {
            if (core_ids.count(id) == 0)
            {
                auto tensor = arena_->getSharedTensor(id);
                if (tensor)
                {
                    state_.extension_buffers[id] = std::move(tensor);
                }
            } });

        // Validate required buffers
        if (!state_.hidden || !state_.logits || !state_.normalized ||
            !state_.Q || !state_.K || !state_.V ||
            !state_.attn_output || !state_.attn_proj ||
            !state_.gate || !state_.up || !state_.ffn_output)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Missing required buffers from arena. "
                      "Ensure Qwen2Schema provides all layer_buffers and model_buffers.");
            return false;
        }

        // =====================================================================
        // Non-arena state: K_head_scales (HybridQ16 only)
        // =====================================================================
        ActivationPrecision act_prec = config.activation_precision;
        if (act_prec == ActivationPrecision::HybridQ16)
        {
            int buffer_n_kv_heads = config.qkv_column_parallel ? config.local_n_kv_heads : config.n_kv_heads;
            const size_t k_head_scales_size = static_cast<size_t>(batch_size * activation_seq_len * buffer_n_kv_heads);
            state_.K_head_scales.resize(k_head_scales_size, 1.0f);
            LOG_DEBUG("[DeviceGraphOrchestrator] HybridQ16 K precision fix: allocated K_head_scales ("
                      << k_head_scales_size << " floats)");
        }

        // =====================================================================
        // Snapshot buffers (allocated directly, not in schema yet — Phase 2)
        // =====================================================================
#ifdef ENABLE_PIPELINE_SNAPSHOTS
        if (tensor_factory_)
        {
            int buffer_n_heads = config.qkv_column_parallel ? config.local_n_heads : config.n_heads;
            int head_dim = config.head_dim;
            int d_model = config.d_model;

            state_.context_snapshot = tensor_factory_->createFP32(
                {static_cast<size_t>(batch_size * activation_seq_len), static_cast<size_t>(buffer_n_heads * head_dim)},
                device);
            state_.attention_output_snapshot = tensor_factory_->createFP32(
                {static_cast<size_t>(batch_size * activation_seq_len), static_cast<size_t>(d_model)},
                device);
            state_.attention_residual_snapshot = tensor_factory_->createFP32(
                {static_cast<size_t>(batch_size * activation_seq_len), static_cast<size_t>(d_model)},
                device);
            LOG_DEBUG("[DeviceGraphOrchestrator] Allocated snapshot buffers from arena path");
        }
#endif

        // =====================================================================
        // KV cache creation (not arena-managed)
        // =====================================================================
        int n_layers = pp_stage_config_.has_value()
                           ? pp_stage_config_.value().layerCount()
                           : config.n_layers;

        std::shared_ptr<IMPIContext> local_mpi_ctx = mpi_ctx_;
        if (!local_mpi_ctx)
        {
            local_mpi_ctx = std::make_shared<MPIContext>(0, 1, MPI_COMM_NULL);
        }

        if (!initializeKVCaches(batch_size, max_seq_len, n_layers, device, local_mpi_ctx))
        {
            return false;
        }

        // =====================================================================
        // Initialize position tracking and config
        // =====================================================================
        state_.positions.assign(batch_size, 0);
        state_.sequence_lengths.assign(batch_size, 0);
        state_.batch_size = batch_size;
        state_.max_seq_len = max_seq_len;
        state_.activation_seq_len = activation_seq_len;
        state_.d_model = config.d_model;
        state_.vocab_size = config.vocab_size;
        state_.device_id = device;
        if (retainsMTPGraphCapacity(config.mtp) &&
            !initializeLiveCheckpointStoragePool())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to initialize transient MTP checkpoint storage");
            return false;
        }
        if (device.is_gpu())
        {
            /*
             * These buffers are persistent graph inputs, not overwrite-only
             * scratch. allocateDeviceStorage() establishes stable addresses but
             * intentionally does not invent initialized bytes or authority.
             * Publish every empty device-owned state bank now so a fresh runner
             * may enter a prefix-cache miss followed by MTP verification without
             * a synthetic request reset or first-use event allocation.
             */
            void *const initialization_stream =
                explicitGPUStreamForOperation(
                    "initialize_generation_persistent_state");
            // Admission and terminal response share these banks regardless of
            // the selected generation algorithm. Allocating them for ordinary
            // inference but initializing them only for MTP left capture inputs
            // without authority and deferred event creation until first use.
            if (!initializeAndPublishDeviceGenerationStorageOnStream(
                    initialization_stream,
                    "initial_inference_state"))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to initialize device-owned generation state");
                return false;
            }
            // Durable generated-token history belongs to the request, not
            // the speculative algorithm. Every retained request row needs
            // initialized, published bytes before its first captured sample.
            if (!zeroAndPublishGenerationSamplingInputsOnStream(
                    initialization_stream,
                    "initial_inference_state"))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to initialize device-owned sampling history");
                return false;
            }
            // Every retained participant owns these initial arena publications,
            // including follower verifier-input banks and their ready events.
            // Predictor-only graph ownership does not remove common setup roots.
            if (retainsMTPGraphCapacity(config.mtp) &&
                (!initializeAndPublishMTPTerminalHiddenArchiveOnStream(
                    initialization_stream,
                    "initial_inference_state") ||
                !initializeAndPublishMTPSampleBanksOnStream(
                    initialization_stream,
                    "initial_inference_state")))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to initialize device-owned MTP persistent state");
                return false;
            }
        }

        LOG_DEBUG("[DeviceGraphOrchestrator] Inference state initialized from arena: "
                  << "batch_size=" << batch_size
                  << " max_seq_len=" << max_seq_len
                  << " activation_seq_len=" << activation_seq_len
                  << " device=" << device.toString());
        return true;
    }

    bool DeviceGraphOrchestrator::initializeKVCaches(
        int batch_size, int max_seq_len, int n_layers,
        DeviceId device, const std::shared_ptr<IMPIContext> &local_mpi_ctx)
    {
        const auto &config = graph_builder_->config();
        if (!physical_memory_authority_ &&
            memory_authority_mode_ == MemoryAuthorityMode::ProductionRequired)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Production KV-cache construction has no rank-local physical-memory authority");
            return false;
        }
        const int n_kv_heads = config.n_kv_heads;
        const int head_dim = config.head_dim;

        // Resolve activation precision from config
        ActivationPrecision act_prec = config.activation_precision;

        // Resolve KV cache precision and layout
        ActivationPrecision kv_cache_prec = resolveBufferPrecision(
            act_prec, HybridBufferType::KV_Cache, nullptr);
        kv_cache_prec = resolveKVCacheStoragePrecision(config.kv_cache_precision, device.is_cpu());
        LOG_DEBUG("[DeviceGraphOrchestrator] KV cache precision: " << activationPrecisionToString(kv_cache_prec));
        LOG_DEBUG("[DeviceGraphOrchestrator] KV cache precision mode: "
                  << kvCachePrecisionToString(config.kv_cache_precision));

        // Determine KV cache layout mode:
        // - Q16_1 precision requires HEAD_MAJOR layout for Q16IntegerAttention kernel
        // - Other precisions use POSITION_MAJOR (legacy layout)
        KVCacheLayoutMode kv_layout_mode = (kv_cache_prec == ActivationPrecision::Q16_1)
                                               ? KVCacheLayoutMode::HEAD_MAJOR
                                               : KVCacheLayoutMode::POSITION_MAJOR;
        LOG_DEBUG("[DeviceGraphOrchestrator] KV cache layout mode: "
                  << (kv_layout_mode == KVCacheLayoutMode::HEAD_MAJOR ? "HEAD_MAJOR" : "POSITION_MAJOR"));

        // Set sharding parameters if needed (tensor parallelism)
        // Sharding is needed when local_n_kv_heads < n_kv_heads, AND tensor parallelism is active.
        // TP can be:
        // - GLOBAL TP: Multiple MPI ranks (mpi_ctx_->world_size() > 1)
        // - LOCAL TP: Multiple devices within single rank (tp_ctx->isLocal() && degree() > 1)
        // - NODE_LOCAL TP: Cross-rank same node (tp_ctx->isNodeLocal())
        // - GLOBAL TP: Cross-rank (tp_ctx->isGlobal()) or MPI world_size > 1
        const bool use_replicated_attention_state_cache =
            config.dense_tp_enabled && config.dense_tp_decode_replicated;
        bool use_sharded_cache =
            !use_replicated_attention_state_cache &&
            (config.local_n_kv_heads > 0 && config.local_n_kv_heads < n_kv_heads);
        bool has_tp = config.tp_ctx && config.tp_ctx->degree() > 1;
        bool is_global_tp = !has_tp && mpi_ctx_ && mpi_ctx_->world_size() > 1;
        if (use_replicated_attention_state_cache)
        {
            LOG_DEBUG("[DeviceGraphOrchestrator] Dense decode replication requires full per-device KV cache; "
                      "disabling TP KV sharding for this graph");
        }

        const auto query_main_heads = resolveQueryIntervalCacheHeads(config, device,
            use_replicated_attention_state_cache ? KVHeadConsumerDemand::FullModel
                                                : KVHeadConsumerDemand::AssignedQueryInterval);
        if (query_main_heads && n_kv_heads != config.n_kv_heads)
            throw std::invalid_argument("main cache original KV geometry differs from its query assignment");

        // One participant owns one main cache, including a PP layer shard.
        {
            llaminar::v2::kernels::KVCacheConfig kv_config;
            kv_config.precision = kv_cache_prec;
            kv_config.device = device;
            kv_config.num_layers = n_layers;
            kv_config.batch_size = batch_size;
            kv_config.max_seq_len = max_seq_len;
            kv_config.n_kv_heads = n_kv_heads;
            kv_config.head_dim = head_dim;
            kv_config.layout_mode = kv_layout_mode;
            kv_config.mpi_ctx = local_mpi_ctx.get();
            kv_config.physical_memory_authority =
                physical_memory_authority_;

            // PP layer offset for hybrid KV cache: ensures GDN kernel init
            // loop uses global layer indices (e.g., 12..23 for PP stage 2).
            if (pp_stage_config_.has_value())
            {
                kv_config.first_layer_index = pp_stage_config_.value().first_layer;
            }
            kv_config.turboquant_ctx = config.turboquant_ctx;

            // For hybrid models (e.g., Qwen 3.5 with GDN + FA layers),
            // configure the hybrid KV cache with layer type mapping and GDN sizing
            std::unique_ptr<HybridKVCacheConfig> hybrid_config_storage;
            if (config.hasGDN() && !config.layer_types.empty())
            {
                hybrid_config_storage = std::make_unique<HybridKVCacheConfig>();
                // Freeze the physical recurrence format before bank allocation.
                // Every graph, snapshot and prefix payload consumes that owner;
                // execution never rereads this diagnostic environment policy.
                hybrid_config_storage->recurrence_layout = gdnRecurrenceStateLayoutForDevice(
                    device, debugEnv().execution.cuda_gdn_state_layout,
                    debugEnv().execution.rocm_gdn_state_layout);
                hybrid_config_storage->first_layer_index = kv_config.first_layer_index;
                const int first_layer = std::max(0, kv_config.first_layer_index);
                const int layer_count = std::max(0, kv_config.num_layers);
                if (first_layer < static_cast<int>(config.layer_types.size()) &&
                    layer_count > 0 &&
                    first_layer + layer_count <= static_cast<int>(config.layer_types.size()))
                {
                    hybrid_config_storage->layer_types.assign(
                        config.layer_types.begin() + first_layer,
                        config.layer_types.begin() + first_layer + layer_count);
                }
                else
                {
                    hybrid_config_storage->first_layer_index = 0;
                    hybrid_config_storage->layer_types = config.layer_types;
                }
                hybrid_config_storage->gdn_conv_kernel_size = config.gdn.conv_kernel_size;
                hybrid_config_storage->gdn_state_size = config.gdn.state_size;
                hybrid_config_storage->gdn_inner_size = config.gdn.inner_size;
                hybrid_config_storage->gdn_group_count = config.gdn.group_count;
                hybrid_config_storage->gdn_time_step_rank = config.gdn.time_step_rank;
                hybrid_config_storage->n_heads = config.n_heads;
                hybrid_config_storage->local_n_heads = config.local_n_heads;
                /*
                 * GraphConfig owns the resolved dense head interval.  In an
                 * ExpertOverlay graph tp_device_idx can identify a sparse
                 * collective participant while dense weights and GDN state
                 * remain fully replicated.  Reconstructing an offset from
                 * that participant index therefore invents a second, false
                 * dense-sharding authority (participant 1 would begin after
                 * the end of the full head range).  Assignment builders set
                 * head_start for genuinely sharded graphs and full-dimension
                 * builders set it to zero, so consume that typed interval
                 * directly for both regimes.
                 */
                hybrid_config_storage->local_head_start = config.head_start;
                kv_config.hybrid_config = hybrid_config_storage.get();

                LOG_DEBUG("[DeviceGraphOrchestrator] Hybrid KV cache config: "
                          << "layers [" << hybrid_config_storage->first_layer_index
                          << ", " << (hybrid_config_storage->first_layer_index +
                                      static_cast<int>(hybrid_config_storage->layer_types.size()))
                          << "), "
                          << hybrid_config_storage->countKVLayers() << " KV layers, "
                          << (static_cast<int>(hybrid_config_storage->layer_types.size()) -
                              hybrid_config_storage->countKVLayers())
                          << " GDN layers");
            }
            kv_config.max_seq_len = max_seq_len;
            kv_config.n_kv_heads = n_kv_heads;
            kv_config.head_dim = head_dim;
            kv_config.layout_mode = kv_layout_mode;
            kv_config.mpi_ctx = local_mpi_ctx.get();
            kv_config.turboquant_ctx = config.turboquant_ctx;

            if (query_main_heads)
            {
                // Original query coverage, never participant*localCount, owns subset replica origins.
                kv_config.local_n_kv_heads = query_main_heads->count;
                kv_config.kv_head_start = query_main_heads->first;
            }
            else if (use_sharded_cache && (has_tp || is_global_tp))
            {
                kv_config.local_n_kv_heads = config.local_n_kv_heads;

                // Calculate kv_head_start based on TP mode:
                // - Any TP context: Use tp_device_idx (works for LOCAL, NODE_LOCAL, GLOBAL)
                // - Legacy GLOBAL TP (no tp_ctx): Use MPI rank
                if (has_tp)
                {
                    kv_config.kv_head_start = config.tp_device_idx * config.local_n_kv_heads;
                    LOG_DEBUG("[DeviceGraphOrchestrator] Creating sharded KV cache (TP scope="
                              << static_cast<int>(config.tp_ctx->scope()) << "): "
                              << n_kv_heads << " total KV heads, "
                              << config.local_n_kv_heads << " local KV heads (tp_idx="
                              << config.tp_device_idx << ", start=" << kv_config.kv_head_start << ")"
                              << " precision=" << activationPrecisionToString(kv_cache_prec));
                }
                else
                {
                    kv_config.kv_head_start = mpi_ctx_->rank() * config.local_n_kv_heads;
                    LOG_DEBUG("[DeviceGraphOrchestrator] Creating sharded KV cache (GLOBAL TP): "
                              << n_kv_heads << " total KV heads, "
                              << config.local_n_kv_heads << " local KV heads (rank="
                              << mpi_ctx_->rank() << ", start=" << kv_config.kv_head_start << ")"
                              << " precision=" << activationPrecisionToString(kv_cache_prec));
                }
            }

            // Create cache via factory (handles sharded vs non-sharded automatically)
            state_.kv_cache = llaminar::v2::kernels::KernelFactory::createKVCache(kv_config);
            if (!state_.kv_cache)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to create KV cache for device "
                          << device.to_string());
                return false;
            }
            state_.kv_cache->bindStateOwnership({
                .role = pp_stage_config_.has_value()
                            ? IKVCache::StateRole::PipelineShard
                            : IKVCache::StateRole::CommittedMain,
            });
            // Model state must exist before prefix-layout and rollback-capacity
            // admission inspect this cache. The graph builder binds only its
            // declared arena views; the cache remains the lifecycle authority.
            graph_builder_->initializeCacheAuxiliaryState(*state_.kv_cache,
                device.is_gpu() ? explicitGPUStreamForOperation("cache_auxiliary_setup_publication") : nullptr);
            logKVCacheBom("main", device, state_.kv_cache.get(), batch_size);
            if (query_main_heads && !use_replicated_attention_state_cache && physical_memory_authority_ &&
                PerfStatsCollector::isDomainEnabled("gpu_graph_inventory"))
                query_kv_geometry::recordCache(config, device, *state_.kv_cache,
                    *physical_memory_authority_);
        }

        // Pipeline followers still retain main-model verification and rollback.
        // Only a terminal participant owns the learned predictor and its shifted
        // cache. This is the same typed ownership projection used by the PMA BOM.
        const auto mtp_state_role = resolveMTPStateRole(
            retainsMTPGraphCapacity(config.mtp),
            !pp_stage_config_ || pp_stage_config_->has_lm_head);
        if (mtp_state_role == MTPStateRole::PredictorOwner)
        {
            if (!initializeMTPKVCaches(
                         batch_size,
                         max_seq_len,
                         kv_cache_prec,
                         kv_layout_mode,
                         device,
                         local_mpi_ctx,
                         use_sharded_cache,
                         has_tp,
                         is_global_tp))
            {
                return false;
            }
        }
        else
        {
            state_.mtp_kv_caches.clear();
        }

        return true;
    }

    bool DeviceGraphOrchestrator::forward(const int *tokens, int seq_len)
    {
        return forwardMainImpl(tokens, seq_len, 1).has_value();
    }

    const float *DeviceGraphOrchestrator::forward(const int *tokens, int seq_len, int batch_size)
    {
        return forwardMainImpl(tokens, seq_len, batch_size).value_or(nullptr);
    }

    std::optional<const float *> DeviceGraphOrchestrator::forwardMainImpl(
        const int *tokens,
        int seq_len,
        int batch_size)
    {
        if (defer_next_mtp_main_decode_sync_ && snapshot_context_.empty())
        {
            /*
             * Scalar MTP consumes its condition token through this ordinary
             * decode entry point before sidecar drafting.  Preserve that exact
             * graph's integration snapshots under a stable scope so grouped
             * request-batch condition rows can be compared with their true
             * serial equivalent after later verifier graphs overwrite the bare
             * aliases.
             */
            ScopedStringOverride snapshot_scope(
                snapshot_context_,
                "MTP_SCALAR_CONDITION");
            return forwardImpl(
                tokens,
                /*token_ids_device=*/nullptr,
                seq_len,
                batch_size,
                ForwardExecutionRole::MainInference);
        }
        return forwardImpl(
            tokens,
            /*token_ids_device=*/nullptr,
            seq_len,
            batch_size,
            ForwardExecutionRole::MainInference);
    }

    bool DeviceGraphOrchestrator::forwardPrefill(const int *tokens, int seq_len)
    {
        return forwardImpl(
                   tokens,
                   /*token_ids_device=*/nullptr,
                   seq_len,
                   /*batch_size=*/1,
                   ForwardExecutionRole::MainInference,
                   ForwardInvocationKind::ExplicitPrefill).has_value();
    }

    bool DeviceGraphOrchestrator::forwardRestoredPrefixMTPDecodeBridge(
        const RestoredPrefixMTPDecodeBridgeRequest &request)
    {
        if (!graph_builder_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Restored-prefix decode bridge has no initialized graph owner");
            return false;
        }
        if (!request.valid() || state_.positions.empty() ||
            state_.positions.front() != request.restored_prefix_tokens)
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Restored-prefix MTP decode bridge "
                "does not match the published request history"
                << " restored=" << request.restored_prefix_tokens
                << " position="
                << (state_.positions.empty() ? -1 : state_.positions.front()));
            return false;
        }
        const int token = request.token_id;
        // A typed outer prefix boundary does not authorize MTP execution. An
        // Off request reuses its admitted serial decode graph and canonical
        // logits roots, including when physical sidecar capacity is retained.
        // Enabled followers still need MTPCondition for their frozen receive
        // mailbox, while the enabled vocabulary owner also advances shifted KV.
        const auto runtime_role = resolveMTPRuntimeTransactionRole(
            graph_builder_->config().mtp.enabled,
            !pp_stage_config_ || pp_stage_config_->has_lm_head);
        if (runtime_role == MTPRuntimeTransactionRole::Disabled)
        {
            return forwardImpl(&token, /*token_ids_device=*/nullptr,
                /*seq_len=*/1, /*batch_size=*/1,
                ForwardExecutionRole::MainInference,
                ForwardInvocationKind::ExplicitDecode).has_value();
        }

        const ForwardExecutionRole role =
            state_.device_id.is_gpu()
                ? ForwardExecutionRole::MTPCondition
                : ForwardExecutionRole::MainInference;
        return forwardImpl(
                   &token,
                   /*token_ids_device=*/nullptr,
                   /*seq_len=*/1,
                   /*batch_size=*/1,
                   role,
                   ForwardInvocationKind::RestoredPrefixMTPDecodeBridge).has_value();
    }

    /**
     * @brief Accept the orchestration-owned generation for graph-native MoE wire packets.
     *
     * A local @ref session_epoch_ is intentionally not substituted here: a
     * root graph may be captured, invalidated, or reset on a different local
     * schedule than its remote expert graph. The orchestration layer is the
     * only authority that can publish one request identity to both ranks.
     */
    bool DeviceGraphOrchestrator::setMoEOverlayCollectiveRequestGeneration(
        uint64_t generation_id)
    {
        const auto plan = graph_builder_
                              ? graph_builder_->config().moe.routed_expert_plan
                              : nullptr;
        if (generation_id == 0 || !plan ||
            !plan->usesExpertOverlayAuthority())
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Refusing an invalid graph-native "
                "MoE overlay collective request generation");
            return false;
        }
        moe_overlay_collective_request_generation_ = generation_id;
        return true;
    }

    bool DeviceGraphOrchestrator::
        setMoEOverlayInferenceTransactionCoordinator(
            std::shared_ptr<MoEOverlayInferenceTransactionCoordinator>
                coordinator,
            int continuation_participant_index)
    {
        const auto *const moe = graph_builder_
                                    ? &graph_builder_->config().moe
                                    : nullptr;
        const auto plan = moe ? moe->routed_expert_plan : nullptr;
        const auto authority = moeOverlayAuthorityExecution();

        /*
         * This object coordinates submission of independently retained rank
         * graphs; it is not the residency-policy authority. A CPU-containing
         * overlay naturally uses it beside the host-resident authority. An
         * all-GPU heterogeneous overlay may also use it as the explicit graph
         * API boundary, but only after a mapped device fabric proves that
         * policy and durable epoch ownership remain on GPU. Static and Dynamic
         * share this submission-only coordinator: Dynamic policy, physical
         * movement, and runtime-bank publication are owned by the mapped
         * device-controller fabric and its retained background graphs.
         */
        const bool valid_submission_locus =
            authority == MoEOverlayAuthorityExecutionKind::HostResident ||
            (authority ==
                 MoEOverlayAuthorityExecutionKind::DeviceResident &&
             moe && moe->device_controller_fabric);
        if (!coordinator ||
            (state_.device_id.is_gpu() && declaredDeviceGenerationExecutionPolicy() !=
                DeviceGenerationExecutionPolicy::HostScheduledCapturedTransactions) ||
            continuation_participant_index < 0 ||
            continuation_participant_index >=
                coordinator->continuationParticipantCount() ||
            !plan || !plan->usesExpertOverlayAuthority() ||
            !valid_submission_locus)
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Refusing an invalid "
                "heterogeneous ExpertOverlay graph-submission binding");
            return false;
        }
        if (serving_graph_family_lifecycle_ !=
            ServingGraphFamilyLifecycle::Sealed)
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] ExpertOverlay transaction authority cannot be installed before the serving graph family is sealed");
            return false;
        }
        if (moe_overlay_inference_transaction_coordinator_ ||
            moe_overlay_inference_transaction_participant_index_ >= 0)
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] ExpertOverlay transaction "
                "authority is setup-only and cannot be replaced");
            return false;
        }

        const auto completion_boundary =
            coordinator->participantCompletionBoundary(
                continuation_participant_index);
        const bool valid_boundary =
            (completion_boundary ==
                 MoEOverlayInferenceCompletionBoundaryKind::DeviceEvent &&
             state_.device_id.is_gpu()) ||
            (completion_boundary ==
                 MoEOverlayInferenceCompletionBoundaryKind::HostSynchronous &&
             state_.device_id.is_cpu());
        if (!valid_boundary)
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] ExpertOverlay completion policy disagrees with participant device "
                << state_.device_id.toString()
                << " participant=" << continuation_participant_index);
            return false;
        }
        if (completion_boundary ==
            MoEOverlayInferenceCompletionBoundaryKind::DeviceEvent)
        {
            IBackend *const backend = getBackendFor(state_.device_id);
            try
            {
                moe_overlay_inference_completion_fence_ =
                    std::make_shared<
                        BackendMoEOverlayInferenceCompletionEvent>(
                        backend,
                        state_.device_id);
            }
            catch (const std::exception &exception)
            {
                LOG_ERROR(
                    "[DeviceGraphOrchestrator] Could not allocate the ExpertOverlay participant calibration event: "
                    << exception.what());
                return false;
            }
        }

        moe_overlay_inference_transaction_coordinator_ =
            std::move(coordinator);
        moe_overlay_inference_transaction_participant_index_ =
            continuation_participant_index;
        PerfStatsCollector::addCounter(
            "moe_overlay_transaction",
            "graph_submission_bindings",
            1.0,
            "graph_setup",
            state_.device_id.toString(),
            {{"authority_execution", toString(authority)},
             {"controller_fabric",
              moe && moe->device_controller_fabric ? "mapped_device" :
                                                      "host"},
             {"host_policy_mirror", "false"},
             {"participant_index",
              std::to_string(continuation_participant_index)}});
        return true;
    }

    bool DeviceGraphOrchestrator::forwardGroupedMTPVerifierWithHostTokenIds(
        const std::vector<std::vector<int>> &token_batches)
    {
        if (!state_.device_id.is_cpu())
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Host-token grouped MTP verification "
                "is CPU-only; GPU callers must publish a device-resident token row");
            return false;
        }
        return forwardHostTokenBatchImpl(
            token_batches,
            ForwardExecutionRole::GroupedMTPVerifier);
    }

    bool DeviceGraphOrchestrator::forwardGroupedMTPVerifierWithDeviceTokenIds(
        const int *token_shadow,
        const void *token_ids_device,
        int seq_len)
    {
        if (!token_shadow || !token_ids_device || seq_len <= 0)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] forwardGroupedMTPVerifierWithDeviceTokenIds "
                      "requires a host token shadow, a device token pointer, and a positive "
                      "sequence length");
            return false;
        }
        if (!state_.device_id.is_gpu())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] "
                      "forwardGroupedMTPVerifierWithDeviceTokenIds is only valid for GPU runners");
            return false;
        }

        std::string width_policy_error;
        const auto width_policy =
            mtpVerifierPhysicalWidthPolicyForRequest(
                /*request_count=*/1,
                &width_policy_error);
        const int physical_seq_len =
            width_policy
                ? mtpVerifierPhysicalPaddedSeqLen(
                      /*request_count=*/1,
                      seq_len,
                      mtp_max_verifier_rows_,
                      *width_policy)
                : 0;
        if (physical_seq_len <= 0)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Grouped MTP verifier logical row count has no valid physical graph policy"
                      << " logical_rows=" << seq_len
                      << " max_rows=" << mtp_max_verifier_rows_
                      << " policy_error=" << width_policy_error);
            return false;
        }

        /*
         * The host shadow contains only logical tokens and is never a GPU graph
         * input. The preparation graph writes the complete physical token bank,
         * including a zeroed inactive suffix, while this one-element span keeps
         * the exact logical width available for host bookkeeping. Every
         * stateful GPU stage consumes the corresponding device length scalar.
         */
        const std::array<int, 1> logical_request_lengths{seq_len};
        return forwardImpl(
                   token_shadow,
                   token_ids_device,
                   physical_seq_len,
                   /*batch_size=*/1,
                   ForwardExecutionRole::GroupedMTPVerifier,
                   ForwardInvocationKind::Automatic,
                   /*position_ids_device_override=*/nullptr,
                   /*sequence_lengths_device_override=*/nullptr,
                   logical_request_lengths).has_value();
    }

    bool DeviceGraphOrchestrator::
        advanceMTPMainConditionFromDeviceResidentLogicalState(
            int32_t token_shadow,
            const DeviceResidentLogicalSequenceStateHandle &logical_state,
            MTPConditionForwardPurpose purpose,
            int request_index)
    {
        auto fail = [&](const std::string &reason) -> bool
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Resident main-condition advance failed: "
                << reason);
            PerfStatsCollector::addCounter(
                "mtp",
                "resident_main_condition_advance_failures",
                1.0,
                "decode",
                state_.device_id.toString(),
                {{"reason", reason}});
            return false;
        };

        if (!state_.device_id.is_gpu())
            return fail("operation requires a GPU runner");
        if (request_index != 0)
        {
            return fail(
                "scalar main-condition advance only accepts request zero; "
                "request batches must use advanceMTPRequestBatchConditionOnDevice");
        }
        if (!logical_state.coversRequest(request_index) ||
            logical_state.device != state_.device_id ||
            !device_resident_logical_sequence_state_mailbox_.ownsHandle(
                logical_state,
                live_replay_state_epoch_))
        {
            return fail("logical-state mailbox is stale, foreign, or incomplete");
        }

        const int32_t *condition_token_device =
            logical_state.nextConditionTokenDeviceForRequest(request_index);
        const int32_t *condition_position_device =
            logical_state.targetPositionDeviceForRequest(request_index);
        const int32_t *condition_sequence_length_device =
            logical_state.targetSequenceLengthDeviceForRequest(request_index);
        if (!condition_token_device ||
            !condition_position_device ||
            !condition_sequence_length_device)
        {
            return fail(
                "logical-state mailbox does not expose a complete token/position/length row");
        }

        /*
         * Token, RoPE position, sparse-expert replica selection, and attention
         * length all consume one event-published transaction. forwardImpl() binds
         * these stable arena addresses into the MTP-condition graph, while
         * prepareLiveStateForForwardGraphExecution() queues the mailbox event on
         * the exact replay stream. The host scalar remains inert bookkeeping.
         */
        ScopedStringOverride snapshot_scope(
            snapshot_context_,
            snapshot_context_.empty()
                ? "MTP_SCALAR_CONDITION"
                : snapshot_context_);
        const bool ok =
            forwardImpl(
                &token_shadow,
                condition_token_device,
                /*seq_len=*/1,
                /*batch_size=*/1,
                ForwardExecutionRole::MTPCondition,
                mtpConditionForwardInvocation(purpose),
                condition_position_device,
                condition_sequence_length_device).has_value();
        if (!ok)
            return fail("captured main condition graph execution failed");

        PerfStatsCollector::addCounter(
            "mtp",
            "resident_main_condition_advances",
            1.0,
            "decode",
            state_.device_id.toString(),
            {{"request_index", std::to_string(request_index)},
             {"token_owner", "device_logical_state_mailbox"},
             {"position_owner", "device_logical_state_mailbox"}});
        return true;
    }

    bool DeviceGraphOrchestrator::
        advanceOrdinaryMainConditionFromDeviceResidentLogicalState(
            int32_t token_shadow,
            const DeviceResidentLogicalSequenceStateHandle &logical_state,
            int request_index)
    {
        auto fail = [&](const std::string &reason) -> bool
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Resident ordinary-condition advance failed: "
                << reason);
            PerfStatsCollector::addCounter(
                "generation",
                "resident_ordinary_condition_advance_failures",
                1.0,
                "decode",
                state_.device_id.toString(),
                {{"reason", reason}});
            return false;
        };

        if (!state_.device_id.is_gpu() || request_index != 0)
            return fail("operation requires one scalar GPU request");
        if (!logical_state.coversRequest(request_index) ||
            logical_state.device != state_.device_id ||
            !device_resident_logical_sequence_state_mailbox_.ownsHandle(
                logical_state, live_replay_state_epoch_))
        {
            return fail("logical-state mailbox is stale, foreign, or incomplete");
        }

        const int32_t *condition_token_device =
            logical_state.nextConditionTokenDeviceForRequest(request_index);
        if (!condition_token_device)
        {
            return fail("mailbox does not expose its request-zero token row");
        }

        /*
         * The mailbox owns the condition token, but it does not own scalar
         * decode geometry. Its target-length row describes the already-resident
         * sequence (for example 39), whereas KV append interprets a device
         * sequence-length binding as the number of rows to append. Passing that
         * row through an ordinary one-token decode consequently makes the graph
         * try to append the whole existing sequence instead of its single row.
         *
         * Leave position and length overrides absent so forwardImpl() selects
         * the canonical scalar device-decode binding: the captured root
         * snapshots the live KV count into REQUEST_POSITION_IDS and every KV
         * append stage uses its exact captured width of one. The token still
         * comes directly from the immutable mailbox, and the graph prelude
         * retains its event-backed reader edge; no host token, position, or
         * synchronization enters this path.
         */
        const bool ok =
            forwardImpl(
                &token_shadow,
                condition_token_device,
                /*seq_len=*/1,
                /*batch_size=*/1,
                ForwardExecutionRole::MainInference,
                ForwardInvocationKind::ExplicitDecode,
                /*position_ids_device_override=*/nullptr,
                /*sequence_lengths_device_override=*/nullptr).has_value();
        if (!ok)
            return fail("captured ordinary condition graph execution failed");

        PerfStatsCollector::addCounter(
            "generation",
            "resident_ordinary_condition_advances",
            1.0,
            "decode",
            state_.device_id.toString(),
            {{"request_index", std::to_string(request_index)},
             {"token_owner", "device_logical_state_mailbox"},
             {"position_owner", "canonical_device_kv_count"},
             {"append_rows", "captured_scalar_one"}});
        return true;
    }

    bool DeviceGraphOrchestrator::
        advanceMTPMainConditionFromDeviceTargetSample(
            int32_t token_shadow,
            int target_sample_slot,
            MTPConditionForwardPurpose purpose)
    {
        if (!state_.device_id.is_gpu() ||
            target_sample_slot < 0 ||
            target_sample_slot >= stochastic_target_row_capacity_)
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Device-target main-condition advance "
                "received an invalid runner or target slot");
            return false;
        }
        if (!state_.kv_cache ||
            !state_.kv_cache->supportsDeviceResidentSequenceStatePublication())
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Device-target main-condition advance "
                "requires a device-owned live KV position");
            return false;
        }

        const int32_t *live_position_device =
            state_.kv_cache->deviceSequenceCachedTokenCountPtr(/*seq_idx=*/0);
        if (!live_position_device)
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Device-target main-condition advance "
                "could not resolve the canonical live KV count");
            return false;
        }

        /*
         * Compose the target token and its pre-forward KV position into the same
         * durable mailbox before replay. Binding the live KV count directly would
         * be incorrect because attention advances that count before later MoE
         * routing stages read their logical-position input.
         */
        if (!publishDeviceResidentLogicalSequenceStateFromTargetSample(
                target_sample_slot,
                live_position_device))
        {
            return false;
        }
        const DeviceResidentLogicalSequenceStateHandle logical_state =
            deviceResidentLogicalSequenceState();
        if (!logical_state.valid())
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Device-target main-condition advance "
                "did not produce a resident logical-state mailbox");
            return false;
        }

        /*
         * Publishing the sampled token into the durable logical-state mailbox
         * is the last consumer of the preceding main-logits transaction.  The
         * next condition forward will publish a new main-logits producer, so
         * retire the old one before that replay rather than allowing a second
         * producer to overwrite an unconsumed one-shot handoff.  This mirrors
         * the request-batched condition path: neither branch synchronizes the
         * stream; the mailbox/event graph has already established the required
         * device ordering.
         */
        (void)consumePendingLogitsStream(
            PendingLogitsStreamRole::MainDecode,
            "advanceMTPMainConditionFromDeviceTargetSample");
        return advanceMTPMainConditionFromDeviceResidentLogicalState(
            token_shadow,
            logical_state,
            purpose,
            /*request_index=*/0);
    }

    bool DeviceGraphOrchestrator::forwardBatchWithDeviceTokenIds(
        const std::vector<std::vector<int>> &token_batches,
        const void *token_ids_device,
        int padded_seq_len)
    {
        ScopedDeviceLog device_log(state_.device_id);

        if (token_batches.empty())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] forwardBatchWithDeviceTokenIds called with empty batch");
            return false;
        }
        if (!token_ids_device || padded_seq_len <= 0)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] forwardBatchWithDeviceTokenIds requires a device token pointer "
                      "and a positive padded sequence length");
            return false;
        }
        if (!state_.device_id.is_gpu())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] forwardBatchWithDeviceTokenIds is only valid for GPU runners");
            return false;
        }

        const int batch_size = static_cast<int>(token_batches.size());
        if (batch_size > state_.batch_size)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-token batch size " << batch_size
                                                                           << " exceeds initialized batch size "
                                                                           << state_.batch_size);
            return false;
        }

        std::vector<int> actual_lengths(batch_size, 0);
        for (int request = 0; request < batch_size; ++request)
        {
            const int len = static_cast<int>(token_batches[request].size());
            if (len <= 0 || len > padded_seq_len)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Invalid device-token batch row length for request "
                          << request << ": len=" << len
                          << " padded_seq_len=" << padded_seq_len);
                return false;
            }
            actual_lengths[request] = len;
        }

        /*
         * Build a flat host shadow with the same padded row layout as the
         * device buffer. The graph does not read this memory when
         * token_ids_device is set; it exists for diagnostics, cache metadata,
         * and CPU-side row-coordinate bookkeeping.
         */
        std::vector<int> flat_shadow(
            static_cast<size_t>(batch_size) * static_cast<size_t>(padded_seq_len),
            0);
        for (int request = 0; request < batch_size; ++request)
        {
            const auto &tokens = token_batches[request];
            for (int col = 0; col < actual_lengths[request]; ++col)
            {
                flat_shadow[static_cast<size_t>(request) *
                                static_cast<size_t>(padded_seq_len) +
                            static_cast<size_t>(col)] = tokens[static_cast<size_t>(col)];
            }
        }

        padded_seq_len_ = padded_seq_len;
        const std::vector<int> old_positions = state_.positions;
        const std::vector<int> old_sequence_lengths = state_.sequence_lengths;

        state_.sequence_lengths.resize(static_cast<size_t>(batch_size));
        for (int request = 0; request < batch_size; ++request)
        {
            state_.sequence_lengths[static_cast<size_t>(request)] =
                actual_lengths[request];
        }

        const bool compact_prefill_logits =
            state_.device_id.is_gpu() &&
            batch_size > 1 &&
            !compute_all_position_logits_;
        if (compact_prefill_logits &&
            graph_builder_ &&
            graph_builder_->config().mtp.enabled)
        {
            /*
             * The first sampled token immediately becomes a resident mailbox
             * condition row. Size the persistent metadata consumer before the
             * compact-prefill graph allocates its workspace; resizing after replay
             * would invalidate the logits and arena pointers that sampling owns.
             */
            MTPSpecDecodeMetadataShape metadata_shape =
                mtp_spec_decode_metadata_binding_.capacity();
            metadata_shape.max_requests =
                std::max(metadata_shape.max_requests, batch_size);
            metadata_shape.max_draft_tokens =
                std::max(
                    metadata_shape.max_draft_tokens,
                    std::max(1, mtp_max_draft_depth_));
            mtp_spec_decode_metadata_binding_.ensureCapacity(metadata_shape);
        }
        auto restore_prefill_logits_mode = [&]()
        {
            if (!compact_prefill_logits)
                return;
            request_batched_prefill_logits_row_count_ = 0;
            if (graph_builder_)
                graph_builder_->setRowIndexedAllPositionLogitRows({});
            setComputeRowIndexedAllPositionLogits(false, 0);
            setComputeAllPositionLogits(false);
        };
        if (compact_prefill_logits)
        {
            if (!setComputeAllPositionLogits(true) ||
                !setComputeRowIndexedAllPositionLogits(true, batch_size) ||
                !graph_builder_)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to configure "
                          "device-token request-batched terminal logits row "
                          "selection");
                restore_prefill_logits_mode();
                state_.sequence_lengths = old_sequence_lengths;
                return false;
            }
            request_batched_prefill_logits_row_count_ = batch_size;
        }

        /*
         * This API is the grouped MTP verifier entry point. Its row lengths are
         * verifier geometry, not request positions, and must never overwrite
         * the prefill-admitted persistent request-length rows or introduce a
         * second H2D fence. RoPE, recurrent stages, and publication consume the
         * event-fenced device logical-state handle installed before this call.
         */

        const auto result = forwardImpl(
            flat_shadow.data(),
            token_ids_device,
            padded_seq_len,
            batch_size,
            ForwardExecutionRole::GroupedMTPVerifier,
            ForwardInvocationKind::Automatic,
            /*position_ids_device_override=*/nullptr,
            /*sequence_lengths_device_override=*/nullptr,
            actual_lengths);
        if (compact_prefill_logits)
        {
            if (graph_builder_)
                graph_builder_->setRowIndexedAllPositionLogitRows({});
            setComputeRowIndexedAllPositionLogits(false, 0);
            setComputeAllPositionLogits(false);
            if (!result)
                request_batched_prefill_logits_row_count_ = 0;
        }

        if (!result)
        {
            state_.positions = old_positions;
            state_.sequence_lengths = old_sequence_lengths;
            return false;
        }

        state_.positions = old_positions;
        state_.sequence_lengths = old_sequence_lengths;
        state_.positions.resize(std::max(
            state_.positions.size(),
            static_cast<size_t>(batch_size)),
            0);
        state_.sequence_lengths.resize(std::max(
            state_.sequence_lengths.size(),
            static_cast<size_t>(batch_size)),
            0);
        for (int request = 0; request < batch_size; ++request)
        {
            const int old_pos = state_.positions[static_cast<size_t>(request)];
            const int old_len =
                request < static_cast<int>(old_sequence_lengths.size())
                    ? old_sequence_lengths[static_cast<size_t>(request)]
                    : old_pos;
            state_.positions[static_cast<size_t>(request)] =
                old_pos + actual_lengths[request];
            state_.sequence_lengths[static_cast<size_t>(request)] =
                old_len + actual_lengths[request];
        }

        return true;
    }

    const void *DeviceGraphOrchestrator::prepareMTPVerifierInputTokensOnDevice(
        int32_t first_token,
        int first_draft_slot,
        int draft_token_count,
        int total_verifier_input_tokens)
    {
        if (!state_.device_id.is_gpu() ||
            !mtp_verifier_input_tokens_dev_ ||
            !stochastic_target_sample_tokens_dev_ ||
            !stochastic_draft_sample_tokens_dev_)
        {
            return nullptr;
        }
        if (total_verifier_input_tokens <= 0 ||
            total_verifier_input_tokens > mtp_max_verifier_rows_ ||
            draft_token_count < 0 ||
            draft_token_count > mtp_max_draft_depth_ ||
            draft_token_count + 1 != total_verifier_input_tokens ||
            first_draft_slot < 0 ||
            first_draft_slot + draft_token_count >
                stochastic_draft_row_capacity_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Invalid MTP verifier device-token plan: total="
                      << total_verifier_input_tokens
                      << " drafts=" << draft_token_count
                      << " first_draft_slot=" << first_draft_slot);
            return nullptr;
        }

        /*
         * Do not copy immediately. The forward engine chooses the graph replay
         * stream later, so eager copies could race captured replay on another
         * stream.  The metadata hook will materialize this plan on the exact
         * stream that consumes the token row.
         */
        pending_mtp_verifier_device_token_plan_ =
            PendingMTPVerifierDeviceTokenPlan{
                first_token,
                -1,
                false,
                false,
                first_draft_slot,
                draft_token_count,
                total_verifier_input_tokens};
        pending_mtp_verifier_device_token_batch_plan_.reset();
        materialized_mtp_verifier_device_token_batch_ = {};
        return mtp_verifier_input_tokens_dev_;
    }

    const void *DeviceGraphOrchestrator::prepareMTPVerifierInputTokenBatchOnDevice(
        const DeviceMTPVerifierInputBatchRequest *requests,
        int request_count,
        int logical_padded_seq_len)
    {
        if (!state_.device_id.is_gpu() ||
            !mtp_verifier_input_tokens_dev_ ||
            !stochastic_target_sample_tokens_dev_ ||
            !stochastic_draft_sample_tokens_dev_)
        {
            return nullptr;
        }
        if (!requests ||
            request_count <= 0 ||
            request_count > stochastic_batch_output_request_capacity_ ||
            logical_padded_seq_len <= 0 ||
            logical_padded_seq_len > mtp_max_verifier_rows_ ||
            request_count * logical_padded_seq_len >
                stochastic_target_row_capacity_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Invalid MTP verifier input token batch: requests="
                      << request_count
                      << " logical_padded_seq_len="
                      << logical_padded_seq_len
                      << " capacity=" << stochastic_target_row_capacity_);
            return nullptr;
        }

        PendingMTPVerifierDeviceTokenBatchPlan plan;
        plan.request_count = request_count;
        plan.logical_padded_seq_len = logical_padded_seq_len;
        plan.requests.resize(static_cast<size_t>(request_count));
        for (int i = 0; i < request_count; ++i)
        {
            const DeviceMTPVerifierInputBatchRequest &row = requests[i];
            if (!row.first_token_from_device ||
                row.total_verifier_input_tokens <= 0 ||
                row.total_verifier_input_tokens >
                    logical_padded_seq_len ||
                row.draft_token_count < 0 ||
                row.draft_token_count + 1 != row.total_verifier_input_tokens ||
                row.first_target_sample_slot < 0 ||
                row.first_target_sample_slot >=
                    stochastic_target_row_capacity_ ||
                row.first_draft_slot < 0 ||
                row.first_draft_slot + row.draft_token_count >
                    stochastic_draft_row_capacity_)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Invalid or host-owned MTP verifier input token batch row "
                          << i
                          << "; GPU grouped verifier rows require an explicit device first-token source");
                return nullptr;
            }
            plan.requests[static_cast<size_t>(i)] = row;
        }

        pending_mtp_verifier_device_token_plan_.reset();
        pending_mtp_verifier_device_token_batch_plan_ = plan;
        materialized_mtp_verifier_device_token_row_ = {};
        materialized_mtp_verifier_device_token_batch_ = {};
        return mtp_verifier_input_tokens_dev_;
    }

    const void *DeviceGraphOrchestrator::prepareMTPVerifierInputTokensOnDeviceFromHostRow(
        const int32_t *verifier_tokens,
        int total_verifier_input_tokens,
        int draft_token_count)
    {
        if (!state_.device_id.is_gpu() || !mtp_verifier_input_tokens_dev_)
            return nullptr;
        if (!verifier_tokens ||
            total_verifier_input_tokens <= 0 ||
            total_verifier_input_tokens > mtp_max_verifier_rows_ ||
            draft_token_count < 0 ||
            draft_token_count > mtp_max_draft_depth_ ||
            draft_token_count + 1 != total_verifier_input_tokens)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Invalid host MTP verifier device-token row: total="
                      << total_verifier_input_tokens
                      << " drafts=" << draft_token_count);
            return nullptr;
        }

        PendingMTPVerifierDeviceTokenPlan plan;
        plan.all_tokens_from_host = true;
        plan.draft_token_count = draft_token_count;
        plan.total_verifier_input_tokens = total_verifier_input_tokens;
        plan.host_tokens.assign(
            verifier_tokens,
            verifier_tokens + total_verifier_input_tokens);
        pending_mtp_verifier_device_token_plan_ = plan;
        pending_mtp_verifier_device_token_batch_plan_.reset();
        materialized_mtp_verifier_device_token_batch_ = {};
        return mtp_verifier_input_tokens_dev_;
    }

    const void *DeviceGraphOrchestrator::prepareMTPVerifierInputTokensOnDeviceFromDeviceFirstToken(
        int first_target_sample_slot,
        int first_draft_slot,
        int draft_token_count,
        int total_verifier_input_tokens)
    {
        if (!state_.device_id.is_gpu() ||
            !mtp_verifier_input_tokens_dev_ ||
            !stochastic_target_sample_tokens_dev_ ||
            !stochastic_draft_sample_tokens_dev_)
        {
            return nullptr;
        }
        if (total_verifier_input_tokens <= 0 ||
            total_verifier_input_tokens > mtp_max_verifier_rows_ ||
            draft_token_count < 0 ||
            draft_token_count > mtp_max_draft_depth_ ||
            draft_token_count + 1 != total_verifier_input_tokens ||
            first_target_sample_slot < 0 ||
            first_target_sample_slot >= stochastic_target_row_capacity_ ||
            first_draft_slot < 0 ||
            first_draft_slot + draft_token_count >
                stochastic_draft_row_capacity_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Invalid MTP verifier device-first-token plan: total="
                      << total_verifier_input_tokens
                      << " drafts=" << draft_token_count
                      << " first_target_slot=" << first_target_sample_slot
                      << " first_draft_slot=" << first_draft_slot);
            return nullptr;
        }
        const auto &first_token_ready =
            stochastic_target_sample_ready_[static_cast<size_t>(first_target_sample_slot)];
        if (!first_token_ready.valid)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] MTP verifier device-first-token plan requires a ready target sample slot="
                      << first_target_sample_slot);
            return nullptr;
        }

        /*
         * The sampled target token is still on the sampler stream.  Store a
         * logical copy plan here and let materializePendingMTPVerifierInputTokensOnDevice()
         * wait/copy on the verifier graph stream once that stream is known.
         */
        pending_mtp_verifier_device_token_plan_ =
            PendingMTPVerifierDeviceTokenPlan{
                -1,
                first_target_sample_slot,
                true,
                false,
                first_draft_slot,
                draft_token_count,
                total_verifier_input_tokens};
        pending_mtp_verifier_device_token_batch_plan_.reset();
        materialized_mtp_verifier_device_token_batch_ = {};
        return mtp_verifier_input_tokens_dev_;
    }
} // namespace llaminar2
