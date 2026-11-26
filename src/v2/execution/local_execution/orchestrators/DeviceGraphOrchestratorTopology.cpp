/**
 * @file DeviceGraphOrchestratorTopology.cpp
 * @brief DeviceGraphOrchestrator method shard (9169..13851).
 */
#include "DeviceGraphOrchestratorInternal.h"

namespace llaminar2
{

    bool DeviceGraphOrchestrator::observeDeviceMoERebalanceDispatchTicket(
        DeviceMoERebalanceDispatchTicket *out_ticket)
    {
        if (out_ticket)
            *out_ticket = DeviceMoERebalanceDispatchTicket{};
        if (!out_ticket ||
            deviceMoERebalanceMaintenanceExecutionPolicy() !=
                DeviceMoERebalanceMaintenanceExecutionPolicy::
                    HostScheduledCapturedMaintenance)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] HIP Device MoE ticket observation has no exact hosted policy");
            return false;
        }

        auto &cache = device_moe_rebalance_maintenance_graph_;
        IBackend *const backend = getBackendFor(state_.device_id);
        void *const maintenance_stream =
            cache.segment_cache.capture_stream;
        if (!backend || !maintenance_stream ||
            !cache.cadence_transaction_capability_checked ||
            cache.cadence_transaction_capture ||
            !cache.cadence_boundary_publication_capture ||
            !cache.cadence_boundary_publication_capture->hasExecutable() ||
            !cache.cadence_boundary_acknowledgement_capture ||
            !cache.cadence_boundary_acknowledgement_capture->hasExecutable() ||
            !cache.hosted_decode_commit_boundary_embedded ||
            !cache.hosted_observation_schedule.valid() ||
            !device_moe_rebalance_dispatch_ticket_host_scratch_ ||
            !device_moe_rebalance_dispatch_ticket_host_scratch_->valid() ||
            !device_moe_rebalance_dispatch_ticket_ready_event_ ||
            device_moe_rebalance_dispatch_ticket_copy_pending_ ||
            last_device_moe_rebalance_dispatch_ticket_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] HIP Device MoE ticket observation has incomplete or overlapping ownership"
                      << " device=" << state_.device_id.toString()
                      << " capability_checked="
                      << cache.cadence_transaction_capability_checked
                      << " copy_pending="
                      << device_moe_rebalance_dispatch_ticket_copy_pending_
                      << " unsubmitted_ticket="
                      << last_device_moe_rebalance_dispatch_ticket_.has_value());
            return false;
        }

        std::string controller_lookup_error;
        DeviceMoERebalanceGraphControllerState *const controller =
            deviceMoERebalanceControllerStateDevice(&controller_lookup_error);
        if (!controller)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] HIP Device MoE ticket observation has no controller: "
                      << controller_lookup_error);
            return false;
        }

        const MoEDeviceRebalanceStage *transaction_owner = nullptr;
        for (const auto &node_name : cache.graph->getExecutionOrder())
        {
            const ComputeNode *node = cache.graph->getNode(node_name);
            const auto *stage =
                node && node->stage
                    ? dynamic_cast<const MoEDeviceRebalanceStage *>(
                          node->stage.get())
                    : nullptr;
            if (stage && stage->ownsRequestTransactionState())
            {
                if (transaction_owner)
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] HIP Device MoE ticket observation found multiple transaction owners");
                    return false;
                }
                transaction_owner = stage;
            }
        }
        if (!transaction_owner)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] HIP Device MoE ticket observation found no transaction owner");
            return false;
        }

        PerfStatsCollector::addCounter(
            "moe_rebalance",
            "device_maintenance_participant_boundary_calls",
            1.0,
            "decode",
            state_.device_id.toString(),
            {{"scheduler", "authenticated_dispatch_ticket"},
             {"ticket_observation", "performed"}});

        /* This is the sole host-visible boundary in HIP scheduling. Join every
         * committed producer before the captured publisher samples cadence.
         * Do not release the overlay reader speculatively: a non-due ticket has
         * no placement writer. The authenticated due submission queues that
         * release on this same stream immediately before maintenance. */
        if (!waitForLiveInferenceStateReadyForObservation(
                maintenance_stream,
                "hosted_device_moe_rebalance_ticket_publisher",
                DeviceTimelineRole::MoERebalanceMaintenance) ||
            !cache.cadence_boundary_publication_capture->launchOnStream(
                maintenance_stream))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] HIP Device MoE ticket publisher could not close and publish the committed boundary on "
                      << state_.device_id.toString());
            return false;
        }

        const size_t ticket_bytes =
            sizeof(DeviceMoERebalanceDispatchTicket);
        auto *const dispatch_ticket_device =
            reinterpret_cast<DeviceMoERebalanceDispatchTicket *>(
                reinterpret_cast<std::byte *>(controller) +
                offsetof(
                    DeviceMoERebalanceGraphControllerState,
                    dispatch_ticket));
        if (!backend->deviceToHostOnStream(
                device_moe_rebalance_dispatch_ticket_host_scratch_->ticket,
                dispatch_ticket_device,
                ticket_bytes,
                state_.device_id.gpu_ordinal(),
                maintenance_stream) ||
            !backend->recordEvent(
                device_moe_rebalance_dispatch_ticket_ready_event_.get(),
                state_.device_id.gpu_ordinal(),
                maintenance_stream))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] HIP Device MoE ticket D2H publication failed on "
                      << state_.device_id.toString());
            return false;
        }
        device_moe_rebalance_dispatch_ticket_copy_pending_ = true;
        PerfStatsCollector::addCounter(
            "moe_rebalance",
            "device_moe_rebalance_dispatch_ticket_d2h_submissions",
            1.0,
            "decode",
            state_.device_id.toString(),
            {{"bytes", std::to_string(ticket_bytes)},
             {"authority", "immutable_scheduler_snapshot"},
             {"state_payload", "false"}});

        if (!backend->waitForEvent(
                device_moe_rebalance_dispatch_ticket_ready_event_.get(),
                state_.device_id.gpu_ordinal()))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] HIP Device MoE ticket-ready event did not complete on "
                      << state_.device_id.toString());
            return false;
        }
        device_moe_rebalance_dispatch_ticket_copy_pending_ = false;

        const auto ticket =
            *device_moe_rebalance_dispatch_ticket_host_scratch_->ticket;
        const auto &config = transaction_owner->getParams().config;
        const uint64_t generation = workspaceGeneration(state_.device_id);
        /* An outstanding boundary must advance the clock beyond the last
         * consumed decision.  An acknowledged idle snapshot may legitimately
         * carry the same count: the immutable HIP bound can observe before a
         * serial state row exists (most visibly for the token sampled directly
         * from prefill logits).  The freshly replayed publisher, event edge,
         * request epoch, and workspace generation authenticate that snapshot;
         * requiring fake arithmetic progress here would corrupt the clock. */
        const bool cadence_progress_valid =
            ticket.decode_boundary_advanced != 0u
                ? ticket.decode_rounds_committed >
                      hosted_device_moe_rebalance_last_committed_round_
                : ticket.decode_rounds_committed >=
                      hosted_device_moe_rebalance_last_committed_round_;
        const bool valid_ticket =
            ticket.matchesLifecycle(
                session_epoch_,
                generation,
                config.participant_id,
                config.participant_count) &&
            ticket.healthy == 1u && ticket.error_code == 0u &&
            cadence_progress_valid;
        if (!valid_ticket)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] HIP Device MoE dispatch ticket failed authentication"
                      << " device=" << state_.device_id.toString()
                      << " abi=" << ticket.abi_version
                      << " session=" << ticket.sessionEpoch()
                      << " expected_session=" << session_epoch_
                      << " workspace=" << ticket.workspaceGeneration()
                      << " expected_workspace=" << generation
                      << " participant=" << ticket.participant_id
                      << " expected_participant=" << config.participant_id
                      << " committed=" << ticket.decode_rounds_committed
                      << " previous_committed="
                      << hosted_device_moe_rebalance_last_committed_round_
                      << " remaining="
                      << ticket.decode_rounds_until_maintenance
                      << " due=" << ticket.maintenance_due
                      << " advanced=" << ticket.decode_boundary_advanced
                      << " healthy=" << ticket.healthy
                      << " error=" << ticket.error_code);
            return false;
        }

        last_device_moe_rebalance_dispatch_ticket_ = ticket;
        *out_ticket = ticket;
        PerfStatsCollector::addCounter(
            "moe_rebalance",
            "device_moe_rebalance_dispatch_tickets_observed",
            1.0,
            "decode",
            state_.device_id.toString(),
            {{"committed_rounds",
              std::to_string(ticket.decode_rounds_committed)},
             {"maintenance_due",
              ticket.maintenance_due ? "true" : "false"},
             {"participant", std::to_string(ticket.participant_id)}});
        return true;
    }

    bool DeviceGraphOrchestrator::
        submitHostScheduledDeviceMoERebalanceMaintenance(
            const DeviceMoERebalanceDispatchTicket &ticket)
    {
        auto &cache = device_moe_rebalance_maintenance_graph_;
        if (deviceMoERebalanceMaintenanceExecutionPolicy() !=
                DeviceMoERebalanceMaintenanceExecutionPolicy::
                    HostScheduledCapturedMaintenance ||
            !last_device_moe_rebalance_dispatch_ticket_ ||
            !ticket.hasSameDispatchDecision(
                *last_device_moe_rebalance_dispatch_ticket_) ||
            ticket.participant_id !=
                last_device_moe_rebalance_dispatch_ticket_->participant_id ||
            device_moe_rebalance_dispatch_ticket_copy_pending_ ||
            !cache.cadence_boundary_acknowledgement_capture ||
            !cache.cadence_boundary_acknowledgement_capture->hasExecutable())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] HIP Device MoE maintenance submission rejected a stale, altered, or overlapping ticket");
            return false;
        }

        bool submitted = false;
        const DeviceMoERebalanceDispatchAction dispatch_action =
            ticket.dispatchAction();
        if (dispatch_action ==
            DeviceMoERebalanceDispatchAction::Maintain)
        {
            void *const stream = cache.segment_cache.capture_stream;
            if (!stream ||
                (moe_overlay_epoch_execution_binding_ &&
                 !claimMoEOverlayEpochForPlacementMaintenance(
                     stream,
                     "hosted_device_moe_rebalance_due_ticket")))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] HIP due MoE ticket could not claim its ExpertOverlay inference boundary on "
                          << state_.device_id.toString());
                return false;
            }
            submitted = maybeRunDeviceMoERebalanceMaintenanceGraph(
                MoEOverlayEpochMaintenanceBoundarySource::
                    AuthenticatedDispatchTicket);
        }
        else if (dispatch_action ==
                 DeviceMoERebalanceDispatchAction::Acknowledge)
        {
            IBackend *const backend = getBackendFor(state_.device_id);
            void *const stream = cache.segment_cache.capture_stream;
            if (!backend || !stream || !cache.completion_event ||
                !cache.cadence_boundary_acknowledgement_capture
                     ->launchOnStream(stream) ||
                !backend->recordEvent(
                    cache.completion_event.get(),
                    state_.device_id.gpu_ordinal(),
                    stream))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] HIP Device MoE non-due acknowledgement submission failed on "
                          << state_.device_id.toString());
                return false;
            }
            ++cache.launch_count;
            cache.completion_event_in_flight = true;
            submitted = true;
            PerfStatsCollector::addCounter(
                "moe_rebalance",
                "device_maintenance_scheduler_transactions",
                1.0,
                "decode",
                state_.device_id.toString(),
                {{"backend", "ROCm"},
                 {"scheduler", "authenticated_dispatch_ticket"},
                 {"maintenance_due", "false"},
                 {"host_state_upload", "false"}});
        }
        else if (dispatch_action ==
                 DeviceMoERebalanceDispatchAction::None)
        {
            /* The serial decode graph already acknowledged this non-due edge.
             * Replaying the acknowledgement graph would be an unnecessary HIP
             * launch and completion event.  The authenticated ticket is enough
             * to arm the next conservative observation interval. */
            submitted = true;
            PerfStatsCollector::addCounter(
                "moe_rebalance",
                "device_maintenance_scheduler_transactions",
                1.0,
                "decode",
                state_.device_id.toString(),
                {{"backend", "ROCm"},
                 {"scheduler", "authenticated_dispatch_ticket"},
                 {"maintenance_due", "false"},
                 {"boundary_state", "acknowledged_idle"},
                 {"standalone_graph_launch", "false"},
                 {"host_state_upload", "false"}});
            PerfStatsCollector::addCounter(
                "moe_rebalance",
                "device_moe_rebalance_idle_ticket_acknowledgements_elided",
                1.0,
                "decode",
                state_.device_id.toString());
        }
        else
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Authenticated HIP Device MoE ticket decoded to no legal dispatch action on "
                      << state_.device_id.toString());
            return false;
        }
        if (!submitted)
            return false;

        hosted_device_moe_rebalance_last_committed_round_ =
            ticket.decode_rounds_committed;
        last_device_moe_rebalance_dispatch_ticket_.reset();
        if (ticket.maintenance_due != 0u)
        {
            PerfStatsCollector::addCounter(
                "moe_rebalance",
                "device_maintenance_host_ticket_due_submissions",
                1.0,
                "decode",
                state_.device_id.toString(),
                {{"committed_rounds",
                  std::to_string(ticket.decode_rounds_committed)},
                 {"scheduler", "authenticated_dispatch_ticket"}});
        }
        return true;
    }

    bool DeviceGraphOrchestrator::exportCompletedDeviceMoERebalanceMaintenanceStats(
        DeviceMoERebalanceMaintenanceGraphCache &cache,
        void *maintenance_stream,
        const std::string &device_key,
        const std::map<std::string, std::string> &maintenance_tags,
        DeviceMoERebalanceMaintenanceOutcome *outcome)
    {
        const std::string trace_path = deviceMoERebalanceTracePathFromEnv();
        const bool export_trace = !trace_path.empty();
        const bool export_native_journal = nativeMoEMovementOwnerStage() != nullptr;
        if (!export_native_journal && !PerfStatsCollector::isDomainEnabled("moe_rebalance") &&
            !outcome && !export_trace)
            return true;

        if (!cache.graph || !workspace_allocator_)
            return true;

        if (!maintenance_stream)
        {
            LOG_ERROR("[DGO] Device MoE rebalance diagnostic export requires an explicit non-null maintenance stream");
            return false;
        }

        const MoEDeviceRebalanceStage *rebalance_stage = nullptr;
        DeviceMoERebalanceStatusPublicationContract publication_contract;
        for (const auto &node_name : cache.graph->getExecutionOrder())
        {
            const ComputeNode *node = cache.graph->getNode(node_name);
            if (!node || !node->stage)
                continue;

            auto *candidate =
                dynamic_cast<const MoEDeviceRebalanceStage *>(node->stage.get());
            if (!candidate)
                continue;

            publication_contract.merge(
                candidate->statusPublicationContract());
            const auto phase = candidate->getParams().phase;
            if (phase == DeviceMoERebalanceStagePhase::PlanCopyApply)
            {
                rebalance_stage = candidate;
                break;
            }
        }

        if (!rebalance_stage)
        {
            LOG_ERROR("[DGO] Device MoE rebalance diagnostic export could not find maintenance stage");
            return false;
        }

        DeviceWorkspaceManager *workspace =
            workspace_allocator_->getDeviceWorkspace(state_.device_id);
        if (!workspace)
        {
            LOG_ERROR("[DGO] Device MoE rebalance diagnostic export missing device workspace for "
                      << device_key);
            return false;
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
        {
            LOG_ERROR("[DGO] Device MoE rebalance diagnostic export could not resolve backend for "
                      << device_key);
            return false;
        }
        const int device_ordinal = state_.device_id.gpu_ordinal();

        const auto &params = rebalance_stage->getParams();
        const std::string &workspace_name = params.workspace_name.empty()
                                                ? params.stage_name
                                                : params.workspace_name;
        const uint32_t command_buffer_count =
            params.local_transfer_slot_count > 0 ? 2u : 1u;
        const uint32_t trace_plan_capacity =
            export_trace
                ? static_cast<uint32_t>(std::min<uint64_t>(
                      deviceMoERebalanceCommandPlanCapacity(params.config, params.transfer_mode),
                      static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())))
                : 0u;

        auto buffer_name = [&](const char *base)
        {
            return MoEDeviceRebalanceStage::workspaceBufferName(base, workspace_name);
        };

        auto copy_required =
            [&](void *dst, const std::string &name, size_t bytes) -> bool
        {
            void *src = workspace->getBuffer(name);
            if (!src)
            {
                LOG_ERROR("[DGO] Device MoE rebalance diagnostic export missing workspace buffer "
                          << name << " on " << device_key);
                return false;
            }
            if (!backend->deviceToHostOnStream(
                    dst,
                    src,
                    bytes,
                    device_ordinal,
                    maintenance_stream))
            {
                LOG_ERROR("[DGO] Device MoE rebalance diagnostic export failed to enqueue D2H for "
                          << name << " on " << device_key);
                return false;
            }
            return true;
        };

        DeviceMoERebalanceStatus status{};
        DeviceMoETransferSlotClaimIndex transfer_slot_claim_index{};
        DeviceMoERebalanceGraphControllerState controller_state{};
        DeviceMoERebalanceApplyStatus copy_status{};
        DeviceMoERebalanceApplyStatus apply_status{};
        std::vector<DeviceMoERebalanceCommandBufferHeader> command_headers(command_buffer_count);
        std::vector<DeviceMoERebalanceWaveState> wave_states(command_buffer_count);
        std::vector<uint32_t> plan_counts(command_buffer_count, 0);
        std::vector<DeviceMoERebalancePlanEntry> trace_plan_entries(
            static_cast<size_t>(command_buffer_count) *
            static_cast<size_t>(trace_plan_capacity));
        const uint32_t trace_histogram_layer_count =
            export_trace
                ? static_cast<uint32_t>(std::min<size_t>(
                      rebalance_stage->traceHistogramLayerCount(),
                      static_cast<size_t>(std::numeric_limits<uint32_t>::max())))
                : 0u;
        std::vector<uint64_t> trace_gathered_histograms(
            export_trace
                ? rebalance_stage->traceGatheredHistogramEntries()
                : 0u);

        if (!copy_required(&status,
                           buffer_name(MoEDeviceRebalanceStage::WS_STATUS),
                           sizeof(status)))
        {
            return false;
        }

        const bool export_perfstats =
            PerfStatsCollector::isDomainEnabled("moe_rebalance");
        const bool export_diagnostics = export_native_journal || export_perfstats || export_trace || outcome;
        if (export_diagnostics)
        {
            if (!copy_required(&controller_state,
                               buffer_name(MoEDeviceRebalanceStage::WS_CONTROLLER_STATE),
                               sizeof(controller_state)) ||
                !copy_required(
                    &transfer_slot_claim_index,
                    buffer_name(
                        MoEDeviceRebalanceStage::
                            WS_TRANSFER_SLOT_CLAIM_INDEX),
                    sizeof(transfer_slot_claim_index)) ||
                !copy_required(command_headers.data(),
                               buffer_name(MoEDeviceRebalanceStage::WS_COMMAND_HEADER),
                               command_headers.size() * sizeof(command_headers.front())) ||
                !copy_required(wave_states.data(),
                               buffer_name(MoEDeviceRebalanceStage::WS_WAVE_STATE),
                               wave_states.size() * sizeof(wave_states.front())) ||
                !copy_required(plan_counts.data(),
                               buffer_name(MoEDeviceRebalanceStage::WS_TRANSFER_PLAN_COUNT),
                               plan_counts.size() * sizeof(plan_counts.front())))
            {
                return false;
            }
            /*
             * Completion records are mandatory only when the concrete stage
             * phase declares ownership. Shared workspace allocation is not a
             * publication signal. Atomic maintenance owns both copy and apply
             * records, while steady decode apply-only stages own only apply.
             */
            if (publication_contract.copy_status &&
                !copy_required(
                    &copy_status,
                    buffer_name(MoEDeviceRebalanceStage::WS_COPY_STATUS),
                    sizeof(copy_status)))
            {
                return false;
            }
            if (publication_contract.apply_status &&
                !copy_required(
                    &apply_status,
                    buffer_name(MoEDeviceRebalanceStage::WS_APPLY_STATUS),
                    sizeof(apply_status)))
            {
                return false;
            }
            if (export_trace && trace_plan_capacity > 0u)
            {
                if (!copy_required(trace_plan_entries.data(),
                                   buffer_name(MoEDeviceRebalanceStage::WS_TRANSFER_PLAN),
                                   trace_plan_entries.size() * sizeof(trace_plan_entries.front())))
                {
                    return false;
                }
            }
            if (export_trace && !trace_gathered_histograms.empty())
            {
                const std::string histogram_name =
                    buffer_name(MoEDeviceRebalanceStage::WS_GATHERED_HISTOGRAM);
                if (workspace->getBuffer(histogram_name))
                {
                    if (!copy_required(trace_gathered_histograms.data(),
                                       histogram_name,
                                       trace_gathered_histograms.size() *
                                           sizeof(trace_gathered_histograms.front())))
                    {
                        return false;
                    }
                }
                else
                {
                    trace_gathered_histograms.clear();
                }
            }
        }

        auto finish_terminal_readback = [&]() -> bool
        {
            PerfStatsCollector::ScopedTimer readback_timer(
                "moe_rebalance",
                "device_rebalance_status_readback",
                "decode",
                device_key,
                maintenance_tags);
            if (!backend->synchronizeStream(maintenance_stream, device_ordinal))
            {
                LOG_ERROR("[DGO] Device MoE rebalance diagnostic export failed to synchronize readback stream for "
                          << device_key);
                return false;
            }
            return true;
        };
        if (!finish_terminal_readback())
            return false;
        if (export_diagnostics)
        {
            // These are the already joined terminal copies. They describe
            // native policy progress without adding a device observation.
            const auto &config = rebalance_stage->getParams().config;
            LOG_DEBUG("[DGO] Native MoE terminal decision evidence"
                      << " device=" << device_key
                      << " launch_count=" << cache.launch_count
                      << " completed_decisions=" << controller_state.completed_decision.completed_windows
                      << " committed_decode_rounds=" << controller_state.decode_rounds_committed
                      << " maintenance_due=" << controller_state.maintenance_due
                      << " rounds_until_maintenance=" << controller_state.decode_rounds_until_maintenance
                      << " status_code=" << status.status_code
                      << " windows_observed=" << status.windows_observed
                      << " window_required_slots=" << status.window_required_slots
                      << " window_ready_slots=" << status.window_ready_slots
                      << " skipped_not_ready=" << status.skipped_not_ready
                      << " skipped_busy_wave=" << status.skipped_busy_wave
                      << " flags=" << config.flags
                      << " participant=" << config.participant_id
                      << " root=" << config.root_participant
                      << " runtime_layers=" << config.num_layers
                      << " layer_window_start=" << config.layer_window_start
                      << " layer_window_count=" << config.layer_window_count
                      << " layer_wave_count=" << config.layer_wave_count);
        }
        if (export_native_journal)
        {
            // Counts become readable only at this terminal producer join. Copy
            // the populated prefix, never the full capacity or live histogram.
            // This path is mandatory even with every PerfStats domain disabled.
            const auto view = rebalance_stage->movementJournalView();
            const auto &journal = controller_state.movement_journal;
            if (controller_state.magic != kDeviceMoERebalanceMagic ||
                controller_state.version != kDeviceMoERebalanceVersion ||
                !view.state || !view.waves || !view.edges || !view.wave_capacity || !view.edge_capacity ||
                journal.committed_waves > view.wave_capacity || journal.committed_edges > view.edge_capacity)
                throw std::runtime_error("Native MoE terminal journal has invalid ABI or populated bounds");
            std::vector<DeviceMoERebalanceMovementWave> waves(journal.committed_waves);
            std::vector<DeviceMoERebalanceMovementEdge> edges(journal.committed_edges);
            if ((!waves.empty() && !copy_required(waves.data(),
                    buffer_name(MoEDeviceRebalanceStage::WS_MOVEMENT_WAVES), waves.size() * sizeof(waves.front()))) ||
                (!edges.empty() && !copy_required(edges.data(),
                    buffer_name(MoEDeviceRebalanceStage::WS_MOVEMENT_EDGES), edges.size() * sizeof(edges.front()))))
                return false;
            if ((!waves.empty() || !edges.empty()) && !finish_terminal_readback())
                return false;
            archiveCompletedNativeMoEMovement(*rebalance_stage, controller_state,
                cache.workspace_generation, waves, edges);
        }
        if (outcome)
        {
            outcome->valid =
                status.magic == kDeviceMoERebalanceMagic &&
                status.version == kDeviceMoERebalanceVersion;
            outcome->controller_valid =
                controller_state.magic == kDeviceMoERebalanceMagic &&
                controller_state.version == kDeviceMoERebalanceVersion;
            outcome->copy_status_present =
                publication_contract.copy_status;
            outcome->copy_status_valid =
                !publication_contract.copy_status ||
                (copy_status.magic == kDeviceMoERebalanceMagic &&
                 copy_status.version == kDeviceMoERebalanceVersion);
            outcome->apply_status_present =
                publication_contract.apply_status;
            outcome->apply_status_valid =
                !publication_contract.apply_status ||
                (apply_status.magic == kDeviceMoERebalanceMagic &&
                 apply_status.version == kDeviceMoERebalanceVersion);
            outcome->status_code = status.status_code;
            outcome->invalid_runtime_layers = status.invalid_runtime_layers;
            outcome->first_invalid_runtime_layer =
                transfer_slot_claim_index.first_invalid_runtime_layer;
            outcome->first_invalid_runtime_active_bank =
                transfer_slot_claim_index
                    .first_invalid_runtime_active_bank;
            outcome->first_invalid_runtime_active_epoch =
                transfer_slot_claim_index
                    .first_invalid_runtime_active_epoch;
            outcome->first_invalid_runtime_expert_count =
                transfer_slot_claim_index
                    .first_invalid_runtime_expert_count;
            outcome->first_invalid_runtime_participant_id =
                transfer_slot_claim_index
                    .first_invalid_runtime_participant_id;
            outcome->first_invalid_runtime_participant_count =
                transfer_slot_claim_index
                    .first_invalid_runtime_participant_count;
            outcome->plan_overflow = status.plan_overflow;
            outcome->capacity_limited_candidates =
                status.capacity_limited_candidates;
            outcome->payload_bucket_overflow =
                status.payload_bucket_overflow;
            outcome->controller_last_error_code =
                controller_state.last_error_code;
            outcome->controller_last_error_wave_index =
                controller_state.last_error_wave_index;
            outcome->controller_last_error_epoch =
                controller_state.last_error_epoch;
            outcome->controller_last_error_expected_arrivals =
                controller_state.last_error_expected_arrivals;
            outcome->controller_last_error_copied_arrivals =
                controller_state.last_error_copied_arrivals;
            outcome->controller_last_error_copy_status_code =
                controller_state.last_error_copy_status_code;
            outcome->controller_last_error_copy_failure_flags =
                controller_state.last_error_copy_failure_flags;
            outcome->controller_last_error_copy_plan_entries_seen =
                controller_state.last_error_copy_plan_entries_seen;
            outcome
                ->controller_last_error_copy_skipped_wrong_destination =
                controller_state
                    .last_error_copy_skipped_wrong_destination;
            outcome->controller_last_error_missing_destination_slot =
                controller_state.last_error_missing_destination_slot;
            outcome->controller_last_error_missing_destination_layer =
                controller_state.last_error_missing_destination_layer;
            outcome->controller_last_error_missing_destination_expert =
                controller_state.last_error_missing_destination_expert;
            outcome->controller_last_error_missing_destination_source =
                controller_state.last_error_missing_destination_source;
            outcome->controller_last_error_local_transfer_slot_count =
                controller_state.last_error_local_transfer_slot_count;
            const uint32_t controller_wave_count =
                std::min<uint32_t>(controller_state.wave_count, 2u);
            for (uint32_t wave_index = 0;
                 wave_index < controller_wave_count;
                 ++wave_index)
            {
                const auto &wave = controller_state.waves[wave_index];
                if (wave.state ==
                        static_cast<uint32_t>(
                            DeviceMoERebalanceWaveLifecycle::Error) ||
                    wave.error_code != 0u)
                {
                    ++outcome->controller_error_waves;
                    if (wave_index ==
                        outcome->controller_last_error_wave_index)
                    {
                        outcome->controller_last_error_participant =
                            wave.error_participant;
                    }
                }
            }
            outcome->copy_status_code = copy_status.status_code;
            outcome->copy_transaction_wave_index =
                copy_status.transaction_wave_index;
            outcome->copy_transaction_epoch =
                copy_status.transaction_epoch;
            outcome->copy_transaction_command_count =
                copy_status.transaction_command_count;
            outcome->copy_plan_entries_seen =
                copy_status.plan_entries_seen;
            outcome->copy_copied_arrivals =
                copy_status.copied_arrivals;
            outcome->copy_skipped_wrong_destination =
                copy_status.skipped_wrong_destination;
            outcome->copy_invalid_plan_entries =
                copy_status.invalid_plan_entries;
            outcome->copy_missing_source_descriptors =
                copy_status.missing_source_descriptors;
            outcome->copy_missing_destination_slots =
                copy_status.missing_destination_slots;
            outcome->copy_descriptor_mismatches =
                copy_status.descriptor_mismatches;
            outcome->copy_incomplete = copy_status.copy_incomplete;
            outcome->copy_required_local_arrivals =
                copy_status.required_local_arrivals;
            outcome->copy_ready_local_arrivals =
                copy_status.ready_local_arrivals;
            outcome->apply_status_code = apply_status.status_code;
            outcome->apply_invalid_plan_entries =
                apply_status.invalid_plan_entries;
            outcome->apply_missing_source_descriptors =
                apply_status.missing_source_descriptors;
            outcome->apply_missing_destination_slots =
                apply_status.missing_destination_slots;
            outcome->apply_descriptor_mismatches =
                apply_status.descriptor_mismatches;
            outcome->apply_copy_incomplete = apply_status.copy_incomplete;
            outcome->apply_required_local_arrivals =
                apply_status.required_local_arrivals;
            outcome->apply_ready_local_arrivals =
                apply_status.ready_local_arrivals;
            outcome->windows_applied = status.windows_applied;
            outcome->selected_replicas = status.selected_replicas;
            outcome->planned_arrivals = status.planned_arrivals;
            const bool apply_status_valid =
                apply_status.magic == kDeviceMoERebalanceMagic &&
                apply_status.version == kDeviceMoERebalanceVersion &&
                apply_status.status_code ==
                    static_cast<uint32_t>(DeviceMoERebalanceApplyStatusCode::Ok);
            outcome->runtime_changed_layers =
                apply_status_valid ? apply_status.changed_layers : status.changed_layers;
            outcome->runtime_applied_arrivals =
                apply_status_valid ? apply_status.applied_arrivals : 0u;
            outcome->prefill_current_batch_movement_layers =
                status.prefill_current_batch_movement_layers;
            outcome->prefill_current_batch_non_owner_assignment_layers =
                status.prefill_current_batch_non_owner_assignment_layers;
            outcome->prefill_active_transfer_slot_experts =
                status.prefill_active_transfer_slot_experts;
            outcome->prefill_unique_transfer_slot_claims =
                status.prefill_unique_transfer_slot_claims;
            outcome->prefill_duplicate_transfer_slot_claims =
                status.prefill_duplicate_transfer_slot_claims;
            outcome->prefill_invalid_transfer_slot_claims =
                status.prefill_invalid_transfer_slot_claims;
            outcome->prefill_max_transfer_slot =
                status.prefill_max_transfer_slot;
            outcome->prefill_max_transfer_slot_layer =
                status.prefill_max_transfer_slot_layer;
            outcome->prefill_max_transfer_slot_expert =
                status.prefill_max_transfer_slot_expert;
            outcome->prefill_first_duplicate_transfer_slot =
                status.prefill_first_duplicate_transfer_slot;
            outcome->prefill_first_duplicate_layer =
                status.prefill_first_duplicate_layer;
            outcome->prefill_first_duplicate_expert =
                status.prefill_first_duplicate_expert;
            outcome->prefill_first_invalid_transfer_slot =
                status.prefill_first_invalid_transfer_slot;
            outcome->prefill_first_invalid_layer =
                status.prefill_first_invalid_layer;
            outcome->prefill_first_invalid_expert =
                status.prefill_first_invalid_expert;
            outcome->prefill_first_invalid_reasons =
                status.prefill_first_invalid_reasons;
            outcome->prefill_first_invalid_flags =
                status.prefill_first_invalid_flags;
            outcome->prefill_first_invalid_resident_mask =
                status.prefill_first_invalid_resident_mask;
            outcome->prefill_first_invalid_owner =
                status.prefill_first_invalid_owner;
            outcome->local_transfer_slot_count =
                params.local_transfer_slot_count;
            outcome->payload_bucket_slots = status.payload_bucket_slots;
            outcome->payload_source_participant_mask =
                status.payload_source_participant_mask;
            outcome->payload_destination_participant_mask =
                status.payload_destination_participant_mask;
            outcome->payload_edge_mask = status.payload_edge_mask;
            outcome->useful_work =
                outcome->valid &&
                (status.prefill_current_batch_movement_layers != 0u ||
                 status.prefill_current_batch_non_owner_assignment_layers != 0u ||
                 status.prefill_active_transfer_slot_experts != 0u ||
                 (status.status_code ==
                      static_cast<uint32_t>(DeviceMoERebalanceStatusCode::Ok) &&
                  (status.windows_applied != 0u ||
                   status.changed_layers != 0u ||
                   status.selected_replicas != 0u ||
                   status.planned_arrivals != 0u ||
                   outcome->runtime_changed_layers != 0u ||
                   outcome->runtime_applied_arrivals != 0u ||
                   status.payload_bucket_slots != 0u)));
        }
        const bool valid_status_object =
            status.magic == kDeviceMoERebalanceMagic &&
            status.version == kDeviceMoERebalanceVersion;
        const bool valid_status =
            valid_status_object &&
            status.status_code ==
                static_cast<uint32_t>(DeviceMoERebalanceStatusCode::Ok);
        const bool valid_apply_status =
            apply_status.magic == kDeviceMoERebalanceMagic &&
            apply_status.version == kDeviceMoERebalanceVersion &&
            apply_status.status_code ==
                static_cast<uint32_t>(DeviceMoERebalanceApplyStatusCode::Ok);
        const bool valid_copy_status =
            copy_status.magic == kDeviceMoERebalanceMagic &&
            copy_status.version == kDeviceMoERebalanceVersion &&
            copy_status.status_code ==
                static_cast<uint32_t>(DeviceMoERebalanceApplyStatusCode::Ok);
        const bool runtime_moved =
            valid_status_object &&
            (status.prefill_current_batch_movement_layers != 0u ||
             status.prefill_active_transfer_slot_experts != 0u ||
             (valid_status &&
              (status.windows_applied != 0u ||
               status.changed_layers != 0u ||
               status.selected_replicas != 0u ||
               status.planned_arrivals != 0u ||
               status.payload_bucket_slots != 0u ||
               (valid_apply_status &&
                (apply_status.changed_layers != 0u ||
                 apply_status.applied_arrivals != 0u)))));
        if (runtime_moved)
        {
            ++moe_runtime_movement_epoch_;
        }
        cache.last_status_publication_launch_count = cache.launch_count;
        if (!export_diagnostics)
            return true;

        std::vector<int32_t> trace_owner_participants;
        if (export_trace &&
            trace_histogram_layer_count > 0u &&
            params.moe_runtime_table != nullptr &&
            params.config.num_layers > 0u &&
            params.config.num_experts > 0u)
        {
            const uint32_t expert_count =
                std::min<uint32_t>(params.config.num_experts, kDeviceMoEMaxExperts);
            trace_owner_participants.assign(
                static_cast<size_t>(trace_histogram_layer_count) *
                    static_cast<size_t>(expert_count),
                -1);
            const uint32_t layer_window_count =
                params.config.layer_window_count == 0u
                    ? params.config.num_layers
                    : std::min(params.config.layer_window_count,
                               params.config.num_layers);
            const uint32_t layer_window_start =
                params.config.layer_window_start % params.config.num_layers;
            for (uint32_t wave_layer = 0; wave_layer < trace_histogram_layer_count; ++wave_layer)
            {
                const uint32_t layer =
                    (layer_window_start +
                     (layer_window_count == 0u ? wave_layer : wave_layer % layer_window_count)) %
                    params.config.num_layers;
                if (params.moe_runtime_table->decodeRuntimePublicationRequired(
                        static_cast<int>(layer)))
                {
                    continue;
                }
                const auto &runtime = params.moe_runtime_table->hostLayerState(
                    static_cast<int>(layer));
                if (runtime.active_bank > 1u)
                    continue;
                const auto &bank = runtime.banks[runtime.active_bank];
                for (uint32_t expert = 0; expert < expert_count; ++expert)
                {
                    trace_owner_participants[
                        static_cast<size_t>(wave_layer) * static_cast<size_t>(expert_count) +
                        static_cast<size_t>(expert)] =
                        bank.experts[expert].owner_participant;
                }
            }
        }

        // Request/command epochs and measured load magnitudes are observation
        // words, not dimensions. Keep the exact primary totals and status/policy
        // partitions while long-lived serving retains a fixed number of rows.
        auto tags = maintenance_tags;
        tags["workspace"] = workspace_name;
        tags["stage"] = params.stage_name;
        tags["publishes_copy_status"] =
            boolTag(publication_contract.copy_status);
        tags["publishes_apply_status"] =
            boolTag(publication_contract.apply_status);
        tags["status_code"] = std::to_string(status.status_code);
        tags["transfer_mode"] = deviceMoERebalanceTransferModeTag(params.transfer_mode);
        tags["participant_count"] = std::to_string(params.config.participant_count);
        tags["local_transfer_slot_count"] = std::to_string(params.local_transfer_slot_count);
        tags["collective_payload_slot_capacity"] =
            std::to_string(params.collective_payload_slot_capacity);
        tags["collective_payload_slot_bytes"] =
            std::to_string(params.collective_payload_slot_bytes);
        tags["load_stats_enabled"] =
            boolTag(hasDeviceMoERebalanceFlag(
                params.config.flags,
                DeviceMoERebalanceFlags::CollectLoadStats));
        tags["min_load_spread_improvement"] =
            std::to_string(params.config.min_load_spread_improvement);
        tags["min_load_spread_improvement_divisor"] =
            std::to_string(params.config.min_load_spread_improvement_divisor);
        tags["min_wave_spread_improvement_per_payload_slot"] =
            std::to_string(params.config.min_wave_spread_improvement_per_payload_slot);
        tags["min_foreign_rows_per_critical_path_payload_slot"] =
            std::to_string(
                params.config
                    .min_foreign_rows_per_critical_path_payload_slot);
        tags["min_router_spread_improvement_per_payload_slot"] =
            std::to_string(params.config.min_router_spread_improvement_per_payload_slot);
        tags["max_post_wave_load_spread_per_mille"] =
            std::to_string(params.config.max_post_wave_load_spread_per_mille);
        tags["dynamic_imbalance_threshold_per_mille"] =
            std::to_string(params.config.dynamic_imbalance_threshold_per_mille);
        tags["dynamic_min_improvement_per_mille"] =
            std::to_string(params.config.dynamic_min_improvement_per_mille);
        tags["dynamic_max_swaps_per_layer"] =
            std::to_string(params.config.dynamic_max_swaps_per_layer);
        tags["dynamic_min_window_activations"] =
            std::to_string(params.config.dynamic_min_window_activations);

        auto emit_u64 = [&](const std::string &name, uint64_t value)
        {
            PerfStatsCollector::addCounterWithSequence(
                "moe_rebalance",
                name,
                static_cast<double>(value),
                {cache.launch_count, status.last_epoch},
                "decode",
                device_key,
                tags);
        };
        auto emit_u32 = [&](const std::string &name, uint32_t value)
        {
            emit_u64(name, static_cast<uint64_t>(value));
        };
        auto emit_double = [&](const std::string &name, double value)
        {
            PerfStatsCollector::addCounterWithSequence(
                "moe_rebalance",
                name,
                value,
                {cache.launch_count, status.last_epoch},
                "decode",
                device_key,
                tags);
        };
        auto safe_ratio = [](uint64_t numerator, uint64_t denominator) -> double
        {
            return denominator == 0 ? 0.0
                                    : static_cast<double>(numerator) /
                                          static_cast<double>(denominator);
        };
        auto popcount_u64 = [](uint64_t value) -> uint32_t
        {
            uint32_t count = 0;
            while (value != 0)
            {
                count += static_cast<uint32_t>(value & 1ULL);
                value >>= 1u;
            }
            return count;
        };

        emit_u32("device_rebalance_status_code", status.status_code);
        emit_u32("device_rebalance_windows_observed", status.windows_observed);
        emit_u32("device_rebalance_windows_applied", status.windows_applied);
        emit_u32("device_rebalance_changed_layers", status.changed_layers);
        emit_u32("device_rebalance_selected_replicas", status.selected_replicas);
        emit_u32("device_rebalance_skipped_not_ready", status.skipped_not_ready);
        emit_u32("device_rebalance_skipped_busy_wave", status.skipped_busy_wave);
        emit_u32("device_rebalance_skipped_histogram_not_ready",
                 status.status_code ==
                         static_cast<uint32_t>(DeviceMoERebalanceStatusCode::WindowNotReady) &&
                         status.skipped_not_ready != 0u &&
                         status.skipped_busy_wave == 0u
                     ? 1u
                     : 0u);
        emit_u32("device_rebalance_window_ready_slots", status.window_ready_slots);
        emit_u32("device_rebalance_window_required_slots", status.window_required_slots);
        emit_u32("device_rebalance_skipped_no_resident", status.skipped_no_resident);
        emit_u32("device_rebalance_skipped_no_improvement", status.skipped_no_improvement);
        emit_u32("device_rebalance_dynamic_ownership_swap_attempts",
                 status.dynamic_ownership_swap_attempts);
        emit_u32("device_rebalance_dynamic_ownership_swap_accepts",
                 status.dynamic_ownership_swap_accepts);
        emit_u32("device_rebalance_dynamic_ownership_swap_rejections",
                 status.dynamic_ownership_swap_rejections);
        emit_u32("device_rebalance_skipped_wave_cost_floor",
                 status.skipped_wave_cost_floor);
        emit_u32("device_rebalance_skipped_low_router_benefit",
                 status.skipped_low_router_benefit);
        emit_u32("device_rebalance_skipped_post_load_spread_ceiling",
                 status.skipped_post_load_spread_ceiling);
        emit_u32("device_rebalance_skipped_participant_load_spread",
                 status.skipped_participant_load_spread);
        emit_u32("device_rebalance_skipped_aggregate_load_spread",
                 status.skipped_aggregate_load_spread);
        emit_u32("device_rebalance_skipped_configured_load_spread_ceiling",
                 status.skipped_configured_load_spread_ceiling);
        emit_u32("device_rebalance_candidate_arrivals_considered",
                 status.candidate_arrivals_considered);
        emit_u32("device_rebalance_candidate_arrivals_below_floor",
                 status.candidate_arrivals_below_floor);
        emit_u32("device_rebalance_candidate_arrivals_pruned_by_count_bound",
                 status.candidate_arrivals_pruned_by_count_bound);
        emit_u32("device_rebalance_llep_assignment_span_count",
                 status.llep_assignment_span_count);
        emit_u32("device_rebalance_llep_weight_transfer_count",
                 status.llep_weight_transfer_count);
        emit_u32("device_rebalance_prefill_current_batch_movement_layers",
                 status.prefill_current_batch_movement_layers);
        emit_u32(
            "device_rebalance_prefill_current_batch_non_owner_assignment_layers",
            status.prefill_current_batch_non_owner_assignment_layers);
        emit_u32("device_rebalance_prefill_active_transfer_slot_experts",
                 status.prefill_active_transfer_slot_experts);
        emit_u32("device_rebalance_prefill_unique_transfer_slot_claims",
                 status.prefill_unique_transfer_slot_claims);
        emit_u32("device_rebalance_prefill_duplicate_transfer_slot_claims",
                 status.prefill_duplicate_transfer_slot_claims);
        emit_u32("device_rebalance_prefill_invalid_transfer_slot_claims",
                 status.prefill_invalid_transfer_slot_claims);
        emit_u32("device_rebalance_prefill_max_transfer_slot",
                 status.prefill_max_transfer_slot);
        emit_u32("device_rebalance_prefill_max_transfer_slot_layer",
                 status.prefill_max_transfer_slot_layer);
        emit_u32("device_rebalance_prefill_max_transfer_slot_expert",
                 status.prefill_max_transfer_slot_expert);
        emit_u32("device_rebalance_prefill_first_duplicate_transfer_slot",
                 status.prefill_first_duplicate_transfer_slot);
        emit_u32("device_rebalance_prefill_first_duplicate_layer",
                 status.prefill_first_duplicate_layer);
        emit_u32("device_rebalance_prefill_first_duplicate_expert",
                 status.prefill_first_duplicate_expert);
        emit_u32("device_rebalance_prefill_first_invalid_transfer_slot",
                 status.prefill_first_invalid_transfer_slot);
        emit_u32("device_rebalance_prefill_first_invalid_layer",
                 status.prefill_first_invalid_layer);
        emit_u32("device_rebalance_prefill_first_invalid_expert",
                 status.prefill_first_invalid_expert);
        emit_u32("device_rebalance_prefill_first_invalid_reasons",
                 status.prefill_first_invalid_reasons);
        emit_u32("device_rebalance_prefill_first_invalid_flags",
                 status.prefill_first_invalid_flags);
        emit_u32("device_rebalance_prefill_first_invalid_resident_mask",
                 status.prefill_first_invalid_resident_mask);
        emit_u32("device_rebalance_prefill_first_invalid_owner",
                 static_cast<uint32_t>(
                     status.prefill_first_invalid_owner));
        emit_u32("device_rebalance_llep_standard_ep_selected",
                 status.llep_standard_ep_selected);
        emit_u32("device_rebalance_llep_skipped_balanced",
                 status.llep_skipped_balanced);
        emit_u32("device_rebalance_llep_skipped_insufficient_spread_improvement",
                 status.llep_skipped_insufficient_spread_improvement);
        emit_u32("device_rebalance_llep_skipped_insufficient_foreign_rows",
                 status.llep_skipped_insufficient_foreign_rows);
        emit_u32("device_rebalance_llep_min_chunk_skips",
                 status.llep_min_chunk_skips);
        emit_u32("device_rebalance_llep_forced_spills",
                 status.llep_forced_spills);
        emit_u64("device_rebalance_candidate_load_spread_improvement_total",
                 status.candidate_load_spread_improvement_total);
        emit_u64("device_rebalance_candidate_load_spread_improvement_max",
                 status.candidate_load_spread_improvement_max);
        emit_u64("device_rebalance_accepted_load_spread_improvement_total",
                 status.accepted_load_spread_improvement_total);
        emit_u64("device_rebalance_accepted_load_spread_improvement_max",
                 status.accepted_load_spread_improvement_max);
        emit_u64("device_rebalance_llep_native_rows",
                 status.llep_native_rows);
        emit_u64("device_rebalance_llep_spilled_rows",
                 status.llep_spilled_rows);
        emit_u64("device_rebalance_llep_required_spread_improvement",
                 status.llep_required_spread_improvement);
        emit_u64("device_rebalance_llep_required_foreign_rows",
                 status.llep_required_foreign_rows);
        emit_u64("device_rebalance_pre_wave_load_total",
                 status.pre_wave_load_total);
        emit_u64("device_rebalance_pre_wave_load_spread",
                 status.pre_wave_load_spread);
        emit_u64("device_rebalance_post_wave_load_total",
                 status.post_wave_load_total);
        emit_u64("device_rebalance_post_wave_load_spread",
                 status.post_wave_load_spread);
        emit_u64("device_rebalance_router_hot_cache_eligible_dispatches",
                 status.router_hot_cache_eligible_dispatches);
        emit_u64("device_rebalance_router_hot_cache_used_dispatches",
                 status.router_hot_cache_used_dispatches);
        emit_u64("device_rebalance_router_hot_cache_improved_dispatches",
                 status.router_hot_cache_improved_dispatches);
        emit_u64("device_rebalance_router_hot_cache_default_load_spread_total",
                 status.router_hot_cache_default_load_spread_total);
        emit_u64("device_rebalance_router_hot_cache_actual_load_spread_total",
                 status.router_hot_cache_actual_load_spread_total);
        emit_u64("device_rebalance_router_hot_cache_load_spread_improvement_total",
                 status.router_hot_cache_load_spread_improvement_total);
        emit_u64("device_rebalance_router_hot_cache_active_dispatches",
                 status.router_hot_cache_active_dispatches);
        emit_u64("device_rebalance_router_hot_cache_miss_dispatches",
                 status.router_hot_cache_miss_dispatches);
        emit_u64("device_rebalance_router_hot_cache_selected_expert_slots",
                 status.router_hot_cache_selected_expert_slots);
        emit_u64("device_rebalance_router_hot_cache_replicated_selected_expert_slots",
                 status.router_hot_cache_replicated_selected_expert_slots);
        emit_u32("device_rebalance_invalid_runtime_layers", status.invalid_runtime_layers);
        emit_u32("device_rebalance_planned_arrivals", status.planned_arrivals);
        emit_u32("device_rebalance_plan_overflow", status.plan_overflow);
        emit_u32("device_rebalance_capacity_limited_candidates",
                 status.capacity_limited_candidates);
        emit_u32("device_rebalance_payload_bucket_requested_slots",
                 status.payload_bucket_requested_slots);
        emit_u32("device_rebalance_payload_bucket_slots",
                 status.payload_bucket_slots);
        emit_u32("device_rebalance_payload_bucket_index",
                 status.payload_bucket_index);
        emit_u32("device_rebalance_payload_bucket_overflow",
                 status.payload_bucket_overflow);
        emit_u32("device_rebalance_payload_source_participant_mask",
                 status.payload_source_participant_mask);
        emit_u32("device_rebalance_payload_destination_participant_mask",
                 status.payload_destination_participant_mask);
        emit_u32("device_rebalance_payload_edge_mask_low32",
                 static_cast<uint32_t>(status.payload_edge_mask & 0xffffffffULL));
        emit_u32("device_rebalance_payload_edge_mask_high32",
                 static_cast<uint32_t>(status.payload_edge_mask >> 32u));
        emit_u32("device_rebalance_payload_active_edge_count",
                 popcount_u64(status.payload_edge_mask));

        uint64_t wave_command_count_total = 0;
        uint64_t wave_copied_arrivals_total = 0;
        uint64_t wave_applied_arrivals_total = 0;
        uint64_t wave_applied_layer_count_total = 0;
        for (uint32_t i = 0; i < controller_state.wave_count && i < 2u; ++i)
        {
            const auto &wave = controller_state.waves[i];
            wave_command_count_total += wave.command_count;
            wave_copied_arrivals_total += wave.copied_arrivals;
            wave_applied_arrivals_total += wave.applied_arrivals;
            wave_applied_layer_count_total += wave.applied_layer_count;
        }
        uint32_t transfer_copied_arrivals = 0;
        bool transfer_wave_matched = false;
        uint32_t transfer_wave_index = 0;
        uint32_t transfer_command_epoch = 0;
        uint32_t transfer_wave_epoch = 0;
        uint32_t transfer_wave_state = static_cast<uint32_t>(DeviceMoERebalanceWaveLifecycle::Idle);
        uint32_t transfer_command_count = 0;
        uint32_t transfer_applied_arrivals = 0;
        uint32_t transfer_applied_layer_count = 0;
        uint32_t transfer_requested_payload_slots = status.payload_bucket_requested_slots;
        uint32_t transfer_payload_bucket_slots = status.payload_bucket_slots;
        uint32_t transfer_payload_bucket_index = status.payload_bucket_index;
        uint32_t transfer_payload_bucket_overflow = status.payload_bucket_overflow;
        uint32_t transfer_payload_source_participant_mask =
            status.payload_source_participant_mask;
        uint32_t transfer_payload_destination_participant_mask =
            status.payload_destination_participant_mask;
        uint64_t transfer_payload_edge_mask = status.payload_edge_mask;
        if (status.status_code == static_cast<uint32_t>(DeviceMoERebalanceStatusCode::Ok) &&
            status.windows_applied != 0u &&
            status.last_epoch != 0u)
        {
            const uint32_t wave_limit =
                std::min<uint32_t>(std::min<uint32_t>(controller_state.wave_count, 2u),
                                   command_buffer_count);
            for (uint32_t i = 0; i < wave_limit; ++i)
            {
                const auto &header = command_headers[i];
                const auto &wave = controller_state.waves[i];
                const uint32_t header_count =
                    std::min<uint32_t>(header.command_count, header.command_capacity);
                const uint32_t command_capacity =
                    header.command_capacity != 0u ? header.command_capacity : wave.command_count;
                const uint32_t wave_count =
                    std::min<uint32_t>(wave.command_count, command_capacity);
                if (wave.epoch == 0u ||
                    wave.epoch != status.last_epoch ||
                    wave_count == 0u)
                {
                    continue;
                }

                transfer_wave_matched = true;
                transfer_wave_index = i;
                transfer_command_epoch = header.epoch != 0u ? header.epoch : wave.epoch;
                transfer_wave_epoch = wave.epoch;
                transfer_wave_state = wave.state;
                transfer_command_count = header_count != 0u ? header_count : wave_count;
                transfer_applied_arrivals =
                    std::min<uint32_t>(wave.applied_arrivals, wave_count);
                transfer_applied_layer_count =
                    std::min<uint32_t>(wave.applied_layer_count,
                                       params.config.num_layers);
                transfer_requested_payload_slots = wave.requested_payload_slots;
                transfer_payload_bucket_slots = wave.payload_bucket_slots;
                transfer_payload_bucket_index = wave.payload_bucket_index;
                transfer_payload_bucket_overflow = wave.payload_bucket_overflow;
                if (wave.state >=
                        static_cast<uint32_t>(DeviceMoERebalanceWaveLifecycle::TransferInFlight) &&
                    wave.state <=
                        static_cast<uint32_t>(DeviceMoERebalanceWaveLifecycle::Applied))
                {
                    transfer_copied_arrivals =
                        std::min<uint32_t>(wave.copied_arrivals, wave_count);
                }
                break;
            }
        }

        const auto transfer_cost =
            estimateDeviceMoERebalanceTransferCost(
                params.config,
                params.transfer_mode,
                params.local_transfer_slot_count,
                params.collective_payload_slot_bytes,
                transfer_copied_arrivals,
                params.collective_payload_slot_capacity,
                transfer_payload_bucket_slots,
                transfer_payload_edge_mask != 0ULL
                    ? popcount_u64(transfer_payload_edge_mask)
                    : std::numeric_limits<uint32_t>::max());
        const auto request_movement =
            deviceMoERebalanceRequestMovementEvidence(
                {controller_state.waves,
                 std::min<uint32_t>(controller_state.wave_count, 2u)},
                status.prefill_current_batch_movement_layers,
                transfer_cost.slot_payload_bytes);
        recordCompletedDeviceMoEMovement(request_movement, "decode", device_key, "maintenance_status");
        if (transfer_cost.transfer_plan_capacity > 0 ||
            params.local_transfer_slot_count > 0 ||
            params.collective_payload_slot_bytes > 0)
        {
            emit_u32("device_rebalance_transfer_current_copied_arrivals",
                     transfer_copied_arrivals);
            emit_u32("device_rebalance_transfer_current_wave_index",
                     transfer_wave_index);
            emit_u32("device_rebalance_transfer_current_wave_matched",
                     transfer_wave_matched ? 1u : 0u);
            emit_u32("device_rebalance_transfer_current_command_epoch",
                     transfer_command_epoch);
            emit_u32("device_rebalance_transfer_current_wave_epoch",
                     transfer_wave_epoch);
            emit_u32("device_rebalance_transfer_current_wave_state",
                     transfer_wave_state);
            emit_u32("device_rebalance_transfer_current_command_count",
                     transfer_command_count);
            emit_u32("device_rebalance_transfer_current_applied_arrivals",
                     transfer_applied_arrivals);
            emit_u32("device_rebalance_transfer_current_applied_layer_count",
                     transfer_applied_layer_count);
            emit_u32("device_rebalance_transfer_current_requested_payload_slots",
                     transfer_requested_payload_slots);
            emit_u32("device_rebalance_transfer_current_payload_bucket_slots",
                     transfer_payload_bucket_slots);
            emit_u32("device_rebalance_transfer_current_payload_bucket_index",
                     transfer_payload_bucket_index);
            emit_u32("device_rebalance_transfer_current_payload_bucket_overflow",
                     transfer_payload_bucket_overflow);
            emit_u32("device_rebalance_transfer_current_payload_source_participant_mask",
                     transfer_payload_source_participant_mask);
            emit_u32("device_rebalance_transfer_current_payload_destination_participant_mask",
                     transfer_payload_destination_participant_mask);
            emit_u32("device_rebalance_transfer_current_payload_edge_mask_low32",
                     static_cast<uint32_t>(transfer_payload_edge_mask & 0xffffffffULL));
            emit_u32("device_rebalance_transfer_current_payload_edge_mask_high32",
                     static_cast<uint32_t>(transfer_payload_edge_mask >> 32u));
            emit_u32("device_rebalance_transfer_current_payload_active_edge_count",
                     popcount_u64(transfer_payload_edge_mask));
            emit_u64("device_rebalance_transfer_command_buffer_count",
                     transfer_cost.command_buffer_count);
            emit_u64("device_rebalance_transfer_plan_capacity",
                     transfer_cost.transfer_plan_capacity);
            emit_u64("device_rebalance_transfer_payload_slot_capacity",
                     transfer_cost.payload_slot_capacity);
            emit_u64("device_rebalance_transfer_payload_slot_count",
                     transfer_cost.payload_slot_count);
            emit_u64("device_rebalance_transfer_slot_payload_bytes",
                     transfer_cost.slot_payload_bytes);
            emit_u64("device_rebalance_transfer_plan_local_bytes",
                     transfer_cost.plan_local_bytes);
            emit_u64("device_rebalance_transfer_plan_gathered_bytes",
                     transfer_cost.plan_gathered_bytes);
            emit_u64("device_rebalance_transfer_header_local_bytes",
                     transfer_cost.header_local_bytes);
            emit_u64("device_rebalance_transfer_header_gathered_bytes",
                     transfer_cost.header_gathered_bytes);
            emit_u64("device_rebalance_transfer_source_descriptor_local_bytes",
                     transfer_cost.source_descriptor_local_bytes);
            emit_u64("device_rebalance_transfer_source_descriptor_gathered_bytes",
                     transfer_cost.source_descriptor_gathered_bytes);
            emit_u64("device_rebalance_transfer_selected_payload_bucket_slots",
                     transfer_cost.selected_payload_bucket_slots);
            emit_u64("device_rebalance_transfer_selected_payload_slot_count",
                     transfer_cost.selected_payload_slot_count);
            emit_u64("device_rebalance_transfer_selected_payload_local_capacity_bytes",
                     transfer_cost.selected_payload_local_capacity_bytes);
            emit_u64("device_rebalance_transfer_selected_payload_gathered_capacity_bytes",
                     transfer_cost.selected_payload_gathered_capacity_bytes);
            emit_u64("device_rebalance_transfer_selected_payload_transport_edge_count",
                     transfer_cost.selected_payload_transport_edge_count);
            emit_u64("device_rebalance_transfer_selected_payload_transport_capacity_bytes",
                     transfer_cost.selected_payload_transport_capacity_bytes);
            emit_u64("device_rebalance_transfer_payload_local_capacity_bytes",
                     transfer_cost.payload_local_capacity_bytes);
            emit_u64("device_rebalance_transfer_payload_gathered_capacity_bytes",
                     transfer_cost.payload_gathered_capacity_bytes);
            emit_u64("device_rebalance_transfer_captured_payload_slack_bytes",
                     transfer_cost.captured_payload_slack_bytes);
            emit_u64("device_rebalance_transfer_useful_payload_bytes",
                     transfer_cost.useful_payload_bytes);
            emit_u64(
                "device_rebalance_request_copied_payload_lower_bound",
                request_movement.copied_payload_lower_bound);
            emit_u64(
                "device_rebalance_request_applied_payload_lower_bound",
                request_movement.applied_payload_lower_bound);
            emit_u64(
                "device_rebalance_request_useful_payload_bytes_lower_bound",
                request_movement.useful_payload_bytes_lower_bound);
            emit_u64("device_rebalance_transfer_wasted_payload_capacity_bytes",
                     transfer_cost.wasted_payload_capacity_bytes);
            emit_u64("device_rebalance_transfer_wasted_payload_transport_capacity_bytes",
                     transfer_cost.wasted_payload_transport_capacity_bytes);
            emit_double("device_rebalance_transfer_payload_local_utilization_ratio",
                        safe_ratio(transfer_cost.useful_payload_bytes,
                                   transfer_cost.payload_local_capacity_bytes));
            emit_double("device_rebalance_transfer_payload_collective_utilization_ratio",
                        safe_ratio(transfer_cost.useful_payload_bytes,
                                   transfer_cost.payload_gathered_capacity_bytes));
            emit_double("device_rebalance_transfer_selected_payload_utilization_ratio",
                        safe_ratio(transfer_cost.useful_payload_bytes,
                                   transfer_cost.selected_payload_gathered_capacity_bytes));
            emit_double("device_rebalance_transfer_selected_payload_transport_utilization_ratio",
                        safe_ratio(transfer_cost.useful_payload_bytes,
                                   transfer_cost.selected_payload_transport_capacity_bytes));
            emit_double("device_rebalance_transfer_useful_bytes_per_accepted_spread_unit",
                        safe_ratio(transfer_cost.useful_payload_bytes,
                                   status.accepted_load_spread_improvement_total));
            emit_double("device_rebalance_transfer_collective_bytes_per_accepted_spread_unit",
                        safe_ratio(transfer_cost.selected_payload_gathered_capacity_bytes,
                                   status.accepted_load_spread_improvement_total));
            emit_double("device_rebalance_transfer_transport_bytes_per_accepted_spread_unit",
                        safe_ratio(transfer_cost.selected_payload_transport_capacity_bytes,
                                   status.accepted_load_spread_improvement_total));
        }

        auto ratio = [](uint64_t numerator, uint64_t denominator) -> double
        {
            return denominator == 0 ? 0.0
                                    : static_cast<double>(numerator) /
                                          static_cast<double>(denominator);
        };
        const double pre_ratio =
            ratio(status.pre_policy_imbalance_numerator,
                  status.pre_policy_imbalance_denominator);
        const double post_ratio =
            ratio(status.post_policy_imbalance_numerator,
                  status.post_policy_imbalance_denominator);
        const uint64_t pre_spread =
            deviceMoELoadSpread(status.pre_policy_load_min,
                                status.pre_policy_load_max);
        const uint64_t post_spread =
            deviceMoELoadSpread(status.post_policy_load_min,
                                status.post_policy_load_max);
        const uint64_t spread_improvement =
            pre_spread > post_spread ? pre_spread - post_spread : 0;

        auto emit_ratio =
            [&](const char *phase,
                double value,
                uint64_t total,
                uint64_t min_load,
                uint64_t max_load,
                uint64_t numerator,
                uint64_t denominator)
        {
            auto ratio_tags = tags;
            ratio_tags["policy_phase"] = phase;
            PerfStatsCollector::addCounterWithSequence(
                "moe_rebalance",
                "device_rebalance_policy_load_imbalance_ratio",
                value,
                {cache.launch_count, status.last_epoch, total, min_load, max_load, numerator, denominator},
                "decode",
                device_key,
                std::move(ratio_tags));
        };
        emit_ratio("pre",
                   pre_ratio,
                   status.pre_policy_load_total,
                   status.pre_policy_load_min,
                   status.pre_policy_load_max,
                   status.pre_policy_imbalance_numerator,
                   status.pre_policy_imbalance_denominator);
        emit_ratio("post",
                   post_ratio,
                   status.post_policy_load_total,
                   status.post_policy_load_min,
                   status.post_policy_load_max,
                   status.post_policy_imbalance_numerator,
                   status.post_policy_imbalance_denominator);
        auto emit_spread =
            [&](const char *phase, uint64_t value)
        {
            auto spread_tags = tags;
            spread_tags["policy_phase"] = phase;
            PerfStatsCollector::addCounterWithSequence(
                "moe_rebalance",
                "device_rebalance_policy_load_spread_units",
                static_cast<double>(value),
                {cache.launch_count, status.last_epoch},
                "decode",
                device_key,
                std::move(spread_tags));
        };
        emit_spread("pre", pre_spread);
        emit_spread("post", post_spread);
        emit_spread("delta", spread_improvement);
        emit_double("device_rebalance_policy_load_spread_delta_units",
                    static_cast<double>(spread_improvement));
        emit_double("device_rebalance_policy_accepted_spread_improvement_per_arrival",
                    safe_ratio(status.accepted_load_spread_improvement_total,
                               std::max<uint32_t>(1u, status.planned_arrivals)));
        emit_double("device_rebalance_post_wave_load_spread_fraction",
                    safe_ratio(status.post_wave_load_spread,
                               status.post_wave_load_total));
        emit_double("device_rebalance_llep_spilled_row_ratio",
                    safe_ratio(status.llep_spilled_rows,
                               status.llep_native_rows + status.llep_spilled_rows));
        emit_double(
                    "device_rebalance_llep_spilled_rows_per_critical_path_payload_slot",
                    safe_ratio(status.llep_spilled_rows,
                               status.payload_bucket_requested_slots));
        emit_double("device_rebalance_router_hot_cache_spread_improvement_ratio",
                    safe_ratio(status.router_hot_cache_load_spread_improvement_total,
                               status.router_hot_cache_default_load_spread_total));
        emit_double("device_rebalance_router_hot_cache_avg_spread_improvement_per_used_dispatch",
                    safe_ratio(status.router_hot_cache_load_spread_improvement_total,
                               status.router_hot_cache_used_dispatches));
        emit_double("device_rebalance_router_hot_cache_spread_improvement_per_requested_payload_slot",
                    safe_ratio(status.router_hot_cache_load_spread_improvement_total,
                               status.payload_bucket_requested_slots));
        emit_double("device_rebalance_router_hot_cache_miss_ratio",
                    safe_ratio(status.router_hot_cache_miss_dispatches,
                               status.router_hot_cache_active_dispatches));
        emit_double("device_rebalance_router_hot_cache_selected_replica_slot_ratio",
                    safe_ratio(status.router_hot_cache_replicated_selected_expert_slots,
                               status.router_hot_cache_selected_expert_slots));
        const uint32_t participant_count =
            std::min<uint32_t>(params.config.participant_count,
                               kDeviceMoEMaxParticipants);
        for (uint32_t participant = 0; participant < participant_count; ++participant)
        {
            auto load_tags = tags;
            load_tags["participant"] = std::to_string(participant);
            load_tags["policy_phase"] = "pre";
            PerfStatsCollector::addCounterWithSequence(
                "moe_rebalance",
                "device_rebalance_policy_participant_load",
                static_cast<double>(
                    status.pre_policy_participant_load[participant]),
                {cache.launch_count, status.last_epoch},
                "decode",
                device_key,
                load_tags);

            load_tags["policy_phase"] = "post";
            PerfStatsCollector::addCounterWithSequence(
                "moe_rebalance",
                "device_rebalance_policy_participant_load",
                static_cast<double>(
                    status.post_policy_participant_load[participant]),
                {cache.launch_count, status.last_epoch},
                "decode",
                device_key,
                load_tags);

            load_tags["policy_phase"] = "delta";
            PerfStatsCollector::addCounterWithSequence(
                "moe_rebalance",
                "device_rebalance_policy_participant_load_delta",
                static_cast<double>(
                    status.post_policy_participant_load[participant]) -
                    static_cast<double>(
                        status.pre_policy_participant_load[participant]),
                {cache.launch_count, status.last_epoch},
                "decode",
                device_key,
                load_tags);
        }
        PerfStatsCollector::addCounterWithSequence(
                "moe_rebalance",
                "device_rebalance_policy_load_imbalance_delta",
                pre_ratio - post_ratio,
                {cache.launch_count, status.last_epoch},
                "decode",
                device_key,
                tags);

        if (export_trace)
        {
            DeviceMoERuntimeTable *trace_runtime_table =
                dynamic_cast<DeviceMoERuntimeTable *>(
                    params.moe_runtime_table);
            if (trace_runtime_table)
            {
                /* Trace export already owns the completed maintenance stream.
                 * This D2H is diagnostic-only and never participates in the
                 * placement or request lifecycle. */
                trace_runtime_table->syncRuntimeStateToHost(
                    maintenance_stream);
            }
            if (!appendDeviceMoERebalanceTraceJsonl(
                    trace_path,
                    device_key,
                    maintenance_tags,
                    workspace_name,
                    params,
                    status,
                    controller_state,
                    copy_status,
                    valid_copy_status,
                    apply_status,
                    command_headers,
                    wave_states,
                    plan_counts,
                    trace_plan_entries,
                    trace_plan_capacity,
                    transfer_cost,
                    transfer_copied_arrivals,
                    transfer_applied_arrivals,
                    transfer_payload_bucket_slots,
                    transfer_payload_edge_mask,
                    trace_gathered_histograms,
                    trace_histogram_layer_count,
                    trace_owner_participants,
                    trace_runtime_table))
            {
                return false;
            }
        }

        if (!export_perfstats)
            return true;

        if (publication_contract.apply_status)
        {
            emit_u32("device_rebalance_apply_status_code", apply_status.status_code);
            emit_u32("device_rebalance_apply_plan_entries_seen", apply_status.plan_entries_seen);
            emit_u32("device_rebalance_apply_applied_arrivals", apply_status.applied_arrivals);
            emit_u32("device_rebalance_apply_changed_layers", apply_status.changed_layers);
            emit_u32("device_rebalance_apply_copied_arrivals", apply_status.copied_arrivals);
            emit_u32("device_rebalance_apply_copy_incomplete", apply_status.copy_incomplete);
            emit_u32("device_rebalance_apply_required_local_arrivals",
                     apply_status.required_local_arrivals);
            emit_u32("device_rebalance_apply_ready_local_arrivals",
                     apply_status.ready_local_arrivals);
            emit_u32("device_rebalance_apply_post_apply_multi_resident_experts",
                     apply_status.post_apply_multi_resident_experts);
            emit_u32("device_rebalance_apply_missing_source_descriptors",
                     apply_status.missing_source_descriptors);
            emit_u32("device_rebalance_apply_missing_destination_slots",
                     apply_status.missing_destination_slots);
            emit_u32("device_rebalance_apply_descriptor_mismatches",
                     apply_status.descriptor_mismatches);
        }
        if (publication_contract.copy_status)
        {
            emit_u32("device_rebalance_copy_status_valid", valid_copy_status ? 1u : 0u);
            emit_u32("device_rebalance_copy_transaction_wave_index",
                     copy_status.transaction_wave_index);
            emit_u32("device_rebalance_copy_transaction_epoch",
                     copy_status.transaction_epoch);
            emit_u32("device_rebalance_copy_transaction_command_count",
                     copy_status.transaction_command_count);
            emit_u32("device_rebalance_copy_plan_entries_seen", copy_status.plan_entries_seen);
            emit_u32("device_rebalance_copy_copied_arrivals", copy_status.copied_arrivals);
            emit_u32("device_rebalance_copy_required_local_arrivals",
                     copy_status.required_local_arrivals);
            emit_u32("device_rebalance_copy_ready_local_arrivals",
                     copy_status.ready_local_arrivals);
            emit_u32("device_rebalance_copy_missing_source_descriptors",
                     copy_status.missing_source_descriptors);
            emit_u32("device_rebalance_copy_missing_destination_slots",
                     copy_status.missing_destination_slots);
            emit_u32("device_rebalance_copy_descriptor_mismatches",
                     copy_status.descriptor_mismatches);
            emit_u32("device_rebalance_copy_invalid_plan_entries",
                     copy_status.invalid_plan_entries);
        }

        emit_u32("device_rebalance_controller_maintenance_launches",
                 controller_state.maintenance_launches);
        emit_u32("device_rebalance_controller_decode_apply_polls",
                 controller_state.decode_apply_polls);
        emit_u32("device_rebalance_controller_decode_apply_hits",
                 controller_state.decode_apply_hits);
        emit_u32("device_rebalance_controller_last_error_code",
                 controller_state.last_error_code);
        emit_u32("device_rebalance_controller_last_error_wave_index",
                 controller_state.last_error_wave_index);
        emit_u32("device_rebalance_controller_last_error_epoch",
                 controller_state.last_error_epoch);
        emit_u32("device_rebalance_controller_last_error_expected_arrivals",
                 controller_state.last_error_expected_arrivals);
        emit_u32("device_rebalance_controller_last_error_copied_arrivals",
                 controller_state.last_error_copied_arrivals);
        emit_u32("device_rebalance_controller_last_error_copy_status_code",
                 controller_state.last_error_copy_status_code);
        emit_u32("device_rebalance_controller_last_error_copy_failure_flags",
                 controller_state.last_error_copy_failure_flags);
        emit_u32("device_rebalance_controller_last_error_copy_plan_entries_seen",
                 controller_state.last_error_copy_plan_entries_seen);
        emit_u32(
            "device_rebalance_controller_last_error_copy_skipped_wrong_destination",
            controller_state
                .last_error_copy_skipped_wrong_destination);
        emit_u32(
            "device_rebalance_controller_last_error_missing_destination_slot",
            controller_state.last_error_missing_destination_slot);
        emit_u32(
            "device_rebalance_controller_last_error_missing_destination_layer",
            controller_state.last_error_missing_destination_layer);
        emit_u32(
            "device_rebalance_controller_last_error_missing_destination_expert",
            controller_state.last_error_missing_destination_expert);
        emit_u32(
            "device_rebalance_controller_last_error_missing_destination_source",
            controller_state.last_error_missing_destination_source);
        emit_u32(
            "device_rebalance_controller_last_error_local_transfer_slot_count",
            controller_state.last_error_local_transfer_slot_count);

        for (uint32_t i = 0; i < command_buffer_count; ++i)
        {
            auto slot_tags = tags;
            slot_tags["buffer_slot"] = std::to_string(i);
            slot_tags["command_capacity"] =
                std::to_string(command_headers[i].command_capacity);
            PerfStatsCollector::addCounterWithSequence(
                "moe_rebalance",
                "device_rebalance_command_count",
                static_cast<double>(command_headers[i].command_count),
                {cache.launch_count, status.last_epoch, command_headers[i].epoch, wave_states[i].epoch},
                "decode",
                device_key,
                slot_tags);
            PerfStatsCollector::addCounterWithSequence(
                "moe_rebalance",
                "device_rebalance_plan_count",
                static_cast<double>(plan_counts[i]),
                {cache.launch_count, status.last_epoch, command_headers[i].epoch, wave_states[i].epoch},
                "decode",
                device_key,
                slot_tags);
            PerfStatsCollector::addCounterWithSequence(
                "moe_rebalance",
                "device_rebalance_wave_planned_layer_count",
                static_cast<double>(wave_states[i].planned_layer_count),
                {cache.launch_count, status.last_epoch, command_headers[i].epoch, wave_states[i].epoch},
                "decode",
                device_key,
                slot_tags);
        }

        for (uint32_t i = 0; i < controller_state.wave_count && i < 2u; ++i)
        {
            const auto &wave = controller_state.waves[i];
            auto wave_tags = tags;
            wave_tags["wave"] = std::to_string(i);
            wave_tags["wave_state"] = std::to_string(wave.state);
            wave_tags["wave_error_code"] = std::to_string(wave.error_code);
            PerfStatsCollector::addCounterWithSequence(
                "moe_rebalance",
                "device_rebalance_wave_command_count",
                static_cast<double>(wave.command_count),
                {cache.launch_count, status.last_epoch, wave.epoch},
                "decode",
                device_key,
                wave_tags);
            PerfStatsCollector::addCounterWithSequence(
                "moe_rebalance",
                "device_rebalance_wave_copied_arrivals",
                static_cast<double>(wave.copied_arrivals),
                {cache.launch_count, status.last_epoch, wave.epoch},
                "decode",
                device_key,
                wave_tags);
            PerfStatsCollector::addCounterWithSequence(
                "moe_rebalance",
                "device_rebalance_wave_applied_arrivals",
                static_cast<double>(wave.applied_arrivals),
                {cache.launch_count, status.last_epoch, wave.epoch},
                "decode",
                device_key,
                wave_tags);
            PerfStatsCollector::addCounterWithSequence(
                "moe_rebalance",
                "device_rebalance_wave_applied_layer_count",
                static_cast<double>(wave.applied_layer_count),
                {cache.launch_count, status.last_epoch, wave.epoch},
                "decode",
                device_key,
                wave_tags);
            PerfStatsCollector::addCounterWithSequence(
                "moe_rebalance",
                "device_rebalance_wave_payload_bucket_requested_slots",
                static_cast<double>(wave.requested_payload_slots),
                {cache.launch_count, status.last_epoch, wave.epoch},
                "decode",
                device_key,
                wave_tags);
            PerfStatsCollector::addCounterWithSequence(
                "moe_rebalance",
                "device_rebalance_wave_payload_bucket_slots",
                static_cast<double>(wave.payload_bucket_slots),
                {cache.launch_count, status.last_epoch, wave.epoch},
                "decode",
                device_key,
                wave_tags);
            PerfStatsCollector::addCounterWithSequence(
                "moe_rebalance",
                "device_rebalance_wave_payload_bucket_index",
                static_cast<double>(wave.payload_bucket_index),
                {cache.launch_count, status.last_epoch, wave.epoch},
                "decode",
                device_key,
                wave_tags);
            PerfStatsCollector::addCounterWithSequence(
                "moe_rebalance",
                "device_rebalance_wave_payload_bucket_overflow",
                static_cast<double>(wave.payload_bucket_overflow),
                {cache.launch_count, status.last_epoch, wave.epoch},
                "decode",
                device_key,
                wave_tags);
        }
        emit_u64("device_rebalance_wave_command_count_total",
                 wave_command_count_total);
        emit_u64("device_rebalance_wave_copied_arrivals_total",
                 wave_copied_arrivals_total);
        emit_u64("device_rebalance_wave_applied_arrivals_total",
                 wave_applied_arrivals_total);
        emit_u64("device_rebalance_wave_applied_layer_count_total",
                 wave_applied_layer_count_total);

        cache.last_status_publication_launch_count = cache.launch_count;
        return true;
    }

    void DeviceGraphOrchestrator::drainCompletedDeviceMoERebalanceMaintenanceDiagnostics(
        const char *boundary,
        const char *reset_operation)
    {
        auto &cache = device_moe_rebalance_maintenance_graph_;
        const auto has_unpublished_status =
            [](const DeviceMoERebalanceMaintenanceGraphCache &target_cache)
        {
            return target_cache.launch_count >
                   target_cache.last_status_publication_launch_count;
        };
        if (!cache.completion_event_in_flight &&
            !has_unpublished_status(cache))
        {
            return;
        }

        const std::string device_key = state_.device_id.toString();
        const std::string boundary_tag =
            boundary && boundary[0] ? boundary : "request_epilogue";
        IWorkerGPUContext *gpu_ctx = nullptr;
        try
        {
            gpu_ctx = &GPUDeviceContextPool::instance().getContext(state_.device_id);
        }
        catch (const std::exception &e)
        {
            throw std::runtime_error(
                "Device MoE rebalance diagnostic epilogue could not resolve GPU context for " +
                device_key + ": " + e.what());
        }
        if (!gpu_ctx)
        {
            throw std::runtime_error(
                "Device MoE rebalance diagnostic epilogue resolved a null GPU context for " +
                device_key);
        }

        PerfStatsCollector::Tags epilogue_tags = {
            {"window", boundary_tag},
            {"scheduling_authority", "device_commit_boundary"},
            {"diagnostic_boundary", boundary_tag}};
        if (reset_operation && reset_operation[0])
        {
            /*
             * The maintenance result remains device-owned until this explicit
             * request epilogue. Publishing the reset name as a diagnostic tag
             * does not create a host execution dependency; it only identifies
             * which already-completed lifecycle boundary authorized the final
             * D2H status readback.
             */
            epilogue_tags["reset"] = reset_operation;
        }

        auto drain_one =
            [&](DeviceMoERebalanceMaintenanceGraphCache &target_cache,
                const char *kind)
        {
            const bool status_unpublished =
                has_unpublished_status(target_cache);
            if (!target_cache.completion_event_in_flight &&
                !status_unpublished)
            {
                return;
            }
            if (!target_cache.graph)
            {
                throw std::runtime_error(
                    std::string("Device MoE rebalance diagnostic epilogue found pending ") +
                    kind + " maintenance status without a graph on " + device_key);
            }
            if (!target_cache.completion_event)
            {
                throw std::runtime_error(
                    std::string("Device MoE rebalance diagnostic epilogue found pending ") +
                    kind + " maintenance without a completion event on " + device_key);
            }
            void *maintenance_stream = target_cache.segment_cache.capture_stream;
            if (!maintenance_stream)
            {
                throw std::runtime_error(
                    std::string("Device MoE rebalance diagnostic epilogue requires an explicit ") +
                    kind + " maintenance stream on " + device_key);
            }

            if (target_cache.completion_event_in_flight)
            {
                bool ready = false;
                if (!gpu_ctx->queryEventChecked(
                        target_cache.completion_event.get(),
                        ready))
                {
                    throw std::runtime_error(
                        std::string("Device MoE rebalance diagnostic epilogue failed to query ") +
                        kind + " maintenance completion event on " + device_key);
                }
                auto readiness_tags = epilogue_tags;
                readiness_tags["maintenance_graph_kind"] = kind;
                readiness_tags["ready_before_readback"] = boolTag(ready);
                PerfStatsCollector::addCounter(
                    "moe_rebalance",
                    "device_maintenance_graph_epilogue_event_queries",
                    1.0,
                    "decode",
                    device_key,
                    std::move(readiness_tags));

                /*
                 * An unfinished terminal event is expected when the host
                 * reaches a request epilogue before the dedicated maintenance
                 * stream. Do not synchronize here. The export below enqueues
                 * every diagnostic D2H after the graph on this exact stream
                 * and performs one stream wait after the final copy. That
                 * single boundary both surfaces asynchronous backend errors
                 * and makes all copied status records host-readable.
                 */
            }

            auto tags = epilogue_tags;
            tags["maintenance_graph_kind"] = kind;
            DeviceMoERebalanceMaintenanceOutcome outcome{};
            if (!exportCompletedDeviceMoERebalanceMaintenanceStats(
                    target_cache,
                    maintenance_stream,
                    device_key,
                    tags,
                    &outcome))
            {
                throw std::runtime_error(
                    std::string("Device MoE rebalance diagnostic epilogue failed to export ") +
                    kind + " maintenance diagnostics on " + device_key);
            }

            if (outcome.hasFatalError())
            {
                std::ostringstream error;
                error
                    << "Device MoE rebalance maintenance violated a fatal publication invariant"
                    << " on " << device_key
                    << " at " << boundary_tag
                    << " graph_kind=" << kind
                    << " status_valid=" << outcome.valid
                    << " status_code=" << outcome.status_code
                    << " invalid_runtime_layers=" << outcome.invalid_runtime_layers
                    << " first_invalid_runtime_layer="
                    << outcome.first_invalid_runtime_layer
                    << " first_invalid_runtime_active_bank="
                    << outcome.first_invalid_runtime_active_bank
                    << " first_invalid_runtime_active_epoch="
                    << outcome.first_invalid_runtime_active_epoch
                    << " first_invalid_runtime_expert_count="
                    << outcome.first_invalid_runtime_expert_count
                    << " first_invalid_runtime_participant_id="
                    << outcome.first_invalid_runtime_participant_id
                    << " first_invalid_runtime_participant_count="
                    << outcome.first_invalid_runtime_participant_count
                    << " plan_overflow=" << outcome.plan_overflow
                    << " capacity_limited_candidates="
                    << outcome.capacity_limited_candidates
                    << " payload_bucket_overflow=" << outcome.payload_bucket_overflow
                    << " transfer_slot_active_claims="
                    << outcome.prefill_active_transfer_slot_experts
                    << " transfer_slot_unique_claims="
                    << outcome.prefill_unique_transfer_slot_claims
                    << " transfer_slot_duplicate_claims="
                    << outcome.prefill_duplicate_transfer_slot_claims
                    << " transfer_slot_invalid_claims="
                    << outcome.prefill_invalid_transfer_slot_claims
                    << " transfer_slot_max="
                    << outcome.prefill_max_transfer_slot
                    << " transfer_slot_max_layer="
                    << outcome.prefill_max_transfer_slot_layer
                    << " transfer_slot_max_expert="
                    << outcome.prefill_max_transfer_slot_expert
                    << " transfer_slot_capacity="
                    << outcome.local_transfer_slot_count
                    << " transfer_slot_first_duplicate="
                    << outcome.prefill_first_duplicate_transfer_slot
                    << " transfer_slot_first_duplicate_layer="
                    << outcome.prefill_first_duplicate_layer
                    << " transfer_slot_first_duplicate_expert="
                    << outcome.prefill_first_duplicate_expert
                    << " transfer_slot_first_invalid="
                    << outcome.prefill_first_invalid_transfer_slot
                    << " transfer_slot_first_invalid_layer="
                    << outcome.prefill_first_invalid_layer
                    << " transfer_slot_first_invalid_expert="
                    << outcome.prefill_first_invalid_expert
                    << " transfer_slot_first_invalid_reasons="
                    << outcome.prefill_first_invalid_reasons
                    << " transfer_slot_first_invalid_flags=0x"
                    << std::hex
                    << outcome.prefill_first_invalid_flags
                    << " transfer_slot_first_invalid_resident_mask=0x"
                    << outcome.prefill_first_invalid_resident_mask
                    << std::dec
                    << " transfer_slot_first_invalid_owner="
                    << outcome.prefill_first_invalid_owner
                    << " controller_valid=" << outcome.controller_valid
                    << " controller_error=" << outcome.controller_last_error_code
                    << " controller_error_waves=" << outcome.controller_error_waves
                    << " controller_error_wave="
                    << outcome.controller_last_error_wave_index
                    << " controller_error_epoch="
                    << outcome.controller_last_error_epoch
                    << " controller_error_participant="
                    << outcome.controller_last_error_participant
                    << " controller_error_expected_arrivals="
                    << outcome.controller_last_error_expected_arrivals
                    << " controller_error_copied_arrivals="
                    << outcome.controller_last_error_copied_arrivals
                    << " controller_error_copy_status="
                    << outcome.controller_last_error_copy_status_code
                    << " controller_error_copy_flags="
                    << outcome.controller_last_error_copy_failure_flags
                    << " controller_error_copy_plan_seen="
                    << outcome.controller_last_error_copy_plan_entries_seen
                    << " controller_error_copy_skipped_wrong_destination="
                    << outcome
                           .controller_last_error_copy_skipped_wrong_destination
                    << " controller_error_missing_destination_slot="
                    << outcome
                           .controller_last_error_missing_destination_slot
                    << " controller_error_missing_destination_layer="
                    << outcome
                           .controller_last_error_missing_destination_layer
                    << " controller_error_missing_destination_expert="
                    << outcome
                           .controller_last_error_missing_destination_expert
                    << " controller_error_missing_destination_source="
                    << outcome
                           .controller_last_error_missing_destination_source
                    << " controller_error_local_transfer_slot_count="
                    << outcome
                           .controller_last_error_local_transfer_slot_count
                    << " copy_present=" << outcome.copy_status_present
                    << " copy_valid=" << outcome.copy_status_valid
                    << " copy_status=" << outcome.copy_status_code
                    << " copy_transaction_wave="
                    << outcome.copy_transaction_wave_index
                    << " copy_transaction_epoch="
                    << outcome.copy_transaction_epoch
                    << " copy_transaction_command_count="
                    << outcome.copy_transaction_command_count
                    << " copy_plan_seen=" << outcome.copy_plan_entries_seen
                    << " copy_copied_arrivals="
                    << outcome.copy_copied_arrivals
                    << " copy_skipped_wrong_destination="
                    << outcome.copy_skipped_wrong_destination
                    << " copy_invalid_plan=" << outcome.copy_invalid_plan_entries
                    << " copy_missing_source=" << outcome.copy_missing_source_descriptors
                    << " copy_missing_destination=" << outcome.copy_missing_destination_slots
                    << " copy_mismatch=" << outcome.copy_descriptor_mismatches
                    << " copy_incomplete=" << outcome.copy_incomplete
                    << " copy_required=" << outcome.copy_required_local_arrivals
                    << " copy_ready=" << outcome.copy_ready_local_arrivals
                    << " apply_present=" << outcome.apply_status_present
                    << " apply_valid=" << outcome.apply_status_valid
                    << " apply_status=" << outcome.apply_status_code
                    << " apply_invalid_plan=" << outcome.apply_invalid_plan_entries
                    << " apply_missing_source=" << outcome.apply_missing_source_descriptors
                    << " apply_missing_destination=" << outcome.apply_missing_destination_slots
                    << " apply_mismatch=" << outcome.apply_descriptor_mismatches
                    << " apply_incomplete=" << outcome.apply_copy_incomplete
                    << " apply_required=" << outcome.apply_required_local_arrivals
                    << " apply_ready=" << outcome.apply_ready_local_arrivals;
                throw std::runtime_error(error.str());
            }

            LOG_INFO("[DGO] Device MoE rebalance diagnostic epilogue"
                     << " device=" << device_key
                     << " boundary=" << boundary_tag
                     << " graph_kind=" << kind
                     << " launch_count=" << target_cache.launch_count
                     << " valid=" << outcome.valid
                     << " useful_work=" << outcome.useful_work
                     << " status_code=" << outcome.status_code
                     << " windows_applied=" << outcome.windows_applied
                     << " selected_replicas=" << outcome.selected_replicas
                     << " planned_arrivals=" << outcome.planned_arrivals
                     << " runtime_changed_layers=" << outcome.runtime_changed_layers
                     << " runtime_applied_arrivals=" << outcome.runtime_applied_arrivals
                     << " prefill_current_batch_movement_layers="
                     << outcome.prefill_current_batch_movement_layers
                     << " prefill_active_transfer_slot_experts="
                     << outcome.prefill_active_transfer_slot_experts
                     << " payload_bucket_slots=" << outcome.payload_bucket_slots
                     << " payload_edge_mask=" << outcome.payload_edge_mask);

            target_cache.completion_event_in_flight = false;
            PerfStatsCollector::addCounter(
                "moe_rebalance",
                "device_maintenance_graph_diagnostic_exports",
                1.0,
                "decode",
                device_key,
                tags);
            if (boundary_tag == "request_reset")
            {
                PerfStatsCollector::addCounter(
                    "moe_rebalance",
                    "device_maintenance_graph_request_reset_exports",
                    1.0,
                    "decode",
                    device_key,
                    tags);
            }
        };

        drain_one(cache, "atomic");
    }

    void DeviceGraphOrchestrator::
        publishTerminalCurrentBatchLLEPEvidenceDiagnostics()
    {
        using namespace sampling_math;
        if (!PerfStatsCollector::isDomainEnabled("moe_rebalance") ||
            !state_.device_id.is_gpu() ||
            !graph_builder_)
        {
            return;
        }

        const DeviceMoECurrentBatchLLEPEvidenceSource source =
            graph_builder_->deviceMoECurrentBatchLLEPEvidenceSource(
                state_.device_id);
        if (!source.valid())
            return;
        if (!device_generation_storage_.control_device ||
            device_generation_storage_.control_stride <
                kDeviceGenerationControlCount)
        {
            throw std::runtime_error(
                "Terminal current-batch LLEP evidence has no persistent "
                "device generation-control row on " +
                state_.device_id.toString());
        }

        IBackend *const backend = getBackendFor(state_.device_id);
        if (!backend)
        {
            throw std::runtime_error(
                "Terminal current-batch LLEP evidence could not resolve the "
                "GPU backend for " +
                state_.device_id.toString());
        }

        IWorkerGPUContext &gpu_context =
            GPUDeviceContextPool::instance().getContext(state_.device_id);
        void *const diagnostic_stream =
            gpu_context.getOrCreateAuxiliaryStream(
                "moe_current_batch_llep_diagnostics");
        if (!diagnostic_stream)
        {
            throw std::runtime_error(
                "Terminal current-batch LLEP evidence could not acquire its "
                "named diagnostics stream on " +
                state_.device_id.toString());
        }
        if (!waitForForwardGraphOutputReady(
                diagnostic_stream,
                DeviceTimelineRole::Diagnostics,
                ForwardGraphOutputKind::Any,
                "current_batch_llep_snapshot_evidence"))
        {
            throw std::runtime_error(
                "Terminal current-batch LLEP evidence could not consume the "
                "latest forward publication on " +
                state_.device_id.toString());
        }
        if (!backend->
                enqueuePublishMoECurrentBatchLLEPEvidenceToGenerationControl(
                    source.runtime_layers_device,
                    source.layer_count,
                    device_generation_storage_.control_device,
                    device_generation_storage_.control_stride,
                    /*request_count=*/1,
                    state_.device_id.gpu_ordinal(),
                    diagnostic_stream))
        {
            throw std::runtime_error(
                "Terminal current-batch LLEP evidence reduction failed on " +
                state_.device_id.toString());
        }

        std::array<int, 2> evidence{};
        const bool trace_assignment_runtime =
            !DebugEnv::isFalseyEnv("LLAMINAR_MOE_PREFILL_ASSIGNMENT_TRACE");
        std::vector<DeviceMoELayerRuntime> traced_runtime_layers;
        if (trace_assignment_runtime)
        {
            /*
             * Stage-local assignment tracing is deliberately deferred while a
             * native CUDA/HIP graph is being recorded.  Complete that explicit
             * diagnostic request here, after the latest forward publication has
             * been consumed and outside captured inference.  Reading the full
             * table lets one observation distinguish a planner that published no
             * spans from a consumer marker that was never set or was later
             * overwritten.  Ordinary inference never allocates or transfers this
             * diagnostic vector.
             */
            traced_runtime_layers.resize(
                static_cast<size_t>(source.layer_count));
            if (!backend->deviceToHostOnStream(
                    traced_runtime_layers.data(),
                    source.runtime_layers_device,
                    traced_runtime_layers.size() *
                        sizeof(DeviceMoELayerRuntime),
                    state_.device_id.gpu_ordinal(),
                    diagnostic_stream))
            {
                throw std::runtime_error(
                    "Terminal current-batch LLEP runtime trace readback failed on " +
                    state_.device_id.toString());
            }
        }
        const int *const evidence_device =
            device_generation_storage_.control_device +
            kDeviceGenerationControlCurrentBatchLLEPMovementLayerCount;
        if (!backend->deviceToHostOnStream(
                evidence.data(),
                evidence_device,
                sizeof(evidence),
                state_.device_id.gpu_ordinal(),
                diagnostic_stream))
        {
            throw std::runtime_error(
                "Terminal current-batch LLEP evidence readback enqueue failed on " +
                state_.device_id.toString());
        }
        completeMTPDiagnosticObservation(
            *backend,
            state_.device_id,
            diagnostic_stream,
            "terminal current-batch LLEP evidence readback");

        const int movement_layers = evidence[0];
        const int non_owner_assignment_layers = evidence[1];
        if (movement_layers < 0 || movement_layers > source.layer_count ||
            non_owner_assignment_layers < 0 ||
            non_owner_assignment_layers > source.layer_count)
        {
            throw std::runtime_error(
                "Terminal current-batch LLEP evidence contains an invalid "
                "layer count on " +
                state_.device_id.toString());
        }

        for (size_t layer = 0; layer < traced_runtime_layers.size(); ++layer)
        {
            const DeviceMoELayerRuntime &runtime =
                traced_runtime_layers[layer];
            const uint32_t active_bank = runtime.active_bank;
            const uint32_t transient_placement =
                active_bank <= 1u
                    ? runtime.banks[active_bank].transient_placement_observed
                    : 0u;
            LOG_INFO(
                "[DeviceGraphOrchestrator] current-batch LLEP terminal runtime trace"
                << " device=" << state_.device_id.toString()
                << " layer=" << layer
                << " active_bank=" << active_bank
                << " active_epoch=" << runtime.active_epoch
                << " participant_id=" << runtime.participant_id
                << " participant_count=" << runtime.participant_count
                << " span_count=" << runtime.reserved_u64[2]
                << " transfer_count=" << runtime.reserved_u64[3]
                << " movement_observed="
                << runtime.current_batch_llep_movement_observed
                << " non_owner_assignment_observed="
                << runtime.current_batch_llep_non_owner_assignment_observed
                << " transient_placement_observed="
                << transient_placement);
        }

        const PerfStatsCollector::Tags tags{
            {"source", "terminal_diagnostic_control"},
            {"diagnostic_only", "true"},
            {"layer_count", std::to_string(source.layer_count)},
            {"expert_payload_slot_bytes",
             std::to_string(source.expert_payload_slot_bytes)}};
        const auto request_movement =
            deviceMoERebalanceRequestMovementEvidence(
                /*waves=*/{},
                static_cast<uint32_t>(movement_layers),
                source.expert_payload_slot_bytes);
        recordCompletedDeviceMoEMovement(request_movement, "prefill", state_.device_id.toString(), "terminal_diagnostic_control");
        PerfStatsCollector::addCounter(
            "moe_rebalance",
            "device_rebalance_prefill_current_batch_movement_layers",
            static_cast<double>(movement_layers),
            "prefill",
            state_.device_id.toString(),
            tags);
        PerfStatsCollector::addCounter(
            "moe_rebalance",
            "device_rebalance_prefill_current_batch_non_owner_assignment_layers",
            static_cast<double>(non_owner_assignment_layers),
            "prefill",
            state_.device_id.toString(),
            tags);
        PerfStatsCollector::addCounter(
            "moe_rebalance",
            "device_rebalance_request_copied_payload_lower_bound",
            static_cast<double>(
                request_movement.copied_payload_lower_bound),
            "prefill",
            state_.device_id.toString(),
            tags);
        PerfStatsCollector::addCounter(
            "moe_rebalance",
            "device_rebalance_request_applied_payload_lower_bound",
            static_cast<double>(
                request_movement.applied_payload_lower_bound),
            "prefill",
            state_.device_id.toString(),
            tags);
        PerfStatsCollector::addCounter(
            "moe_rebalance",
            "device_rebalance_request_useful_payload_bytes_lower_bound",
            static_cast<double>(
                request_movement.useful_payload_bytes_lower_bound),
            "prefill",
            state_.device_id.toString(),
            tags);
    }

    void DeviceGraphOrchestrator::
        drainCompletedDecodeBoundaryMaintenanceDiagnostics()
    {
        drainCompletedDeviceMoERebalanceMaintenanceDiagnostics(
            "request_epilogue");
        publishTerminalCurrentBatchLLEPEvidenceDiagnostics();
    }

    const char *DeviceGraphOrchestrator::moeOverlayEpochLeaseStateName(
        MoEOverlayEpochLeaseState state) noexcept
    {
        switch (state)
        {
        case MoEOverlayEpochLeaseState::Idle:
            return "idle";
        case MoEOverlayEpochLeaseState::ExternalReader:
            return "external_reader";
        case MoEOverlayEpochLeaseState::ExternalSequence:
            return "external_sequence";
        case MoEOverlayEpochLeaseState::HostedParent:
            return "hosted_parent";
        case MoEOverlayEpochLeaseState::CapturedHostedParent:
            return "captured_hosted_parent";
        case MoEOverlayEpochLeaseState::CapturedMainForward:
            return "captured_main_forward";
        case MoEOverlayEpochLeaseState::ExternalForward:
            return "external_forward";
        case MoEOverlayEpochLeaseState::HostedChildForward:
            return "hosted_child_forward";
        case MoEOverlayEpochLeaseState::ReleasePublished:
            return "release_published";
        }
        return "invalid";
    }

    DeviceGraphOrchestrator::MoEOverlayExternalSequenceGraphLease::
        ~MoEOverlayExternalSequenceGraphLease() noexcept
    {
        if (!active())
            return;
        if (!finish(/*execution_succeeded=*/false))
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Failed to abort an unfinished "
                "ExpertOverlay execution-sequence graph at "
                << (consumer_name_ ? consumer_name_ : "unnamed"));
            std::terminate();
        }
    }

    bool DeviceGraphOrchestrator::MoEOverlayExternalSequenceGraphLease::admit(
        DeviceGraphOrchestrator &owner,
        const MoEOverlayInferenceParticipantGraphBinding &binding,
        void *execution_stream,
        DeviceTimelineRole consumer_role,
        const char *consumer_name)
    {
        if (active() || owner_ || binding_ || execution_stream_)
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] ExpertOverlay sequence graph lease "
                "cannot be admitted more than once");
            return false;
        }
        if (!binding.active)
        {
            finished_ = true;
            return true;
        }
        if (!owner.admitMoEOverlayExternalSequenceGraph(
                binding,
                execution_stream,
                consumer_role,
                consumer_name))
        {
            return false;
        }
        owner_ = &owner;
        binding_ = &binding;
        execution_stream_ = execution_stream;
        consumer_name_ = consumer_name;
        finished_ = false;
        return true;
    }

    bool DeviceGraphOrchestrator::MoEOverlayExternalSequenceGraphLease::finish(
        bool execution_succeeded) noexcept
    {
        if (!active())
            return true;

        bool ok = false;
        try
        {
            ok = owner_->finishMoEOverlayExternalSequenceGraph(
                *binding_,
                execution_stream_,
                execution_succeeded,
                consumer_name_);
        }
        catch (const std::exception &exception)
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] ExpertOverlay execution-sequence "
                "graph terminal threw at "
                << (consumer_name_ ? consumer_name_ : "unnamed")
                << ": " << exception.what());
            return false;
        }
        catch (...)
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] ExpertOverlay execution-sequence "
                "graph terminal threw an unknown exception at "
                << (consumer_name_ ? consumer_name_ : "unnamed"));
            return false;
        }

        if (ok)
            finished_ = true;
        return ok;
    }

    bool DeviceGraphOrchestrator::
        initializeMoEOverlayEpochExecutionBinding()
    {
        const auto initial_lease_state =
            moe_overlay_epoch_lease_lifecycle_->load();
        if ((initial_lease_state != MoEOverlayEpochLeaseState::Idle &&
             initial_lease_state !=
                 MoEOverlayEpochLeaseState::ReleasePublished) ||
            moe_overlay_external_sequence_identity_.valid() ||
            moe_overlay_epoch_forward_submission_stream_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] ExpertOverlay epoch topology cannot change while a request lease is active"
                      << " lease_state="
                      << moeOverlayEpochLeaseStateName(initial_lease_state));
            return false;
        }

        DeviceMoEOverlayEpochExecutionBinding binding;
        try
        {
            if (state_.device_id.is_gpu() && graph_builder_)
            {
                binding =
                    graph_builder_->deviceMoEOverlayEpochExecutionBinding(
                        state_.device_id);
            }
        }
        catch (const std::exception &exception)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] ExpertOverlay epoch binding resolution failed on "
                      << state_.device_id.toString() << ": "
                      << exception.what());
            return false;
        }

        const auto retire_pending_release_before_identity_change = [&]() -> bool
        {
            auto submission =
                moe_overlay_epoch_lease_lifecycle_->beginSubmission();
            if (submission.state() !=
                MoEOverlayEpochLeaseState::ReleasePublished)
            {
                return submission.state() ==
                       MoEOverlayEpochLeaseState::Idle;
            }
            IBackend *const backend = getBackendFor(state_.device_id);
            if (!backend || !moe_overlay_epoch_released_event_ ||
                !submission.publishedReleaseProducerStream() ||
                !backend->waitForEvent(
                    moe_overlay_epoch_released_event_.get(),
                    state_.device_id.gpu_ordinal()))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] ExpertOverlay epoch identity cannot change before its prior release event retires on "
                          << state_.device_id.toString());
                return false;
            }
            if (!submission.commit(MoEOverlayEpochLeaseState::Idle))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] ExpertOverlay epoch identity retirement lost release ownership"
                          << " observed_state="
                          << moeOverlayEpochLeaseStateName(
                                 submission.state()));
                return false;
            }
            return true;
        };

        if (!binding)
        {
            if (!retire_pending_release_before_identity_change())
                return false;
            moe_overlay_finite_placement_graphs_.invalidate();
            moe_overlay_epoch_acquire_graph_.invalidate();
            moe_overlay_epoch_release_graph_.invalidate();
            moe_overlay_epoch_acquired_event_.reset();
            moe_overlay_epoch_released_event_.reset();
            moe_overlay_epoch_acquire_producer_stream_ = nullptr;
            moe_overlay_external_sequence_identity_.reset();
            moe_overlay_epoch_forward_submission_stream_ = nullptr;
            moe_overlay_epoch_execution_binding_ = {};
            return true;
        }
        if (!state_.device_id.is_gpu() ||
            binding.arena->deviceId() != state_.device_id ||
            binding.request_slot >= binding.arena->requestSlotCapacity())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] ExpertOverlay epoch binding has mismatched device or request-slot topology on "
                      << state_.device_id.toString());
            return false;
        }

        const bool identity_changed =
            moe_overlay_epoch_execution_binding_.arena != binding.arena ||
            moe_overlay_epoch_execution_binding_.request_slot !=
                binding.request_slot;
        if (identity_changed)
        {
            if (!retire_pending_release_before_identity_change())
                return false;
            invalidateSparseQueryParents();
            moe_overlay_finite_placement_graphs_.invalidate();
            moe_overlay_epoch_acquire_graph_.invalidate();
            moe_overlay_epoch_release_graph_.invalidate();
            moe_overlay_epoch_acquired_event_.reset();
            moe_overlay_epoch_released_event_.reset();
            moe_overlay_epoch_acquire_producer_stream_ = nullptr;
            moe_overlay_external_sequence_identity_.reset();
            moe_overlay_epoch_forward_submission_stream_ = nullptr;
            moe_overlay_epoch_execution_binding_ = binding;
        }

        IBackend *const backend = getBackendFor(state_.device_id);
        if (!backend)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] ExpertOverlay epoch binding has no backend on "
                      << state_.device_id.toString());
            return false;
        }
        const int ordinal = state_.device_id.gpu_ordinal();
        const auto make_event = [&](std::shared_ptr<void> &owner,
                                    const char *edge_name) -> bool
        {
            if (owner)
                return true;
            void *const raw_event = backend->createEvent(ordinal);
            if (!raw_event)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Could not preallocate ExpertOverlay epoch "
                          << edge_name << " event on "
                          << state_.device_id.toString());
                return false;
            }
            owner.reset(
                raw_event,
                [backend, ordinal](void *event)
                {
                    if (event)
                        backend->destroyEvent(event, ordinal);
                });
            return true;
        };
        if (!make_event(
                moe_overlay_epoch_acquired_event_,
                "acquired") ||
            !make_event(
                moe_overlay_epoch_released_event_,
                "released"))
        {
            return false;
        }

        /*
         * The device workspace generation is published later by eager graph-
         * family materialization. Resolve ownership and preallocate events now;
         * build the two graph definitions only once that generation exists.
         */
        return true;
    }

    bool DeviceGraphOrchestrator::prepareMoEOverlayFinitePlacementGraphs()
    {
        if(!moe_overlay_epoch_execution_binding_ ||
           !moe_overlay_epoch_execution_binding_.retirement_readiness_controller ||
           workspaceGeneration(state_.device_id)==0u)return false;
        auto *context=getDeviceContext(state_.device_id);
        if(!context)return false;
        auto &worker=GPUDeviceContextPool::instance().getContext(state_.device_id);
        worker.submitAndWait([&] {
            moe_overlay_finite_placement_graphs_.prepare(executor_,*context,worker,
                moe_overlay_epoch_execution_binding_.arena,
                *moe_overlay_epoch_execution_binding_.retirement_readiness_controller,
                workspaceGeneration(state_.device_id),true);
        });
        return true;
    }

    bool DeviceGraphOrchestrator::runMoEOverlayFinitePlacementBoundary(
        DeviceMoEOverlayEpochParentAction action)
    {
        // Actual parent completion is observed inside the original graph cache,
        // independently of the background maintenance worker's reader wait.
        moe_overlay_finite_placement_graphs_.run(action,[&](void *stream) {
            return enqueueMoEOverlayDeviceInferenceBoundary(stream,
                {.purpose=MoEOverlayInferenceBoundaryPurpose::PlacementDecisionSnapshot})==
                MoEOverlayInferenceBoundaryStatus::Submitted;
        });
        return true;
    }

    bool DeviceGraphOrchestrator::materializeMoEOverlayEpochBoundaryGraphs()
    {
        if (!moe_overlay_epoch_execution_binding_)
            return true;

        const uint64_t generation = workspaceGeneration(state_.device_id);
        if (!state_.device_id.is_gpu() || generation == 0)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] ExpertOverlay epoch graphs require a finalized GPU workspace generation");
            return false;
        }

        const auto materialize_one =
            [&](MoEOverlayEpochBoundaryStage::Operation operation,
                MoEOverlayEpochBoundaryGraphCache &cache,
                const char *node_name) -> bool
        {
            MoEOverlayEpochBoundaryStage::Params params;
            params.device_id = state_.device_id;
            params.arena = moe_overlay_epoch_execution_binding_.arena;
            params.request_slot =
                moe_overlay_epoch_execution_binding_.request_slot;
            params.operation = operation;
            if (operation ==
                MoEOverlayEpochBoundaryStage::Operation::Release)
            {
                params.retirement_readiness_controller =
                    moe_overlay_epoch_execution_binding_
                        .retirement_readiness_controller;
            }
            params.stage_name = node_name;

            if (cache.valid && cache.graph && cache.stage &&
                cache.workspace_generation == generation &&
                cache.stage->hasSameCaptureIdentity(params))
            {
                return true;
            }

            std::unique_ptr<MoEOverlayEpochBoundaryStage> stage;
            try
            {
                stage =
                    std::make_unique<MoEOverlayEpochBoundaryStage>(params);
            }
            catch (const std::exception &exception)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Could not construct "
                          << node_name << ": " << exception.what());
                return false;
            }
            auto *const stage_view = stage.get();
            auto graph = std::make_unique<ComputeGraph>();
            graph->addNode(node_name, std::move(stage), state_.device_id);

            invalidateSparseQueryParents();
            cache.invalidate();
            cache.graph = std::move(graph);
            cache.stage = stage_view;
            cache.workspace_generation = generation;
            cache.valid = true;
            return true;
        };

        return materialize_one(
                   MoEOverlayEpochBoundaryStage::Operation::Acquire,
                   moe_overlay_epoch_acquire_graph_,
                   "moe_overlay_epoch_acquire") &&
               materialize_one(
                   MoEOverlayEpochBoundaryStage::Operation::Release,
                   moe_overlay_epoch_release_graph_,
                   "moe_overlay_epoch_release");
    }

    bool DeviceGraphOrchestrator::executeMoEOverlayEpochBoundaryCaptured(
        MoEOverlayEpochBoundaryStage::Operation operation,
        void *execution_stream,
        const char *consumer_name)
    {
        if (!moe_overlay_epoch_execution_binding_)
            return true;
        if (!execution_stream ||
            !materializeMoEOverlayEpochBoundaryGraphs())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] ExpertOverlay epoch boundary requires a current graph and exact execution stream for "
                      << (consumer_name ? consumer_name : "unnamed"));
            return false;
        }

        auto &cache =
            operation == MoEOverlayEpochBoundaryStage::Operation::Acquire
                ? moe_overlay_epoch_acquire_graph_
                : moe_overlay_epoch_release_graph_;
        const uint64_t generation = workspaceGeneration(state_.device_id);
        if (!cache.valid || !cache.graph || !cache.stage ||
            cache.workspace_generation != generation ||
            cache.stage->getParams().operation != operation)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] ExpertOverlay epoch boundary graph has stale capture identity");
            return false;
        }
        if (!debugEnv().execution.gpu_graphs)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] GPU ExpertOverlay epoch boundaries require graph capture");
            return false;
        }

        IDeviceContext *const ctx = getDeviceContext(state_.device_id);
        if (!ctx)
            return false;
        auto &gpu_ctx =
            GPUDeviceContextPool::instance().getContext(state_.device_id);
        if (!cache.segment_cache.ensureCaptureStream(
                &gpu_ctx,
                state_.device_id,
                /*context_from_process_pool=*/true) ||
            !cache.segment_cache.orderCaptureStreamAfter(
                &gpu_ctx,
                execution_stream))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] ExpertOverlay epoch boundary could not join its producer stream");
            return false;
        }

        cache.segment_cache.perf_context =
            operation == MoEOverlayEpochBoundaryStage::Operation::Acquire
                ? "moe_overlay_epoch_acquire"
                : "moe_overlay_epoch_release";
        cache.segment_cache.replay_workload = {
            .seq_len = 1,
            .batch_size = 1,
            .m = 1,
            .decode = true,
        };
        auto policy = buildDecodeCapturePolicy(
            /*has_collectives=*/false,
            ctx);
        policy.defer_final_sync = true;
        policy.force_recapture = false;
        policy.graph_replay_plan_policy =
            DeviceGraphExecutor::GraphReplayPlanPolicy::RequireFullGraph;
        if (!policy.allow_cached_graph_replay)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] ExpertOverlay epoch boundary requires mandatory graph replay");
            return false;
        }

        cache.graph->reset();
        bool used_graph_replay = false;
        if (!executor_.executeDecodeWithCapturePolicy(
                *cache.graph,
                ctx,
                &cache.segment_cache,
                cache.segment_cache.capture_stream,
                &gpu_ctx,
                /*collective_nodes=*/nullptr,
                policy,
                &used_graph_replay) ||
            !cache.segment_cache.orderStreamAfterCapture(
                &gpu_ctx,
                execution_stream))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] ExpertOverlay epoch boundary failed strict captured execution");
            return false;
        }

        PerfStatsCollector::addCounter(
            "moe_overlay_epoch",
            operation == MoEOverlayEpochBoundaryStage::Operation::Acquire
                ? "external_transaction_acquires"
                : "external_transaction_releases",
            1.0,
            perfPhaseName(),
            state_.device_id.toString(),
            {{"consumer", consumer_name ? consumer_name : "unnamed"},
             {"graph_replay", boolTag(used_graph_replay)},
             {"request_slot",
              std::to_string(
                  moe_overlay_epoch_execution_binding_.request_slot)}});
        return true;
    }

    bool DeviceGraphOrchestrator::
        acquireMoEOverlayEpochForExternalTransaction(
            void *consumer_stream,
            DeviceTimelineRole consumer_role,
            const char *consumer_name,
            MoEOverlayEpochLeaseState acquired_state)
    {
        if (!moe_overlay_epoch_execution_binding_)
            return true;
        if (!consumer_stream || !moe_overlay_epoch_acquired_event_ ||
            !moe_overlay_epoch_released_event_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] ExpertOverlay epoch acquire has incomplete stream/event ownership");
            return false;
        }
        IBackend *const backend = getBackendFor(state_.device_id);
        if (!backend)
            return false;
        if (acquired_state != MoEOverlayEpochLeaseState::ExternalReader &&
            acquired_state != MoEOverlayEpochLeaseState::ExternalSequence &&
            acquired_state != MoEOverlayEpochLeaseState::ExternalForward)
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] External ExpertOverlay acquire "
                "received an invalid terminal lease state="
                << moeOverlayEpochLeaseStateName(acquired_state));
            return false;
        }
        const int ordinal = state_.device_id.gpu_ordinal();
        auto submission =
            moe_overlay_epoch_lease_lifecycle_->beginSubmission();
        const auto lease_state = submission.state();
        if (lease_state == MoEOverlayEpochLeaseState::ExternalReader &&
            (acquired_state == MoEOverlayEpochLeaseState::ExternalReader ||
             acquired_state == MoEOverlayEpochLeaseState::ExternalForward))
        {
            /*
             * A later child graph may use another stream, and a complete forward
             * may consume a reader acquired by its enclosing live-state prelude.
             * Both wait on the one durable acquire publication instead of
             * incrementing the device ticket twice. The forward-owner transition
             * is committed under this same submission authority, so maintenance
             * cannot release the reader between event wait and graph admission.
             */
            if (!moe_overlay_epoch_acquire_producer_stream_ ||
                !backend->streamWaitEvent(
                    consumer_stream,
                    moe_overlay_epoch_acquired_event_.get(),
                    ordinal))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] ExpertOverlay child could not consume the acquired epoch event");
                return false;
            }
            if (acquired_state ==
                    MoEOverlayEpochLeaseState::ExternalForward &&
                !submission.commit(
                    MoEOverlayEpochLeaseState::ExternalForward))
            {
                LOG_ERROR(
                    "[DeviceGraphOrchestrator] ExpertOverlay forward could not publish ownership of its existing external reader");
                std::terminate();
            }
            PerfStatsCollector::addCounter(
                "moe_overlay_epoch",
                "external_transaction_reader_event_waits",
                1.0,
                perfPhaseName(),
                state_.device_id.toString(),
                {{"consumer", consumer_name ? consumer_name : "unnamed"},
                 {"consumer_role",
                  std::string(deviceTimelineRoleName(consumer_role))},
                 {"reader_use",
                  acquired_state == MoEOverlayEpochLeaseState::ExternalForward
                      ? "forward_owner"
                      : "shared_child"}});
            return true;
        }

        const auto predecessor_state = lease_state;
        if (lease_state != MoEOverlayEpochLeaseState::Idle &&
            lease_state != MoEOverlayEpochLeaseState::ReleasePublished)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] External ExpertOverlay acquire overlaps another lease phase"
                      << " lease_state="
                      << moeOverlayEpochLeaseStateName(lease_state));
            return false;
        }
        if (predecessor_state ==
            MoEOverlayEpochLeaseState::ReleasePublished)
        {
            if (!submission.publishedReleaseProducerStream() ||
                !backend->streamWaitEvent(
                    consumer_stream,
                    moe_overlay_epoch_released_event_.get(),
                    ordinal))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] ExpertOverlay acquire could not join the prior release");
                return false;
            }
        }

        if (!executeMoEOverlayEpochBoundaryCaptured(
                MoEOverlayEpochBoundaryStage::Operation::Acquire,
                consumer_stream,
                consumer_name))
        {
            return false;
        }
        if (!backend->recordEvent(
                moe_overlay_epoch_acquired_event_.get(),
                ordinal,
                consumer_stream))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] ExpertOverlay acquire was submitted without a publishable completion event");
            std::terminate();
        }
        moe_overlay_epoch_acquire_producer_stream_ = consumer_stream;
        if (!submission.commit(acquired_state))
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] ExpertOverlay acquire could not publish its complete semantic owner");
            std::terminate();
        }
        return true;
    }

    bool DeviceGraphOrchestrator::admitMoEOverlayExternalSequenceGraph(
        const MoEOverlayInferenceParticipantGraphBinding &binding,
        void *execution_stream,
        DeviceTimelineRole consumer_role,
        const char *consumer_name)
    {
        if (!binding.active)
            return true;
        if (!state_.device_id.is_gpu() || !execution_stream ||
            binding.request_generation == 0 || binding.sequence_id == 0 ||
            binding.sequence_graph_count <= 1 ||
            binding.sequence_graph_ordinal < 0 ||
            binding.sequence_graph_ordinal >=
                binding.sequence_graph_count - 1 ||
            binding.descriptor.graph_role !=
                MoEOverlayInferenceGraphRole::MTPDraft ||
            binding.descriptor.placement_epoch == 0)
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Direct ExpertOverlay MTP graph "
                "received an invalid execution-sequence binding"
                << " device=" << state_.device_id.toString()
                << " request_generation=" << binding.request_generation
                << " sequence_id=" << binding.sequence_id
                << " ordinal=" << binding.sequence_graph_ordinal
                << " graph_count=" << binding.sequence_graph_count
                << " placement_epoch="
                << binding.descriptor.placement_epoch);
            return false;
        }

        /*
         * Some GPU participants can be outside the topology-wide epoch arena.
         * They still authenticate the rank-wide graph sequence, but have no
         * local residency ticket to acquire or retain.
         */
        if (!moe_overlay_epoch_execution_binding_)
            return true;

        auto &identity = moe_overlay_external_sequence_identity_;
        if (binding.beginsSequence())
        {
            if (identity.valid())
            {
                LOG_ERROR(
                    "[DeviceGraphOrchestrator] ExpertOverlay MTP sequence "
                    "began while another sequence identity remained live"
                    << " live_sequence=" << identity.sequence_id
                    << " incoming_sequence=" << binding.sequence_id);
                return false;
            }
            if (!acquireMoEOverlayEpochForExternalTransaction(
                    execution_stream,
                    consumer_role,
                    consumer_name,
                    MoEOverlayEpochLeaseState::ExternalSequence))
            {
                return false;
            }
            identity.request_generation = binding.request_generation;
            identity.sequence_id = binding.sequence_id;
            identity.placement_epoch =
                binding.descriptor.placement_epoch;
            identity.graph_count = binding.sequence_graph_count;
            identity.last_graph_ordinal = binding.sequence_graph_ordinal;
        }
        else
        {
            const auto lease_state =
                moe_overlay_epoch_lease_lifecycle_->load();
            const bool identity_matches =
                identity.valid() &&
                identity.request_generation == binding.request_generation &&
                identity.sequence_id == binding.sequence_id &&
                identity.placement_epoch ==
                    binding.descriptor.placement_epoch &&
                identity.graph_count == binding.sequence_graph_count &&
                binding.sequence_graph_ordinal ==
                    identity.last_graph_ordinal + 1;
            if (lease_state !=
                    MoEOverlayEpochLeaseState::ExternalSequence ||
                !identity_matches)
            {
                LOG_ERROR(
                    "[DeviceGraphOrchestrator] ExpertOverlay MTP sequence "
                    "graph is duplicated, skipped, crossed, or changed epoch"
                    << " lease_state="
                    << moeOverlayEpochLeaseStateName(lease_state)
                    << " live_request=" << identity.request_generation
                    << " incoming_request=" << binding.request_generation
                    << " live_sequence=" << identity.sequence_id
                    << " incoming_sequence=" << binding.sequence_id
                    << " live_ordinal=" << identity.last_graph_ordinal
                    << " incoming_ordinal="
                    << binding.sequence_graph_ordinal
                    << " live_count=" << identity.graph_count
                    << " incoming_count=" << binding.sequence_graph_count
                    << " live_epoch=" << identity.placement_epoch
                    << " incoming_epoch="
                    << binding.descriptor.placement_epoch);
                return false;
            }

            IBackend *const backend = getBackendFor(state_.device_id);
            if (!backend || !moe_overlay_epoch_acquired_event_ ||
                !moe_overlay_epoch_acquire_producer_stream_ ||
                !backend->streamWaitEvent(
                    execution_stream,
                    moe_overlay_epoch_acquired_event_.get(),
                    state_.device_id.gpu_ordinal()))
            {
                LOG_ERROR(
                    "[DeviceGraphOrchestrator] ExpertOverlay MTP sequence "
                    "graph could not consume its immutable acquire edge");
                return false;
            }
            identity.last_graph_ordinal =
                binding.sequence_graph_ordinal;
        }

        PerfStatsCollector::addCounter(
            "moe_overlay_epoch",
            binding.beginsSequence()
                ? "external_sequence_acquires"
                : "external_sequence_graph_joins",
            1.0,
            perfPhaseName(),
            state_.device_id.toString(),
            {{"sequence_id", std::to_string(binding.sequence_id)},
             {"graph_ordinal",
              std::to_string(binding.sequence_graph_ordinal)},
             {"graph_count",
              std::to_string(binding.sequence_graph_count)},
             {"placement_epoch",
              std::to_string(binding.descriptor.placement_epoch)},
             {"blocking", "false"}});
        return true;
    }

    bool DeviceGraphOrchestrator::finishMoEOverlayExternalSequenceGraph(
        const MoEOverlayInferenceParticipantGraphBinding &binding,
        void *execution_stream,
        bool execution_succeeded,
        const char *consumer_name)
    {
        if (!binding.active || !moe_overlay_epoch_execution_binding_)
            return true;

        const auto &identity = moe_overlay_external_sequence_identity_;
        const auto lease_state =
            moe_overlay_epoch_lease_lifecycle_->load();
        const bool identity_matches =
            identity.valid() &&
            identity.request_generation == binding.request_generation &&
            identity.sequence_id == binding.sequence_id &&
            identity.placement_epoch == binding.descriptor.placement_epoch &&
            identity.graph_count == binding.sequence_graph_count &&
            identity.last_graph_ordinal ==
                binding.sequence_graph_ordinal;
        if (lease_state != MoEOverlayEpochLeaseState::ExternalSequence ||
            !identity_matches || !execution_stream)
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] ExpertOverlay MTP graph terminal "
                "lost its exact sequence lease"
                << " boundary="
                << (consumer_name ? consumer_name : "unnamed")
                << " lease_state="
                << moeOverlayEpochLeaseStateName(lease_state)
                << " live_sequence=" << identity.sequence_id
                << " incoming_sequence=" << binding.sequence_id
                << " live_ordinal=" << identity.last_graph_ordinal
                << " incoming_ordinal="
                << binding.sequence_graph_ordinal);
            return false;
        }

        if (execution_succeeded)
        {
            /* The grouped verifier owns the only successful terminal release. */
            return true;
        }

        if (!releaseMoEOverlayEpochForExternalTransaction(
                execution_stream,
                consumer_name,
                MoEOverlayEpochLeaseState::ExternalSequence))
        {
            return false;
        }
        moe_overlay_external_sequence_identity_.reset();
        PerfStatsCollector::addCounter(
            "moe_overlay_epoch",
            "external_sequence_aborts",
            1.0,
            perfPhaseName(),
            state_.device_id.toString(),
            {{"sequence_id", std::to_string(binding.sequence_id)},
             {"graph_ordinal",
              std::to_string(binding.sequence_graph_ordinal)},
             {"blocking", "false"}});
        return true;
    }

    bool DeviceGraphOrchestrator::
        claimMoEOverlayEpochForPlacementMaintenance(
            void *maintenance_stream,
            const char *boundary_name)
    {
        if (!moe_overlay_epoch_execution_binding_)
            return true;
        if (!maintenance_stream ||
            !moe_overlay_epoch_acquired_event_ ||
            !moe_overlay_epoch_released_event_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] ExpertOverlay placement maintenance reached an incomplete boundary at "
                      << (boundary_name ? boundary_name : "unnamed"));
            return false;
        }

        auto submission =
            moe_overlay_epoch_lease_lifecycle_->beginSubmission();
        const auto lease_state = submission.state();
        const auto action =
            moeOverlayEpochPlacementMaintenanceAction(lease_state);
        const char *action_name = "invalid";
        switch (action)
        {
        case MoEOverlayEpochPlacementMaintenanceAction::CloseExternalReader:
            action_name = "close_external_reader";
            if (!submitMoEOverlayEpochExternalRelease(
                    submission,
                    maintenance_stream,
                    boundary_name,
                    MoEOverlayEpochLeaseState::ExternalReader))
            {
                return false;
            }
            break;
        case MoEOverlayEpochPlacementMaintenanceAction::JoinPublishedRelease:
        {
            action_name = "join_published_release";
            IBackend *const backend = getBackendFor(state_.device_id);
            if (!backend ||
                !submission.publishedReleaseProducerStream() ||
                !backend->streamWaitEvent(
                    maintenance_stream,
                    moe_overlay_epoch_released_event_.get(),
                    state_.device_id.gpu_ordinal()))
            {
                LOG_ERROR(
                    "[DeviceGraphOrchestrator] ExpertOverlay placement maintenance could not join the published release at "
                    << (boundary_name ? boundary_name : "unnamed"));
                return false;
            }
            break;
        }
        case MoEOverlayEpochPlacementMaintenanceAction::
            RejectMissingBoundary:
            LOG_ERROR(
                "[DeviceGraphOrchestrator] ExpertOverlay placement maintenance has no committed inference boundary at "
                << (boundary_name ? boundary_name : "unnamed"));
            return false;
        case MoEOverlayEpochPlacementMaintenanceAction::
            RejectLiveInferenceOwner:
            LOG_ERROR(
                "[DeviceGraphOrchestrator] ExpertOverlay placement maintenance overlaps a live inference owner"
                << " boundary="
                << (boundary_name ? boundary_name : "unnamed")
                << " lease_state="
                << moeOverlayEpochLeaseStateName(lease_state));
            return false;
        }

        PerfStatsCollector::addCounter(
            "moe_overlay_epoch",
            "placement_maintenance_boundary_claims",
            1.0,
            perfPhaseName(),
            state_.device_id.toString(),
            {{"boundary", boundary_name ? boundary_name : "unnamed"},
             {"action", action_name},
             {"blocking", "false"}});
        return true;
    }

    bool DeviceGraphOrchestrator::submitMoEOverlayEpochExternalRelease(
        MoEOverlayEpochLeaseLifecycle::Submission &submission,
        void *release_stream,
        const char *boundary_name,
        MoEOverlayEpochLeaseState expected_state)
    {
        if (!moe_overlay_epoch_execution_binding_ || !release_stream ||
            !moe_overlay_epoch_acquired_event_ ||
            !moe_overlay_epoch_released_event_ ||
            submission.state() != expected_state)
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] ExpertOverlay external release primitive received an incomplete or mismatched lease"
                      << " boundary="
                      << (boundary_name ? boundary_name : "unnamed")
                      << " lease_state="
                      << moeOverlayEpochLeaseStateName(submission.state())
                      << " expected_state="
                      << moeOverlayEpochLeaseStateName(expected_state));
            return false;
        }

        IBackend *const backend = getBackendFor(state_.device_id);
        const int ordinal = state_.device_id.gpu_ordinal();
        if (!backend || !moe_overlay_epoch_acquire_producer_stream_ ||
            !backend->streamWaitEvent(
                release_stream,
                moe_overlay_epoch_acquired_event_.get(),
                ordinal) ||
            !executeMoEOverlayEpochBoundaryCaptured(
                MoEOverlayEpochBoundaryStage::Operation::Release,
                release_stream,
                boundary_name))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] ExpertOverlay release could not consume and close its acquired ticket");
            return false;
        }
        if (!backend->recordEvent(
                moe_overlay_epoch_released_event_.get(),
                ordinal,
                release_stream))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] ExpertOverlay release was submitted without a durable completion event");
            std::terminate();
        }

        moe_overlay_epoch_acquire_producer_stream_ = nullptr;
        if (!submission.publishRelease(release_stream))
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] ExpertOverlay release could not publish its durable semantic receipt");
            std::terminate();
        }
        return true;
    }

    bool DeviceGraphOrchestrator::
        releaseMoEOverlayEpochForExternalTransaction(
            void *release_stream,
            const char *boundary_name,
            MoEOverlayEpochLeaseState expected_state)
    {
        if (!moe_overlay_epoch_execution_binding_)
            return true;
        if (!release_stream ||
            !moe_overlay_epoch_acquired_event_ ||
            !moe_overlay_epoch_released_event_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] ExpertOverlay release reached an incomplete or inactive transaction at "
                      << (boundary_name ? boundary_name : "unnamed"));
            return false;
        }

        if (expected_state != MoEOverlayEpochLeaseState::ExternalReader &&
            expected_state != MoEOverlayEpochLeaseState::ExternalSequence &&
            expected_state != MoEOverlayEpochLeaseState::ExternalForward)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] ExpertOverlay release received an invalid expected lease state"
                      << " expected_state="
                      << moeOverlayEpochLeaseStateName(expected_state));
            return false;
        }
        auto submission =
            moe_overlay_epoch_lease_lifecycle_->beginSubmission();
        const auto prior_state = submission.state();
        if (prior_state != expected_state)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] ExpertOverlay release has no external lease"
                      << " boundary="
                      << (boundary_name ? boundary_name : "unnamed")
                      << " lease_state="
                      << moeOverlayEpochLeaseStateName(prior_state));
            return false;
        }
        return submitMoEOverlayEpochExternalRelease(
            submission,
            release_stream,
            boundary_name,
            expected_state);
    }

    bool DeviceGraphOrchestrator::acquireMoEOverlayEpochForHostedParent(
        void *acquire_stream,
        const IGPUGraphCapture *acquire_capture,
        const char *parent_name)
    {
        if (!moe_overlay_epoch_execution_binding_)
            return true;
        if (!acquire_stream || !acquire_capture ||
            !moe_overlay_epoch_acquired_event_ ||
            !moe_overlay_epoch_released_event_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Hosted ExpertOverlay parent acquire has incomplete or overlapping ownership at "
                      << (parent_name ? parent_name : "unnamed"));
            return false;
        }

        IBackend *const backend = getBackendFor(state_.device_id);
        const int ordinal = state_.device_id.gpu_ordinal();
        if (!backend)
            return false;
        auto submission =
            moe_overlay_epoch_lease_lifecycle_->beginSubmission();
        const auto lease_state = submission.state();
        const auto predecessor_state = lease_state;
        if (lease_state != MoEOverlayEpochLeaseState::Idle &&
            lease_state != MoEOverlayEpochLeaseState::ReleasePublished)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Hosted ExpertOverlay parent acquire overlaps another lease phase"
                      << " lease_state="
                      << moeOverlayEpochLeaseStateName(lease_state));
            return false;
        }
        if (predecessor_state ==
            MoEOverlayEpochLeaseState::ReleasePublished)
        {
            if (!submission.publishedReleaseProducerStream() ||
                !backend->streamWaitEvent(
                    acquire_stream,
                    moe_overlay_epoch_released_event_.get(),
                    ordinal))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Hosted ExpertOverlay parent acquire could not join the prior branch release at "
                          << (parent_name ? parent_name : "unnamed"));
                return false;
            }
        }

        if (!acquire_capture->launchOnStream(acquire_stream) ||
            !backend->recordEvent(
                moe_overlay_epoch_acquired_event_.get(),
                ordinal,
                acquire_stream))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Hosted ExpertOverlay parent acquire could not publish its exact completion edge at "
                      << (parent_name ? parent_name : "unnamed"));
            return false;
        }
        moe_overlay_epoch_acquire_producer_stream_ = acquire_stream;
        if (!submission.commit(MoEOverlayEpochLeaseState::HostedParent))
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Hosted ExpertOverlay parent acquire could not publish its complete owner");
            std::terminate();
        }
        PerfStatsCollector::addCounter(
            "moe_overlay_epoch",
            "hosted_parent_acquires",
            1.0,
            "decode",
            state_.device_id.toString(),
            {{"parent", parent_name ? parent_name : "unnamed"},
             {"blocking", "false"}});
        return true;
    }

    bool DeviceGraphOrchestrator::releaseMoEOverlayEpochForHostedParent(
        void *release_stream,
        const IGPUGraphCapture *release_capture,
        const char *parent_name)
    {
        if (!moe_overlay_epoch_execution_binding_)
            return true;
        if (!release_stream || !release_capture ||
            !moe_overlay_epoch_acquired_event_ ||
            !moe_overlay_epoch_released_event_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Hosted ExpertOverlay parent release has incomplete or contradictory ownership at "
                      << (parent_name ? parent_name : "unnamed"));
            return false;
        }

        auto submission =
            moe_overlay_epoch_lease_lifecycle_->beginSubmission();
        const auto lease_state = submission.state();
        if (lease_state != MoEOverlayEpochLeaseState::HostedParent)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Hosted ExpertOverlay parent release lost its atomic lease claim"
                      << " lease_state="
                      << moeOverlayEpochLeaseStateName(lease_state));
            return false;
        }

        IBackend *const backend = getBackendFor(state_.device_id);
        const int ordinal = state_.device_id.gpu_ordinal();
        if (!backend || !moe_overlay_epoch_acquire_producer_stream_ ||
            !backend->streamWaitEvent(
                release_stream,
                moe_overlay_epoch_acquired_event_.get(),
                ordinal) ||
            !release_capture->launchOnStream(release_stream) ||
            !backend->recordEvent(
                moe_overlay_epoch_released_event_.get(),
                ordinal,
                release_stream))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Hosted ExpertOverlay parent release could not close and publish its ticket at "
                      << (parent_name ? parent_name : "unnamed"));
            return false;
        }

        moe_overlay_epoch_acquire_producer_stream_ = nullptr;
        if (!submission.publishRelease(release_stream))
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Hosted ExpertOverlay parent release could not publish its durable receipt");
            std::terminate();
        }
        PerfStatsCollector::addCounter(
            "moe_overlay_epoch",
            "hosted_parent_releases",
            1.0,
            "decode",
            state_.device_id.toString(),
            {{"parent", parent_name ? parent_name : "unnamed"},
             {"blocking", "false"}});
        return true;
    }

    bool DeviceGraphOrchestrator::consumeMoEOverlayEpochForHostedChild(
        void *consumer_stream,
        const char *child_name)
    {
        if (!moe_overlay_epoch_execution_binding_)
            return true;
        if (!consumer_stream ||
            moe_overlay_epoch_lease_lifecycle_->load() !=
                MoEOverlayEpochLeaseState::HostedParent ||
            !moe_overlay_epoch_acquired_event_ ||
            !moe_overlay_epoch_acquire_producer_stream_)
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Hosted ExpertOverlay child has no complete parent acquire edge at "
                << (child_name ? child_name : "unnamed"));
            return false;
        }

        IBackend *const backend = getBackendFor(state_.device_id);
        if (!backend || !backend->streamWaitEvent(
                            consumer_stream,
                            moe_overlay_epoch_acquired_event_.get(),
                            state_.device_id.gpu_ordinal()))
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Hosted ExpertOverlay child could not consume its parent acquire event at "
                << (child_name ? child_name : "unnamed"));
            return false;
        }
        PerfStatsCollector::addCounter(
            "moe_overlay_epoch",
            "hosted_child_acquire_event_consumptions",
            1.0,
            "decode",
            state_.device_id.toString(),
            {{"child", child_name ? child_name : "unnamed"},
             {"blocking", "false"}});
        return true;
    }

    bool DeviceGraphOrchestrator::prepareMoEOverlayEpochForInternalParent(
        void *parent_stream,
        const char *parent_name)
    {
        if (!moe_overlay_epoch_execution_binding_)
            return true;
        const auto lease_state = moe_overlay_epoch_lease_lifecycle_->load();
        if (!parent_stream ||
            lease_state == MoEOverlayEpochLeaseState::HostedParent ||
            lease_state == MoEOverlayEpochLeaseState::HostedChildForward)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] ExpertOverlay internal parent requires an exact stream and no already-active hosted parent at "
                      << (parent_name ? parent_name : "unnamed"));
            return false;
        }

        /*
         * A static overlay has no maintenance graph to close the externally
         * scheduled main transaction. Release it here, after the caller joined
         * all first-transaction state, so the parent's first captured acquire
         * sees an empty request ticket.
         */
        if (lease_state == MoEOverlayEpochLeaseState::ExternalReader)
        {
            return releaseMoEOverlayEpochForExternalTransaction(
                parent_stream,
                parent_name);
        }

        /*
         * Dynamic maintenance closes the ticket before its own graph launch.
         * The maintenance completion edge normally subsumes this release, but
         * retaining the direct event dependency makes the parent contract valid
         * even when a future topology has no maintenance payload to execute.
         */
        auto release_submission =
            moe_overlay_epoch_lease_lifecycle_->beginSubmission();
        if (release_submission.state() !=
            MoEOverlayEpochLeaseState::ReleasePublished)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] ExpertOverlay internal parent has neither an active external reader nor a published release at "
                      << (parent_name ? parent_name : "unnamed")
                      << " lease_state="
                      << moeOverlayEpochLeaseStateName(
                             release_submission.state()));
            return false;
        }
        IBackend *const backend = getBackendFor(state_.device_id);
        if (!backend || !moe_overlay_epoch_released_event_ ||
            !release_submission.publishedReleaseProducerStream() ||
            !backend->streamWaitEvent(
                parent_stream,
                moe_overlay_epoch_released_event_.get(),
                state_.device_id.gpu_ordinal()))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] ExpertOverlay internal parent could not consume the external release event at "
                      << (parent_name ? parent_name : "unnamed"));
            return false;
        }
        PerfStatsCollector::addCounter(
            "moe_overlay_epoch",
            "internal_parent_external_release_waits",
            1.0,
            perfPhaseName(),
            state_.device_id.toString(),
            {{"parent", parent_name ? parent_name : "unnamed"}});
        return true;
    }

    bool DeviceGraphOrchestrator::waitForPendingDeviceMoERebalanceMaintenance(
        void *consumer_stream,
        DeviceTimelineRole consumer_role,
        const char *consumer_name)
    {
        if (!state_.device_id.is_gpu())
            return true;

        auto &cache = device_moe_rebalance_maintenance_graph_;
        const bool maintenance_wait_required = cache.launch_count != 0;
        if (!consumer_stream)
        {
            throw std::runtime_error(
                "Pending device MoE maintenance requires an explicit consumer "
                "stream on " +
                state_.device_id.toString() + " consumer=" +
                (consumer_name ? consumer_name : "unnamed"));
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
        {
            throw std::runtime_error(
                "Pending device MoE maintenance could not resolve backend for " +
                state_.device_id.toString() + " consumer=" +
                (consumer_name ? consumer_name : "unnamed"));
        }

        const std::string device_key = state_.device_id.toString();
        const std::string consumer =
            consumer_name && *consumer_name ? consumer_name : "unnamed";
        const auto wait_one =
            [&](const DeviceMoERebalanceMaintenanceGraphCache &target_cache,
                const char *kind)
        {
            if (!target_cache.completion_event)
            {
                throw std::runtime_error(
                    "Device MoE " + std::string(kind) +
                    " maintenance is in flight without a completion event on " +
                    device_key + " consumer=" + consumer);
            }
            if (!DeviceEventEdge::at(
                     DeviceTimelinePoint::MoERebalanceMaintenanceReady)
                     .from(DeviceTimelineRole::MoERebalanceMaintenance)
                     .to(consumer_role)
                     .enqueuePublishedWait(
                         *backend,
                         state_.device_id,
                         target_cache.completion_event.get(),
                         consumer_stream))
            {
                throw std::runtime_error(
                    "Failed to order " + consumer + " after device MoE " +
                    kind + " maintenance on " + device_key);
            }

            PerfStatsCollector::Tags tags = {
                {"consumer", consumer},
                {"consumer_role",
                 std::string(deviceTimelineRoleName(consumer_role))},
                {"maintenance_graph_kind", kind},
                {"diagnostic_status_in_flight",
                 boolTag(target_cache.completion_event_in_flight)}};
            PerfStatsCollector::addCounterWithSequence(
                "moe_rebalance",
                "device_maintenance_graph_consumer_event_waits",
                1.0,
                {target_cache.launch_count},
                "decode",
                device_key,
                std::move(tags));
            return true;
        };

        if (maintenance_wait_required &&
            !wait_one(
                device_moe_rebalance_maintenance_graph_,
                "atomic"))
        {
            throw std::runtime_error(
                "Atomic device MoE maintenance wait returned without either "
                "succeeding or throwing");
        }
        return true;
    }

    bool DeviceGraphOrchestrator::
        materializeDeviceMoERebalanceMaintenanceGraphForFamily()
    {
        if (!state_.device_id.is_gpu() ||
            !isMoeRebalancingActive() ||
            !usesDeviceResidentMoEOverlayAuthority())
        {
            return true;
        }
        if (!graph_builder_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device MoE maintenance family declaration has no graph builder");
            return false;
        }

        auto &cache = device_moe_rebalance_maintenance_graph_;
        if (usesTopologyWideDeviceMoEOverlayController())
        {
            /*
             * The main forward-family declaration has already materialized the
             * model-lifetime runtime table and its epoch arena.  Certify that
             * exact binding here; the orchestration layer will pass it unchanged
             * to MoEOverlayDeviceControllerGraphService after all rank-local
             * participants have completed construction.
             *
             * Do not build a local maintenance graph as a temporary bridge.
             * Both graph types can publish the same bank selector, so even an
             * apparently idle duplicate would violate the single-writer RCU
             * contract as soon as its cadence became due.
             */
            if (cache.graph || !cache.collective_nodes.empty())
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Mapped topology-wide ExpertOverlay found a forbidden participant-local maintenance graph on "
                          << state_.device_id.toString());
                return false;
            }

            const auto bindings =
                moeOverlayDeviceControllerRuntimeBindings();
            if (bindings.size() != 1u ||
                !bindings.front().backgroundPublicationValid() ||
                !bindings.front().initialRuntimePublicationValid() ||
                bindings.front().device != state_.device_id ||
                bindings.front().inference_boundary != this)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Mapped topology-wide ExpertOverlay has no unique production runtime/publication binding on "
                          << state_.device_id.toString()
                          << " bindings=" << bindings.size());
                return false;
            }

            PerfStatsCollector::addCounter(
                "moe_overlay_controller",
                "mapped_runtime_bindings_certified",
                1.0,
                "graph_setup",
                state_.device_id.toString(),
                {{"authority", "topology_wide_device_controller"},
                 {"participant_local_maintenance_graph", "false"},
                 {"durable_writer_count", "1"},
                 {"blocking", "false"}});
            return true;
        }

        if (!debugEnv().moe_rebalance.device_rebalance_maintenance_graph)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Single-domain native device-resident ExpertOverlay was selected, but its captured maintenance graph is disabled");
            return false;
        }

        if (!cache.graph)
        {
            ComputeGraph graph =
                graph_builder_->buildDeviceMoERebalanceMaintenanceGraph(
                    state_.device_id);
            if (graph.size() == 0)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Active device MoE rebalance did not materialize its maintenance graph during family declaration on "
                          << state_.device_id.toString());
                return false;
            }
            cache.graph =
                std::make_unique<ComputeGraph>(std::move(graph));
        }

        cache.collective_nodes =
            cache.graph->collectiveNodeNames();
        if (cache.collective_nodes.empty())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device MoE maintenance family participant declares no collective nodes on "
                      << state_.device_id.toString());
            cache.invalidate();
            return false;
        }

        /*
         * Dynamic residency maintenance is a child transaction of the enclosing
         * LocalTP graph family, not an independently communicating workload.
         * Every producer and consumer transition is represented by a typed
         * device event, so the maintenance stage must use the family's already
         * initialized NCCL/RCCL context.  A second communicator would allocate
         * transport resources after model placement and would reintroduce an
         * unordered communication lane that the graph cannot express.
         *
         * Validate the pointer identity while the immutable family graph is
         * declared.  This makes a future accidental dedicated communicator a
         * construction failure before capture or GPU submission, rather than
         * a late request-time OOM or collective-ordering race.
         */
        auto *domain_collective_ctx =
            dynamic_cast<ILocalTPContext *>(
                graph_builder_->config().tp_ctx);
        size_t maintenance_stage_count = 0;
        if (!domain_collective_ctx)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device MoE maintenance requires the enclosing LocalTP collective context on "
                      << state_.device_id.toString());
            cache.invalidate();
            return false;
        }
        for (const auto &node_name : cache.graph->getExecutionOrder())
        {
            const ComputeNode *node = cache.graph->getNode(node_name);
            const auto *stage =
                node && node->stage
                    ? dynamic_cast<const MoEDeviceRebalanceStage *>(
                          node->stage.get())
                    : nullptr;
            if (!stage)
                continue;
            ++maintenance_stage_count;
            if (stage->getParams().tp_ctx != domain_collective_ctx)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Device MoE maintenance attempted to bind a collective context outside its graph family on "
                          << state_.device_id.toString());
                cache.invalidate();
                return false;
            }
        }
        if (maintenance_stage_count != 1u)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device MoE maintenance graph requires exactly one domain-collective transaction owner on "
                      << state_.device_id.toString()
                      << " owners=" << maintenance_stage_count);
            cache.invalidate();
            return false;
        }

        return true;
    }

    bool DeviceGraphOrchestrator::materializePipelineMaintenanceWithoutLaunch(ILocalTPContext &original, int participant)
    {
        if (!usesParticipantLocalDeviceMoERebalanceController() || !graph_builder_ ||
            graph_builder_->config().tp_ctx != &original || participant < 0 || participant >= original.degree() ||
            original.devices()[participant].toLocalDeviceId() != state_.device_id)
            return false;
        auto &cache = device_moe_rebalance_maintenance_graph_;
        auto &gpu = GPUDeviceContextPool::instance().getContext(state_.device_id);
        auto *backend = getBackendFor(state_.device_id);
        auto *context = getDeviceContext(state_.device_id);
        if (!backend || !context || !cache.graph || cache.collective_nodes.empty() ||
            cache.workspace_generation != workspaceGeneration(state_.device_id) ||
            !cache.segment_cache.ensureCaptureStream(&gpu, state_.device_id, true)) return false;
        void *const stream = cache.segment_cache.capture_stream;
        std::string error;
        if (!prepareDeviceMoERebalanceCadenceTransactionSources(&gpu, backend, stream, &error)) return false;
        auto policy = buildDecodeCapturePolicy(true, context);
        if (!policy.allow_cached_graph_replay || !policy.collectives_graph_capturable) return false;
        policy.defer_final_sync = true;
        policy.graph_replay_plan_policy = DeviceGraphExecutor::GraphReplayPlanPolicy::RequireFullGraph;
        // Only actual members of this independent stage participate. An idle
        // singleton/Static stage must not join a manufactured pipeline barrier.
        policy.capture_boundary = [&original, participant](const std::string &phase, void *capture_stream) {
            return original.graphCaptureBoundaryOnStream("pipeline_original_stage_maintenance:" + phase,
                participant, capture_stream, collective_timeout_policy::effectiveCollectTimeoutMs(0));
        };
        cache.segment_cache.perf_context = "pipeline_original_stage_maintenance_source";
        dgo_detail::bindMoEMaintenanceReplayContext(cache.segment_cache);
        cache.graph->reset();
        bool replay = false;
        if (!executor_.executeDecodeWithCapturePolicy(*cache.graph, context, &cache.segment_cache, stream,
                &gpu, &cache.collective_nodes, policy, &replay,
                DeviceGraphExecutor::GraphInitialSubmissionPolicy::MaterializeWithoutLaunch) ||
            !cache.segment_cache.deviceLoopGraphTemplate(*cache.graph, &error)) return false;
        return composeDeviceMoERebalanceCadenceTransaction(&error);
    }

    bool DeviceGraphOrchestrator::
        appendHostedDeviceMoEDecodeCommitBoundary(
            ComputeGraph &graph,
            const ForwardInput &input,
            std::string *error)
    {
        if (error)
            error->clear();

        const bool owns_serial_decode_boundary =
            ownsSerialDecodeCommitBoundary(
                input.execution_role,
                input.execution_phase,
                input.state_transaction);
        if (!owns_serial_decode_boundary ||
            deviceMoERebalanceMaintenanceExecutionPolicy() !=
                DeviceMoERebalanceMaintenanceExecutionPolicy::
                    HostScheduledCapturedMaintenance)
        {
            return true;
        }

        if (graph.size() == 0u)
        {
            if (error)
                *error = "cannot append a decode commit boundary to an empty graph";
            return false;
        }
        if (!materializeDeviceMoERebalanceMaintenanceGraphForFamily())
        {
            if (error)
                *error = "could not materialize the shared device MoE maintenance participant";
            return false;
        }

        const MoEDeviceRebalanceStage *transaction_owner = nullptr;
        const auto &maintenance =
            *device_moe_rebalance_maintenance_graph_.graph;
        for (const auto &node_name : maintenance.getExecutionOrder())
        {
            const ComputeNode *const node = maintenance.getNode(node_name);
            const auto *const stage =
                node && node->stage
                    ? dynamic_cast<const MoEDeviceRebalanceStage *>(
                          node->stage.get())
                    : nullptr;
            if (!stage || !stage->ownsRequestTransactionState())
                continue;
            if (transaction_owner)
            {
                if (error)
                    *error = "maintenance graph has multiple cadence transaction owners";
                return false;
            }
            transaction_owner = stage;
        }
        if (!transaction_owner)
        {
            if (error)
                *error = "maintenance graph has no cadence transaction owner";
            return false;
        }

        const auto &owner_params = transaction_owner->getParams();
        const std::string workspace_name =
            owner_params.workspace_name.empty()
                ? owner_params.stage_name
                : owner_params.workspace_name;
        IBackend *const backend = getBackendFor(state_.device_id);
        if (!backend || workspace_name.empty())
        {
            if (error)
                *error = "HIP decode cadence has no backend or workspace identity";
            return false;
        }

        constexpr const char *kBoundaryNode =
            "moe_device_decode_commit_boundary";
        if (graph.getNode(kBoundaryNode))
        {
            if (error)
                *error = "main decode graph already contains a cadence boundary";
            return false;
        }

        const std::string output_terminal = graph.terminalNode();
        if (ComputeNode *const output_node = graph.getNode(output_terminal))
        {
            /*
             * DeviceGraphExecutor normally marks only the final schedule node
             * as an event-publishing tensor producer.  The cadence stage has no
             * tensor outputs, so preserve that responsibility on the model's
             * logical output before installing a new control-only terminal.
             */
            output_node->is_final_output = true;
        }

        const std::vector<std::string> prior_leaves = graph.getLeafNodes(GraphLeafScope::MainExecution);
        if (prior_leaves.empty())
        {
            if (error)
                *error = "main decode graph has no terminal leaves";
            return false;
        }

        MoEDeviceDecodeCommitBoundaryStage::Params params;
        params.device_id = state_.device_id;
        params.backend = backend;
        params.workspace_name = workspace_name;
        params.stage_name = kBoundaryNode;
        graph.addNode(
            kBoundaryNode,
            ComputeStageFactory::createMoEDeviceDecodeCommitBoundary(params),
            state_.device_id);
        for (const auto &leaf : prior_leaves)
            graph.addDependency(kBoundaryNode, leaf);
        graph.setTerminalNode(kBoundaryNode);

        device_moe_rebalance_maintenance_graph_
            .hosted_decode_commit_boundary_embedded = true;
        return true;
    }

    bool DeviceGraphOrchestrator::
        appendCapturedMainForwardMoEOverlayEpochTransaction(
            ComputeGraph &graph,
            const ForwardInput &input,
            std::string *error)
    {
        if (error)
            error->clear();

        const bool owns_captured_main_transaction =
            input.execution_role == ForwardExecutionRole::MainInference &&
            static_cast<bool>(moe_overlay_epoch_execution_binding_);
        if (!owns_captured_main_transaction)
            return true;

        if (!state_.device_id.is_gpu() ||
            input.device != state_.device_id ||
            !moe_overlay_epoch_execution_binding_)
        {
            if (error)
            {
                *error =
                    "captured main graph has no matching GPU epoch binding";
            }
            return false;
        }
        if (graph.size() == 0u)
        {
            if (error)
                *error = "cannot enclose an empty main graph in an epoch transaction";
            return false;
        }

        constexpr const char *kAcquireNode =
            "moe_overlay_main_forward_epoch_acquire";
        constexpr const char *kReleaseNode =
            "moe_overlay_main_forward_epoch_release";
        if (graph.getNode(kAcquireNode) || graph.getNode(kReleaseNode))
        {
            if (error)
                *error = "main graph already contains an epoch transaction boundary";
            return false;
        }

        const std::vector<std::string> prior_roots = graph.getRootNodes();
        const std::vector<std::string> prior_leaves = graph.getLeafNodes(GraphLeafScope::MainExecution);
        if (prior_roots.empty() || prior_leaves.empty())
        {
            if (error)
                *error = "main graph has no complete root/leaf frontier";
            return false;
        }

        /*
         * The release is a control-only terminal. Preserve the model's logical
         * output marker so tensor publication remains attached to the stage
         * that actually produced the output, while the executor's terminal
         * event is recorded after the RCU ticket has been released.
         */
        const std::string prior_terminal = graph.terminalNode();
        if (ComputeNode *const output_node = graph.getNode(prior_terminal))
            output_node->is_final_output = true;

        const auto make_boundary =
            [&](MoEOverlayEpochBoundaryStage::Operation operation,
                const char *stage_name)
        {
            MoEOverlayEpochBoundaryStage::Params params;
            params.device_id = state_.device_id;
            params.arena = moe_overlay_epoch_execution_binding_.arena;
            params.request_slot =
                moe_overlay_epoch_execution_binding_.request_slot;
            params.operation = operation;
            if (operation ==
                MoEOverlayEpochBoundaryStage::Operation::Release)
            {
                params.retirement_readiness_controller =
                    moe_overlay_epoch_execution_binding_
                        .retirement_readiness_controller;
            }
            params.stage_name = stage_name;
            return std::make_unique<MoEOverlayEpochBoundaryStage>(
                std::move(params));
        };

        try
        {
            graph.addNode(
                kAcquireNode,
                make_boundary(
                    MoEOverlayEpochBoundaryStage::Operation::Acquire,
                    kAcquireNode),
                state_.device_id);
            graph.addNode(
                kReleaseNode,
                make_boundary(
                    MoEOverlayEpochBoundaryStage::Operation::Release,
                    kReleaseNode),
                state_.device_id);
        }
        catch (const std::exception &exception)
        {
            if (error)
                *error = exception.what();
            return false;
        }

        /* Every original reader is downstream of the acquire. */
        for (const std::string &root : prior_roots)
            graph.addDependency(root, kAcquireNode);
        /* The release is downstream of every original leaf, including any HIP
         * cadence stage, so no auxiliary MoE stream can outlive the request
         * ticket. */
        for (const std::string &leaf : prior_leaves)
            graph.addDependency(kReleaseNode, leaf);
        try
        {
            if (requiresHeterogeneousTicketSegmentation(
                    graph.nativeCaptureEnvelope()))
            {
                /* The model builder sealed its logical tensor producer before
                 * this control-only decorator ran. Move the typed ticket-unit
                 * terminal to Release as one graph mutation so capture cannot
                 * close the heterogeneous transaction at the earlier LM head. */
                graph.moveHeterogeneousTicketTransactionTerminal(
                    prior_terminal, kReleaseNode);
            }
            else
            {
                graph.setTerminalNode(kReleaseNode);
            }
        }
        catch (const std::exception &exception)
        {
            if (error)
                *error = exception.what();
            return false;
        }

        PerfStatsCollector::addCounter(
            "moe_overlay_epoch",
            "captured_main_forward_transactions_declared",
            1.0,
            input.execution_phase == ForwardExecutionPhase::Decode
                ? "decode"
                : "prefill",
            state_.device_id.toString(),
            {{"ownership", "device_graph_root_terminal"},
             {"request_slot",
              std::to_string(
                  moe_overlay_epoch_execution_binding_.request_slot)}});
        return true;
    }

    DeviceMoERebalanceGraphControllerState *
    DeviceGraphOrchestrator::deviceMoERebalanceControllerStateDevice(
        std::string *error)
    {
        if (!usesParticipantLocalDeviceMoERebalanceController())
        {
            if (error)
                *error = "participant-local device maintenance authority is not selected";
            return nullptr;
        }
        if (!device_moe_rebalance_maintenance_graph_.graph)
        {
            if (error)
                *error = "retained maintenance graph is absent";
            return nullptr;
        }
        if (!workspace_allocator_)
        {
            if (error)
                *error = "workspace allocator is absent";
            return nullptr;
        }

        DeviceWorkspaceManager *workspace =
            workspace_allocator_->getDeviceWorkspace(state_.device_id);
        if (!workspace)
        {
            if (error)
                *error = "device workspace is absent";
            return nullptr;
        }

        const MoEDeviceRebalanceStage *rebalance_stage = nullptr;
        const auto &graph = *device_moe_rebalance_maintenance_graph_.graph;
        for (const auto &node_name : graph.getExecutionOrder())
        {
            const ComputeNode *node = graph.getNode(node_name);
            if (!node || !node->stage)
                continue;
            const auto *candidate =
                dynamic_cast<const MoEDeviceRebalanceStage *>(
                    node->stage.get());
            if (!candidate)
                continue;
            rebalance_stage = candidate;
            if (candidate->getParams().phase ==
                DeviceMoERebalanceStagePhase::PlanCopyApply)
            {
                break;
            }
        }
        if (!rebalance_stage)
        {
            if (error)
                *error = "retained maintenance graph has no rebalance stage";
            return nullptr;
        }

        const auto &params = rebalance_stage->getParams();
        const std::string &workspace_name =
            params.workspace_name.empty()
                ? params.stage_name
                : params.workspace_name;
        void *controller = workspace->getBuffer(
            MoEDeviceRebalanceStage::workspaceBufferName(
                MoEDeviceRebalanceStage::WS_CONTROLLER_STATE,
                workspace_name));
        if (!controller && error)
        {
            *error =
                "device workspace has no controller buffer named " +
                MoEDeviceRebalanceStage::workspaceBufferName(
                    MoEDeviceRebalanceStage::WS_CONTROLLER_STATE,
                    workspace_name);
        }
        return static_cast<DeviceMoERebalanceGraphControllerState *>(
            controller);
    }

    bool DeviceGraphOrchestrator::
        initializeDeviceMoERebalanceDispatchTicketOnStream(
            void *stream,
            uint64_t ticket_session_epoch)
    {
        if (!usesParticipantLocalDeviceMoERebalanceController())
        {
            return true;
        }
        const uint64_t generation =
            device_moe_rebalance_maintenance_graph_.workspace_generation;
        if (!state_.device_id.is_gpu() || !stream ||
            ticket_session_epoch == 0u || generation == 0u)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device MoE ticket initialization has incomplete lifecycle identity"
                      << " device=" << state_.device_id.toString()
                      << " session=" << ticket_session_epoch
                      << " workspace_generation=" << generation
                      << " stream=" << stream);
            return false;
        }

        std::string controller_lookup_error;
        DeviceMoERebalanceGraphControllerState *const controller =
            deviceMoERebalanceControllerStateDevice(&controller_lookup_error);
        if (!controller)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device MoE ticket initialization has no controller: "
                      << controller_lookup_error);
            return false;
        }

        const MoEDeviceRebalanceStage *transaction_owner = nullptr;
        const auto &graph =
            *device_moe_rebalance_maintenance_graph_.graph;
        for (const auto &node_name : graph.getExecutionOrder())
        {
            const ComputeNode *node = graph.getNode(node_name);
            const auto *stage =
                node && node->stage
                    ? dynamic_cast<const MoEDeviceRebalanceStage *>(
                          node->stage.get())
                    : nullptr;
            if (stage && stage->ownsRequestTransactionState())
            {
                if (transaction_owner)
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Device MoE ticket initialization found multiple transaction owners");
                    return false;
                }
                transaction_owner = stage;
            }
        }
        if (!transaction_owner)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device MoE ticket initialization found no transaction owner");
            return false;
        }

        const auto &config = transaction_owner->getParams().config;
        IBackend *const backend = getBackendFor(state_.device_id);
        auto *const dispatch_ticket_device =
            reinterpret_cast<DeviceMoERebalanceDispatchTicket *>(
                reinterpret_cast<std::byte *>(controller) +
                offsetof(
                    DeviceMoERebalanceGraphControllerState,
                    dispatch_ticket));
        if (!backend ||
            !backend->enqueueInitializeDeviceMoERebalanceDispatchTicket(
                ticket_session_epoch,
                generation,
                config.participant_id,
                config.participant_count,
                dispatch_ticket_device,
                state_.device_id.gpu_ordinal(),
                stream))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Backend rejected device MoE ticket initialization"
                      << " device=" << state_.device_id.toString()
                      << " participant=" << config.participant_id
                      << " participants=" << config.participant_count);
            return false;
        }

        /* Host fields track observation ownership only. Clearing them cannot
         * affect the device controller and prevents a ticket from the prior
         * request being resubmitted after the reset event is published. */
        last_device_moe_rebalance_dispatch_ticket_.reset();
        device_moe_rebalance_dispatch_ticket_copy_pending_ = false;
        hosted_device_moe_rebalance_last_committed_round_ = 0u;
        return true;
    }

    bool DeviceGraphOrchestrator::
        prepareDeviceMoERebalanceCadenceTransactionSources(
            IWorkerGPUContext *gpu_ctx,
            IBackend *backend,
            void *maintenance_stream,
            std::string *error)
    {
        auto &cache = device_moe_rebalance_maintenance_graph_;
        if (error)
            error->clear();
        if (cache.cadence_transaction_capability_checked)
            return true;
        if (!gpu_ctx || !backend || !maintenance_stream)
        {
            if (error)
                *error = "cadence source capture has no GPU context, backend, or stream";
            return false;
        }

        std::unique_ptr<IGPUGraphCapture> parent;
        try
        {
            parent = gpu_ctx->createGraphCapture(maintenance_stream);
        }
        catch (const std::exception &exception)
        {
            if (error)
                *error = "cadence parent creation failed: " +
                         std::string(exception.what());
            return false;
        }
        if (!parent)
        {
            if (error)
                *error = "GPU context returned no cadence parent owner";
            return false;
        }
        const bool native_transaction =
            parent->supportsDeviceControlledTransaction();
        const std::string capture_backend = parent->backendName();

        DeviceMoERebalanceHostedObservationSchedule hosted_schedule;
        if (!native_transaction)
        {
            const MoEDeviceRebalanceStage *transaction_owner = nullptr;
            for (const auto &node_name : cache.graph->getExecutionOrder())
            {
                const ComputeNode *const node = cache.graph->getNode(node_name);
                const auto *const stage =
                    node && node->stage
                        ? dynamic_cast<const MoEDeviceRebalanceStage *>(
                              node->stage.get())
                        : nullptr;
                if (!stage || !stage->ownsRequestTransactionState())
                    continue;
                if (transaction_owner)
                {
                    if (error)
                        *error = "hosted cadence has multiple transaction owners";
                    return false;
                }
                transaction_owner = stage;
            }
            if (!transaction_owner || !graph_builder_)
            {
                if (error)
                    *error = "hosted cadence has no immutable transaction configuration";
                return false;
            }

            const auto &transaction_config =
                transaction_owner->getParams().config;
            const bool mtp_enabled = graph_builder_->config().mtp.enabled;
            const uint32_t maximum_committed_rounds_per_boundary =
                mtp_enabled
                    ? static_cast<uint32_t>(
                          std::max(1, mtp_max_draft_depth_) + 1)
                    : 1u;
            hosted_schedule = {
                .initial_round_interval =
                    transaction_config.initial_maintenance_period_tokens,
                .recurring_round_interval =
                    transaction_config.maintenance_period_tokens,
                .maximum_committed_rounds_per_boundary =
                    maximum_committed_rounds_per_boundary,
            };
            if (!hosted_schedule.valid())
            {
                if (error)
                    *error = "hosted cadence produced an invalid observation schedule";
                return false;
            }
        }

        std::string controller_error;
        DeviceMoERebalanceGraphControllerState *const controller =
            deviceMoERebalanceControllerStateDevice(&controller_error);
        if (!controller)
        {
            if (error)
                *error = "cadence source capture has no controller: " +
                         controller_error;
            return false;
        }
        auto *const controller_bytes =
            reinterpret_cast<std::byte *>(controller);
        auto *const magic = reinterpret_cast<uint32_t *>(
            controller_bytes +
            offsetof(DeviceMoERebalanceGraphControllerState, magic));
        auto *const version = reinterpret_cast<uint32_t *>(
            controller_bytes +
            offsetof(DeviceMoERebalanceGraphControllerState, version));
        auto *const controller_error_device = reinterpret_cast<uint32_t *>(
            controller_bytes +
            offsetof(
                DeviceMoERebalanceGraphControllerState,
                last_error_code));
        auto *const committed = reinterpret_cast<uint32_t *>(
            controller_bytes +
            offsetof(
                DeviceMoERebalanceGraphControllerState,
                decode_rounds_committed));
        auto *const remaining = reinterpret_cast<uint32_t *>(
            controller_bytes +
            offsetof(
                DeviceMoERebalanceGraphControllerState,
                decode_rounds_until_maintenance));
        auto *const due = reinterpret_cast<uint32_t *>(
            controller_bytes +
            offsetof(
                DeviceMoERebalanceGraphControllerState,
                maintenance_due));
        auto *const advanced = reinterpret_cast<uint32_t *>(
            controller_bytes +
            offsetof(
                DeviceMoERebalanceGraphControllerState,
                decode_boundary_advanced));
        auto *const dispatch_ticket =
            reinterpret_cast<DeviceMoERebalanceDispatchTicket *>(
                controller_bytes +
                offsetof(
                    DeviceMoERebalanceGraphControllerState,
                    dispatch_ticket));

        std::unique_ptr<IGPUGraphCapture> publication;
        std::unique_ptr<IGPUGraphCapture> acknowledgement;
        try
        {
            publication = gpu_ctx->createGraphCapture(maintenance_stream);
            acknowledgement = gpu_ctx->createGraphCapture(maintenance_stream);
        }
        catch (const std::exception &exception)
        {
            if (error)
                *error = "cadence fragment owner creation failed: " +
                         std::string(exception.what());
            return false;
        }
        if (!publication || !acknowledgement)
        {
            if (error)
                *error = "GPU context returned an incomplete cadence fragment pair";
            return false;
        }

        const auto capture_fragment =
            [&](IGPUGraphCapture &capture,
                const std::function<bool()> &enqueue,
                const char *name) -> bool
        {
            if (!capture.beginCapture())
            {
                if (error)
                    *error = std::string("could not begin ") + name +
                             " capture";
                return false;
            }
            const bool enqueued = enqueue();
            /* End capture exactly once even when enqueueing fails; otherwise
             * the shared maintenance stream remains poisoned for every graph. */
            const bool ended = capture.endCapture();
            if (!enqueued || !ended || capture.nodeCount() == 0)
            {
                if (error)
                    *error = std::string("could not capture complete ") + name;
                return false;
            }
            return true;
        };

        if (!capture_fragment(
                *publication,
                [&]()
                {
                    if (native_transaction)
                    {
                        return backend
                            ->enqueuePublishSerialDecodeCommitBoundary(
                                committed,
                                remaining,
                                due,
                                advanced,
                                state_.device_id.gpu_ordinal(),
                                maintenance_stream);
                    }
                    /* HIP main decode or MTP accepted-state publication already
                     * advanced this exact edge. The hosted fragment may only
                     * authenticate the device-owned state; advancing here would
                     * double count conservative early observations. */
                    return backend
                        ->enqueuePublishDeviceMoERebalanceDispatchTicket(
                            magic,
                            version,
                            controller_error_device,
                            committed,
                            remaining,
                            due,
                            advanced,
                            dispatch_ticket,
                            state_.device_id.gpu_ordinal(),
                            maintenance_stream);
                },
                "decode-boundary publication") ||
            !capture_fragment(
                *acknowledgement,
                [&]()
                {
                    return backend->enqueueAcknowledgeDecodeCommitBoundary(
                        remaining,
                        due,
                        advanced,
                        state_.device_id.gpu_ordinal(),
                        maintenance_stream);
                },
                "decode-boundary acknowledgement"))
        {
            return false;
        }

        /* CUDA clones these source graphs into one native conditional parent,
         * whereas HIP launches only its ticket publisher and an occasional
         * acknowledgement directly. Both standalone HIP fragments therefore
         * need setup-time executables. Delaying instantiation until the first
         * observed boundary would put graph allocation in the hot path. */
        if (!native_transaction &&
            (!publication->instantiate() ||
             !publication->hasExecutable() ||
             !acknowledgement->instantiate() ||
             !acknowledgement->hasExecutable()))
        {
            if (error)
            {
                *error =
                    "host-scheduled cadence fragments could not be instantiated";
            }
            return false;
        }

        cache.cadence_boundary_publication_capture =
            std::move(publication);
        cache.cadence_boundary_acknowledgement_capture =
            std::move(acknowledgement);
        cache.hosted_observation_schedule = hosted_schedule;
        if (native_transaction)
            cache.cadence_transaction_capture = std::move(parent);
        cache.cadence_transaction_capability_checked = true;
        PerfStatsCollector::addCounter(
            "moe_rebalance",
            "device_maintenance_cadence_native_capability_checks",
            1.0,
            "graph_setup",
            state_.device_id.toString(),
            {{"backend", capture_backend},
             {"native_transaction", boolTag(native_transaction)},
             {"scheduler",
              native_transaction ? "device_conditional"
                                 : "authenticated_dispatch_ticket"}});
        return true;
    }

    bool DeviceGraphOrchestrator::
        composeDeviceMoERebalanceCadenceTransaction(std::string *error)
    {
        auto &cache = device_moe_rebalance_maintenance_graph_;
        if (error)
            error->clear();
        if (!cache.cadence_transaction_capability_checked)
        {
            if (error)
                *error = "cadence transaction capability was not preflighted";
            return false;
        }
        if (!cache.cadence_transaction_capture)
            return true;
        if (cache.cadence_transaction_capture->hasExecutable())
            return true;
        if (!cache.segment_cache.initialized ||
            cache.segment_cache.needs_capture)
        {
            // The first production maintenance transaction owns source capture.
            return true;
        }
        if (!cache.cadence_boundary_publication_capture ||
            !cache.cadence_boundary_acknowledgement_capture)
        {
            if (error)
                *error = "native cadence parent has incomplete source fragments";
            return false;
        }

        std::string template_error;
        const auto maintenance = cache.segment_cache.deviceLoopGraphTemplate(
            *cache.graph,
            &template_error);
        if (!maintenance || !maintenance->capture)
        {
            if (error)
                *error = "maintenance graph cannot be composed: " +
                         template_error;
            return false;
        }
        std::string controller_error;
        const DeviceMoERebalanceGraphControllerState *const controller =
            deviceMoERebalanceControllerStateDevice(&controller_error);
        if (!controller)
        {
            if (error)
                *error = "cadence composition has no controller: " +
                         controller_error;
            return false;
        }
        const auto *const due = reinterpret_cast<const uint32_t *>(
            reinterpret_cast<const std::byte *>(controller) +
            offsetof(
                DeviceMoERebalanceGraphControllerState,
                maintenance_due));
        const std::array<DeviceControlledLoopFragment, 3> fragments = {{
            {
                .name = "decode boundary publication",
                .capture =
                    cache.cadence_boundary_publication_capture.get(),
            },
            {
                .name = "conditional device MoE maintenance",
                .capture = maintenance->capture,
                .execution =
                    DeviceControlledLoopFragmentExecution::
                        IfDeviceWordNonZero,
                .condition_word_device = due,
            },
            {
                .name = "decode boundary acknowledgement",
                .capture =
                    cache.cadence_boundary_acknowledgement_capture.get(),
            },
        }};
        if (!cache.cadence_transaction_capture
                 ->buildDeviceControlledTransaction(fragments) ||
            !cache.cadence_transaction_capture->instantiate() ||
            !cache.cadence_transaction_capture->hasExecutable())
        {
            if (error)
                *error = "native cadence transaction could not be built and instantiated";
            return false;
        }

        PerfStatsCollector::addCounter(
            "moe_rebalance",
            "device_maintenance_cadence_transaction_materializations",
            1.0,
            "graph_setup",
            state_.device_id.toString(),
            {{"backend",
              cache.cadence_transaction_capture->backendName()},
             {"fragments", "3"},
             {"conditional_fragments", "1"},
             {"maintenance_nodes",
              std::to_string(maintenance->captured_node_count)},
             {"workspace_generation",
              std::to_string(cache.workspace_generation)}});
        return true;
    }
} // namespace llaminar2
