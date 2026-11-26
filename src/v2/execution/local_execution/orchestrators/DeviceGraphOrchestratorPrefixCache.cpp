/**
 * @file DeviceGraphOrchestratorPrefixCache.cpp
 * @brief DeviceGraphOrchestrator method shard (41846..46667).
 */
#include "DeviceGraphOrchestratorInternal.h"

namespace llaminar2
{

    bool DeviceGraphOrchestrator::prepareLiveStateForForwardGraphExecution(
        const ForwardInput &input,
        void *execution_stream,
        DeviceId execution_device)
    {
        if (!state_.device_id.is_gpu())
            return true;
        // Every role here executes the main model, including restored-prefix
        // MTP conditions and grouped verification. A full prefix hit can skip
        // MainInference prefill entirely, so the first real condition must arm
        // the same request interval. Repeated entries remain epoch-idempotent.
        // Predictor sidecars do not enter this main-model forward prelude.
        // Setup-only graph materialization neither starts I/O nor launches it.
        if(input.graph_submission_intent!=ForwardGraphSubmissionIntent::MaterializeExecutableWithoutLaunch&&
            !beginRetainedStorageAdmission(execution_stream))return false;
        if (pending_device_resident_logical_state_forward_read_.valid())
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Forward graph reached a new live-state prelude before the prior graph published its mailbox read completion");
            std::terminate();
        }

        const bool request_consumers_started_before_prelude =
            request_input_reuse_ready_.consumers_started;
        const bool request_token_admission_pending_before_prelude =
            request_input_admission_ready_.valid &&
            request_input_admission_ready_.token_count==1 &&
            request_input_admission_ready_.request_count==1 &&
            input.token_ids_device==request_token_ids_dev_;
        const auto epoch_submission_policy =
            moeOverlayForwardEpochSubmissionPolicy(
                static_cast<bool>(moe_overlay_epoch_execution_binding_),
                input.execution_role == ForwardExecutionRole::MainInference);
        const auto admit_forward_epoch = [&]() -> bool
        {
            using Policy = MoEOverlayForwardEpochSubmissionPolicy;
            if (epoch_submission_policy == Policy::Unbound)
                return true;
            if (!execution_stream)
            {
                LOG_ERROR(
                    "[DeviceGraphOrchestrator] ExpertOverlay forward admission has no exact execution stream on "
                    << state_.device_id.toString());
                return false;
            }

            if (epoch_submission_policy == Policy::CapturedMainTransaction)
            {
                auto submission =
                    moe_overlay_epoch_lease_lifecycle_->beginSubmission();
                const auto lease_state = submission.state();
                const auto predecessor_state = lease_state;
                if (lease_state != MoEOverlayEpochLeaseState::Idle &&
                    lease_state !=
                        MoEOverlayEpochLeaseState::ReleasePublished)
                {
                    LOG_ERROR(
                        "[DeviceGraphOrchestrator] Captured main ExpertOverlay graph overlaps another lease phase on "
                        << state_.device_id.toString()
                        << " lease_state="
                        << moeOverlayEpochLeaseStateName(lease_state));
                    return false;
                }
                if (predecessor_state ==
                    MoEOverlayEpochLeaseState::ReleasePublished)
                {
                    IBackend *const backend =
                        getBackendFor(state_.device_id);
                    if (!backend || !moe_overlay_epoch_released_event_ ||
                        !submission.publishedReleaseProducerStream() ||
                        !backend->streamWaitEvent(
                            execution_stream,
                            moe_overlay_epoch_released_event_.get(),
                            state_.device_id.gpu_ordinal()))
                    {
                        LOG_ERROR(
                            "[DeviceGraphOrchestrator] Captured main ExpertOverlay graph could not join the prior terminal release on "
                            << state_.device_id.toString());
                        return false;
                    }
                }
                if (!submission.commit(
                        MoEOverlayEpochLeaseState::CapturedMainForward))
                {
                    LOG_ERROR(
                        "[DeviceGraphOrchestrator] Captured main ExpertOverlay graph could not publish its complete submission owner on "
                        << state_.device_id.toString());
                    std::terminate();
                }
            }
            else if (moe_overlay_epoch_lease_lifecycle_->load() ==
                     MoEOverlayEpochLeaseState::ExternalSequence)
            {
                auto &identity = moe_overlay_external_sequence_identity_;
                const bool final_verifier_matches =
                    input.execution_role ==
                        ForwardExecutionRole::GroupedMTPVerifier &&
                    identity.valid() &&
                    input.moe_overlay_collective_generation_id ==
                        identity.request_generation &&
                    input.moe_overlay_sequence_id == identity.sequence_id &&
                    input.moe_overlay_sequence_placement_epoch ==
                        identity.placement_epoch &&
                    input.moe_overlay_sequence_graph_count ==
                        identity.graph_count &&
                    input.moe_overlay_sequence_graph_ordinal ==
                        identity.graph_count - 1 &&
                    identity.last_graph_ordinal ==
                        input.moe_overlay_sequence_graph_ordinal - 1;
                if (!final_verifier_matches)
                {
                    LOG_ERROR(
                        "[DeviceGraphOrchestrator] Grouped verifier does not "
                        "close the exact live ExpertOverlay MTP sequence"
                        << " device=" << state_.device_id.toString()
                        << " live_request=" << identity.request_generation
                        << " incoming_request="
                        << input.moe_overlay_collective_generation_id
                        << " live_sequence=" << identity.sequence_id
                        << " incoming_sequence="
                        << input.moe_overlay_sequence_id
                        << " live_ordinal="
                        << identity.last_graph_ordinal
                        << " incoming_ordinal="
                        << input.moe_overlay_sequence_graph_ordinal
                        << " live_count=" << identity.graph_count
                        << " incoming_count="
                        << input.moe_overlay_sequence_graph_count
                        << " live_epoch=" << identity.placement_epoch
                        << " incoming_epoch="
                        << input.moe_overlay_sequence_placement_epoch);
                    return false;
                }

                auto submission =
                    moe_overlay_epoch_lease_lifecycle_->beginSubmission();
                const auto sequence_state = submission.state();
                if (sequence_state !=
                    MoEOverlayEpochLeaseState::ExternalSequence)
                {
                    LOG_ERROR(
                        "[DeviceGraphOrchestrator] Grouped verifier lost its "
                        "ExpertOverlay sequence lease claim"
                        << " device=" << state_.device_id.toString()
                        << " lease_state="
                        << moeOverlayEpochLeaseStateName(sequence_state));
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
                        "[DeviceGraphOrchestrator] Grouped verifier could not "
                        "consume the MTP sequence acquire publication on "
                        << state_.device_id.toString());
                    return false;
                }
                identity.last_graph_ordinal =
                    input.moe_overlay_sequence_graph_ordinal;
                if (!submission.commit(
                        MoEOverlayEpochLeaseState::ExternalForward))
                {
                    LOG_ERROR(
                        "[DeviceGraphOrchestrator] Grouped verifier could not publish its complete sequence-forward owner on "
                        << state_.device_id.toString());
                    std::terminate();
                }
                PerfStatsCollector::addCounter(
                    "moe_overlay_epoch",
                    "external_sequence_verifier_joins",
                    1.0,
                    perfPhaseName(),
                    state_.device_id.toString(),
                    {{"sequence_id", std::to_string(identity.sequence_id)},
                     {"graph_ordinal",
                      std::to_string(identity.last_graph_ordinal)},
                     {"graph_count", std::to_string(identity.graph_count)},
                     {"blocking", "false"}});
            }
            else if (moe_overlay_epoch_lease_lifecycle_->load() ==
                     MoEOverlayEpochLeaseState::HostedParent)
            {
                auto submission =
                    moe_overlay_epoch_lease_lifecycle_->beginSubmission();
                const auto hosted_state = submission.state();
                if (hosted_state != MoEOverlayEpochLeaseState::HostedParent)
                {
                    LOG_ERROR(
                        "[DeviceGraphOrchestrator] Hosted ExpertOverlay child lost its atomic parent lease claim on "
                        << state_.device_id.toString()
                        << " lease_state="
                        << moeOverlayEpochLeaseStateName(hosted_state));
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
                        "[DeviceGraphOrchestrator] Hosted ExpertOverlay child forward could not consume its parent reader publication on "
                        << state_.device_id.toString());
                    return false;
                }
                if (!submission.commit(
                        MoEOverlayEpochLeaseState::HostedChildForward))
                {
                    LOG_ERROR(
                        "[DeviceGraphOrchestrator] Hosted ExpertOverlay child could not publish its complete forward owner on "
                        << state_.device_id.toString());
                    std::terminate();
                }
            }
            else
            {
                if (input.execution_role ==
                        ForwardExecutionRole::GroupedMTPVerifier &&
                    input.moe_overlay_sequence_id != 0)
                {
                    LOG_ERROR(
                        "[DeviceGraphOrchestrator] Direct grouped verifier "
                        "reached forward admission without its complete "
                        "ExpertOverlay MTP sequence lease"
                        << " device=" << state_.device_id.toString()
                        << " sequence_id="
                        << input.moe_overlay_sequence_id
                        << " lease_state="
                        << moeOverlayEpochLeaseStateName(
                               moe_overlay_epoch_lease_lifecycle_->load()));
                    return false;
                }
                if (!acquireMoEOverlayEpochForExternalTransaction(
                        execution_stream,
                        DeviceTimelineRole::MainForwardGraph,
                        "forward_graph_exact_submission",
                        MoEOverlayEpochLeaseState::ExternalForward))
                {
                    LOG_ERROR(
                        "[DeviceGraphOrchestrator] Retained ExpertOverlay forward could not acquire its exact per-submission reader on "
                        << state_.device_id.toString());
                    return false;
                }
            }
            moe_overlay_epoch_forward_submission_stream_ =
                execution_stream;
            const auto admitted_state =
                moe_overlay_epoch_lease_lifecycle_->load();
            PerfStatsCollector::addCounter(
                "moe_overlay_epoch",
                "forward_submission_admissions",
                1.0,
                perfPhaseName(),
                state_.device_id.toString(),
                {{"lifetime", "one_forward_submission"},
                 {"boundary_owner",
                  epoch_submission_policy == Policy::CapturedMainTransaction
                      ? "captured_main_graph"
                      : admitted_state ==
                                MoEOverlayEpochLeaseState::HostedChildForward
                            ? "borrowed_hosted_parent"
                            : "retained_external_graphs"},
                 {"blocking", "false"}});
            return true;
        };
        auto admit_serial_decode_position = [&]() -> bool
        {
            if (!input.device_decode_position)
                return true;
            if (!execution_stream ||
                input.execution_role !=
                    ForwardExecutionRole::MainInference ||
                input.batch_size != 1 ||
                input.seq_len != 1 ||
                input.position_ids_device != request_position_ids_dev_ ||
                !state_.kv_cache)
            {
                LOG_ERROR(
                    "[DeviceGraphOrchestrator] Device-owned serial-decode "
                    "position materialization received an invalid graph "
                    "transaction");
                return false;
            }

            const int32_t *const live_position_device =
                state_.kv_cache
                    ->deviceSequenceCachedTokenCountPtr(/*seq_idx=*/0);
            IBackend *const backend = getBackendFor(state_.device_id);
            if (!live_position_device || !backend ||
                input.device_decode_position->backend != backend ||
                input.device_decode_position->cached_tokens != live_position_device ||
                input.device_decode_position->position != request_position_ids_dev_)
            {
                LOG_ERROR(
                    "[DeviceGraphOrchestrator] Device-owned serial decode "
                    "cannot resolve its live KV position");
                return false;
            }

            auto &reuse = request_input_reuse_ready_;
            if (request_consumers_started_before_prelude)
            {
                LOG_ERROR(
                    "[DeviceGraphOrchestrator] Serial decode reached a request "
                    "input bank whose prior graph never published reuse "
                    "readiness");
                std::terminate();
            }
            if (reuse.valid)
            {
                if (!reuse.event || !reuse.producer_stream ||
                    !DeviceEventEdge::at(
                         DeviceTimelinePoint::RequestInputReuseReady)
                         .from(DeviceTimelineRole::MainForwardGraph)
                         .to(DeviceTimelineRole::MainForwardGraph)
                         .enqueueWait(
                             *backend,
                             state_.device_id,
                             reuse.event.get(),
                             reuse.producer_stream,
                             execution_stream))
                {
                    LOG_ERROR(
                        "[DeviceGraphOrchestrator] Failed to order serial "
                        "decode position publication after the prior graph");
                    std::terminate();
                }
                // Position-only serial admission reuses the same input bank
                // as host prefill. Report its real release/acquire edge too;
                // observers must not mistake omitted telemetry for no wait.
                PerfStatsCollector::addCounter(
                    "request_admission", "device_input_reuse_waits", 1.0,
                    "decode", state_.device_id.toString(),
                    {{"same_stream", boolTag(reuse.producer_stream == execution_stream)},
                     {"ordering", "consumer_release_event"},
                     {"admission", "serial_decode_position_snapshot"}});
                reuse.valid = false;
                reuse.producer_stream = nullptr;
            }

            // Admission joins prior readers once. The captured root owns the
            // actual snapshot, so native parent iterations no longer depend on
            // executing this host prelude between child replays.
            const bool shared_with_token_admission =
                reuse.consumers_started;
            if (!shared_with_token_admission)
            {
                beginRequestInputReaderTransaction(
                    "serial_decode_position_snapshot",
                    /*token_count=*/1,
                    /*request_count=*/1);
            }
            else if (!(request_token_admission_pending_before_prelude &&
                       !request_input_admission_ready_.valid &&
                       input.token_ids_device==request_token_ids_dev_) &&
                     input.state_transaction != ForwardStateTransaction::RestoredPrefixMTPDecodeBridge)
            {
                LOG_ERROR(
                    "[DeviceGraphOrchestrator] Serial decode unexpectedly "
                    "shared an active request-input reader transaction");
                std::terminate();
            }
            PerfStatsCollector::addCounter(
                "request_admission",
                "device_serial_decode_position_admissions",
                1.0,
                "decode",
                state_.device_id.toString(),
                {{"owner", "canonical_kv_count"},
                 {"destination", "arena_request_position"},
                 {"transfer", "d2d"},
                 {"ordering", "captured_root_after_event_wait"},
                 {"token_admission",
                  shared_with_token_admission
                      ? "shared_transaction"
                      : "position_only"}});
            return true;
        };
        if (!waitForPendingRequestStateReset(
                execution_stream,
                DeviceTimelineRole::MainForwardGraph,
                "forward_graph_request_state_reset"))
        {
            return false;
        }
        if (!waitForPendingGraphBuildDeviceStateReady(
                main_graph_build_device_state_ready_,
                execution_stream,
                DeviceTimelineRole::MainForwardGraph,
                "forward_graph_build_device_state"))
        {
            return false;
        }
        if (!waitForPendingRequestInputAdmission(
                execution_stream,
                "forward_graph_request_input"))
        {
            return false;
        }
        if (!waitForPendingDeviceMoERebalanceMaintenance(
                execution_stream,
                DeviceTimelineRole::MainForwardGraph,
                "forward_graph"))
        {
            return false;
        }
        /*
         * A KV-only MTP sidecar still reads the preceding main terminal hidden
         * row. It publishes shifted-KV completion, not a logits handoff. Join
         * that read retirement before this graph can overwrite its source.
         * This is a non-consuming observation: the next sidecar/cache mutation
         * retains its own completion obligation. Keep it before the no-other-
         * publications fast path, including budget-limited and forced tokens.
         */
        if (!waitForPendingShiftedMTPKVReadyForObservation(
                execution_stream,
                "forward_graph_previous_terminal_hidden_reader"))
        {
            return false;
        }
        // A follower has no sampler/mailbox edge between forwards. A resident
        // prefill family likewise shares activation and chunk metadata across
        // independently captured widths. Acquire the actual preceding terminal
        // on this exact consumer stream before any root can read/write them.
        // Native parent iterations retain their own internal DAG, not this
        // external-submission prelude. No host/device synchronization is added.
        const auto output_role = pp_stage_config_ && !pp_stage_config_->has_lm_head
            ? ForwardParticipantOutputRole::PipelineFollower
            : ForwardParticipantOutputRole::TerminalOwner;
        if (forwardGraphEntryDependency(input, state_.device_id, output_role) ==
                ForwardGraphEntryDependency::PreviousForwardCompletion &&
            !waitForForwardGraphOutputReady(
                execution_stream,
                DeviceTimelineRole::MainForwardGraph,
                ForwardGraphOutputKind::Any,
                output_role == ForwardParticipantOutputRole::PipelineFollower
                    ? "pipeline_follower_previous_forward"
                    : "prefill_previous_bucket"))
        {
            return false;
        }
        // Pipeline admission joins prefix restoration on its publication
        // stream. The initial verifier owns a separate retained execution
        // stream: transfer the controller frontier here, before the empty
        // live-publication fast path. Borrowing on the input-publication
        // stream would leave this graph's KV/GDN reads unordered. Standalone
        // verifier preparation has no admitted controller to borrow.
        if (output_role == ForwardParticipantOutputRole::PipelineFollower &&
            input.execution_role == ForwardExecutionRole::GroupedMTPVerifier &&
            device_generation_storage_.active_request_count != 0)
        {
            if ((execution_device.is_gpu() && execution_device != state_.device_id) ||
                !consumeDeviceGenerationStateReady(execution_stream,
                    DeviceTimelineRole::AllPositionVerifier, 1,
                    "pipeline_initial_mtp_verifier"))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Pipeline verifier could not acquire admission on its exact execution stream");
                return false;
            }
            PerfStatsCollector::addCounter("mtp", "pipeline_initial_verifier_admission_waits",
                1.0, perfPhaseName(), state_.device_id.toString());
        }
        const bool has_accepted_publication =
            accepted_spec_publication_ready_.valid;
        const bool has_live_checkpoint =
            live_prefix_checkpoint_ready_.valid;
        const bool has_prefill_terminal_archive =
            mtp_prefill_terminal_archive_ready_.valid;
        const bool has_live_mutation =
            live_prefix_mutation_ready_.valid;
        const bool has_logical_mailbox =
            device_resident_logical_sequence_state_mailbox_.valid();
        if (!has_accepted_publication &&
            !has_live_checkpoint &&
            !has_prefill_terminal_archive &&
            !has_live_mutation &&
            !has_logical_mailbox)
        {
            return admit_serial_decode_position() &&
                   admit_forward_epoch();
        }
        if (!execution_stream)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Pending resident live-state handoff requires an explicit forward stream");
            return false;
        }

        const DeviceId consumer_device =
            execution_device.is_gpu() ? execution_device : state_.device_id;
        if (consumer_device != state_.device_id)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Resident live-state handoff wait requested on "
                      << consumer_device.toString()
                      << " but live state belongs to " << state_.device_id.toString());
            return false;
        }
        if (!waitForPendingLivePrefixCheckpointReady(
                execution_stream,
                "forward_graph_live_checkpoint_source"))
        {
            return false;
        }
        /*
         * A main graph is one of two users of the terminal-selection event: it
         * must not overwrite the source hidden tensor before archival
         * completes, while the next MTP sidecar still needs to read the
         * archived payload. Queue the source-protection wait without consuming
         * the payload handoff; the sidecar or an explicit terminal refresh
         * remains its semantic owner.
         */
        if (!waitForPendingMTPPrefillTerminalArchiveReadyForObservation(
                execution_stream,
                "forward_graph_mtp_prefill_terminal_archive_source"))
        {
            return false;
        }
        if (!joinPublishedLiveStateHandoffs(
                execution_stream,
                DeviceTimelineRole::MainForwardGraph,
                "forward_graph_live_state_consumer"))
        {
            return false;
        }
        if (!admit_serial_decode_position())
            return false;

        if (has_logical_mailbox)
        {
            uint64_t publication_generation = 0;
            std::shared_lock<std::shared_mutex> admission_lock;
            if (!admitDeviceResidentLogicalSequenceStateRead(
                    execution_stream,
                    "forward_graph_logical_state",
                    &publication_generation,
                    &admission_lock))
            {
                return false;
            }
            if (publication_generation == 0 || !admission_lock.owns_lock())
            {
                LOG_ERROR(
                    "[DeviceGraphOrchestrator] Forward graph logical-state "
                    "admission lost the mailbox it was selected to consume");
                return false;
            }
            pending_device_resident_logical_state_forward_read_.stream =
                execution_stream;
            pending_device_resident_logical_state_forward_read_
                .publication_generation = publication_generation;
            pending_device_resident_logical_state_forward_read_
                .admission_lock = std::move(admission_lock);
        }
        return admit_forward_epoch();
    }

    GraphCaptureAuxiliaryBranchFactory
    DeviceGraphOrchestrator::forwardGraphAuxiliaryBranchFactory(
        DeviceId execution_device)
    {
        if (!moe_overlay_transfer_progress_epoch_)
            return {};
        if (!state_.device_id.is_gpu() ||
            execution_device != state_.device_id ||
            moe_overlay_transfer_progress_epoch_->device() !=
                state_.device_id)
        {
            throw std::logic_error(
                "ExpertOverlay transfer-progress branch received mismatched forward/device ownership");
        }

        // CUDA must enter progress from the graph root, not from a native queue
        // that future inference observers can occupy. Its graph-private native
        // timeline wait consumes no SM between the eager check and a real work
        // notification; the terminal releases an otherwise idle wait without
        // running copy work. ROCm's native SDMA authority intentionally returns
        // an empty compute branch.
        const auto factory = moe_overlay_transfer_progress_epoch_->graphBranchFactory();
        PerfStatsCollector::addCounter(
            "moe_overlay_residency",
            factory.valid() ? "mapped_transfer_progress_forward_branches"
                            : "mapped_transfer_progress_native_sdma_intervals",
            1.0,
            "forward",
            state_.device_id.toString(),
            {{"authority", "maintenance_epoch"},
             {"command_completion_join", "false"}});
        return factory;
    }

    bool DeviceGraphOrchestrator::prepareGraphBuildStateForMaterialization(
        const ForwardInput &input,
        void *capture_stream,
        DeviceId execution_device)
    {
        if (!state_.device_id.is_gpu())
            return true;
        if (input.graph_submission_intent !=
                ForwardGraphSubmissionIntent::
                    MaterializeExecutableWithoutLaunch ||
            !capture_stream || execution_device != state_.device_id ||
            input.device != state_.device_id)
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Graph-build setup handoff received "
                "an invalid capture owner"
                << " input_device=" << input.device.toString()
                << " execution_device=" << execution_device.toString()
                << " owner_device=" << state_.device_id.toString()
                << " stream=" << capture_stream);
            return false;
        }

        return waitForPendingGraphBuildDeviceStateReady(
            main_graph_build_device_state_ready_,
            capture_stream,
            DeviceTimelineRole::MainForwardGraph,
            "setup_graph_materialization_device_state");
    }

    bool DeviceGraphOrchestrator::completeLiveStateForForwardGraphExecution(
        const ForwardInput &input,
        void *execution_stream,
        DeviceId execution_device)
    {
        if (!state_.device_id.is_gpu())
            return true;

        auto &pending =
            pending_device_resident_logical_state_forward_read_;
        if (pending.valid())
        {
            const uint64_t consumed_publication_generation =
                pending.publication_generation;
            if (!execution_stream || execution_device != state_.device_id ||
                execution_stream != pending.stream)
            {
                LOG_ERROR(
                    "[DeviceGraphOrchestrator] Forward graph mailbox read completion did not name its exact admitted stream"
                    << " expected_stream=" << pending.stream
                    << " actual_stream=" << execution_stream
                    << " expected_device=" << state_.device_id.toString()
                    << " actual_device=" << execution_device.toString());
                return false;
            }
            if (!recordDeviceResidentLogicalSequenceStateReadCompletion(
                    execution_stream,
                    "forward_graph_logical_state",
                    pending.publication_generation))
            {
                LOG_ERROR(
                    "[DeviceGraphOrchestrator] Forward graph failed at the resident logical-state reader terminal"
                    << " device=" << state_.device_id.toString()
                    << " stream=" << execution_stream
                    << " publication_generation="
                    << pending.publication_generation
                    << " access_generation="
                    << device_resident_logical_state_access_epoch_
                           .publication_generation
                    << " access_reader_streams="
                    << device_resident_logical_state_access_epoch_
                           .reader_stream_count);
                return false;
            }
            pending.clear();

            /*
             * An MTP-condition graph consumes the mailbox as an immutable
             * input transaction. Its forward output has already advanced
             * canonical KV on this same stream, so retaining the pre-forward
             * token/position row would expose two competing live-state owners.
             * Retire only after the reader completion above is published; the
             * access epoch preserves that event for safe arena-row reuse.
             */
            if (input.execution_role == ForwardExecutionRole::MTPCondition &&
                !retireConsumedMTPConditionInputMailbox(
                    execution_stream,
                    consumed_publication_generation))
            {
                return false;
            }
        }

        const auto overlay_lease_state =
            moe_overlay_epoch_lease_lifecycle_->load();
        if (overlay_lease_state ==
                MoEOverlayEpochLeaseState::CapturedMainForward ||
            overlay_lease_state == MoEOverlayEpochLeaseState::ExternalForward ||
            overlay_lease_state ==
                MoEOverlayEpochLeaseState::HostedChildForward)
        {
            if (!execution_stream || execution_device != state_.device_id ||
                execution_stream !=
                    moe_overlay_epoch_forward_submission_stream_)
            {
                LOG_ERROR(
                    "[DeviceGraphOrchestrator] Topology-wide ExpertOverlay forward completion lost its exact submission stream"
                    << " expected_stream="
                    << moe_overlay_epoch_forward_submission_stream_
                    << " actual_stream=" << execution_stream
                    << " expected_device=" << state_.device_id.toString()
                    << " actual_device=" << execution_device.toString());
                return false;
            }

            const bool captured_main =
                overlay_lease_state ==
                MoEOverlayEpochLeaseState::CapturedMainForward;
            const bool borrowed_hosted_parent =
                overlay_lease_state ==
                MoEOverlayEpochLeaseState::HostedChildForward;
            const bool external_sequence_verifier =
                overlay_lease_state ==
                    MoEOverlayEpochLeaseState::ExternalForward &&
                moe_overlay_external_sequence_identity_.valid();
            if (captured_main)
            {
                if (input.execution_role !=
                    ForwardExecutionRole::MainInference)
                {
                    LOG_ERROR(
                        "[DeviceGraphOrchestrator] Captured main ExpertOverlay completion found contradictory host reader state on "
                        << state_.device_id.toString());
                    return false;
                }

                auto submission =
                    moe_overlay_epoch_lease_lifecycle_->beginSubmission();
                const auto captured_state = submission.state();
                if (captured_state !=
                    MoEOverlayEpochLeaseState::CapturedMainForward)
                {
                    LOG_ERROR(
                        "[DeviceGraphOrchestrator] Captured main ExpertOverlay completion lost lease authority on "
                        << state_.device_id.toString()
                        << " lease_state="
                        << moeOverlayEpochLeaseStateName(captured_state));
                    return false;
                }

                /*
                 * The graph's sole terminal is the captured Release stage.
                 * Recording on the exact launch stream creates an immutable
                 * receipt after that terminal; it does not submit another graph
                 * or inspect the device-owned ticket.
                 */
                IBackend *const backend = getBackendFor(state_.device_id);
                if (!backend || !moe_overlay_epoch_released_event_ ||
                    !backend->recordEvent(
                        moe_overlay_epoch_released_event_.get(),
                        state_.device_id.gpu_ordinal(),
                        execution_stream))
                {
                    LOG_ERROR(
                        "[DeviceGraphOrchestrator] Captured main ExpertOverlay graph could not publish its terminal release receipt on "
                        << state_.device_id.toString());
                    std::terminate();
                }

                moe_overlay_epoch_acquire_producer_stream_ = nullptr;
                if (!submission.publishRelease(execution_stream))
                {
                    LOG_ERROR(
                        "[DeviceGraphOrchestrator] Captured main ExpertOverlay graph could not publish its terminal semantic receipt on "
                        << state_.device_id.toString());
                    std::terminate();
                }
            }
            else if (borrowed_hosted_parent)
            {
                if (input.execution_role ==
                        ForwardExecutionRole::MainInference ||
                    !moe_overlay_epoch_acquire_producer_stream_)
                {
                    LOG_ERROR(
                        "[DeviceGraphOrchestrator] Hosted ExpertOverlay child completion lost its parent-owned reader on "
                        << state_.device_id.toString());
                    return false;
                }
                /*
                 * This forward consumed the parent's acquired-event on its exact
                 * stream. Its semantic replay handoff subsequently joins that
                 * stream back to the hosted scheduler. Clearing only the local
                 * submission recipe leaves the parent ticket untouched until
                 * the typed branch-terminal release fragment runs.
                 */
                auto submission =
                    moe_overlay_epoch_lease_lifecycle_->beginSubmission();
                const auto child_state = submission.state();
                if (child_state !=
                    MoEOverlayEpochLeaseState::HostedChildForward)
                {
                    LOG_ERROR(
                        "[DeviceGraphOrchestrator] Hosted ExpertOverlay child could not return its parent lease on "
                        << state_.device_id.toString()
                        << " lease_state="
                        << moeOverlayEpochLeaseStateName(child_state));
                    return false;
                }
                if (!submission.commit(MoEOverlayEpochLeaseState::HostedParent))
                {
                    LOG_ERROR(
                        "[DeviceGraphOrchestrator] Hosted ExpertOverlay child could not republish its parent owner on "
                        << state_.device_id.toString());
                    std::terminate();
                }
            }
            else
            {
                if (external_sequence_verifier)
                {
                    const auto &identity =
                        moe_overlay_external_sequence_identity_;
                    if (input.execution_role !=
                            ForwardExecutionRole::GroupedMTPVerifier ||
                        input.moe_overlay_collective_generation_id !=
                            identity.request_generation ||
                        input.moe_overlay_sequence_id !=
                            identity.sequence_id ||
                        input.moe_overlay_sequence_placement_epoch !=
                            identity.placement_epoch ||
                        input.moe_overlay_sequence_graph_count !=
                            identity.graph_count ||
                        input.moe_overlay_sequence_graph_ordinal !=
                            identity.graph_count - 1 ||
                        identity.last_graph_ordinal !=
                            input.moe_overlay_sequence_graph_ordinal)
                    {
                        LOG_ERROR(
                            "[DeviceGraphOrchestrator] Grouped verifier "
                            "terminal no longer matches its admitted "
                            "ExpertOverlay MTP sequence");
                        return false;
                    }
                }
                if (!releaseMoEOverlayEpochForExternalTransaction(
                        execution_stream,
                        "forward_graph_exact_submission_terminal",
                        MoEOverlayEpochLeaseState::ExternalForward))
                {
                    return false;
                }
                if (external_sequence_verifier)
                {
                    PerfStatsCollector::addCounter(
                        "moe_overlay_epoch",
                        "external_sequence_releases",
                        1.0,
                        perfPhaseName(),
                        state_.device_id.toString(),
                        {{"sequence_id",
                          std::to_string(
                              moe_overlay_external_sequence_identity_
                                  .sequence_id)},
                         {"graph_count",
                          std::to_string(
                              moe_overlay_external_sequence_identity_
                                  .graph_count)},
                         {"blocking", "false"}});
                    moe_overlay_external_sequence_identity_.reset();
                }
            }

            moe_overlay_epoch_forward_submission_stream_ = nullptr;
            PerfStatsCollector::addCounter(
                "moe_overlay_epoch",
                "forward_submission_terminal_receipts",
                1.0,
                perfPhaseName(),
                state_.device_id.toString(),
                {{"lifetime", "one_forward_submission"},
                 {"forward_role",
                  captured_main
                      ? "main_inference"
                      : borrowed_hosted_parent
                            ? "hosted_parent_child"
                            : "auxiliary_forward"},
                 {"boundary_owner",
                 captured_main
                      ? "captured_main_graph"
                      : borrowed_hosted_parent
                            ? "borrowed_hosted_parent"
                            : "retained_external_graphs"},
                 {"blocking", "false"}});
        }

        if (moe_overlay_inference_transaction_coordinator_ &&
            moe_overlay_inference_completion_fence_ &&
            input.execution_role == ForwardExecutionRole::MainInference &&
            input.execution_phase == ForwardExecutionPhase::Prefill &&
            input.moe_overlay_collective_step_id != 0)
        {
            /*
             * This hook runs after the forward executor has enqueued the graph
             * and after the captured overlay Release terminal above. Recording
             * one dedicated event here gives calibration the same device-side
             * completion boundary as the production graph without blocking the
             * inference thread or reusing the epoch-publication receipt.
             */
            std::string completion_error;
            if (!moe_overlay_inference_transaction_coordinator_
                     ->deferPrefillInterferenceCompletionAtDeviceTerminal(
                         input.moe_overlay_collective_step_id,
                         moe_overlay_inference_transaction_participant_index_,
                         moe_overlay_inference_completion_fence_,
                         execution_stream,
                         &completion_error))
            {
                LOG_ERROR(
                    "[DeviceGraphOrchestrator] ExpertOverlay prefill could not publish its exact device-terminal calibration receipt on "
                    << state_.device_id.toString() << ": "
                    << completion_error);
                return false;
            }
        }
        return true;
    }

    const float *DeviceGraphOrchestrator::getAllPositionLogits() const
    {
        TensorBase *tensor = state_.all_position_logits.get();
        return publishLogitsTensorToHost(
            tensor,
            HostLogitsSurface::AllPositionVerifier,
            "getAllPositionLogits");
    }

    std::string DeviceGraphOrchestrator::mtpDecodeUnsupportedReason() const
    {
        return {};
    }

    bool DeviceGraphOrchestrator::supportsMTPTokenCoordination() const
    {
        const IGlobalTPContext *ctx = globalTPContextForMTPCoordination();
        return ctx && ctx->degree() > 1;
    }

    bool DeviceGraphOrchestrator::publishesFullVocabularyMTPLogits() const
    {
        if (!graph_builder_)
            return false;

        const auto &config = graph_builder_->config();
        return config.mtpParticipantOwnsFullVocabulary() || config.mtpPublishesNativeGatheredVocabulary();
    }

    bool DeviceGraphOrchestrator::supportsMTPSidecarSampleFusion() const
    {
        return state_.device_id.is_gpu();
    }

    bool DeviceGraphOrchestrator::supportsMTPSidecarLogitsStreamHandoff() const
    {
        return graph_builder_ &&
               graph_builder_->config().mtp.enabled &&
               state_.device_id.is_gpu() &&
               debugEnv().execution.gpu_graphs;
    }

    bool DeviceGraphOrchestrator::supportsMTPDeviceDraftTokenInput() const
    {
        return supportsMTPSidecarLogitsStreamHandoff() &&
               mtp_sidecar_condition_token_dev_ &&
               mtp_sidecar_position_ids_dev_ &&
               stochastic_target_sample_tokens_dev_ &&
               stochastic_draft_sample_tokens_dev_;
    }

    bool DeviceGraphOrchestrator::supportsMTPSidecarPreservesMainState() const
    {
        if (!graph_builder_ ||
            !graph_builder_->config().mtp.enabled ||
            !state_.kv_cache ||
            state_.mtp_kv_caches.empty())
        {
            return false;
        }

        /*
         * The model graph owns every stage and buffer in the sidecar, so it is
         * the only layer that can state this mutation contract authoritatively.
         * Backend and topology are deliberately absent: CPU, CUDA, ROCm, and
         * multi-participant runners may skip the restore only when every child
         * graph declares the same isolated ownership contract.
         */
        return graph_builder_->mtpSidecarStateContract().preservesMainState();
    }

    bool DeviceGraphOrchestrator::supportsMTPShiftedRowReuseFromSidecar() const
    {
        if (!supportsMTPSidecarPreservesMainState() || !graph_builder_)
        {
            return false;
        }

        return graph_builder_->mtpSidecarStateContract()
            .reusesSidecarShiftedRow();
    }

    IGlobalTPContext *DeviceGraphOrchestrator::globalTPContextForMTPCoordination() const
    {
        if (global_tp_ctx_)
        {
            return global_tp_ctx_.get();
        }
        if (!graph_builder_)
        {
            return nullptr;
        }

        ITPContext *tp_ctx = graph_builder_->config().tp_ctx;
        if (!tp_ctx || tp_ctx->isLocal())
        {
            return nullptr;
        }

        auto *global_ctx = dynamic_cast<IGlobalTPContext *>(tp_ctx);
        return global_ctx && global_ctx->degree() > 1 ? global_ctx : nullptr;
    }

    int DeviceGraphOrchestrator::sampleGreedyFromMTPLogitsOnDevice()
    {
        auto it = state_.extension_buffers.find(mtpSamplingLogitsBufferId());
        if (it == state_.extension_buffers.end() || !it->second)
        {
            return -1;
        }

        const int token_offset = mtpSidecarLogitsAreColumnParallel() && graph_builder_
                                     ? vocabOffsetForTPConfig(graph_builder_->config())
                                     : 0;
        void *pending_stream = consumePendingLogitsStream(
            PendingLogitsStreamRole::MTPSidecar,
            "sampleGreedyFromMTPLogitsOnDevice");
        TensorBase *mtp_logits = it->second.get();
        if (!pending_stream && mtp_logits)
        {
            auto device_opt = mtp_logits->current_device();
            if (device_opt.has_value() && device_opt->is_gpu())
            {
                pending_stream = explicitGPUStreamForOperation("sampleGreedyOnMTPLogits");
                if (!pending_stream)
                {
                    return -1;
                }
            }
        }
        return coordinateGreedyCandidate(
            sampleGreedyCandidateFromTensor(mtp_logits, 0, token_offset,
                                            argmax_partial_vals_dev_,
                                            argmax_partial_idxs_dev_,
                                            argmax_partial_capacity_,
                                            pending_stream,
                                            "mtp_logits"),
            globalTPContextForMTPCoordination());
    }

    std::vector<MTPMirroredTensorDigest>
    DeviceGraphOrchestrator::captureMirroredMTPDigests(
        void *ordered_stream)
    {
        std::vector<MTPMirroredTensorDigest> digests;
        if (!state_.device_id.is_gpu())
            return digests;

        IBackend *backend = getBackendFor(state_.device_id);
        void *stream = ordered_stream
                           ? ordered_stream
                           : explicitGPUStreamForOperation(
                                 "captureMirroredMTPDigests");
        if (!backend || !stream)
            return digests;

        constexpr uint64_t kFnvOffsetBasis = 1469598103934665603ULL;
        constexpr uint64_t kFnvPrime = 1099511628211ULL;
        auto hash_bytes = [](const uint8_t *bytes, size_t byte_count)
        {
            uint64_t hash = kFnvOffsetBasis;
            for (size_t i = 0; i < byte_count; ++i)
            {
                hash ^= static_cast<uint64_t>(bytes[i]);
                hash *= kFnvPrime;
            }
            return hash;
        };

        /*
         * Proposal sidecars are one-row graphs.  Hashing one logical row keeps
         * preallocated capacity and stale tail rows out of the comparison.
         * Direct backend copies intentionally bypass TensorBase::ensureOnHost:
         * diagnostics observe device truth without changing tensor authority.
         */
        auto append_tensor_row =
            [&](const char *name, const TensorBase *tensor)
        {
            MTPMirroredTensorDigest digest;
            digest.name = name ? name : "(unnamed)";
            if (!tensor || !tensor->gpu_data_ptr() || !tensor->deviceValid())
            {
                digests.push_back(std::move(digest));
                return;
            }

            const size_t rows = std::max<size_t>(1, tensor->rows());
            const size_t total_bytes = tensor->size_bytes();
            if (total_bytes == 0 || total_bytes % rows != 0)
            {
                digests.push_back(std::move(digest));
                return;
            }

            digest.byte_count = total_bytes / rows;
            std::vector<uint8_t> host_bytes(digest.byte_count);
            if (backend->deviceToHostFast(
                    host_bytes.data(),
                    tensor->gpu_data_ptr(),
                    digest.byte_count,
                    state_.device_id.gpu_ordinal(),
                    stream))
            {
                digest.hash = hash_bytes(
                    host_bytes.data(),
                    host_bytes.size());
                digest.available = true;
            }
            digests.push_back(std::move(digest));
        };

        auto append_device_bytes =
            [&](const char *name, const void *device_ptr, size_t byte_count)
        {
            MTPMirroredTensorDigest digest;
            digest.name = name ? name : "(unnamed)";
            digest.byte_count = byte_count;
            if (!device_ptr || byte_count == 0)
            {
                digests.push_back(std::move(digest));
                return;
            }

            std::vector<uint8_t> host_bytes(byte_count);
            if (backend->deviceToHostFast(
                    host_bytes.data(),
                    const_cast<void *>(device_ptr),
                    byte_count,
                    state_.device_id.gpu_ordinal(),
                    stream))
            {
                digest.hash = hash_bytes(
                    host_bytes.data(),
                    host_bytes.size());
                digest.available = true;
            }
            digests.push_back(std::move(digest));
        };

        /*
         * Compact control arrays need their literal values as well as a hash.
         * A hash identifies the first divergent boundary, while the values
         * immediately distinguish a bad token publication from a bad embedding
         * launch. The preview is bounded by the caller and exists only on the
         * already-fatal mirrored-participant mismatch path.
         */
        auto append_device_int32_values =
            [&](const char *name, const int32_t *device_ptr, size_t value_count)
        {
            MTPMirroredTensorDigest digest;
            digest.name = name ? name : "(unnamed)";
            digest.byte_count = value_count * sizeof(int32_t);
            if (!device_ptr || value_count == 0)
            {
                digests.push_back(std::move(digest));
                return;
            }

            std::vector<int32_t> host_values(value_count);
            if (backend->deviceToHostFast(
                    host_values.data(),
                    const_cast<int32_t *>(device_ptr),
                    digest.byte_count,
                    state_.device_id.gpu_ordinal(),
                    stream))
            {
                digest.hash = hash_bytes(
                    reinterpret_cast<const uint8_t *>(host_values.data()),
                    digest.byte_count);
                digest.available = true;
                std::ostringstream preview;
                preview << "{";
                for (size_t index = 0; index < host_values.size(); ++index)
                {
                    if (index != 0)
                        preview << ',';
                    preview << host_values[index];
                }
                preview << "}";
                digest.value_preview = preview.str();
            }
            digests.push_back(std::move(digest));
        };

        auto append_prehashed_bytes =
            [&](std::string name,
                uint64_t hash,
                size_t byte_count,
                bool available)
        {
            MTPMirroredTensorDigest digest;
            digest.name = std::move(name);
            digest.hash = hash;
            digest.byte_count = byte_count;
            digest.available = available;
            digests.push_back(std::move(digest));
        };

        auto append_host_bytes =
            [&](std::string name,
                const void *host_ptr,
                size_t byte_count,
                std::string preview = {})
        {
            MTPMirroredTensorDigest digest;
            digest.name = std::move(name);
            digest.byte_count = byte_count;
            if (host_ptr && byte_count > 0)
            {
                digest.hash = hash_bytes(
                    static_cast<const uint8_t *>(host_ptr),
                    byte_count);
                digest.available = true;
                digest.value_preview = std::move(preview);
            }
            digests.push_back(std::move(digest));
        };

        auto extension = [&](BufferId id) -> const TensorBase *
        {
            const auto it = state_.extension_buffers.find(id);
            return it == state_.extension_buffers.end()
                       ? nullptr
                       : it->second.get();
        };

        /*
         * Optional graph-captured checkpoints preserve only one terminal row
         * at each main-model layer boundary. Select the graph regime that
         * produced the currently published terminal hidden row; other cached
         * prefill and verifier variants may have valid but intentionally stale
         * checkpoint banks from earlier requests.
         */
        if (const auto *qwen_moe =
                dynamic_cast<const Qwen35MoEGraph *>(graph_builder_.get()))
        {
            const auto checkpoints =
                qwen_moe->mirroredLayerCheckpoints();
            if (DebugEnv::isTruthyEnv(
                    "LLAMINAR_MTP_MIRROR_LAYER_DIAGNOSTICS"))
            {
                size_t selected_matches = 0;
                std::map<std::string, size_t> bank_counts;
                for (const auto &checkpoint : checkpoints)
                {
                    if (!last_mirrored_layer_checkpoint_prefix_.empty() &&
                        checkpoint.name.starts_with(
                            last_mirrored_layer_checkpoint_prefix_))
                    {
                        ++selected_matches;
                    }
                    const size_t layer_marker =
                        checkpoint.name.find("_layer");
                    const std::string bank =
                        layer_marker == std::string::npos
                            ? checkpoint.name
                            : checkpoint.name.substr(0, layer_marker + 1);
                    ++bank_counts[bank];
                }
                std::ostringstream bank_summary;
                bool first_bank = true;
                for (const auto &[bank, count] : bank_counts)
                {
                    if (!first_bank)
                        bank_summary << ',';
                    first_bank = false;
                    bank_summary << bank << ':' << count;
                }
                LOG_ERROR(
                    "[MirroredLayerDiagnostics] device="
                    << state_.device_id.toString()
                    << " builder=" << qwen_moe->architectureName()
                    << " selected_bank="
                    << (last_mirrored_layer_checkpoint_prefix_.empty()
                            ? "(none)"
                            : last_mirrored_layer_checkpoint_prefix_)
                    << " retained_checkpoints=" << checkpoints.size()
                    << " selected_matches=" << selected_matches
                    << " banks={" << bank_summary.str() << '}'
                    << " first="
                    << (checkpoints.empty()
                            ? "(none)"
                            : checkpoints.front().name)
                    << " last="
                    << (checkpoints.empty()
                            ? "(none)"
                            : checkpoints.back().name));
            }
            for (const auto &checkpoint : checkpoints)
            {
                if (last_mirrored_layer_checkpoint_prefix_.empty() ||
                    !checkpoint.name.starts_with(
                        last_mirrored_layer_checkpoint_prefix_))
                    continue;
                append_tensor_row(
                    ("main_checkpoint_" + checkpoint.name).c_str(),
                    checkpoint.tensor);
            }
        }
        else if (DebugEnv::isTruthyEnv(
                     "LLAMINAR_MTP_MIRROR_LAYER_DIAGNOSTICS"))
        {
            LOG_ERROR(
                "[MirroredLayerDiagnostics] device="
                << state_.device_id.toString()
                << " builder="
                << (graph_builder_
                        ? graph_builder_->architectureName()
                        : "(null)")
                << " selected_bank="
                << (last_mirrored_layer_checkpoint_prefix_.empty()
                        ? "(none)"
                        : last_mirrored_layer_checkpoint_prefix_)
                << " retained_checkpoints=unavailable "
                   "(builder is not Qwen35MoEGraph)");
        }

        append_tensor_row(
            "terminal_hidden",
            state_.prefix_terminal_hidden.get());

        /*
         * The sidecar's first operation normalizes the terminal hidden row.
         * Include the immutable sidecar weights in the failed-participant
         * snapshot so an expert-overlay write that escapes its allocation can
         * be distinguished from an activation-lifetime or launch-order bug.
         * Matrix tensors contribute one logical row, which is enough to expose
         * participant-specific placement or collateral corruption without
         * copying a complete projection matrix on the error path.
         */
        const FrozenModelWeightSet *mtp_weight_set =
            selectMTPDecodeWeightSet("failed mirrored MTP digest");
        if (mtp_weight_set)
        {
            const ModelWeightBindings bindings =
                makeModelWeightBindings(
                    *mtp_weight_set, state_.device_id);
            if (!bindings.mtp.empty() && !bindings.mtp.depths.empty())
            {
                const MTPDepthWeightBindings &depth =
                    bindings.mtp.depths.front();
                append_tensor_row(
                    "mtp_weight_norm_hidden",
                    legacyTensor(depth.pre_fc_norm_hidden));
                append_tensor_row(
                    "mtp_weight_norm_embedding",
                    legacyTensor(depth.pre_fc_norm_embedding));
                append_tensor_row(
                    "mtp_weight_fc",
                    legacyTensor(depth.fc));
                append_tensor_row(
                    "mtp_weight_final_norm",
                    legacyTensor(depth.final_norm));
            }
        }

        append_tensor_row("mtp_embedding", extension(BufferId::MTP_EMBEDDING));
        append_tensor_row("mtp_norm_hidden", extension(BufferId::MTP_NORM_HIDDEN));
        append_tensor_row(
            "mtp_norm_embedding",
            extension(BufferId::MTP_NORM_EMBEDDING));
        append_tensor_row("mtp_concat", extension(BufferId::MTP_CONCAT));
        append_tensor_row("mtp_projected", extension(BufferId::MTP_PROJECTED));
        append_tensor_row("mtp_q", extension(BufferId::MTP_Q_PROJ));
        append_tensor_row("mtp_k", extension(BufferId::MTP_K_PROJ));
        append_tensor_row("mtp_v", extension(BufferId::MTP_V_PROJ));
        append_tensor_row(
            "mtp_attention_output",
            extension(BufferId::MTP_ATTN_OUTPUT));
        append_tensor_row(
            "mtp_attention_projection",
            extension(BufferId::MTP_ATTN_PROJ));
        append_tensor_row(
            "mtp_moe_expert_indices",
            extension(BufferId::MOE_EXPERT_INDICES));
        append_tensor_row(
            "mtp_moe_expert_weights",
            extension(BufferId::MOE_EXPERT_WEIGHTS));
        append_tensor_row(
            "mtp_moe_routed_output",
            extension(BufferId::MOE_COMBINED_OUTPUT));
        append_tensor_row(
            "mtp_moe_canonical_route_contributions",
            extension(BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS));
        append_tensor_row(
            "mtp_moe_shared_output",
            extension(BufferId::MOE_SHARED_EXPERT_OUTPUT));
        append_tensor_row("mtp_hidden", extension(BufferId::MTP_HIDDEN));
        append_tensor_row("mtp_logits", extension(BufferId::MTP_LOGITS));

        /*
         * Stateful attention is the only main-graph operation whose result can
         * differ across mirrored participants despite equal incoming hidden
         * rows and immutable weights.  Capture a bounded logical K/V tail plus
         * both GDN bank geometries after a failed proposal.  The policy is
         * invocation-scoped and deliberately omits complete long-context KV
         * payloads, keeping this already-exceptional diagnostic proportional
         * to layer count rather than sequence length.
         */
        if (state_.kv_cache)
        {
            PrefixProbeCapturePolicy capture_policy;
            capture_policy.trailing_kv_tokens = 4;
            capture_policy.hash_gdn_device_state = true;

            const PrefixKVCacheProbe kv_probe =
                inspectKVCacheForPrefixProbe(
                    *state_.kv_cache,
                    "failed_mirrored_mtp",
                    state_.device_id,
                    /*sequence_count=*/1,
                    stream,
                    capture_policy);
            for (const PrefixKVLayerProbe &layer : kv_probe.layers)
            {
                const auto segment_it = std::find_if(
                    layer.segments.begin(),
                    layer.segments.end(),
                    [](const PrefixKVSegmentProbe &segment)
                    {
                        return segment.name == "diagnostic_tail";
                    });
                const bool available =
                    segment_it != layer.segments.end() &&
                    segment_it->hash_available;
                const std::string suffix =
                    "_layer" + std::to_string(layer.global_layer) +
                    "_seq" + std::to_string(layer.seq_idx);
                append_prehashed_bytes(
                    "main_live_kv_k" + suffix,
                    available ? segment_it->k_payload_hash : 0,
                    available ? segment_it->k_payload_bytes : 0,
                    available);
                append_prehashed_bytes(
                    "main_live_kv_v" + suffix,
                    available ? segment_it->v_payload_hash : 0,
                    available ? segment_it->v_payload_bytes : 0,
                    available);
            }

            const std::vector<PrefixGDNLayerProbe> gdn_probes =
                inspectHybridGDNForPrefixProbe(
                    *state_.kv_cache,
                    stream,
                    capture_policy);
            for (const PrefixGDNLayerProbe &layer : gdn_probes)
            {
                const std::string suffix =
                    "_layer" + std::to_string(layer.global_layer);
                append_prehashed_bytes(
                    "main_live_gdn_full_conv" + suffix,
                    layer.conv_device_hash,
                    layer.conv_device_bytes,
                    layer.device_state_hash_available);
                append_prehashed_bytes(
                    "main_live_gdn_full_recurrence" + suffix,
                    layer.recurrence_device_hash,
                    layer.recurrence_device_bytes,
                    layer.device_state_hash_available);
                append_prehashed_bytes(
                    "main_live_gdn_local_conv" + suffix,
                    layer.conv_local_device_hash,
                    layer.conv_local_device_bytes,
                    layer.local_device_state_hash_available);
                append_prehashed_bytes(
                    "main_live_gdn_local_recurrence" + suffix,
                    layer.recurrence_local_device_hash,
                    layer.recurrence_local_device_bytes,
                    layer.local_device_state_hash_available);
            }
        }

        /*
         * The replicated predictor reads its own shifted KV cache, not the
         * main-model cache inspected above.  Observe the cache-owned device
         * token count through the retained MTP transaction fence, then export
         * the complete short diagnostic history from every shifted-cache
         * layer (bounded at one verifier transaction).  A four-row tail can
         * falsely certify two caches whose prompt-seeded prefixes differ, so
         * the failure oracle must cover every row the attention launch can
         * currently read.  Do
         * not consult IKVCache's host progress mirror here: device-owned MTP
         * execution is intentionally allowed to leave that diagnostic mirror
         * stale between explicit observation boundaries.
         */
        if (!state_.mtp_kv_caches.empty() && state_.mtp_kv_caches[0])
        {
            const IKVCache &mtp_cache = *state_.mtp_kv_caches[0];
            const std::optional<int> observed_tokens =
                deviceResidentShiftedMTPKVTokenCountForObservation(
                    /*depth=*/0,
                    /*request_index=*/0,
                    stream,
                    "failed_mirrored_mtp_shifted_cache");
            constexpr int kDiagnosticTransactionTokens = 32;
            for (int cache_layer = 0;
                 cache_layer < mtp_cache.n_layers();
                 ++cache_layer)
            {
                const int global_layer =
                    mtp_cache.first_layer_index() + cache_layer;
                const std::string suffix =
                    "_layer" + std::to_string(global_layer) + "_seq0";
                bool available = false;
                size_t k_byte_count = 0;
                size_t v_byte_count = 0;
                uint64_t k_hash = 0;
                uint64_t v_hash = 0;
                if (observed_tokens && *observed_tokens > 0)
                {
                    const int token_count = std::min(
                        kDiagnosticTransactionTokens,
                        *observed_tokens);
                    const auto layout = mtp_cache.logicalBlockLayout(
                        global_layer,
                        token_count);
                    if (layout.k_bytes > 0 && layout.v_bytes > 0)
                    {
                        std::vector<uint8_t> k_bytes(layout.k_bytes);
                        std::vector<uint8_t> v_bytes(layout.v_bytes);
                        IKVCache::KVCacheLogicalBlockDescriptor descriptor;
                        descriptor.layer = global_layer;
                        descriptor.seq_idx = 0;
                        descriptor.logical_token_start =
                            *observed_tokens - token_count;
                        descriptor.token_count = token_count;
                        descriptor.stream = stream;
                        available = mtp_cache.exportLogicalBlock(
                            descriptor,
                            k_bytes.data(),
                            v_bytes.data());
                        if (available)
                        {
                            k_byte_count = k_bytes.size();
                            v_byte_count = v_bytes.size();
                            k_hash = hash_bytes(
                                k_bytes.data(),
                                k_bytes.size());
                            v_hash = hash_bytes(
                                v_bytes.data(),
                                v_bytes.size());
                        }
                    }
                }
                append_prehashed_bytes(
                    "mtp_live_kv_k" + suffix,
                    k_hash,
                    k_byte_count,
                    available);
                append_prehashed_bytes(
                    "mtp_live_kv_v" + suffix,
                    v_hash,
                    v_byte_count,
                    available);
            }
        }

        append_device_int32_values(
            "main_live_position",
            state_.kv_cache
                ? state_.kv_cache->deviceSequenceCachedTokenCountPtr(0)
                : nullptr,
            1);
        append_device_int32_values(
            "mtp_live_position",
            !state_.mtp_kv_caches.empty() && state_.mtp_kv_caches[0]
                ? state_.mtp_kv_caches[0]->deviceSequenceCachedTokenCountPtr(0)
                : nullptr,
            1);
        append_device_bytes(
            "resident_draft_slots",
            stochastic_draft_sample_tokens_dev_,
            static_cast<size_t>(
                std::max(0, stochastic_draft_row_capacity_)) *
                sizeof(int32_t));
        append_device_bytes(
            "resident_target_slots",
            stochastic_target_sample_tokens_dev_,
            static_cast<size_t>(
                std::max(0, stochastic_target_row_capacity_)) *
                sizeof(int32_t));
        /*
         * Completion clears active loop geometry before RankOrchestrator asks
         * for a fatal digest. Allocation capacities remain stable for the
         * request lifetime, so use them for the preview and let the copied
         * request-length values identify the logical prefix. MTP depth is at
         * most fifteen; the bounded token preview therefore covers a complete
         * single-request verifier transaction while remaining compact for
         * request-batched configurations.
         */
        const size_t verifier_value_capacity =
            static_cast<size_t>(std::max(0, stochastic_target_row_capacity_));
        const size_t verifier_values_to_preview =
            std::min<size_t>(verifier_value_capacity, 64u);
        const size_t verifier_requests_to_preview =
            std::min<size_t>(
                static_cast<size_t>(
                    std::max(0, stochastic_batch_output_request_capacity_)),
                4u);
        append_device_int32_values(
            "grouped_verifier_input_tokens",
            static_cast<const int32_t *>(mtp_verifier_input_tokens_dev_),
            verifier_values_to_preview);
        append_device_int32_values(
            "grouped_verifier_position_ids",
            static_cast<const int32_t *>(mtp_verifier_position_ids_dev_),
            verifier_values_to_preview);
        append_device_int32_values(
            "grouped_verifier_request_lengths",
            static_cast<const int32_t *>(mtp_verifier_request_lengths_dev_),
            verifier_requests_to_preview);

        const bool first_transaction_diagnostic_requested = mtp_diagnostic_record_mode_ == sampling_math::MTPDiagnosticRecordMode::FirstTransaction;
        if (first_transaction_diagnostic_requested)
        {
            using DiagnosticRecord =
                sampling_math::MTPFirstTransactionDiagnosticRecord;
            DiagnosticRecord record{};
            const bool record_available =
                mtp_first_transaction_diagnostic_dev_ &&
                backend->deviceToHostFast(
                    &record,
                    mtp_first_transaction_diagnostic_dev_,
                    sizeof(record),
                    state_.device_id.gpu_ordinal(),
                    stream) &&
                record.valid == 1 &&
                record.version ==
                    sampling_math::kMTPFirstTransactionDiagnosticVersion &&
                record.transaction_count == 0 &&
                record.comparison_row_count > 0 &&
                record.comparison_row_count <=
                    sampling_math::kSpeculativeBatchMaxRows &&
                record.top_k > 0 &&
                record.top_k <= sampling_math::kMaxTopK;
            if (!record_available)
            {
                append_host_bytes(
                    "first_transaction_diagnostic",
                    nullptr,
                    sizeof(record));
                return digests;
            }

            const size_t sample_rows =
                static_cast<size_t>(record.comparison_row_count + 1);
            const size_t comparison_rows =
                static_cast<size_t>(record.comparison_row_count);
            const std::array<uint64_t, 9> header_words = {
                static_cast<uint64_t>(record.version),
                record.threshold_seed,
                static_cast<uint64_t>(
                    static_cast<uint32_t>(record.threshold_base_position)),
                static_cast<uint64_t>(
                    static_cast<uint32_t>(record.threshold_position_offset)),
                static_cast<uint64_t>(
                    static_cast<uint32_t>(record.comparison_row_count)),
                static_cast<uint64_t>(
                    static_cast<uint32_t>(record.top_k)),
                static_cast<uint64_t>(
                    static_cast<uint32_t>(record.transaction_count)),
                static_cast<uint64_t>(
                    static_cast<uint32_t>(
                        record.transaction_commit_budget)),
                static_cast<uint64_t>(
                    static_cast<uint32_t>(
                        record.leading_committed_output_count)),
            };
            std::ostringstream header_preview;
            header_preview
                << "{version=" << record.version
                << ",seed=0x" << std::hex << record.threshold_seed
                << std::dec
                << ",base_position=" << record.threshold_base_position
                << ",position_offset=" << record.threshold_position_offset
                << ",rows=" << record.comparison_row_count
                << ",top_k=" << record.top_k
                << ",transaction=" << record.transaction_count
                << ",budget=" << record.transaction_commit_budget
                << ",leading="
                << record.leading_committed_output_count << '}';
            append_host_bytes(
                "first_transaction_header",
                header_words.data(),
                sizeof(header_words),
                header_preview.str());

            auto int_preview = [](const int32_t *values, size_t count)
            {
                std::ostringstream preview;
                preview << '{';
                for (size_t index = 0; index < count; ++index)
                {
                    if (index != 0)
                        preview << ',';
                    preview << values[index];
                }
                preview << '}';
                return preview.str();
            };
            append_host_bytes(
                "first_transaction_verifier_inputs",
                record.verifier_input_tokens,
                sample_rows * sizeof(record.verifier_input_tokens[0]),
                int_preview(record.verifier_input_tokens, sample_rows));
            append_host_bytes(
                "first_transaction_sampled_targets",
                record.sampled_target_tokens,
                sample_rows * sizeof(record.sampled_target_tokens[0]),
                int_preview(record.sampled_target_tokens, sample_rows));
            append_host_bytes(
                "first_transaction_sample_matches",
                record.sampled_matches_verifier_input,
                comparison_rows *
                    sizeof(record.sampled_matches_verifier_input[0]),
                int_preview(
                    record.sampled_matches_verifier_input,
                    comparison_rows));

            std::ostringstream threshold_preview;
            threshold_preview << '{';
            for (size_t row = 0; row < sample_rows; ++row)
            {
                if (row != 0)
                    threshold_preview << ',';
                threshold_preview << "0x" << std::hex
                                  << record.threshold_bits[row]
                                  << std::dec;
            }
            threshold_preview << '}';
            append_host_bytes(
                "first_transaction_threshold_bits",
                record.threshold_bits,
                sample_rows * sizeof(record.threshold_bits[0]),
                threshold_preview.str());

            std::ostringstream distribution_preview;
            distribution_preview << '{';
            for (size_t row = 0; row < sample_rows; ++row)
            {
                if (row != 0)
                    distribution_preview << ',';
                distribution_preview << "0x" << std::hex
                                     << record.target_distribution_hashes[row]
                                     << std::dec;
            }
            distribution_preview << '}';
            append_host_bytes(
                "first_transaction_target_distribution_hashes",
                record.target_distribution_hashes,
                sample_rows *
                    sizeof(record.target_distribution_hashes[0]),
                distribution_preview.str());
            append_host_bytes(
                "first_transaction_output_tokens",
                record.output_tokens,
                sample_rows * sizeof(record.output_tokens[0]),
                int_preview(record.output_tokens, sample_rows));
            append_host_bytes(
                "first_transaction_output_meta",
                record.output_meta,
                sizeof(record.output_meta),
                int_preview(
                    record.output_meta,
                    sampling_math::kSpeculativeBatchMetaCount));

            const int retained_draft_depth =
                std::clamp(
                    record.draft_diagnostic_depth,
                    0,
                    sampling_math::kSpeculativeBatchMaxRows);
            if (retained_draft_depth > 0)
            {
                append_host_bytes(
                    "first_transaction_draft_conditions",
                    record.draft_condition_tokens,
                    static_cast<size_t>(retained_draft_depth) *
                        sizeof(record.draft_condition_tokens[0]),
                    int_preview(
                        record.draft_condition_tokens,
                        static_cast<size_t>(retained_draft_depth)));
                append_host_bytes(
                    "first_transaction_draft_positions",
                    record.draft_position_ids,
                    static_cast<size_t>(retained_draft_depth) *
                        sizeof(record.draft_position_ids[0]),
                    int_preview(
                        record.draft_position_ids,
                        static_cast<size_t>(retained_draft_depth)));

                constexpr std::array<
                    const char *,
                    sampling_math::kMTPFirstTransactionDraftBoundaryCount>
                    kDraftBoundaryNames = {
                        "terminal_hidden_input",
                        "embedding",
                        "normalized_terminal_hidden",
                        "normalized_embedding",
                        "concatenated_input",
                        "projected_input",
                        "attention_query",
                        "attention_key",
                        "attention_value",
                        "attention_output",
                        "attention_projection",
                        "moe_expert_indices",
                        "moe_expert_weights",
                        "moe_routed_output",
                        "moe_shared_output",
                        "ffn_output",
                        "final_hidden",
                        "scored_logits",
                    };
                for (int slot = 0; slot < retained_draft_depth; ++slot)
                {
                    for (size_t boundary = 0;
                         boundary < kDraftBoundaryNames.size();
                         ++boundary)
                    {
                        const size_t word_count =
                            record.draft_boundary_word_counts[slot]
                                                              [boundary];
                        append_prehashed_bytes(
                            "first_transaction_draft" +
                                std::to_string(slot) + "_" +
                                kDraftBoundaryNames[boundary],
                            record.draft_boundary_hashes[slot][boundary],
                            word_count * sizeof(uint32_t),
                            word_count > 0);
                    }
                }
            }
        }
        return digests;
    }

    bool DeviceGraphOrchestrator::sampleGreedyFromMTPLogitsToDeviceDraftSlot(
        int draft_sample_slot,
        int32_t *out_token)
    {
        if (out_token)
            *out_token = -1;
        if (globalTPContextForMTPCoordination())
        {
            /*
             * Multi-rank vocab-sharded MTP logits need a coordinated winner
             * before a single participant can write the verifier draft slot.
             * Keep those lanes on the existing host-visible coordinated path
             * until the TP device-slot contract is implemented domain-wide.
             */
            return false;
        }

        SamplingParams greedy_params;
        greedy_params.temperature = 0.0f;
        const int vocab = vocab_size();
        return sampleStochasticDraftProposalOnDeviceImpl(
            DeviceLogitsSource::MTP,
            /*row=*/0,
            draft_sample_slot,
            greedy_params,
            vocab,
            /*threshold=*/0.0f,
            out_token,
            /*verifier_consumer_pending=*/true,
            mtp_request_penalty_policy_);
    }

    bool DeviceGraphOrchestrator::sampleGreedyFromMainLogitsToDeviceTargetSlot(
        int target_sample_slot,
        int32_t *out_token)
    {
        if (!supportsDeviceStochasticMTPVerification() ||
            globalTPContextForMTPCoordination() ||
            target_sample_slot < 0 ||
            target_sample_slot >= stochastic_target_row_capacity_ ||
            !stochastic_target_sample_tokens_dev_ ||
            !stochastic_verify_accept_probs_dev_ ||
            !argmax_partial_vals_dev_ ||
            !argmax_partial_idxs_dev_ ||
            argmax_partial_capacity_ <= 0)
        {
            return false;
        }

        TensorBase *main_logits = requireCurrentMainLogits(
            DeviceLogitsSource::Main,
            "sampleGreedyFromMainLogitsToDeviceTargetSlot");
        if (!main_logits->deviceValid() ||
            !main_logits->gpu_data_ptr())
        {
            return false;
        }

        /*
         * The first target-token slot is the device-resident owner for this
         * token until the compact verifier summary returns.  Clear any stale
         * readiness marker before enqueueing a new argmax into the slot.
         */
        stochastic_target_top_k_[static_cast<size_t>(target_sample_slot)] = 0;
        stochastic_target_row_formats_[static_cast<size_t>(target_sample_slot)] =
            StochasticRowFormat::Empty;
        stochastic_target_distribution_streams_[static_cast<size_t>(target_sample_slot)] = nullptr;
        clearStochasticTargetSampleReadySlot(
            target_sample_slot,
            StochasticSampleReadyClearMode::Force);

        const auto &shape = main_logits->shape();
        if (shape.empty())
            return false;
        const size_t rows = shape.size() >= 2 ? shape[0] : 1;
        const size_t cols = shape.size() >= 2 ? shape[1] : shape[0];
        if (cols == 0 ||
            rows == 0 ||
            cols > static_cast<size_t>(std::numeric_limits<int>::max()))
        {
            return false;
        }

        void *stream = prepareMainLogitsDeviceConsumer(
            "sampleGreedyFromMainLogitsToDeviceTargetSlot",
            MainLogitsHandoffMode::Consume);
        if (!stream)
            return false;

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
            return false;

        const float *row_ptr =
            static_cast<const float *>(main_logits->gpu_data_ptr());
        auto *out_token_dev =
            static_cast<int *>(stochastic_target_sample_tokens_dev_) +
            target_sample_slot;
        auto *out_value_dev =
            static_cast<float *>(stochastic_verify_accept_probs_dev_) +
            target_sample_slot;
        {
            PerfStatsCollector::ScopedTimer timer(
                "mtp",
                "first_token_greedy_device_target_slot_enqueue",
                "decode",
                state_.device_id.toString(),
                {{"slot", std::to_string(target_sample_slot)},
                 {"vocab", std::to_string(cols)}});
            if (!backend->enqueueArgmaxF32BatchedRowsDevice(
                    row_ptr,
                    /*rows=*/1,
                    static_cast<int>(cols),
                    state_.device_id.gpu_ordinal(),
                    stream,
                    out_value_dev,
                    out_token_dev,
                    argmax_partial_vals_dev_,
                    argmax_partial_idxs_dev_,
                    argmax_partial_capacity_))
            {
                return false;
            }
        }

        if (!recordStochasticTargetSampleReady(
                target_sample_slot,
                stream,
                /*verifier_consumer_pending=*/out_token == nullptr, StochasticTargetPublication::NewValue))
            return false;

        if (!out_token)
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "first_token_greedy_deferred_host_reads",
                1.0,
                "decode",
                state_.device_id.toString(),
                {{"slot", std::to_string(target_sample_slot)}});
            return true;
        }

        int32_t token = -1;
        {
            PerfStatsCollector::ScopedTimer timer(
                "mtp",
                "first_token_greedy_device_target_slot_d2h_sync",
                "decode",
                state_.device_id.toString(),
                {{"slot", std::to_string(target_sample_slot)}});
            if (!backend->deviceToHostFast(&token,
                                           out_token_dev,
                                           sizeof(int),
                                           state_.device_id.gpu_ordinal(),
                                           stream))
            {
                return false;
            }
        }
        *out_token = token;
        return token >= 0;
    }

    int DeviceGraphOrchestrator::sampleGreedyFromAllPositionLogitsOnDevice(int row)
    {
        if (row < 0)
        {
            return -1;
        }

        if (state_.all_position_logits)
        {
            void *stream = nullptr;
            auto device_opt = state_.all_position_logits->current_device();
            if (device_opt.has_value() && device_opt->is_gpu())
            {
                stream = peekPendingLogitsStream(
                    PendingLogitsStreamRole::AllPositionVerifier);
                if (!stream)
                    stream = explicitGPUStreamForOperation("sampleGreedyFromAllPositionLogitsOnDevice");
                if (!stream)
                {
                    return -1;
                }
            }
            return coordinateGreedyCandidate(
                sampleGreedyCandidateFromTensor(state_.all_position_logits.get(), row, 0,
                                                argmax_partial_vals_dev_,
                                                argmax_partial_idxs_dev_,
                                                argmax_partial_capacity_,
                                                stream,
                                                "all_position_logits"),
                globalTPContextForMTPCoordination());
        }

        if (state_.all_position_logits_local)
        {
            const int token_offset = graph_builder_
                                         ? vocabOffsetForTPConfig(graph_builder_->config())
                                         : 0;
            void *stream = nullptr;
            auto device_opt = state_.all_position_logits_local->current_device();
            if (device_opt.has_value() && device_opt->is_gpu())
            {
                stream = peekPendingLogitsStream(
                    PendingLogitsStreamRole::AllPositionVerifier);
                if (!stream)
                    stream = explicitGPUStreamForOperation("sampleGreedyFromAllPositionLogitsLocalOnDevice");
                if (!stream)
                {
                    return -1;
                }
            }
            return coordinateGreedyCandidate(
                sampleGreedyCandidateFromTensor(state_.all_position_logits_local.get(), row, token_offset,
                                                argmax_partial_vals_dev_,
                                                argmax_partial_idxs_dev_,
                                                argmax_partial_capacity_,
                                                stream,
                                                "all_position_logits_local"),
                globalTPContextForMTPCoordination());
        }

        return -1;
    }

    bool DeviceGraphOrchestrator::sampleGreedyFromAllPositionLogitsOnDeviceRows(
        int start_row,
        int row_count,
        int32_t *out_tokens)
    {
        if (start_row < 0 || row_count <= 0 || !out_tokens)
            return false;

        const IGlobalTPContext *global_ctx = globalTPContextForMTPCoordination();
        if (global_ctx && global_ctx->degree() > 1)
        {
            return IInferenceRunner::sampleGreedyFromAllPositionLogitsOnDeviceRows(
                start_row, row_count, out_tokens);
        }

        TensorBase *tensor = nullptr;
        int token_offset = 0;
        const char *sample_source = "all_position_logits_rows";
        if (state_.all_position_logits)
        {
            tensor = state_.all_position_logits.get();
        }
        else if (state_.all_position_logits_local)
        {
            tensor = state_.all_position_logits_local.get();
            sample_source = "all_position_logits_local_rows";
            token_offset = graph_builder_
                               ? vocabOffsetForTPConfig(graph_builder_->config())
                               : 0;
        }
        if (!tensor)
            return false;

        constexpr int kMaxStackVerifierRows = 16;
        if (row_count > kMaxStackVerifierRows)
        {
            if (peekPendingLogitsStream(PendingLogitsStreamRole::AllPositionVerifier))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Deferred all-position verifier sampling exceeds stack row capacity");
                clearPendingLogitsStream(
                    PendingLogitsStreamRole::AllPositionVerifier,
                    "sampleGreedyFromAllPositionLogitsOnDeviceRows_over_capacity");
                return false;
            }
            return IInferenceRunner::sampleGreedyFromAllPositionLogitsOnDeviceRows(
                start_row, row_count, out_tokens);
        }

        const auto &shape = tensor->shape();
        if (shape.empty())
            return false;
        const size_t rows = shape.size() >= 2 ? shape[0] : 1;
        const size_t cols = shape.size() >= 2 ? shape[1] : shape[0];
        if (cols == 0 ||
            static_cast<size_t>(start_row) >= rows ||
            static_cast<size_t>(start_row + row_count) > rows ||
            cols > static_cast<size_t>(std::numeric_limits<int>::max()))
        {
            return false;
        }

        auto device_opt = tensor->current_device();
        if (device_opt.has_value() && device_opt->is_gpu() && tensor->deviceValid())
        {
            IBackend *backend = getBackendFor(*device_opt);
            const void *gpu_ptr = tensor->gpu_data_ptr();
            if (!backend || !gpu_ptr)
                return false;

            void *stream = peekPendingLogitsStream(
                PendingLogitsStreamRole::AllPositionVerifier);
            const bool using_deferred_stream = stream != nullptr;
            if (!stream)
                stream = explicitGPUStreamForOperation("sampleGreedyFromAllPositionLogitsOnDeviceRows");
            if (!stream)
            {
                return false;
            }

            const auto *base = static_cast<const float *>(gpu_ptr);
            const void *first_row =
                base + static_cast<size_t>(start_row) * static_cast<size_t>(cols);
            std::array<float, kMaxStackVerifierRows> values{};
            std::array<int, kMaxStackVerifierRows> indices{};
            const bool ok = backend->argmaxF32BatchedRows(
                first_row,
                row_count,
                static_cast<int>(cols),
                device_opt->gpu_ordinal(),
                values.data(),
                indices.data(),
                stream,
                argmax_partial_vals_dev_,
                argmax_partial_idxs_dev_,
                argmax_partial_capacity_);
            if (!ok)
            {
                if (using_deferred_stream)
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Batched verifier argmax failed after deferred verifier replay");
                    return false;
                }
                if (backend->backendDeviceType() == DeviceType::ROCm)
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] ROCm batched verifier argmax failed");
                    return false;
                }
                return IInferenceRunner::sampleGreedyFromAllPositionLogitsOnDeviceRows(
                    start_row, row_count, out_tokens);
            }

            for (int i = 0; i < row_count; ++i)
            {
                if (indices[static_cast<size_t>(i)] < 0)
                    return false;
                out_tokens[i] = static_cast<int32_t>(
                    token_offset + indices[static_cast<size_t>(i)]);
            }
            if (greedyMarginStatsEnabled())
            {
                for (int i = 0; i < row_count; ++i)
                {
                    const void *row_ptr =
                        base + static_cast<size_t>(start_row + i) * static_cast<size_t>(cols);
                    float top_values[2] = {};
                    int top_indices[2] = {};
                    if (backend->topKF32(row_ptr,
                                         static_cast<int>(cols),
                                         2,
                                         device_opt->gpu_ordinal(),
                                         top_values,
                                         top_indices,
                                         stream))
                    {
                        recordGreedyMarginStats(
                            sample_source,
                            *device_opt,
                            start_row + i,
                            top_values[0],
                            top_values[1]);
                    }
                    else
                    {
                        recordGreedyMarginUnavailable(sample_source, *device_opt, start_row + i);
                    }
                }
            }
            PerfStatsCollector::addCounter(
                "mtp",
                "verifier_token_device_batch_samples",
                static_cast<double>(row_count),
                "decode",
                state_.device_id.toString(),
                {{"rows", std::to_string(row_count)}});
            return true;
        }

        return IInferenceRunner::sampleGreedyFromAllPositionLogitsOnDeviceRows(
            start_row, row_count, out_tokens);
    }

    bool DeviceGraphOrchestrator::configureMTPRequestStopTokens(
        const std::vector<int32_t> &stop_tokens)
    {
        using namespace sampling_math;

        if (stop_tokens.size() > kSpeculativeBatchMaxStopTokens)
        {
            throw std::invalid_argument(
                "MTP request stop-token policy exceeds the fixed device "
                "control capacity");
        }
        if (mtp_verifier_outcome_graph_mode_ !=
                MTPVerifierOutcomeGraphMode::Disabled ||
            greedy_verifier_outcome_graph_transaction_.state !=
                GreedyVerifierOutcomeGraphState::Idle)
        {
            throw std::logic_error(
                "MTP request stop-token policy cannot change during an active "
                "verifier transaction");
        }

        std::array<int32_t, kSpeculativeBatchMaxStopTokens> staged = {
            -1, -1, -1, -1, -1, -1, -1, -1};
        std::copy(stop_tokens.begin(), stop_tokens.end(), staged.begin());
        const int staged_count = static_cast<int>(stop_tokens.size());
        if (staged == mtp_request_stop_tokens_ &&
            staged_count == mtp_request_stop_token_count_)
        {
            return true;
        }

        mtp_request_stop_tokens_ = staged;
        mtp_request_stop_token_count_ = staged_count;
        mtp_request_stop_tokens_published_session_epoch_.reset();
        return true;
    }

    bool DeviceGraphOrchestrator::initializePromptRepetitionHistory(
        std::span<const int32_t> unique_tokens, int request_index)
    {
        if (!state_.device_id.is_gpu() || (pp_stage_config_ && !pp_stage_config_->has_lm_head))
            return true;
        if (!arena_ || !mtp_generated_token_counts_dev_ || unique_tokens.empty() ||
            unique_tokens.size() > static_cast<size_t>(state_.vocab_size) || request_index < 0 ||
            static_cast<size_t>(request_index) >= arena_->getRows(BufferId::MTP_GENERATED_TOKEN_COUNTS))
            return false;
        // Prefill and prefix restore have both completed their producer edges.
        // Publish prompt membership before any sampler can consume their logits.
        void *stream = prepareMainLogitsDeviceConsumer("initializePromptRepetitionHistory",
            MainLogitsHandoffMode::Observe, /*require_main_forward=*/false);
        auto *backend = getBackendFor(state_.device_id);
        if (!stream || !backend ||
            !arena_->prepareForRead(BufferId::MTP_GENERATED_TOKEN_COUNTS, state_.device_id, stream) ||
            !backend->initializePromptRepetitionHistory(
                static_cast<int32_t *>(mtp_generated_token_counts_dev_) +
                    static_cast<size_t>(request_index) * mtp_generated_token_count_capacity_,
                unique_tokens.data(), static_cast<int>(unique_tokens.size()), state_.vocab_size,
                state_.device_id.gpu_ordinal(), stream) ||
            !publishPreparedArenaGraphInput(BufferId::MTP_GENERATED_TOKEN_COUNTS,
                mtp_generated_token_counts_dev_, stream, state_.device_id, "prompt_repetition_history"))
            return false;
        PerfStatsCollector::addCounter("sampling", "prompt_repetition_history_tokens",
            static_cast<double>(unique_tokens.size()), "request_admission", state_.device_id.toString(),
            {{"request", std::to_string(request_index)}, {"ordering", "event_wait"}});
        return true;
    }

    bool DeviceGraphOrchestrator::configureMTPRequestPenaltyPolicy(
        const MTPRequestPenaltyPolicy &policy)
    {
        if (!std::isfinite(policy.presence_penalty) ||
            !std::isfinite(policy.frequency_penalty) ||
            !std::isfinite(policy.repetition_penalty) || policy.repetition_penalty <= 0.0f)
        {
            throw std::invalid_argument(
                "MTP request penalties must be finite and repetition_penalty must be positive");
        }
        if (mtp_verifier_outcome_graph_mode_ !=
                MTPVerifierOutcomeGraphMode::Disabled ||
            greedy_verifier_outcome_graph_transaction_.state !=
                GreedyVerifierOutcomeGraphState::Idle)
        {
            throw std::logic_error(
                "MTP request penalty policy cannot change during an active "
                "verifier transaction");
        }

        /*
         * Configuration may precede the reset that retires the previous request.
         * Keep the currently published policy and its epoch intact until that
         * boundary occurs; the next admission publishes this staged value after
         * session_epoch_ advances.  Current-request graph consumers therefore
         * cannot observe a next-request policy early.
         */
        mtp_request_penalty_policy_ = policy;
        return true;
    }

    bool DeviceGraphOrchestrator::prepareGreedyAllPositionBatchOutcomeGraph(
        int verifier_token_count,
        const int32_t *stop_tokens,
        int stop_token_count,
        const MTPRequestPenaltyPolicy &penalty_policy)
    {
        using namespace sampling_math;

        if (!state_.device_id.is_gpu() ||
            verifier_token_count <= 0 ||
            verifier_token_count > mtp_max_verifier_rows_ ||
            stop_token_count < 0 ||
            stop_token_count > kSpeculativeBatchMaxStopTokens ||
            (stop_token_count > 0 && !stop_tokens) ||
            !mtp_verifier_stop_tokens_dev_ ||
            !mtp_greedy_penalty_policy_dev_ ||
            !mtp_generated_token_counts_dev_ ||
            mtp_generated_token_count_capacity_ != state_.vocab_size ||
            !mtp_verifier_input_tokens_dev_ ||
            !stochastic_verify_tokens_dev_ ||
            !stochastic_verify_accept_probs_dev_ ||
            !stochastic_batch_output_tokens_dev_ ||
            !stochastic_batch_output_meta_dev_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Invalid graph-owned greedy outcome preparation"
                      << " device=" << state_.device_id.toString()
                      << " verifier_tokens=" << verifier_token_count
                      << " stop_tokens=" << stop_token_count);
            return false;
        }
        if (mtp_verifier_outcome_graph_mode_ !=
                MTPVerifierOutcomeGraphMode::Disabled ||
            greedy_verifier_outcome_graph_transaction_.state !=
                GreedyVerifierOutcomeGraphState::Idle)
        {
            throw std::logic_error(
                "Cannot arm a graph-owned greedy MTP outcome while another "
                "outcome transaction is active");
        }
        if (stop_token_count != mtp_request_stop_token_count_ ||
            !mtp_request_stop_tokens_published_session_epoch_ ||
            *mtp_request_stop_tokens_published_session_epoch_ !=
                session_epoch_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Greedy verifier stop controls were not published by the current request boundary"
                      << " requested_count=" << stop_token_count
                      << " configured_count="
                      << mtp_request_stop_token_count_
                      << " session=" << session_epoch_
                      << " published_session="
                      << (mtp_request_stop_tokens_published_session_epoch_
                              ? std::to_string(
                                    *mtp_request_stop_tokens_published_session_epoch_)
                              : "none"));
            return false;
        }
        for (int i = 0; i < stop_token_count; ++i)
        {
            if (mtp_request_stop_tokens_[static_cast<size_t>(i)] !=
                stop_tokens[i])
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Greedy verifier stop controls differ from request admission at index "
                          << i);
                return false;
            }
        }

        GreedyVerifierOutcomeGraphTransaction transaction;
        transaction.state = GreedyVerifierOutcomeGraphState::Armed;
        transaction.verifier_token_count = verifier_token_count;
        transaction.stop_token_count = stop_token_count;
        transaction.penalty_policy = penalty_policy;

        greedy_verifier_outcome_graph_transaction_ = transaction;
        mtp_verifier_outcome_graph_mode_ =
            MTPVerifierOutcomeGraphMode::Greedy;
        PerfStatsCollector::addCounter(
            "mtp",
            "graph_owned_greedy_outcome_transactions_armed",
            1.0,
            "decode",
            state_.device_id.toString(),
            {{"rows", std::to_string(verifier_token_count)},
             {"stop_tokens", std::to_string(stop_token_count)}});
        return true;
    }

    bool DeviceGraphOrchestrator::verifyGreedyAllPositionBatchOutcomeOnDeviceResident(
        const int32_t *draft_tokens,
        int draft_token_count,
        const int32_t *stop_tokens,
        int stop_token_count,
        DeviceSpeculativeOutcomeHandle *out_handle)
    {
        using namespace sampling_math;
        if (out_handle)
            *out_handle = DeviceSpeculativeOutcomeHandle{};

        if (!out_handle)
            return false;

        if (!draft_tokens ||
            draft_token_count <= 0 ||
            draft_token_count > mtp_max_verifier_rows_ ||
            stop_token_count < 0 ||
            stop_token_count > kSpeculativeBatchMaxStopTokens ||
            (stop_token_count > 0 && !stop_tokens))
        {
            return false;
        }

        const auto &transaction =
            greedy_verifier_outcome_graph_transaction_;
        if (mtp_verifier_outcome_graph_mode_ !=
                MTPVerifierOutcomeGraphMode::Greedy ||
            transaction.state !=
                GreedyVerifierOutcomeGraphState::Produced ||
            transaction.verifier_token_count != draft_token_count ||
            transaction.stop_token_count != stop_token_count)
        {
            throw std::logic_error(
                "Greedy resident outcome consumption does not match a "
                "produced graph-owned transaction");
        }
        for (int i = 0; i < stop_token_count; ++i)
        {
            if (mtp_request_stop_tokens_[static_cast<size_t>(i)] !=
                stop_tokens[i])
            {
                throw std::logic_error(
                    "Greedy resident outcome stop controls differ from the "
                    "controls captured for graph replay");
            }
        }

        void *stream = consumePendingLogitsStream(
            PendingLogitsStreamRole::AllPositionVerifier,
            "consumeGraphOwnedGreedyAllPositionBatchOutcome");
        if (!stream)
        {
            throw std::logic_error(
                "Graph-owned greedy outcome has no verifier stream handoff");
        }
        const auto *grouped_ready = selectForwardGraphOutputPublication(
            ForwardGraphOutputKind::GroupedVerifier);
        if (!grouped_ready || !grouped_ready->event ||
            !grouped_ready->all_position_logits)
        {
            throw std::logic_error(
                "Graph-owned greedy outcome has no durable grouped-forward "
                "completion publication");
        }

        const bool materialized_first_token_from_device =
            materialized_mtp_verifier_device_token_row_.first_token_from_device;
        const int materialized_first_target_sample_slot =
            materialized_mtp_verifier_device_token_row_.first_target_sample_slot;

        if (!sealDeviceSpeculativeOutcomeVerifierGeometry(
                out_handle,
                /*request_count=*/1,
                draft_token_count,
                "graph_owned_greedy_device_outcome"))
        {
            return false;
        }

        if (materialized_first_token_from_device)
        {
            clearStochasticTargetSampleReadySlot(
                materialized_first_target_sample_slot,
                StochasticSampleReadyClearMode::Force);
        }
        materialized_mtp_verifier_device_token_row_ = {};
        materialized_mtp_verifier_device_token_batch_ = {};

        if (!recordDeviceResidentMTPTransactionMutation(
                /*request_count=*/1,
                stream,
                "graph_owned_greedy_device_outcome"))
        {
            return false;
        }

        out_handle->output_tokens_device =
            static_cast<const int32_t *>(stochastic_batch_output_tokens_dev_);
        out_handle->meta_device =
            static_cast<int *>(stochastic_batch_output_meta_dev_);
        out_handle->output_token_stride =
            stochastic_batch_output_token_stride_;
        out_handle->meta_stride = kSpeculativeBatchMetaCount;
        out_handle->device = state_.device_id;
        out_handle->stream = stream;
        out_handle->response_ready_event = grouped_ready->event;
        out_handle->mtp_transaction =
            currentDeviceResidentMTPTransactionLease();
        out_handle->device_generation_controller_owned = true;
        out_handle->sampling_mode = DeviceGenerationSamplingMode::Greedy;
        out_handle->local_tp_participant_complete =
            publishesFullVocabularyMTPLogits() &&
            graph_builder_ &&
            graph_builder_->config().tp_ctx &&
            graph_builder_->config().tp_ctx->degree() > 1;

        greedy_verifier_outcome_graph_transaction_ = {};
        mtp_verifier_outcome_graph_mode_ =
            MTPVerifierOutcomeGraphMode::Disabled;

        PerfStatsCollector::addCounter(
            "mtp",
            "all_position_greedy_device_resident_outcomes",
            1.0,
            "decode",
            state_.device_id.toString(),
            {{"compare_rows", std::to_string(draft_token_count - 1)},
             {"implementation", "graph_owned_outcome"}});
        PerfStatsCollector::addCounter(
            "mtp",
            "graph_owned_greedy_outcome_consumptions",
            1.0,
            "decode",
            state_.device_id.toString(),
            {{"rows", std::to_string(draft_token_count)}});
        return out_handle->valid();
    }

    bool DeviceGraphOrchestrator::verifyGreedyAllPositionRequestBatchOutcomesOnDeviceResident(
        const DeviceGreedyBatchOutcomeRequest *requests,
        int request_count,
        DeviceSpeculativeOutcomeHandle *out_handle)
    {
        using namespace sampling_math;
        if (out_handle)
            *out_handle = DeviceSpeculativeOutcomeHandle{};
        if (!out_handle ||
            !requests ||
            request_count <= 0 ||
            request_count > stochastic_batch_output_request_capacity_ ||
            !state_.all_position_logits ||
            !state_.device_id.is_gpu() ||
            !materialized_mtp_verifier_device_token_batch_.valid ||
            materialized_mtp_verifier_device_token_batch_.request_count !=
                request_count ||
            materialized_mtp_verifier_device_token_batch_
                    .physical_padded_seq_len <
                materialized_mtp_verifier_device_token_batch_
                    .logical_padded_seq_len ||
            (materialized_mtp_verifier_device_token_batch_
                         .physical_padded_seq_len !=
                     materialized_mtp_verifier_device_token_batch_
                         .logical_padded_seq_len &&
             request_count != 1) ||
            materialized_mtp_verifier_device_token_batch_
                    .total_token_capacity !=
                request_count *
                    materialized_mtp_verifier_device_token_batch_
                        .physical_padded_seq_len ||
            !mtp_verifier_input_tokens_dev_ ||
            !stochastic_verify_tokens_dev_ ||
            !stochastic_verify_accept_probs_dev_ ||
            !stochastic_batch_output_tokens_dev_ ||
            !stochastic_batch_output_meta_dev_)
        {
            return false;
        }

        int max_target_row = 0;
        for (int i = 0; i < request_count; ++i)
        {
            const DeviceGreedyBatchOutcomeRequest &request = requests[i];
            if (request.first_target_row < 0 ||
                request.verifier_token_count <= 0 ||
                request.verifier_token_count > mtp_max_verifier_rows_ ||
                request.token_row_offset < 0 ||
                request.token_row_stride <= 0 ||
                request.token_row_stride !=
                    materialized_mtp_verifier_device_token_batch_
                        .logical_padded_seq_len ||
                request.stop_token_count < 0 ||
                request.stop_token_count >
                    static_cast<int>(kSpeculativeBatchMaxStopTokens) ||
                request.token_row_offset + request.verifier_token_count >
                    materialized_mtp_verifier_device_token_batch_
                        .total_token_capacity)
            {
                return false;
            }
            max_target_row = std::max(
                max_target_row,
                request.first_target_row + request.verifier_token_count);
        }
        if (max_target_row <= 0 ||
            max_target_row > stochastic_target_row_capacity_ ||
            activeAllPositionLogitsAreColumnParallel(max_target_row))
        {
            return false;
        }

        const IGlobalTPContext *global_ctx =
            globalTPContextForMTPCoordination();
        if (global_ctx && global_ctx->degree() > 1)
            return false;

        TensorBase *tensor = state_.all_position_logits.get();
        const auto &shape = tensor->shape();
        if (shape.empty())
            return false;
        const size_t rows = shape.size() >= 2 ? shape[0] : 1;
        const size_t cols = shape.size() >= 2 ? shape[1] : shape[0];
        if (cols == 0 ||
            static_cast<size_t>(max_target_row) > rows ||
            cols > static_cast<size_t>(std::numeric_limits<int>::max()) ||
            !state_.all_position_logits->deviceValid())
        {
            return false;
        }

        auto device_opt = tensor->current_device();
        if (!device_opt.has_value() || !device_opt->is_gpu())
            return false;

        IBackend *backend = getBackendFor(*device_opt);
        const void *gpu_ptr = tensor->gpu_data_ptr();
        if (!backend || !gpu_ptr)
            return false;

        void *stream = consumePendingLogitsStream(
            PendingLogitsStreamRole::AllPositionVerifier,
            "verifyGreedyAllPositionRequestBatchOutcomesOnDeviceResident");
        if (!stream)
        {
            stream = explicitGPUStreamForOperation(
                "verifyGreedyAllPositionRequestBatchOutcomesOnDeviceResident");
        }
        if (!stream)
            return false;

        std::shared_ptr<void> producer_start_timing_event;
        std::shared_ptr<void> producer_stop_timing_event;
        const bool collect_producer_gpu_timing =
            PerfStatsCollector::isDomainEnabled("mtp") &&
            state_.device_id.is_gpu();
        if (collect_producer_gpu_timing)
        {
            const int device_ordinal = device_opt->gpu_ordinal();
            if (!acquirePersistentMTPGpuTimingEvents(
                    "greedy_request_batch_compact_outcome_summary",
                    &producer_start_timing_event,
                    &producer_stop_timing_event))
            {
                return false;
            }
            if (!backend->recordEvent(
                    producer_start_timing_event.get(),
                    device_ordinal,
                    stream))
            {
                PerfStatsCollector::addCounter(
                    "mtp",
                    "greedy_request_batch_summary_gpu_timing_record_failures",
                    1.0,
                    "decode",
                    state_.device_id.toString(),
                    {{"event", "start"}});
                return false;
            }
        }

        const auto *base = static_cast<const float *>(gpu_ptr);
        if (!backend->enqueueArgmaxF32BatchedRowsDevice(
                base,
                max_target_row,
                static_cast<int>(cols),
                device_opt->gpu_ordinal(),
                stream,
                stochastic_verify_accept_probs_dev_,
                stochastic_verify_tokens_dev_,
                argmax_partial_vals_dev_,
                argmax_partial_idxs_dev_,
                argmax_partial_capacity_))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Greedy request-batch verifier device argmax failed on "
                      << device_opt->toString());
            return false;
        }

        auto *prepared_tokens =
            static_cast<const int32_t *>(mtp_verifier_input_tokens_dev_);
        auto *verify_tokens =
            static_cast<const int32_t *>(stochastic_verify_tokens_dev_);
        auto *output_tokens =
            static_cast<int32_t *>(stochastic_batch_output_tokens_dev_);
        auto *output_meta =
            static_cast<int *>(stochastic_batch_output_meta_dev_);
        for (int request_index = 0;
             request_index < request_count;
             ++request_index)
        {
            const DeviceGreedyBatchOutcomeRequest &request =
                requests[request_index];
            std::array<int, kSpeculativeBatchMaxStopTokens>
                packed_stop_tokens = {-1, -1, -1, -1, -1, -1, -1, -1};
            for (int stop_index = 0;
                 stop_index < request.stop_token_count;
                 ++stop_index)
            {
                packed_stop_tokens[static_cast<size_t>(stop_index)] =
                    request.stop_tokens[static_cast<size_t>(stop_index)];
            }

            if (!backend->enqueueSummarizeGreedySpeculativeVerifyBatch(
                    verify_tokens + request.first_target_row,
                    prepared_tokens + request.token_row_offset,
                    request.verifier_token_count - 1,
                    request.first_token,
                    packed_stop_tokens.data(),
                    request.stop_token_count,
                    device_opt->gpu_ordinal(),
                    stream,
                    stochastic_batch_output_token_stride_,
                    output_tokens +
                        static_cast<size_t>(request_index) *
                            static_cast<size_t>(
                                stochastic_batch_output_token_stride_),
                    output_meta +
                        static_cast<size_t>(request_index) *
                            static_cast<size_t>(kSpeculativeBatchMetaCount)))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Greedy request-batch verifier summary failed on "
                          << device_opt->toString()
                          << " request_index=" << request_index);
                return false;
            }
        }

        if (producer_stop_timing_event)
        {
            if (!backend->recordEvent(
                    producer_stop_timing_event.get(),
                    device_opt->gpu_ordinal(),
                    stream))
            {
                PerfStatsCollector::addCounter(
                    "mtp",
                    "greedy_request_batch_summary_gpu_timing_record_failures",
                    1.0,
                    "decode",
                    state_.device_id.toString(),
                    {{"event", "stop"}});
                return false;
            }
        }

        std::shared_ptr<void> response_ready_event =
            acquirePersistentMTPOutcomeReadyEvent(
                "greedy_request_batch_compact_outcome");
        if (!response_ready_event)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Greedy request-batch outcome has no available preallocated response-ready event");
            return false;
        }
        if (!sealDeviceSpeculativeOutcomeVerifierGeometry(
                out_handle,
                request_count,
                materialized_mtp_verifier_device_token_batch_
                    .logical_padded_seq_len,
                "greedy_request_batch_device_outcome"))
        {
            return false;
        }
        if (!DeviceEventEdge::at(
                 DeviceTimelinePoint::CompactSpeculativeResponseReady)
                 .from(DeviceTimelineRole::VerifierSummary)
                 .publish(
                     *backend,
                     state_.device_id,
                     response_ready_event.get(),
                     stream))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to record greedy request-batch outcome response-ready event");
            return false;
        }

        materialized_mtp_verifier_device_token_batch_ = {};
        if (!recordDeviceResidentMTPTransactionMutation(
                request_count,
                stream,
                "greedy_request_batch_device_outcome"))
        {
            return false;
        }
        out_handle->output_tokens_device =
            static_cast<const int32_t *>(stochastic_batch_output_tokens_dev_);
        out_handle->meta_device =
            static_cast<int *>(stochastic_batch_output_meta_dev_);
        out_handle->output_token_stride =
            stochastic_batch_output_token_stride_;
        out_handle->meta_stride = kSpeculativeBatchMetaCount;
        out_handle->device = state_.device_id;
        out_handle->stream = stream;
        out_handle->response_ready_event = std::move(response_ready_event);
        out_handle->mtp_transaction =
            currentDeviceResidentMTPTransactionLease();
        out_handle->local_tp_participant_complete =
            publishesFullVocabularyMTPLogits() &&
            graph_builder_ &&
            graph_builder_->config().tp_ctx &&
            graph_builder_->config().tp_ctx->isLocal() &&
            graph_builder_->config().tp_ctx->degree() > 1;
        out_handle->producer_start_timing_event =
            std::move(producer_start_timing_event);
        out_handle->producer_stop_timing_event =
            std::move(producer_stop_timing_event);

        PerfStatsCollector::addCounter(
            "mtp",
            "request_batch_greedy_device_resident_outcomes",
            static_cast<double>(request_count),
            "decode",
            state_.device_id.toString(),
            {{"rows", std::to_string(max_target_row)}});
        return out_handle->valid();
    }

    bool DeviceGraphOrchestrator::verifyGreedyAllPositionBatchOutcomeOnDevice(
        const int32_t *draft_tokens,
        int draft_token_count,
        const int32_t *stop_tokens,
        int stop_token_count,
        DeviceSpeculativeVerifyBatchOutcome *out)
    {
        using namespace sampling_math;
        if (!out)
            return false;
        *out = DeviceSpeculativeVerifyBatchOutcome{};

        const int compare_rows = draft_token_count - 1;
        if (!draft_tokens ||
            draft_token_count <= 0 ||
            draft_token_count > mtp_max_verifier_rows_ ||
            compare_rows < 0 ||
            compare_rows > mtp_max_draft_depth_ ||
            stop_token_count < 0 ||
            stop_token_count > kSpeculativeBatchMaxStopTokens ||
            (stop_token_count > 0 && !stop_tokens) ||
            !state_.all_position_logits)
        {
            return false;
        }

        /*
         * See the resident counterpart above: only true local-shard verifier
         * logits are excluded. Mirrored LocalTP MTP heads keep full-vocab rows
         * despite the model's ordinary column-parallel LM-head setting.
         */
        if (activeAllPositionLogitsAreColumnParallel(draft_token_count))
            return false;

        const IGlobalTPContext *global_ctx = globalTPContextForMTPCoordination();
        if (global_ctx && global_ctx->degree() > 1)
            return false;

        TensorBase *tensor = state_.all_position_logits.get();
        const auto &shape = tensor->shape();
        if (shape.empty())
            return false;
        const size_t rows = shape.size() >= 2 ? shape[0] : 1;
        const size_t cols = shape.size() >= 2 ? shape[1] : shape[0];
        if (cols == 0 ||
            static_cast<size_t>(draft_token_count) > rows ||
            cols > static_cast<size_t>(std::numeric_limits<int>::max()))
        {
            return false;
        }

        std::array<int, kSpeculativeBatchMaxStopTokens> packed_stop_tokens =
            {-1, -1, -1, -1, -1, -1, -1, -1};
        for (int i = 0; i < stop_token_count; ++i)
            packed_stop_tokens[static_cast<size_t>(i)] = stop_tokens[i];

        if (state_.device_id.is_cpu())
        {
            /*
             * CPU runners keep verifier logits host-visible already, so the
             * compact vLLM-style greedy outcome can use the same shared summary
             * math without pretending to have a GPU-side reducer. This keeps
             * the SingleDevice CPU correctness gate symmetric with CUDA/ROCm
             * while still excluding sharded TP logits above.
             */
            const float *logits = tensor->fp32_data();
            if (!logits)
                return false;

            std::vector<int> verifier_tokens(
                static_cast<size_t>(draft_token_count));
            for (int row = 0; row < draft_token_count; ++row)
            {
                const float *row_logits =
                    logits + static_cast<size_t>(row) * static_cast<size_t>(cols);
                int best = 0;
                float best_value = row_logits[0];
                for (int col = 1; col < static_cast<int>(cols); ++col)
                {
                    if (row_logits[col] > best_value)
                    {
                        best = col;
                        best_value = row_logits[col];
                    }
                }
                verifier_tokens[static_cast<size_t>(row)] = best;
            }

            std::vector<int> packed_draft_tokens(
                static_cast<size_t>(draft_token_count));
            for (int i = 0; i < draft_token_count; ++i)
                packed_draft_tokens[static_cast<size_t>(i)] = draft_tokens[i];

            std::vector<int> output_tokens_int(
                static_cast<size_t>(draft_token_count));
            std::array<int, kSpeculativeBatchMetaCount> meta{};
            summarize_greedy_speculative_verify_batch(
                draft_tokens[0],
                verifier_tokens.data(),
                packed_draft_tokens.data(),
                compare_rows,
                packed_stop_tokens.data(),
                stop_token_count,
                output_tokens_int.data(),
                static_cast<int>(output_tokens_int.size()),
                meta.data());

            std::vector<int32_t> output_tokens(
                static_cast<size_t>(draft_token_count));
            for (size_t i = 0; i < output_tokens.size(); ++i)
                output_tokens[i] = static_cast<int32_t>(output_tokens_int[i]);

            if (!fillSpeculativeVerifyOutcomeFromMeta(
                    output_tokens.data(),
                    draft_token_count,
                    meta.data(),
                    out))
                return false;

            PerfStatsCollector::addCounter(
                "mtp",
                "all_position_greedy_device_batched_rows",
                static_cast<double>(draft_token_count),
                "decode",
                state_.device_id.toString(),
                {{"compare_rows", std::to_string(compare_rows)},
                 {"implementation", "cpu_compact_outcome"}});
            return true;
        }

        DeviceSpeculativeOutcomeHandle handle;
        if (!verifyGreedyAllPositionBatchOutcomeOnDeviceResident(
                draft_tokens,
                draft_token_count,
                stop_tokens,
                stop_token_count,
                &handle))
        {
            return false;
        }

        if (!copyDeviceSpeculativeOutcomesToHostForDiagnostics(handle, out))
            return false;

        PerfStatsCollector::addCounter(
            "mtp",
            "all_position_greedy_device_batched_rows",
            static_cast<double>(draft_token_count),
            "decode",
            state_.device_id.toString(),
            {{"compare_rows", std::to_string(compare_rows)},
             {"implementation", "device_batch_outcome"}});
        return true;
    }

    bool DeviceGraphOrchestrator::supportsGreedyAllPositionBatchOutcomeOnDevice() const
    {
        /*
         * The compact reducer consumes one device's all-position logits tensor.
         * SingleDevice CPU can execute the same reducer contract over
         * host-visible logits. Rank-level sharded TP still needs a domain-wide
         * reduction contract, so only concrete CPU/GPU device runners advertise
         * this path.
         */
        return state_.device_id.is_cpu() || state_.device_id.is_gpu();
    }

    PrefixRuntimeStateSnapshot DeviceGraphOrchestrator::prefixStateProbe(
        const PrefixProbeCapturePolicy &capture_policy) const
    {
        void *probe_stream = explicitGPUStreamForOperation("prefixStateProbe");
        DeviceResidentLogicalStateReadScope logical_state_read(
            const_cast<DeviceGraphOrchestrator &>(*this),
            probe_stream,
            "prefix_state_probe_logical_state");
        if (state_.device_id.is_gpu() &&
            (!probe_stream ||
             !logical_state_read.ready() ||
             !waitForLiveInferenceStateReadyForObservation(
                 probe_stream,
                 "prefix_state_probe",
                 DeviceTimelineRole::Diagnostics)))
        {
            return {};
        }
        PrefixRuntimeStateSnapshot snapshot;
        snapshot.initialized = state_.isInitialized();
        snapshot.has_hidden = static_cast<bool>(state_.hidden);
        snapshot.has_logits = static_cast<bool>(state_.logits);
        snapshot.architecture = architecture();
        snapshot.execution_path = "graph";
        snapshot.primary_device = state_.device_id;
        snapshot.current_position = getPosition(0);
        snapshot.session_epoch = session_epoch_;
        snapshot.moe_runtime_movement_epoch = PrefixMovementEpochSnapshot::leaf(moeRuntimeMovementEpoch());
        snapshot.live_state_epoch = live_replay_state_epoch_;
        snapshot.live_state_mutations = live_state_mutation_count_;
        snapshot.last_live_state_mutation_reason =
            livePrefixMutationReasonName(last_live_state_mutation_reason_);
        snapshot.last_live_state_mutation_operation =
            last_live_state_mutation_operation_;
        snapshot.live_state_accepted_publications =
            live_state_accepted_publications_;
        snapshot.live_state_rejected_corrections =
            live_state_rejected_corrections_;
        snapshot.live_state_prefix_restores = live_state_prefix_restores_;
        snapshot.live_state_prefix_truncates = live_state_prefix_truncates_;
        snapshot.live_state_session_resets = live_state_session_resets_;
        snapshot.prefix_cache_config_enabled =
            graph_builder_ && graph_builder_->config().prefix_cache.enabled;
        snapshot.prefix_cache_bypassed =
            snapshot.prefix_cache_config_enabled && prefix_cache_bypassed_;
        snapshot.prefix_cache_bypass_reason =
            snapshot.prefix_cache_bypassed ? prefix_cache_bypass_reason_ : "";
        snapshot.prefix_cache_ready =
            static_cast<bool>(prefix_cache_) && !snapshot.prefix_cache_bypassed;
        if (prefix_cache_)
        {
            const auto &stats = prefix_cache_->stats();
            snapshot.prefix_cache_lookups = stats.lookups;
            snapshot.prefix_cache_hits = stats.hits;
            snapshot.prefix_cache_partial_hits = stats.partial_hits;
            snapshot.prefix_cache_misses = stats.misses;
            snapshot.prefix_cache_matched_blocks = stats.matched_blocks;
            snapshot.prefix_cache_matched_tokens = stats.matched_tokens;
            snapshot.prefix_cache_stores = stats.stores;
            snapshot.prefix_cache_inserts = stats.inserts;
            snapshot.prefix_cache_evictions = stats.evictions;
            snapshot.prefix_cache_promotions = stats.promotions;
            snapshot.prefix_cache_ram_to_disk_demotions =
                stats.ram_to_disk_demotions;
            snapshot.prefix_cache_device_hot_promotions =
                stats.device_hot_promotions;
            snapshot.prefix_cache_device_hot_repromotions =
                stats.device_hot_repromotions;
            snapshot.prefix_cache_device_hot_evictions =
                stats.device_hot_evictions;
            snapshot.prefix_cache_disk_evictions = stats.disk_evictions;
            snapshot.prefix_cache_device_hot_direct_hits =
                stats.device_hot_direct_hits;
            snapshot.prefix_cache_disk_hydrations = stats.disk_hydrations;
            snapshot.prefix_cache_terminal_state_hits = stats.terminal_state_hits;
            snapshot.prefix_cache_ram_bytes = stats.ram_bytes;
            snapshot.prefix_cache_device_bytes = stats.device_bytes;
            snapshot.prefix_cache_disk_bytes = stats.disk_bytes;
            snapshot.prefix_cache_hybrid_state_bytes = stats.hybrid_state_bytes;
            snapshot.prefix_cache_mtp_state_bytes = stats.mtp_state_bytes;
        }
        snapshot.prefix_cache_bypasses = prefix_cache_stats_.bypasses;
        snapshot.prefix_cache_unsupported_backend_bypasses =
            prefix_cache_stats_.unsupported_backend_bypasses;
        snapshot.prefix_cache_fingerprint_bypasses =
            prefix_cache_stats_.fingerprint_bypasses;
        snapshot.prefix_cache_terminal_state_bypasses =
            prefix_cache_stats_.terminal_state_bypasses;
        snapshot.mtp_config_enabled =
            graph_builder_ && graph_builder_->config().mtp.enabled;

        /*
         * Terminal state is the only persistent input consumed before the
         * first decode transaction has produced new hidden/logit rows.  Hash
         * it at this explicit diagnostic result boundary so a prefix-restore
         * investigation can distinguish bad archive bytes from stale
         * graph-owned workspace.  The opt-in copy is deliberately absent from
         * production execution.
         */
        if (capture_policy.hash_terminal_state)
        {
            auto hash_terminal_tensor =
                [&](const TensorBase *tensor,
                    size_t bytes,
                    bool *available,
                    size_t *reported_bytes,
                    uint64_t *reported_hash,
                    std::vector<float> *reported_fp32_values,
                    const char *name,
                    size_t source_byte_offset = 0)
            {
                if (!tensor || bytes == 0)
                    return;

                std::vector<uint8_t> payload(bytes);
                if (state_.device_id.is_gpu())
                {
                    IBackend *backend = getBackendFor(state_.device_id);
                    const auto *base = static_cast<const uint8_t *>(tensor->gpu_data_ptr());
                    const void *source = base ? base + source_byte_offset : nullptr;
                    if (!backend || !probe_stream || !source ||
                        !backend->deviceToHostFast(
                            payload.data(),
                            source,
                            bytes,
                            state_.device_id.gpu_ordinal(),
                            probe_stream))
                    {
                        throw std::runtime_error(
                            std::string("Prefix-state diagnostics could not "
                                        "materialize terminal ") +
                            name);
                    }
                }
                else
                {
                    const auto *base = static_cast<const uint8_t *>(tensor->raw_data());
                    const void *source = base ? base + source_byte_offset : nullptr;
                    if (!source)
                    {
                        throw std::runtime_error(
                            std::string("Prefix-state diagnostics could not "
                                        "read terminal ") +
                            name);
                    }
                    std::memcpy(payload.data(), source, bytes);
                }

                *available = true;
                *reported_bytes = bytes;
                *reported_hash =
                    hashByteBufferForPrefixProbe(payload.data(), payload.size());
                if (reported_fp32_values)
                {
                    if (tensor->native_type() != TensorType::FP32 ||
                        bytes % sizeof(float) != 0u)
                    {
                        throw std::runtime_error(
                            std::string("Prefix-state diagnostics cannot retain non-FP32 terminal ") +
                            name);
                    }
                    reported_fp32_values->resize(bytes / sizeof(float));
                    std::memcpy(
                        reported_fp32_values->data(),
                        payload.data(),
                        bytes);
                }
            };

            if (state_.prefix_terminal_hidden &&
                state_.mtp_terminal_hidden_publication.current() &&
                prefix_layout_.terminal_hidden_bytes > 0)
            {
                hash_terminal_tensor(
                    state_.prefix_terminal_hidden.get(),
                    prefix_layout_.terminal_hidden_bytes,
                    &snapshot.terminal_hidden_hash_available,
                    &snapshot.terminal_hidden_bytes,
                    &snapshot.terminal_hidden_hash,
                    capture_policy.capture_terminal_hidden_values
                        ? &snapshot.terminal_hidden_values
                        : nullptr,
                    "hidden row");
            }

            /*
             * Observe the tensor named by the latest ordered producer.  A
             * restored-prefix MTP bridge uses scalar main semantics but writes
             * row zero of the preplanned all-position surface; the dormant
             * canonical LOGITS allocation still contains the prompt row.
             */
            const auto terminal_logits = currentPrefixArchiveLogits();
            if (terminal_logits)
            {
                hash_terminal_tensor(
                    terminal_logits->tensor,
                    terminal_logits->slice.byteCount(),
                    &snapshot.terminal_logits_hash_available,
                    &snapshot.terminal_logits_bytes,
                    &snapshot.terminal_logits_hash,
                    capture_policy.capture_terminal_logits_values
                        ? &snapshot.terminal_logits_values
                        : nullptr,
                    "logits row",
                    terminal_logits->slice.byteOffset());
            }

            /*
             * The aggregate probe above is the durable correctness evidence.
             * A focused trace additionally names the participant-local source
             * row and both embedded addresses so a captured graph that writes
             * an obsolete archive allocation can be distinguished from bad
             * row arithmetic.  This extra D2H is reachable only when both the
             * deep terminal-state probe and explicit prefix trace are enabled.
             */
            if (prefixCacheTraceEnabled() && state_.hidden &&
                state_.hidden->native_type() == TensorType::FP32 &&
                state_.d_model > 0 &&
                state_.last_forward_batch_size == 1 &&
                state_.last_forward_seq_len > 0 &&
                !state_.last_forward_request_lengths.empty())
            {
                const int terminal_row = std::clamp(
                    state_.last_forward_request_lengths.front() - 1,
                    0,
                    state_.last_forward_seq_len - 1);
                const size_t row_bytes =
                    static_cast<size_t>(state_.d_model) * sizeof(float);
                std::vector<uint8_t> live_hidden(row_bytes);
                const auto *hidden_base = static_cast<const uint8_t *>(
                    state_.device_id.is_gpu()
                        ? state_.hidden->gpu_data_ptr()
                        : state_.hidden->raw_data());
                const void *hidden_source =
                    hidden_base
                        ? hidden_base +
                              static_cast<size_t>(terminal_row) * row_bytes
                        : nullptr;
                bool materialized = false;
                if (state_.device_id.is_gpu())
                {
                    IBackend *backend = getBackendFor(state_.device_id);
                    materialized = backend && probe_stream && hidden_source &&
                        backend->deviceToHostFast(
                            live_hidden.data(),
                            hidden_source,
                            row_bytes,
                            state_.device_id.gpu_ordinal(),
                            probe_stream);
                }
                else if (hidden_source)
                {
                    std::memcpy(
                        live_hidden.data(), hidden_source, row_bytes);
                    materialized = true;
                }

                LOG_DEBUG(
                    "[PrefixTerminalTrace] device="
                    << state_.device_id.toString()
                    << " position=" << snapshot.current_position
                    << " last_forward_seq_len="
                    << state_.last_forward_seq_len
                    << " terminal_row=" << terminal_row
                    << " hidden_tensor=" << state_.hidden.get()
                    << " hidden_source=" << hidden_source
                    << " hidden_hash="
                    << (materialized
                            ? hashByteBufferForPrefixProbe(
                                  live_hidden.data(), live_hidden.size())
                            : uint64_t{0})
                    << " archive_tensor="
                    << state_.prefix_terminal_hidden.get()
                    << " archive_source="
                    << (state_.prefix_terminal_hidden
                            ? state_.prefix_terminal_hidden->gpu_data_ptr()
                            : nullptr)
                    << " archive_hash="
                    << (snapshot.terminal_hidden_hash_available
                            ? snapshot.terminal_hidden_hash
                            : uint64_t{0}));
            }
        }
        snapshot.positions = state_.positions;
        snapshot.sequence_lengths = state_.sequence_lengths;
        if (state_.device_id.is_gpu() &&
            !capture_policy.capture_device_logical_state)
        {
            /*
             * GPU logical sequence state is fully device owned. An ordinary
             * runtime summary may report host-owned counters and graph/cache
             * inventory, but it must not revive the scheduler mirror or issue
             * a D2H merely to populate convenience vectors. A focused deep
             * probe opts in explicitly above; production benchmark and server
             * summaries therefore cannot perturb or synchronize execution.
             */
            snapshot.current_position = -1;
            snapshot.positions.clear();
            snapshot.sequence_lengths.clear();
        }
        if (mtpPublicationDiagnosticsEnabled())
        {
            const auto &mailbox =
                device_resident_logical_sequence_state_mailbox_;
            LOG_INFO(
                "[MTPPublicationDiagnostics] phase=prefix_state_probe_admission"
                << " device=" << state_.device_id.toString()
                << " mailbox_valid=" << (mailbox.valid() ? "true" : "false")
                << " mailbox_epoch=" << mailbox.live_state_epoch
                << " publication_generation="
                << mailbox.publication_generation
                << " publication_kind="
                << deviceResidentLogicalStatePublicationKindName(
                       mailbox.publication_kind)
                << " live_state_epoch=" << live_replay_state_epoch_
                << " storage_published="
                << (device_resident_logical_sequence_state_storage_
                            .hasLivePublication(live_replay_state_epoch_)
                        ? "true"
                        : "false")
                << " last_mutation_reason="
                << livePrefixMutationReasonName(last_live_state_mutation_reason_)
                << " last_mutation_operation="
                << last_live_state_mutation_operation_);
        }
        const DeviceResidentLogicalSequenceStateHandle logical_state =
            deviceResidentLogicalSequenceState();
        if (logical_state.valid() &&
            capture_policy.capture_device_logical_state)
        {
            /*
             * A diagnostic snapshot is a host-visible result boundary, so it
             * may materialize the small logical metadata row.  Read the entire
             * row from the authoritative mailbox in one operation rather than
             * mixing request zero with stale host vectors for later requests.
             * Any failed observation is fatal: returning the host mirror would
             * make a coherence defect look like a valid runtime state.
             */
            IBackend *backend = getBackendFor(state_.device_id);
            if (!backend)
            {
                throw std::runtime_error(
                    "Prefix-state diagnostics could not resolve the GPU backend");
            }

            const auto &storage =
                device_resident_logical_sequence_state_storage_;
            const size_t request_count =
                static_cast<size_t>(logical_state.request_count);
            const size_t request_capacity =
                static_cast<size_t>(storage.request_capacity);

            /*
             * The logical mailbox is one packed arena allocation. Preserve
             * that ownership boundary during diagnostics: copy one immutable
             * publication into the dedicated device diagnostic owner, then
             * export that owner once instead of issuing pageable-host copies
             * from individual live fields. Adjacent device-only phase
             * snapshots proved that the live publication remained valid while
             * direct CUDA observations left their host destinations untouched.
             *
             * The handle is a typed view over this exact storage. Reject any
             * future implementation that silently redirects one field to a
             * transient workspace, because combining owners would make even a
             * packed diagnostic internally inconsistent.
             */
            if (!storage.validFor(logical_state.request_count) ||
                logical_state.target_positions_device !=
                    storage.target_cached_tokens_device ||
                logical_state.target_sequence_lengths_device !=
                    storage.target_cached_tokens_device ||
                logical_state.accepted_state_counts_device !=
                    storage.accepted_state_counts_device ||
                logical_state.next_condition_tokens_device !=
                    storage.next_condition_tokens_device ||
                logical_state.all_drafts_accepted_flags_device !=
                    storage.all_drafts_accepted_flags_device ||
                logical_state.stopped_flags_device !=
                    storage.stopped_flags_device ||
                logical_state.publication_ok_flags_device !=
                    storage.publication_ok_flags_device)
            {
                throw std::runtime_error(
                    "Prefix-state diagnostics rejected a logical-state handle "
                    "that does not alias the canonical packed arena owner");
            }

            const size_t packed_element_count =
                DeviceResidentLogicalSequenceStateStorage::kFieldCount *
                request_capacity;
            auto &diagnostic_storage =
                device_resident_logical_state_diagnostic_storage_;
            int32_t *const staging_device =
                diagnostic_storage.probeStagingBase();
            if (!diagnostic_storage.validFor(logical_state.request_count) ||
                !staging_device)
            {
                throw std::runtime_error(
                    "Deep GPU logical-state diagnostics require preallocated "
                    "device staging; enable the capture policy before runner "
                    "initialization");
            }
            if (!backend->deviceCopyAsync(
                    staging_device,
                    storage.initialization_base_cached_tokens_scratch_device,
                    packed_element_count * sizeof(int32_t),
                    state_.device_id.gpu_ordinal(),
                    probe_stream))
            {
                throw std::runtime_error(
                    "Prefix-state diagnostics could not stage the packed "
                    "device-resident logical-state publication");
            }

            const size_t diagnostic_element_count =
                DeviceResidentLogicalStateDiagnosticStorage::kRowCount *
                request_capacity;
            std::vector<int32_t> diagnostic_host_values(
                diagnostic_element_count,
                -1);
            if (!backend->deviceToHostFast(
                    diagnostic_host_values.data(),
                    diagnostic_storage.base_device,
                    diagnostic_element_count * sizeof(int32_t),
                    state_.device_id.gpu_ordinal(),
                    probe_stream))
            {
                throw std::runtime_error(
                    "Prefix-state diagnostics could not export the staged "
                    "device-resident logical-state publication");
            }

            const size_t staging_offset =
                DeviceResidentLogicalStateDiagnosticStorage::
                    kSnapshotRowCount *
                request_capacity;
            const auto packed_begin =
                diagnostic_host_values.begin() +
                static_cast<std::ptrdiff_t>(staging_offset);
            std::vector<int32_t> packed_state(
                packed_begin,
                packed_begin +
                    static_cast<std::ptrdiff_t>(packed_element_count));

            PerfStatsCollector::addCounter(
                "mtp",
                "device_logical_state_deep_probe_exports",
                1.0,
                perfPhaseName(),
                state_.device_id.toString(),
                {{"requests", std::to_string(logical_state.request_count)},
                 {"source_owner", "arena_persistent"},
                 {"transfer_owner", "diagnostic_staging"}});

            auto materialize_packed_row =
                [&](DeviceResidentLogicalSequenceStateStorage::Field field)
            {
                const size_t row_offset =
                    static_cast<size_t>(field) * request_capacity;
                return std::vector<int32_t>(
                    packed_state.begin() + static_cast<std::ptrdiff_t>(row_offset),
                    packed_state.begin() + static_cast<std::ptrdiff_t>(
                                               row_offset + request_count));
            };

            std::vector<int32_t> resident_positions =
                materialize_packed_row(
                    DeviceResidentLogicalSequenceStateStorage::Field::
                        TargetCachedTokens);
            std::vector<int32_t> resident_sequence_lengths = resident_positions;

            const auto contains_negative = [](const std::vector<int32_t> &values)
            {
                return std::any_of(
                    values.begin(),
                    values.end(),
                    [](int32_t value) { return value < 0; });
            };
            if (contains_negative(resident_positions) ||
                contains_negative(resident_sequence_lengths))
            {
                const std::vector<int32_t> accepted_state_counts =
                    materialize_packed_row(
                        DeviceResidentLogicalSequenceStateStorage::Field::
                            AcceptedStateCounts);
                const std::vector<int32_t> next_condition_tokens =
                    materialize_packed_row(
                        DeviceResidentLogicalSequenceStateStorage::Field::
                            NextConditionTokens);
                const std::vector<int32_t> all_drafts_accepted_flags =
                    materialize_packed_row(
                        DeviceResidentLogicalSequenceStateStorage::Field::
                            AllDraftsAcceptedFlags);
                const std::vector<int32_t> stopped_flags =
                    materialize_packed_row(
                        DeviceResidentLogicalSequenceStateStorage::Field::
                            StoppedFlags);
                const std::vector<int32_t> publication_ok_flags =
                    materialize_packed_row(
                        DeviceResidentLogicalSequenceStateStorage::Field::
                            PublicationOkFlags);

                const auto &mailbox =
                    device_resident_logical_sequence_state_mailbox_;
                std::ostringstream message;
                message
                    << "Prefix-state diagnostics observed invalid negative "
                       "device-resident logical metadata"
                    << " device=" << state_.device_id.toString()
                    << " requests=" << logical_state.request_count
                    << " mailbox_epoch=" << logical_state.live_state_epoch
                    << " publication_generation="
                    << device_resident_logical_sequence_state_mailbox_
                           .publication_generation
                    << " publication_kind="
                    << deviceResidentLogicalStatePublicationKindName(
                           device_resident_logical_sequence_state_mailbox_
                               .publication_kind)
                    << " live_state_epoch=" << live_replay_state_epoch_
                    << " publication_generation="
                    << mailbox.publication_generation
                    << " publication_kind="
                    << deviceResidentLogicalStatePublicationKindName(
                           mailbox.publication_kind)
                    << " positions="
                    << formatMTPDiagnosticIntVector(resident_positions)
                    << " sequence_lengths="
                    << formatMTPDiagnosticIntVector(
                           resident_sequence_lengths)
                    << " accepted_state_counts="
                    << formatMTPDiagnosticIntVector(accepted_state_counts)
                    << " next_condition_tokens="
                    << formatMTPDiagnosticIntVector(next_condition_tokens)
                    << " all_drafts_accepted_flags="
                    << formatMTPDiagnosticIntVector(
                           all_drafts_accepted_flags)
                    << " stopped_flags="
                    << formatMTPDiagnosticIntVector(stopped_flags)
                    << " publication_ok_flags="
                    << formatMTPDiagnosticIntVector(publication_ok_flags)
                    << " logical_state_addresses={target="
                    << static_cast<const void *>(
                           logical_state.target_positions_device)
                    << ",accepted="
                    << static_cast<const void *>(
                           logical_state.accepted_state_counts_device)
                    << ",next_token="
                    << static_cast<const void *>(
                           logical_state.next_condition_tokens_device)
                    << ",all_accepted="
                    << static_cast<const void *>(
                           logical_state.all_drafts_accepted_flags_device)
                    << ",stopped="
                    << static_cast<const void *>(
                           logical_state.stopped_flags_device)
                    << ",ok="
                    << static_cast<const void *>(
                           logical_state.publication_ok_flags_device)
                    << ",init_scratch="
                    << static_cast<const void *>(
                           device_resident_logical_sequence_state_storage_
                               .initialization_base_cached_tokens_scratch_device)
                    << ",request_capacity="
                    << device_resident_logical_sequence_state_storage_
                           .request_capacity
                    << ",accepted_index_workspace="
                    << static_cast<const void *>(
                           mtp_spec_decode_metadata_binding_
                               .devicePointers()
                               .accepted_state_slot_indices)
                    << "}"
                    << " last_mutation_reason="
                    << livePrefixMutationReasonName(
                           last_live_state_mutation_reason_)
                    << " last_mutation_operation="
                    << last_live_state_mutation_operation_
                    << ' '
                    << materializeDeviceResidentLogicalStatePhaseDiagnostics(
                           probe_stream);
                PerfStatsCollector::addCounterWithSequence(
                    "mtp",
                    "device_resident_logical_state_invalid_observations",
                    1.0,
                    {static_cast<uint64_t>(mailbox.publication_generation)},
                    "decode",
                    state_.device_id.toString(),
                    {{"publication_kind",
                      deviceResidentLogicalStatePublicationKindName(
                          mailbox.publication_kind)}});
                throw std::runtime_error(message.str());
            }

            if (logical_state.request_count == 1)
            {
                // Reuse the SAME ordered packed export. No second readback or
                // host response token supplies this continuation identity.
                const auto scalar = [&](DeviceResidentLogicalSequenceStateStorage::Field field) {
                    return packed_state[static_cast<size_t>(field) * request_capacity];
                };
                snapshot.resident_condition = PrefixResidentConditionProbe{
                    .device = logical_state.device,
                    .publication_generation = logical_state.publication_generation,
                    .live_state_epoch = logical_state.live_state_epoch,
                    .target_position = resident_positions.front(),
                    .next_condition_token = scalar(DeviceResidentLogicalSequenceStateStorage::Field::NextConditionTokens),
                    .publication_ok = scalar(DeviceResidentLogicalSequenceStateStorage::Field::PublicationOkFlags) == 1,
                    .stopped = scalar(DeviceResidentLogicalSequenceStateStorage::Field::StoppedFlags) != 0};
            }
            snapshot.current_position = resident_positions.front();
            snapshot.positions.assign(
                resident_positions.begin(),
                resident_positions.end());
            snapshot.sequence_lengths.assign(
                resident_sequence_lengths.begin(),
                resident_sequence_lengths.end());
            if (mtpPublicationDiagnosticsEnabled())
            {
                LOG_INFO(
                    "[MTPPublicationDiagnostics] phase=prefix_state_probe_observed"
                    << " device=" << state_.device_id.toString()
                    << " mailbox_epoch=" << logical_state.live_state_epoch
                    << " live_state_epoch=" << live_replay_state_epoch_
                    << " resident_positions="
                    << formatMTPDiagnosticIntVector(resident_positions)
                    << " resident_sequence_lengths="
                    << formatMTPDiagnosticIntVector(
                           resident_sequence_lengths));
            }
        }
        else if (state_.device_id.is_gpu() &&
                 state_.kv_cache &&
                 capture_policy.capture_device_logical_state)
        {
            /*
             * A restore/truncate boundary invalidates the prior speculative
             * outcome mailbox because its token and acceptance fields no
             * longer describe the live timeline. That does not make the host
             * position vector authoritative. Materialize each canonical
             * cache-owned count only at this explicit diagnostic boundary,
             * after the mutation event has already been inherited above.
             *
             * Failure is fatal. Returning state_.positions here would turn a
             * missing device ownership contract into a plausible but stale
             * probe, exactly the failure mode this diagnostic is intended to
             * expose.
             */
            std::vector<int32_t> canonical_positions;
            canonical_positions.reserve(
                static_cast<size_t>(std::max(1, state_.batch_size)));
            for (int request = 0;
                 request < std::max(1, state_.batch_size);
                 ++request)
            {
                const std::optional<int> canonical_position =
                    deviceResidentLogicalTokenCountForObservation(
                        request,
                        probe_stream,
                        "prefix_state_probe_canonical_kv_count");
                if (!canonical_position)
                {
                    throw std::runtime_error(
                        "Prefix-state diagnostics could not observe the "
                        "canonical GPU KV sequence count after mailbox retirement");
                }
                canonical_positions.push_back(*canonical_position);
            }

            snapshot.current_position = canonical_positions.front();
            snapshot.positions.assign(
                canonical_positions.begin(),
                canonical_positions.end());
            snapshot.sequence_lengths.assign(
                canonical_positions.begin(),
                canonical_positions.end());
            if (mtpPublicationDiagnosticsEnabled())
            {
                LOG_INFO(
                    "[MTPPublicationDiagnostics] phase=prefix_state_probe_observed"
                    << " device=" << state_.device_id.toString()
                    << " mailbox_epoch=retired"
                    << " live_state_epoch=" << live_replay_state_epoch_
                    << " state_owner=canonical_kv_sequence_count"
                    << " resident_positions="
                    << formatMTPDiagnosticIntVector(canonical_positions)
                    << " resident_sequence_lengths="
                    << formatMTPDiagnosticIntVector(canonical_positions));
            }
        }

        const int sequence_count = state_.batch_size > 0 ? state_.batch_size : 1;
        if (state_.kv_cache)
        {
            auto cache_probe = inspectKVCacheForPrefixProbe(
                *state_.kv_cache,
                "primary",
                state_.device_id,
                sequence_count,
                probe_stream,
                capture_policy);
            auto gdn_probes = inspectHybridGDNForPrefixProbe(
                *state_.kv_cache,
                probe_stream,
                capture_policy);
            snapshot.gdn_layers.insert(snapshot.gdn_layers.end(),
                                       gdn_probes.begin(), gdn_probes.end());
            snapshot.kv_caches.push_back(std::move(cache_probe));
        }

        for (size_t depth = 0; depth < state_.mtp_kv_caches.size(); ++depth)
        {
            const auto &cache = state_.mtp_kv_caches[depth];
            if (!cache)
            {
                continue;
            }
            auto cache_probe = inspectKVCacheForPrefixProbe(
                *cache,
                "mtp:" + std::to_string(depth),
                state_.device_id,
                sequence_count,
                probe_stream,
                capture_policy);
            if (capture_policy.capture_device_logical_state)
            {
                if (auto resident_mtp_tokens =
                    deviceResidentShiftedMTPKVTokenCountForObservation(
                        static_cast<int>(depth),
                        /*request_index=*/0,
                        probe_stream,
                        "prefix_state_probe_shifted_mtp"))
                {
                    for (auto &layer_probe : cache_probe.layers)
                    {
                        if (layer_probe.seq_idx == 0)
                        {
                            layer_probe.cached_tokens = *resident_mtp_tokens;
                            layer_probe.ring_head = *resident_mtp_tokens;
                        }
                    }
                }
            }
            snapshot.mtp_kv_caches.push_back(std::move(cache_probe));
        }

        if (forward_engine_)
        {
            auto phase_name = [](PrefillGraphPhase phase) -> const char *
            {
                switch (phase)
                {
                case PrefillGraphPhase::Disabled:
                    return "disabled";
                case PrefillGraphPhase::Cold:
                    return "cold";
                case PrefillGraphPhase::Initialized:
                    return "initialized";
                case PrefillGraphPhase::Warmup:
                    return "warmup";
                case PrefillGraphPhase::Capturing:
                    return "capturing";
                case PrefillGraphPhase::Ready:
                    return "ready";
                }
                return "unknown";
            };

            const auto prefill_snapshots = forward_engine_->prefillGraphCacheSnapshots();
            snapshot.prefill_graphs.reserve(prefill_snapshots.size());
            for (const auto &entry : prefill_snapshots)
            {
                PrefillGraphRuntimeProbe probe;
                probe.forward_cache_valid = entry.forward_cache_valid;
                probe.prefill_cache_initialized = entry.prefill_cache_initialized;
                probe.phase = phase_name(entry.phase);
                probe.cache_size = entry.cache_size;
                probe.node_count = entry.node_count;
                probe.replay_count = entry.replay_count;
                probe.warmup_count = entry.warmup_count;
                probe.initialized_count = entry.initialized_count;
                probe.capture_count = entry.capture_count;
                probe.eviction_count = entry.eviction_count;
                probe.observation_valid = entry.observation_valid;
                probe.chunk_index = entry.chunk_index;
                probe.bucket_seq_len = entry.bucket_seq_len;
                probe.real_token_start = entry.real_token_start;
                probe.real_token_count = entry.real_token_count;
                probe.real_token_end = entry.real_token_end;
                probe.domain_id = entry.domain_id;
                probe.participant_id = entry.participant_id;
                probe.placement_epoch = entry.placement_epoch;
                probe.topology_signature = entry.topology_signature;
                probe.capture_phase = entry.capture_phase;
                probe.recapture_reason = entry.recapture_reason;
                probe.reject_stage_name = entry.reject_stage_name;
                probe.reject_stage_type = entry.reject_stage_type;
                snapshot.prefill_graphs.push_back(std::move(probe));
            }
        }

        if ((snapshot_enabled_ ||
             capture_policy.capture_device_logical_state) &&
            graph_builder_ &&
            graph_builder_->config().mtp.enabled &&
            mtp_max_draft_depth_ > 0)
        {
            /*
             * The reusable verifier row cannot prove which committed state
             * consumed it. GPU response/state publication snapshots its exact
             * active prefix on device; the CPU grouped publisher commits the
             * same record in host authority. Both are one typed fact containing
             * transaction ordinal, selected depth, and token bytes.
             */
            sampling_math::MTPCommittedVerifierIdentityRecord identity{};
            if (state_.device_id.is_gpu())
            {
                IBackend *const backend = getBackendFor(state_.device_id);
                if (!mtp_committed_verifier_identity_dev_ ||
                    !arena_ ||
                    !arena_->isRegistered(
                        BufferId::MTP_COMMITTED_VERIFIER_IDENTITY) ||
                    !arena_->prepareForRead(
                        BufferId::MTP_COMMITTED_VERIFIER_IDENTITY,
                        state_.device_id,
                        probe_stream))
                {
                    throw std::runtime_error(
                        "Prefix-state diagnostics could not join the exact "
                        "committed MTP verifier-identity producer event");
                }
                if (!backend || !probe_stream ||
                    !backend->deviceToHostFast(
                        &identity,
                        mtp_committed_verifier_identity_dev_,
                        sizeof(identity),
                        state_.device_id.gpu_ordinal(),
                        probe_stream))
                {
                    throw std::runtime_error(
                        "Prefix-state diagnostics could not export the resident "
                        "committed MTP verifier identity");
                }
            }
            else
            {
                if (mtp_committed_verifier_identities_host_.empty())
                {
                    throw std::runtime_error(
                        "Prefix-state diagnostics found no CPU-owned committed "
                        "MTP verifier-identity storage");
                }
                identity = mtp_committed_verifier_identities_host_.front();
            }

            /*
             * Admission clears `valid` on the controller's exact stream. A
             * non-valid record therefore means no transaction has committed in
             * this request, while every malformed valid record is fatal rather
             * than silently interpreted as an absent proposal.
             */
            if (identity.valid != 0)
            {
                const bool header_valid =
                    identity.version ==
                        sampling_math::
                            kMTPCommittedVerifierIdentityVersion &&
                    identity.transaction_count > 0 &&
                    identity.draft_depth > 0 &&
                    identity.draft_depth <= mtp_max_draft_depth_ &&
                    identity.draft_depth <=
                        sampling_math::kSpeculativeBatchMaxRows;
                bool tokens_valid = header_valid;
                for (int row = 0;
                     tokens_valid &&
                     row <= identity.draft_depth;
                     ++row)
                {
                    tokens_valid =
                        identity.verifier_input_tokens[row] >= 0;
                }
                for (int row = identity.draft_depth + 1;
                     tokens_valid &&
                     row <
                         sampling_math::
                             kSpeculativeBatchMaxOutputTokens;
                     ++row)
                {
                    tokens_valid =
                        identity.verifier_input_tokens[row] == -1;
                }
                if (!tokens_valid)
                {
                    throw std::runtime_error(
                        "Prefix-state diagnostics observed a malformed "
                        "committed MTP verifier identity");
                }

                snapshot.mtp_observed_verifier_transaction_count =
                    identity.transaction_count;
                snapshot.mtp_observed_verifier_draft_depth =
                    identity.draft_depth;
                snapshot.mtp_observed_verifier_condition_token = identity.verifier_input_tokens[0];
                snapshot.mtp_observed_verifier_draft_tokens.assign(
                    identity.verifier_input_tokens + 1,
                    identity.verifier_input_tokens +
                        identity.draft_depth + 1);
            }
        }

        return snapshot;
    }

    void DeviceGraphOrchestrator::disablePrefixCacheForRunner(const std::string &reason)
    {
        const bool should_record =
            !prefix_cache_bypassed_ &&
            graph_builder_ &&
            graph_builder_->config().prefix_cache.enabled &&
            reason != "feature disabled";

        if (should_record)
        {
            ++prefix_cache_stats_.bypasses;
            const std::string normalized_reason = lowerASCII(reason);
            if (containsAny(normalized_reason, {"fingerprint", "moe placement policy"}))
            {
                ++prefix_cache_stats_.fingerprint_bypasses;
            }
            else if (containsAny(normalized_reason, {"terminal"}))
            {
                ++prefix_cache_stats_.terminal_state_bypasses;
            }
            else if (containsAny(normalized_reason,
                                 {"unavailable",
                                  "not implemented",
                                  "does not expose",
                                  "without an initialized",
                                  "graph builder unavailable"}))
            {
                ++prefix_cache_stats_.unsupported_backend_bypasses;
            }
        }
        prefix_cache_bypassed_ = true;
        prefix_cache_bypass_reason_ = reason;
        if (!reason.empty())
        {
            LOG_DEBUG("[DeviceGraphOrchestrator] Prefix cache bypassed: " << reason);
        }
    }

    bool DeviceGraphOrchestrator::isPrefixCacheMoEModel() const
    {
        if (!graph_builder_)
            return false;
        const std::string architecture = graph_builder_->architectureName();
        return architecture.find("moe") != std::string::npos ||
               architecture.find("MoE") != std::string::npos;
    }

    void *DeviceGraphOrchestrator::explicitGPUStreamForOperation(const char *operation) const
    {
        if (!state_.device_id.is_gpu())
        {
            return nullptr;
        }

        try
        {
            void *stream = GPUDeviceContextPool::instance()
                               .getContext(state_.device_id)
                               .defaultStream();
            if (!stream)
            {
                throw std::runtime_error(
                    std::string(operation ? operation : "gpu_operation") +
                    " requires an explicit non-null GPU stream on " +
                    state_.device_id.toString());
            }
            return stream;
        }
        catch (const std::exception &e)
        {
            throw std::runtime_error(
                std::string("[DeviceGraphOrchestrator] ") +
                (operation ? operation : "gpu_operation") +
                " could not acquire an explicit GPU stream on " +
                state_.device_id.toString() + ": " + e.what());
        }
    }

    const char *DeviceGraphOrchestrator::livePrefixMutationReasonName(
        LivePrefixMutationReason reason)
    {
        switch (reason)
        {
        case LivePrefixMutationReason::AcceptedSpecPublication:
            return "accepted_publication";
        case LivePrefixMutationReason::RejectedCorrection:
            return "rejected_correction";
        case LivePrefixMutationReason::PrefixRestore:
            return "prefix_restore";
        case LivePrefixMutationReason::PrefixTruncate:
            return "prefix_truncate";
        case LivePrefixMutationReason::SessionReset:
            return "session_reset";
        case LivePrefixMutationReason::Unknown:
        default:
            return "unknown";
        }
    }

    DeviceGraphOrchestrator::LivePrefixMutationRecord
    DeviceGraphOrchestrator::recordLivePrefixMutation(
        LivePrefixMutationReason reason,
        const char *operation)
    {
        /*
         * An epoch change invalidates every logical-state handle from the old
         * main-state timeline. The owner must first order reusable storage after
         * all readers and retire the mailbox. Enforcing that precondition here
         * makes it impossible to create a valid-looking stale mailbox merely by
         * incrementing the epoch. Auxiliary shifted-MTP KV publication never
         * enters this method; it owns a separate event-backed generation.
         */
        if (state_.device_id.is_gpu() &&
            device_resident_logical_sequence_state_mailbox_.valid())
        {
            std::ostringstream message;
            message
                << "[DeviceGraphOrchestrator] Main live-state epoch mutation "
                   "requires the prior logical-state mailbox to be ordered and "
                   "retired"
                << " operation=" << (operation ? operation : "unknown")
                << " reason=" << livePrefixMutationReasonName(reason)
                << " mailbox_epoch="
                << device_resident_logical_sequence_state_mailbox_
                       .live_state_epoch
                << " live_state_epoch=" << live_replay_state_epoch_;
            throw std::logic_error(message.str());
        }

        const uint64_t previous_epoch = live_replay_state_epoch_;
        ++live_replay_state_epoch_;
        ++live_state_mutation_count_;
        last_live_state_mutation_reason_ = reason;
        last_live_state_mutation_operation_ =
            operation ? operation : "unknown";

        switch (reason)
        {
        case LivePrefixMutationReason::AcceptedSpecPublication:
            ++live_state_accepted_publications_;
            break;
        case LivePrefixMutationReason::RejectedCorrection:
            ++live_state_rejected_corrections_;
            break;
        case LivePrefixMutationReason::PrefixRestore:
            ++live_state_prefix_restores_;
            break;
        case LivePrefixMutationReason::PrefixTruncate:
            ++live_state_prefix_truncates_;
            break;
        case LivePrefixMutationReason::SessionReset:
            ++live_state_session_resets_;
            break;
        case LivePrefixMutationReason::Unknown:
        default:
            break;
        }

        return LivePrefixMutationRecord{
            previous_epoch,
            live_replay_state_epoch_,
            livePrefixMutationReasonName(reason)};
    }

    void DeviceGraphOrchestrator::recordLivePrefixSessionReset(
        const char *operation,
        bool preserve_gpu_replay_state)
    {
        clearPendingAllPositionVerifierStateReady();
        clearPendingAcceptedSpecPublicationReady();
        PerfStatsCollector::Tags tags{{"operation", operation ? operation : "unknown"}};
        const LivePrefixMutationRecord mutation =
            recordLivePrefixMutation(
                LivePrefixMutationReason::SessionReset,
                operation);
        tags["mutation_reason"] = mutation.reason_name;
        if (isPrefixCacheMoEModel())
        {
            tags["model"] = "moe";
        }
        if (preserve_gpu_replay_state)
        {
            tags["forward_replay_reset_scope"] = "request_boundary_preserve";
            tags["replay_state"] = "preserved";
            tags["sidecar_replay_state"] = "preserved";
            tags["kernel_dynamic_state"] = "preserved";
            tags["gpu_replay_preserve_reason"] =
                operation ? operation : "session_reset";
        }
        else
        {
            tags["forward_replay_reset_scope"] = "session";
            tags["replay_state"] = "reset";
            tags["sidecar_replay_state"] = "reset";
            tags["kernel_dynamic_state"] = "reset";
        }
        // Ordered lifecycle evidence keeps epochs out of retained map keys.
        PerfStatsCollector::addCounterWithSequence("mtp",
                                       "live_prefix_replay_state_after_mutation",
                                       1.0,
                                       {mutation.previous_epoch, mutation.live_state_epoch,
                                        live_state_mutation_count_,
                                        isPrefixCacheMoEModel() ? moePlacementEpoch() : 0,
                                        isPrefixCacheMoEModel() ? moeRuntimeMovementEpoch() : 0},
                                       "decode",
                                       state_.device_id.toString(),
                                       std::move(tags));
    }

    bool DeviceGraphOrchestrator::purgePrefixCache()
    {
        /*
         * PrefixStateCache owns every capacity-tier index. Dropping that one
         * authority retires RAM, device-hot, and disk records together while
         * shared payload/event owners keep any already-admitted asynchronous
         * producer safe. The operation intentionally does not touch live KV,
         * graph captures, prepared weights, or placement state.
         */
        if (!prefix_cache_)
            return true;
        if (!prefix_cache_->clear())
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Prefix-cache purge was rejected by an active explicit lease"
                << " device=" << state_.device_id.toString());
            return false;
        }
        prefix_cache_stats_ = prefix_cache_->stats();
        return true;
    }

    void DeviceGraphOrchestrator::resetMTPSidecarDepth0ReplayState()
    {
        mtp_sidecar_depth0_cache_.resetReplayState();
        mtp_sidecar_depth0_device_token_cache_.resetReplayState();
        mtp_sidecar_depth0_chained_cache_.resetReplayState();
        mtp_sidecar_depth0_chained_device_token_cache_.resetReplayState();
        mtp_sidecar_depth0_kv_only_cache_.resetReplayState();
        mtp_sidecar_depth0_kv_only_device_token_cache_.resetReplayState();
        for (auto &cache : mtp_sidecar_depth0_kv_only_batch_caches_)
        {
            cache->resetReplayState();
        }
    }

    void DeviceGraphOrchestrator::handleLivePrefixReplayStateAfterMutation(
        LivePrefixMutationReason reason,
        const char *operation)
    {
        if (reason == LivePrefixMutationReason::PrefixRestore)
        {
            /*
             * Prefix restore is a full live-state replacement.  Clearing every
             * transient handoff here makes the next forward depend only on the
             * freshly imported snapshot plus the mutation-ready event recorded by
             * the caller, not on any stream/mailbox token from the discarded
             * timeline.
             */
            clearLivePrefixRestoreTransientHandoffs(operation);
        }
        else
        {
            clearPendingAllPositionVerifierStateReady();
        }
        if (reason != LivePrefixMutationReason::AcceptedSpecPublication &&
            reason != LivePrefixMutationReason::RejectedCorrection)
        {
            clearPendingAcceptedSpecPublicationReady();
        }

        PerfStatsCollector::Tags tags{{"operation", operation ? operation : "unknown"}};
        const LivePrefixMutationRecord mutation =
            recordLivePrefixMutation(reason, operation);
        tags["mutation_reason"] = mutation.reason_name;
        if (isPrefixCacheMoEModel())
        {
            tags["model"] = "moe";
        }
        bool preserves_live_graph_replay = false;
        if (forward_engine_)
        {
            const bool correction_replay_boundary =
                reason == LivePrefixMutationReason::AcceptedSpecPublication ||
                reason == LivePrefixMutationReason::RejectedCorrection;
            if (correction_replay_boundary)
            {
                /*
                 * Accepted/rejected MTP publication is a live-state boundary:
                 * KV, GDN, short-conv, terminal hidden, and shifted sidecar
                 * caches have just been advanced to a compact verifier row.
                 * The capture lifetime is determined by the graph signature,
                 * not by the backend alone. Single-token ordinary decode reads
                 * backend-owned live buffers at stable addresses, while GDN and
                 * short-conv verifier work lives in separate speculative slots.
                 * CUDA and ROCm integration regressions capture the ordinary
                 * graph once, publish every accepted row at M=2/3/4, and prove
                 * the replayed continuation plus full recurrent state byte-equal
                 * to serial M=1 decode. All-position verifier graphs similarly
                 * refresh their stage-owned row metadata before every launch.
                 *
                 * Multi-row ordinary decode remains live-state-versioned: its
                 * captured executable encodes progression across several live
                 * rows, so publication invalidates that replay state. Delegate
                 * this distinction to the typed cache policy and retain every
                 * executable whose device-state ownership contract is proven.
                 */
                const bool preserve_single_token_decode_replay =
                    state_.device_id.is_gpu();
                const ForwardExecutionEngine::ReplayStateResetSummary summary =
                    forward_engine_->resetCapturedReplayStateForCorrectionReplay(
                        live_replay_state_epoch_,
                        preserve_single_token_decode_replay);
                preserves_live_graph_replay =
                    summary.preserved_for_stream_rebind > 0;
                tags["forward_replay_reset_scope"] =
                    "correction_replay_typed_live_state";
                tags["forward_replay_single_token_decode_replay"] =
                    preserve_single_token_decode_replay
                        ? "preserved_byte_proven_device_state"
                        : "reset_for_publication_boundary";
                tags["forward_replay_reset_cache_count"] =
                    std::to_string(summary.reset_replay_state);
                tags["forward_replay_stream_rebind_cache_count"] =
                    std::to_string(summary.preserved_for_stream_rebind);
                tags["forward_replay_ordinary_decode_reset_count"] =
                    std::to_string(summary.ordinary_decode_reset);
                tags["forward_replay_all_position_verifier_rebind_count"] =
                    std::to_string(summary.all_position_verifier_preserved);
                tags["forward_replay_other_rebind_count"] =
                    std::to_string(summary.other_preserved);
            }
            else
            {
                if (reason == LivePrefixMutationReason::PrefixRestore)
                {
                    if (state_.device_id.is_gpu())
                    {
                        /*
                         * GPU prefix restore replaces contents behind stable
                         * cache/recurrent-state addresses and publishes one
                         * explicit mutation event. Replay-safe executables consume
                         * that event and bind their exact launch streams to it.
                         */
                        const ForwardExecutionEngine::ReplayStateResetSummary
                            summary =
                                forward_engine_
                                    ->rebindCapturedReplayStateAfterPrefixRestore(
                                        live_replay_state_epoch_);
                        preserves_live_graph_replay =
                            summary.preserved_for_stream_rebind > 0;
                        tags["forward_replay_reset_scope"] =
                            "prefix_restore_typed_live_state";
                        tags["forward_replay_reset_cache_count"] =
                            std::to_string(summary.reset_replay_state);
                        tags["forward_replay_stream_rebind_cache_count"] =
                            std::to_string(summary.preserved_for_stream_rebind);
                        tags["forward_replay_ordinary_decode_reset_count"] =
                            std::to_string(summary.ordinary_decode_reset);
                        tags["forward_replay_all_position_verifier_rebind_count"] =
                            std::to_string(
                                summary.all_position_verifier_preserved);
                        tags["forward_replay_other_rebind_count"] =
                            std::to_string(summary.other_preserved);
                    }
                    else
                    {
                        /*
                         * CPU execution has no captured GPU executable or
                         * producer event to rebind. A restored prefix therefore
                         * invalidates cached replay observations outright.
                         */
                        forward_engine_->discardAllCachedGraphs();
                        tags["forward_replay_reset_scope"] =
                            "prefix_restore_cpu_discard_cached_graphs";
                    }
                }
                else
                {
                    forward_engine_->resetCapturedReplayState();
                    tags["forward_replay_reset_scope"] = "all";
                }
            }

            /*
             * GPU sidecar graph replay is a mandatory device-ownership
             * contract, not an advertised optional capability. Dense and MoE
             * sidecars own stable dynamic buffers; MoE routed metadata is
             * depth-scoped as part of graph construction. Accepted/rejected
             * publication therefore changes buffer contents behind stable
             * addresses and is ordered by explicit events without recapture.
             */
            const bool preserve_sidecar_replay =
                state_.device_id.is_gpu() &&
                (correction_replay_boundary ||
                 reason == LivePrefixMutationReason::PrefixRestore);
            if (!preserve_sidecar_replay)
            {
                /*
                 * Resetting sidecar graph replay tears down captured segments.
                 * GraphSegmentCache::reset() synchronizes the explicit capture
                 * stream before resources are discarded, so this timer makes
                 * the real recapture/synchronization tax visible in Phase 10
                 * tuning runs instead of hiding it inside publication latency.
                 */
                PerfStatsCollector::ScopedTimer reset_timer(
                    "mtp",
                    "sidecar_replay_reset",
                    "decode",
                    state_.device_id.toString(),
                    {{"operation", operation ? operation : "unknown"},
                     {"mutation_reason", tags["mutation_reason"]},
                     {"correction_replay_boundary",
                      boolTag(correction_replay_boundary)},
                     {"reset_scope",
                      correction_replay_boundary
                          ? "after_spec_publication"
                          : "all"}});
                resetMTPSidecarDepth0ReplayState();
                tags["sidecar_replay_state"] =
                    correction_replay_boundary
                        ? "reset_after_spec_publication"
                        : "reset";
            }
            else
            {
                tags["sidecar_replay_state"] =
                    reason == LivePrefixMutationReason::PrefixRestore
                        ? "preserved_for_prefix_restore"
                        : "preserved_for_spec_publication";
                preserves_live_graph_replay = true;
            }
            tags["replay_state"] =
                preserves_live_graph_replay
                    ? "typed_reset_with_replay_safe_graphs_preserved"
                    : "reset";
        }
        else
        {
            tags["replay_state"] = "not_initialized";
            tags["sidecar_replay_state"] = "not_initialized";
        }
        /*
         * KernelFactory::resetAllDynamicState() clears per-kernel dynamic
         * device-param validity and stream bindings.  That is correct for a
         * hard graph reset, but unsafe when this boundary deliberately keeps
         * graph executables alive: CUDA/HIP graphs capture kernel argument
         * pointer identity, while updateDynamicParams() only restamps the
         * already-captured dynamic buffers before replay.  Preserve kernel
         * dynamic objects for accepted/rejected MTP publication and prefix
         * restore boundaries, then rely on explicit stream rebinding plus
         * per-stage dynamic updates.
         */
        if (preserves_live_graph_replay)
        {
            tags["kernel_dynamic_state"] =
                "preserved_for_live_state_graph_replay";
        }
        else
        {
            resetKernelDynamicState();
            tags["kernel_dynamic_state"] = "reset";
        }
        // Ordered lifecycle evidence keeps epochs out of retained map keys.
        PerfStatsCollector::addCounterWithSequence("mtp",
                                       "live_prefix_replay_state_after_mutation",
                                       1.0,
                                       {mutation.previous_epoch, mutation.live_state_epoch,
                                        live_state_mutation_count_,
                                        isPrefixCacheMoEModel() ? moePlacementEpoch() : 0,
                                        isPrefixCacheMoEModel() ? moeRuntimeMovementEpoch() : 0},
                                       "decode",
                                       state_.device_id.toString(),
                                       std::move(tags));
    }

    PrefixCacheFingerprintResult DeviceGraphOrchestrator::buildCurrentPrefixFingerprint(
        const PrefixCacheRuntimeConfig &prefix_config) const
    {
        // Background publication is free to advance after this observation.
        // Bind the hash and its admission metadata to this one value; harvest
        // will independently detect a newer identity and discard stale state.
        const uint64_t placement_epoch = moeRuntimeMovementEpoch();
        const auto &config = graph_builder_->config();
        PrefixFingerprintMaterial material;
        material.model = {
            {"architecture", graph_builder_->architectureName()},
            {"n_layers", std::to_string(config.n_layers)},
            {"d_model", std::to_string(config.d_model)},
            {"n_heads", std::to_string(config.n_heads)},
            {"n_kv_heads", std::to_string(config.n_kv_heads)},
            {"vocab_size", std::to_string(config.vocab_size)},
        };
        material.runtime = {
            {"activation_precision", activationPrecisionToString(config.activation_precision)},
            {"kv_cache_precision", kvCachePrecisionToString(config.kv_cache_precision)},
            {"k_precision", activationPrecisionToString(prefix_layout_.k_precision)},
            {"v_precision", activationPrecisionToString(prefix_layout_.v_precision)},
            {"kv_layout", std::to_string(static_cast<int>(prefix_layout_.kv_layout))},
            {"block_size", std::to_string(prefix_layout_.block_size)},
            {"terminal_state", prefixCacheTerminalStateModeToString(prefix_config.terminal_state)},
        };
        appendProjectionArithmeticFingerprintFields(
            material.runtime, debugEnv().gemm.gdn_projection_arithmetic,
            resolveMoERouterArithmetic(debugEnv().gemm.moe_router_arithmetic, graph_builder_->architectureName()),
            debugEnv().gemm.cuda_moe_router_prefill);
        if (config.flash_next)
            appendQwen4ExpArithmeticFingerprintFields(material.runtime, config.qfn_hc_up_activation, config.qfn_gdn_output_activation, config.qfn_fa_output_activation, config.qfn_gdn_qkv_activation, config.attention_phase_schedule);
        material.topology = {
            {"first_layer_index", std::to_string(prefix_layout_.first_layer_index)},
            {"total_layers", std::to_string(prefix_layout_.total_layers)},
            {"head_dim", std::to_string(prefix_layout_.head_dim)},
        };
        if (config.tp_config && config.tp_config->worldSize() > 1)
        {
            const int tp_participant = config.local_rank;
            if (tp_participant < 0 ||
                tp_participant >= config.tp_config->worldSize())
            {
                PrefixCacheFingerprintResult invalid;
                invalid.bypass = true;
                invalid.bypass_reason =
                    "tensor-parallel prefix participant rank is outside its domain";
                return invalid;
            }
            const auto &assignment =
                config.tp_config->forRank(tp_participant);
            material.topology.push_back({"scope", "tp-domain"});
            material.topology.push_back({"tp_world_size", std::to_string(config.tp_config->worldSize())});
            /*
             * Participants share the logical prompt and coordination domain,
             * but own different KV and terminal-logit payload slices. The
             * participant identity must therefore be part of the durable key:
             * otherwise one shard overwrites another in the shared archive.
             * Physical device identity is intentionally excluded so the same
             * logical shard remains cross-restorable after a device remap.
             */
            material.topology.push_back({"tp_participant", std::to_string(tp_participant)});
            material.topology.push_back({"tp_total_heads", std::to_string(config.tp_config->totalHeads())});
            material.topology.push_back({"tp_total_kv_heads", std::to_string(config.tp_config->totalKVHeads())});
            material.topology.push_back({"tp_total_d_ff", std::to_string(config.tp_config->totalDFF())});
            material.topology.push_back({"tp_total_vocab", std::to_string(config.tp_config->totalVocab())});
            material.topology.push_back({"tp_proportional", config.tp_config->isProportional() ? "true" : "false"});
            material.topology.push_back({"local_kv_heads", std::to_string(prefix_layout_.local_kv_heads)});
            material.topology.push_back({"kv_head_start", std::to_string(prefix_layout_.kv_head_start)});
            material.topology.push_back({"vocab_start", std::to_string(assignment.vocab_start)});
            material.topology.push_back({"vocab_count", std::to_string(assignment.vocab_count)});
            material.topology.push_back({"qkv_column_parallel", config.qkv_column_parallel ? "true" : "false"});
            material.topology.push_back({"lm_head_column_parallel", config.lm_head_column_parallel ? "true" : "false"});
        }
        else
        {
            material.topology.push_back({"scope", "single-participant"});
            material.topology.push_back({"device", state_.device_id.toString()});
            material.topology.push_back({"local_kv_heads", std::to_string(prefix_layout_.local_kv_heads)});
            material.topology.push_back({"kv_head_start", std::to_string(prefix_layout_.kv_head_start)});
        }
        material.hybrid = prefix_layout_.includes_hybrid_state
                              ? std::vector<PrefixFingerprintField>{
                                    {"enabled", "true"},
                                    {"gdn_layers", std::to_string(prefix_layout_.gdn_layers)},
                                    {"hybrid_host_state_bytes", std::to_string(prefix_layout_.hybrid_host_state_bytes)},
                                    {"hybrid_device_state_bytes", std::to_string(prefix_layout_.hybrid_device_state_bytes)},
                                    {"hybrid_state_bytes", std::to_string(prefix_layout_.hybrid_state_bytes)},
                                }
                              : std::vector<PrefixFingerprintField>{{"enabled", "false"}};
        if (prefix_layout_.includes_hybrid_state)
        {
            const auto *hybrid = dynamic_cast<const IHybridKVCache *>(state_.kv_cache.get());
            if (!hybrid)
                throw std::logic_error("Hybrid prefix identity requires canonical cache metadata");
            const auto metadata = hybrid->hybridPrefixStateMetadata();
            appendGDNStateLayoutFingerprintFields(
                material.hybrid,
                metadata.host_bytes > 0 ? std::optional{metadata.host_recurrence_layout}
                                        : std::nullopt,
                metadata.device_bytes > 0 ? std::optional{metadata.device_recurrence_layout}
                                          : std::nullopt);
            if (metadata.host_bytes > 0)
                appendCPUHostPrefixEncodingFingerprintFields(material.hybrid, metadata.host_encoding);
        }
        material.mtp = config.mtp.enabled
                           ? std::vector<PrefixFingerprintField>{
                                 {"enabled", "true"},
                                 {"draft_tokens", std::to_string(config.mtp.draft_tokens)},
                                 {"verify_mode", mtpVerifyModeToString(config.mtp.verify_mode)},
                                 {"require_terminal_hidden_for_full_hit",
                                  config.mtp.require_terminal_hidden_for_full_hit ? "true" : "false"},
                                 {"terminal_hidden_bytes", std::to_string(prefix_layout_.terminal_hidden_bytes)},
                             }
                           : std::vector<PrefixFingerprintField>{{"enabled", "false"}};

        const bool model_is_moe = isPrefixCacheMoEModel();
        if (model_is_moe)
        {
            const bool invalidate_on_rebalance =
                prefix_config.moe_policy == PrefixCacheMoEPolicy::InvalidateOnRebalance;
            material.moe.push_back({"policy", prefixCacheMoEPolicyToString(prefix_config.moe_policy)});
            material.moe.push_back({"model_is_moe", "true"});
            if (invalidate_on_rebalance)
            {
                material.moe.push_back({"graph_placement_epoch", std::to_string(moePlacementEpoch())});
            }
            if (moe_rebalance_controller_)
            {
                const auto *controller = moe_rebalance_controller_.get();
                material.moe.push_back({"controller.num_layers", std::to_string(controller->numLayers())});
                material.moe.push_back({"controller.num_experts", std::to_string(controller->numExperts())});
                material.moe.push_back({"controller.top_k", std::to_string(controller->topK())});
                if (invalidate_on_rebalance)
                {
                    material.moe.push_back({"controller.placement_epoch", std::to_string(controller->placementEpoch())});
                    material.moe.push_back({"controller.total_rebalances", std::to_string(controller->totalRebalances())});
                }

                const auto &ownership = controller->currentOwnership();
                if (invalidate_on_rebalance)
                {
                    for (int layer_idx = 0;
                         layer_idx < ownership.layerCount();
                         ++layer_idx)
                    {
                        for (int expert_id = 0;
                             expert_id < ownership.expertCount();
                             ++expert_id)
                        {
                            material.moe.push_back({
                                "controller.layer." + std::to_string(layer_idx) +
                                    ".expert." + std::to_string(expert_id) +
                                    ".owner_participant",
                                std::to_string(
                                    ownership.owner(layer_idx, expert_id))});
                        }
                    }
                }

                const auto &replicas = controller->currentReplicas();
                if (invalidate_on_rebalance)
                {
                    material.moe.push_back({"controller.replicas.count", std::to_string(replicas.num_replicated)});
                    for (size_t expert = 0;
                         expert < replicas.replicated_in_any_layer.size();
                         ++expert)
                    {
                        const std::string prefix = "controller.replica." + std::to_string(expert);
                        material.moe.push_back({
                            prefix + ".enabled",
                            replicas.replicated_in_any_layer[expert] ? "1" : "0"});
                    }
                    for (size_t layer = 0; layer < replicas.replica_participants_by_layer.size(); ++layer)
                    {
                        const auto &layer_replicas = replicas.replica_participants_by_layer[layer];
                        for (size_t expert = 0; expert < layer_replicas.size(); ++expert)
                        {
                            const auto &participants = layer_replicas[expert];
                            for (size_t participant = 0; participant < participants.size(); ++participant)
                            {
                                if (!participants[participant])
                                    continue;
                                material.moe.push_back({"controller.replica.layer." + std::to_string(layer) +
                                                            ".expert." + std::to_string(expert) +
                                                            ".participant." + std::to_string(participant),
                                                        "1"});
                            }
                        }
                    }
                }
            }
        }

        if (graph_builder_)
            graph_builder_->appendPrefixCacheFingerprintMaterial(material);

        return buildPrefixCacheFingerprint(
            std::move(material),
            model_is_moe,
            prefix_config.moe_policy,
            placement_epoch);
    }

    bool DeviceGraphOrchestrator::ensurePrefixArchiveDeviceStaging()
    {
        if (!state_.device_id.is_gpu())
        {
            prefix_archive_device_staging_.reset();
            return true;
        }
        if (!arena_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Prefix archive staging requires an initialized BufferArena");
            return false;
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Prefix archive staging cannot resolve backend for "
                      << state_.device_id.toString());
            return false;
        }
        if (!prefix_archive_device_staging_ ||
            prefix_archive_device_staging_->device != state_.device_id)
        {
            prefix_archive_device_staging_ =
                std::make_unique<PrefixArchiveDeviceStaging>(
                    backend,
                    state_.device_id);
        }

        struct Requirement
        {
            BufferId id;
            size_t bytes;
        };
        const std::array<Requirement, 5> requirements = {{
            {BufferId::PREFIX_K_STAGING, prefix_layout_.bytes_per_fa_layer_k},
            {BufferId::PREFIX_V_STAGING, prefix_layout_.bytes_per_fa_layer_v},
            {BufferId::PREFIX_HYBRID_STATE_STAGING, prefix_layout_.hybrid_device_state_bytes},
            {BufferId::PREFIX_MTP_K_STAGING, prefix_layout_.bytes_per_mtp_layer_k},
            {BufferId::PREFIX_MTP_V_STAGING, prefix_layout_.bytes_per_mtp_layer_v},
        }};

        for (const Requirement &requirement : requirements)
        {
            if (requirement.bytes == 0)
                continue;

            auto *slot = prefix_archive_device_staging_->slot(requirement.id);
            if (!slot ||
                !prefix_archive_device_staging_->ensure(*slot, requirement.bytes) ||
                !slot->view)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to allocate "
                          << bufferIdName(requirement.id)
                          << " capacity=" << requirement.bytes);
                return false;
            }

            const bool registered = arena_->isRegistered(requirement.id);
            const bool bound =
                registered
                    ? arena_->bindExternalBuffer(requirement.id, slot->view.get())
                    : arena_->registerExternalBuffer(requirement.id, slot->view.get());
            if (!bound)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to "
                          << (registered ? "bind " : "register ")
                          << bufferIdName(requirement.id)
                          << " with BufferArena");
                return false;
            }
        }

        PerfStatsCollector::addCounter(
            "prefix_cache",
            "device_archive_staging_ready",
            1.0,
            "initialization",
            state_.device_id.toString(),
            {
                {"k_bytes", std::to_string(prefix_layout_.bytes_per_fa_layer_k)},
                {"v_bytes", std::to_string(prefix_layout_.bytes_per_fa_layer_v)},
                {"mtp_k_bytes", std::to_string(prefix_layout_.bytes_per_mtp_layer_k)},
                {"mtp_v_bytes", std::to_string(prefix_layout_.bytes_per_mtp_layer_v)},
                {"hybrid_bytes", std::to_string(prefix_layout_.hybrid_device_state_bytes)},
            });
        return true;
    }

    bool DeviceGraphOrchestrator::preparePrefixArchivePayloadReady(
        PrefixBlockHandle *handle,
        void *producer_stream,
        const char *producer_name) const
    {
        if (!state_.device_id.is_gpu())
            return true;
        if (!handle || !handle->payload_readiness || !producer_stream)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Prefix archive readiness requires a GPU RAM handle and explicit stream");
            return false;
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
            return false;
        const int ordinal = state_.device_id.gpu_ordinal();
        void *raw_event = backend->createEvent(ordinal);
        if (!raw_event)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to create prefix archive readiness event");
            return false;
        }
        std::shared_ptr<void> event(
            raw_event,
            [backend, ordinal](void *value)
            {
                if (value)
                    backend->destroyEvent(value, ordinal);
            });

        if (!handle->payload_readiness->prepare(
                std::move(event),
                state_.device_id,
                producer_stream, *backend))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to prepare prefix archive readiness event"
                      << " producer="
                      << (producer_name && producer_name[0] != '\0'
                              ? producer_name
                              : "unknown"));
            return false;
        }
        return true;
    }

    bool DeviceGraphOrchestrator::recordPrefixArchivePayloadReady(
        PrefixBlockHandle *handle,
        void *producer_stream,
        const char *producer_name) const
    {
        if (!state_.device_id.is_gpu())
            return true;
        if (!handle ||
            !handle->payload_readiness ||
            !handle->payload_readiness->prepared() ||
            !handle->payload_readiness->event() ||
            handle->payload_readiness->producerDevice() != state_.device_id ||
            handle->payload_readiness->producerStream() != producer_stream)
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Prefix archive completion reached "
                "commit without its exact preflighted event and stream");
            std::terminate();
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Prefix archive completion lost its "
                "GPU backend after asynchronous copies were submitted");
            std::terminate();
        }
        const int ordinal = state_.device_id.gpu_ordinal();
        if (!backend->recordEvent(
                handle->payload_readiness->event(),
                ordinal,
                producer_stream))
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Failed to record the preflighted "
                "prefix archive readiness event after asynchronous copies");
            std::terminate();
        }
        if (!handle->payload_readiness->publishRecorded())
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Failed to publish the recorded "
                "prefix archive readiness event");
            std::terminate();
        }

        PerfStatsCollector::addCounter(
            "prefix_cache",
            "archive_ready_events",
            1.0,
            "harvest",
            state_.device_id.toString(),
            {{"producer", producer_name && producer_name[0] != '\0'
                              ? producer_name
                              : "unknown"}});
        return true;
    }
} // namespace llaminar2
