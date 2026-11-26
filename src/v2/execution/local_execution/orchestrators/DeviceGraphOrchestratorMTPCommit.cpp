/**
 * @file DeviceGraphOrchestratorMTPCommit.cpp
 * @brief DeviceGraphOrchestrator method shard (32478..37247).
 */
#include "DeviceGraphOrchestratorInternal.h"

namespace llaminar2
{

    const char *DeviceGraphOrchestrator::deviceResidentLogicalStateDiagnosticPhaseName(
        DeviceResidentLogicalStateDiagnosticPhase phase)
    {
        switch (phase)
        {
        case DeviceResidentLogicalStateDiagnosticPhase::PrimaryDerive:
            return "primary_derive";
        case DeviceResidentLogicalStateDiagnosticPhase::MailboxPublished:
            return "mailbox_published";
        case DeviceResidentLogicalStateDiagnosticPhase::MaintenanceEntry:
            return "maintenance_entry";
        case DeviceResidentLogicalStateDiagnosticPhase::MaintenanceExit:
            return "maintenance_exit";
        case DeviceResidentLogicalStateDiagnosticPhase::SidecarEntry:
            return "sidecar_entry";
        case DeviceResidentLogicalStateDiagnosticPhase::SidecarExit:
            return "sidecar_exit";
        case DeviceResidentLogicalStateDiagnosticPhase::ObservationMailboxReady:
            return "observation_mailbox_ready";
        case DeviceResidentLogicalStateDiagnosticPhase::ObservationPrefixCheckpointJoined:
            return "observation_prefix_checkpoint_joined";
        case DeviceResidentLogicalStateDiagnosticPhase::ObservationPublishedHandoffsJoined:
            return "observation_published_handoffs_joined";
        case DeviceResidentLogicalStateDiagnosticPhase::ObservationGraphProducersJoined:
            return "observation_graph_producers_joined";
        case DeviceResidentLogicalStateDiagnosticPhase::ObservationMTPTransactionJoined:
            return "observation_mtp_transaction_joined";
        case DeviceResidentLogicalStateDiagnosticPhase::Count:
            break;
        }
        return "unknown";
    }

    const char *DeviceGraphOrchestrator::deviceResidentLogicalStateDiagnosticFieldName(
        DeviceResidentLogicalStateDiagnosticField field)
    {
        switch (field)
        {
        case DeviceResidentLogicalStateDiagnosticField::TargetCachedTokens:
            return "target_cached_tokens";
        case DeviceResidentLogicalStateDiagnosticField::AcceptedStateCounts:
            return "accepted_state_counts";
        case DeviceResidentLogicalStateDiagnosticField::NextConditionTokens:
            return "next_condition_tokens";
        case DeviceResidentLogicalStateDiagnosticField::AllDraftsAcceptedFlags:
            return "all_drafts_accepted_flags";
        case DeviceResidentLogicalStateDiagnosticField::StoppedFlags:
            return "stopped_flags";
        case DeviceResidentLogicalStateDiagnosticField::PublicationOkFlags:
            return "publication_ok_flags";
        case DeviceResidentLogicalStateDiagnosticField::Count:
            break;
        }
        return "unknown";
    }

    void DeviceGraphOrchestrator::snapshotDeviceResidentLogicalStatePhase(
        DeviceResidentLogicalStateDiagnosticPhase phase,
        void *producer_stream,
        int request_count,
        uint64_t publication_generation) const
    {
        if (!debugEnv().runtime_debug.mtp_device_phase_snapshots)
            return;

        const size_t phase_index = static_cast<size_t>(phase);
        if (!state_.device_id.is_gpu() ||
            !producer_stream ||
            phase_index >=
                device_resident_logical_state_diagnostic_records_.size() ||
            !device_resident_logical_sequence_state_storage_.validFor(
                request_count) ||
            !device_resident_logical_state_diagnostic_storage_.validFor(
                request_count))
        {
            throw std::runtime_error(
                std::string("Device-only MTP logical-state snapshot has an invalid boundary: phase=") +
                deviceResidentLogicalStateDiagnosticPhaseName(phase) +
                " device=" + state_.device_id.toString() +
                " requests=" + std::to_string(request_count));
        }

        IBackend *backend = getBackendFor(state_.device_id);
        auto &record =
            device_resident_logical_state_diagnostic_records_[phase_index];
        if (!backend || !record.ready_event)
        {
            throw std::runtime_error(
                std::string("Device-only MTP logical-state snapshot is missing its backend/event: phase=") +
                deviceResidentLogicalStateDiagnosticPhaseName(phase));
        }

        if (record.snapshot_count >=
            DeviceResidentLogicalStateDiagnosticStorage::kSlotsPerPhase)
        {
            throw std::runtime_error(
                std::string("Device-only MTP logical-state snapshot exhausted its non-reusing phase slots: phase=") +
                deviceResidentLogicalStateDiagnosticPhaseName(phase) +
                " snapshots=" + std::to_string(record.snapshot_count));
        }
        const size_t snapshot_slot = record.snapshot_count;

        const auto &source =
            device_resident_logical_sequence_state_storage_;
        const std::array<const int32_t *,
                         DeviceResidentLogicalStateDiagnosticStorage::kFieldCount>
            source_rows = {
                source.target_cached_tokens_device,
                source.accepted_state_counts_device,
                source.next_condition_tokens_device,
                source.all_drafts_accepted_flags_device,
                source.stopped_flags_device,
                source.publication_ok_flags_device,
            };
        const size_t row_bytes =
            sizeof(int32_t) * static_cast<size_t>(request_count);
        for (size_t field_index = 0;
             field_index < source_rows.size();
             ++field_index)
        {
            const auto field =
                static_cast<DeviceResidentLogicalStateDiagnosticField>(
                    field_index);
            int32_t *destination =
                device_resident_logical_state_diagnostic_storage_.row(
                    phase,
                    snapshot_slot,
                    field);
            if (!source_rows[field_index] ||
                !destination ||
                !backend->deviceCopyAsync(
                    destination,
                    source_rows[field_index],
                    row_bytes,
                    state_.device_id.gpu_ordinal(),
                    producer_stream))
            {
                throw std::runtime_error(
                    std::string("Device-only MTP logical-state snapshot D2D copy failed: phase=") +
                    deviceResidentLogicalStateDiagnosticPhaseName(phase) +
                    " field=" +
                    deviceResidentLogicalStateDiagnosticFieldName(field));
            }
        }

        if (!backend->recordEvent(
                record.ready_event.get(),
                state_.device_id.gpu_ordinal(),
                producer_stream))
        {
            throw std::runtime_error(
                std::string("Device-only MTP logical-state snapshot event publication failed: phase=") +
                deviceResidentLogicalStateDiagnosticPhaseName(phase) +
                ". The phase boundary must be outside graph capture.");
        }

        record.producer_stream = producer_stream;
        record.request_count = request_count;
        record.live_state_epoch = live_replay_state_epoch_;
        record.publication_generation = publication_generation;
        record.slot = snapshot_slot;
        ++record.snapshot_count;
        record.valid = true;
    }

    std::string DeviceGraphOrchestrator::materializeDeviceResidentLogicalStatePhaseDiagnostics(
        void *observation_stream) const
    {
        if (!debugEnv().runtime_debug.mtp_device_phase_snapshots)
            return "phase_snapshots=disabled";
        if (!state_.device_id.is_gpu() ||
            !observation_stream ||
            !device_resident_logical_state_diagnostic_storage_.validFor(1))
        {
            return "phase_snapshots_error=invalid_observation_boundary";
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
            return "phase_snapshots_error=backend_unavailable";

        for (const auto &record :
             device_resident_logical_state_diagnostic_records_)
        {
            if (!record.valid || !record.ready_event)
                continue;
            if (record.producer_stream != observation_stream &&
                !backend->streamWaitEvent(
                    observation_stream,
                    record.ready_event.get(),
                    state_.device_id.gpu_ordinal()))
            {
                return "phase_snapshots_error=event_wait_failed";
            }
        }

        const auto &storage =
            device_resident_logical_state_diagnostic_storage_;
        const size_t element_count =
            DeviceResidentLogicalStateDiagnosticStorage::kRowCount *
            static_cast<size_t>(storage.request_capacity);
        std::vector<int32_t> host_values(element_count, -1);
        if (!backend->deviceToHostFast(
                host_values.data(),
                storage.base_device,
                element_count * sizeof(int32_t),
                state_.device_id.gpu_ordinal(),
                observation_stream))
        {
            return "phase_snapshots_error=d2h_failed";
        }

        std::ostringstream output;
        output << "phase_snapshots=[";
        bool first_phase = true;
        for (size_t phase_index = 0;
             phase_index <
                 device_resident_logical_state_diagnostic_records_.size();
             ++phase_index)
        {
            const auto &record =
                device_resident_logical_state_diagnostic_records_[phase_index];
            if (!record.valid)
                continue;
            if (!first_phase)
                output << ',';
            first_phase = false;

            const auto phase =
                static_cast<DeviceResidentLogicalStateDiagnosticPhase>(
                    phase_index);
            output << "{phase="
                   << deviceResidentLogicalStateDiagnosticPhaseName(phase)
                   << ",epoch=" << record.live_state_epoch
                   << ",generation=" << record.publication_generation
                   << ",requests=" << record.request_count;
            for (size_t field_index = 0;
                 field_index <
                     DeviceResidentLogicalStateDiagnosticStorage::kFieldCount;
                 ++field_index)
            {
                const auto field =
                    static_cast<DeviceResidentLogicalStateDiagnosticField>(
                        field_index);
                const size_t row_index =
                    (phase_index *
                         DeviceResidentLogicalStateDiagnosticStorage::kSlotsPerPhase +
                     record.slot) *
                        DeviceResidentLogicalStateDiagnosticStorage::kFieldCount +
                    field_index;
                const size_t row_offset =
                    row_index *
                    static_cast<size_t>(storage.request_capacity);
                output << ','
                       << deviceResidentLogicalStateDiagnosticFieldName(field)
                       << "=[";
                const int shown_requests =
                    std::min(record.request_count, storage.request_capacity);
                for (int request = 0;
                     request < shown_requests;
                     ++request)
                {
                    if (request != 0)
                        output << ',';
                    output << host_values[
                        row_offset + static_cast<size_t>(request)];
                }
                output << ']';
            }
            output << '}';
        }
        output << ']';
        return output.str();
    }

    DeviceGraphOrchestrator::DeviceResidentLogicalStateReadScope::
        DeviceResidentLogicalStateReadScope(
            DeviceGraphOrchestrator &owner,
            void *stream,
            const char *consumer_name)
        : owner_(&owner),
          stream_(stream),
          consumer_name_(consumer_name)
    {
        if (!owner.state_.device_id.is_gpu())
        {
            ready_ = true;
            return;
        }

        ready_ = owner.admitDeviceResidentLogicalSequenceStateRead(
            stream_,
            consumer_name_,
            &publication_generation_,
            &admission_lock_);
        active_ = ready_ && publication_generation_ > 0 &&
                  admission_lock_.owns_lock();
    }

    DeviceGraphOrchestrator::DeviceResidentLogicalStateReadScope::
        ~DeviceResidentLogicalStateReadScope() noexcept
    {
        if (!active_)
            return;
        if (complete())
            return;

        LOG_ERROR(
            "[DeviceGraphOrchestrator] Fatal resident logical-state reader "
            "completion failure for consumer="
            << (consumer_name_ && consumer_name_[0] != '\0'
                    ? consumer_name_
                    : "unknown"));
        std::terminate();
    }

    bool DeviceGraphOrchestrator::DeviceResidentLogicalStateReadScope::
        complete() noexcept
    {
        if (!active_)
            return ready_;
        if (!owner_ ||
            !owner_->recordDeviceResidentLogicalSequenceStateReadCompletion(
                stream_,
                consumer_name_,
                publication_generation_))
        {
            return false;
        }
        active_ = false;
        if (admission_lock_.owns_lock())
            admission_lock_.unlock();
        return true;
    }

    /**
     * @brief Publish one logical-state writer to both tensor and mailbox readers.
     * @param request_count Number of initialized request rows in the arena bank.
     * @param producer_stream Exact stream that enqueued the current writer.
     * @param publication_kind Typed provenance of the new logical-state values.
     * @param error Receives a diagnostic when admission or publication fails.
     * @return Whether the event-backed mailbox now names the published arena rows.
     */
    bool DeviceGraphOrchestrator::recordDeviceResidentLogicalSequenceStateMailbox(
        int request_count,
        void *producer_stream,
        DeviceResidentLogicalStatePublicationKind publication_kind,
        std::string *error)
    {
        if (request_count <= 0)
        {
            if (error)
                *error = "device logical-state mailbox requires a positive request count";
            return false;
        }
        if (!producer_stream)
        {
            if (error)
                *error = "device logical-state mailbox requires an explicit producer stream";
            return false;
        }
        if (!device_resident_logical_sequence_state_storage_.validFor(
                request_count))
        {
            if (error)
            {
                *error =
                    "device logical-state mailbox requires complete "
                    "arena-owned publication storage for the request batch";
            }
            return false;
        }

        if (!state_.device_id.is_gpu())
        {
            if (error)
                *error = "device logical-state mailbox requires a GPU device";
            return false;
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
        {
            if (error)
                *error = "device logical-state mailbox could not resolve backend";
            return false;
        }

        std::unique_lock<std::shared_mutex> admission_lock(
            device_resident_logical_state_admission_sync_->admission_mutex);
        if (!pending_device_resident_logical_state_writer_.valid() ||
            pending_device_resident_logical_state_writer_.stream !=
                producer_stream)
        {
            if (error)
            {
                *error =
                    "device logical-state mailbox publication has no matching "
                    "exclusive writer transaction";
            }
            return false;
        }

        if (!device_resident_logical_sequence_state_ready_event_)
        {
            if (error)
            {
                *error =
                    "device logical-state mailbox readiness event was not "
                    "preallocated";
            }
            return false;
        }
        /*
         * The mailbox is a typed view, not a second coherence authority. Its
         * raw-pointer readers already join the mailbox event, but captured
         * pipeline metadata reads the original arena tensor through
         * TransferEngine. Publish that same writer there before making the
         * mailbox visible. In particular, a first one-token request has no
         * preceding verifier output that could have published this bank.
         * Both events describe this exact producer stream; neither downloads
         * state nor waits for the GPU on the host.
         */
        TransferEngine::publishDeviceWrite(
            arena_->getTensor(BufferId::MTP_LOGICAL_SEQUENCE_STATE),
            state_.device_id,
            producer_stream);
        if (!backend->recordEvent(
                device_resident_logical_sequence_state_ready_event_.get(),
                state_.device_id.gpu_ordinal(),
                producer_stream))
        {
            if (error)
                *error = "device logical-state mailbox could not record readiness event";
            return false;
        }
        DeviceResidentLogicalSequenceStateMailbox mailbox;
        mailbox.request_count = request_count;
        mailbox.target_positions_device =
            device_resident_logical_sequence_state_storage_
                .target_cached_tokens_device;
        mailbox.target_sequence_lengths_device =
            device_resident_logical_sequence_state_storage_
                .target_cached_tokens_device;
        mailbox.accepted_state_counts_device =
            device_resident_logical_sequence_state_storage_
                .accepted_state_counts_device;
        mailbox.next_condition_tokens_device =
            device_resident_logical_sequence_state_storage_
                .next_condition_tokens_device;
        mailbox.all_drafts_accepted_flags_device =
            device_resident_logical_sequence_state_storage_
                .all_drafts_accepted_flags_device;
        mailbox.stopped_flags_device =
            device_resident_logical_sequence_state_storage_
                .stopped_flags_device;
        mailbox.publication_ok_flags_device =
            device_resident_logical_sequence_state_storage_
                .publication_ok_flags_device;
        mailbox.producer_stream = producer_stream;
        mailbox.ready_event =
            device_resident_logical_sequence_state_ready_event_;
        mailbox.live_state_epoch = live_replay_state_epoch_;
        mailbox.publication_generation =
            ++device_resident_logical_state_publication_generation_;
        mailbox.publication_kind = publication_kind;
        if (!mailbox.valid())
        {
            if (error)
                *error = "device logical-state mailbox is incomplete";
            return false;
        }

        device_resident_logical_state_access_epoch_.begin(
            producer_stream,
            device_resident_logical_sequence_state_ready_event_,
            mailbox.publication_generation);

        if (!device_resident_logical_sequence_state_storage_.markPublished(
                request_count,
                live_replay_state_epoch_))
        {
            if (error)
            {
                *error =
                    "device logical-state arena publication marker rejected "
                    "the request batch";
            }
            return false;
        }

        device_resident_logical_sequence_state_mailbox_ = mailbox;
        pending_device_resident_logical_state_writer_.clear();
        snapshotDeviceResidentLogicalStatePhase(
            DeviceResidentLogicalStateDiagnosticPhase::MailboxPublished,
            producer_stream,
            request_count,
            mailbox.publication_generation);
        PerfStatsCollector::addCounterWithSequence(
            "mtp",
            "device_resident_logical_state_mailboxes",
            1.0,
            {static_cast<uint64_t>(mailbox.publication_generation)},
            "decode",
            state_.device_id.toString(),
            {{"requests", std::to_string(request_count)},
             {"publication_kind",
              deviceResidentLogicalStatePublicationKindName(publication_kind)},
             {"storage_owner", "arena_persistent"}});
        PerfStatsCollector::addCounter(
            "mtp",
            "device_resident_logical_state_arena_publications",
            1.0,
            "decode",
            state_.device_id.toString(),
            {{"requests", std::to_string(request_count)}});
        return true;
    }

    bool DeviceGraphOrchestrator::admitDeviceResidentLogicalSequenceStateRead(
        void *consumer_stream,
        const char *consumer_name,
        uint64_t *publication_generation,
        std::shared_lock<std::shared_mutex> *admission_lock)
    {
        if (!publication_generation || !admission_lock)
            return false;
        *publication_generation = 0;
        *admission_lock = {};
        if (!state_.device_id.is_gpu())
            return true;

        std::shared_lock<std::shared_mutex> read_admission(
            device_resident_logical_state_admission_sync_->admission_mutex);
        if (pending_device_resident_logical_state_writer_.valid())
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Resident logical-state reader was "
                "admitted while replacement publication is in flight"
                << " consumer="
                << (consumer_name && consumer_name[0] != '\0'
                        ? consumer_name
                        : "unknown")
                << " writer="
                << (pending_device_resident_logical_state_writer_.writer_name &&
                            pending_device_resident_logical_state_writer_
                                    .writer_name[0] != '\0'
                        ? pending_device_resident_logical_state_writer_.writer_name
                        : "unknown"));
            return false;
        }

        auto &mailbox = device_resident_logical_sequence_state_mailbox_;
        if (!mailbox.valid())
            return true;
        if (mailbox.live_state_epoch != live_replay_state_epoch_)
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Resident logical-state reader found "
                "a stale publication epoch"
                << " mailbox_epoch=" << mailbox.live_state_epoch
                << " live_state_epoch=" << live_replay_state_epoch_
                << " consumer="
                << (consumer_name && consumer_name[0] != '\0'
                        ? consumer_name
                        : "unknown"));
            return false;
        }
        if (!consumer_stream)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Resident logical-state mailbox consumer requires an explicit stream");
            return false;
        }
        const auto &access_epoch =
            device_resident_logical_state_access_epoch_;
        if (!access_epoch.valid() ||
            access_epoch.publication_generation !=
                mailbox.publication_generation)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Resident logical-state mailbox is valid without a matching access epoch");
            return false;
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
            return false;

        /*
         * Fan readers out from the immutable publication edge. Waiting on the
         * another reader here would serialize otherwise independent sidecar,
         * verifier, sampler, and maintenance work. Each read scope records its
         * own stream-lane completion only after its GPU work has been enqueued.
         */
        const bool same_stream = mailbox.producer_stream == consumer_stream;
        if (!same_stream &&
            !backend->streamWaitEvent(
                consumer_stream,
                mailbox.ready_event.get(),
                state_.device_id.gpu_ordinal()))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to queue resident logical-state mailbox wait for consumer="
                      << (consumer_name && consumer_name[0] != '\0'
                              ? consumer_name
                              : "unknown"));
            return false;
        }

        PerfStatsCollector::addCounterWithSequence(
            "mtp",
            "device_resident_logical_state_mailbox_waits",
            1.0,
            {static_cast<uint64_t>(mailbox.live_state_epoch)},
            perfPhaseName(),
            state_.device_id.toString(),
            {{"consumer", consumer_name && consumer_name[0] != '\0'
                              ? consumer_name
                              : "unknown"},
             {"same_stream", boolTag(same_stream)}});
        *publication_generation = mailbox.publication_generation;
        *admission_lock = std::move(read_admission);
        return true;
    }

    bool DeviceGraphOrchestrator::publishPipelineFollowerLogicalRowWriteCompletion(
        void *producer_stream)
    {
        if (!state_.device_id.is_gpu() || !producer_stream ||
            !pp_stage_config_ || pp_stage_config_->has_lm_head ||
            device_generation_storage_.active_request_count != 1 ||
            !device_resident_logical_sequence_state_storage_.validFor(1))
            return false;
        auto *backend = getBackendFor(state_.device_id);
        if (!backend || !device_resident_logical_sequence_state_ready_event_)
            return false;
        std::unique_lock<std::shared_mutex> admission_lock(
            device_resident_logical_state_admission_sync_->admission_mutex);
        if (!pending_device_resident_logical_state_writer_.valid() ||
            pending_device_resident_logical_state_writer_.stream != producer_stream)
            return false;

        // The received token is a physical write to the original arena bank.
        // Its other logical fields are not produced by this follower. Publish
        // only the actual writer event; never expose a full outcome mailbox.
        TransferEngine::publishDeviceWrite(
            arena_->getTensor(BufferId::MTP_LOGICAL_SEQUENCE_STATE),
            state_.device_id, producer_stream);
        if (!backend->recordEvent(device_resident_logical_sequence_state_ready_event_.get(),
                state_.device_id.gpu_ordinal(), producer_stream))
            return false;
        device_resident_logical_state_access_epoch_.begin(
            producer_stream, device_resident_logical_sequence_state_ready_event_,
            ++device_resident_logical_state_publication_generation_);
        device_resident_logical_sequence_state_mailbox_.clear();
        device_resident_logical_sequence_state_storage_.retirePublication();
        pending_device_resident_logical_state_writer_.clear();
        PerfStatsCollector::addCounterWithSequence(
            "mtp", "device_resident_logical_state_partial_write_completions", 1.0,
            {device_resident_logical_state_publication_generation_}, perfPhaseName(),
            state_.device_id.toString(), {{"owner", "pipeline_follower"}, {"semantic_mailbox", "absent"}});
        return true;
    }

    bool DeviceGraphOrchestrator::waitForDeviceResidentLogicalSequenceStateRowReuse(
        void *writer_stream,
        const char *writer_name)
    {
        if (!state_.device_id.is_gpu())
            return true;
        if (!writer_stream)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Resident logical-state row reuse requires an explicit writer stream");
            return false;
        }

        std::unique_lock<std::shared_mutex> write_admission(
            device_resident_logical_state_admission_sync_->admission_mutex);
        if (pending_device_resident_logical_state_writer_.valid())
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Resident logical-state replacement "
                "writer overlapped an unfinished writer transaction"
                << " requested_writer="
                << (writer_name && writer_name[0] != '\0'
                        ? writer_name
                        : "unknown")
                << " active_writer="
                << (pending_device_resident_logical_state_writer_.writer_name &&
                            pending_device_resident_logical_state_writer_
                                    .writer_name[0] != '\0'
                        ? pending_device_resident_logical_state_writer_.writer_name
                        : "unknown"));
            return false;
        }

        const auto &access_epoch =
            device_resident_logical_state_access_epoch_;
        if (!access_epoch.valid())
        {
            pending_device_resident_logical_state_writer_.begin(
                writer_stream,
                /*replaced_generation=*/0,
                writer_name);
            device_resident_logical_sequence_state_mailbox_.clear();
            return true;
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
            return false;

        /*
         * The publication root covers the no-reader case. Independent reader
         * completion lanes then form the leaves of the only fan-in in this
         * protocol; readers themselves never wait one another.
         */
        const bool producer_same_stream =
            access_epoch.producer_stream == writer_stream;
        if (!producer_same_stream &&
            !backend->streamWaitEvent(
                writer_stream,
                access_epoch.publication_ready_event.get(),
                state_.device_id.gpu_ordinal()))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to queue resident logical-state publication-root wait for writer="
                      << (writer_name && writer_name[0] != '\0'
                              ? writer_name
                              : "unknown"));
            return false;
        }

        size_t cross_stream_reader_waits = 0;
        for (size_t lane = 0;
             lane < access_epoch.reader_stream_count;
             ++lane)
        {
            const auto &completion =
                access_epoch.reader_completions[lane];
            if (!completion.validFor(
                    access_epoch.publication_generation))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Resident logical-state access epoch contains an invalid reader completion lane="
                          << lane << " writer="
                          << (writer_name && writer_name[0] != '\0'
                                  ? writer_name
                                  : "unknown"));
                return false;
            }
            if (completion.stream == writer_stream)
                continue;

            const auto &completion_event =
                device_resident_logical_state_reader_completion_events_[lane];
            if (!completion_event ||
                !backend->streamWaitEvent(
                    writer_stream,
                    completion_event.get(),
                    state_.device_id.gpu_ordinal()))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to queue resident logical-state reader fan-in lane="
                          << lane << " for writer="
                          << (writer_name && writer_name[0] != '\0'
                                  ? writer_name
                                  : "unknown"));
                return false;
            }
            ++cross_stream_reader_waits;
        }

        PerfStatsCollector::addCounterWithSequence(
            "mtp",
            "device_resident_logical_state_row_reuse_waits",
            1.0,
            {static_cast<uint64_t>(access_epoch.publication_generation), static_cast<uint64_t>(access_epoch.access_sequence)},
            perfPhaseName(),
            state_.device_id.toString(),
            {{"writer", writer_name && writer_name[0] != '\0'
                            ? writer_name
                            : "unknown"},
             {"producer_same_stream", boolTag(producer_same_stream)},
             {"reader_streams",
              std::to_string(access_epoch.reader_stream_count)},
             {"cross_stream_reader_waits",
              std::to_string(cross_stream_reader_waits)}});
        pending_device_resident_logical_state_writer_.begin(
            writer_stream,
            access_epoch.publication_generation,
            writer_name);
        device_resident_logical_sequence_state_mailbox_.clear();
        return true;
    }

    bool DeviceGraphOrchestrator::
        recordDeviceResidentLogicalSequenceStateReadCompletion(
            void *consumer_stream,
            const char *consumer_name,
            uint64_t publication_generation)
    {
        if (!state_.device_id.is_gpu())
            return true;
        if (!consumer_stream || publication_generation == 0)
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Resident logical-state reader completion has invalid publication identity"
                << " consumer="
                << (consumer_name && consumer_name[0] != '\0'
                        ? consumer_name
                        : "unknown")
                << " stream=" << consumer_stream
                << " publication_generation=" << publication_generation
                << " device=" << state_.device_id.toString());
            return false;
        }

        std::lock_guard<std::mutex> completion_lock(
            device_resident_logical_state_admission_sync_
                ->reader_completion_mutex);

        IBackend *backend = getBackendFor(state_.device_id);
        auto &access_epoch =
            device_resident_logical_state_access_epoch_;
        if (!backend || !access_epoch.valid() ||
            access_epoch.publication_generation != publication_generation)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Resident logical-state reader attempted to complete a replaced access epoch"
                      << " consumer="
                      << (consumer_name && consumer_name[0] != '\0'
                              ? consumer_name
                              : "unknown")
                      << " expected_generation=" << publication_generation
                      << " current_access_generation="
                      << access_epoch.publication_generation);
            return false;
        }

        size_t lane = access_epoch.reader_stream_count;
        for (size_t candidate = 0;
             candidate < access_epoch.reader_stream_count;
             ++candidate)
        {
            if (access_epoch.reader_completions[candidate].stream ==
                consumer_stream)
            {
                lane = candidate;
                break;
            }
        }

        if (lane == access_epoch.reader_stream_count)
        {
            if (lane >= access_epoch.reader_completions.size())
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Resident logical-state reader stream capacity exhausted"
                          << " capacity="
                          << access_epoch.reader_completions.size()
                          << " consumer="
                          << (consumer_name && consumer_name[0] != '\0'
                                  ? consumer_name
                                  : "unknown"));
                return false;
            }
            ++access_epoch.reader_stream_count;
        }

        const auto &completion_event =
            device_resident_logical_state_reader_completion_events_[lane];
        if (!completion_event ||
            !backend->recordEvent(
                completion_event.get(),
                state_.device_id.gpu_ordinal(),
                consumer_stream))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to publish resident logical-state reader completion lane="
                      << lane << " consumer="
                      << (consumer_name && consumer_name[0] != '\0'
                              ? consumer_name
                              : "unknown"));
            return false;
        }

        auto &completion = access_epoch.reader_completions[lane];
        completion.stream = consumer_stream;
        completion.publication_generation = publication_generation;
        completion.completion_sequence = ++access_epoch.access_sequence;

        PerfStatsCollector::addCounterWithSequence(
            "mtp",
            "device_resident_logical_state_reader_completions",
            1.0,
            {static_cast<uint64_t>(publication_generation)},
            perfPhaseName(),
            state_.device_id.toString(),
            {{"consumer", consumer_name && consumer_name[0] != '\0'
                              ? consumer_name
                              : "unknown"},
             {"reader_lane", std::to_string(lane)},
             {"reader_streams",
              std::to_string(access_epoch.reader_stream_count)}});
        return true;
    }

    bool DeviceGraphOrchestrator::waitForDeviceResidentLogicalSequenceStateMailboxForObservation(
        void *consumer_stream,
        const char *consumer_name,
        bool *replacement_writer_pending) const
    {
        if (replacement_writer_pending)
            *replacement_writer_pending = false;
        if (!state_.device_id.is_gpu())
            return true;

        std::shared_lock<std::shared_mutex> observation_admission(
            device_resident_logical_state_admission_sync_->admission_mutex);
        if (pending_device_resident_logical_state_writer_.valid())
        {
            if (replacement_writer_pending)
                *replacement_writer_pending = true;
            // This is an expected polling result while a retained generation
            // writer is live, potentially many times per token. Formatting it
            // at DEBUG makes passive telemetry compete with inference progress.
            LOG_TRACE(
                "[DeviceGraphOrchestrator] Resident logical-state observation "
                "deferred behind an unfinished replacement writer"
                << " consumer="
                << (consumer_name && consumer_name[0] != '\0'
                        ? consumer_name
                        : "unknown")
                << " writer="
                << (pending_device_resident_logical_state_writer_.writer_name &&
                            pending_device_resident_logical_state_writer_
                                    .writer_name[0] != '\0'
                        ? pending_device_resident_logical_state_writer_.writer_name
                        : "unknown")
                << " replaced_generation="
                << pending_device_resident_logical_state_writer_
                       .replaced_publication_generation);
            return false;
        }

        const auto &mailbox = device_resident_logical_sequence_state_mailbox_;
        if (!mailbox.valid())
            return true;
        if (mailbox.live_state_epoch != live_replay_state_epoch_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Resident logical-state mailbox observation found stale epoch "
                      << mailbox.live_state_epoch
                      << " while live state is at epoch "
                      << live_replay_state_epoch_);
            return false;
        }
        if (!consumer_stream)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Resident logical-state mailbox observation requires an explicit stream");
            return false;
        }
        const auto &access_epoch =
            device_resident_logical_state_access_epoch_;
        if (!access_epoch.valid() ||
            access_epoch.publication_generation !=
                mailbox.publication_generation)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Resident logical-state mailbox observation is valid without a matching access epoch");
            return false;
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
            return false;

        /*
         * Observation callers synchronously materialize their diagnostic row
         * before returning to request control. They wait the immutable
         * publication root, not unrelated asynchronous readers.
         */
        const bool same_stream =
            access_epoch.producer_stream == consumer_stream;
        if (!same_stream &&
            !backend->streamWaitEvent(
                consumer_stream,
                access_epoch.publication_ready_event.get(),
                state_.device_id.gpu_ordinal()))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to queue resident logical-state mailbox observation wait for consumer="
                      << (consumer_name && consumer_name[0] != '\0'
                              ? consumer_name
                              : "unknown"));
            return false;
        }

        PerfStatsCollector::addCounterWithSequence(
            "mtp",
            "device_resident_logical_state_mailbox_observation_waits",
            1.0,
            {static_cast<uint64_t>(mailbox.live_state_epoch)},
            perfPhaseName(),
            state_.device_id.toString(),
            {{"consumer", consumer_name && consumer_name[0] != '\0'
                              ? consumer_name
                              : "unknown"},
             {"same_stream", boolTag(same_stream)}});
        return true;
    }

    bool DeviceGraphOrchestrator::supportsDeviceResidentMTPSpecStatePublication() const
    {
        /*
         * Direct publication is safe only when both halves of the handoff are
         * present: the cache can update device-visible head/count mirrors from
         * device metadata, and DGO can expose the resulting logical state through
         * a resident mailbox.  GPU consumers keep using those canonical device
         * rows; host-visible probes may observe them explicitly, but no host
         * adoption step is part of the production transaction.
         *
         * Do not require supportsMTPSpecStatePublication() here.  That older
         * capability answers a stronger policy question: whether the runner may
         * select the all-position verifier path directly.  Phase 10's grouped
         * outcome path has a separate row-equivalence proof and should be able
         * to consume the same resident publisher without accidentally promoting
         * the older all-position policy.
         */
        return graph_builder_ &&
               graph_builder_->config().mtp.enabled &&
               supportsDeviceResidentLogicalSequenceStatePublication();
    }

    bool DeviceGraphOrchestrator::supportsLogicalMTPVerifierBaseCheckpoint() const
    {
        /*
         * A token-count-only verifier base can be replayed exactly for KV-only
         * decoder state. Hybrid/GDN models need the recurrent and short-conv
         * payload captured in PrefixStateSnapshot; otherwise replay checkers
         * restore positions while leaving live recurrent state from a later row.
         */
        return graph_builder_ &&
               graph_builder_->config().mtp.enabled &&
               state_.kv_cache != nullptr &&
               !mtpSpecStatePublicationRequiresCapturedStage();
    }

    bool DeviceGraphOrchestrator::supportsDeviceResidentLogicalSequenceStatePublication() const
    {
        /*
         * This is intentionally split from IKVCache device publication.  A cache
         * can update its device head/count mirrors only when DGO also records a
         * resident logical-state mailbox and refuses stale host-position reads.
         * Graph signatures retain only static capture geometry. Dynamic logical
         * positions and sequence lengths are read from the mailbox and cache
         * device rows on every replay, so no host mirror becomes authoritative.
         */
        return state_.device_id.is_gpu() &&
               state_.kv_cache != nullptr &&
               state_.kv_cache->supportsDeviceResidentSequenceStatePublication() &&
               device_resident_logical_sequence_state_storage_.validFor(
                   /*request_count=*/1);
    }

    bool DeviceGraphOrchestrator::materializeMTPDraftTokenPublicationGraph(
        int row,
        int slot,
        const MTPRequestPenaltyPolicy &penalty_policy,
        std::string *error)
    {
        auto fail = [&](const std::string &reason) -> bool
        {
            if (error)
                *error = reason;
            LOG_ERROR("[DeviceGraphOrchestrator] " << reason);
            return false;
        };
        if (error)
            error->clear();

        if (!mtp_graph_owner_plan_.ownsControllerExecutables())
            return fail("Pipeline follower cannot materialize controller-owned MTP helper executables");

        const uint64_t generation = workspaceGeneration(state_.device_id);
        if (!state_.device_id.is_gpu() || generation == 0 || row < 0 ||
            slot < 0 || slot >= mtp_max_draft_depth_ ||
            slot >= stochastic_draft_row_capacity_ ||
            slot >= static_cast<int>(
                        mtp_draft_token_publication_graphs_.size()) ||
            !mtp_draft_token_publication_graphs_[
                static_cast<size_t>(slot)])
        {
            return fail(
                "captured MTP draft publication requires finalized GPU bindings and an in-range fixed-depth slot");
        }

        IBackend *backend = getBackendFor(state_.device_id);
        auto logits_it = state_.extension_buffers.find(mtpSamplingLogitsBufferId());
        TensorBase *logits =
            logits_it == state_.extension_buffers.end()
                ? nullptr
                : logits_it->second.get();
        if (!backend || !logits || !logits->deviceValid() ||
            !logits->gpu_data_ptr())
        {
            return fail(
                "captured MTP draft publication could not bind fresh sidecar logits");
        }
        const auto &shape = logits->shape();
        const size_t rows = shape.size() >= 2 ? shape[0] : 1;
        const size_t cols = shape.size() >= 2 ? shape[1] : shape[0];
        if (static_cast<size_t>(row) >= rows || cols == 0 ||
            cols != static_cast<size_t>(state_.vocab_size))
        {
            return fail(
                "captured MTP draft publication does not match the sidecar logit geometry");
        }
        if (!stochastic_draft_sample_tokens_dev_ ||
            !stochastic_draft_sample_probs_dev_ ||
            !argmax_partial_vals_dev_ || !argmax_partial_idxs_dev_ ||
            argmax_partial_capacity_ <= 0)
        {
            return fail(
                "captured MTP draft publication has incomplete persistent argmax workspace");
        }

        const bool apply_penalties = penalty_policy.enabled();
        const int full_condition_slot =
            mtp_sidecar_capture_layout_.conditionTokenSlot(
                MTPSidecarCaptureRole::Full,
                /*total_rows=*/1);
        const int condition_offset =
            full_condition_slot *
            mtp_sidecar_condition_token_slot_width_;
        const int chained_condition_slot =
            mtp_sidecar_capture_layout_.conditionTokenSlot(
                MTPSidecarCaptureRole::Chained,
                /*total_rows=*/1);
        const int chained_condition_offset =
            chained_condition_slot *
            mtp_sidecar_condition_token_slot_width_;
        if (!mtp_sidecar_condition_token_dev_ ||
            !mtp_sidecar_position_ids_dev_ ||
            full_condition_slot < 0 ||
            condition_offset < 0 ||
            condition_offset >=
                mtp_sidecar_condition_token_capacity_ ||
            chained_condition_offset < 0 ||
            chained_condition_offset >=
                mtp_sidecar_condition_token_capacity_)
        {
            return fail(
                "captured MTP draft publication has no persistent chained-sidecar token/position mailbox");
        }
        if (apply_penalties &&
            (!mtp_sidecar_condition_token_dev_ ||
             condition_offset < 0 ||
             condition_offset >=
                 mtp_sidecar_condition_token_capacity_ ||
             !mtp_generated_token_counts_dev_ ||
             !mtp_greedy_penalty_policy_dev_ ||
             mtp_generated_token_count_capacity_ != state_.vocab_size))
        {
            return fail(
                "captured MTP draft penalties have incomplete resident branch-history bindings");
        }

        MTPDraftTokenPublicationStage::Params stage_params;
        stage_params.device_id = state_.device_id;
        stage_params.backend = backend;
        stage_params.logits_row_device =
            static_cast<float *>(logits->gpu_data_ptr()) +
            static_cast<size_t>(row) * cols;
        stage_params.logits_row = row;
        stage_params.vocab_size = state_.vocab_size;
        stage_params.apply_penalties = apply_penalties;
        stage_params.first_condition_token_device =
            apply_penalties
                ? static_cast<const int32_t *>(
                      mtp_sidecar_condition_token_dev_) +
                      condition_offset
                : nullptr;
        stage_params.prior_draft_tokens_device =
            apply_penalties
                ? static_cast<const int32_t *>(
                      stochastic_draft_sample_tokens_dev_)
                : nullptr;
        stage_params.prior_draft_count = slot;
        stage_params.generated_token_counts_device =
            apply_penalties
                ? static_cast<const int32_t *>(
                      mtp_generated_token_counts_dev_)
                : nullptr;
        stage_params.penalty_policy_device =
            apply_penalties ? mtp_greedy_penalty_policy_dev_ : nullptr;
        stage_params.draft_values_device =
            static_cast<float *>(stochastic_draft_sample_probs_dev_);
        stage_params.draft_tokens_device =
            static_cast<int32_t *>(stochastic_draft_sample_tokens_dev_);
        stage_params.destination_slot = slot;
        stage_params.next_chain_condition_token_device =
            static_cast<int32_t *>(mtp_sidecar_condition_token_dev_) +
            chained_condition_offset;
        stage_params.next_chain_position_id_device =
            static_cast<int32_t *>(mtp_sidecar_position_ids_dev_);
        stage_params.chain_position_increment = 1;
        stage_params.argmax_partial_values_device =
            static_cast<float *>(argmax_partial_vals_dev_);
        stage_params.argmax_partial_indices_device =
            static_cast<int32_t *>(argmax_partial_idxs_dev_);
        stage_params.argmax_partial_capacity = argmax_partial_capacity_;

        const bool retain_first_transaction_draft_diagnostics = mtp_diagnostic_record_mode_ == sampling_math::MTPDiagnosticRecordMode::FirstTransaction;
        if (retain_first_transaction_draft_diagnostics)
        {
            if (!mtp_first_transaction_diagnostic_dev_ ||
                !device_generation_storage_.validFor(1))
            {
                return fail(
                    "captured MTP draft diagnostics require the arena-owned transaction record and device controller");
            }

            stage_params.diagnostic_condition_token_device =
                static_cast<const int32_t *>(
                    mtp_sidecar_condition_token_dev_) +
                (slot == 0 ? condition_offset : chained_condition_offset);
            stage_params.diagnostic_position_id_device =
                static_cast<const int32_t *>(
                    mtp_sidecar_position_ids_dev_);
            stage_params.diagnostic_generation_control_device =
                device_generation_storage_.control_device;
            stage_params.diagnostic_generation_control_stride =
                device_generation_storage_.control_stride;
            stage_params.first_transaction_diagnostic_device =
                mtp_first_transaction_diagnostic_dev_;

            const FrozenModelWeightSet *diagnostic_weight_set =
                selectMTPDecodeWeightSet(
                    "transaction-zero MTP draft diagnostic");
            if (!diagnostic_weight_set)
            {
                return fail(
                    "captured MTP draft diagnostics could not resolve the active decode weight set");
            }
            const ModelWeightBindings diagnostic_bindings =
                makeModelWeightBindings(
                    *diagnostic_weight_set, state_.device_id);
            if (diagnostic_bindings.mtp.empty() ||
                diagnostic_bindings.mtp.depths.empty())
            {
                return fail(
                    "captured MTP draft diagnostics found no active MTP depth binding");
            }
            const auto &diagnostic_depth =
                diagnostic_bindings.mtp.depths.front();
            const bool diagnostic_moe_sidecar =
                diagnostic_depth.fa_block.moe_gate ||
                diagnostic_depth.fa_block.moe_gate_exps ||
                diagnostic_depth.fa_block.moe_up_exps ||
                diagnostic_depth.fa_block.moe_down_exps;

            auto extension = [&](BufferId id) -> TensorBase *
            {
                const auto it = state_.extension_buffers.find(id);
                return it == state_.extension_buffers.end()
                           ? nullptr
                           : it->second.get();
            };
            auto boundary_name =
                [](sampling_math::MTPFirstTransactionDraftBoundary boundary)
                -> const char *
            {
                using Boundary =
                    sampling_math::MTPFirstTransactionDraftBoundary;
                switch (boundary)
                {
                case Boundary::TerminalHiddenInput:
                    return "terminal_hidden_input";
                case Boundary::Embedding:
                    return "embedding";
                case Boundary::NormalizedTerminalHidden:
                    return "normalized_terminal_hidden";
                case Boundary::NormalizedEmbedding:
                    return "normalized_embedding";
                case Boundary::ConcatenatedInput:
                    return "concatenated_input";
                case Boundary::ProjectedInput:
                    return "projected_input";
                case Boundary::AttentionQuery:
                    return "attention_query";
                case Boundary::AttentionKey:
                    return "attention_key";
                case Boundary::AttentionValue:
                    return "attention_value";
                case Boundary::AttentionOutput:
                    return "attention_output";
                case Boundary::AttentionProjection:
                    return "attention_projection";
                case Boundary::MoEExpertIndices:
                    return "moe_expert_indices";
                case Boundary::MoEExpertWeights:
                    return "moe_expert_weights";
                case Boundary::MoERoutedOutput:
                    return "moe_routed_output";
                case Boundary::MoESharedOutput:
                    return "moe_shared_output";
                case Boundary::FFNOutput:
                    return "ffn_output";
                case Boundary::FinalHidden:
                    return "final_hidden";
                case Boundary::ScoredLogits:
                    return "scored_logits";
                case Boundary::Count:
                    break;
                }
                return "invalid";
            };
            auto bind_boundary =
                [&](sampling_math::MTPFirstTransactionDraftBoundary boundary,
                    const TensorBase *tensor,
                    int tensor_row,
                    bool allow_absent = false) -> bool
            {
                const size_t boundary_index =
                    static_cast<size_t>(boundary);
                if (boundary_index >=
                    stage_params.diagnostic_boundary_word_counts.size())
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Transaction-zero MTP draft diagnostic has an invalid boundary id="
                              << boundary_index);
                    return false;
                }
                if (!tensor)
                {
                    if (!allow_absent)
                    {
                        LOG_ERROR("[DeviceGraphOrchestrator] Transaction-zero MTP draft diagnostic is missing required boundary="
                                  << boundary_name(boundary)
                                  << " slot=" << slot);
                        return false;
                    }
                    stage_params.diagnostic_boundary_words_device
                        [boundary_index] = nullptr;
                    stage_params.diagnostic_boundary_word_counts
                        [boundary_index] = 0;
                    return true;
                }
                const size_t tensor_rows = tensor->rows();
                const size_t tensor_words = tensor->numel();
                /*
                 * These are captured producer outputs, so their host-side
                 * coherence bit is deliberately still invalid while the
                 * graph is being materialized.  The diagnostic node consumes
                 * them after their producer in the same captured stream.  A
                 * stable arena pointer and exact row geometry are therefore
                 * the binding contract; requiring deviceValid() here rejects
                 * correctly ordered first-use outputs such as MTP_FFN_OUTPUT.
                 */
                if (!tensor->gpu_data_ptr() || tensor_rows == 0 ||
                    tensor_words == 0 ||
                    tensor_words % tensor_rows != 0 ||
                    tensor->size_bytes() != tensor_words * sizeof(uint32_t) ||
                    tensor_row < 0 ||
                    static_cast<size_t>(tensor_row) >= tensor_rows)
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Transaction-zero MTP draft diagnostic boundary has invalid geometry"
                              << " boundary=" << boundary_name(boundary)
                              << " slot=" << slot
                              << " row=" << tensor_row
                              << " native_type="
                              << static_cast<int>(tensor->native_type())
                              << " rows=" << tensor_rows
                              << " words=" << tensor_words
                              << " bytes=" << tensor->size_bytes()
                              << " expected_bytes="
                              << tensor_words * sizeof(uint32_t)
                              << " device_valid=" << tensor->deviceValid()
                              << " device_ptr=" << tensor->gpu_data_ptr());
                    return false;
                }
                const size_t row_words = tensor_words / tensor_rows;
                if (row_words > static_cast<size_t>(
                                    std::numeric_limits<int>::max()))
                {
                    return false;
                }
                stage_params.diagnostic_boundary_words_device
                    [boundary_index] =
                    static_cast<const uint32_t *>(tensor->gpu_data_ptr()) +
                    static_cast<size_t>(tensor_row) * row_words;
                stage_params.diagnostic_boundary_word_counts
                    [boundary_index] = static_cast<int>(row_words);
                return true;
            };

            using DraftBoundary =
                sampling_math::MTPFirstTransactionDraftBoundary;
            const bool diagnostic_geometry_valid =
                bind_boundary(
                    DraftBoundary::TerminalHiddenInput,
                    slot == 0
                        ? state_.prefix_terminal_hidden.get()
                        : nullptr,
                    /*tensor_row=*/0,
                    /*allow_absent=*/slot != 0) &&
                bind_boundary(
                    DraftBoundary::Embedding,
                    extension(BufferId::MTP_EMBEDDING),
                    row) &&
                bind_boundary(
                    DraftBoundary::NormalizedTerminalHidden,
                    extension(BufferId::MTP_NORM_HIDDEN),
                    row) &&
                bind_boundary(
                    DraftBoundary::NormalizedEmbedding,
                    extension(BufferId::MTP_NORM_EMBEDDING),
                    row) &&
                bind_boundary(
                    DraftBoundary::ConcatenatedInput,
                    extension(BufferId::MTP_CONCAT),
                    row) &&
                bind_boundary(
                    DraftBoundary::ProjectedInput,
                    extension(BufferId::MTP_PROJECTED),
                    row) &&
                bind_boundary(
                    DraftBoundary::AttentionQuery,
                    extension(BufferId::MTP_Q_PROJ),
                    row) &&
                bind_boundary(
                    DraftBoundary::AttentionKey,
                    extension(BufferId::MTP_K_PROJ),
                    row) &&
                bind_boundary(
                    DraftBoundary::AttentionValue,
                    extension(BufferId::MTP_V_PROJ),
                    row) &&
                bind_boundary(
                    DraftBoundary::AttentionOutput,
                    extension(BufferId::MTP_ATTN_OUTPUT),
                    row) &&
                bind_boundary(
                    DraftBoundary::AttentionProjection,
                    extension(BufferId::MTP_ATTN_PROJ),
                    row) &&
                bind_boundary(
                    DraftBoundary::MoEExpertIndices,
                    diagnostic_moe_sidecar
                        ? extension(BufferId::MOE_EXPERT_INDICES)
                        : nullptr,
                    row,
                    /*allow_absent=*/!diagnostic_moe_sidecar) &&
                bind_boundary(
                    DraftBoundary::MoEExpertWeights,
                    diagnostic_moe_sidecar
                        ? extension(BufferId::MOE_EXPERT_WEIGHTS)
                        : nullptr,
                    row,
                    /*allow_absent=*/!diagnostic_moe_sidecar) &&
                bind_boundary(
                    DraftBoundary::MoERoutedOutput,
                    diagnostic_moe_sidecar
                        ? extension(BufferId::MOE_COMBINED_OUTPUT)
                        : nullptr,
                    row,
                    /*allow_absent=*/!diagnostic_moe_sidecar) &&
                bind_boundary(
                    DraftBoundary::MoESharedOutput,
                    diagnostic_moe_sidecar
                        ? extension(BufferId::MOE_SHARED_EXPERT_OUTPUT)
                        : nullptr,
                    row,
                    /*allow_absent=*/!diagnostic_moe_sidecar) &&
                bind_boundary(
                    DraftBoundary::FFNOutput,
                    diagnostic_moe_sidecar
                        ? nullptr
                        : extension(BufferId::MTP_FFN_OUTPUT),
                    row,
                    /*allow_absent=*/diagnostic_moe_sidecar) &&
                bind_boundary(
                    DraftBoundary::FinalHidden,
                    extension(BufferId::MTP_HIDDEN),
                    row) &&
                bind_boundary(
                    DraftBoundary::ScoredLogits,
                    logits,
                    row);
            if (!diagnostic_geometry_valid)
            {
                return fail(
                    "captured MTP draft diagnostics found a non-FP32/INT32 or malformed sidecar boundary");
            }
        }

        auto &cache = *mtp_draft_token_publication_graphs_[
            static_cast<size_t>(slot)];
        if (cache.valid && cache.graph && cache.stage &&
            cache.workspace_generation == generation &&
            cache.stage->hasSameCaptureIdentity(stage_params))
        {
            return true;
        }

        auto stage = std::make_unique<MTPDraftTokenPublicationStage>(
            std::move(stage_params));
        auto *draft_stage = stage.get();
        auto graph = std::make_unique<ComputeGraph>();
        graph->setExecutableMemoryClass(GPUGraphExecutableClass::BoundedFlatHelper);
        graph->addNode(
            "mtp_draft_token_publication_" + std::to_string(slot),
            std::move(stage),
            state_.device_id);

        invalidateSparseQueryParents();
        cache.invalidate();
        cache.graph = std::move(graph);
        cache.stage = draft_stage;
        cache.workspace_generation = generation;
        cache.valid = true;

        PerfStatsCollector::addCounter(
            "mtp",
            "draft_token_publication_graph_materializations",
            1.0,
            "graph_setup",
            state_.device_id.toString(),
            {{"row", std::to_string(row)},
             {"slot", std::to_string(slot)},
             {"penalties", apply_penalties ? "true" : "false"},
             {"workspace_generation", std::to_string(generation)}});
        return true;
    }

    bool DeviceGraphOrchestrator::executeMTPDraftTokenPublicationCaptured(
        void *producer_stream,
        int row,
        int slot,
        std::string *error,
        DeviceGraphExecutor::GraphInitialSubmissionPolicy submission)
    {
        auto fail = [&](const std::string &reason) -> bool
        {
            if (error)
                *error = reason;
            LOG_ERROR("[DeviceGraphOrchestrator] " << reason);
            return false;
        };
        if (slot < 0 ||
            slot >= static_cast<int>(
                        mtp_draft_token_publication_graphs_.size()) ||
            !mtp_draft_token_publication_graphs_[
                static_cast<size_t>(slot)])
        {
            return fail(
                "captured MTP draft publication has no graph owner for its slot");
        }

        auto &cache = *mtp_draft_token_publication_graphs_[
            static_cast<size_t>(slot)];
        const uint64_t generation = workspaceGeneration(state_.device_id);
        if (!state_.device_id.is_gpu() || !producer_stream ||
            !cache.valid || !cache.graph || !cache.stage ||
            cache.workspace_generation == 0 ||
            cache.workspace_generation != generation ||
            cache.stage->getParams().logits_row != row ||
            cache.stage->getParams().destination_slot != slot ||
            cache.stage->getParams().prior_draft_count != slot)
        {
            return fail(
                "captured MTP draft publication requires a current fixed-depth graph and exact sidecar stream");
        }
        if (!debugEnv().execution.gpu_graphs)
        {
            return fail(
                "MTP draft publication requires graph capture; eager execution is forbidden");
        }

        if (submission == DeviceGraphExecutor::GraphInitialSubmissionPolicy::MaterializeWithoutLaunch &&
            cache.segment_cache.initialized)
            return cache.segment_cache.deviceLoopGraphTemplate(*cache.graph, error).has_value();
        IDeviceContext *ctx = getDeviceContext(state_.device_id);
        if (!ctx)
            return fail("captured MTP draft publication has no device context");
        auto &gpu_ctx =
            GPUDeviceContextPool::instance().getContext(state_.device_id);
        if (!cache.segment_cache.ensureCaptureStream(
                &gpu_ctx,
                state_.device_id,
                /*context_from_process_pool=*/true) ||
            !cache.segment_cache.orderCaptureStreamAfter(
                &gpu_ctx,
                producer_stream))
        {
            return fail(
                "captured MTP draft publication could not consume sidecar-logit readiness");
        }

        cache.segment_cache.perf_context =
            "mtp_draft_token_publication";
        cache.segment_cache.replay_workload = {
            .seq_len = 1,
            .batch_size = 1,
            .m = 1,
            .decode = true,
        };
        auto capture_policy = buildDecodeCapturePolicy(
            /*has_collectives=*/false,
            ctx);
        if (!capture_policy.allow_cached_graph_replay)
        {
            return fail(
                "captured MTP draft publication requires mandatory graph replay");
        }
        capture_policy.defer_final_sync = true;
        capture_policy.force_recapture = false;
        capture_policy.graph_replay_plan_policy =
            DeviceGraphExecutor::GraphReplayPlanPolicy::RequireFullGraph;

        cache.graph->reset();
        bool used_graph_replay = false;
        if (!executor_.executeDecodeWithCapturePolicy(
                *cache.graph,
                ctx,
                &cache.segment_cache,
                cache.segment_cache.capture_stream,
                &gpu_ctx,
                /*collective_nodes=*/nullptr,
                capture_policy,
                &used_graph_replay,
                submission))
        {
            return fail(
                "MTP draft publication failed under strict monolithic capture policy");
        }
        if (submission == DeviceGraphExecutor::GraphInitialSubmissionPolicy::MaterializeWithoutLaunch)
            return cache.segment_cache.deviceLoopGraphTemplate(*cache.graph, error).has_value();
        if (!cache.segment_cache.orderStreamAfterCapture(
                &gpu_ctx,
                producer_stream))
        {
            return fail(
                "captured MTP draft publication could not return completion to the sidecar stream");
        }
        return true;
    }

    bool DeviceGraphOrchestrator::materializeMTPVerifierPreparationGraph(
        const MTPVerifierPreparationStage::Params &params,
        MTPVerifierPreparationGraphKey *key,
        std::string *error)
    {
        auto fail = [&](const std::string &reason) -> bool
        {
            if (key)
                *key = {};
            if (error)
                *error = reason;
            LOG_ERROR("[DeviceGraphOrchestrator] " << reason);
            return false;
        };
        if (key)
            *key = {};
        if (error)
            error->clear();

        if (!mtp_graph_owner_plan_.ownsControllerExecutables())
            return fail("Pipeline follower cannot materialize controller-owned MTP helper executables");

        const uint64_t generation = workspaceGeneration(state_.device_id);
        if (!key || !state_.device_id.is_gpu() || generation == 0 ||
            params.device_id != state_.device_id ||
            mtp_verifier_preparation_graphs_.empty())
        {
            return fail(
                "captured MTP verifier preparation requires finalized GPU bindings and a preallocated registry");
        }

        const MTPVerifierPreparationGraphKey requested_key{
            .control_policy = params.generation_control_device
                                  ? MTPVerifierPreparationControlPolicy::
                                        DeviceGenerationControlled
                                  : MTPVerifierPreparationControlPolicy::
                                        Standalone,
            .request_count = params.request_count,
            .padded_seq_len = params.padded_seq_len,
        };
        std::string registry_error;
        const auto registry_index =
            mtpVerifierPreparationRegistryIndex(
                requested_key,
                &registry_error);
        if (!registry_index)
            return fail(registry_error);

        auto &cache =
            *mtp_verifier_preparation_graphs_[*registry_index];
        if (cache.valid && cache.workspace_generation != generation)
            cache.invalidate();

        if (cache.valid)
        {
            if (cache.key != requested_key || !cache.graph || !cache.stage ||
                cache.workspace_generation != generation)
            {
                return fail(
                    "captured MTP verifier preparation registry slot is internally inconsistent");
            }
            if (!cache.stage->hasSameCaptureIdentity(params))
            {
                std::ostringstream diagnostic;
                diagnostic
                    << "captured MTP verifier preparation key collision; "
                       "same-generation graph replacement is forbidden"
                    << " registry_index=" << *registry_index
                    << " key={control_policy="
                    << static_cast<int>(requested_key.control_policy)
                    << ",requests=" << requested_key.request_count
                    << ",padded_seq_len=" << requested_key.padded_seq_len
                    << "} retained={"
                    << cache.stage->captureIdentityDescription()
                    << "} incoming={"
                    << MTPVerifierPreparationStage::describeCaptureIdentity(
                           params)
                    << '}';
                return fail(diagnostic.str());
            }
            *key = requested_key;
            return true;
        }

        auto stage =
            std::make_unique<MTPVerifierPreparationStage>(params);
        auto *preparation_stage = stage.get();
        auto graph = std::make_unique<ComputeGraph>();
        graph->setExecutableMemoryClass(
            mtp_graph_owner_plan_.verifierPreparationExecutableClassForOwner(params.request_count));
        graph->addNode(
            "mtp_verifier_preparation_" +
                std::to_string(*registry_index),
            std::move(stage),
            state_.device_id);

        invalidateSparseQueryParents();
        cache.invalidate();
        cache.key = requested_key;
        cache.graph = std::move(graph);
        cache.stage = preparation_stage;
        cache.workspace_generation = generation;
        cache.valid = true;
        *key = requested_key;

        PerfStatsCollector::addCounter(
            "mtp",
            "verifier_preparation_graph_materializations",
            1.0,
            "graph_setup",
            state_.device_id.toString(),
            {{"registry_index", std::to_string(*registry_index)},
             {"requests", std::to_string(params.request_count)},
             {"padded_seq_len", std::to_string(params.padded_seq_len)},
             {"control_policy",
              std::to_string(
                  static_cast<int>(requested_key.control_policy))},
             {"rows",
              std::to_string(
                  params.request_count * params.padded_seq_len)},
             {"workspace_generation", std::to_string(generation)}});
        return true;
    }

    bool DeviceGraphOrchestrator::executeMTPVerifierPreparationCaptured(
        void *producer_stream,
        const MTPVerifierPreparationGraphKey &key,
        std::string *error,
        DeviceGraphExecutor::GraphInitialSubmissionPolicy submission)
    {
        auto fail = [&](const std::string &reason) -> bool
        {
            if (error)
                *error = reason;
            LOG_ERROR("[DeviceGraphOrchestrator] " << reason);
            return false;
        };
        if (error)
            error->clear();
        std::string registry_error;
        const auto registry_index =
            mtpVerifierPreparationRegistryIndex(key, &registry_error);
        if (!registry_index)
            return fail(registry_error);

        auto &cache = *mtp_verifier_preparation_graphs_[*registry_index];
        const uint64_t generation = workspaceGeneration(state_.device_id);
        if (!state_.device_id.is_gpu() || !producer_stream ||
            !cache.valid || cache.key != key || !cache.graph || !cache.stage ||
            cache.workspace_generation == 0 ||
            cache.workspace_generation != generation)
        {
            return fail(
                "captured MTP verifier preparation requires a current graph and the exact forward producer stream");
        }
        if (!debugEnv().execution.gpu_graphs)
        {
            return fail(
                "MTP verifier preparation requires graph capture; eager execution is forbidden");
        }

        IDeviceContext *ctx = getDeviceContext(state_.device_id);
        if (!ctx)
            return fail(
                "captured MTP verifier preparation has no device context");
        auto &gpu_ctx =
            GPUDeviceContextPool::instance().getContext(state_.device_id);
        if (!cache.segment_cache.ensureCaptureStream(
                &gpu_ctx,
                state_.device_id,
                /*context_from_process_pool=*/true))
        {
            return fail(
                "captured MTP verifier preparation has no persistent capture stream");
        }
        if (submission == DeviceGraphExecutor::GraphInitialSubmissionPolicy::MaterializeWithoutLaunch &&
            cache.segment_cache.initialized)
            return cache.segment_cache.deviceLoopGraphTemplate(*cache.graph, error).has_value();
        if (submission != DeviceGraphExecutor::GraphInitialSubmissionPolicy::MaterializeWithoutLaunch &&
            key.control_policy ==
                MTPVerifierPreparationControlPolicy::
                    DeviceGenerationControlled &&
            !consumeDeviceGenerationStateReady(
                producer_stream,
                DeviceTimelineRole::AllPositionVerifier,
                key.request_count,
                "mtp_verifier_preparation"))
        {
            return fail(
                "captured MTP verifier preparation could not consume its device-generation admission");
        }
        if (!cache.segment_cache.orderCaptureStreamAfter(
                &gpu_ctx,
                producer_stream))
        {
            return fail(
                "captured MTP verifier preparation could not consume its producer event frontier");
        }

        const auto &params = cache.stage->getParams();
        cache.segment_cache.perf_context = "mtp_verifier_preparation";
        cache.segment_cache.replay_workload = {
            .seq_len = params.padded_seq_len,
            .batch_size = params.request_count,
            .m = params.request_count * params.padded_seq_len,
            .decode = true,
        };
        auto capture_policy = buildDecodeCapturePolicy(
            /*has_collectives=*/false,
            ctx);
        if (!capture_policy.allow_cached_graph_replay)
        {
            return fail(
                "captured MTP verifier preparation requires mandatory graph replay");
        }
        capture_policy.defer_final_sync = true;
        capture_policy.force_recapture = false;
        capture_policy.graph_replay_plan_policy =
            DeviceGraphExecutor::GraphReplayPlanPolicy::RequireFullGraph;

        if (pipeline_maintenance_admission_) {
            const auto owner = pipeline_maintenance_admission_;
            const auto member = pipeline_maintenance_participant_;
            capture_policy.graph_replay_plan_policy = DeviceGraphExecutor::GraphReplayPlanPolicy::RequireRetainedParentComposition;
            capture_policy.retained_parent_composer = [owner, member](IGPUGraphCapture &destination, const ComputeGraph &,
                std::span<const DeviceGraphExecutor::GraphSegmentCache::RetainedCaptureUnitTemplateView> units) {
                if (units.size() != 1 || !units.front().capture) return false;
                const GPUOrderedTimelineStep steps[] = {
                    {.name = "original pipeline maintenance budget", .capture = owner->recording(member)},
                    {.name = "original tail verifier preparation", .capture = units.front().capture}};
                return destination.buildOrderedTimelineTransaction(steps);
            };
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
                capture_policy,
                &used_graph_replay,
                submission))
        {
            return fail(
                "MTP verifier preparation failed under strict monolithic capture policy");
        }
        if (submission == DeviceGraphExecutor::GraphInitialSubmissionPolicy::MaterializeWithoutLaunch)
            return true;
        if (!cache.segment_cache.orderStreamAfterCapture(
                &gpu_ctx,
                producer_stream))
        {
            return fail(
                "captured MTP verifier preparation could not publish completion to the forward stream");
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "verifier_preparation_graph_replays",
            1.0,
            "decode",
            state_.device_id.toString(),
            {{"registry_index", std::to_string(*registry_index)},
             {"requests", std::to_string(params.request_count)},
             {"padded_seq_len", std::to_string(params.padded_seq_len)},
             {"full_graph", "true"}});
        return true;
    }

    bool DeviceGraphOrchestrator::
        materializeMTPStochasticTargetDistributionGraph(
            int row_count,
            const SamplingParams &params,
            const MTPRequestPenaltyPolicy &penalty_policy,
            int vocab_size,
            std::string *error)
    {
        auto fail = [&](const std::string &reason) -> bool
        {
            if (error)
                *error = reason;
            LOG_ERROR("[DeviceGraphOrchestrator] " << reason);
            return false;
        };
        if (error)
            error->clear();

        if (!mtp_graph_owner_plan_.ownsControllerExecutables())
            return fail("Pipeline follower cannot materialize controller-owned MTP helper executables");

        const uint64_t generation = workspaceGeneration(state_.device_id);
        if (!state_.device_id.is_gpu() || generation == 0 || row_count <= 0 ||
            !supportsDeviceStochasticMTPVerification())
        {
            return fail(
                "captured stochastic verifier target preparation requires finalized GPU bindings and positive row geometry");
        }
        if (params.top_k <= 0 ||
            params.top_k > static_cast<int>(kStochasticDistributionMaxK) ||
            params.top_p <= 0.0F || params.top_p > 1.0F ||
            params.temperature <= 0.0F ||
            vocab_size <= 0 || vocab_size != state_.vocab_size)
        {
            return fail(
                "captured stochastic verifier target preparation received invalid sampling policy or vocabulary geometry");
        }
        if (row_count > stochastic_target_row_capacity_ ||
            !stochastic_target_token_ids_dev_ ||
            !stochastic_target_probs_dev_ ||
            !stochastic_topk_partial_vals_dev_ ||
            !stochastic_topk_partial_idxs_dev_ ||
            stochastic_topk_partial_capacity_ == 0)
        {
            return fail(
                "captured stochastic verifier target preparation has incomplete persistent distribution workspace");
        }

        IBackend *backend = getBackendFor(state_.device_id);
        TensorBase *logits = state_.all_position_logits.get();
        if (!backend || !logits || !logits->deviceValid() ||
            !logits->gpu_data_ptr())
        {
            return fail(
                "captured stochastic verifier target preparation could not bind all-position logits");
        }
        const auto &shape = logits->shape();
        if (shape.size() < 2 || row_count > static_cast<int>(shape[0]) ||
            shape[1] != static_cast<size_t>(vocab_size))
        {
            return fail(
                "captured stochastic verifier target preparation does not match the materialized logits tensor");
        }

        const bool apply_penalties = penalty_policy.enabled();
        const bool generation_controller_owned =
            device_generation_storage_.active_request_count > 0;
        if (generation_controller_owned &&
            (device_generation_storage_.active_request_count != 1 ||
             !device_generation_storage_.validFor(1) ||
             !mtp_verifier_request_lengths_dev_))
        {
            return fail(
                "captured device-loop target preparation requires one canonical resident verifier-length row");
        }
        if (apply_penalties &&
            (!arena_ || !mtp_verifier_input_tokens_dev_ ||
             !mtp_generated_token_counts_dev_ ||
             !mtp_greedy_penalty_policy_dev_ ||
             mtp_generated_token_count_capacity_ != vocab_size))
        {
            return fail(
                "captured stochastic verifier penalties have incomplete persistent history bindings");
        }

        MTPStochasticTargetDistributionStage::Params stage_params;
        stage_params.device_id = state_.device_id;
        stage_params.backend = backend;
        stage_params.logits_device =
            static_cast<float *>(logits->gpu_data_ptr());
        stage_params.first_logit_row = 0;
        stage_params.row_count = row_count;
        stage_params.vocab_size = vocab_size;
        stage_params.logits_row_stride = vocab_size;
        stage_params.active_rows_device =
            generation_controller_owned
                ? static_cast<const int32_t *>(
                      mtp_verifier_request_lengths_dev_)
                : nullptr;
        stage_params.apply_penalties = apply_penalties;
        stage_params.verifier_input_tokens_device =
            apply_penalties
                ? static_cast<const int32_t *>(
                      mtp_verifier_input_tokens_dev_)
                : nullptr;
        stage_params.generated_token_counts_device =
            apply_penalties
                ? static_cast<const int32_t *>(
                      mtp_generated_token_counts_dev_)
                : nullptr;
        stage_params.penalty_policy_device =
            apply_penalties ? mtp_greedy_penalty_policy_dev_ : nullptr;
        stage_params.top_k = params.top_k;
        stage_params.top_p = params.top_p;
        stage_params.temperature = params.temperature;
        stage_params.target_token_ids_device =
            static_cast<int32_t *>(stochastic_target_token_ids_dev_);
        stage_params.target_probs_device =
            static_cast<float *>(stochastic_target_probs_dev_);
        stage_params.first_target_slot = 0;
        stage_params.target_row_stride =
            static_cast<int>(kStochasticDistributionMaxK);
        stage_params.topk_partial_values_device =
            static_cast<float *>(stochastic_topk_partial_vals_dev_);
        stage_params.topk_partial_indices_device =
            static_cast<int32_t *>(stochastic_topk_partial_idxs_dev_);
        stage_params.topk_partial_capacity =
            stochastic_topk_partial_capacity_;

        auto &cache = mtpStochasticTargetDistributionCache(row_count);
        if (cache.valid && cache.graph && cache.stage &&
            cache.workspace_generation == generation &&
            cache.stage->hasSameCaptureIdentity(stage_params))
        {
            return true;
        }

        auto stage =
            std::make_unique<MTPStochasticTargetDistributionStage>(
                std::move(stage_params));
        auto *target_stage = stage.get();
        auto graph = std::make_unique<ComputeGraph>();
        graph->addNode(
            "mtp_stochastic_target_distribution",
            std::move(stage),
            state_.device_id);

        invalidateSparseQueryParents();
        cache.invalidate();
        cache.graph = std::move(graph);
        cache.stage = target_stage;
        cache.workspace_generation = generation;
        cache.valid = true;

        PerfStatsCollector::addCounter(
            "mtp",
            "stochastic_target_distribution_graph_materializations",
            1.0,
            "graph_setup",
            state_.device_id.toString(),
            {{"rows", std::to_string(row_count)},
             {"top_k", std::to_string(params.top_k)},
             {"penalties", apply_penalties ? "true" : "false"},
             {"workspace_generation", std::to_string(generation)}});
        return true;
    }

    bool DeviceGraphOrchestrator::
        executeMTPStochasticTargetDistributionCaptured(
            void *producer_stream,
            int row_count,
            std::string *error,
        DeviceGraphExecutor::GraphInitialSubmissionPolicy submission)
    {
        auto fail = [&](const std::string &reason) -> bool
        {
            if (error)
                *error = reason;
            LOG_ERROR("[DeviceGraphOrchestrator] " << reason);
            return false;
        };

        auto &cache = mtpStochasticTargetDistributionCache(row_count);
        const uint64_t generation = workspaceGeneration(state_.device_id);
        if (!state_.device_id.is_gpu() || !producer_stream ||
            !cache.valid || !cache.graph || !cache.stage ||
            cache.workspace_generation == 0 ||
            cache.workspace_generation != generation ||
            cache.stage->getParams().first_logit_row != 0 ||
            cache.stage->getParams().first_target_slot != 0 ||
            cache.stage->getParams().row_count != row_count)
        {
            return fail(
                "captured stochastic verifier target preparation requires a current zero-based graph and exact producer stream");
        }
        if (!debugEnv().execution.gpu_graphs)
        {
            return fail(
                "stochastic verifier target preparation requires graph capture; eager execution is forbidden");
        }

        if (submission == DeviceGraphExecutor::GraphInitialSubmissionPolicy::MaterializeWithoutLaunch &&
            cache.segment_cache.initialized)
            return cache.segment_cache.deviceLoopGraphTemplate(*cache.graph, error).has_value();
        IDeviceContext *ctx = getDeviceContext(state_.device_id);
        if (!ctx)
        {
            return fail(
                "captured stochastic verifier target preparation has no device context");
        }
        auto &gpu_ctx =
            GPUDeviceContextPool::instance().getContext(state_.device_id);
        if (!cache.segment_cache.ensureCaptureStream(
                &gpu_ctx,
                state_.device_id,
                /*context_from_process_pool=*/true) ||
            !cache.segment_cache.orderCaptureStreamAfter(
                &gpu_ctx,
                producer_stream))
        {
            return fail(
                "captured stochastic verifier target preparation could not consume verifier-logit readiness");
        }

        cache.segment_cache.perf_context =
            "mtp_stochastic_target_distribution";
        cache.segment_cache.replay_workload = {
            .seq_len = row_count,
            .batch_size = 1,
            .m = row_count,
            .all_position_rows = row_count,
            .decode = true,
            .all_position_logits = true,
        };
        auto capture_policy = buildDecodeCapturePolicy(
            /*has_collectives=*/false,
            ctx);
        if (!capture_policy.allow_cached_graph_replay)
        {
            return fail(
                "captured stochastic verifier target preparation requires mandatory graph replay");
        }
        capture_policy.defer_final_sync = true;
        capture_policy.force_recapture = false;
        capture_policy.graph_replay_plan_policy =
            DeviceGraphExecutor::GraphReplayPlanPolicy::RequireFullGraph;

        cache.graph->reset();
        bool used_graph_replay = false;
        if (!executor_.executeDecodeWithCapturePolicy(
                *cache.graph,
                ctx,
                &cache.segment_cache,
                cache.segment_cache.capture_stream,
                &gpu_ctx,
                /*collective_nodes=*/nullptr,
                capture_policy,
                &used_graph_replay,
                submission))
        {
            return fail(
                "stochastic verifier target preparation failed under strict monolithic capture policy");
        }
        if (submission == DeviceGraphExecutor::GraphInitialSubmissionPolicy::MaterializeWithoutLaunch)
            return true;
        if (!cache.segment_cache.orderStreamAfterCapture(
                &gpu_ctx,
                producer_stream))
        {
            return fail(
                "captured stochastic verifier target preparation could not publish completion to the verifier stream");
        }
        return true;
    }

    bool DeviceGraphOrchestrator::materializeMTPStochasticOutcomeGraph(
        const DeviceStochasticBatchOutcomeRequest *requests,
        int request_count,
        std::string *error)
    {
        using namespace sampling_math;

        auto fail = [&](const std::string &reason) -> bool
        {
            if (error)
                *error = reason;
            LOG_ERROR("[DeviceGraphOrchestrator] " << reason);
            return false;
        };
        if (error)
            error->clear();

        if (!mtp_graph_owner_plan_.ownsControllerExecutables())
            return fail("Pipeline follower cannot materialize controller-owned MTP helper executables");

        const uint64_t generation = workspaceGeneration(state_.device_id);
        if (!state_.device_id.is_gpu() || generation == 0 || !requests ||
            request_count <= 0 ||
            request_count > stochastic_batch_output_request_capacity_ ||
            !supportsDeviceStochasticMTPVerification())
        {
            return fail(
                "stochastic outcome graph requires finalized GPU bindings and positive request geometry");
        }
        if (!device_generation_storage_.validFor(request_count) ||
            device_generation_storage_.active_request_count != request_count ||
            !active_device_generation_admission_ ||
            active_device_generation_admission_->request_count != request_count ||
            !active_device_generation_admission_->sampling_seeds ||
            !mtp_publication_base_cache_snapshot_ready_ ||
            mtp_publication_base_cache_snapshot_request_count_ < request_count)
        {
            return fail(
                "stochastic outcome graph requires an admitted controller, request seeds and verifier-base snapshot");
        }
        const auto admitted_seeds =
            active_device_generation_admission_->sampling_seeds->values();
        if (admitted_seeds.size() != static_cast<size_t>(request_count))
            return fail("stochastic outcome request seeds do not match admitted request geometry");

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
            return fail("stochastic outcome graph could not resolve its backend");

        const MTPSpecDecodeMetadataDevicePointers &publication_metadata =
            mtp_spec_decode_metadata_binding_.devicePointers();
        if (!publication_metadata.base_cached_tokens ||
            !stochastic_target_token_ids_dev_ ||
            !stochastic_target_probs_dev_ ||
            !stochastic_verify_tokens_dev_ ||
            !mtp_verifier_input_tokens_dev_ ||
            !mtp_verifier_stop_tokens_dev_ ||
            !stochastic_batch_output_tokens_dev_ ||
            !stochastic_batch_output_meta_dev_)
        {
            return fail(
                "stochastic outcome graph has incomplete persistent arena bindings");
        }

        const int comparison_rows = requests[0].row_count;
        const int verifier_rows = comparison_rows + 1;
        const int verifier_input_stride = requests[0].token_row_stride;
        if (comparison_rows <= 0 || comparison_rows > mtp_max_draft_depth_ ||
            verifier_input_stride < verifier_rows)
        {
            return fail(
                "stochastic outcome graph received invalid verifier depth or input stride");
        }
        const bool materialized_batch_matches =
            materialized_mtp_verifier_device_token_batch_.valid &&
            materialized_mtp_verifier_device_token_batch_.request_count ==
                request_count &&
            materialized_mtp_verifier_device_token_batch_
                    .logical_padded_seq_len == verifier_input_stride &&
            materialized_mtp_verifier_device_token_batch_
                    .physical_padded_seq_len >= verifier_input_stride &&
            (materialized_mtp_verifier_device_token_batch_
                         .physical_padded_seq_len == verifier_input_stride ||
             request_count == 1) &&
            materialized_mtp_verifier_device_token_batch_
                    .total_token_capacity ==
                request_count *
                    materialized_mtp_verifier_device_token_batch_
                        .physical_padded_seq_len;
        const bool materialized_scalar_matches =
            request_count == 1 &&
            materialized_mtp_verifier_device_token_row_.valid &&
            materialized_mtp_verifier_device_token_row_
                    .total_verifier_input_tokens == verifier_rows &&
            materialized_mtp_verifier_device_token_row_.draft_token_count ==
                comparison_rows;
        if (!materialized_batch_matches && !materialized_scalar_matches)
        {
            return fail(
                "stochastic outcome graph requires a complete materialized verifier-token matrix");
        }
        if (!mtp_request_stop_tokens_published_session_epoch_ ||
            *mtp_request_stop_tokens_published_session_epoch_ != session_epoch_)
        {
            return fail(
                "stochastic outcome graph reached capture without request-admitted stop controls");
        }

        // Admission already owns and publishes this bank on the controller's
        // incoming event. A request's entropy changes data, never graph identity.
        const auto seed_tensor = arena_
            ? arena_->getSharedTensor(BufferId::SAMPLING_REQUEST_SEEDS) : nullptr;
        if (!seed_tensor || !seed_tensor->gpu_data_ptr() ||
            seed_tensor->size_bytes() < static_cast<size_t>(request_count) * sizeof(uint64_t) ||
            reinterpret_cast<uintptr_t>(seed_tensor->gpu_data_ptr()) % alignof(uint64_t) != 0)
        {
            return fail("stochastic outcome graph requires the admitted device request-seed bank");
        }
        const auto verification = requests[0].serial_sample_equivalent
            ? MTPStochasticOutcomeStage::Verification::SerialEquivalent
            : MTPStochasticOutcomeStage::Verification::OneHotProbabilityRejection;
        int shared_top_k = 0;
        for (int request_index = 0;
             request_index < request_count;
             ++request_index)
        {
            const DeviceStochasticBatchOutcomeRequest &request =
                requests[request_index];
            // Compare immutable admission metadata before capture. The device
            // bank remains the sole runtime seed input; this check neither
            // downloads it nor embeds per-request entropy in graph identity.
            if (request.inverse_sample_seed != admitted_seeds[request_index])
                return fail("stochastic outcome seed differs from request admission at request " +
                            std::to_string(request_index));
            const int expected_first_target_slot =
                request_index * verifier_rows;
            const int expected_token_row_offset =
                request_index * verifier_input_stride;
            if (request.serial_sample_equivalent == request.use_vllm_probability_rejection ||
                request.serial_sample_equivalent != requests[0].serial_sample_equivalent ||
                !request.derive_thresholds_from_seed ||
                request.inverse_sample_seed == 0 ||
                request.inverse_sample_first_logical_position >= 0 ||
                request.draw_position_source !=
                    DeviceStochasticDrawPositionSource::VerifierBaseSnapshot ||
                request.request_id != request_index ||
                request.row_count != comparison_rows ||
                request.first_target_slot != expected_first_target_slot ||
                request.first_draft_slot !=
                    request_index * comparison_rows ||
                request.bonus_target_slot !=
                    expected_first_target_slot + comparison_rows ||
                request.token_row_offset != expected_token_row_offset ||
                request.token_row_stride != verifier_input_stride ||
                !request.use_device_draft_tokens ||
                request.stop_token_count !=
                    mtp_request_stop_token_count_)
            {
                return fail(
                    "stochastic outcome descriptor does not match uniform captured geometry at request " +
                    std::to_string(request_index));
            }
            for (int stop_index = 0;
                 stop_index < request.stop_token_count;
                 ++stop_index)
            {
                if (request.stop_tokens[static_cast<size_t>(stop_index)] !=
                    mtp_request_stop_tokens_[static_cast<size_t>(stop_index)])
                {
                    return fail(
                        "stochastic outcome stop controls differ from request admission");
                }
            }

            const int request_top_k =
                stochastic_target_top_k_[
                    static_cast<size_t>(expected_first_target_slot)];
            if (request_top_k <= 0 ||
                request_top_k >
                    static_cast<int>(kStochasticDistributionMaxK) ||
                (shared_top_k != 0 && shared_top_k != request_top_k))
            {
                return fail(
                    "stochastic outcome target rows disagree on top-k width");
            }
            shared_top_k = request_top_k;
            for (int row = 0; row < verifier_rows; ++row)
            {
                const int slot = expected_first_target_slot + row;
                if (slot < 0 || slot >= stochastic_target_row_capacity_ ||
                    stochastic_target_top_k_[static_cast<size_t>(slot)] !=
                        shared_top_k ||
                    stochastic_target_row_formats_[static_cast<size_t>(slot)] !=
                        StochasticRowFormat::CompactDistribution)
                {
                    return fail(
                        "stochastic outcome target matrix is incomplete at slot " +
                        std::to_string(slot));
                }
            }
        }
        if (request_count * verifier_rows > stochastic_target_row_capacity_)
        {
            return fail(
                "stochastic outcome geometry exceeds target-row capacity");
        }

        /*
         * Static ExpertOverlay still uses the device generation controller,
         * but it intentionally has no rebalance graph or maintenance clock.
         * Bind this second controller only when Dynamic maintenance is an
         * active child of the homogeneous device authority. Manufacturing a
         * maintenance graph for Static would violate its no-movement contract.
         */
        const bool requires_rebalance_controller =
            usesParticipantLocalDeviceMoERebalanceController();
        std::string rebalance_controller_error;
        DeviceMoERebalanceGraphControllerState *rebalance_controller =
            requires_rebalance_controller
                ? deviceMoERebalanceControllerStateDevice(
                      &rebalance_controller_error)
                : nullptr;
        if (requires_rebalance_controller && !rebalance_controller)
        {
            return fail(
                "stochastic outcome graph cannot bind the required MoE maintenance controller: " +
                rebalance_controller_error);
        }
        auto *rebalance_bytes =
            reinterpret_cast<std::byte *>(rebalance_controller);
        auto controller_field = [&](size_t offset) -> uint32_t *
        {
            return rebalance_controller
                       ? reinterpret_cast<uint32_t *>(rebalance_bytes + offset)
                       : nullptr;
        };

        MTPStochasticOutcomeStage::Params params;
        params.device_id = state_.device_id;
        params.backend = backend;
        params.verification = verification;
        params.vocabulary_size =
            graph_builder_ ? graph_builder_->config().vocab_size : state_.vocab_size;
        params.accepted_rows_device =
            static_cast<int32_t *>(stochastic_verify_accepted_dev_);
        params.accepted_row_stride = verifier_rows;
        params.target_token_ids_device =
            static_cast<const int32_t *>(stochastic_target_token_ids_dev_);
        params.target_probs_device =
            static_cast<const float *>(stochastic_target_probs_dev_);
        params.target_distribution_row_stride =
            static_cast<int>(kStochasticDistributionMaxK);
        params.top_k = shared_top_k;
        params.verifier_input_tokens_device =
            static_cast<const int32_t *>(mtp_verifier_input_tokens_dev_);
        params.verifier_input_token_stride = verifier_input_stride;
        params.stop_tokens_device =
            static_cast<const int32_t *>(mtp_verifier_stop_tokens_dev_);
        params.stop_token_stride = 0;
        params.threshold_base_positions_device =
            publication_metadata.base_cached_tokens;
        params.threshold_position_offset = 1;
        params.threshold_seeds_device =
            static_cast<const uint64_t *>(seed_tensor->gpu_data_ptr());
        params.generation_control_device =
            device_generation_storage_.control_device;
        params.generation_control_stride =
            device_generation_storage_.control_stride;
        params.maintenance_rows_remaining_device = pipeline_maintenance_admission_
            ? pipeline_maintenance_admission_->remaining(pipeline_maintenance_participant_)
            : controller_field(offsetof(DeviceMoERebalanceGraphControllerState, decode_rounds_until_maintenance));
        /*
         * Dynamic MTP captures a distinct verifier geometry for the currently
         * selected depth. The resident response controller may commit only rows
         * produced by this exact graph. Using the configured maximum depth here
         * authorizes a larger transaction than publication's captured
         * `verifier_rows` storage and correctly trips the fail-hard metadata
         * guard when the controller starts below its maximum depth.
         */
        params.verifier_row_capacity = verifier_rows;
        params.sampled_target_tokens_device =
            static_cast<int32_t *>(stochastic_verify_tokens_dev_);
        params.sampled_target_token_stride = verifier_rows;
        params.output_tokens_device =
            static_cast<int32_t *>(stochastic_batch_output_tokens_dev_);
        params.output_token_stride = stochastic_batch_output_token_stride_;
        params.output_meta_device =
            static_cast<int *>(stochastic_batch_output_meta_dev_);
        params.output_meta_stride = kSpeculativeBatchMetaCount;
        const bool retain_first_transaction_diagnostic = mtp_diagnostic_record_mode_ == sampling_math::MTPDiagnosticRecordMode::FirstTransaction;
        if (retain_first_transaction_diagnostic &&
            !mtp_first_transaction_diagnostic_dev_)
        {
            return fail(
                "stochastic outcome graph has no arena-owned first-transaction diagnostic record");
        }
        params.first_transaction_diagnostic_device =
            retain_first_transaction_diagnostic
                ? mtp_first_transaction_diagnostic_dev_
                : nullptr;
        params.request_count = request_count;
        params.comparison_rows_per_request = comparison_rows;
        params.advance_maintenance_boundary = rebalance_controller != nullptr;
        params.decode_rounds_committed_device =
            controller_field(offsetof(
                DeviceMoERebalanceGraphControllerState,
                decode_rounds_committed));
        params.decode_rounds_until_maintenance_device =
            controller_field(offsetof(
                DeviceMoERebalanceGraphControllerState,
                decode_rounds_until_maintenance));
        params.maintenance_due_device =
            controller_field(offsetof(
                DeviceMoERebalanceGraphControllerState,
                maintenance_due));
        params.decode_boundary_advanced_device =
            controller_field(offsetof(
                DeviceMoERebalanceGraphControllerState,
                decode_boundary_advanced));

        auto &cache = mtpStochasticOutcomeCache(verifier_rows);
        if (cache.valid && cache.graph && cache.stage &&
            cache.workspace_generation == generation &&
            cache.stage->hasSameCaptureIdentity(params))
        {
            return true;
        }

        auto stage =
            std::make_unique<MTPStochasticOutcomeStage>(
                std::move(params));
        auto *outcome_stage = stage.get();
        auto graph = std::make_unique<ComputeGraph>();
        graph->addNode(
            "mtp_stochastic_outcome",
            std::move(stage),
            state_.device_id);

        invalidateSparseQueryParents();
        cache.invalidate();
        cache.graph = std::move(graph);
        cache.stage = outcome_stage;
        cache.workspace_generation = generation;
        cache.valid = true;

        PerfStatsCollector::addCounter(
            "mtp",
            "stochastic_outcome_graph_materializations",
            1.0,
            "graph_setup",
            state_.device_id.toString(),
            {{"requests", std::to_string(request_count)},
             {"comparison_rows", std::to_string(comparison_rows)},
             {"top_k", std::to_string(shared_top_k)},
             {"workspace_generation", std::to_string(generation)}});
        return true;
    }

    bool DeviceGraphOrchestrator::executeMTPStochasticOutcomeCaptured(
        void *producer_stream,
        int request_count,
        int comparison_rows_per_request,
        std::string *error,
        DeviceGraphExecutor::GraphInitialSubmissionPolicy submission)
    {
        auto fail = [&](const std::string &reason) -> bool
        {
            if (error)
                *error = reason;
            LOG_ERROR("[DeviceGraphOrchestrator] " << reason);
            return false;
        };

        auto &cache = mtpStochasticOutcomeCache(comparison_rows_per_request + 1);
        const uint64_t generation = workspaceGeneration(state_.device_id);
        if (!state_.device_id.is_gpu() || !producer_stream ||
            !cache.valid || !cache.graph || !cache.stage ||
            cache.workspace_generation == 0 ||
            cache.workspace_generation != generation ||
            cache.stage->getParams().request_count != request_count ||
            cache.stage->getParams().comparison_rows_per_request !=
                comparison_rows_per_request)
        {
            return fail(
                "captured stochastic outcome requires a current exact-geometry graph and producer stream");
        }
        if (!debugEnv().execution.gpu_graphs)
        {
            return fail(
                "stochastic GPU outcome reduction requires graph capture; eager execution is forbidden");
        }

        if (submission == DeviceGraphExecutor::GraphInitialSubmissionPolicy::MaterializeWithoutLaunch &&
            cache.segment_cache.initialized)
            return cache.segment_cache.deviceLoopGraphTemplate(*cache.graph, error).has_value();
        IDeviceContext *ctx = getDeviceContext(state_.device_id);
        if (!ctx)
            return fail("captured stochastic outcome has no device context");
        auto &gpu_ctx =
            GPUDeviceContextPool::instance().getContext(state_.device_id);
        if (!cache.segment_cache.ensureCaptureStream(
                &gpu_ctx,
                state_.device_id,
                /*context_from_process_pool=*/true) ||
            !cache.segment_cache.orderCaptureStreamAfter(
                &gpu_ctx,
                producer_stream))
        {
            return fail(
                "captured stochastic outcome could not consume target-distribution readiness");
        }

        cache.segment_cache.perf_context =
            "mtp_stochastic_outcome";
        cache.segment_cache.replay_workload = {
            .seq_len = comparison_rows_per_request + 1,
            .batch_size = request_count,
            .m = (comparison_rows_per_request + 1) * request_count,
            .all_position_rows = comparison_rows_per_request + 1,
            .decode = true,
            .all_position_logits = true,
        };
        auto capture_policy = buildDecodeCapturePolicy(
            /*has_collectives=*/false,
            ctx);
        if (!capture_policy.allow_cached_graph_replay)
        {
            return fail(
                "captured stochastic outcome requires mandatory graph replay");
        }
        capture_policy.defer_final_sync = true;
        capture_policy.force_recapture = false;
        capture_policy.graph_replay_plan_policy =
            DeviceGraphExecutor::GraphReplayPlanPolicy::RequireFullGraph;

        cache.graph->reset();
        bool used_graph_replay = false;
        if (!executor_.executeDecodeWithCapturePolicy(
                *cache.graph,
                ctx,
                &cache.segment_cache,
                cache.segment_cache.capture_stream,
                &gpu_ctx,
                /*collective_nodes=*/nullptr,
                capture_policy,
                &used_graph_replay,
                submission))
        {
            return fail(
                "stochastic outcome failed under the strict monolithic capture policy");
        }
        if (submission == DeviceGraphExecutor::GraphInitialSubmissionPolicy::MaterializeWithoutLaunch)
            return true;
        if (!cache.segment_cache.orderStreamAfterCapture(
                &gpu_ctx,
                producer_stream))
        {
            return fail(
                "stochastic outcome could not publish completion to its producer stream");
        }
        return true;
    }

    bool DeviceGraphOrchestrator::sealDeviceSpeculativeOutcomeVerifierGeometry(
        DeviceSpeculativeOutcomeHandle *handle,
        int request_count,
        int logical_rows_per_request,
        const char *producer_name)
    {
        const char *producer =
            producer_name && producer_name[0] != '\0'
                ? producer_name
                : "unknown";
        auto fail = [&](const std::string &reason) -> bool
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Compact outcome geometry seal failed"
                      << " producer=" << producer << ": " << reason);
            return false;
        };

        if (!handle || !state_.device_id.is_gpu() || request_count <= 0 ||
            logical_rows_per_request <= 0)
        {
            return fail("invalid GPU outcome or logical request geometry");
        }
        if (!forward_engine_)
            return fail("no initialized forward engine");

        auto verifier_graph =
            forward_engine_->lastAllPositionVerifierForwardGraph();
        if (!verifier_graph || !*verifier_graph || !verifier_graph->graph ||
            !verifier_graph->is_decode ||
            !verifier_graph->all_position_logits)
        {
            return fail("no retained all-position verifier producer graph");
        }
        if (verifier_graph->signature.batch_size != request_count)
        {
            return fail(
                "request count differs from retained verifier graph: logical=" +
                std::to_string(request_count) + " physical=" +
                std::to_string(verifier_graph->signature.batch_size));
        }

        std::string width_policy_error;
        const auto width_policy =
            mtpVerifierPhysicalWidthPolicyForRequest(
                request_count,
                &width_policy_error);
        const int expected_physical_rows =
            width_policy
                ? mtpVerifierPhysicalPaddedSeqLen(
                      request_count,
                      logical_rows_per_request,
                      mtp_max_verifier_rows_,
                      *width_policy)
                : 0;
        const int graph_physical_rows = verifier_graph->signature.seq_len;
        if (expected_physical_rows <= 0 ||
            graph_physical_rows != expected_physical_rows)
        {
            return fail(
                "retained verifier graph violates the logical-to-physical "
                "bucket policy: logical_rows=" +
                std::to_string(logical_rows_per_request) +
                " expected_physical_rows=" +
                std::to_string(expected_physical_rows) +
                " graph_physical_rows=" +
                std::to_string(graph_physical_rows) +
                " policy_error=" + width_policy_error);
        }

        handle->request_count = request_count;
        handle->logical_verifier_rows_per_request = logical_rows_per_request;
        handle->physical_verifier_rows_per_request = graph_physical_rows;
        return true;
    }


    bool DeviceGraphOrchestrator::finalizeMTPSpeculativeStatePublicationLaunch(
        const DeviceSpeculativePublicationRequest &request,
        std::string *error)
    {
        auto fail = [&](const std::string &reason) -> bool
        {
            if (error)
                *error = reason;
            LOG_ERROR("[DeviceGraphOrchestrator] " << reason);
            return false;
        };
        if (!request.outcome.stream)
            return fail("MTP publication lifecycle requires an explicit completed producer stream");

        if (request.outcome.device_generation_controller_owned &&
            !publishDeviceGenerationStateReady(
                request.outcome.stream,
                request.requestCount(),
                DeviceGenerationStatePublicationKind::
                    CommittedTransaction))
        {
            return fail(
                "MTP publication could not publish generation-controller readiness");
        }

        if (request.requestCount() == 1)
        {
            try
            {
                arena_->markWritten(
                    BufferId::MTP_GREEDY_PENALTY_POLICY,
                    state_.device_id,
                    request.outcome.stream);
                arena_->markWritten(
                    BufferId::MTP_GENERATED_TOKEN_COUNTS,
                    state_.device_id,
                    request.outcome.stream);
            }
            catch (const std::exception &exception)
            {
                return fail(
                    "MTP publication could not publish penalty-history readiness: " +
                    std::string(exception.what()));
            }
        }

        handleLivePrefixReplayStateAfterMutation(
            LivePrefixMutationReason::AcceptedSpecPublication,
            "mtp_spec_state_publication_device_resident");

        if (request.publish_mtp_shifted_kv &&
            !recordDeviceResidentMTPTransactionMutation(
                request.requestCount(),
                request.outcome.stream,
                "mtp_spec_state_publication_device_resident_shifted_kv"))
        {
            return fail(
                "MTP publication could not advance its shifted-cache transaction fence");
        }

        /*
         * The fused publication kernel has refreshed target slots [0, R) with
         * the next condition tokens. Publish one exact event per slot on the
         * outcome stream so the next verifier preparation consumes a single
         * canonical pointer topology regardless of acceptance/rejection. This
         * is metadata-only host bookkeeping: token values never leave device.
         */
        if (request.requestCount() >
            static_cast<int>(stochastic_target_sample_ready_.size()))
        {
            return fail(
                "MTP publication produced more canonical verifier conditions than the preallocated target-slot family");
        }
        for (int request_index = 0;
             request_index < request.requestCount();
             ++request_index)
        {
            clearStochasticTargetSampleReadySlot(
                request_index,
                StochasticSampleReadyClearMode::Force);
            if (!recordStochasticTargetSampleReady(
                    request_index,
                    request.outcome.stream,
                    /*verifier_consumer_pending=*/true, StochasticTargetPublication::NewValue))
            {
                return fail(
                    "MTP publication could not publish canonical verifier-condition readiness for request " +
                    std::to_string(request_index));
            }
        }

        std::string mailbox_error;
        if (!recordDeviceResidentLogicalSequenceStateMailbox(
                request.requestCount(),
                request.outcome.stream,
                DeviceResidentLogicalStatePublicationKind::AcceptedSpecState,
                &mailbox_error))
        {
            return fail(
                mailbox_error.empty()
                    ? "MTP publication could not record its logical-state mailbox"
                    : "MTP publication could not record its logical-state mailbox: " +
                          mailbox_error);
        }
        if (!recordAcceptedSpecPublicationReady(
                request.outcome.stream,
                "mtp_spec_state_publication_device_resident"))
        {
            return fail(
                "MTP publication could not record accepted-state readiness");
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "device_resident_kv_sequence_state_publications",
            1.0,
            "decode",
            state_.device_id.toString(),
            {{"request_count", std::to_string(request.requestCount())}});
        PerfStatsCollector::addCounter(
            "mtp",
            "canonical_verifier_condition_publications",
            static_cast<double>(request.requestCount()),
            "decode",
            state_.device_id.toString(),
            {{"requests", std::to_string(request.requestCount())},
             {"owner", "captured_speculative_state_publication"},
             {"ordering", "producer_event"}});
        PerfStatsCollector::addCounter(
            "mtp",
            "spec_state_publications",
            1.0,
            "decode",
            state_.device_id.toString(),
            {{"implementation", "captured_device_resident"},
             {"requests", std::to_string(request.requestCount())},
             {"publishes_mtp_shifted_kv",
              request.publish_mtp_shifted_kv ? "true" : "false"}});
        return true;
    }

    bool DeviceGraphOrchestrator::publishAcceptedMTPSpecStateBatchFromDeviceOutcome(
        const DeviceSpeculativePublicationRequest &request,
        std::string *error)
    {
        std::string prepare_error;
        {
            PerfStatsCollector::ScopedTimer timer(
                "mtp",
                "device_resident_publication_prepare_metadata",
                perfPhaseName(),
                state_.device_id.toString(),
                {{"request_count", std::to_string(request.requestCount())},
                 {"logical_verifier_rows",
                  std::to_string(request.logicalVerifierRowsPerRequest())},
                 {"physical_verifier_rows",
                  std::to_string(request.physicalVerifierRowsPerRequest())}});
            if (!prepareDeviceResidentMTPSpecPublicationMetadata(
                    request,
                    &prepare_error))
            {
                if (error)
                {
                    *error =
                        "device-resident MTP publication preflight failed: " +
                        prepare_error;
                }
                return false;
            }
        }

        auto verifier_graph =
            forward_engine_
                ? forward_engine_->lastAllPositionVerifierForwardGraph()
                : decltype(
                      forward_engine_
                          ->lastAllPositionVerifierForwardGraph()){};
        if (!verifier_graph || !*verifier_graph ||
            !verifier_graph->graph)
        {
            if (error)
            {
                *error =
                    "device-resident MTP publication lost its verifier graph after preflight";
            }
            return false;
        }

        /*
         * The compact reducer, shifted sidecars, verifier-state capture, and
         * main-KV checkpoints may use different explicit streams. Join their
         * immutable completion events on the compact outcome stream before
         * handing that stream to the publication graph. The graph cache then
         * creates one producer -> capture and one capture -> consumer event
         * edge; no stream or device synchronization is involved.
         */
        if (request.publish_mtp_shifted_kv &&
            !waitForPendingShiftedMTPKVReady(
                request.outcome.stream,
                "mtp_spec_state_publication_device_resident"))
        {
            if (error)
            {
                *error =
                    "device-resident MTP publication could not join shifted-KV readiness";
            }
            return false;
        }
        if (!waitForPendingAllPositionVerifierStateReady(
                request.outcome.stream,
                "mtp_spec_state_publication_device_resident"))
        {
            if (error)
            {
                *error =
                    "device-resident MTP publication could not join verifier-state readiness";
            }
            return false;
        }

        if (request.requestCount() == 1 &&
            (!arena_ || !mtp_greedy_penalty_policy_dev_ ||
             !mtp_generated_token_counts_dev_ ||
             mtp_generated_token_count_capacity_ != state_.vocab_size ||
             !consumeMTPRequestPenaltyPolicyOnDevice(
                 request.penalty_policy,
                 request.outcome.stream,
                 "mtp_speculative_state_publication") ||
             !arena_->prepareForRead(
                 BufferId::MTP_GENERATED_TOKEN_COUNTS,
                 state_.device_id,
                 request.outcome.stream)))
        {
            if (error)
            {
                *error =
                    "device-resident MTP publication could not consume the persistent penalty history";
            }
            return false;
        }

        // Cold capture, producer readiness and penalty preparation have all
        // succeeded. Open the original exclusive writer immediately before
        // submission. Once submission can enqueue writes, failure stays closed
        // until authentic retirement; it must never restore an older mailbox.
        if (!waitForDeviceResidentLogicalSequenceStateRowReuse(
                request.outcome.stream,
                "mtp_spec_publication_logical_state_row_reuse"))
        {
            if (error)
                *error = "device-resident MTP publication could not order reusable logical-state rows";
            return false;
        }

        std::string publication_error;
        {
            PerfStatsCollector::ScopedTimer timer(
                "mtp",
                "captured_device_resident_publication",
                perfPhaseName(),
                state_.device_id.toString(),
                {{"request_count", std::to_string(request.requestCount())},
                 {"logical_verifier_rows",
                  std::to_string(request.logicalVerifierRowsPerRequest())},
                 {"physical_verifier_rows",
                  std::to_string(request.physicalVerifierRowsPerRequest())}});
            if (!executeMTPStatePublicationGraph(
                    request.outcome.stream,
                    &publication_error))
            {
                if (error)
                {
                    *error =
                        "device-resident MTP publication graph failed: " +
                        publication_error;
                }
                return false;
            }
        }

        /*
         * Accepted terminal hidden is a separately reusable child fragment.
         * Keeping this as an explicit graph dependency does not imply a future
         * runtime launch boundary: the parent loop will clone publication and
         * selection consecutively into one native graph. It also avoids fusing a
         * large d_model row copy into the scalar metadata transaction.
         */
        if (request.publish_mtp_shifted_kv)
        {
            const int total_graph_rows =
                verifier_graph->signature.seq_len *
                verifier_graph->signature.batch_size;
            if (!selectMTPTerminalHiddenRowsFromDeviceAcceptedState(
                    request.requestCount(),
                    total_graph_rows,
                    request.outcome.stream))
            {
                if (error)
                {
                    *error =
                        "device-resident MTP publication could not publish accepted terminal hidden";
                }
                return false;
            }
            PerfStatsCollector::addCounter(
                "mtp",
                "spec_state_terminal_hidden_publications",
                1.0,
                "decode",
                state_.device_id.toString(),
                {{"implementation", "captured_device_resident"},
                 {"requests", std::to_string(request.requestCount())},
                 {"target_rows",
                  std::to_string(verifier_graph->signature.seq_len)}});
        }

        if (!finalizeMTPSpeculativeStatePublicationLaunch(
                request,
                &publication_error))
        {
            if (error)
            {
                *error =
                    "device-resident MTP publication lifecycle failed: " +
                    publication_error;
            }
            return false;
        }

        mtp_publication_main_kv_base_checkpoints_ready_ = false;
        mtp_publication_main_kv_base_checkpoint_request_count_ = 0;
        if (forward_engine_)
            forward_engine_->clearLastAllPositionVerifierForwardGraph();
        return true;
    }


    bool DeviceGraphOrchestrator::publishAcceptedMTPSpecState(
        const MTPSpecStepPlan &plan,
        std::string *error)
    {
        auto fail = [&](std::string reason) -> bool
        {
            if (error)
                *error = reason;
            LOG_ERROR("[DeviceGraphOrchestrator] " << reason);
            return false;
        };

        if (!supportsMTPSpecStatePublication())
        {
            return fail(
                "MTP spec-state publication is not advertised for this runner");
        }
        if (!graph_builder_ || !graph_builder_->config().mtp.enabled)
        {
            return fail("MTP spec-state publication requires an MTP-enabled graph builder");
        }
        if (!forward_engine_)
        {
            return fail("MTP spec-state publication requires an initialized forward engine");
        }
        if (state_.device_id.is_gpu())
        {
            return fail(
                "GPU MTP spec-state publication from host step plans is retired; use compact device-resident outcome publication");
        }

        auto verifier_graph = forward_engine_->lastAllPositionVerifierForwardGraph();
        if (!verifier_graph || !*verifier_graph || !verifier_graph->graph)
        {
            return fail("MTP spec-state publication has no cached verifier graph from the last forward");
        }
        if (!verifier_graph->is_decode || !verifier_graph->all_position_logits)
        {
            return fail("MTP spec-state publication requires the last forward to be an all-position verifier decode graph");
        }
        if (verifier_graph->signature.seq_len != plan.draft_count)
        {
            std::ostringstream msg;
            msg << "MTP spec-state publication verifier-row mismatch: plan="
                << plan.draft_count << " graph=" << verifier_graph->signature.seq_len
                << " metadata_target_rows=" << plan.target_rows;
            return fail(msg.str());
        }

        void *stream = verifier_graph->stream;

        if (!state_.kv_cache)
        {
            return fail("MTP spec-state publication requires an initialized main KV cache");
        }
        std::vector<IKVCache *> mtp_caches;
        if (plan.publish_mtp_shifted_kv)
        {
            mtp_caches.reserve(state_.mtp_kv_caches.size());
            for (const auto &cache : state_.mtp_kv_caches)
            {
                if (!cache)
                {
                    return fail("MTP spec-state publication requires initialized MTP KV caches");
                }
                mtp_caches.push_back(cache.get());
            }
        }

        MTPSpecKVPublicationResult kv_result =
            publishAcceptedMTPSpecKVState(
                plan,
                *state_.kv_cache,
                mtp_caches,
                /*seq_idx=*/0,
                stream);
        if (!kv_result.ok)
        {
            return fail(kv_result.error);
        }

        MTPSpecStatePublicationResult result =
            llaminar2::publishAcceptedMTPSpecState(
                plan,
                *verifier_graph->graph,
                state_.device_id,
                stream,
                /*require_captured_stage=*/mtpSpecStatePublicationRequiresCapturedStage());
        if (!result.ok)
        {
            return fail(result.error);
        }

        /*
         * The next MTP sidecar consumes PREFIX_TERMINAL_HIDDEN as its main-model
         * hidden input.  GDN/short-conv verifier-state publication restores
         * recurrent state, and KV publication truncates/appends cache state, but
         * neither one updates this terminal-hidden buffer.  Select the accepted
         * verifier row explicitly and enqueue the copy on the same stream used
         * for publication so the readiness event below covers every live-state
         * component the next sidecar will read.
         */
        if (plan.publish_mtp_shifted_kv && result.accepted_count > 0)
        {
            const int accepted_hidden_row = result.accepted_count - 1;
            if (!selectMTPTerminalHiddenRow(
                    accepted_hidden_row,
                    verifier_graph->signature.seq_len,
                    stream))
            {
                return fail("MTP spec-state publication could not restore accepted terminal hidden row");
            }
            PerfStatsCollector::addCounter(
                "mtp",
                "spec_state_terminal_hidden_publications",
                1.0,
                "decode",
                state_.device_id.toString(),
                {{"accepted_row", std::to_string(accepted_hidden_row)},
                 {"target_rows", std::to_string(verifier_graph->signature.seq_len)}});
        }

        if (result.accepted_count > 0 && !publishCPUAcceptedVerifierLogits(
                PrefixLogitsRowSelection(static_cast<size_t>(result.accepted_count - 1),
                    static_cast<size_t>(verifier_graph->signature.seq_len), session_epoch_)))
            return fail("MTP spec-state publication could not publish accepted terminal logits");

        if (!state_.positions.empty())
            state_.positions[0] = plan.target_cached_tokens;
        if (!state_.sequence_lengths.empty())
            state_.sequence_lengths[0] = plan.target_cached_tokens;
        /*
         * Publication advances live state to a verifier-captured accepted row.
         * That is a hard mutation boundary for every captured decode/sidecar
         * graph, even when the speculative step accepted all draft rows.  The
         * verifier graph was captured against the old live-state epoch and may
         * contain backend-owned dynamic pointers or row metadata that are not
         * part of PrefixStateSnapshot.  Resetting replay state here keeps the
         * published state atomic: future graph launches must be captured from
         * the newly published positions, KV/GDN state, and shifted MTP cache.
         */
        const LivePrefixMutationReason mutation_reason =
            plan.requiresCorrectionReplay()
                ? LivePrefixMutationReason::RejectedCorrection
                : LivePrefixMutationReason::AcceptedSpecPublication;
        handleLivePrefixReplayStateAfterMutation(
            mutation_reason,
            "mtp_spec_state_publication");

        PerfStatsCollector::addCounter(
            "mtp",
            "spec_state_publications",
            1.0,
            "decode",
            state_.device_id.toString(),
            {{"accepted_count", std::to_string(result.accepted_count)},
	             {"main_kv_tokens", std::to_string(kv_result.main_truncated_tokens)},
	             {"publishes_mtp_shifted_kv", plan.publish_mtp_shifted_kv ? "true" : "false"},
	             {"restored_stages", std::to_string(result.restored_stage_count)},
	             {"skipped_stages", std::to_string(result.skipped_stage_count)},
		             {"target_rows", std::to_string(plan.target_rows)}});
        if (forward_engine_)
            forward_engine_->clearLastAllPositionVerifierForwardGraph();
        return true;
    }

    bool DeviceGraphOrchestrator::publishAcceptedMTPSpecStateBatch(
        const MTPSpecStepPlanBatch &plans,
        std::string *error)
    {
        auto fail = [&](std::string reason) -> bool
        {
            if (error)
                *error = reason;
            LOG_ERROR("[DeviceGraphOrchestrator] " << reason);
            return false;
        };

        if (!supportsMTPSpecStatePublication() &&
            !grouped_decode_equivalent_spec_publication_scope_)
        {
            return fail(
                "MTP batched spec-state publication is not advertised for this runner");
        }
        if (!plans.ok)
            return fail("MTP batched spec-state publication received an invalid plan batch: " + plans.error);
        if (!graph_builder_ || !graph_builder_->config().mtp.enabled)
            return fail("MTP batched spec-state publication requires an MTP-enabled graph builder");
        if (!forward_engine_)
            return fail("MTP batched spec-state publication requires an initialized forward engine");
        if (state_.device_id.is_gpu())
        {
            return fail(
                "GPU MTP batched spec-state publication from host step plans is retired; use compact device-resident outcome publication");
        }
        if (plans.request_count <= 0 ||
            static_cast<int>(plans.steps.size()) != plans.request_count)
        {
            return fail("MTP batched spec-state publication received inconsistent request count");
        }
        if (plans.request_count > state_.batch_size)
        {
            return fail("MTP batched spec-state publication exceeds initialized runner batch size");
        }
        if (static_cast<int>(state_.positions.size()) < plans.request_count ||
            static_cast<int>(state_.sequence_lengths.size()) < plans.request_count)
        {
            return fail("MTP batched spec-state publication requires per-request position state");
        }

        auto verifier_graph = forward_engine_->lastAllPositionVerifierForwardGraph();
        if (!verifier_graph || !*verifier_graph || !verifier_graph->graph)
            return fail("MTP batched spec-state publication has no cached verifier graph from the last forward");
        if (!verifier_graph->is_decode || !verifier_graph->all_position_logits)
            return fail("MTP batched spec-state publication requires the last forward to be an all-position verifier decode graph");
        if (verifier_graph->signature.batch_size != plans.request_count)
        {
            std::ostringstream msg;
            msg << "MTP batched spec-state publication request-count mismatch: plan="
                << plans.request_count << " graph=" << verifier_graph->signature.batch_size;
            return fail(msg.str());
        }

        const int padded_seq_len = verifier_graph->signature.seq_len;
        if (padded_seq_len <= 0)
            return fail("MTP batched spec-state publication received an invalid verifier graph sequence length");

        int max_draft_count = 0;
        std::vector<bool> seen_request(static_cast<size_t>(plans.request_count), false);
        std::vector<int> restore_rows(static_cast<size_t>(plans.request_count), -1);
        bool any_shifted_kv_publication = false;
        bool any_correction = false;
        for (const MTPSpecStepPlan &step : plans.steps)
        {
            if (step.request_index < 0 || step.request_index >= plans.request_count)
                return fail("MTP batched spec-state publication received an out-of-range request index");
            if (seen_request[static_cast<size_t>(step.request_index)])
                return fail("MTP batched spec-state publication received a duplicate request index");
            seen_request[static_cast<size_t>(step.request_index)] = true;

            if (step.draft_count < 0 || step.draft_count > padded_seq_len)
                return fail("MTP batched spec-state publication step draft count is outside the padded verifier shape");
            if (step.target_rows != step.draft_count + 1)
                return fail("MTP batched spec-state publication step target rows do not match draft count");
            if (step.accepted_count < 0 || step.accepted_count > step.draft_count)
                return fail("MTP batched spec-state publication step accepted count is outside draft count");
            if (step.target_cached_tokens != step.base_cached_tokens + step.accepted_count)
                return fail("MTP batched spec-state publication step target cache count drifted from base plus accepted");

            max_draft_count = std::max(max_draft_count, step.draft_count);
            any_shifted_kv_publication =
                any_shifted_kv_publication || step.publish_mtp_shifted_kv;
            any_correction = any_correction || step.requiresCorrectionReplay();
            if (step.accepted_count > 0)
            {
                restore_rows[static_cast<size_t>(step.request_index)] =
                    step.request_index * padded_seq_len + step.accepted_count - 1;
            }
        }
        for (bool seen : seen_request)
        {
            if (!seen)
                return fail("MTP batched spec-state publication plan is missing a request index");
        }
        if (padded_seq_len != max_draft_count)
        {
            std::ostringstream msg;
            msg << "MTP batched spec-state publication verifier padded length mismatch: plan="
                << max_draft_count << " graph=" << padded_seq_len;
            return fail(msg.str());
        }
        if (any_shifted_kv_publication && plans.request_count > 1)
        {
            for (const MTPSpecStepPlan &step : plans.steps)
            {
                if (step.accepted_count <= 0)
                {
                    return fail(
                        "MTP batched spec-state publication with shifted sidecar KV requires every request to publish an accepted terminal hidden row");
                }
            }
        }


        void *stream = verifier_graph->stream;
        if (!state_.kv_cache)
            return fail("MTP batched spec-state publication requires an initialized main KV cache");

        int restored_stage_total = 0;
        int skipped_stage_total = 0;
        std::vector<int> terminal_rows;
        terminal_rows.resize(
            any_shifted_kv_publication ? static_cast<size_t>(plans.request_count) : 0u,
            -1);

        {
            PerfStatsCollector::ScopedTimer timer(
                "mtp",
                "spec_state_batch_publish_kv_cpu",
                "decode");
            for (const MTPSpecStepPlan &step : plans.steps)
            {
                std::vector<IKVCache *> mtp_caches;
                if (step.publish_mtp_shifted_kv)
                {
                    mtp_caches.reserve(state_.mtp_kv_caches.size());
                    for (const auto &cache : state_.mtp_kv_caches)
                    {
                        if (!cache)
                            return fail("MTP batched spec-state publication requires initialized MTP KV caches");
                        mtp_caches.push_back(cache.get());
                    }
                }

                MTPSpecKVPublicationResult kv_result =
                    publishAcceptedMTPSpecKVState(
                        step,
                        *state_.kv_cache,
                        mtp_caches,
                        step.request_index,
                        stream);
                if (!kv_result.ok)
                    return fail(kv_result.error);

                const int restore_row = restore_rows[static_cast<size_t>(step.request_index)];
                state_.positions[static_cast<size_t>(step.request_index)] =
                    step.target_cached_tokens;
                state_.sequence_lengths[static_cast<size_t>(step.request_index)] =
                    step.target_cached_tokens;
                if (step.publish_mtp_shifted_kv && step.accepted_count > 0)
                {
                    terminal_rows[static_cast<size_t>(step.request_index)] =
                        restore_row;
                }
            }
        }

        /*
         * Stateful CPU stages own one live bank per request.  Publish every
         * accepted verifier row in one stage call after request-local KV
         * truncation is complete; scalar publication would overwrite request
         * zero repeatedly and clear the capture binding before later lanes.
         */
        MTPSpecStatePublicationResult state_result;
        {
            PerfStatsCollector::ScopedTimer timer(
                "mtp",
                "spec_state_batch_publish_recurrent_state_cpu",
                "decode");
            state_result = publishAcceptedMTPSpecStateFromVerifierRows(
                plans,
                restore_rows.data(),
                *verifier_graph->graph,
                state_.device_id,
                stream,
                /*require_captured_stage=*/mtpSpecStatePublicationRequiresCapturedStage());
        }
        if (!state_result.ok)
            return fail(state_result.error);
        restored_stage_total += state_result.restored_stage_count;
        skipped_stage_total += state_result.skipped_stage_count;

        if (!terminal_rows.empty())
        {
            PerfStatsCollector::ScopedTimer timer(
                "mtp",
                "spec_state_batch_publish_terminal_hidden_cpu",
                "decode");
            for (int row : terminal_rows)
            {
                if (row < 0)
                    return fail("MTP batched spec-state publication could not resolve every terminal hidden row");
            }
            const int total_graph_rows =
                verifier_graph->signature.seq_len *
                verifier_graph->signature.batch_size;
            if (!selectMTPTerminalHiddenRows(terminal_rows, total_graph_rows, stream))
            {
                return fail("MTP batched spec-state publication could not restore accepted terminal hidden rows");
            }
        }

        // Request zero is the public scalar main-logit authority. The batch
        // retains explicit request-row consumers for its other participants.
        if (restore_rows.front() >= 0 && !publishCPUAcceptedVerifierLogits(
                PrefixLogitsRowSelection(static_cast<size_t>(restore_rows.front()),
                    static_cast<size_t>(padded_seq_len) * plans.request_count, session_epoch_)))
            return fail("MTP batched spec-state publication could not publish accepted terminal logits");

        const LivePrefixMutationReason mutation_reason =
            any_correction
                ? LivePrefixMutationReason::RejectedCorrection
                : LivePrefixMutationReason::AcceptedSpecPublication;
        {
            PerfStatsCollector::ScopedTimer timer(
                "mtp",
                "spec_state_batch_publish_replay_state_cpu",
                "decode");
            handleLivePrefixReplayStateAfterMutation(
                mutation_reason,
                "mtp_spec_state_publication_batch");
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "spec_state_batch_publications",
            1.0,
            "decode",
            state_.device_id.toString(),
            {{"requests", std::to_string(plans.request_count)},
             {"padded_seq_len", std::to_string(padded_seq_len)},
		             {"publishes_mtp_shifted_kv", any_shifted_kv_publication ? "true" : "false"},
	             {"restored_stages", std::to_string(restored_stage_total)},
	             {"skipped_stages", std::to_string(skipped_stage_total)}});
        if (forward_engine_)
            forward_engine_->clearLastAllPositionVerifierForwardGraph();
        return true;
    }

    /**
     * @brief Publish accepted rows for the grouped decode-equivalent MTP lane.
     *
     * This wrapper deliberately reuses publishAcceptedMTPSpecStateBatch() for
     * the actual KV/GDN/short-conv/terminal-hidden work.  Reuse keeps direct
     * and grouped publication from drifting, while the scoped flag below makes
     * the entry point narrow: external direct-publisher calls still require
     * supportsMTPSpecStatePublication().
     */
    bool DeviceGraphOrchestrator::publishGroupedDecodeEquivalentMTPSpecStateBatch(
        const MTPSpecStepPlanBatch &plans,
        std::string *error)
    {
        auto fail = [&](const std::string &reason) -> bool
        {
            if (error)
                *error = reason;
            LOG_ERROR("[DeviceGraphOrchestrator] " << reason);
            return false;
        };

        if (!state_.device_id.is_gpu())
        {
            if (static_cast<int>(
                    mtp_committed_verifier_identities_host_.size()) <
                plans.request_count)
            {
                return fail(
                    "Grouped CPU MTP publication has no persistent verifier-identity slot for every request");
            }
            for (const MTPSpecStepPlan &step : plans.steps)
            {
                if (step.request_index < 0 ||
                    step.request_index >= plans.request_count ||
                    !step.verifier_input_identity ||
                    !step.verifier_input_identity->valid() ||
                    step.verifier_input_identity->draft_depth !=
                        step.draft_count - 1 ||
                    step.verifier_input_identity->draft_depth >
                        mtp_max_draft_depth_)
                {
                    return fail(
                        "Grouped CPU MTP publication requires the exact typed verifier-input identity for every request");
                }
                const auto &current =
                    mtp_committed_verifier_identities_host_[
                        static_cast<size_t>(step.request_index)];
                if (current.valid != 0 &&
                    current.transaction_count ==
                        std::numeric_limits<int32_t>::max())
                {
                    return fail(
                        "Grouped CPU MTP verifier transaction count overflowed");
                }
            }
        }

        /*
         * The scope object is intentionally tiny and local.  It flips one
         * boolean for the duration of this call so the shared batch publisher
         * can pass the direct all-position capability check, then restores the
         * previous value even if publication exits through an early failure.
         * The shared publisher still validates the real structural contract:
         * MTP graph, cached verifier graph, KV caches, captured state, streams,
         * and the exact accepted-row step plan.
         */
        struct ScopedGroupedDecodeEquivalentPublication
        {
            bool &slot;
            bool previous;

            explicit ScopedGroupedDecodeEquivalentPublication(bool &slot_in)
                : slot(slot_in), previous(slot_in)
            {
                slot = true;
            }

            ~ScopedGroupedDecodeEquivalentPublication()
            {
                slot = previous;
            }
        } scope(grouped_decode_equivalent_spec_publication_scope_);

        bool ok = publishAcceptedMTPSpecStateBatch(plans, error);
        if (ok)
        {
            if (!state_.device_id.is_gpu())
            {
                for (const MTPSpecStepPlan &step : plans.steps)
                {
                    auto &record =
                        mtp_committed_verifier_identities_host_[
                            static_cast<size_t>(step.request_index)];
                    const int next_transaction_count =
                        record.valid != 0
                            ? record.transaction_count + 1
                            : 1;
                    const MTPSpecVerifierInputIdentity &identity =
                        *step.verifier_input_identity;
                    if (!sampling_math::
                            materialize_committed_verifier_identity(
                                identity.verifier_input_tokens.data(),
                                static_cast<int>(
                                    identity.verifier_input_tokens.size()),
                                next_transaction_count,
                                identity.draft_depth,
                                &record))
                    {
                        throw std::runtime_error(
                            "Validated CPU MTP verifier identity failed during committed publication");
                    }
                }
            }
            PerfStatsCollector::addCounter(
                "mtp",
                "grouped_decode_equivalent_spec_state_publications",
                1.0,
                "decode",
                state_.device_id.toString(),
                {{"requests", std::to_string(plans.request_count)}});
        }
        return ok;
    }

    bool DeviceGraphOrchestrator::prepareDeviceResidentMTPSpecPublicationMetadata(
        const DeviceSpeculativePublicationRequest &request,
        std::string *error)
    {
        auto fail = [&](const std::string &reason) -> bool
        {
            if (error)
                *error = reason;
            LOG_ERROR("[DeviceGraphOrchestrator] " << reason);
            return false;
        };

        if (!request.valid())
            return fail("device-resident MTP publication request is invalid");
        if (!state_.device_id.is_gpu())
            return fail("device-resident MTP publication requires a GPU runner");
        if (request.outcome.device != state_.device_id)
            return fail("device-resident MTP publication device does not match runner device");
        if (!request.outcome.stream)
            return fail("device-resident MTP publication requires the exact compact-outcome stream");
        if (!debugEnv().execution.gpu_graphs)
            return fail("device-resident MTP publication requires GPU graph capture");

        if (!forward_engine_)
            return fail("device-resident MTP publication requires an initialized forward engine");
        auto verifier_graph =
            forward_engine_->lastAllPositionVerifierForwardGraph();
        if (!verifier_graph || !*verifier_graph ||
            !verifier_graph->graph)
        {
            return fail(
                "device-resident MTP publication has no cached verifier graph");
        }
        if (!verifier_graph->is_decode ||
            !verifier_graph->all_position_logits)
        {
            return fail(
                "device-resident MTP publication requires the last forward to be an all-position verifier graph");
        }
        if (verifier_graph->signature.batch_size !=
            request.requestCount())
        {
            return fail(
                "device-resident MTP publication outcome request count does not "
                "match its retained producer graph: outcome=" +
                std::to_string(request.requestCount()) +
                " graph=" +
                std::to_string(verifier_graph->signature.batch_size));
        }
        std::string width_policy_error;
        const auto width_policy =
            mtpVerifierPhysicalWidthPolicyForRequest(
                request.requestCount(),
                &width_policy_error);
        const int expected_physical_rows =
            width_policy
                ? mtpVerifierPhysicalPaddedSeqLen(
                      request.requestCount(),
                      request.logicalVerifierRowsPerRequest(),
                      mtp_max_verifier_rows_,
                      *width_policy)
                : 0;
        if (expected_physical_rows <= 0 ||
            request.physicalVerifierRowsPerRequest() != expected_physical_rows)
        {
            return fail(
                "device-resident MTP publication outcome violates the canonical "
                "logical-to-physical row policy: logical=" +
                std::to_string(request.logicalVerifierRowsPerRequest()) +
                " outcome_physical=" +
                std::to_string(request.physicalVerifierRowsPerRequest()) +
                " expected_physical=" +
                std::to_string(expected_physical_rows) +
                " policy_error=" + width_policy_error);
        }
        if (verifier_graph->signature.seq_len !=
            request.physicalVerifierRowsPerRequest())
        {
            return fail(
                "device-resident MTP publication physical verifier stride does "
                "not match its retained producer graph: outcome=" +
                std::to_string(request.physicalVerifierRowsPerRequest()) +
                " graph=" +
                std::to_string(verifier_graph->signature.seq_len));
        }

        if (!state_.kv_cache ||
            !state_.kv_cache
                 ->supportsDeviceResidentSequenceStatePublication() ||
            !supportsDeviceResidentLogicalSequenceStatePublication())
        {
            return fail(
                "device-resident MTP publication requires primary KV and logical-state device publishers");
        }
        if (!getBackendFor(state_.device_id))
            return fail("device-resident MTP publication could not resolve its GPU backend");

        /*
         * Workspace geometry is a graph-family setup responsibility. Allocation
         * or rebinding here would invalidate verifier/capture pointers inside the
         * decode hot path, so an uncovered request is a construction error.
         */
        const MTPSpecDecodeMetadataShape current_shape =
            mtp_spec_decode_metadata_binding_.capacity();
        if (!mtp_spec_decode_metadata_binding_.hasWorkspace() ||
            current_shape.max_requests < request.requestCount() ||
            current_shape.max_draft_tokens <
                request.physicalVerifierRowsPerRequest())
        {
            return fail(
                "device-resident MTP publication metadata workspace was not "
                "pre-materialized for request geometry: has_workspace=" +
                std::to_string(
                    mtp_spec_decode_metadata_binding_.hasWorkspace() ? 1 : 0) +
                " capacity_requests=" +
                std::to_string(current_shape.max_requests) +
                " capacity_rows=" +
                std::to_string(current_shape.max_draft_tokens) +
                " outcome_requests=" +
                std::to_string(request.requestCount()) +
                " outcome_logical_rows=" +
                std::to_string(request.logicalVerifierRowsPerRequest()) +
                " outcome_physical_rows=" +
                std::to_string(request.physicalVerifierRowsPerRequest()) +
                " binding_error=" +
                mtp_spec_decode_metadata_binding_.bindingError());
        }

        const MTPSpecDecodeMetadataDevicePointers &workspace_ptrs =
            mtp_spec_decode_metadata_binding_.devicePointers();
        MTPSpecDecodeMetadataDevicePointers publication_ptrs =
            workspace_ptrs;
        if (!workspace_ptrs.base_cached_tokens ||
            !workspace_ptrs.shifted_target_cached_tokens ||
            !workspace_ptrs.shifted_accepted_state_counts ||
            !workspace_ptrs.accepted_state_slot_indices ||
            !device_resident_logical_sequence_state_storage_
                 .bindPublicationOutputs(
                     &publication_ptrs,
                     request.requestCount()))
        {
            return fail(
                "device-resident MTP publication is missing persistent scratch or logical-state outputs");
        }

        if (!mtp_publication_base_cache_snapshot_ready_ ||
            mtp_publication_base_cache_snapshot_request_count_ <
                request.requestCount())
        {
            return fail(
                "device-resident MTP publication requires the pre-verifier device base-cache snapshot");
        }
        if (!mtp_publication_main_kv_base_checkpoints_ready_ ||
            mtp_publication_main_kv_base_checkpoint_request_count_ <
                request.requestCount() ||
            static_cast<int>(
                mtp_publication_main_kv_base_checkpoints_.size()) <
                request.requestCount())
        {
            return fail(
                "device-resident MTP publication requires one current main-KV checkpoint per request");
        }

        if (request.outcome.device_generation_controller_owned &&
            (!device_generation_storage_.validFor(
                 request.requestCount()) ||
             device_generation_storage_.active_request_count !=
                 request.requestCount() ||
             !device_generation_state_ready_.handoff.borrowed()))
        {
            return fail(
                "device-resident stochastic publication does not own an exclusive consumed generation controller");
        }

        /*
         * All checks above are side-effect free. Validate the producer lease and
         * join its latest mutation event before materializing a stage that freezes
         * shifted-cache bindings. A stale lease must not replace a valid captured
         * publication family.
         *
         * The transaction wait only enqueues an event dependency on the compact
         * outcome stream; it neither observes nor synchronizes the device and it
         * does not open an exclusive lifecycle transaction.
         */
        if (!waitForDeviceResidentMTPTransaction(
                request.outcome.mtp_transaction,
                request.outcome.stream,
                "mtp_spec_publication_metadata"))
        {
            return fail(
                "device-resident MTP publication received a stale or foreign transaction lease");
        }

        if (!materializeMTPSpeculativeStatePublicationGraph(
                request,
                publication_ptrs,
                *verifier_graph->graph,
                verifier_graph->signature,
                verifier_graph->signature.seq_len,
                error))
        {
            return false;
        }

        // Finish the exact cache's cold capture before opening any replacement
        // writer. A prelaunch capture failure leaves the original logical-state
        // mailbox and access epoch intact; no device publication is invented.
        if (!executeMTPStatePublicationGraph(
                request.outcome.stream, error,
                DeviceGraphExecutor::GraphInitialSubmissionPolicy::MaterializeWithoutLaunch))
        {
            return false;
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "device_resident_publication_metadata_prepares",
            1.0,
            "decode",
            state_.device_id.toString(),
            {{"requests", std::to_string(request.requestCount())},
             {"logical_verifier_rows",
              std::to_string(request.logicalVerifierRowsPerRequest())},
             {"physical_verifier_rows",
              std::to_string(request.physicalVerifierRowsPerRequest())},
             {"operation", "capture_preflight_only"}});
        return true;
    }

    std::vector<ForwardExecutionEngine::ReplayCacheObservation>
    DeviceGraphOrchestrator::forwardReplayCacheObservations() const
    {
        if (!forward_engine_)
        {
            return {};
        }
        return forward_engine_->replayCacheObservations(live_replay_state_epoch_);
    }

    bool DeviceGraphOrchestrator::commitMTPShiftedRowsFromLastForward(
        const int32_t *tokens,
        int token_count,
        int already_appended_tokens)
    {
        return commitMTPShiftedRowsFromPartialForward(
            tokens,
            token_count,
            already_appended_tokens,
            token_count);
    }

    bool DeviceGraphOrchestrator::commitMTPShiftedRowFromCurrentTerminalHidden(
        int32_t token,
        int already_appended_tokens,
        bool allow_speculative_discard,
        int position_offset_override)
    {
        if (!graph_builder_ || !graph_builder_->config().mtp.enabled)
            return true;
        if (already_appended_tokens < 0)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Sequential MTP shifted-row commit received negative already_appended_tokens");
            return false;
        }
        if (state_.positions.empty())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Sequential MTP shifted-row commit requires position state");
            return false;
        }

        const int position_offset =
            position_offset_override >= 0
                ? position_offset_override
                : state_.positions[0] - already_appended_tokens;
        if (position_offset < 0)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Invalid sequential MTP shifted-row commit position_offset="
                      << position_offset << " already_appended=" << already_appended_tokens
                      << " override=" << position_offset_override);
            return false;
        }

        IKVCache *cache = state_.mtp_kv_caches.empty() ? nullptr : state_.mtp_kv_caches[0].get();
        if (!cache)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Sequential MTP shifted-row commit requires an initialized MTP KV cache");
            return false;
        }

        void *stream = nullptr;
        DeviceResidentMTPTransactionLease transaction;
        if (state_.device_id.is_gpu())
        {
            stream = explicitGPUStreamForOperation("commitMTPShiftedRowFromCurrentTerminalHidden");
            if (!stream)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Sequential MTP shifted-row commit requires an explicit GPU stream");
                return false;
            }
            /*
             * Metadata admission observes the sidecar but does not take its
             * write frontier. The eventual mailbox writer or sidecar executes
             * on another stream and must still acquire this same publication.
             * Consuming it here would strand the wait on the metadata stream.
             */
            if (!waitForPendingShiftedMTPKVReadyForObservation(
                    stream,
                    "shifted_row_sequential_metadata"))
            {
                return false;
            }
            transaction = currentDeviceResidentMTPTransactionLease();
            if (!transaction.valid())
            {
                if (!recordDeviceResidentMTPTransactionMutation(
                        /*request_count=*/1,
                        stream,
                        "sequential_shifted_row_commit_admission"))
                {
                    return false;
                }
                transaction = currentDeviceResidentMTPTransactionLease();
            }
        }

        const int expected_cached_tokens =
            mtpShiftedAppendBoundaryRows(position_offset,already_appended_tokens,cache->stateOwnership().shifted_origin);
        if (!prepareShiftedMTPKVCommitBoundary(
                *cache,
                /*depth=*/0,
                /*request_index=*/0,
                expected_cached_tokens,
                allow_speculative_discard,
                stream,
                state_.device_id.is_gpu() ? &transaction : nullptr,
                "sequential_shifted_row_commit"))
        {
            return false;
        }

        TensorBase *terminal_hidden = nullptr;
        BufferId terminal_hidden_buffer_id = BufferId::HIDDEN_STATE;
        if (!resolveMTPTerminalHiddenInput(
                &terminal_hidden,
                &terminal_hidden_buffer_id,
                "commitMTPShiftedRowFromCurrentTerminalHidden"))
            return false;

        PerfStatsCollector::ScopedTimer timer(
            "mtp",
            "shifted_row_sequential_commit",
            perfPhaseName(),
            state_.device_id.toString());
        const uint64_t workspace_generation_before_commit =
            state_.device_id.is_gpu() ? workspaceGeneration(state_.device_id) : 0;

        if (!executeMTPDepth0(token,
                              terminal_hidden,
                              position_offset + already_appended_tokens,
                              kMTPDecodeCatchupContext,
                              true,
                              terminal_hidden_buffer_id,
                              /*defer_final_sync=*/state_.device_id.is_gpu()))
        {
            return false;
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "shifted_rows_committed",
            1.0,
            perfPhaseName(),
            state_.device_id.toString());
        PerfStatsCollector::addCounter(
            "mtp",
            "shifted_row_sequential_commits",
            1.0,
            perfPhaseName(),
            state_.device_id.toString(),
            {{"already_appended", std::to_string(already_appended_tokens)},
             {"rebuilt_first_row", already_appended_tokens == 0 ? "true" : "false"}});

        if (state_.device_id.is_gpu())
        {
            const uint64_t workspace_generation_after_commit =
                workspaceGeneration(state_.device_id);
            if (workspace_generation_after_commit != workspace_generation_before_commit)
            {
                handleLivePrefixReplayStateAfterMutation(
                    LivePrefixMutationReason::Unknown,
                    "mtp_shifted_row_sequential_commit_workspace_rebind");
            }
        }
        return true;
    }

    bool DeviceGraphOrchestrator::commitMTPShiftedRowFromCheckpointTerminalHidden(
        const PrefixStateSnapshot &checkpoint,
        int32_t token,
        int already_appended_tokens,
        bool allow_speculative_discard,
        int position_offset_override)
    {
        if (!graph_builder_ || !graph_builder_->config().mtp.enabled)
            return true;
        if (!checkpoint.valid)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Checkpoint-backed MTP shifted-row commit requires a valid checkpoint");
            return false;
        }
        const int position_offset =
            position_offset_override >= 0
                ? position_offset_override
                : checkpoint.cached_tokens;
        if (position_offset != checkpoint.cached_tokens)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Checkpoint-backed MTP shifted-row commit anchor mismatch: checkpoint_cached="
                      << checkpoint.cached_tokens
                      << " position_offset=" << position_offset
                      << " token=" << token);
            return false;
        }

        void *stream = nullptr;
        if (state_.device_id.is_gpu())
        {
            stream = explicitGPUStreamForOperation(
                "commitMTPShiftedRowFromCheckpointTerminalHidden");
            if (!stream)
                return false;
        }
        if (!importMTPCheckpointTerminalHidden(
                checkpoint,
                stream,
                "commit_mtp_shifted_row_checkpoint_terminal_hidden"))
        {
            return false;
        }

        PerfStatsCollector::addCounterWithSequence(
            "mtp",
            "checkpoint_terminal_hidden_shifted_commits",
            1.0,
            {static_cast<uint64_t>(checkpoint.cached_tokens)},
            perfPhaseName(),
            state_.device_id.toString(),
            {{"already_appended", std::to_string(already_appended_tokens)}});

        /*
         * The import above made PREFIX_TERMINAL_HIDDEN current for exactly this
         * checkpoint boundary.  Reuse the ordinary append path so shifted-KV
         * truncation, sidecar execution, and replay-state invalidation remain
         * centralized in one implementation.
         */
        return commitMTPShiftedRowFromCurrentTerminalHidden(
            token,
            already_appended_tokens,
            allow_speculative_discard,
            position_offset);
    }

    bool DeviceGraphOrchestrator::commitMTPInitialShiftedRowFromDeviceOutcome(
        const PrefixStateSnapshot &checkpoint,
        const DeviceSpeculativeOutcomeHandle &outcome,
        int request_index,
        bool allow_speculative_discard)
    {
        if (!graph_builder_ || !graph_builder_->config().mtp.enabled)
            return true;
        if (!state_.device_id.is_gpu())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-outcome initial MTP shifted-row commit requires a GPU runner");
            return false;
        }
        if (!supportsMTPDeviceDraftTokenInput())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-outcome initial MTP shifted-row commit requires device-token sidecar input support");
            return false;
        }
        if (!checkpoint.valid)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-outcome initial MTP shifted-row commit requires a valid verifier-base checkpoint");
            return false;
        }
        if (!outcome.valid() || outcome.device != state_.device_id ||
            request_index < 0 || request_index >= outcome.request_count)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-outcome initial MTP shifted-row commit received an invalid or foreign outcome handle");
            return false;
        }

        const MTPSpecDecodeMetadataDevicePointers &metadata =
            mtp_spec_decode_metadata_binding_.devicePointers();
        if (!mtp_publication_base_cache_snapshot_ready_ ||
            request_index >= mtp_publication_base_cache_snapshot_request_count_ ||
            !metadata.base_cached_tokens)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-outcome initial MTP shifted-row commit requires the resident pre-verifier base-position snapshot"
                      << " request=" << request_index
                      << " snapshot_requests="
                      << mtp_publication_base_cache_snapshot_request_count_);
            return false;
        }

        IKVCache *cache = state_.mtp_kv_caches.empty() ? nullptr : state_.mtp_kv_caches[0].get();
        if (!cache)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-outcome initial MTP shifted-row commit requires an initialized MTP KV cache");
            return false;
        }

        void *stream =
            explicitGPUStreamForOperation("commitMTPInitialShiftedRowFromDeviceOutcome");
        if (!stream)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-outcome initial MTP shifted-row commit requires an explicit GPU stream");
            return false;
        }
        if (!waitForPendingShiftedMTPKVReadyForObservation(
                stream,
                "shifted_row_initial_device_outcome_metadata"))
        {
            return false;
        }

        if (!prepareShiftedMTPKVCommitBoundary(
                *cache,
                /*depth=*/0,
                /*request_index=*/0,
                /*expected_cached_tokens=*/-1,
                allow_speculative_discard,
                stream,
                &outcome.mtp_transaction,
                "initial_shifted_row_device_outcome_commit"))
        {
            return false;
        }

        if (!importMTPCheckpointTerminalHidden(
                checkpoint,
                stream,
                "commit_mtp_initial_shifted_row_device_outcome_checkpoint_terminal_hidden"))
        {
            return false;
        }

        PerfStatsCollector::ScopedTimer timer(
            "mtp",
            "initial_shifted_row_device_outcome_commit",
            perfPhaseName(),
            state_.device_id.toString(),
            {{"request_index", std::to_string(request_index)}});
        const uint64_t workspace_generation_before_commit =
            workspaceGeneration(state_.device_id);

        if (!executeMTPDepth0Batched(
                /*draft_condition_tokens=*/nullptr,
                /*token_count=*/1,
                state_.prefix_terminal_hidden.get(),
                /*position_id=*/0,
                kMTPDecodeCatchupContext,
                /*kv_cache_only=*/true,
                BufferId::PREFIX_TERMINAL_HIDDEN,
                /*defer_final_sync=*/true,
                /*draft_condition_tokens_device=*/nullptr,
                /*draft_condition_ready_slot=*/-1,
                /*draft_condition_ready_is_target=*/false,
                /*request_batch=*/1,
                /*position_ids_override=*/nullptr,
                metadata.base_cached_tokens,
                outcome.meta_device,
                outcome.meta_stride,
                outcome.output_tokens_device,
                outcome.output_token_stride,
                request_index,
                /*speculative_first_output_token_index=*/0,
                outcome.response_ready_event.get(),
                /*draft_condition_token_stride=*/1,
                /*draft_condition_ready_count=*/1,
                /*draft_condition_ready_stride=*/1,
                /*device_position_offset=*/0))
        {
            return false;
        }
        // The checkpoint row was scratch input for the just-submitted sidecar.
        // Its read lease and shifted-KV completion event were published by
        // executeMTPDepth0Batched; invalidating semantic currency does not drop
        // those event edges or release storage. The suffix selects its own
        // verifier rows, and captured accepted-state publication produces the
        // sole final terminal row. A prefill refresh here used the wrong length
        // owner and added a redundant captured copy between those transactions.
        state_.mtp_terminal_hidden_publication.invalidate();

        PerfStatsCollector::addCounter(
            "mtp",
            "initial_shifted_rows_device_outcome_committed",
            1.0,
            perfPhaseName(),
            state_.device_id.toString(),
            {{"request_index", std::to_string(request_index)},
             {"position_owner", "resident_pre_verifier_snapshot"}});

        const uint64_t workspace_generation_after_commit =
            workspaceGeneration(state_.device_id);
        if (workspace_generation_after_commit != workspace_generation_before_commit)
        {
            handleLivePrefixReplayStateAfterMutation(
                LivePrefixMutationReason::Unknown,
                "mtp_initial_shifted_row_device_outcome_commit_workspace_rebind");
        }
        return true;
    }

    bool DeviceGraphOrchestrator::commitMTPShiftedRowFromDeviceTargetSample(
        int target_sample_slot,
        int already_appended_tokens,
        bool allow_speculative_discard)
    {
        if (!graph_builder_ || !graph_builder_->config().mtp.enabled)
            return true;
        if (!state_.device_id.is_gpu())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-target MTP shifted-row commit requires a GPU runner");
            return false;
        }
        if (!supportsMTPDeviceDraftTokenInput() || !stochastic_target_sample_tokens_dev_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-target MTP shifted-row commit requires device sample-token support");
            return false;
        }
        if (target_sample_slot < 0 ||
            target_sample_slot >= stochastic_target_row_capacity_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-target MTP shifted-row commit received invalid target slot="
                      << target_sample_slot);
            return false;
        }
        if (already_appended_tokens < 0)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-target MTP shifted-row commit received negative already_appended_tokens");
            return false;
        }
        IKVCache *cache = state_.mtp_kv_caches.empty() ? nullptr : state_.mtp_kv_caches[0].get();
        if (!cache)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-target MTP shifted-row commit requires an initialized MTP KV cache");
            return false;
        }

        void *stream = explicitGPUStreamForOperation("commitMTPShiftedRowFromDeviceTargetSample");
        if (!stream)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-target MTP shifted-row commit requires an explicit GPU stream");
            return false;
        }
        if (!waitForPendingShiftedMTPKVReadyForObservation(
                stream,
                "shifted_row_device_target_metadata"))
        {
            return false;
        }

        DeviceResidentMTPTransactionLease transaction =
            currentDeviceResidentMTPTransactionLease();
        if (!transaction.valid())
        {
            if (!recordDeviceResidentMTPTransactionMutation(
                    /*request_count=*/1,
                    stream,
                    "shifted_row_device_target_commit_admission"))
            {
                return false;
            }
            transaction = currentDeviceResidentMTPTransactionLease();
        }

        if (!prepareShiftedMTPKVCommitBoundary(
                *cache,
                /*depth=*/0,
                /*request_index=*/0,
                /*expected_cached_tokens=*/-1,
                allow_speculative_discard,
                stream,
                &transaction,
                "shifted_row_device_target_commit"))
        {
            return false;
        }

        TensorBase *terminal_hidden = nullptr;
        BufferId terminal_hidden_buffer_id = BufferId::HIDDEN_STATE;
        if (!resolveMTPTerminalHiddenInput(
                &terminal_hidden,
                &terminal_hidden_buffer_id,
                "commitMTPShiftedRowFromDeviceTargetSample"))
            return false;

        PerfStatsCollector::ScopedTimer timer(
            "mtp",
            "shifted_row_device_target_commit",
            perfPhaseName(),
            state_.device_id.toString());
        const uint64_t workspace_generation_before_commit =
            workspaceGeneration(state_.device_id);

        const int *target_token_device =
            static_cast<const int *>(stochastic_target_sample_tokens_dev_) +
            target_sample_slot;
        const int32_t *shifted_cached_tokens_device =
            transaction.state->cachedTokensForDepth(/*depth=*/0);
        if (!shifted_cached_tokens_device)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-target MTP shifted-row commit could not resolve the canonical shifted-cache count");
            return false;
        }

        const int shifted_position_offset =
            mtpShiftedPrefixOffset(graph_builder_->config().mtpShiftedCacheOrigin());
        if (shifted_position_offset < 0)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-target MTP shifted-row commit has an invalid cache origin");
            return false;
        }
        // The input normalizer converts the authoritative shifted cache count
        // into the RoPE position using the model's origin. QFN already counts
        // its visible position-zero row, so that model uses no displacement.
        if (!executeMTPDepth0Batched(
                /*draft_condition_tokens=*/nullptr,
                /*token_count=*/1,
                terminal_hidden,
                /*position_id=*/0,
                kMTPDecodeCatchupContext,
                /*kv_cache_only=*/true,
                terminal_hidden_buffer_id,
                /*defer_final_sync=*/true,
                target_token_device,
                target_sample_slot,
                /*draft_condition_ready_is_target=*/true,
                /*request_batch=*/1,
                /*position_ids_override=*/nullptr,
                shifted_cached_tokens_device,
                /*speculative_outcome_meta_device=*/nullptr,
                /*speculative_outcome_meta_stride=*/0,
                /*speculative_outcome_output_tokens_device=*/nullptr,
                /*speculative_outcome_output_token_stride=*/0,
                /*speculative_outcome_request_index=*/-1,
                /*speculative_first_output_token_index=*/0,
                /*speculative_outcome_ready_event=*/nullptr,
                /*draft_condition_token_stride=*/1,
                /*draft_condition_ready_count=*/1,
                /*draft_condition_ready_stride=*/1,
                /*device_position_offset=*/shifted_position_offset))
        {
            return false;
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "shifted_rows_committed",
            1.0,
            perfPhaseName(),
            state_.device_id.toString());
        PerfStatsCollector::addCounter(
            "mtp",
            "shifted_row_device_target_commits",
            1.0,
            perfPhaseName(),
            state_.device_id.toString(),
            {{"target_slot", std::to_string(target_sample_slot)},
             {"already_appended", std::to_string(already_appended_tokens)},
             {"rebuilt_first_row", already_appended_tokens == 0 ? "true" : "false"}});

        const uint64_t workspace_generation_after_commit =
            workspaceGeneration(state_.device_id);
        if (workspace_generation_after_commit != workspace_generation_before_commit)
        {
            handleLivePrefixReplayStateAfterMutation(
                LivePrefixMutationReason::Unknown,
                "mtp_shifted_row_device_target_commit_workspace_rebind");
        }
        return true;
    }

    bool DeviceGraphOrchestrator::commitMTPShiftedRowFromDeviceResidentLogicalState(
        const DeviceResidentLogicalSequenceStateHandle &logical_state,
        int request_index,
        int already_appended_tokens,
        bool allow_speculative_discard)
    {
        if (!graph_builder_ || !graph_builder_->config().mtp.enabled)
            return true;
        if (!state_.device_id.is_gpu())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Resident logical-state MTP shifted-row commit requires a GPU runner");
            return false;
        }
        if (!supportsMTPDeviceDraftTokenInput())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Resident logical-state MTP shifted-row commit requires device-token sidecar input support");
            return false;
        }
        if (!logical_state.coversRequest(request_index) ||
            !device_resident_logical_sequence_state_mailbox_.ownsHandle(
                logical_state,
                live_replay_state_epoch_))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Resident logical-state MTP shifted-row commit received a stale or foreign mailbox handle");
            return false;
        }
        if (already_appended_tokens < 0)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Resident logical-state MTP shifted-row commit received negative already_appended_tokens");
            return false;
        }

        const int32_t *next_condition_token_device =
            logical_state.nextConditionTokenDeviceForRequest(request_index);
        const int32_t *target_position_device =
            logical_state.targetPositionDeviceForRequest(request_index);
        if (!next_condition_token_device || !target_position_device)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Resident logical-state MTP shifted-row commit has incomplete request-local device metadata"
                      << " request=" << request_index);
            return false;
        }

        IKVCache *cache = state_.mtp_kv_caches.empty() ? nullptr : state_.mtp_kv_caches[0].get();
        if (!cache)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Resident logical-state MTP shifted-row commit requires an initialized MTP KV cache");
            return false;
        }

        void *stream = explicitGPUStreamForOperation("commitMTPShiftedRowFromDeviceResidentLogicalState");
        if (!stream)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Resident logical-state MTP shifted-row commit requires an explicit GPU stream");
            return false;
        }
        DeviceResidentLogicalStateReadScope logical_state_read(
            *this,
            stream,
            "shifted_row_resident_logical_state_commit");
        if (!logical_state_read.ready())
            return false;
        if (!waitForPendingShiftedMTPKVReadyForObservation(
                stream,
                "shifted_row_resident_logical_state_metadata"))
        {
            return false;
        }

        if (!prepareShiftedMTPKVCommitBoundary(
                *cache,
                /*depth=*/0,
                request_index,
                /*expected_cached_tokens=*/-1,
                allow_speculative_discard,
                stream,
                &logical_state.mtp_transaction,
                "shifted_row_resident_logical_state_commit"))
        {
            return false;
        }

        TensorBase *terminal_hidden = nullptr;
        BufferId terminal_hidden_buffer_id = BufferId::HIDDEN_STATE;
        if (!resolveMTPTerminalHiddenInput(
                &terminal_hidden,
                &terminal_hidden_buffer_id,
                "commitMTPShiftedRowFromDeviceResidentLogicalState"))
            return false;
        if (terminal_hidden_buffer_id != BufferId::PREFIX_TERMINAL_HIDDEN)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Resident logical-state MTP shifted-row commit requires the persistent terminal-hidden mailbox");
            return false;
        }
        PerfStatsCollector::ScopedTimer timer(
            "mtp",
            "shifted_row_resident_logical_state_commit",
            perfPhaseName(),
            state_.device_id.toString());
        const uint64_t workspace_generation_before_commit =
            workspaceGeneration(state_.device_id);

        /*
         * Both dynamic inputs remain device-owned.  entry 0 of the sidecar
         * token row comes from next_condition_token_device, while the RoPE/KV
         * position comes from target_position_device.  If an earlier sidecar
         * already appended rows for this transaction, the backend composes the
         * small integer offset into its persistent position row before graph
         * replay; no target position is copied through the CPU.
         */
        if (!executeMTPDepth0Batched(
                /*draft_condition_tokens=*/nullptr,
                /*token_count=*/1,
                terminal_hidden,
                /*position_id=*/0,
                kMTPDecodeCatchupContext,
                /*kv_cache_only=*/true,
                terminal_hidden_buffer_id,
                /*defer_final_sync=*/true,
                next_condition_token_device,
                /*draft_condition_ready_slot=*/-1,
                /*draft_condition_ready_is_target=*/false,
                /*request_batch=*/1,
                /*position_ids_override=*/nullptr,
                target_position_device,
                /*speculative_outcome_meta_device=*/nullptr,
                /*speculative_outcome_meta_stride=*/0,
                /*speculative_outcome_output_tokens_device=*/nullptr,
                /*speculative_outcome_output_token_stride=*/0,
                /*speculative_outcome_request_index=*/0,
                /*speculative_first_output_token_index=*/0,
                /*speculative_outcome_ready_event=*/nullptr,
                /*draft_condition_token_stride=*/1,
                /*draft_condition_ready_count=*/1,
                /*draft_condition_ready_stride=*/1,
                /*device_position_offset=*/already_appended_tokens))
        {
            return false;
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "shifted_rows_committed",
            1.0,
            perfPhaseName(),
            state_.device_id.toString());
        PerfStatsCollector::addCounter(
            "mtp",
            "shifted_row_resident_logical_state_commits",
            1.0,
            perfPhaseName(),
            state_.device_id.toString(),
            {{"request_index", std::to_string(request_index)},
             {"already_appended", std::to_string(already_appended_tokens)}});

        const uint64_t workspace_generation_after_commit =
            workspaceGeneration(state_.device_id);
        if (workspace_generation_after_commit != workspace_generation_before_commit)
        {
            handleLivePrefixReplayStateAfterMutation(
                LivePrefixMutationReason::Unknown,
                "mtp_shifted_row_resident_logical_state_commit_workspace_rebind");
        }
        else
        {
            /*
             * This commit consumes a one-shot logical-state handle. The caller
             * has already preserved the emitted token in a persistent target
             * slot, and the immediately following main graph establishes the
             * next live-state epoch. Carrying the old compact values across
             * this mutation would create a stale mailbox with a fresh epoch.
             */
            if (!logical_state_read.complete())
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Resident logical-state shifted-row commit could not publish reader completion");
                return false;
            }
            clearDeviceResidentLogicalSequenceStateMailbox();
            PerfStatsCollector::addCounter(
                "mtp",
                "device_resident_logical_state_mailbox_consumptions",
                1.0,
                perfPhaseName(),
                state_.device_id.toString(),
                {{"consumer",
                  "mtp_shifted_row_resident_logical_state_commit"},
                 {"next_owner", "persistent_target_slot_then_main_graph"}});
        }
        return true;
    }

    bool DeviceGraphOrchestrator::commitMTPShiftedRowsFromDeviceOutcome(
        const DeviceSpeculativeOutcomeHandle &outcome,
        int request_index,
        int already_appended_tokens,
        int max_state_commit_rows,
        int main_forward_token_count,
        bool allow_speculative_discard)
    {
        if (!graph_builder_ || !graph_builder_->config().mtp.enabled)
            return true;
        if (!state_.device_id.is_gpu())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-outcome MTP shifted-row commit requires a GPU runner");
            return false;
        }
        if (!supportsMTPDeviceDraftTokenInput())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-outcome MTP shifted-row commit requires device-token sidecar input support");
            return false;
        }
        if (!outcome.valid() || outcome.device != state_.device_id ||
            request_index < 0 || request_index >= outcome.request_count)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-outcome MTP shifted-row commit received an invalid or foreign outcome handle");
            return false;
        }
        if (already_appended_tokens < 0)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-outcome MTP shifted-row commit received negative already_appended_tokens");
            return false;
        }
        if (max_state_commit_rows <= already_appended_tokens)
            return true;

        const int catchup_token_count =
            max_state_commit_rows - already_appended_tokens;
        if (catchup_token_count <= 0)
            return true;
        if (catchup_token_count > mtp_max_draft_depth_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-outcome MTP shifted-row commit exceeds graph capacity: "
                      << catchup_token_count);
            return false;
        }

        const int hidden_source_row_start = already_appended_tokens - 1;
        const int hidden_source_row_end =
            hidden_source_row_start + catchup_token_count;
        if (main_forward_token_count <= 0 ||
            hidden_source_row_start < 0 ||
            hidden_source_row_end > main_forward_token_count)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-outcome MTP shifted-row commit received invalid verifier hidden-row bounds: start="
                      << hidden_source_row_start
                      << " end=" << hidden_source_row_end
                      << " main_forward_token_count=" << main_forward_token_count
                      << " max_state_commit_rows=" << max_state_commit_rows);
            return false;
        }
        const MTPSpecDecodeMetadataDevicePointers &metadata =
            mtp_spec_decode_metadata_binding_.devicePointers();
        if (!mtp_publication_base_cache_snapshot_ready_ ||
            request_index >= mtp_publication_base_cache_snapshot_request_count_ ||
            !metadata.base_cached_tokens)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-outcome MTP shifted-row commit requires the resident pre-verifier base-position snapshot"
                      << " request=" << request_index
                      << " snapshot_requests="
                      << mtp_publication_base_cache_snapshot_request_count_);
            return false;
        }

        IKVCache *cache = state_.mtp_kv_caches.empty() ? nullptr : state_.mtp_kv_caches[0].get();
        if (!cache)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-outcome MTP shifted-row commit requires an initialized MTP KV cache");
            return false;
        }

        void *stream =
            explicitGPUStreamForOperation("commitMTPShiftedRowsFromDeviceOutcome");
        if (!stream)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-outcome MTP shifted-row commit requires an explicit GPU stream");
            return false;
        }
        if (!waitForPendingShiftedMTPKVReadyForObservation(
                stream,
                "shifted_row_device_outcome_metadata"))
        {
            return false;
        }

        if (!prepareShiftedMTPKVCommitBoundary(
                *cache,
                /*depth=*/0,
                /*request_index=*/0,
                /*expected_cached_tokens=*/-1,
                allow_speculative_discard,
                stream,
                &outcome.mtp_transaction,
                "shifted_row_device_outcome_commit"))
        {
            return false;
        }

        PerfStatsCollector::ScopedTimer timer(
            "mtp",
            "shifted_row_device_outcome_commit",
            perfPhaseName(),
            state_.device_id.toString(),
            {{"rows", std::to_string(catchup_token_count)},
             {"request_index", std::to_string(request_index)}});
        const uint64_t workspace_generation_before_commit =
            workspaceGeneration(state_.device_id);

        /*
         * Select on the verifier outcome's exact producer stream. The terminal
         * mailbox publishes its row-copy event for the sidecar capture stream,
         * so neither hidden-state provenance nor compact metadata crosses the
         * host merely to establish ordering.
         */
        if (!selectMTPTerminalHiddenRows(
                hidden_source_row_start,
                catchup_token_count,
                main_forward_token_count,
                outcome.stream))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-outcome MTP shifted-row commit failed to select verifier hidden rows "
                      << hidden_source_row_start << ".."
                      << (hidden_source_row_end - 1));
            return false;
        }

        if (!executeMTPDepth0Batched(
                /*draft_condition_tokens=*/nullptr,
                catchup_token_count,
                state_.prefix_terminal_hidden.get(),
                /*position_id=*/0,
                kMTPDecodeCatchupContext,
                /*kv_cache_only=*/true,
                BufferId::PREFIX_TERMINAL_HIDDEN,
                /*defer_final_sync=*/true,
                /*draft_condition_tokens_device=*/nullptr,
                /*draft_condition_ready_slot=*/-1,
                /*draft_condition_ready_is_target=*/false,
                /*request_batch=*/1,
                /*position_ids_override=*/nullptr,
                metadata.base_cached_tokens,
                outcome.meta_device,
                outcome.meta_stride,
                outcome.output_tokens_device,
                outcome.output_token_stride,
                request_index,
                already_appended_tokens,
                outcome.response_ready_event.get(),
                /*draft_condition_token_stride=*/1,
                /*draft_condition_ready_count=*/1,
                /*draft_condition_ready_stride=*/1,
                /*device_position_offset=*/already_appended_tokens))
        {
            return false;
        }
        // These verifier rows were temporary KV-only inputs, not committed
        // terminal state. The existing accepted-state graph selects from its
        // device-owned accepted indices after consuming shifted-KV readiness.
        // Retire the scratch generation so no unrelated reader can borrow it
        // while preserving the persistent mailbox's read-completion event.
        state_.mtp_terminal_hidden_publication.invalidate();

        PerfStatsCollector::addCounter(
            "mtp",
            "shifted_rows_device_outcome_committed",
            static_cast<double>(catchup_token_count),
            perfPhaseName(),
            state_.device_id.toString(),
            {{"request_index", std::to_string(request_index)},
             {"already_appended", std::to_string(already_appended_tokens)},
             {"max_state_commit_rows", std::to_string(max_state_commit_rows)}});

        const uint64_t workspace_generation_after_commit =
            workspaceGeneration(state_.device_id);
        if (workspace_generation_after_commit != workspace_generation_before_commit)
        {
            handleLivePrefixReplayStateAfterMutation(
                LivePrefixMutationReason::Unknown,
                "mtp_shifted_row_device_outcome_commit_workspace_rebind");
        }
        return true;
    }
} // namespace llaminar2
