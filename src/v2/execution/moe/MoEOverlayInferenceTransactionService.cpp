/**
 * @file MoEOverlayInferenceTransactionService.cpp
 * @brief Fixed-slot MPI control transport and remote transaction execution.
 */

#include "MoEOverlayInferenceTransactionService.h"

#include "collective/CollectiveTimeoutPolicy.h"
#include "interfaces/IMPIContext.h"
#include "utils/Logger.h"
#include "utils/PerfStatsCollector.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <stdexcept>

namespace llaminar2
{
    namespace
    {
        /** MPI guarantees at least 32767 as the maximum valid tag. */
        constexpr int kInferenceTransactionTicketTag = 32742;

        /** @brief Store a diagnostic only when requested. */
        bool fail(std::string message, std::string *error)
        {
            if (error)
                *error = std::move(message);
            return false;
        }

        /** @brief Publish one control-plane witness without model-state payloads. */
        void recordTicket(
            const MoEOverlayInferenceTransactionTicket &ticket,
            const char *endpoint,
            const char *result)
        {
            PerfStatsCollector::addCounter(
                "moe_overlay_transaction",
                "control_tickets",
                1.0,
                "inference",
                "mpi",
                {{"action", std::to_string(static_cast<std::uint32_t>(ticket.action))},
                 {"command", std::to_string(ticket.command_id)},
                 {"draft_depth", std::to_string(ticket.draft_depth)},
                 {"endpoint", endpoint},
                 {"logical_step", std::to_string(ticket.logical_step_id)},
                 {"ordinal", std::to_string(ticket.transaction_ordinal)},
                 {"result", result},
                 {"role", std::to_string(static_cast<std::uint32_t>(ticket.graph_role))},
                 {"target_world_rank", std::to_string(ticket.target_world_rank)}});
        }

        /** @brief Attribute one follower control or lifecycle interval. */
        void recordTicketTiming(
            const MoEOverlayInferenceTransactionTicket &ticket,
            const char *name,
            const char *endpoint,
            std::chrono::steady_clock::time_point begin,
            std::chrono::steady_clock::time_point end)
        {
            if (!PerfStatsCollector::isDomainEnabled(
                    "moe_overlay_transaction"))
            {
                return;
            }
            const auto elapsed =
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    end - begin)
                    .count();
            PerfStatsCollector::recordTimingNs(
                "moe_overlay_transaction",
                name,
                static_cast<std::uint64_t>(std::max<std::int64_t>(1, elapsed)),
                "inference",
                "mpi",
                {{"command", std::to_string(ticket.command_id)},
                 {"endpoint", endpoint},
                 {"logical_step", std::to_string(ticket.logical_step_id)},
                 {"ordinal", std::to_string(ticket.transaction_ordinal)},
                 {"role", std::to_string(
                              static_cast<std::uint32_t>(ticket.graph_role))},
                 {"target_world_rank",
                  std::to_string(ticket.target_world_rank)}});
        }
    } // namespace

    MoEOverlayMPIInferenceTransactionChannel::
        MoEOverlayMPIInferenceTransactionChannel(Config config)
        : config_(std::move(config))
    {
        if (!config_.mpi_ctx || config_.source_world_rank < 0 ||
            config_.target_world_rank < 0 ||
            config_.source_world_rank == config_.target_world_rank ||
            config_.source_world_rank >= config_.mpi_ctx->world_size() ||
            config_.target_world_rank >= config_.mpi_ctx->world_size() ||
            (config_.mpi_ctx->rank() != config_.source_world_rank &&
             config_.mpi_ctx->rank() != config_.target_world_rank) ||
            config_.send_slot_count == 0)
        {
            throw std::invalid_argument(
                "ExpertOverlay inference transaction channel requires one valid rank pair and fixed send ring");
        }
        send_slots_.resize(config_.send_slot_count);
    }

    MoEOverlayMPIInferenceTransactionChannel::
        ~MoEOverlayMPIInferenceTransactionChannel()
    {
        drainNoexcept();
    }

    void MoEOverlayMPIInferenceTransactionChannel::progressSendSlots() const
    {
        if (config_.mpi_ctx->rank() != config_.source_world_rank)
            return;
        for (auto &slot : send_slots_)
        {
            if (!slot.in_flight)
                continue;
            if (config_.mpi_ctx->test(&slot.request))
            {
                slot.request = MPI_REQUEST_NULL;
                slot.in_flight = false;
            }
        }
    }

    bool MoEOverlayMPIInferenceTransactionChannel::
        progressRequestToCompletion(
            MPI_Request *request,
            MPI_Status *status,
            const char *operation,
            std::string *error) const
    {
        if (!request || !operation)
            return fail("ExpertOverlay ticket progress request is invalid", error);

        const auto deadline =
            std::chrono::steady_clock::now() +
            std::chrono::milliseconds(
                collective_timeout_policy::kDefaultCollectiveTimeoutMs);
        while (!config_.mpi_ctx->test(request, status))
        {
            /*
             * MPI_Test is the progress engine on the deployed Open MPI build.
             * This loop exists only for a true graph-selection dependency; a
             * publisher never enters it unless its fixed ring is backpressured.
             */
            if (std::chrono::steady_clock::now() >= deadline)
            {
                return fail(
                    std::string(operation) + " timed out after " +
                        std::to_string(
                            collective_timeout_policy::
                                kDefaultCollectiveTimeoutMs) +
                        "ms",
                    error);
            }
        }
        return true;
    }

    MoEOverlayMPIInferenceTransactionChannel::SendSlot *
    MoEOverlayMPIInferenceTransactionChannel::acquireSendSlot(
        std::string *error)
    {
        const auto find_available = [this]() -> SendSlot *
        {
            progressSendSlots();
            for (std::size_t offset = 0; offset < send_slots_.size(); ++offset)
            {
                const std::size_t index =
                    (next_send_slot_ + offset) % send_slots_.size();
                if (!send_slots_[index].in_flight)
                {
                    next_send_slot_ = (index + 1u) % send_slots_.size();
                    return &send_slots_[index];
                }
            }
            return nullptr;
        };

        if (auto *slot = find_available())
            return slot;

        const auto begin = std::chrono::steady_clock::now();
        const auto deadline =
            begin + std::chrono::milliseconds(
                        collective_timeout_policy::
                            kDefaultCollectiveTimeoutMs);
        while (std::chrono::steady_clock::now() < deadline)
        {
            if (auto *slot = find_available())
            {
                PerfStatsCollector::recordTimingNs(
                    "moe_overlay_transaction",
                    "control_send_ring_backpressure",
                    static_cast<std::uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - begin)
                            .count()),
                    "inference",
                    "mpi",
                    {{"slots", std::to_string(send_slots_.size())}});
                return slot;
            }
        }
        fail(
            "ExpertOverlay transaction control ring timed out waiting for a reusable slot",
            error);
        return nullptr;
    }

    bool MoEOverlayMPIInferenceTransactionChannel::publish(
        const MoEOverlayInferenceTransactionTicket &ticket,
        std::string *error)
    {
        if (config_.mpi_ctx->rank() != config_.source_world_rank)
        {
            return fail(
                "Only the continuation authority may publish an ExpertOverlay transaction ticket",
                error);
        }
        if (!ticket.valid() ||
            ticket.source_world_rank != config_.source_world_rank ||
            ticket.target_world_rank != config_.target_world_rank)
        {
            return fail(
                "ExpertOverlay transaction ticket does not match its control lane",
                error);
        }

        SendSlot *slot = acquireSendSlot(error);
        if (!slot)
            return false;
        slot->ticket = ticket;
        try
        {
            slot->request = config_.mpi_ctx->isend(
                &slot->ticket,
                sizeof(slot->ticket),
                MPI_BYTE,
                config_.target_world_rank,
                kInferenceTransactionTicketTag);
            slot->in_flight = slot->request != MPI_REQUEST_NULL;
            recordTicket(ticket, "publisher", "submitted");
            return true;
        }
        catch (const std::exception &exception)
        {
            slot->request = MPI_REQUEST_NULL;
            slot->in_flight = false;
            return fail(
                std::string("ExpertOverlay transaction ticket send failed: ") +
                    exception.what(),
                error);
        }
    }

    MoEOverlayInferenceTransactionReceiveResult
    MoEOverlayMPIInferenceTransactionChannel::receive()
    {
        MoEOverlayInferenceTransactionReceiveResult result;
        if (config_.mpi_ctx->rank() != config_.target_world_rank)
        {
            result.error =
                "Only the remote follower may receive an ExpertOverlay transaction ticket";
            return result;
        }

        try
        {
            receive_ticket_ = {};
            MPI_Status status{};
            MPI_Request request = config_.mpi_ctx->irecv(
                &receive_ticket_,
                sizeof(receive_ticket_),
                MPI_BYTE,
                config_.source_world_rank,
                kInferenceTransactionTicketTag);
            const auto receive_begin = std::chrono::steady_clock::now();
            if (!progressRequestToCompletion(
                    &request,
                    &status,
                    "ExpertOverlay transaction ticket receive",
                    &result.error))
            {
                return result;
            }
            const int received = config_.mpi_ctx->getCount(status, MPI_BYTE);
            if (received != static_cast<int>(sizeof(receive_ticket_)))
            {
                result.error =
                    "ExpertOverlay transaction control message has the wrong fixed size";
                return result;
            }
            if (!receive_ticket_.valid() ||
                receive_ticket_.source_world_rank !=
                    config_.source_world_rank ||
                receive_ticket_.target_world_rank !=
                    config_.target_world_rank)
            {
                result.error =
                    "ExpertOverlay transaction control message failed lane authentication";
                return result;
            }
            result.ok = true;
            result.ticket = receive_ticket_;
            recordTicketTiming(
                result.ticket,
                "control_receive_wait",
                "follower",
                receive_begin,
                std::chrono::steady_clock::now());
            recordTicket(result.ticket, "follower", "received");
            return result;
        }
        catch (const std::exception &exception)
        {
            result.error =
                std::string("ExpertOverlay transaction ticket receive failed: ") +
                exception.what();
            return result;
        }
    }

    std::size_t MoEOverlayMPIInferenceTransactionChannel::
        inFlightSendCount() const noexcept
    {
        try
        {
            progressSendSlots();
        }
        catch (...)
        {
            return send_slots_.size();
        }
        std::size_t count = 0;
        for (const auto &slot : send_slots_)
            count += slot.in_flight ? 1u : 0u;
        return count;
    }

    int MoEOverlayMPIInferenceTransactionChannel::localWorldRank() const noexcept
    {
        return config_.mpi_ctx ? config_.mpi_ctx->rank() : -1;
    }

    void MoEOverlayMPIInferenceTransactionChannel::drainNoexcept() noexcept
    {
        if (!config_.mpi_ctx ||
            config_.mpi_ctx->rank() != config_.source_world_rank)
        {
            return;
        }
        try
        {
            for (auto &slot : send_slots_)
            {
                if (!slot.in_flight)
                    continue;
                std::string error;
                if (!progressRequestToCompletion(
                        &slot.request,
                        nullptr,
                        "ExpertOverlay transaction ticket teardown",
                        &error))
                {
                    LOG_ERROR(error);
                    std::terminate();
                }
                slot.request = MPI_REQUEST_NULL;
                slot.in_flight = false;
            }
        }
        catch (const std::exception &exception)
        {
            LOG_ERROR(
                "ExpertOverlay transaction ticket teardown failed: "
                << exception.what());
            std::terminate();
        }
    }

    MoEOverlayInferenceTransactionPublisher::
        MoEOverlayInferenceTransactionPublisher(Config config)
        : channel_(std::move(config.channel)),
          protocol_(std::move(config.protocol))
    {
        validateConstruction();
    }

    void MoEOverlayInferenceTransactionPublisher::
        validateConstruction() const
    {
        if (!channel_)
        {
            throw std::invalid_argument(
                "ExpertOverlay transaction publisher requires a control channel");
        }
        const auto &topology = protocol_.topologyIdentity();
        if (channel_->localWorldRank() != channel_->sourceWorldRank() ||
            topology.source_world_rank != channel_->sourceWorldRank() ||
            topology.target_world_rank != channel_->targetWorldRank())
        {
            throw std::invalid_argument(
                "ExpertOverlay transaction publisher channel and protocol topology must name this source rank");
        }
    }

    bool MoEOverlayInferenceTransactionPublisher::beginCommand(
        const MoEOverlayInferenceCommandIdentity &command,
        std::string *error)
    {
        if (!protocol_.beginCommand(command, error))
            return false;
        active_command_ = command;
        return true;
    }

    MoEOverlayPublishedInferenceTransaction
    MoEOverlayInferenceTransactionPublisher::publish(
        const MoEOverlayInferenceExecutionDescriptor &descriptor)
    {
        MoEOverlayPublishedInferenceTransaction result;
        if (protocol_.state() != MoEOverlayInferenceProtocolState::Active ||
            !active_command_.valid())
        {
            result.error =
                "ExpertOverlay transaction publisher has no active command";
            return result;
        }

        try
        {
            result.ticket = makeMoEOverlayInferenceExecutionTicket(
                protocol_.topologyIdentity(),
                active_command_,
                protocol_.nextTransactionOrdinal(),
                descriptor.logical_step_id,
                descriptor.placement_epoch,
                descriptor.graph_role,
                descriptor.request_count,
                descriptor.logical_rows_per_request,
                descriptor.physical_rows_per_request,
                descriptor.draft_depth,
                descriptor.sidecar_depth);
        }
        catch (const std::exception &exception)
        {
            result.error = exception.what();
            return result;
        }

        const MoEOverlayInferenceAdmission admission =
            protocol_.accept(result.ticket);
        if (!admission.accepted())
        {
            result.error = admission.error.empty()
                               ? "ExpertOverlay source protocol rejected its next execution ticket"
                               : admission.error;
            return result;
        }
        result.slot_index = admission.slot_index;
        if (!protocol_.markSubmitted(
                result.slot_index, result.ticket, &result.error))
        {
            return result;
        }
        if (!channel_->publish(result.ticket, &result.error))
        {
            return result;
        }
        result.ok = true;
        recordTicket(result.ticket, "publisher_authority", "graph_admitted");
        return result;
    }

    bool MoEOverlayInferenceTransactionPublisher::retire(
        const MoEOverlayPublishedInferenceTransaction &transaction,
        std::string *error)
    {
        if (!transaction.ok ||
            transaction.slot_index ==
                MoEOverlayInferenceAdmission::kNoSlot)
        {
            return fail(
                "ExpertOverlay transaction publisher cannot retire an unpublished handle",
                error);
        }
        if (!protocol_.markReturnReady(
                transaction.slot_index, transaction.ticket, error) ||
            !protocol_.retire(
                transaction.slot_index, transaction.ticket, error))
        {
            return false;
        }
        recordTicket(
            transaction.ticket,
            "publisher_authority",
            "data_return_retired");
        return true;
    }

    bool MoEOverlayInferenceTransactionPublisher::publishTerminal(
        MoEOverlayInferenceTransactionAction action,
        std::uint64_t placement_epoch,
        int error_code,
        std::string *error)
    {
        if (protocol_.state() != MoEOverlayInferenceProtocolState::Active ||
            !active_command_.valid())
        {
            return fail(
                "ExpertOverlay transaction publisher has no active command to close",
                error);
        }

        MoEOverlayInferenceTransactionTicket ticket;
        try
        {
            ticket = makeMoEOverlayInferenceTerminalTicket(
                protocol_.topologyIdentity(),
                active_command_,
                protocol_.nextTransactionOrdinal(),
                placement_epoch,
                action,
                error_code);
        }
        catch (const std::exception &exception)
        {
            return fail(exception.what(), error);
        }

        const MoEOverlayInferenceAdmission admission =
            protocol_.accept(ticket);
        const bool expected_terminal =
            (action == MoEOverlayInferenceTransactionAction::Complete &&
             admission.status ==
                 MoEOverlayInferenceAdmissionStatus::Complete) ||
            (action == MoEOverlayInferenceTransactionAction::Abort &&
             admission.status ==
                 MoEOverlayInferenceAdmissionStatus::Aborted);
        if (!expected_terminal)
        {
            return fail(
                admission.error.empty()
                    ? "ExpertOverlay source protocol rejected its terminal ticket"
                    : admission.error,
                error);
        }
        if (!channel_->publish(ticket, error))
            return false;
        recordTicket(ticket, "publisher_authority", "command_terminal");
        return true;
    }

    bool MoEOverlayInferenceTransactionPublisher::complete(
        std::uint64_t placement_epoch,
        std::string *error)
    {
        return publishTerminal(
            MoEOverlayInferenceTransactionAction::Complete,
            placement_epoch,
            /*error_code=*/0,
            error);
    }

    bool MoEOverlayInferenceTransactionPublisher::abort(
        std::uint64_t placement_epoch,
        int error_code,
        std::string *error)
    {
        return publishTerminal(
            MoEOverlayInferenceTransactionAction::Abort,
            placement_epoch,
            error_code,
            error);
    }

    MoEOverlayInferenceTransactionCoordinator::
        MoEOverlayInferenceTransactionCoordinator(Config config)
        : config_(std::move(config))
    {
        if (config_.publishers.empty() ||
            config_.continuation_participant_count <= 0 ||
            config_.ticket_authority_participant_index < 0 ||
            config_.ticket_authority_participant_index >=
                config_.continuation_participant_count ||
            config_.max_transactions_per_command == 0 ||
            config_.max_mtp_draft_depth < 0)
        {
            throw std::invalid_argument(
                "ExpertOverlay transaction coordinator requires publishers, continuation participants, one in-range ticket authority, and fixed command capacity");
        }

        int source_rank = -1;
        std::vector<int> target_ranks;
        target_ranks.reserve(config_.publishers.size());
        for (const auto &publisher : config_.publishers)
        {
            if (!publisher || !publisher->topologyIdentity().valid())
            {
                throw std::invalid_argument(
                    "ExpertOverlay transaction coordinator received an invalid publisher topology");
            }
            const auto &topology = publisher->topologyIdentity();
            if (source_rank < 0)
                source_rank = topology.source_world_rank;
            if (topology.source_world_rank != source_rank ||
                std::find(
                    target_ranks.begin(), target_ranks.end(),
                    topology.target_world_rank) != target_ranks.end())
            {
                throw std::invalid_argument(
                    "ExpertOverlay transaction coordinator publishers must share one source and have unique targets");
            }
            target_ranks.push_back(topology.target_world_rank);
        }

        retained_transactions_.resize(
            config_.max_transactions_per_command);
        for (auto &transaction : retained_transactions_)
        {
            transaction.target_transactions.resize(
                config_.publishers.size());
        }
        entered_participants_.resize(
            static_cast<std::size_t>(
                config_.continuation_participant_count));
        finished_participants_.resize(
            static_cast<std::size_t>(
                config_.continuation_participant_count));
    }

    bool MoEOverlayInferenceTransactionCoordinator::failLocked(
        std::string message,
        std::string *error)
    {
        if (failure_.empty())
            failure_ = std::move(message);
        state_ = MoEOverlayInferenceProtocolState::Failed;
        graph_group_completion_cv_.notify_all();
        if (error)
            *error = failure_;
        return false;
    }

    bool MoEOverlayInferenceTransactionCoordinator::allParticipantsMarked(
        const std::vector<std::uint8_t> &marks) const
    {
        return std::all_of(
            marks.begin(), marks.end(),
            [](std::uint8_t marked) { return marked != 0; });
    }

    bool MoEOverlayInferenceTransactionCoordinator::beginCommand(
        const MoEOverlayInferenceCommandIdentity &command,
        std::string *error)
    {
        std::lock_guard lock(mutex_);
        if (error)
            error->clear();
        if (!command.valid())
            return failLocked(
                "ExpertOverlay coordinator received an invalid command identity",
                error);
        if (state_ == MoEOverlayInferenceProtocolState::Active)
        {
            return failLocked(
                "ExpertOverlay coordinator cannot overlap outer commands",
                error);
        }
        if (state_ == MoEOverlayInferenceProtocolState::Failed)
        {
            if (error)
                *error = failure_;
            return false;
        }

        resetGraphSequenceLocked();
        std::fill(
            entered_participants_.begin(),
            entered_participants_.end(), std::uint8_t{0});
        std::fill(
            finished_participants_.begin(),
            finished_participants_.end(), std::uint8_t{0});
        active_command_ = command;
        total_transaction_count_ = 0;
        graph_sequence_count_ = 0;
        current_placement_epoch_ = command.initial_placement_epoch;
        failure_.clear();

        std::size_t begun = 0;
        for (; begun < config_.publishers.size(); ++begun)
        {
            std::string publisher_error;
            if (!config_.publishers[begun]->beginCommand(
                    command, &publisher_error))
            {
                for (std::size_t index = 0; index < begun; ++index)
                {
                    std::string ignored;
                    (void)config_.publishers[index]->abort(
                        command.initial_placement_epoch,
                        /*error_code=*/1,
                        &ignored);
                }
                return failLocked(
                    publisher_error.empty()
                        ? "ExpertOverlay target publisher rejected command admission"
                        : publisher_error,
                    error);
            }
        }
        state_ = MoEOverlayInferenceProtocolState::Active;
        PerfStatsCollector::addCounter(
            "moe_overlay_transaction",
            "coordinator_commands",
            1.0,
            "inference",
            "continuation_rank",
            {{"action", "begin"},
             {"command", std::to_string(command.command_id)},
             {"participants",
              std::to_string(config_.continuation_participant_count)},
             {"ticket_authority_participant",
              std::to_string(
                  config_.ticket_authority_participant_index)},
             {"targets", std::to_string(config_.publishers.size())}});
        return true;
    }

    bool MoEOverlayInferenceTransactionCoordinator::beginGraphSequence(
        int draft_depth,
        std::string *error)
    {
        std::lock_guard lock(mutex_);
        return beginGraphSequenceLocked(draft_depth, error);
    }

    bool MoEOverlayInferenceTransactionCoordinator::
        beginGraphSequenceLocked(
            int draft_depth,
            std::string *error)
    {
        if (error)
            error->clear();
        if (state_ != MoEOverlayInferenceProtocolState::Active)
            return failLocked(
                "ExpertOverlay graph sequence requires an active command", error);
        if (graph_group_active_ || transaction_count_ != 0 ||
            mtp_depth_declared_)
            return failLocked(
                "ExpertOverlay graph sequence cannot overlap live graph slots",
                error);
        if (draft_depth < 0 ||
            draft_depth > config_.max_mtp_draft_depth)
        {
            return failLocked(
                "ExpertOverlay MTP depth exceeds the retained graph family",
                error);
        }
        mtp_depth_declared_ = true;
        active_mtp_draft_depth_ = draft_depth;
        next_sidecar_ordinal_ = 0;
        sequence_main_graph_count_ = 0;
        sequence_sidecar_graph_count_ = 0;
        sequence_verifier_graph_count_ = 0;
        ++graph_sequence_count_;
        return true;
    }

    bool MoEOverlayInferenceTransactionCoordinator::
        admitSerialPrefillGraph(
            int participant_index,
            std::string *error)
    {
        std::unique_lock lock(mutex_);
        if (error)
            error->clear();
        if (state_ != MoEOverlayInferenceProtocolState::Active)
        {
            return failLocked(
                "ExpertOverlay serial prefill requires an active command",
                error);
        }
        if (participant_index < 0 ||
            participant_index >= config_.continuation_participant_count)
        {
            return failLocked(
                "ExpertOverlay serial prefill received an out-of-range participant",
                error);
        }

        const std::size_t participant =
            static_cast<std::size_t>(participant_index);
        if (graph_group_active_ && entered_participants_[participant] != 0)
        {
            /*
             * GPU graph submission is asynchronous, so one LocalTP worker may
             * return from chunk N and request chunk N+1 while a sibling is
             * still submitting N.  It cannot join the active group twice or
             * reserve the next descriptor early.  Wait only for rank-local
             * host submission bookkeeping; the device streams continue
             * independently and preserve their already-enqueued graph order.
             */
            const auto timeout = std::chrono::milliseconds(
                collective_timeout_policy::kDefaultCollectiveTimeoutMs);
            const bool aligned = graph_group_completion_cv_.wait_for(
                lock,
                timeout,
                [&]
                {
                    return state_ !=
                               MoEOverlayInferenceProtocolState::Active ||
                           !graph_group_active_ ||
                           entered_participants_[participant] == 0;
                });
            if (!aligned)
            {
                return failLocked(
                    "ExpertOverlay serial prefill timed out waiting for "
                    "symmetric LocalTP chunk submission alignment",
                    error);
            }
            if (state_ != MoEOverlayInferenceProtocolState::Active)
            {
                if (error)
                {
                    *error = failure_.empty()
                                 ? "ExpertOverlay serial prefill alignment "
                                   "observed a terminal coordinator"
                                 : failure_;
                }
                return false;
            }
        }

        if (graph_group_active_)
        {
            /*
             * LocalTP siblings call this independently before joining the same
             * graph group. Once the first participant has reserved the group,
             * every later participant must observe exactly a MainPrefill group.
             * Ticket publication remains deferred until the first exact launch
             * edge, and no sequence transition is needed until all siblings
             * finish.
             */
            if (transaction_count_ >= retained_transactions_.size() ||
                !retained_transactions_[transaction_count_].used ||
                retained_transactions_[transaction_count_]
                        .descriptor.graph_role !=
                    MoEOverlayInferenceGraphRole::MainPrefill)
            {
                return failLocked(
                    "ExpertOverlay serial prefill attempted to join a non-prefill graph group",
                    error);
            }
            return true;
        }

        if (!mtp_depth_declared_)
            return beginGraphSequenceLocked(/*draft_depth=*/0, error);

        if (active_mtp_draft_depth_ != 0)
        {
            return failLocked(
                "ExpertOverlay serial prefill cannot enter an MTP graph sequence",
                error);
        }
        if (transaction_count_ == 0)
        {
            /* Another symmetric participant opened this sequence first. */
            return true;
        }

        const bool complete_prefill_chunk =
            transaction_count_ == 1 &&
            sequence_main_graph_count_ == 1 &&
            sequence_sidecar_graph_count_ == 0 &&
            sequence_verifier_graph_count_ == 0;
        if (!complete_prefill_chunk)
        {
            return failLocked(
                "ExpertOverlay serial prefill cannot advance an incomplete or non-prefill graph sequence",
                error);
        }

        /*
         * A bounded prefill schedule is an ordered series of complete serial
         * transactions under one outer MPI command. Retire the previous fixed
         * control slot before the first participant reserves the next chunk;
         * the follower and continuation endpoint streams preserve device-side
         * graph order independently of this CPU metadata transition.
         */
        if (!retireCompletedGraphSequenceLocked(error))
            return false;
        return beginGraphSequenceLocked(/*draft_depth=*/0, error);
    }

    MoEOverlayInferenceParticipantGraphBinding
    MoEOverlayInferenceTransactionCoordinator::beginParticipantGraph(
        MoEOverlayInferenceExecutionDescriptor descriptor,
        int participant_index)
    {
        std::lock_guard lock(mutex_);
        MoEOverlayInferenceParticipantGraphBinding result;
        result.participant_index = participant_index;
        if (state_ == MoEOverlayInferenceProtocolState::Idle ||
            state_ == MoEOverlayInferenceProtocolState::Complete)
        {
            result.ok = true;
            return result;
        }
        if (state_ == MoEOverlayInferenceProtocolState::Failed)
        {
            result.error = failure_.empty()
                               ? "ExpertOverlay coordinator is failed"
                               : failure_;
            return result;
        }
        if (!mtp_depth_declared_)
        {
            failLocked(
                "ExpertOverlay graph entered before graph-sequence admission",
                &result.error);
            return result;
        }
        if (participant_index < 0 ||
            participant_index >= config_.continuation_participant_count ||
            descriptor.graph_role == MoEOverlayInferenceGraphRole::None ||
            descriptor.placement_epoch == 0 ||
            descriptor.request_count <= 0 ||
            descriptor.logical_rows_per_request <= 0 ||
            descriptor.physical_rows_per_request <
                descriptor.logical_rows_per_request)
        {
            failLocked(
                "ExpertOverlay participant graph descriptor is invalid",
                &result.error);
            return result;
        }

        switch (descriptor.graph_role)
        {
        case MoEOverlayInferenceGraphRole::MTPDraft:
        {
            const int group_sidecar_ordinal =
                graph_group_active_ &&
                        transaction_count_ < retained_transactions_.size()
                    ? retained_transactions_[transaction_count_]
                          .descriptor.sidecar_depth
                    : next_sidecar_ordinal_;
            if (active_mtp_draft_depth_ <= 0 ||
                group_sidecar_ordinal < 0 ||
                group_sidecar_ordinal >= active_mtp_draft_depth_)
            {
                failLocked(
                    "ExpertOverlay sidecar count exceeds the admitted MTP depth",
                    &result.error);
                return result;
            }
            descriptor.draft_depth = active_mtp_draft_depth_;
            descriptor.sidecar_depth = group_sidecar_ordinal;
            break;
        }
        case MoEOverlayInferenceGraphRole::MTPGroupedVerifier:
            if (active_mtp_draft_depth_ <= 0 ||
                descriptor.draft_depth != active_mtp_draft_depth_ ||
                descriptor.sidecar_depth != -1)
            {
                failLocked(
                    "ExpertOverlay grouped verifier width disagrees with command admission",
                    &result.error);
                return result;
            }
            break;
        case MoEOverlayInferenceGraphRole::MainPrefill:
        case MoEOverlayInferenceGraphRole::MainDecode:
            if (descriptor.draft_depth != -1 ||
                descriptor.sidecar_depth != -1)
            {
                failLocked(
                    "ExpertOverlay main graph cannot carry MTP sidecar geometry",
                    &result.error);
                return result;
            }
            break;
        case MoEOverlayInferenceGraphRole::None:
            break;
        }

        RetainedTransaction *transaction = nullptr;
        if (graph_group_active_)
        {
            if (transaction_count_ >= retained_transactions_.size())
            {
                failLocked(
                    "ExpertOverlay active graph group exceeds fixed transaction storage",
                    &result.error);
                return result;
            }
            transaction = &retained_transactions_[transaction_count_];
            descriptor.logical_step_id =
                transaction->descriptor.logical_step_id;
            if (!transaction->used ||
                transaction->descriptor != descriptor)
            {
                failLocked(
                    "ExpertOverlay LocalTP participants entered divergent graph descriptors",
                    &result.error);
                return result;
            }
        }
        else
        {
            if (transaction_count_ >= retained_transactions_.size() ||
                next_logical_step_id_ == 0 ||
                next_logical_step_id_ ==
                    std::numeric_limits<std::uint64_t>::max())
            {
                failLocked(
                    "ExpertOverlay command exhausted fixed transaction or operation-id capacity",
                    &result.error);
                return result;
            }
            transaction = &retained_transactions_[transaction_count_];
            descriptor.logical_step_id = next_logical_step_id_++;
            transaction->used = true;
            transaction->armed = false;
            transaction->group_id = next_group_id_++;
            transaction->descriptor = descriptor;
            std::fill(
                entered_participants_.begin(),
                entered_participants_.end(), std::uint8_t{0});
            std::fill(
                finished_participants_.begin(),
                finished_participants_.end(), std::uint8_t{0});

            graph_group_active_ = true;
            current_placement_epoch_ = std::max(
                current_placement_epoch_, descriptor.placement_epoch);
            if (descriptor.graph_role ==
                MoEOverlayInferenceGraphRole::MTPDraft)
            {
                ++next_sidecar_ordinal_;
            }
        }

        const std::size_t participant =
            static_cast<std::size_t>(participant_index);
        if (entered_participants_[participant] != 0)
        {
            failLocked(
                "ExpertOverlay continuation participant entered one graph twice",
                &result.error);
            return result;
        }
        entered_participants_[participant] = 1;
        result.ok = true;
        result.active = true;
        result.owns_ticket_authority =
            participant_index ==
            config_.ticket_authority_participant_index;
        result.group_id = transaction->group_id;
        result.request_generation = active_command_.request_generation;
        result.descriptor = transaction->descriptor;
        return result;
    }

    bool MoEOverlayInferenceTransactionCoordinator::armParticipantGraph(
        const MoEOverlayInferenceParticipantGraphBinding &binding,
        std::string *error)
    {
        std::lock_guard lock(mutex_);
        if (error)
            error->clear();
        if (!binding.ok || !binding.active ||
            state_ != MoEOverlayInferenceProtocolState::Active ||
            !graph_group_active_ ||
            transaction_count_ >= retained_transactions_.size())
        {
            return failLocked(
                "ExpertOverlay participant attempted to arm an inactive graph group",
                error);
        }

        RetainedTransaction &transaction =
            retained_transactions_[transaction_count_];
        if (!transaction.used || binding.group_id != transaction.group_id ||
            binding.descriptor != transaction.descriptor ||
            binding.participant_index < 0 ||
            binding.participant_index >=
                config_.continuation_participant_count ||
            binding.participant_index !=
                config_.ticket_authority_participant_index ||
            !binding.owns_ticket_authority)
        {
            return failLocked(
                "ExpertOverlay executable launch rejected a non-authority, stale, or aliasing participant binding",
                error);
        }
        const std::size_t participant =
            static_cast<std::size_t>(binding.participant_index);
        if (entered_participants_[participant] == 0 ||
            finished_participants_[participant] != 0)
        {
            return failLocked(
                "ExpertOverlay executable launch occurred outside its participant graph scope",
                error);
        }
        if (transaction.armed)
            return true;

        /*
         * Ticket publication is intentionally the final host action before the
         * continuation executable launch. The descriptor and every fixed source
         * slot were reserved at graph entry, so this loop allocates no graph or
         * payload state and introduces no new collective.
         */
        for (std::size_t target = 0;
             target < config_.publishers.size(); ++target)
        {
            auto published =
                config_.publishers[target]->publish(transaction.descriptor);
            transaction.target_transactions[target] = published;
            if (!published.ok)
            {
                return failLocked(
                    published.error.empty()
                        ? "ExpertOverlay target ticket publication failed at executable launch"
                        : published.error,
                    error);
            }
        }
        transaction.armed = true;
        PerfStatsCollector::addCounter(
            "moe_overlay_transaction",
            "coordinator_graph_arms",
            1.0,
            "inference",
            "continuation_rank",
            {{"command", std::to_string(active_command_.command_id)},
             {"group", std::to_string(transaction.group_id)},
             {"role", std::to_string(static_cast<std::uint32_t>(
                          transaction.descriptor.graph_role))},
             {"participant", std::to_string(binding.participant_index)}});
        return true;
    }

    bool MoEOverlayInferenceTransactionCoordinator::finishParticipantGraph(
        const MoEOverlayInferenceParticipantGraphBinding &binding,
        bool execution_succeeded,
        std::string *error)
    {
        std::lock_guard lock(mutex_);
        if (error)
            error->clear();
        if (!binding.ok || !binding.active ||
            state_ != MoEOverlayInferenceProtocolState::Active ||
            !graph_group_active_ ||
            transaction_count_ >= retained_transactions_.size())
        {
            return failLocked(
                "ExpertOverlay participant attempted to finish an inactive graph group",
                error);
        }
        const RetainedTransaction &transaction =
            retained_transactions_[transaction_count_];
        if (binding.group_id != transaction.group_id ||
            binding.descriptor != transaction.descriptor ||
            binding.participant_index < 0 ||
            binding.participant_index >=
                config_.continuation_participant_count)
        {
            return failLocked(
                "ExpertOverlay participant graph finish rejected a stale or aliasing binding",
                error);
        }
        const std::size_t participant =
            static_cast<std::size_t>(binding.participant_index);
        if (entered_participants_[participant] == 0 ||
            finished_participants_[participant] != 0)
        {
            return failLocked(
                "ExpertOverlay participant graph finish lifecycle is out of order",
                error);
        }
        if (execution_succeeded &&
            binding.participant_index ==
                config_.ticket_authority_participant_index &&
            !transaction.armed)
        {
            return failLocked(
                "ExpertOverlay ticket authority reported successful graph execution before its launch ticket was armed",
                error);
        }
        finished_participants_[participant] = 1;
        graph_execution_failed_ =
            graph_execution_failed_ || !execution_succeeded;

        if (allParticipantsMarked(finished_participants_))
        {
            if (!allParticipantsMarked(entered_participants_))
            {
                return failLocked(
                    "ExpertOverlay graph group finished without every continuation participant",
                    error);
            }
            if (!graph_execution_failed_ && !transaction.armed)
            {
                return failLocked(
                    "ExpertOverlay graph group completed successfully without its ticket authority arming the remote transaction",
                    error);
            }
            switch (transaction.descriptor.graph_role)
            {
            case MoEOverlayInferenceGraphRole::MainPrefill:
            case MoEOverlayInferenceGraphRole::MainDecode:
                ++sequence_main_graph_count_;
                break;
            case MoEOverlayInferenceGraphRole::MTPDraft:
                ++sequence_sidecar_graph_count_;
                break;
            case MoEOverlayInferenceGraphRole::MTPGroupedVerifier:
                ++sequence_verifier_graph_count_;
                break;
            case MoEOverlayInferenceGraphRole::None:
                return failLocked(
                    "ExpertOverlay execution sequence sealed a terminal graph role",
                    error);
            }
            graph_group_active_ = false;
            ++transaction_count_;
            graph_group_completion_cv_.notify_all();
        }
        return true;
    }

    void MoEOverlayInferenceTransactionCoordinator::resetGraphSequenceLocked()
        noexcept
    {
        for (auto &transaction : retained_transactions_)
        {
            transaction.used = false;
            transaction.armed = false;
            transaction.group_id = 0;
            transaction.descriptor = {};
            std::fill(
                transaction.target_transactions.begin(),
                transaction.target_transactions.end(),
                MoEOverlayPublishedInferenceTransaction{});
        }
        transaction_count_ = 0;
        graph_group_active_ = false;
        graph_execution_failed_ = false;
        mtp_depth_declared_ = false;
        active_mtp_draft_depth_ = -1;
        next_sidecar_ordinal_ = 0;
        sequence_main_graph_count_ = 0;
        sequence_sidecar_graph_count_ = 0;
        sequence_verifier_graph_count_ = 0;
    }

    bool MoEOverlayInferenceTransactionCoordinator::
        retireCompletedGraphSequenceLocked(std::string *error)
    {
        if (state_ != MoEOverlayInferenceProtocolState::Active ||
            !mtp_depth_declared_ || graph_group_active_ ||
            graph_execution_failed_ || transaction_count_ == 0)
        {
            return failLocked(
                "ExpertOverlay graph sequence cannot retire before every local graph and sparse return complete",
                error);
        }

        const bool serial_shape =
            active_mtp_draft_depth_ == 0 &&
            sequence_main_graph_count_ == 1 &&
            sequence_sidecar_graph_count_ == 0 &&
            sequence_verifier_graph_count_ == 0;
        const bool speculative_shape =
            active_mtp_draft_depth_ > 0 &&
            sequence_main_graph_count_ <= 1 &&
            sequence_sidecar_graph_count_ == active_mtp_draft_depth_ &&
            sequence_verifier_graph_count_ == 1;
        if (!serial_shape && !speculative_shape)
        {
            return failLocked(
                "ExpertOverlay graph sequence does not contain one complete "
                "serial or speculative transaction: admitted_draft_depth=" +
                    std::to_string(active_mtp_draft_depth_) +
                    " main_graphs=" +
                    std::to_string(sequence_main_graph_count_) +
                    " sidecar_graphs=" +
                    std::to_string(sequence_sidecar_graph_count_) +
                    " verifier_graphs=" +
                    std::to_string(sequence_verifier_graph_count_) +
                    " retained_transactions=" +
                    std::to_string(transaction_count_),
                error);
        }

        const std::size_t retired_count = transaction_count_;
        for (std::size_t transaction_index = 0;
             transaction_index < retired_count;
             ++transaction_index)
        {
            auto &transaction = retained_transactions_[transaction_index];
            if (!transaction.used || !transaction.armed)
            {
                return failLocked(
                    "ExpertOverlay graph sequence contains an uninitialized or unarmed retained transaction",
                    error);
            }
            for (std::size_t target = 0;
                 target < config_.publishers.size(); ++target)
            {
                std::string publisher_error;
                if (!config_.publishers[target]->retire(
                        transaction.target_transactions[target],
                        &publisher_error))
                {
                    return failLocked(
                        publisher_error.empty()
                            ? "ExpertOverlay target source slot retirement failed"
                            : publisher_error,
                        error);
                }
            }
        }
        total_transaction_count_ += retired_count;
        PerfStatsCollector::addCounter(
            "moe_overlay_transaction",
            "coordinator_graph_sequences",
            1.0,
            "inference",
            "continuation_rank",
            {{"action", "retire"},
             {"command", std::to_string(active_command_.command_id)},
             {"draft_depth", std::to_string(active_mtp_draft_depth_)},
             {"transactions", std::to_string(retired_count)}});
        resetGraphSequenceLocked();
        return true;
    }

    bool MoEOverlayInferenceTransactionCoordinator::
        retireCompletedGraphSequence(std::string *error)
    {
        std::lock_guard lock(mutex_);
        if (error)
            error->clear();
        return retireCompletedGraphSequenceLocked(error);
    }

    bool MoEOverlayInferenceTransactionCoordinator::completeCommand(
        std::uint64_t placement_epoch,
        std::string *error)
    {
        std::lock_guard lock(mutex_);
        if (error)
            error->clear();
        if (state_ != MoEOverlayInferenceProtocolState::Active ||
            graph_group_active_ || graph_execution_failed_ ||
            placement_epoch < current_placement_epoch_)
        {
            return failLocked(
                "ExpertOverlay command cannot complete with unfinished or failed graph work",
                error);
        }

        if (mtp_depth_declared_ &&
            !retireCompletedGraphSequenceLocked(error))
        {
            return false;
        }

        for (const auto &publisher : config_.publishers)
        {
            std::string publisher_error;
            if (!publisher->complete(
                    current_placement_epoch_, &publisher_error))
            {
                return failLocked(
                    publisher_error.empty()
                        ? "ExpertOverlay target command terminal failed"
                        : publisher_error,
                    error);
            }
        }
        state_ = MoEOverlayInferenceProtocolState::Complete;
        PerfStatsCollector::addCounter(
            "moe_overlay_transaction",
            "coordinator_commands",
            1.0,
            "inference",
            "continuation_rank",
            {{"action", "complete"},
             {"command", std::to_string(active_command_.command_id)},
             {"transactions", std::to_string(total_transaction_count_)},
             {"graph_sequences", std::to_string(graph_sequence_count_)}});
        return true;
    }

    bool MoEOverlayInferenceTransactionCoordinator::abortCommand(
        std::uint64_t placement_epoch,
        int error_code,
        std::string *error) noexcept
    {
        try
        {
            std::lock_guard lock(mutex_);
            if (error)
                error->clear();
            if (state_ == MoEOverlayInferenceProtocolState::Idle ||
                state_ == MoEOverlayInferenceProtocolState::Complete ||
                !active_command_.valid())
            {
                return false;
            }
            const std::uint64_t terminal_epoch = std::max(
                current_placement_epoch_, placement_epoch);
            bool ok = true;
            std::string first_error;
            for (const auto &publisher : config_.publishers)
            {
                std::string publisher_error;
                if (!publisher->abort(
                        terminal_epoch,
                        std::max(1, error_code),
                        &publisher_error))
                {
                    ok = false;
                    if (first_error.empty())
                        first_error = std::move(publisher_error);
                }
            }
            state_ = MoEOverlayInferenceProtocolState::Failed;
            graph_group_completion_cv_.notify_all();
            if (failure_.empty())
            {
                failure_ = first_error.empty()
                               ? "ExpertOverlay command aborted"
                               : first_error;
            }
            if (error)
                *error = failure_;
            return ok;
        }
        catch (const std::exception &exception)
        {
            if (error)
                *error = exception.what();
            return false;
        }
        catch (...)
        {
            if (error)
                *error = "Unknown ExpertOverlay command abort failure";
            return false;
        }
    }

    MoEOverlayInferenceProtocolState
    MoEOverlayInferenceTransactionCoordinator::state() const noexcept
    {
        std::lock_guard lock(mutex_);
        return state_;
    }

    std::uint64_t
    MoEOverlayInferenceTransactionCoordinator::currentPlacementEpoch() const
        noexcept
    {
        std::lock_guard lock(mutex_);
        return current_placement_epoch_;
    }

    int MoEOverlayInferenceTransactionCoordinator::activeMTPDraftDepth() const
        noexcept
    {
        std::lock_guard lock(mutex_);
        return active_mtp_draft_depth_;
    }

    MoEOverlayInferenceParticipantGraphScope::
        MoEOverlayInferenceParticipantGraphScope(
            std::shared_ptr<MoEOverlayInferenceTransactionCoordinator>
                coordinator,
            MoEOverlayInferenceExecutionDescriptor descriptor,
            int participant_index)
        : coordinator_(std::move(coordinator))
    {
        if (!coordinator_)
        {
            binding_.ok = true;
            return;
        }
        binding_ = coordinator_->beginParticipantGraph(
            std::move(descriptor), participant_index);
    }

    bool MoEOverlayInferenceParticipantGraphScope::armForExecutableLaunch(
        void *execution_stream,
        std::string *error)
    {
        if (error)
            error->clear();
        if (finished_)
        {
            if (error)
                *error =
                    "ExpertOverlay graph scope cannot arm after it has finished";
            return false;
        }
        if (!binding_.ok)
        {
            if (error)
                *error = binding_.error;
            return false;
        }
        if (!binding_.active)
            return true;
        if (!binding_.owns_ticket_authority)
        {
            if (error)
                *error =
                    "ExpertOverlay sibling graph cannot arm the ticket authority's executable edge";
            return false;
        }
        if (!execution_stream)
        {
            if (error)
                *error =
                    "ExpertOverlay GPU graph launch arm requires an exact non-null execution stream";
            return false;
        }
        if (armed_)
            return true;
        if (!coordinator_ ||
            !coordinator_->armParticipantGraph(binding_, error))
        {
            return false;
        }
        armed_ = true;
        return true;
    }

    bool MoEOverlayInferenceParticipantGraphScope::armForHostExecution(
        std::string *error)
    {
        if (error)
            error->clear();
        if (finished_)
        {
            if (error)
                *error =
                    "ExpertOverlay graph scope cannot arm after it has finished";
            return false;
        }
        if (!binding_.ok)
        {
            if (error)
                *error = binding_.error;
            return false;
        }
        if (!binding_.active || armed_)
            return true;
        if (!binding_.owns_ticket_authority)
        {
            if (error)
                *error =
                    "ExpertOverlay sibling CPU graph cannot arm the ticket authority's host boundary";
            return false;
        }
        if (!coordinator_ ||
            !coordinator_->armParticipantGraph(binding_, error))
        {
            return false;
        }
        armed_ = true;
        return true;
    }

    MoEOverlayInferenceParticipantGraphScope::
        ~MoEOverlayInferenceParticipantGraphScope() noexcept
    {
        if (!coordinator_ || !binding_.active || finished_)
            return;
        std::string error;
        if (!coordinator_->finishParticipantGraph(
                binding_, /*execution_succeeded=*/false, &error))
        {
            LOG_ERROR(
                "ExpertOverlay participant graph scope could not publish its failed terminal: "
                << error);
        }
    }

    bool MoEOverlayInferenceParticipantGraphScope::finish(
        bool execution_succeeded,
        std::string *error)
    {
        if (error)
            error->clear();
        if (finished_)
            return fail(
                "ExpertOverlay participant graph scope was finished twice",
                error);
        if (!binding_.ok)
            return fail(binding_.error, error);
        if (!binding_.active)
        {
            finished_ = true;
            return true;
        }
        if (!coordinator_->finishParticipantGraph(
                binding_, execution_succeeded, error))
        {
            return false;
        }
        finished_ = true;
        return true;
    }

    MoEOverlayInferenceTransactionFollower::
        MoEOverlayInferenceTransactionFollower(Config config)
        : channel_(std::move(config.channel)),
          executor_(config.executor),
          protocol_(std::move(config.protocol))
    {
        validateConstruction();
    }

    void MoEOverlayInferenceTransactionFollower::validateConstruction() const
    {
        if (!channel_ || !executor_)
        {
            throw std::invalid_argument(
                "ExpertOverlay transaction follower requires a channel and retained-graph executor");
        }

        const auto &topology = protocol_.topologyIdentity();
        if (channel_->localWorldRank() != channel_->targetWorldRank() ||
            topology.source_world_rank != channel_->sourceWorldRank() ||
            topology.target_world_rank != channel_->targetWorldRank())
        {
            throw std::invalid_argument(
                "ExpertOverlay transaction follower channel and protocol topology must name this target rank");
        }
    }

    MoEOverlayInferenceFollowerCommandResult
    MoEOverlayInferenceTransactionFollower::runOneCommand()
    {
        MoEOverlayInferenceFollowerCommandResult result;
        while (true)
        {
            const auto received = channel_->receive();
            if (!received.ok)
            {
                result.error = received.error;
                return result;
            }

            const auto &ticket = received.ticket;
            const auto protocol_begin = std::chrono::steady_clock::now();
            if (protocol_.state() == MoEOverlayInferenceProtocolState::Idle ||
                protocol_.state() ==
                    MoEOverlayInferenceProtocolState::Complete)
            {
                if (!protocol_.beginCommand(
                        ticket.commandIdentity(), &result.error))
                {
                    return result;
                }
            }

            const MoEOverlayInferenceAdmission admission =
                protocol_.accept(ticket);
            if (admission.status ==
                MoEOverlayInferenceAdmissionStatus::Complete)
            {
                result.ok = true;
                return result;
            }
            if (admission.status ==
                MoEOverlayInferenceAdmissionStatus::Aborted)
            {
                result.aborted = true;
                result.error_code = ticket.error_code;
                result.error = "Continuation authority aborted ExpertOverlay transaction command";
                return result;
            }
            if (!admission.accepted())
            {
                result.error = admission.error.empty()
                                   ? "ExpertOverlay follower rejected a transaction ticket"
                                   : admission.error;
                return result;
            }

            if (!protocol_.markSubmitted(
                    admission.slot_index, ticket, &result.error))
            {
                return result;
            }
            const auto protocol_end = std::chrono::steady_clock::now();
            recordTicketTiming(
                ticket,
                "follower_protocol_admission",
                "follower",
                protocol_begin,
                protocol_end);
            const auto execution_begin = protocol_end;
            if (!executor_->executeMoEOverlayInferenceTransaction(
                    ticket, &result.error))
            {
                if (result.error.empty())
                {
                    result.error =
                        "Remote retained ExpertOverlay graph transaction failed";
                }
                recordTicket(ticket, "follower", "execution_failed");
                return result;
            }
            const auto execution_end = std::chrono::steady_clock::now();
            recordTicketTiming(
                ticket,
                "follower_graph_execution",
                "follower",
                execution_begin,
                execution_end);
            if (!protocol_.markReturnReady(
                    admission.slot_index, ticket, &result.error) ||
                !protocol_.retire(
                    admission.slot_index, ticket, &result.error))
            {
                return result;
            }
            recordTicketTiming(
                ticket,
                "follower_protocol_retirement",
                "follower",
                execution_end,
                std::chrono::steady_clock::now());
            ++result.executed_transactions;
            recordTicket(ticket, "follower", "executed");
        }
    }

} // namespace llaminar2
