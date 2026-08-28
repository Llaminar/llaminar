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
            /*
             * A ticket is the only host-visible edge in heterogeneous graph
             * execution.  Keep its DEBUG witness beside the PerfStats record:
             * if a device timeline stalls, this establishes whether control
             * publication, remote receipt, or retained execution was the last
             * completed edge without polling any device-owned model state.
             */
            LOG_DEBUG(
                "[ExpertOverlay][Transaction] endpoint=" << endpoint
                << " result=" << result
                << " action="
                << static_cast<std::uint32_t>(ticket.action)
                << " role="
                << static_cast<std::uint32_t>(ticket.graph_role)
                << " command=" << ticket.command_id
                << " ordinal=" << ticket.transaction_ordinal
                << " logical_step=" << ticket.logical_step_id
                << " rows=" << ticket.request_count << "x"
                << ticket.logical_rows_per_request << "/"
                << ticket.physical_rows_per_request
                << " target_rank=" << ticket.target_world_rank);
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

    MoEOverlayInferenceCoordinatorGraphPlan
    MoEOverlayInferenceCoordinatorGraphPlan::
        sealAfterSynchronizedMaterialization(
            std::uint64_t graph_family_generation,
            std::vector<Segment> segments)
    {
        /* Stable role/rank order is part of the setup certificate. It keeps
         * diagnostics and publisher validation independent of the order in
         * which declarative domains happened to be visited. */
        std::sort(
            segments.begin(),
            segments.end(),
            [](const Segment &left, const Segment &right)
            {
                if (left.role != right.role)
                {
                    return static_cast<std::uint8_t>(left.role) <
                           static_cast<std::uint8_t>(right.role);
                }
                return left.world_rank < right.world_rank;
            });

        MoEOverlayInferenceCoordinatorGraphPlan result;
        result.graph_family_generation_ = graph_family_generation;
        result.segments_ = std::move(segments);
        if (!result.valid())
        {
            throw std::invalid_argument(
                "ExpertOverlay coordinator graph plan requires one unique continuation rank, at least one unique follower rank, sealed materialization, and positive participant counts");
        }
        return result;
    }

    bool MoEOverlayInferenceCoordinatorGraphPlan::valid() const noexcept
    {
        if (graph_family_generation_ == 0 || segments_.size() < 2u)
            return false;

        std::size_t continuation_count = 0u;
        std::size_t follower_count = 0u;
        for (std::size_t index = 0u; index < segments_.size(); ++index)
        {
            const Segment &segment = segments_[index];
            const bool valid_role =
                segment.role ==
                    MoEOverlayInferenceCoordinatorSegmentRole::Continuation ||
                segment.role ==
                    MoEOverlayInferenceCoordinatorSegmentRole::ExpertFollower;
            const bool valid_materialization =
                segment.materialization ==
                    MoEOverlayInferenceSegmentMaterializationKind::
                        NativeDeviceExecutable ||
                segment.materialization ==
                    MoEOverlayInferenceSegmentMaterializationKind::
                        EagerHostGraph;
            if (!valid_role || !valid_materialization ||
                segment.world_rank < 0 ||
                segment.local_participant_count == 0u)
            {
                return false;
            }
            for (std::size_t earlier = 0u; earlier < index; ++earlier)
            {
                if (segments_[earlier].world_rank == segment.world_rank)
                    return false;
            }
            if (segment.role ==
                MoEOverlayInferenceCoordinatorSegmentRole::Continuation)
            {
                ++continuation_count;
            }
            else
            {
                ++follower_count;
            }
        }
        return continuation_count == 1u && follower_count != 0u;
    }

    std::size_t
    MoEOverlayInferenceCoordinatorGraphPlan::followerSegmentCount()
        const noexcept
    {
        return static_cast<std::size_t>(std::count_if(
            segments_.begin(),
            segments_.end(),
            [](const Segment &segment)
            {
                return segment.role ==
                       MoEOverlayInferenceCoordinatorSegmentRole::
                           ExpertFollower;
            }));
    }

    std::size_t
    MoEOverlayInferenceCoordinatorGraphPlan::nativeSegmentCount()
        const noexcept
    {
        return static_cast<std::size_t>(std::count_if(
            segments_.begin(),
            segments_.end(),
            [](const Segment &segment)
            {
                return segment.materialization ==
                       MoEOverlayInferenceSegmentMaterializationKind::
                           NativeDeviceExecutable;
            }));
    }

    std::size_t
    MoEOverlayInferenceCoordinatorGraphPlan::eagerHostSegmentCount()
        const noexcept
    {
        return static_cast<std::size_t>(std::count_if(
            segments_.begin(),
            segments_.end(),
            [](const Segment &segment)
            {
                return segment.materialization ==
                       MoEOverlayInferenceSegmentMaterializationKind::
                           EagerHostGraph;
            }));
    }

    std::size_t
    MoEOverlayInferenceCoordinatorGraphPlan::nativeParticipantCount()
        const noexcept
    {
        std::size_t count = 0u;
        for (const Segment &segment : segments_)
        {
            if (segment.materialization ==
                MoEOverlayInferenceSegmentMaterializationKind::
                    NativeDeviceExecutable)
            {
                count += segment.local_participant_count;
            }
        }
        return count;
    }

    int MoEOverlayInferenceCoordinatorGraphPlan::continuationWorldRank()
        const noexcept
    {
        const auto found = std::find_if(
            segments_.begin(),
            segments_.end(),
            [](const Segment &segment)
            {
                return segment.role ==
                       MoEOverlayInferenceCoordinatorSegmentRole::Continuation;
            });
        return found == segments_.end() ? -1 : found->world_rank;
    }

    const MoEOverlayInferenceCoordinatorGraphPlan::Segment *
    MoEOverlayInferenceCoordinatorGraphPlan::segmentForWorldRank(
        int world_rank) const noexcept
    {
        const auto found = std::find_if(
            segments_.begin(),
            segments_.end(),
            [world_rank](const Segment &segment)
            {
                return segment.world_rank == world_rank;
            });
        return found == segments_.end() ? nullptr : &*found;
    }

    /**
     * @brief Setup-sized poll-only conjunction of participant GPU events.
     *
     * The coordinator arms this object only after every configured local
     * completion boundary has arrived. CPU participants contribute their
     * synchronous return mark to that admission decision; only GPU events are
     * retained here. The maintenance thread may poll a ready leaf repeatedly
     * while another device remains pending, so leaf events deliberately retain
     * their Ready state until the next independently admitted sample records
     * them again.
     */
    class MoEOverlayInferenceTransactionCoordinator::CompletionFenceSet final
        : public IMoEOverlayInferenceCompletionFence
    {
    public:
        /** @brief Allocate the fixed participant slots during coordinator setup. */
        explicit CompletionFenceSet(std::size_t participant_count)
            : fences_(participant_count)
        {
            if (participant_count == 0)
            {
                throw std::invalid_argument(
                    "ExpertOverlay completion fence set requires participants");
            }
        }

        /**
         * @brief Publish one complete immutable set of recorded GPU events.
         * @param events Participant-indexed events; CPU participant slots are null.
         * @param error Optional precise lifecycle diagnostic.
         * @return True when at least one recorded GPU event now owns the set.
         */
        bool arm(
            const std::vector<
                std::shared_ptr<IMoEOverlayInferenceCompletionEvent>> &events,
            std::string *error) noexcept
        {
            if (error)
                error->clear();
            if (events.size() != fences_.size() ||
                active_.load(std::memory_order_acquire))
            {
                if (error)
                {
                    *error =
                        "ExpertOverlay aggregate completion fence is active or has divergent participant geometry";
                }
                return false;
            }

            std::size_t event_count = 0;
            for (std::size_t index = 0; index < events.size(); ++index)
            {
                fences_[index] = events[index];
                event_count += events[index] ? 1u : 0u;
            }
            if (event_count == 0)
            {
                if (error)
                    *error = "ExpertOverlay aggregate completion fence has no GPU events";
                return false;
            }

            /*
             * Every shared event reference and backend record is complete
             * before this release. Probe publication supplies the matching
             * cross-thread ownership edge to maintenance.
             */
            active_.store(true, std::memory_order_release);
            return true;
        }

        /** @copydoc IMoEOverlayInferenceCompletionFence::poll */
        [[nodiscard]] MoEOverlayInferenceCompletionFenceProgress poll(
            std::string *error) noexcept override
        {
            if (error)
                error->clear();
            if (!active_.load(std::memory_order_acquire))
            {
                if (error)
                    *error = "ExpertOverlay aggregate completion fence is not armed";
                return MoEOverlayInferenceCompletionFenceProgress::Failed;
            }

            bool pending = false;
            for (const auto &fence : fences_)
            {
                if (!fence)
                    continue;
                std::string participant_error;
                const auto progress = fence->poll(&participant_error);
                if (progress ==
                    MoEOverlayInferenceCompletionFenceProgress::Failed)
                {
                    if (error)
                    {
                        *error = participant_error.empty()
                                     ? "ExpertOverlay participant completion event query failed"
                                     : std::move(participant_error);
                    }
                    return progress;
                }
                pending = pending ||
                          progress ==
                              MoEOverlayInferenceCompletionFenceProgress::Pending;
            }
            if (pending)
                return MoEOverlayInferenceCompletionFenceProgress::Pending;

            active_.store(false, std::memory_order_release);
            return MoEOverlayInferenceCompletionFenceProgress::Ready;
        }

    private:
        /** Participant-indexed, setup-sized event references. */
        std::vector<std::shared_ptr<IMoEOverlayInferenceCompletionFence>>
            fences_;
        /** True from immutable publication until every leaf reports Ready. */
        std::atomic<bool> active_{false};
    };

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
                descriptor.sidecar_depth,
                descriptor.prefill_schedule_workload);
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
        if (config_.publishers.empty() || !config_.graph_plan.valid() ||
            config_.continuation_participant_count <= 0 ||
            config_.ticket_authority_participant_index < 0 ||
            config_.ticket_authority_participant_index >=
                config_.continuation_participant_count ||
            config_.participant_completion_boundaries.size() !=
                static_cast<std::size_t>(
                    config_.continuation_participant_count) ||
            std::any_of(
                config_.participant_completion_boundaries.begin(),
                config_.participant_completion_boundaries.end(),
                [](MoEOverlayInferenceCompletionBoundaryKind boundary)
                {
                    return boundary ==
                           MoEOverlayInferenceCompletionBoundaryKind::Unspecified;
                }) ||
            config_.max_transactions_per_command == 0 ||
            config_.max_mtp_draft_depth < 0)
        {
            throw std::invalid_argument(
                "ExpertOverlay transaction coordinator requires publishers, one sealed global graph plan, continuation participants, one in-range ticket authority, and fixed command capacity");
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

        const auto *const continuation_segment =
            config_.graph_plan.segmentForWorldRank(source_rank);
        if (!continuation_segment ||
            continuation_segment->role !=
                MoEOverlayInferenceCoordinatorSegmentRole::Continuation ||
            continuation_segment->local_participant_count !=
                static_cast<std::size_t>(
                    config_.continuation_participant_count) ||
            config_.graph_plan.graphFamilyGeneration() !=
                config_.publishers.front()
                    ->topologyIdentity()
                    .workspace_generation ||
            config_.graph_plan.followerSegmentCount() !=
                config_.publishers.size() ||
            config_.graph_plan.segmentCount() !=
                config_.publishers.size() + 1u)
        {
            throw std::invalid_argument(
                "ExpertOverlay transaction coordinator graph plan disagrees with its continuation rank, participant count, graph generation, or follower cardinality");
        }
        for (const auto &publisher : config_.publishers)
        {
            const auto &topology = publisher->topologyIdentity();
            const auto *const follower_segment =
                config_.graph_plan.segmentForWorldRank(
                    topology.target_world_rank);
            if (!follower_segment ||
                follower_segment->role !=
                    MoEOverlayInferenceCoordinatorSegmentRole::
                        ExpertFollower ||
                topology.workspace_generation !=
                    config_.graph_plan.graphFamilyGeneration())
            {
                throw std::invalid_argument(
                    "ExpertOverlay transaction coordinator publisher is absent from the sealed global graph plan");
            }
        }

        graph_group_slots_.resize(
            config_.max_transactions_per_command);
        for (auto &transaction : graph_group_slots_)
        {
            transaction.target_transactions.resize(
                config_.publishers.size());
            transaction.entered_participants.resize(
                static_cast<std::size_t>(
                    config_.continuation_participant_count));
            transaction.terminal_participants.resize(
                static_cast<std::size_t>(
                    config_.continuation_participant_count));
            transaction.participant_completion_events.resize(
                static_cast<std::size_t>(
                    config_.continuation_participant_count));
            transaction.participant_completion_recorded.resize(
                static_cast<std::size_t>(
                    config_.continuation_participant_count));
        }
        prefill_completion_fence_set_ =
            std::make_shared<CompletionFenceSet>(
                static_cast<std::size_t>(
                    config_.continuation_participant_count));

        /* These records mirror the immutable plan constructed only after the
         * synchronized serving-family phase. PerfStats is not consulted by
         * admission or replay; it merely exposes that production authority to
         * integration certification and operational diagnostics. */
        const PerfStatsCollector::Tags plan_tags{
            {"authority", "typed_overlay_transaction_plan"},
            {"continuation_rank", std::to_string(source_rank)},
            {"eager_host_segments",
             std::to_string(config_.graph_plan.eagerHostSegmentCount())},
            {"follower_segments",
             std::to_string(config_.graph_plan.followerSegmentCount())},
            {"graph_family_generation",
             std::to_string(config_.graph_plan.graphFamilyGeneration())},
            {"native_participants",
             std::to_string(config_.graph_plan.nativeParticipantCount())},
            {"native_segments",
             std::to_string(config_.graph_plan.nativeSegmentCount())},
            {"scope", "cross_rank_expert_overlay"},
        };
        PerfStatsCollector::addCounter(
            "forward_graph",
            "segmented_plan_segments",
            static_cast<double>(config_.graph_plan.segmentCount()),
            "setup",
            "continuation_rank",
            plan_tags);
        if (config_.graph_plan.nativeSegmentCount() != 0u)
        {
            PerfStatsCollector::addCounter(
                "forward_graph",
                "segmented_graph_capture_segments",
                static_cast<double>(
                    config_.graph_plan.nativeSegmentCount()),
                "setup",
                "continuation_rank",
                plan_tags);
        }
    }

    bool MoEOverlayInferenceTransactionCoordinator::
        bindPrefillInterferenceProbe(
            std::shared_ptr<MoEOverlayInferenceInterferenceProbe> probe,
            std::string *error)
    {
        std::lock_guard lock(mutex_);
        if (error)
            error->clear();
        if (!probe)
        {
            return fail(
                "ExpertOverlay prefill calibration requires a non-null probe",
                error);
        }
        if (state_ != MoEOverlayInferenceProtocolState::Idle ||
            execution_sequence_.graphInFlight())
        {
            return fail(
                "ExpertOverlay prefill calibration probe must bind before command admission",
                error);
        }
        if (prefill_interference_probe_ &&
            prefill_interference_probe_ != probe)
        {
            return fail(
                "ExpertOverlay transaction coordinator cannot replace its prefill calibration probe",
                error);
        }
        prefill_interference_probe_ = std::move(probe);
        return true;
    }

    bool MoEOverlayInferenceTransactionCoordinator::
        declarePrefillInterferenceSchedule(
            MoEOverlayInferenceWorkloadIdentity workload,
            std::string *error)
    {
        std::lock_guard lock(mutex_);
        if (error)
            error->clear();
        if (!workload.valid() ||
            workload.source != ExpertHistogramSource::PrefillChunk ||
            workload.speculative_depth != 0)
        {
            return failLocked(
                "ExpertOverlay aggregate prefill declaration requires one valid non-speculative prefill workload",
                error);
        }
        if (state_ != MoEOverlayInferenceProtocolState::Active ||
            execution_sequence_.graphInFlight())
        {
            return failLocked(
                "ExpertOverlay aggregate prefill declaration requires an active command between graph groups",
                error);
        }

        const std::size_t completed_transactions =
            total_transaction_count_ + transaction_count_;
        if (declared_prefill_schedule_)
        {
            if (completed_transactions <
                    declared_prefill_schedule_end_ordinal_ ||
                (declared_prefill_probe_ticket_.valid() &&
                 !declared_prefill_probe_released_))
            {
                return failLocked(
                    "ExpertOverlay cannot replace an incomplete aggregate prefill calibration schedule",
                    error);
            }
            if (declared_prefill_probe_ticket_.valid() &&
                !declared_prefill_terminal_published_)
            {
                return failLocked(
                    "ExpertOverlay completed an aggregate prefill schedule without publishing its exact terminal fence",
                    error);
            }
        }

        const auto schedule_transactions = static_cast<std::size_t>(
            workload.transaction_count);
        if (schedule_transactions == 0 ||
            completed_transactions >
                std::numeric_limits<std::size_t>::max() -
                    schedule_transactions)
        {
            return failLocked(
                "ExpertOverlay aggregate prefill transaction cardinality overflowed",
                error);
        }

        declared_prefill_schedule_ = workload;
        declared_prefill_schedule_begin_ordinal_ =
            completed_transactions + 1u;
        declared_prefill_schedule_end_ordinal_ =
            completed_transactions + schedule_transactions;
        declared_prefill_probe_ticket_ =
            prefill_interference_probe_
                ? prefill_interference_probe_->beginSample(workload)
                : MoEOverlayInterferenceProbeTicket{};
        declared_prefill_probe_released_ =
            !declared_prefill_probe_ticket_.valid();
        declared_prefill_terminal_published_ = false;

        PerfStatsCollector::addCounter(
            "moe_overlay_residency",
            "aggregate_prefill_calibration_schedules",
            1.0,
            "prefill",
            "continuation_rank",
            {{"real_rows", std::to_string(workload.real_rows)},
             {"execution_rows",
              std::to_string(workload.execution_rows)},
             {"transactions",
              std::to_string(workload.transaction_count)},
             {"probe_claimed",
              declared_prefill_probe_ticket_.valid() ? "true" : "false"}});
        return true;
    }

    bool MoEOverlayInferenceTransactionCoordinator::failLocked(
        std::string message,
        std::string *error)
    {
        if (failure_.empty())
            failure_ = std::move(message);
        state_ = MoEOverlayInferenceProtocolState::Failed;
        execution_sequence_.state = ExecutionSequenceState::Failed;
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

    bool MoEOverlayInferenceTransactionCoordinator::
        tryPublishPrefillInterferenceCompletionLocked(
            GraphGroupSlot &transaction,
            std::string *error)
    {
        if (!transaction.interference_ticket.valid() ||
            transaction.interference_completion_published ||
            !allParticipantsMarked(
                transaction.participant_completion_recorded))
        {
            return true;
        }
        if (!prefill_interference_probe_ ||
            !prefill_completion_fence_set_)
        {
            return failLocked(
                "ExpertOverlay prefill completion lost its setup-owned probe or aggregate fence",
                error);
        }

        bool has_device_event = false;
        for (std::size_t participant = 0;
             participant <
             config_.participant_completion_boundaries.size();
             ++participant)
        {
            const auto boundary =
                config_.participant_completion_boundaries[participant];
            const bool has_event = static_cast<bool>(
                transaction.participant_completion_events[participant]);
            if (boundary ==
                MoEOverlayInferenceCompletionBoundaryKind::DeviceEvent)
            {
                if (!has_event)
                {
                    return failLocked(
                        "ExpertOverlay GPU participant terminal mark has no recorded event",
                        error);
                }
                has_device_event = true;
            }
            else if (has_event)
            {
                return failLocked(
                    "ExpertOverlay CPU participant published a GPU completion event",
                    error);
            }
        }

        bool published = false;
        if (has_device_event)
        {
            std::string publication_error;
            published = prefill_completion_fence_set_->arm(
                            transaction.participant_completion_events,
                            &publication_error) &&
                        prefill_interference_probe_->deferSampleCompletion(
                            transaction.interference_ticket,
                            prefill_completion_fence_set_,
                            &publication_error);
            if (!published)
            {
                return failLocked(
                    publication_error.empty()
                        ? "ExpertOverlay could not publish its aggregate participant completion fence"
                        : std::move(publication_error),
                    error);
            }
        }
        else
        {
            published = prefill_interference_probe_->finishSample(
                transaction.interference_ticket);
            if (!published)
            {
                return failLocked(
                    "ExpertOverlay CPU participants could not publish their exact synchronous completion interval",
                    error);
            }
        }
        transaction.interference_completion_published = true;
        LOG_DEBUG(
            "[ExpertOverlay][Calibration] Published rank-wide prefill terminal sequence="
            << transaction.interference_ticket.calibration_sequence
            << " participants="
            << transaction.participant_completion_recorded.size()
            << " device_events=" << (has_device_event ? "true" : "false")
            << " logical_step=" << transaction.descriptor.logical_step_id);
        return true;
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
        active_command_ = command;
        total_transaction_count_ = 0;
        graph_sequence_count_ = 0;
        last_hosted_sequence_transition_id_ = 0;
        last_hosted_sequence_next_draft_depth_.reset();
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
        if (execution_sequence_.active() || transaction_count_ != 0)
            return failLocked(
                "ExpertOverlay graph sequence cannot overlap live graph slots",
                error);
        if (draft_depth < 0 ||
            draft_depth > config_.max_mtp_draft_depth ||
            next_sequence_id_ == 0 ||
            next_sequence_id_ ==
                std::numeric_limits<std::uint64_t>::max())
        {
            return failLocked(
                "ExpertOverlay MTP depth or sequence identity exceeds the retained graph family",
                error);
        }

        std::optional<MoEOverlayResidencyAuthority::TicketLease>
            placement_epoch_lease;
        std::uint64_t placement_epoch = current_placement_epoch_;
        if (config_.residency_authority)
        {
            placement_epoch_lease = config_.residency_authority
                                        ->tryAcquireGraphSequenceSnapshot();
            if (!placement_epoch_lease || !*placement_epoch_lease ||
                placement_epoch_lease->epoch() == 0 ||
                placement_epoch_lease->epoch() < current_placement_epoch_)
            {
                return failLocked(
                    "ExpertOverlay graph sequence could not pin a monotonic host residency epoch",
                    error);
            }
            placement_epoch = placement_epoch_lease->epoch();
        }
        execution_sequence_.state = ExecutionSequenceState::Open;
        execution_sequence_.draft_depth = draft_depth;
        execution_sequence_.next_graph_ordinal = 0;
        execution_sequence_.placement_epoch = placement_epoch;
        execution_sequence_.sequence_id = next_sequence_id_++;
        execution_sequence_.placement_epoch_lease =
            std::move(placement_epoch_lease);
        current_placement_epoch_ = placement_epoch;
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
        if (execution_sequence_.graphInFlight() &&
            transaction_count_ < graph_group_slots_.size() &&
            graph_group_slots_[transaction_count_]
                    .entered_participants[participant] != 0)
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
                           !execution_sequence_.graphInFlight() ||
                           (transaction_count_ < graph_group_slots_.size() &&
                            graph_group_slots_[transaction_count_]
                                    .entered_participants[participant] == 0);
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

        if (execution_sequence_.graphInFlight())
        {
            /*
             * LocalTP siblings call this independently before joining the same
             * graph group. Once the first participant has reserved the group,
             * every later participant must observe exactly a MainPrefill group.
             * Ticket publication remains deferred until the first exact launch
             * edge, and no sequence transition is needed until all siblings
             * finish.
             */
            if (transaction_count_ >= graph_group_slots_.size() ||
                !graph_group_slots_[transaction_count_].inFlight() ||
                graph_group_slots_[transaction_count_]
                        .descriptor.graph_role !=
                    MoEOverlayInferenceGraphRole::MainPrefill)
            {
                return failLocked(
                    "ExpertOverlay serial prefill attempted to join a non-prefill graph group",
                    error);
            }
            return true;
        }

        if (!execution_sequence_.active())
            return beginGraphSequenceLocked(/*draft_depth=*/0, error);

        if (execution_sequence_.draft_depth != 0)
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
            execution_sequence_.state == ExecutionSequenceState::Open &&
            execution_sequence_.next_graph_ordinal == 1 &&
            transaction_count_ == 1;
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
        std::unique_lock lock(mutex_);
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
        if (!execution_sequence_.active())
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

        const std::size_t participant =
            static_cast<std::size_t>(participant_index);
        if (execution_sequence_.graphInFlight() &&
            transaction_count_ < graph_group_slots_.size() &&
            graph_group_slots_[transaction_count_]
                    .entered_participants[participant] != 0)
        {
            /*
             * Retained graph launch is asynchronous. A faster LocalTP worker
             * can therefore finish submitting graph N and request graph N+1
             * while a sibling is still publishing N's completion boundary.
             * Keep both workers alive and wait only on coordinator metadata;
             * already-enqueued device work remains fully asynchronous. This is
             * the same bounded heterogeneous-boundary admission used by serial
             * prefill, generalized to every sparse graph role.
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
                           !execution_sequence_.graphInFlight() ||
                           (transaction_count_ < graph_group_slots_.size() &&
                            graph_group_slots_[transaction_count_]
                                    .entered_participants[participant] == 0);
                });
            if (!aligned)
            {
                failLocked(
                    "ExpertOverlay participant timed out waiting for symmetric retained-graph submission alignment",
                    &result.error);
                return result;
            }
            if (state_ != MoEOverlayInferenceProtocolState::Active)
            {
                result.error = failure_.empty()
                                   ? "ExpertOverlay participant alignment observed a terminal coordinator"
                                   : failure_;
                return result;
            }
        }

        const bool joining_active_group =
            execution_sequence_.graphInFlight();
        if (!joining_active_group)
        {
            const int ordinal = execution_sequence_.next_graph_ordinal;
            const int expected_count =
                execution_sequence_.expectedGraphCount();
            const bool serial_role =
                descriptor.graph_role ==
                    MoEOverlayInferenceGraphRole::MainPrefill ||
                descriptor.graph_role ==
                    MoEOverlayInferenceGraphRole::MainDecode;
            const bool expected_role =
                execution_sequence_.draft_depth == 0
                    ? ordinal == 0 && serial_role
                    : ordinal < execution_sequence_.draft_depth
                          ? descriptor.graph_role ==
                                MoEOverlayInferenceGraphRole::MTPDraft
                          : ordinal == execution_sequence_.draft_depth &&
                                descriptor.graph_role ==
                                    MoEOverlayInferenceGraphRole::
                                        MTPGroupedVerifier;
            if (ordinal < 0 || ordinal >= expected_count || !expected_role)
            {
                failLocked(
                    "ExpertOverlay graph role is out of order for its immutable execution-sequence plan",
                    &result.error);
                return result;
            }
        }

        switch (descriptor.graph_role)
        {
        case MoEOverlayInferenceGraphRole::MTPDraft:
        {
            const int group_sidecar_ordinal =
                joining_active_group &&
                        transaction_count_ < graph_group_slots_.size()
                    ? graph_group_slots_[transaction_count_]
                          .descriptor.sidecar_depth
                    : execution_sequence_.next_graph_ordinal;
            if (execution_sequence_.draft_depth <= 0 ||
                group_sidecar_ordinal < 0 ||
                group_sidecar_ordinal >= execution_sequence_.draft_depth)
            {
                failLocked(
                    "ExpertOverlay sidecar count exceeds the admitted MTP depth",
                    &result.error);
                return result;
            }
            descriptor.draft_depth = execution_sequence_.draft_depth;
            descriptor.sidecar_depth = group_sidecar_ordinal;
            break;
        }
        case MoEOverlayInferenceGraphRole::MTPGroupedVerifier:
            if (execution_sequence_.draft_depth <= 0 ||
                descriptor.draft_depth != execution_sequence_.draft_depth ||
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
            if (execution_sequence_.draft_depth != 0 ||
                descriptor.draft_depth != -1 ||
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

        if (descriptor.prefill_schedule_workload.valid())
        {
            failLocked(
                "ExpertOverlay graph caller attempted to inject coordinator-owned aggregate prefill identity",
                &result.error);
            return result;
        }
        const std::size_t absolute_transaction_ordinal =
            total_transaction_count_ + transaction_count_ + 1u;
        const bool inside_declared_prefill_schedule =
            declared_prefill_schedule_ &&
            absolute_transaction_ordinal >=
                declared_prefill_schedule_begin_ordinal_ &&
            absolute_transaction_ordinal <=
                declared_prefill_schedule_end_ordinal_;
        if (inside_declared_prefill_schedule)
        {
            if (descriptor.graph_role !=
                MoEOverlayInferenceGraphRole::MainPrefill)
            {
                failLocked(
                    "ExpertOverlay aggregate prefill schedule encountered a non-prefill graph before its terminal ordinal",
                    &result.error);
                return result;
            }
            descriptor.prefill_schedule_workload =
                *declared_prefill_schedule_;
        }

        GraphGroupSlot *transaction = nullptr;
        if (joining_active_group)
        {
            if (transaction_count_ >= graph_group_slots_.size())
            {
                failLocked(
                    "ExpertOverlay active graph group exceeds fixed transaction storage",
                    &result.error);
                return result;
            }
            transaction = &graph_group_slots_[transaction_count_];
            descriptor.logical_step_id =
                transaction->descriptor.logical_step_id;
            if (!transaction->inFlight() ||
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
            if (transaction_count_ >= graph_group_slots_.size() ||
                next_logical_step_id_ == 0 ||
                next_logical_step_id_ ==
                    std::numeric_limits<std::uint64_t>::max())
            {
                failLocked(
                    "ExpertOverlay command exhausted fixed transaction or operation-id capacity",
                    &result.error);
                return result;
            }
            transaction = &graph_group_slots_[transaction_count_];
            descriptor.logical_step_id = next_logical_step_id_++;
            transaction->state = GraphGroupSlotState::Admitting;
            transaction->group_id = next_group_id_++;
            transaction->sequence_id = execution_sequence_.sequence_id;
            transaction->sequence_graph_ordinal =
                execution_sequence_.next_graph_ordinal;
            transaction->sequence_graph_count =
                execution_sequence_.expectedGraphCount();
            transaction->descriptor = descriptor;
            transaction->interference_ticket = {};
            transaction->interference_completion_published = false;
            std::fill(
                transaction->participant_completion_events.begin(),
                transaction->participant_completion_events.end(),
                nullptr);
            std::fill(
                transaction->participant_completion_recorded.begin(),
                transaction->participant_completion_recorded.end(),
                std::uint8_t{0});
            std::fill(
                transaction->entered_participants.begin(),
                transaction->entered_participants.end(), std::uint8_t{0});
            std::fill(
                transaction->terminal_participants.begin(),
                transaction->terminal_participants.end(), std::uint8_t{0});

            execution_sequence_.state =
                ExecutionSequenceState::GraphInFlight;
            if (execution_sequence_.placement_epoch == 0)
            {
                execution_sequence_.placement_epoch =
                    descriptor.placement_epoch;
            }
            else if (execution_sequence_.placement_epoch !=
                     descriptor.placement_epoch)
            {
                failLocked(
                    "ExpertOverlay graph changed residency epoch inside one execution sequence",
                    &result.error);
                return result;
            }
            current_placement_epoch_ = std::max(
                current_placement_epoch_, descriptor.placement_epoch);

            if (descriptor.graph_role ==
                    MoEOverlayInferenceGraphRole::MainPrefill &&
                inside_declared_prefill_schedule)
            {
                /*
                 * Earlier segments carry the authenticated aggregate identity
                 * but no timing terminal.  The final segment alone inherits
                 * the schedule ticket so its exact participant event set closes
                 * the complete interval without a host synchronization.
                 */
                if (absolute_transaction_ordinal ==
                    declared_prefill_schedule_end_ordinal_)
                {
                    transaction->interference_ticket =
                        declared_prefill_probe_ticket_;
                }
            }
            else if (descriptor.graph_role ==
                         MoEOverlayInferenceGraphRole::MainPrefill &&
                     prefill_interference_probe_)
            {
                const std::int64_t real_rows =
                    static_cast<std::int64_t>(descriptor.request_count) *
                    descriptor.logical_rows_per_request;
                const std::int64_t execution_rows =
                    static_cast<std::int64_t>(descriptor.request_count) *
                    descriptor.physical_rows_per_request;
                if (real_rows <= 0 || execution_rows < real_rows ||
                    real_rows > std::numeric_limits<int>::max() ||
                    execution_rows > std::numeric_limits<int>::max())
                {
                    failLocked(
                        "ExpertOverlay prefill calibration workload geometry overflowed",
                        &result.error);
                    return result;
                }
                transaction->interference_ticket =
                    prefill_interference_probe_->beginSample(
                        makeMoEOverlayInferenceWorkloadIdentity(
                            ExpertHistogramSource::PrefillChunk,
                            static_cast<int>(real_rows),
                            static_cast<int>(execution_rows),
                            /*transaction_count=*/1,
                            /*speculative_depth=*/0));
            }
        }

        if (transaction->entered_participants[participant] != 0)
        {
            failLocked(
                "ExpertOverlay continuation participant entered one graph twice",
                &result.error);
            return result;
        }
        transaction->entered_participants[participant] = 1;
        result.ok = true;
        result.active = true;
        result.owns_ticket_authority =
            participant_index ==
            config_.ticket_authority_participant_index;
        result.group_id = transaction->group_id;
        result.request_generation = active_command_.request_generation;
        result.sequence_id = transaction->sequence_id;
        result.sequence_graph_ordinal =
            transaction->sequence_graph_ordinal;
        result.sequence_graph_count = transaction->sequence_graph_count;
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
            !execution_sequence_.graphInFlight() ||
            transaction_count_ >= graph_group_slots_.size())
        {
            return failLocked(
                "ExpertOverlay participant attempted to arm an inactive graph group",
                error);
        }

        GraphGroupSlot &transaction =
            graph_group_slots_[transaction_count_];
        if (!transaction.inFlight() || transaction.failed() ||
            binding.group_id != transaction.group_id ||
            binding.sequence_id != transaction.sequence_id ||
            binding.sequence_graph_ordinal !=
                transaction.sequence_graph_ordinal ||
            binding.sequence_graph_count !=
                transaction.sequence_graph_count ||
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
        if (transaction.entered_participants[participant] == 0 ||
            transaction.terminal_participants[participant] != 0)
        {
            return failLocked(
                "ExpertOverlay executable launch occurred outside its participant graph scope",
                error);
        }
        if (transaction.armed())
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
        transaction.state = GraphGroupSlotState::Armed;
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

    bool MoEOverlayInferenceTransactionCoordinator::
        deferPrefillInterferenceCompletionAtDeviceTerminal(
            std::uint64_t logical_step_id,
            int participant_index,
            std::shared_ptr<IMoEOverlayInferenceCompletionEvent> event,
            void *producer_stream,
            std::string *error)
    {
        std::lock_guard lock(mutex_);
        if (error)
            error->clear();
        if (state_ != MoEOverlayInferenceProtocolState::Active ||
            !execution_sequence_.graphInFlight() ||
            transaction_count_ >= graph_group_slots_.size())
        {
            return failLocked(
                "ExpertOverlay device terminal reached an inactive graph group",
                error);
        }

        GraphGroupSlot &transaction =
            graph_group_slots_[transaction_count_];
        if (!transaction.inFlight() ||
            transaction.descriptor.graph_role !=
                MoEOverlayInferenceGraphRole::MainPrefill ||
            transaction.descriptor.logical_step_id != logical_step_id ||
            participant_index < 0 ||
            participant_index >=
                config_.continuation_participant_count ||
            config_.participant_completion_boundaries[
                static_cast<std::size_t>(participant_index)] !=
                MoEOverlayInferenceCompletionBoundaryKind::DeviceEvent ||
            !event || !producer_stream)
        {
            return failLocked(
                "ExpertOverlay prefill device terminal disagrees with its active participant or completion policy",
                error);
        }

        /*
         * Most graphs run while no calibration request is armed. Avoid even
         * recording the reusable event in that ordinary path. A claimed ticket
         * is single-shot, so duplicate terminal publication is a protocol bug.
         */
        if (!transaction.interference_ticket.valid())
            return true;
        const std::size_t participant =
            static_cast<std::size_t>(participant_index);
        if (transaction.participant_completion_recorded[participant] != 0)
        {
            return failLocked(
                "ExpertOverlay prefill participant published its device terminal twice",
                error);
        }
        std::string record_error;
        if (!event->record(producer_stream, &record_error))
        {
            return failLocked(
                record_error.empty()
                    ? "ExpertOverlay prefill participant could not record its device-terminal calibration event"
                    : std::move(record_error),
                error);
        }
        transaction.participant_completion_events[participant] =
            std::move(event);
        transaction.participant_completion_recorded[participant] = 1;
        return tryPublishPrefillInterferenceCompletionLocked(
            transaction, error);
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
            !execution_sequence_.graphInFlight() ||
            transaction_count_ >= graph_group_slots_.size())
        {
            return failLocked(
                "ExpertOverlay participant attempted to finish an inactive graph group",
                error);
        }
        GraphGroupSlot &transaction =
            graph_group_slots_[transaction_count_];
        if (binding.group_id != transaction.group_id ||
            binding.sequence_id != transaction.sequence_id ||
            binding.sequence_graph_ordinal !=
                transaction.sequence_graph_ordinal ||
            binding.sequence_graph_count !=
                transaction.sequence_graph_count ||
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
        if (!transaction.inFlight() ||
            transaction.entered_participants[participant] == 0 ||
            transaction.terminal_participants[participant] != 0)
        {
            return failLocked(
                "ExpertOverlay participant graph finish lifecycle is out of order",
                error);
        }
        if (execution_succeeded &&
            binding.participant_index ==
                config_.ticket_authority_participant_index &&
            !transaction.armed())
        {
            return failLocked(
                "ExpertOverlay ticket authority reported successful graph execution before its launch ticket was armed",
                error);
        }

        if (execution_succeeded &&
            transaction.interference_ticket.valid())
        {
            const auto boundary =
                config_.participant_completion_boundaries[participant];
            if (boundary ==
                MoEOverlayInferenceCompletionBoundaryKind::HostSynchronous)
            {
                transaction.participant_completion_recorded[participant] = 1;
            }
            else if (
                transaction.participant_completion_recorded[participant] == 0)
            {
                return failLocked(
                    "ExpertOverlay GPU prefill participant returned before publishing its exact device-terminal event",
                    error);
            }
            if (!tryPublishPrefillInterferenceCompletionLocked(
                    transaction, error))
            {
                return false;
            }
        }
        transaction.terminal_participants[participant] = 1;
        if (!execution_succeeded)
        {
            transaction.state = transaction.armed()
                                    ? GraphGroupSlotState::ArmedFailed
                                    : GraphGroupSlotState::AdmittingFailed;
        }

        if (allParticipantsMarked(transaction.terminal_participants))
        {
            if (!allParticipantsMarked(transaction.entered_participants))
            {
                return failLocked(
                    "ExpertOverlay graph group finished without every continuation participant",
                    error);
            }
            if (!transaction.failed() && !transaction.armed())
            {
                return failLocked(
                    "ExpertOverlay graph group completed successfully without its ticket authority arming the remote transaction",
                    error);
            }
            if (transaction.interference_ticket.valid())
            {
                const bool aggregate_prefill_terminal =
                    declared_prefill_schedule_ &&
                    transaction.descriptor.prefill_schedule_workload ==
                        *declared_prefill_schedule_ &&
                    total_transaction_count_ + transaction_count_ + 1u ==
                        declared_prefill_schedule_end_ordinal_ &&
                    transaction.interference_ticket.probe_generation ==
                        declared_prefill_probe_ticket_.probe_generation;
                bool probe_ok = true;
                if (transaction.failed())
                {
                    if (transaction.interference_completion_published)
                    {
                        return failLocked(
                            "ExpertOverlay graph failed after publishing its aggregate calibration terminal",
                            error);
                    }
                    probe_ok = prefill_interference_probe_->discardSample(
                        transaction.interference_ticket);
                }
                else
                {
                    probe_ok =
                        transaction.interference_completion_published;
                }
                transaction.interference_ticket = {};
                if (aggregate_prefill_terminal)
                {
                    declared_prefill_probe_released_ = probe_ok;
                    declared_prefill_terminal_published_ =
                        probe_ok && !transaction.failed();
                }
                if (!probe_ok)
                {
                    return failLocked(
                        "ExpertOverlay prefill graph group finished without every participant publishing its exact calibration terminal",
                        error);
                }
            }
            if (transaction.failed())
            {
                transaction.state = GraphGroupSlotState::Failed;
                execution_sequence_.state = ExecutionSequenceState::Failed;
            }
            else
            {
                transaction.state = GraphGroupSlotState::Terminal;
                ++execution_sequence_.next_graph_ordinal;
                execution_sequence_.state = ExecutionSequenceState::Open;
            }
            ++transaction_count_;
            graph_group_completion_cv_.notify_all();
        }
        return true;
    }

    void MoEOverlayInferenceTransactionCoordinator::resetGraphSequenceLocked()
        noexcept
    {
        for (auto &transaction : graph_group_slots_)
        {
            transaction.state = GraphGroupSlotState::Available;
            transaction.group_id = 0;
            transaction.sequence_id = 0;
            transaction.sequence_graph_ordinal = -1;
            transaction.sequence_graph_count = 0;
            transaction.descriptor = {};
            transaction.interference_ticket = {};
            transaction.interference_completion_published = false;
            std::fill(
                transaction.entered_participants.begin(),
                transaction.entered_participants.end(),
                std::uint8_t{0});
            std::fill(
                transaction.terminal_participants.begin(),
                transaction.terminal_participants.end(),
                std::uint8_t{0});
            std::fill(
                transaction.participant_completion_events.begin(),
                transaction.participant_completion_events.end(),
                nullptr);
            std::fill(
                transaction.participant_completion_recorded.begin(),
                transaction.participant_completion_recorded.end(),
                std::uint8_t{0});
            std::fill(
                transaction.target_transactions.begin(),
                transaction.target_transactions.end(),
                MoEOverlayPublishedInferenceTransaction{});
        }
        transaction_count_ = 0;
        execution_sequence_.reset();
    }

    bool MoEOverlayInferenceTransactionCoordinator::
        retireCompletedGraphSequenceLocked(std::string *error)
    {
        if (state_ != MoEOverlayInferenceProtocolState::Active ||
            execution_sequence_.state != ExecutionSequenceState::Open ||
            transaction_count_ == 0 ||
            execution_sequence_.next_graph_ordinal !=
                execution_sequence_.expectedGraphCount() ||
            transaction_count_ != static_cast<std::size_t>(
                                      execution_sequence_.expectedGraphCount()))
        {
            return failLocked(
                "ExpertOverlay graph sequence cannot retire before every local graph and sparse return complete",
                error);
        }

        execution_sequence_.state = ExecutionSequenceState::Releasing;
        const int retired_draft_depth = execution_sequence_.draft_depth;
        const std::size_t retired_count = transaction_count_;
        std::uint64_t retired_prefill_tokens = 0u;
        bool all_prefill = true;
        bool all_serial_decode = true;
        for (std::size_t transaction_index = 0;
             transaction_index < retired_count;
             ++transaction_index)
        {
            auto &transaction = graph_group_slots_[transaction_index];
            if (transaction.state != GraphGroupSlotState::Terminal ||
                !transaction.armed())
            {
                return failLocked(
                    "ExpertOverlay graph sequence contains an uninitialized or unarmed retained transaction",
                    error);
            }
            if (transaction.descriptor.graph_role ==
                MoEOverlayInferenceGraphRole::MainPrefill)
            {
                retired_prefill_tokens +=
                    static_cast<std::uint64_t>(
                        transaction.descriptor.request_count) *
                    static_cast<std::uint64_t>(
                        transaction.descriptor.logical_rows_per_request);
            }
            else
            {
                all_prefill = false;
            }
            if (transaction.descriptor.graph_role !=
                MoEOverlayInferenceGraphRole::MainDecode)
            {
                all_serial_decode = false;
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
        if (retired_prefill_tokens != 0u &&
            config_.retired_prefill_progress_sink)
        {
            std::string progress_error;
            if (!config_.retired_prefill_progress_sink(
                    retired_prefill_tokens, &progress_error))
            {
                return failLocked(
                    progress_error.empty()
                        ? "ExpertOverlay continuation prefill progress sideband rejected a retired transaction"
                        : std::move(progress_error),
                    error);
            }
        }
        if (retired_count >
            std::numeric_limits<std::size_t>::max() /
                config_.graph_plan.segmentCount())
        {
            return failLocked(
                "ExpertOverlay completed segment replay cardinality overflowed",
                error);
        }
        const std::size_t completed_segments =
            retired_count * config_.graph_plan.segmentCount();
        total_transaction_count_ += retired_count;
        const char *const replay_phase =
            all_prefill ? "prefill"
                        : (all_serial_decode ? "decode" : "mtp");

        /* Slot retirement follows the continuation graph's exact sparse-return
         * fence. Reaching this edge proves both the local captured submissions
         * and every authenticated follower transaction completed; earlier arm
         * or terminal-submission states are deliberately insufficient. */
        PerfStatsCollector::addCounter(
            "forward_graph",
            "segmented_replay_segments",
            static_cast<double>(completed_segments),
            replay_phase,
            "continuation_rank",
            {{"authority", "typed_overlay_transaction_plan"},
             {"command", std::to_string(active_command_.command_id)},
             {"draft_depth", std::to_string(retired_draft_depth)},
             {"graph_groups", std::to_string(retired_count)},
             {"plan_segments",
              std::to_string(config_.graph_plan.segmentCount())},
             {"scope", "cross_rank_expert_overlay"},
             {"terminal", "sparse_return_retired"}});
        PerfStatsCollector::addCounter(
            "moe_overlay_transaction",
            "coordinator_graph_sequences",
            1.0,
            "inference",
            "continuation_rank",
            {{"action", "retire"},
             {"command", std::to_string(active_command_.command_id)},
             {"draft_depth", std::to_string(retired_draft_depth)},
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

    bool MoEOverlayInferenceTransactionCoordinator::
        advanceHostedGraphSequence(
            std::uint64_t transaction_id,
            std::optional<int> next_draft_depth,
            std::string *error)
    {
        std::lock_guard lock(mutex_);
        if (error)
            error->clear();

        if (transaction_id == 0)
        {
            return failLocked(
                "ExpertOverlay hosted graph transition requires a positive authenticated transaction id",
                error);
        }
        if (next_draft_depth &&
            (*next_draft_depth <= 0 ||
             *next_draft_depth > config_.max_mtp_draft_depth))
        {
            return failLocked(
                "ExpertOverlay hosted graph transition selected a depth outside the retained MTP family",
                error);
        }

        /*
         * Rank-local participants enter independently on persistent workers.
         * Once one participant applies a ticket, every sibling must observe the
         * exact same decision as an idempotent read. This replaces the former
         * participant-zero mutation/sibling-sampling race.
         */
        if (transaction_id == last_hosted_sequence_transition_id_)
        {
            if (next_draft_depth !=
                last_hosted_sequence_next_draft_depth_)
            {
                return failLocked(
                    "ExpertOverlay symmetric participants presented divergent hosted graph transitions",
                    error);
            }
            return true;
        }
        if (transaction_id < last_hosted_sequence_transition_id_)
        {
            return failLocked(
                "ExpertOverlay hosted graph transition replayed a stale transaction id",
                error);
        }

        if (!retireCompletedGraphSequenceLocked(error))
            return false;
        if (next_draft_depth &&
            !beginGraphSequenceLocked(*next_draft_depth, error))
        {
            return false;
        }

        last_hosted_sequence_transition_id_ = transaction_id;
        last_hosted_sequence_next_draft_depth_ = next_draft_depth;
        PerfStatsCollector::addCounter(
            "moe_overlay_transaction",
            "hosted_sequence_transitions",
            1.0,
            "decode",
            "continuation_rank",
            {{"transaction", std::to_string(transaction_id)},
             {"terminal", next_draft_depth ? "false" : "true"},
             {"next_depth",
              next_draft_depth
                  ? std::to_string(*next_draft_depth)
                  : "terminal"},
             {"authority", "idempotent_ticket"}});
        return true;
    }

    bool MoEOverlayInferenceTransactionCoordinator::completeCommand(
        std::uint64_t placement_epoch,
        std::string *error)
    {
        std::lock_guard lock(mutex_);
        if (error)
            error->clear();
        if (state_ != MoEOverlayInferenceProtocolState::Active ||
            execution_sequence_.state ==
                ExecutionSequenceState::GraphInFlight ||
            execution_sequence_.state == ExecutionSequenceState::Releasing ||
            execution_sequence_.state == ExecutionSequenceState::Failed ||
            placement_epoch < current_placement_epoch_)
        {
            return failLocked(
                "ExpertOverlay command cannot complete with unfinished or failed graph work",
                error);
        }

        if (execution_sequence_.active() &&
            !retireCompletedGraphSequenceLocked(error))
        {
            return false;
        }

        if (declared_prefill_schedule_)
        {
            if (total_transaction_count_ !=
                declared_prefill_schedule_end_ordinal_)
            {
                return failLocked(
                    "ExpertOverlay command completed before its declared aggregate prefill schedule cardinality",
                    error);
            }
            if (declared_prefill_probe_ticket_.valid() &&
                (!declared_prefill_probe_released_ ||
                 !declared_prefill_terminal_published_))
            {
                return failLocked(
                    "ExpertOverlay command completed before the aggregate prefill probe owned its exact final-device terminal",
                    error);
            }
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
        declared_prefill_schedule_.reset();
        declared_prefill_schedule_begin_ordinal_ = 0;
        declared_prefill_schedule_end_ordinal_ = 0;
        declared_prefill_probe_ticket_ = {};
        declared_prefill_probe_released_ = false;
        declared_prefill_terminal_published_ = false;
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
            if (declared_prefill_probe_ticket_.valid() &&
                !declared_prefill_probe_released_ &&
                prefill_interference_probe_)
            {
                if (!prefill_interference_probe_->discardSample(
                        declared_prefill_probe_ticket_))
                {
                    ok = false;
                    first_error =
                        "ExpertOverlay abort could not release its aggregate prefill probe ticket";
                }
                else
                {
                    declared_prefill_probe_released_ = true;
                }
            }
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
            resetGraphSequenceLocked();
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
        return execution_sequence_.state == ExecutionSequenceState::Idle
                   ? -1
                   : execution_sequence_.draft_depth;
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
          protocol_(std::move(config.protocol)),
          interference_probe_(std::move(config.interference_probe)),
          retired_prefill_progress_sink_(
              std::move(config.retired_prefill_progress_sink))
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
        std::optional<MoEOverlayInferenceWorkloadIdentity>
            active_prefill_schedule;
        std::optional<MoEOverlayInferenceInterferenceScope>
            active_prefill_scope;
        int active_prefill_transactions = 0;
        const auto discard_active_prefill = [&]() noexcept
        {
            if (active_prefill_scope)
                active_prefill_scope->discard();
            active_prefill_scope.reset();
            active_prefill_schedule.reset();
            active_prefill_transactions = 0;
        };
        while (true)
        {
            const auto received = channel_->receive();
            if (!received.ok)
            {
                discard_active_prefill();
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
                    discard_active_prefill();
                    return result;
                }
            }

            const MoEOverlayInferenceAdmission admission =
                protocol_.accept(ticket);
            if (admission.status ==
                MoEOverlayInferenceAdmissionStatus::Complete)
            {
                if (active_prefill_schedule)
                {
                    discard_active_prefill();
                    result.error =
                        "ExpertOverlay follower received Complete before its aggregate prefill schedule terminal";
                    return result;
                }
                result.ok = true;
                return result;
            }
            if (admission.status ==
                MoEOverlayInferenceAdmissionStatus::Aborted)
            {
                discard_active_prefill();
                result.aborted = true;
                result.error_code = ticket.error_code;
                result.error = "Continuation authority aborted ExpertOverlay transaction command";
                return result;
            }
            if (!admission.accepted())
            {
                discard_active_prefill();
                result.error = admission.error.empty()
                                   ? "ExpertOverlay follower rejected a transaction ticket"
                                   : admission.error;
                return result;
            }

            if (!protocol_.markSubmitted(
                    admission.slot_index, ticket, &result.error))
            {
                discard_active_prefill();
                return result;
            }
            const auto protocol_end = std::chrono::steady_clock::now();
            recordTicketTiming(
                ticket,
                "follower_protocol_admission",
                "follower",
                protocol_begin,
                protocol_end);

            const auto prefill_schedule =
                ticket.prefillScheduleWorkload();
            if (prefill_schedule.valid())
            {
                if (!active_prefill_schedule)
                {
                    active_prefill_schedule = prefill_schedule;
                    active_prefill_transactions = 0;
                    active_prefill_scope.emplace(
                        interference_probe_.get(), prefill_schedule);
                }
                else if (*active_prefill_schedule != prefill_schedule)
                {
                    discard_active_prefill();
                    result.error =
                        "ExpertOverlay follower observed a changing aggregate prefill workload before its declared terminal";
                    return result;
                }
                if (active_prefill_transactions >=
                    prefill_schedule.transaction_count)
                {
                    discard_active_prefill();
                    result.error =
                        "ExpertOverlay follower aggregate prefill schedule exceeded its authenticated transaction count";
                    return result;
                }
            }
            else if (active_prefill_schedule)
            {
                discard_active_prefill();
                result.error =
                    "ExpertOverlay follower aggregate prefill schedule was interrupted by an unscoped transaction";
                return result;
            }

            std::optional<MoEOverlayInferenceInterferenceScope>
                transaction_scope;
            if (!prefill_schedule.valid())
            {
                ExpertHistogramSource source =
                    ExpertHistogramSource::SyntheticTest;
                int speculative_depth = 0;
                if (ticket.graph_role ==
                    MoEOverlayInferenceGraphRole::MainPrefill)
                {
                    source = ExpertHistogramSource::PrefillChunk;
                }
                else if (ticket.graph_role ==
                         MoEOverlayInferenceGraphRole::MainDecode)
                {
                    source = ExpertHistogramSource::DecodeToken;
                }
                else if (ticket.graph_role ==
                         MoEOverlayInferenceGraphRole::MTPGroupedVerifier)
                {
                    source = ExpertHistogramSource::GroupedVerifier;
                    speculative_depth = ticket.draft_depth;
                }
                if (source != ExpertHistogramSource::SyntheticTest)
                {
                    transaction_scope.emplace(
                        interference_probe_.get(),
                        makeMoEOverlayInferenceWorkloadIdentity(
                            source,
                            ticket.request_count *
                                ticket.logical_rows_per_request,
                            ticket.request_count *
                                ticket.physical_rows_per_request,
                            /*transaction_count=*/1,
                            speculative_depth));
                }
            }
            const auto execution_begin = protocol_end;
            if (!executor_->executeMoEOverlayInferenceTransaction(
                    ticket, &result.error))
            {
                if (transaction_scope)
                    transaction_scope->discard();
                discard_active_prefill();
                if (result.error.empty())
                {
                    result.error =
                        "Remote retained ExpertOverlay graph transaction failed";
                }
                recordTicket(ticket, "follower", "execution_failed");
                return result;
            }
            const auto execution_end = std::chrono::steady_clock::now();
            transaction_scope.reset();
            if (prefill_schedule.valid())
            {
                ++active_prefill_transactions;
                if (active_prefill_transactions ==
                    prefill_schedule.transaction_count)
                {
                    /*
                     * executeMoEOverlayInferenceTransaction returns only after
                     * the retained follower graph and sparse return handoff are
                     * terminal, so the last graph closes the same complete
                     * logical schedule carried by the continuation event set.
                     */
                    active_prefill_scope.reset();
                    active_prefill_schedule.reset();
                    active_prefill_transactions = 0;
                }
            }
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
                discard_active_prefill();
                return result;
            }
            if (ticket.graph_role ==
                    MoEOverlayInferenceGraphRole::MainPrefill &&
                retired_prefill_progress_sink_)
            {
                const std::uint64_t completed_tokens =
                    static_cast<std::uint64_t>(ticket.request_count) *
                    static_cast<std::uint64_t>(
                        ticket.logical_rows_per_request);
                if (!retired_prefill_progress_sink_(
                        completed_tokens, &result.error))
                {
                    discard_active_prefill();
                    if (result.error.empty())
                    {
                        result.error =
                            "ExpertOverlay follower prefill progress sideband rejected a retired transaction";
                    }
                    return result;
                }
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
