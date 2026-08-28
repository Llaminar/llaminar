/**
 * @file MoEOverlayMPIResidencyProposalPublisher.cpp
 * @brief Private non-blocking MPI transport for canonical residency proposals.
 */

#include "MoEOverlayMPIResidencyProposalPublisher.h"

#include "MoEOverlayMPIFatal.h"

#include "collective/CollectiveTimeoutPolicy.h"
#include "interfaces/IMPIContext.h"
#include "utils/Logger.h"
#include "utils/PerfStatsCollector.h"

#include <algorithm>
#include <climits>
#include <exception>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace llaminar2
{
    namespace
    {
        constexpr int kProposalPublicationTag = 0;
        constexpr int kProposalAcknowledgementTag = 1;

        /** @brief Assign one optional diagnostic. */
        void setError(std::string *error, std::string message)
        {
            if (error)
                *error = std::move(message);
        }
    } // namespace

    MoEOverlayMPIResidencyProposalPublisher::
        MoEOverlayMPIResidencyProposalPublisher(
        Config config)
        : config_(std::move(config))
    {
        if (!config_.mpi_context ||
            config_.mpi_context->world_size() < 2 ||
            config_.mpi_context->rank() < 0 ||
            config_.mpi_context->rank() >=
                config_.mpi_context->world_size())
        {
            throw std::invalid_argument(
                "ExpertOverlay proposal publisher requires valid multi-rank MPI membership");
        }
        if (config_.coordinator_world_rank < 0 ||
            config_.coordinator_world_rank >=
                config_.mpi_context->world_size())
        {
            throw std::invalid_argument(
                "ExpertOverlay proposal coordinator is outside the MPI world");
        }
        if (config_.num_layers <= 0 || config_.num_experts <= 0)
        {
            throw std::invalid_argument(
                "ExpertOverlay proposal publisher requires positive model geometry");
        }
        if (config_.mpi_context->communicator() == MPI_COMM_NULL)
        {
            throw std::invalid_argument(
                "ExpertOverlay proposal publisher requires a live communicator");
        }
        if (config_.perf_device.empty())
            config_.perf_device = "expert_overlay_distributed";

        int initialized = 0;
        int finalized = 0;
        int thread_support = MPI_THREAD_SINGLE;
        if (MPI_Initialized(&initialized) != MPI_SUCCESS || !initialized ||
            MPI_Finalized(&finalized) != MPI_SUCCESS || finalized ||
            MPI_Query_thread(&thread_support) != MPI_SUCCESS ||
            thread_support < MPI_THREAD_MULTIPLE)
        {
            throw std::runtime_error(
                "ExpertOverlay proposal publisher requires live MPI_THREAD_MULTIPLE support");
        }

        const std::size_t wire_bytes =
            moeOverlayDistributedResidencyProposalWireBytes(
                config_.num_layers, config_.num_experts);
        if (wire_bytes > static_cast<std::size_t>(INT_MAX))
        {
            throw std::overflow_error(
                "ExpertOverlay proposal packet exceeds the MPI count ABI");
        }
        wire_buffer_.resize(wire_bytes);
        send_requests_.resize(
            static_cast<std::size_t>(config_.mpi_context->world_size()),
            MPI_REQUEST_NULL);
        acknowledgement_receive_requests_.resize(
            static_cast<std::size_t>(config_.mpi_context->world_size()),
            MPI_REQUEST_NULL);
        acknowledgement_receive_generations_.resize(
            static_cast<std::size_t>(config_.mpi_context->world_size()),
            0u);

        const int duplicate_result = MPI_Comm_dup(
            config_.mpi_context->communicator(),
            &private_communicator_);
        if (duplicate_result != MPI_SUCCESS ||
            private_communicator_ == MPI_COMM_NULL)
        {
            throw std::runtime_error(
                mpiError("MPI_Comm_dup", duplicate_result));
        }

        if (!isCoordinator())
        {
            std::string error;
            if (!armReceive(&error))
            {
                const int free_result =
                    MPI_Comm_free(&private_communicator_);
                if (free_result != MPI_SUCCESS)
                    std::terminate();
                throw std::runtime_error(error);
            }
        }
        recordCounter("proposal_publication_lanes_materialized");
    }

    MoEOverlayMPIResidencyProposalPublisher::
        ~MoEOverlayMPIResidencyProposalPublisher()
    {
        try
        {
            stopAndDrain();
        }
        catch (...)
        {
            std::terminate();
        }

        if (private_communicator_ == MPI_COMM_NULL)
            return;
        int finalized = 0;
        if (MPI_Finalized(&finalized) != MPI_SUCCESS || finalized ||
            MPI_Comm_free(&private_communicator_) != MPI_SUCCESS)
        {
            LOG_ERROR(
                "[MoEOverlayMPIResidencyProposalPublisher] Private communicator outlived MPI or failed to free");
            std::terminate();
        }
    }

    bool MoEOverlayMPIResidencyProposalPublisher::isCoordinator() const noexcept
    {
        return config_.mpi_context->rank() ==
               config_.coordinator_world_rank;
    }

    std::string MoEOverlayMPIResidencyProposalPublisher::mpiError(
        const char *operation,
        int mpi_error)
    {
        char buffer[MPI_MAX_ERROR_STRING]{};
        int length = 0;
        const int describe_result =
            MPI_Error_string(mpi_error, buffer, &length);
        std::ostringstream message;
        message << "ExpertOverlay " << operation << " failed with MPI code "
                << mpi_error;
        if (describe_result == MPI_SUCCESS && length > 0)
            message << ": " << std::string(buffer, buffer + length);
        return message.str();
    }

    void MoEOverlayMPIResidencyProposalPublisher::recordCounter(
        const char *name,
        double value) const
    {
        PerfStatsCollector::addCounter(
            "moe_overlay_residency",
            name,
            value,
            "maintenance",
            config_.perf_device,
            {{"world_rank",
              std::to_string(config_.mpi_context->rank())},
             {"coordinator_world_rank",
              std::to_string(config_.coordinator_world_rank)},
             {"background", "true"},
             {"blocking", "false"}});
    }

    MoEOverlayResidencyWaveProgress
    MoEOverlayMPIResidencyProposalPublisher::fail(
        std::string message,
        std::string *error)
    {
        if (failure_.empty())
            failure_ = std::move(message);
        state_ = MoEOverlayMPIResidencyProposalPublisherState::Failed;
        setError(error, failure_);
        return MoEOverlayResidencyWaveProgress::Failed;
    }

    bool MoEOverlayMPIResidencyProposalPublisher::armReceive(std::string *error)
    {
        if (isCoordinator())
        {
            setError(
                error,
                "ExpertOverlay proposal coordinator cannot arm a peer receive");
            return false;
        }
        if (state_ != MoEOverlayMPIResidencyProposalPublisherState::Idle ||
            receive_request_ != MPI_REQUEST_NULL ||
            acknowledgement_send_request_ != MPI_REQUEST_NULL)
        {
            setError(
                error,
                "ExpertOverlay proposal receive cannot be armed in the current lane state");
            return false;
        }

        const int mpi_result = MPI_Irecv(
            wire_buffer_.data(),
            static_cast<int>(wire_buffer_.size()),
            MPI_BYTE,
            config_.coordinator_world_rank,
            kProposalPublicationTag,
            private_communicator_,
            &receive_request_);
        if (mpi_result != MPI_SUCCESS ||
            receive_request_ == MPI_REQUEST_NULL)
        {
            ++stats_.mpi_failures;
            receive_request_ = MPI_REQUEST_NULL;
            failure_ = mpiError("MPI_Irecv", mpi_result);
            state_ = MoEOverlayMPIResidencyProposalPublisherState::Failed;
            setError(error, failure_);
            return false;
        }
        operation_started_at_ = std::chrono::steady_clock::now();
        state_ = MoEOverlayMPIResidencyProposalPublisherState::Receiving;
        ++stats_.receives_armed;
        recordCounter("proposal_receives_armed");
        if (error)
            error->clear();
        return true;
    }

    bool MoEOverlayMPIResidencyProposalPublisher::acceptReceivedProposal(
        std::uint64_t histogram_generation,
        std::string *error)
    {
        if (isCoordinator() ||
            state_ !=
                MoEOverlayMPIResidencyProposalPublisherState::AwaitingValidation ||
            awaiting_validation_generation_ == 0u ||
            histogram_generation != awaiting_validation_generation_ ||
            acknowledgement_send_request_ != MPI_REQUEST_NULL)
        {
            ++stats_.validation_failures;
            setError(
                error,
                "ExpertOverlay proposal acceptance does not match the decoded peer generation");
            return false;
        }

        acknowledgement_send_generation_ = histogram_generation;
        const int acknowledgement_result = MPI_Isend(
            &acknowledgement_send_generation_,
            1,
            MPI_UINT64_T,
            config_.coordinator_world_rank,
            kProposalAcknowledgementTag,
            private_communicator_,
            &acknowledgement_send_request_);
        if (acknowledgement_result != MPI_SUCCESS ||
            acknowledgement_send_request_ == MPI_REQUEST_NULL)
        {
            ++stats_.mpi_failures;
            acknowledgement_send_request_ = MPI_REQUEST_NULL;
            failure_ = mpiError(
                "MPI_Isend proposal semantic acknowledgement",
                acknowledgement_result);
            state_ = MoEOverlayMPIResidencyProposalPublisherState::Failed;
            setError(error, failure_);
            return false;
        }

        awaiting_validation_generation_ = 0u;
        operation_started_at_ = std::chrono::steady_clock::now();
        state_ = MoEOverlayMPIResidencyProposalPublisherState::Acknowledging;
        ++stats_.acknowledgements_started;
        recordCounter("proposal_acknowledgements_started");
        if (error)
            error->clear();
        return true;
    }

    void MoEOverlayMPIResidencyProposalPublisher::rejectReceivedProposal(
        std::uint64_t histogram_generation,
        std::string diagnostic)
    {
        std::ostringstream fatal;
        fatal << "Peer rejected authoritative proposal after semantic adoption"
              << " generation=" << histogram_generation
              << " expected_generation=" << awaiting_validation_generation_
              << " state=" << static_cast<int>(state_)
              << " reason="
              << (diagnostic.empty() ? "unspecified" : diagnostic);
        ++stats_.validation_failures;
        abortMoEOverlayMPI(
            private_communicator_,
            config_.mpi_context->rank(),
            "proposal_semantic_adoption",
            fatal.str());
    }

    bool MoEOverlayMPIResidencyProposalPublisher::beginPublish(
        const MoEOverlayDistributedResidencyProposal &proposal,
        std::string *error)
    {
        if (!isCoordinator())
        {
            ++stats_.validation_failures;
            setError(
                error,
                "Only the declared ExpertOverlay coordinator may publish residency proposals");
            return false;
        }
        if (state_ != MoEOverlayMPIResidencyProposalPublisherState::Idle)
        {
            ++stats_.validation_failures;
            setError(
                error,
                "ExpertOverlay proposal publisher already owns an active generation");
            return false;
        }
        if (!proposal.valid() ||
            proposal.plan.num_layers != config_.num_layers ||
            proposal.plan.num_experts != config_.num_experts)
        {
            ++stats_.validation_failures;
            setError(
                error,
                "ExpertOverlay coordinator proposal does not match model geometry");
            return false;
        }
        if (!encodeMoEOverlayDistributedResidencyProposal(
                proposal, wire_buffer_, error))
        {
            ++stats_.validation_failures;
            return false;
        }

        for (auto &request : send_requests_)
            request = MPI_REQUEST_NULL;
        for (auto &request : acknowledgement_receive_requests_)
            request = MPI_REQUEST_NULL;
        std::fill(
            acknowledgement_receive_generations_.begin(),
            acknowledgement_receive_generations_.end(),
            0u);
        publishing_generation_ =
            proposal.plan.histogram_window->generation;
        for (int rank = 0; rank < config_.mpi_context->world_size(); ++rank)
        {
            if (rank == config_.coordinator_world_rank)
                continue;
            /*
             * Arm readiness before sending the large packet. MPI completion
             * of the packet send only proves that a preposted receive matched;
             * this generation echo proves the peer maintenance thread decoded
             * the immutable window and is ready to derive the same proposal.
             */
            int mpi_result = MPI_Irecv(
                &acknowledgement_receive_generations_[
                    static_cast<std::size_t>(rank)],
                1,
                MPI_UINT64_T,
                rank,
                kProposalAcknowledgementTag,
                private_communicator_,
                &acknowledgement_receive_requests_[
                    static_cast<std::size_t>(rank)]);
            if (mpi_result != MPI_SUCCESS ||
                acknowledgement_receive_requests_[
                    static_cast<std::size_t>(rank)] == MPI_REQUEST_NULL)
            {
                ++stats_.mpi_failures;
                failure_ = mpiError(
                    "MPI_Irecv proposal acknowledgement", mpi_result);
                state_ = MoEOverlayMPIResidencyProposalPublisherState::Failed;
                setError(error, failure_);
                return false;
            }
            mpi_result = MPI_Isend(
                wire_buffer_.data(),
                static_cast<int>(wire_buffer_.size()),
                MPI_BYTE,
                rank,
                kProposalPublicationTag,
                private_communicator_,
                &send_requests_[static_cast<std::size_t>(rank)]);
            if (mpi_result != MPI_SUCCESS ||
                send_requests_[static_cast<std::size_t>(rank)] ==
                    MPI_REQUEST_NULL)
            {
                ++stats_.mpi_failures;
                failure_ = mpiError("MPI_Isend", mpi_result);
                state_ = MoEOverlayMPIResidencyProposalPublisherState::Failed;
                setError(error, failure_);
                return false;
            }
        }

        operation_started_at_ = std::chrono::steady_clock::now();
        state_ = MoEOverlayMPIResidencyProposalPublisherState::Publishing;
        ++stats_.publications_started;
        stats_.bytes_sent +=
            static_cast<std::uint64_t>(wire_buffer_.size()) *
            static_cast<std::uint64_t>(
                config_.mpi_context->world_size() - 1);
        recordCounter("proposal_publications_started");
        if (error)
            error->clear();
        return true;
    }

    MoEOverlayResidencyWaveProgress
    MoEOverlayMPIResidencyProposalPublisher::poll(
        std::shared_ptr<const MoEOverlayDistributedResidencyProposal>
            *received_proposal,
        std::string *error)
    {
        ++stats_.progress_polls;
        if (received_proposal)
            received_proposal->reset();
        if (state_ == MoEOverlayMPIResidencyProposalPublisherState::Failed)
            return fail(failure_, error);
        if (state_ ==
            MoEOverlayMPIResidencyProposalPublisherState::AwaitingValidation)
        {
            return fail(
                "ExpertOverlay decoded proposal was polled before semantic acceptance or rejection",
                error);
        }
        if (state_ != MoEOverlayMPIResidencyProposalPublisherState::Receiving &&
            state_ != MoEOverlayMPIResidencyProposalPublisherState::Acknowledging &&
            state_ != MoEOverlayMPIResidencyProposalPublisherState::Publishing)
        {
            return fail(
                "ExpertOverlay proposal lane was polled without active traffic",
                error);
        }

        int complete = 0;
        int mpi_result = MPI_SUCCESS;
        if (state_ == MoEOverlayMPIResidencyProposalPublisherState::Receiving)
        {
            mpi_result = MPI_Test(
                &receive_request_, &complete, MPI_STATUS_IGNORE);
        }
        else if (
            state_ ==
            MoEOverlayMPIResidencyProposalPublisherState::Acknowledging)
        {
            mpi_result = MPI_Test(
                &acknowledgement_send_request_,
                &complete,
                MPI_STATUS_IGNORE);
        }
        else
        {
            int sends_complete = 0;
            int acknowledgements_complete = 0;
            mpi_result = MPI_Testall(
                static_cast<int>(send_requests_.size()),
                send_requests_.data(),
                &sends_complete,
                MPI_STATUSES_IGNORE);
            if (mpi_result == MPI_SUCCESS)
            {
                mpi_result = MPI_Testall(
                    static_cast<int>(
                        acknowledgement_receive_requests_.size()),
                    acknowledgement_receive_requests_.data(),
                    &acknowledgements_complete,
                    MPI_STATUSES_IGNORE);
            }
            complete = sends_complete && acknowledgements_complete;
        }
        if (mpi_result != MPI_SUCCESS)
        {
            ++stats_.mpi_failures;
            const auto message = mpiError("MPI_Test", mpi_result);
            fail(message, error);
            abortMoEOverlayMPI(
                private_communicator_,
                config_.mpi_context->rank(),
                "proposal_publication",
                message);
        }

        if (!complete)
        {
            const auto elapsed =
                std::chrono::steady_clock::now() - operation_started_at_;
            /*
             * A preposted peer Irecv is a passive subscription, not evidence
             * that the coordinator has started an epoch. It can legitimately
             * span model calibration or an idle server. Initiated coordinator
             * publication and peer acknowledgement both retain the canonical
             * deadline because their counterpart has promised progress.
             */
            if (moeOverlayProposalOwnsProgressDeadline(state_) &&
                elapsed >= std::chrono::milliseconds(
                               collective_timeout_policy::
                                   kDefaultCollectiveTimeoutMs))
            {
                ++stats_.mpi_failures;
                std::ostringstream diagnostic;
                diagnostic
                    << "MPI progress exceeded the canonical 30-second timeout"
                    << " state=" << static_cast<int>(state_)
                    << " coordinator_world_rank="
                    << config_.coordinator_world_rank
                    << " packet_bytes=" << wire_buffer_.size();
                fail(diagnostic.str(), error);
                abortMoEOverlayMPI(
                    private_communicator_,
                    config_.mpi_context->rank(),
                    "proposal_publication",
                    diagnostic.str());
            }
            return MoEOverlayResidencyWaveProgress::Pending;
        }

        if (state_ == MoEOverlayMPIResidencyProposalPublisherState::Publishing)
        {
            for (int rank = 0;
                 rank < config_.mpi_context->world_size();
                 ++rank)
            {
                if (rank == config_.coordinator_world_rank)
                    continue;
                if (acknowledgement_receive_generations_[
                        static_cast<std::size_t>(rank)] !=
                    publishing_generation_)
                {
                    ++stats_.validation_failures;
                    std::ostringstream diagnostic;
                    diagnostic
                        << "Proposal peer acknowledged the wrong generation"
                        << " peer_rank=" << rank
                        << " expected_generation=" << publishing_generation_
                        << " received_generation="
                        << acknowledgement_receive_generations_[
                               static_cast<std::size_t>(rank)];
                    fail(diagnostic.str(), error);
                    abortMoEOverlayMPI(
                        private_communicator_,
                        config_.mpi_context->rank(),
                        "proposal_publication",
                        diagnostic.str());
                }
                ++stats_.acknowledgements_received;
            }
            state_ = MoEOverlayMPIResidencyProposalPublisherState::Idle;
            ++stats_.publications_completed;
            recordCounter("proposal_publications_completed");
            recordCounter(
                "proposal_acknowledgements_received",
                static_cast<double>(
                    config_.mpi_context->world_size() - 1));
            if (error)
                error->clear();
            return MoEOverlayResidencyWaveProgress::Ready;
        }

        if (state_ == MoEOverlayMPIResidencyProposalPublisherState::Acknowledging)
        {
            if (acknowledgement_send_generation_ == 0u)
            {
                ++stats_.validation_failures;
                return fail(
                    "Proposal acknowledgement completed without its accepted generation",
                    error);
            }
            state_ = MoEOverlayMPIResidencyProposalPublisherState::Idle;
            ++stats_.acknowledgements_completed;
            recordCounter("proposal_acknowledgements_completed");
            acknowledgement_send_generation_ = 0u;
            if (error)
                error->clear();
            return MoEOverlayResidencyWaveProgress::Ready;
        }

        auto proposal =
            std::make_shared<MoEOverlayDistributedResidencyProposal>();
        std::string decode_error;
        if (!decodeMoEOverlayDistributedResidencyProposal(
                wire_buffer_,
                config_.num_layers,
                config_.num_experts,
                proposal.get(),
                &decode_error))
        {
            ++stats_.validation_failures;
            fail(decode_error, error);
            abortMoEOverlayMPI(
                private_communicator_,
                config_.mpi_context->rank(),
                "proposal_authentication",
                decode_error);
        }

        awaiting_validation_generation_ =
            proposal->plan.histogram_window->generation;
        state_ =
            MoEOverlayMPIResidencyProposalPublisherState::AwaitingValidation;
        ++stats_.windows_received;
        stats_.bytes_received += wire_buffer_.size();
        recordCounter("proposals_received");
        if (received_proposal)
            *received_proposal = std::move(proposal);
        if (error)
            error->clear();
        return MoEOverlayResidencyWaveProgress::Ready;
    }

    void MoEOverlayMPIResidencyProposalPublisher::stopAndDrain()
    {
        if (state_ == MoEOverlayMPIResidencyProposalPublisherState::Stopped)
            return;

        if (receive_request_ != MPI_REQUEST_NULL)
        {
            const int cancel_result = MPI_Cancel(&receive_request_);
            const int wait_result =
                MPI_Wait(&receive_request_, MPI_STATUS_IGNORE);
            if (cancel_result != MPI_SUCCESS || wait_result != MPI_SUCCESS)
            {
                throw std::runtime_error(
                    "Failed to drain ExpertOverlay proposal receive during runner shutdown");
            }
        }

        if (acknowledgement_send_request_ != MPI_REQUEST_NULL)
        {
            const int wait_result = MPI_Wait(
                &acknowledgement_send_request_, MPI_STATUS_IGNORE);
            if (wait_result != MPI_SUCCESS)
            {
                throw std::runtime_error(
                    "Failed to drain ExpertOverlay proposal acknowledgement during runner shutdown");
            }
        }

        bool has_send = false;
        for (const auto &request : send_requests_)
            has_send = has_send || request != MPI_REQUEST_NULL;
        if (has_send)
        {
            const int wait_result = MPI_Waitall(
                static_cast<int>(send_requests_.size()),
                send_requests_.data(),
                MPI_STATUSES_IGNORE);
            if (wait_result != MPI_SUCCESS)
            {
                throw std::runtime_error(
                    "Failed to drain ExpertOverlay proposal sends during runner shutdown");
            }
        }
        for (auto &request : acknowledgement_receive_requests_)
        {
            if (request == MPI_REQUEST_NULL)
                continue;
            const int cancel_result = MPI_Cancel(&request);
            const int wait_result = MPI_Wait(&request, MPI_STATUS_IGNORE);
            if (cancel_result != MPI_SUCCESS || wait_result != MPI_SUCCESS)
            {
                throw std::runtime_error(
                    "Failed to drain ExpertOverlay proposal readiness receive during runner shutdown");
            }
        }
        awaiting_validation_generation_ = 0u;
        acknowledgement_send_generation_ = 0u;
        state_ = MoEOverlayMPIResidencyProposalPublisherState::Stopped;
    }
} // namespace llaminar2
