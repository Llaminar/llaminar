/**
 * @file MoEOverlayMPIHistogramPublisher.cpp
 * @brief Private non-blocking MPI transport for frozen routing evidence.
 */

#include "MoEOverlayMPIHistogramPublisher.h"

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
        constexpr int kHistogramPublicationTag = 0;
        constexpr int kHistogramAcknowledgementTag = 1;

        /** @brief Assign one optional diagnostic. */
        void setError(std::string *error, std::string message)
        {
            if (error)
                *error = std::move(message);
        }
    } // namespace

    MoEOverlayMPIHistogramPublisher::MoEOverlayMPIHistogramPublisher(
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
                "ExpertOverlay histogram publisher requires valid multi-rank MPI membership");
        }
        if (config_.coordinator_world_rank < 0 ||
            config_.coordinator_world_rank >=
                config_.mpi_context->world_size())
        {
            throw std::invalid_argument(
                "ExpertOverlay histogram coordinator is outside the MPI world");
        }
        if (config_.num_layers <= 0 || config_.num_experts <= 0)
        {
            throw std::invalid_argument(
                "ExpertOverlay histogram publisher requires positive model geometry");
        }
        if (config_.mpi_context->communicator() == MPI_COMM_NULL)
        {
            throw std::invalid_argument(
                "ExpertOverlay histogram publisher requires a live communicator");
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
                "ExpertOverlay histogram publisher requires live MPI_THREAD_MULTIPLE support");
        }

        const std::size_t wire_bytes =
            moeOverlayDistributedHistogramWireBytes(
                config_.num_layers, config_.num_experts);
        if (wire_bytes > static_cast<std::size_t>(INT_MAX))
        {
            throw std::overflow_error(
                "ExpertOverlay histogram packet exceeds the MPI count ABI");
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
        recordCounter("histogram_publication_lanes_materialized");
    }

    MoEOverlayMPIHistogramPublisher::~MoEOverlayMPIHistogramPublisher()
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
                "[MoEOverlayMPIHistogramPublisher] Private communicator outlived MPI or failed to free");
            std::terminate();
        }
    }

    bool MoEOverlayMPIHistogramPublisher::isCoordinator() const noexcept
    {
        return config_.mpi_context->rank() ==
               config_.coordinator_world_rank;
    }

    std::string MoEOverlayMPIHistogramPublisher::mpiError(
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

    void MoEOverlayMPIHistogramPublisher::recordCounter(
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
    MoEOverlayMPIHistogramPublisher::fail(
        std::string message,
        std::string *error)
    {
        if (failure_.empty())
            failure_ = std::move(message);
        state_ = MoEOverlayMPIHistogramPublisherState::Failed;
        setError(error, failure_);
        return MoEOverlayResidencyWaveProgress::Failed;
    }

    bool MoEOverlayMPIHistogramPublisher::armReceive(std::string *error)
    {
        if (isCoordinator())
        {
            setError(
                error,
                "ExpertOverlay histogram coordinator cannot arm a peer receive");
            return false;
        }
        if (state_ == MoEOverlayMPIHistogramPublisherState::Stopped ||
            state_ == MoEOverlayMPIHistogramPublisherState::Failed ||
            receive_request_ != MPI_REQUEST_NULL ||
            acknowledgement_send_request_ != MPI_REQUEST_NULL ||
            received_window_pending_acknowledgement_)
        {
            setError(
                error,
                "ExpertOverlay histogram receive cannot be armed in the current lane state");
            return false;
        }

        const int mpi_result = MPI_Irecv(
            wire_buffer_.data(),
            static_cast<int>(wire_buffer_.size()),
            MPI_BYTE,
            config_.coordinator_world_rank,
            kHistogramPublicationTag,
            private_communicator_,
            &receive_request_);
        if (mpi_result != MPI_SUCCESS ||
            receive_request_ == MPI_REQUEST_NULL)
        {
            ++stats_.mpi_failures;
            receive_request_ = MPI_REQUEST_NULL;
            failure_ = mpiError("MPI_Irecv", mpi_result);
            state_ = MoEOverlayMPIHistogramPublisherState::Failed;
            setError(error, failure_);
            return false;
        }
        operation_started_at_ = std::chrono::steady_clock::now();
        state_ = MoEOverlayMPIHistogramPublisherState::Receiving;
        ++stats_.receives_armed;
        recordCounter("histogram_receives_armed");
        if (error)
            error->clear();
        return true;
    }

    bool MoEOverlayMPIHistogramPublisher::beginPublish(
        const DecodeExpertHistogramWindow &window,
        std::string *error)
    {
        if (!isCoordinator())
        {
            ++stats_.validation_failures;
            setError(
                error,
                "Only the declared ExpertOverlay coordinator may publish routing evidence");
            return false;
        }
        if (state_ != MoEOverlayMPIHistogramPublisherState::Idle)
        {
            ++stats_.validation_failures;
            setError(
                error,
                "ExpertOverlay histogram publisher already owns an active generation");
            return false;
        }
        if (window.num_layers != config_.num_layers ||
            window.num_experts != config_.num_experts || !window.valid())
        {
            ++stats_.validation_failures;
            setError(
                error,
                "ExpertOverlay coordinator histogram does not match model geometry");
            return false;
        }
        if (!encodeMoEOverlayDistributedHistogramWindow(
                window, wire_buffer_, error))
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
        publishing_generation_ = window.generation;
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
                kHistogramAcknowledgementTag,
                private_communicator_,
                &acknowledgement_receive_requests_[
                    static_cast<std::size_t>(rank)]);
            if (mpi_result != MPI_SUCCESS ||
                acknowledgement_receive_requests_[
                    static_cast<std::size_t>(rank)] == MPI_REQUEST_NULL)
            {
                ++stats_.mpi_failures;
                failure_ = mpiError(
                    "MPI_Irecv histogram acknowledgement", mpi_result);
                state_ = MoEOverlayMPIHistogramPublisherState::Failed;
                setError(error, failure_);
                return false;
            }
            mpi_result = MPI_Isend(
                wire_buffer_.data(),
                static_cast<int>(wire_buffer_.size()),
                MPI_BYTE,
                rank,
                kHistogramPublicationTag,
                private_communicator_,
                &send_requests_[static_cast<std::size_t>(rank)]);
            if (mpi_result != MPI_SUCCESS ||
                send_requests_[static_cast<std::size_t>(rank)] ==
                    MPI_REQUEST_NULL)
            {
                ++stats_.mpi_failures;
                failure_ = mpiError("MPI_Isend", mpi_result);
                state_ = MoEOverlayMPIHistogramPublisherState::Failed;
                setError(error, failure_);
                return false;
            }
        }

        operation_started_at_ = std::chrono::steady_clock::now();
        state_ = MoEOverlayMPIHistogramPublisherState::Publishing;
        ++stats_.publications_started;
        stats_.bytes_sent +=
            static_cast<std::uint64_t>(wire_buffer_.size()) *
            static_cast<std::uint64_t>(
                config_.mpi_context->world_size() - 1);
        recordCounter("histogram_publications_started");
        if (error)
            error->clear();
        return true;
    }

    MoEOverlayResidencyWaveProgress
    MoEOverlayMPIHistogramPublisher::poll(
        std::shared_ptr<const DecodeExpertHistogramWindow>
            *received_window,
        std::string *error)
    {
        ++stats_.progress_polls;
        if (received_window)
            received_window->reset();
        if (state_ == MoEOverlayMPIHistogramPublisherState::Failed)
            return fail(failure_, error);
        if (state_ != MoEOverlayMPIHistogramPublisherState::Receiving &&
            state_ != MoEOverlayMPIHistogramPublisherState::Acknowledging &&
            state_ != MoEOverlayMPIHistogramPublisherState::Publishing)
        {
            return fail(
                "ExpertOverlay histogram lane was polled without active traffic",
                error);
        }

        int complete = 0;
        int mpi_result = MPI_SUCCESS;
        if (state_ == MoEOverlayMPIHistogramPublisherState::Receiving)
        {
            mpi_result = MPI_Test(
                &receive_request_, &complete, MPI_STATUS_IGNORE);
        }
        else if (
            state_ ==
            MoEOverlayMPIHistogramPublisherState::Acknowledging)
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
                "histogram_publication",
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
            if (moeOverlayHistogramOwnsProgressDeadline(state_) &&
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
                    "histogram_publication",
                    diagnostic.str());
            }
            return MoEOverlayResidencyWaveProgress::Pending;
        }

        if (state_ == MoEOverlayMPIHistogramPublisherState::Publishing)
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
                        << "Histogram peer acknowledged the wrong generation"
                        << " peer_rank=" << rank
                        << " expected_generation=" << publishing_generation_
                        << " received_generation="
                        << acknowledgement_receive_generations_[
                               static_cast<std::size_t>(rank)];
                    fail(diagnostic.str(), error);
                    abortMoEOverlayMPI(
                        private_communicator_,
                        config_.mpi_context->rank(),
                        "histogram_publication",
                        diagnostic.str());
                }
                ++stats_.acknowledgements_received;
            }
            state_ = MoEOverlayMPIHistogramPublisherState::Idle;
            ++stats_.publications_completed;
            recordCounter("histogram_publications_completed");
            recordCounter(
                "histogram_acknowledgements_received",
                static_cast<double>(
                    config_.mpi_context->world_size() - 1));
            if (error)
                error->clear();
            return MoEOverlayResidencyWaveProgress::Ready;
        }

        if (state_ == MoEOverlayMPIHistogramPublisherState::Acknowledging)
        {
            auto window =
                std::move(received_window_pending_acknowledgement_);
            if (!window ||
                window->generation != acknowledgement_send_generation_)
            {
                ++stats_.validation_failures;
                return fail(
                    "Histogram acknowledgement completed without its exact immutable window",
                    error);
            }
            state_ = MoEOverlayMPIHistogramPublisherState::Idle;
            ++stats_.acknowledgements_completed;
            ++stats_.windows_received;
            recordCounter("histogram_acknowledgements_completed");
            recordCounter("histogram_windows_received");
            if (received_window)
                *received_window = std::move(window);
            if (error)
                error->clear();
            return MoEOverlayResidencyWaveProgress::Ready;
        }

        auto window = std::make_shared<DecodeExpertHistogramWindow>();
        window->expert_counts.reserve(
            static_cast<std::size_t>(config_.num_layers) *
            static_cast<std::size_t>(config_.num_experts));
        window->source_expert_counts.reserve(
            kExpertHistogramProductionSourceCount *
            static_cast<std::size_t>(config_.num_layers) *
            static_cast<std::size_t>(config_.num_experts));
        std::string decode_error;
        if (!decodeMoEOverlayDistributedHistogramWindow(
                wire_buffer_,
                config_.num_layers,
                config_.num_experts,
                window.get(),
                &decode_error))
        {
            ++stats_.validation_failures;
            return fail(std::move(decode_error), error);
        }

        acknowledgement_send_generation_ = window->generation;
        received_window_pending_acknowledgement_ = std::move(window);
        const int acknowledgement_result = MPI_Isend(
            &acknowledgement_send_generation_,
            1,
            MPI_UINT64_T,
            config_.coordinator_world_rank,
            kHistogramAcknowledgementTag,
            private_communicator_,
            &acknowledgement_send_request_);
        if (acknowledgement_result != MPI_SUCCESS ||
            acknowledgement_send_request_ == MPI_REQUEST_NULL)
        {
            ++stats_.mpi_failures;
            acknowledgement_send_request_ = MPI_REQUEST_NULL;
            return fail(
                mpiError(
                    "MPI_Isend histogram acknowledgement",
                    acknowledgement_result),
                error);
        }
        operation_started_at_ = std::chrono::steady_clock::now();
        state_ = MoEOverlayMPIHistogramPublisherState::Acknowledging;
        ++stats_.acknowledgements_started;
        stats_.bytes_received += wire_buffer_.size();
        recordCounter("histogram_acknowledgements_started");
        if (error)
            error->clear();
        return MoEOverlayResidencyWaveProgress::Pending;
    }

    void MoEOverlayMPIHistogramPublisher::stopAndDrain()
    {
        if (state_ == MoEOverlayMPIHistogramPublisherState::Stopped)
            return;

        if (receive_request_ != MPI_REQUEST_NULL)
        {
            const int cancel_result = MPI_Cancel(&receive_request_);
            const int wait_result =
                MPI_Wait(&receive_request_, MPI_STATUS_IGNORE);
            if (cancel_result != MPI_SUCCESS || wait_result != MPI_SUCCESS)
            {
                throw std::runtime_error(
                    "Failed to drain ExpertOverlay histogram receive during runner shutdown");
            }
        }

        if (acknowledgement_send_request_ != MPI_REQUEST_NULL)
        {
            const int wait_result = MPI_Wait(
                &acknowledgement_send_request_, MPI_STATUS_IGNORE);
            if (wait_result != MPI_SUCCESS)
            {
                throw std::runtime_error(
                    "Failed to drain ExpertOverlay histogram acknowledgement during runner shutdown");
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
                    "Failed to drain ExpertOverlay histogram sends during runner shutdown");
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
                    "Failed to drain ExpertOverlay histogram readiness receive during runner shutdown");
            }
        }
        received_window_pending_acknowledgement_.reset();
        state_ = MoEOverlayMPIHistogramPublisherState::Stopped;
    }
} // namespace llaminar2
