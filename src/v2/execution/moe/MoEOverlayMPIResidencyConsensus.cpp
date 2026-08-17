/**
 * @file MoEOverlayMPIResidencyConsensus.cpp
 * @brief MPI_THREAD_MULTIPLE vote progress for ExpertOverlay maintenance.
 */

#include "MoEOverlayMPIResidencyConsensus.h"

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
        /** @brief Assign an optional diagnostic destination. */
        void setError(std::string *error, std::string message)
        {
            if (error)
                *error = std::move(message);
        }
    } // namespace

    MoEOverlayMPIResidencyConsensus::MoEOverlayMPIResidencyConsensus(
        Config config)
        : config_(std::move(config))
    {
        if (!config_.mpi_context)
        {
            throw std::invalid_argument(
                "Distributed ExpertOverlay consensus requires an MPI context");
        }
        if (config_.mpi_context->world_size() < 2 ||
            config_.mpi_context->rank() < 0 ||
            config_.mpi_context->rank() >=
                config_.mpi_context->world_size())
        {
            throw std::invalid_argument(
                "Distributed ExpertOverlay consensus requires valid multi-rank membership");
        }
        if (config_.mpi_context->communicator() == MPI_COMM_NULL)
        {
            throw std::invalid_argument(
                "Distributed ExpertOverlay consensus requires a live communicator");
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
                "Distributed ExpertOverlay maintenance requires live MPI_THREAD_MULTIPLE support");
        }

        if (sizeof(MoEOverlayDistributedResidencyVote) >
            static_cast<std::size_t>(INT_MAX))
        {
            throw std::overflow_error(
                "ExpertOverlay vote size exceeds the MPI count ABI");
        }

        /*
         * Complete every rank-local allocation before entering the collective
         * communicator duplication. If one rank cannot satisfy the model-time
         * BOM, no peer can be stranded inside a later collective constructor.
         */
        gathered_votes_.resize(static_cast<std::size_t>(
            config_.mpi_context->world_size()));

        const int duplicate_result = MPI_Comm_dup(
            config_.mpi_context->communicator(),
            &private_communicator_);
        if (duplicate_result != MPI_SUCCESS ||
            private_communicator_ == MPI_COMM_NULL)
        {
            throw std::runtime_error(
                mpiError("MPI_Comm_dup", duplicate_result));
        }

        recordCounter("mpi_consensus_lanes_materialized");
    }

    MoEOverlayMPIResidencyConsensus::~MoEOverlayMPIResidencyConsensus()
    {
        if (request_ != MPI_REQUEST_NULL)
        {
            LOG_ERROR(
                "[MoEOverlayMPIResidencyConsensus] Destroyed with an in-flight vote exchange");
            std::terminate();
        }
        if (private_communicator_ == MPI_COMM_NULL)
            return;

        int finalized = 0;
        if (MPI_Finalized(&finalized) != MPI_SUCCESS || finalized)
        {
            LOG_ERROR(
                "[MoEOverlayMPIResidencyConsensus] Communicator outlived MPI finalization");
            std::terminate();
        }
        if (MPI_Comm_free(&private_communicator_) != MPI_SUCCESS)
        {
            LOG_ERROR(
                "[MoEOverlayMPIResidencyConsensus] Failed to free private communicator");
            std::terminate();
        }
    }

    std::string MoEOverlayMPIResidencyConsensus::mpiError(
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

    void MoEOverlayMPIResidencyConsensus::recordCounter(
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
             {"world_size",
              std::to_string(config_.mpi_context->world_size())},
             {"background", "true"},
             {"blocking", "false"}});
    }

    bool MoEOverlayMPIResidencyConsensus::begin(
        const MoEOverlayDistributedResidencyVote &vote,
        std::string *error)
    {
        if (request_ != MPI_REQUEST_NULL)
        {
            ++stats_.concurrent_start_rejections;
            setError(
                error,
                "ExpertOverlay MPI consensus already owns an active exchange");
            return false;
        }
        if (!vote.valid(config_.mpi_context->world_size()) ||
            vote.world_rank != config_.mpi_context->rank())
        {
            ++stats_.malformed_local_votes;
            setError(
                error,
                "ExpertOverlay MPI consensus received an invalid local-rank vote");
            return false;
        }

        local_vote_ = vote;
        std::fill(
            gathered_votes_.begin(),
            gathered_votes_.end(),
            MoEOverlayDistributedResidencyVote{});
        exchange_started_at_ = std::chrono::steady_clock::now();

        const int vote_bytes =
            static_cast<int>(sizeof(MoEOverlayDistributedResidencyVote));
        const int mpi_result = MPI_Iallgather(
            &local_vote_,
            vote_bytes,
            MPI_BYTE,
            gathered_votes_.data(),
            vote_bytes,
            MPI_BYTE,
            private_communicator_,
            &request_);
        if (mpi_result != MPI_SUCCESS || request_ == MPI_REQUEST_NULL)
        {
            request_ = MPI_REQUEST_NULL;
            ++stats_.mpi_failures;
            setError(error, mpiError("MPI_Iallgather", mpi_result));
            return false;
        }

        ++stats_.exchanges_started;
        recordCounter("mpi_consensus_exchanges_started");
        if (error)
            error->clear();
        return true;
    }

    MoEOverlayResidencyWaveProgress
    MoEOverlayMPIResidencyConsensus::poll(
        std::vector<MoEOverlayDistributedResidencyVote> *votes,
        std::string *error)
    {
        ++stats_.progress_polls;
        if (request_ == MPI_REQUEST_NULL)
        {
            setError(
                error,
                "ExpertOverlay MPI consensus was polled without an active exchange");
            return MoEOverlayResidencyWaveProgress::Failed;
        }

        int complete = 0;
        const int mpi_result =
            MPI_Test(&request_, &complete, MPI_STATUS_IGNORE);
        if (mpi_result != MPI_SUCCESS)
        {
            ++stats_.mpi_failures;
            const auto message = mpiError("MPI_Test", mpi_result);
            setError(error, message);
            /*
             * MPI request ownership is indeterminate after a collective
             * progress error.  Returning to inference would violate message
             * ordering, so terminate the communicator rather than pretending
             * the unpublished wave is locally recoverable.
             */
            abortMoEOverlayMPI(
                private_communicator_,
                config_.mpi_context->rank(),
                "residency_consensus",
                message);
        }

        if (!complete)
        {
            const auto elapsed =
                std::chrono::steady_clock::now() - exchange_started_at_;
            if (elapsed >= std::chrono::milliseconds(
                               collective_timeout_policy::
                                   kDefaultCollectiveTimeoutMs))
            {
                ++stats_.mpi_failures;
                std::ostringstream diagnostic;
                diagnostic
                    << "MPI_Iallgather exceeded the canonical 30-second collective timeout"
                    << " phase=" << static_cast<int>(local_vote_.phase)
                    << " expected_epoch="
                    << local_vote_.identity.expected_epoch
                    << " candidate_epoch="
                    << local_vote_.identity.candidate_epoch
                    << " decision="
                    << static_cast<int>(local_vote_.decision);
                setError(error, diagnostic.str());
                abortMoEOverlayMPI(
                    private_communicator_,
                    config_.mpi_context->rank(),
                    "residency_consensus",
                    diagnostic.str());
            }
            return MoEOverlayResidencyWaveProgress::Pending;
        }

        if (votes)
            *votes = gathered_votes_;
        ++stats_.exchanges_completed;
        recordCounter("mpi_consensus_exchanges_completed");
        if (error)
            error->clear();
        return MoEOverlayResidencyWaveProgress::Ready;
    }

    int MoEOverlayMPIResidencyConsensus::worldRank() const noexcept
    {
        return config_.mpi_context->rank();
    }

    int MoEOverlayMPIResidencyConsensus::worldSize() const noexcept
    {
        return config_.mpi_context->world_size();
    }
} // namespace llaminar2
