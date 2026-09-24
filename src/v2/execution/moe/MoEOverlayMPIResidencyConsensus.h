/**
 * @file MoEOverlayMPIResidencyConsensus.h
 * @brief Non-blocking MPI vote lane for distributed ExpertOverlay residency.
 *
 * Every rank contributes one fixed-layout vote after its local physical stage
 * or inactive-bank commit completes.  This lane exchanges those votes over a
 * model-lifetime private communicator and exposes completion through MPI_Test;
 * it never waits, sleeps, or executes on an inference thread.
 */

#pragma once

#include "MoEOverlayDistributedResidencyProtocol.h"

#include <mpi.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace llaminar2
{
    class IMPIContext;

    /** @brief Process-local evidence for one private MPI consensus lane. */
    struct MoEOverlayMPIResidencyConsensusStats
    {
        std::uint64_t exchanges_started = 0;
        std::uint64_t exchanges_completed = 0;
        std::uint64_t progress_polls = 0;
        std::uint64_t mpi_failures = 0;
        std::uint64_t malformed_local_votes = 0;
        std::uint64_t concurrent_start_rejections = 0;
    };

    /**
     * @brief Model-lifetime asynchronous all-rank vote exchange.
     *
     * Construction duplicates the configured communicator so maintenance
     * collectives can never alias inference sparse collectives, pipeline
     * traffic, or another model instance.  One exchange may be active at a
     * time.  The caller must poll it to completion before destruction; an
     * in-flight MPI collective cannot be safely cancelled or abandoned.
     */
    class MoEOverlayMPIResidencyConsensus final
        : public IMoEOverlayResidencyConsensusLane
    {
    public:
        /** @brief Immutable communicator and PerfStats identity. */
        struct Config
        {
            /** MPI context whose complete world participates in every vote. */
            std::shared_ptr<IMPIContext> mpi_context;
            /** Stable topology label attached to low-frequency evidence. */
            std::string perf_device;
        };

        /**
         * @brief Duplicate the communicator and preallocate the gather buffer.
         * @param config Multi-rank context initialized with MPI_THREAD_MULTIPLE.
         * @throws std::invalid_argument For missing or single-rank membership.
         * @throws std::runtime_error For unavailable thread support or MPI setup.
         */
        explicit MoEOverlayMPIResidencyConsensus(Config config);

        /**
         * @brief Free the idle private communicator.
         *
         * Destruction terminates when an exchange remains active or MPI was
         * already finalized: either condition means the runner violated the
         * lane's explicit drain-before-MPI-finalize ownership contract.
         */
        ~MoEOverlayMPIResidencyConsensus();

        MoEOverlayMPIResidencyConsensus(
            const MoEOverlayMPIResidencyConsensus &) = delete;
        MoEOverlayMPIResidencyConsensus &operator=(
            const MoEOverlayMPIResidencyConsensus &) = delete;

        /**
         * @brief Submit one local vote to a non-blocking all-gather.
         * @param vote Valid vote whose rank matches this MPI context.
         * @param error Receives an exact validation or MPI diagnostic.
         * @return True only when MPI owns the complete exchange request.
         */
        bool begin(
            const MoEOverlayDistributedResidencyVote &vote,
            std::string *error = nullptr) override;

        /**
         * @brief Progress the active exchange once with MPI_Test.
         * @param votes Receives one rank-ordered vote when Ready is returned.
         * @param error Receives an exact MPI diagnostic on Failed.
         * @return Pending, Ready, or Failed without blocking.
         *
         * Exceeding the canonical collective timeout aborts the communicator,
         * because collective ordering and request ownership are then unknown.
         */
        MoEOverlayResidencyWaveProgress poll(
            std::vector<MoEOverlayDistributedResidencyVote> *votes,
            std::string *error = nullptr) override;

        /** @return Whether no MPI request is currently in flight. */
        [[nodiscard]] bool idle() const noexcept override
        {
            return request_ == MPI_REQUEST_NULL;
        }

        /** @return World rank bound to the private communicator. */
        [[nodiscard]] int worldRank() const noexcept override;

        /** @return Number of ranks bound to the private communicator. */
        [[nodiscard]] int worldSize() const noexcept override;

        /** @return Race-free counters; the maintenance thread is sole writer. */
        [[nodiscard]] MoEOverlayMPIResidencyConsensusStats stats()
            const noexcept
        {
            return stats_;
        }

    private:
        /** @brief Convert an MPI status code into a stable diagnostic string. */
        [[nodiscard]] static std::string mpiError(
            const char *operation,
            int mpi_error);

        /** @brief Export one rare lifecycle event to PerfStats. */
        void recordCounter(const char *name, double value = 1.0) const;

        Config config_;
        MPI_Comm private_communicator_ = MPI_COMM_NULL;
        MPI_Request request_ = MPI_REQUEST_NULL;
        MoEOverlayDistributedResidencyVote local_vote_;
        std::vector<MoEOverlayDistributedResidencyVote> gathered_votes_;
        std::chrono::steady_clock::time_point exchange_started_at_{};
        MoEOverlayMPIResidencyConsensusStats stats_;
    };
} // namespace llaminar2
