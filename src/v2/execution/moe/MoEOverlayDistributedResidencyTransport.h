/**
 * @file MoEOverlayDistributedResidencyTransport.h
 * @brief All-rank publication and lease-safe retirement around local movement.
 *
 * A physical tier factory can complete, defer, or fail independently on each
 * MPI rank. Returning that local result directly would let one rank abandon a
 * wave while peers enter commit or wait forever. This transport wraps the
 * process-local wave in fixed-identity stage, inactive-bank, and lease-drain
 * vote exchanges, so every rank reaches the same publication outcome and no
 * rank retires an epoch still referenced by a peer's sparse dispatch.
 */

#pragma once

#include "MoEOverlayDistributedResidencyProtocol.h"

#include <cstdint>
#include <memory>
#include <string>

namespace llaminar2
{
    struct MoEOverlayDistributedResidencyTransportSharedStats;

    /** @brief Process-local proof counters for distributed wave composition. */
    struct MoEOverlayDistributedResidencyTransportStats
    {
        std::uint64_t waves_started = 0;
        std::uint64_t local_stage_started = 0;
        std::uint64_t local_stage_deferred = 0;
        std::uint64_t local_stage_failed = 0;
        std::uint64_t reservation_consensus_started = 0;
        std::uint64_t reservation_consensus_ready = 0;
        std::uint64_t reservation_consensus_deferred = 0;
        std::uint64_t reservation_consensus_failed = 0;
        std::uint64_t stage_consensus_started = 0;
        std::uint64_t stage_consensus_ready = 0;
        std::uint64_t stage_consensus_deferred = 0;
        std::uint64_t stage_consensus_failed = 0;
        std::uint64_t commit_consensus_started = 0;
        std::uint64_t commit_consensus_ready = 0;
        std::uint64_t commit_consensus_failed = 0;
        std::uint64_t local_commit_begin_failed = 0;
        std::uint64_t retirement_consensus_started = 0;
        std::uint64_t retirement_consensus_ready = 0;
        std::uint64_t retirement_consensus_failed = 0;
        std::uint64_t waves_published = 0;
        std::uint64_t waves_aborted = 0;
        std::uint64_t abort_cleanups_completed = 0;
        std::uint64_t inference_thread_waits = 0;
        std::uint64_t blocking_synchronizations = 0;
    };

    /**
     * @brief Compose one local physical transport with an asynchronous vote lane.
     *
     * Even a locally deferred or failed start is represented by an owned wave:
     * its first poll contributes that terminal stage vote and waits for the
     * all-rank result. A local commit-enqueue failure similarly becomes a commit
     * vote instead of returning early and stranding peers in their collective.
     */
    class MoEOverlayDistributedResidencyTransport final
        : public IMoEOverlayResidencyTransport
    {
    public:
        /** @brief Model-lifetime local transport, vote lane, and evidence label. */
        struct Config
        {
            /** Process-local physical migration transport; never null. */
            IMoEOverlayResidencyTransport *local_transport = nullptr;
            /** Private all-rank lane retained by every active/cleanup wave. */
            std::shared_ptr<IMoEOverlayResidencyConsensusLane> consensus;
            /** Stable topology/model label attached to PerfStats evidence. */
            std::string perf_device;
        };

        /**
         * @brief Bind the process-local transport to one exact communicator.
         * @param config Complete model-lifetime distributed dependencies.
         * @throws std::invalid_argument For missing dependencies or bad geometry.
         */
        explicit MoEOverlayDistributedResidencyTransport(Config config);

        /**
         * @brief Begin local work and return an all-rank coordinated wave.
         * @param transaction Valid non-empty transaction shared by every rank.
         * @return Started wrapper, or a structural failure before collectives.
         */
        MoEOverlayResidencyStageStart beginStage(
            const MoEOverlayResidencyTransaction &transaction) override;

        /** @return Race-free cumulative distributed control-plane evidence. */
        [[nodiscard]] MoEOverlayDistributedResidencyTransportStats stats()
            const noexcept;

    private:
        Config config_;
        std::shared_ptr<MoEOverlayDistributedResidencyTransportSharedStats>
            stats_;
    };
} // namespace llaminar2
