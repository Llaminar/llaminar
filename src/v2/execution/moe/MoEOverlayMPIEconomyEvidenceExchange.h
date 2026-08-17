/**
 * @file MoEOverlayMPIEconomyEvidenceExchange.h
 * @brief Private non-blocking MPI lane for ExpertOverlay economy evidence.
 *
 * The lane owns a duplicated communicator distinct from inference collectives,
 * residency votes, remote weight bytes, and histogram publication. Attempt and
 * service packets use versioned fixed layouts with deterministic hashes; MPI
 * progress is queried with `MPI_Test` only by the maintenance worker.
 */

#pragma once

#include "MoEOverlayEconomyEvidenceExchange.h"

#include <cstdint>
#include <memory>
#include <string>

namespace llaminar2
{
    class IMPIContext;

    /** @brief Race-safe process-local proof for the private evidence lane. */
    struct MoEOverlayMPIEconomyEvidenceExchangeStats
    {
        std::uint64_t attempt_exchanges_started = 0;
        std::uint64_t attempt_exchanges_completed = 0;
        std::uint64_t service_readiness_exchanges_started = 0;
        std::uint64_t service_readiness_exchanges_completed = 0;
        std::uint64_t service_readiness_incomplete = 0;
        std::uint64_t service_exchanges_started = 0;
        std::uint64_t service_exchanges_completed = 0;
        std::uint64_t progress_polls = 0;
        std::uint64_t rejected_packets = 0;
        std::uint64_t concurrent_start_rejections = 0;
        std::uint64_t mpi_failures = 0;
    };

    /** @brief MPI_THREAD_MULTIPLE implementation of the economy evidence ABI. */
    class MoEOverlayMPIEconomyEvidenceExchange final
        : public IMoEOverlayEconomyEvidenceExchange
    {
    public:
        /** @brief Immutable communicator, ownership geometry, and label. */
        struct Config
        {
            std::shared_ptr<IMPIContext> mpi_context;
            MoEExpertOwnerMap owner_map;
            int num_layers = 0;
            /** Immutable phases reachable under the instance's MTP policy. */
            ExpertHistogramProductionSourceMask active_sources =
                kAllExpertHistogramProductionSources;
            std::string perf_device;
        };

        /**
         * @brief Duplicate a private communicator and preallocate all packets.
         * @throws std::invalid_argument For bad topology or communicator input.
         * @throws std::runtime_error For MPI lifecycle/thread/setup failures.
         */
        explicit MoEOverlayMPIEconomyEvidenceExchange(Config config);

        /** @brief Free the idle communicator; an active request is fatal. */
        ~MoEOverlayMPIEconomyEvidenceExchange() override;

        MoEOverlayMPIEconomyEvidenceExchange(
            const MoEOverlayMPIEconomyEvidenceExchange &) = delete;
        MoEOverlayMPIEconomyEvidenceExchange &operator=(
            const MoEOverlayMPIEconomyEvidenceExchange &) = delete;

        bool beginAttempt(
            const MoEOverlayCalibrationAttemptEvidence &local,
            std::string *error = nullptr) override;

        MoEOverlayResidencyWaveProgress pollAttempt(
            MoEOverlayCalibrationAttemptResult *result,
            std::string *error = nullptr) override;

        bool beginServiceReadiness(
            bool local_ready,
            std::string *error = nullptr) override;

        MoEOverlayResidencyWaveProgress pollServiceReadiness(
            bool *all_ranks_ready,
            std::string *error = nullptr) override;

        bool beginService(
            const std::vector<MoEOverlayParticipantLayerServiceTotals> &local,
            std::string *error = nullptr) override;

        MoEOverlayResidencyWaveProgress pollService(
            std::vector<MoEOverlayParticipantLayerServiceTotals> *result,
            std::string *error = nullptr) override;

        [[nodiscard]] bool idle() const noexcept override;
        [[nodiscard]] int worldRank() const noexcept override;
        [[nodiscard]] int worldSize() const noexcept override;

        /** @return Race-safe copy of control-plane proof counters. */
        [[nodiscard]] MoEOverlayMPIEconomyEvidenceExchangeStats stats()
            const noexcept;

    private:
        /** Opaque packet storage and MPI request ownership. */
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace llaminar2
