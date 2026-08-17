/**
 * @file MoEOverlayEconomyEvidenceExchange.h
 * @brief All-rank attempt and service evidence exchange for overlay economics.
 *
 * Distributed calibration produces partial projection timings and one local
 * inference-overlap verdict on every rank. Dynamic placement is certified only
 * after those rows are merged conservatively and every rank receives the same
 * pointer-free result. Service measurements use the same principle: each
 * participant's owning rank contributes its exact live prepared-engine totals,
 * and the exchange rejects missing, duplicated, or wrongly owned rows.
 */

#pragma once

#include "MoEOverlayInferenceInterferenceProbe.h"
#include "MoEOverlayMigrationMeasurementExchange.h"

#include <cstdint>
#include <string>
#include <vector>

namespace llaminar2
{
    /** @brief One rank's complete outcome for a synchronized calibration try. */
    struct MoEOverlayCalibrationAttemptEvidence
    {
        std::uint64_t calibration_sequence = 0;
        MoEOverlayMigrationMeasurementCoordinate coordinate;
        ExpertHistogramSource source = ExpertHistogramSource::SyntheticTest;
        MoEOverlayInferenceWorkloadIdentity workload;
        std::uint64_t baseline_nanoseconds = 0;
        std::uint64_t concurrent_nanoseconds = 0;
        bool exact_overlap = false;
        std::vector<MoEOverlayCompletedMigrationMeasurement>
            local_measurements;

        /** @return Whether identity, timings, and two-row swap evidence cohere. */
        [[nodiscard]] bool valid() const noexcept;
    };

    /** @brief Identical all-rank decision returned after one attempt exchange. */
    struct MoEOverlayCalibrationAttemptResult
    {
        bool accepted = false;
        /** Complete conservatively merged rows only when accepted. */
        std::vector<MoEOverlayCompletedMigrationMeasurement> measurements;
        /** Pair from the rank with the greatest non-negative interference. */
        std::uint64_t baseline_nanoseconds = 0;
        std::uint64_t concurrent_nanoseconds = 0;

        /** @return Whether rejection is empty or acceptance is complete. */
        [[nodiscard]] bool valid() const noexcept;
    };

    /**
     * @brief Pure deterministic global reducers used by MPI and protocol tests.
     */
    class MoEOverlayEconomyEvidenceMerger final
    {
    public:
        /**
         * @brief Merge one rank-ordered attempt corpus.
         * @return Rejected when any rank lacks exact overlap; otherwise complete
         *         migration rows and the largest observed local interference.
         * @throws std::invalid_argument For identity or row divergence.
         */
        [[nodiscard]] static MoEOverlayCalibrationAttemptResult mergeAttempt(
            const std::vector<MoEOverlayCalibrationAttemptEvidence> &ranks);

        /**
         * @brief Merge locally owned service rows into the complete topology.
         * @param rank_rows One row vector per world rank.
         * @param owner_map Global participant and world-rank authority.
         * @param num_layers Exact model layer count.
         * @param active_sources Immutable runtime phase availability.
         * @return Canonically participant/layer-ordered complete totals.
         * Active phases may be coherently empty in individual equivalent
         * layers; the certifier subsequently requires measured coverage after
         * exact-class pooling. Disabled phases must remain empty here.
         *
         * @throws std::invalid_argument For missing, duplicate, wrong-rank,
         *         malformed, overflowed, or disabled-phase rows.
         */
        [[nodiscard]] static std::vector<
            MoEOverlayParticipantLayerServiceTotals> mergeService(
            const std::vector<std::vector<
                MoEOverlayParticipantLayerServiceTotals>> &rank_rows,
            const MoEExpertOwnerMap &owner_map,
            int num_layers,
            ExpertHistogramProductionSourceMask active_sources =
                kAllExpertHistogramProductionSources);
    };

    /**
     * @brief Non-blocking all-rank lane for calibration and service evidence.
     *
     * Exactly one exchange may be active. Every rank invokes operations in the
     * same deterministic order; implementations must poll rather than wait.
     */
    class IMoEOverlayEconomyEvidenceExchange
    {
    public:
        virtual ~IMoEOverlayEconomyEvidenceExchange() = default;

        /** @brief Begin one all-rank calibration-attempt exchange. */
        virtual bool beginAttempt(
            const MoEOverlayCalibrationAttemptEvidence &local,
            std::string *error = nullptr) = 0;

        /** @brief Poll the active attempt once without blocking. */
        virtual MoEOverlayResidencyWaveProgress pollAttempt(
            MoEOverlayCalibrationAttemptResult *result,
            std::string *error = nullptr) = 0;

        /**
         * @brief Begin a tiny all-rank vote on local service readiness.
         *
         * Every rank must enter this collective, including ranks whose live
         * sparse traffic has not yet covered every required service class.
         * This vote is the ordering edge that prevents a ready rank from
         * entering the larger service all-gather alone.
         */
        virtual bool beginServiceReadiness(
            bool local_ready,
            std::string *error = nullptr) = 0;

        /**
         * @brief Poll the active readiness vote once without blocking.
         * @param all_ranks_ready Receives the logical AND of every rank's vote.
         */
        virtual MoEOverlayResidencyWaveProgress pollServiceReadiness(
            bool *all_ranks_ready,
            std::string *error = nullptr) = 0;

        /**
         * @brief Begin one all-rank local-service-row exchange.
         *
         * This operation follows a successful readiness vote. Individual
         * equivalent layers may contain coherent zero totals; the certifier
         * pools their live observations over the authenticated layer class.
         */
        virtual bool beginService(
            const std::vector<MoEOverlayParticipantLayerServiceTotals> &local,
            std::string *error = nullptr) = 0;

        /** @brief Poll the active service exchange once without blocking. */
        virtual MoEOverlayResidencyWaveProgress pollService(
            std::vector<MoEOverlayParticipantLayerServiceTotals> *result,
            std::string *error = nullptr) = 0;

        /** @return Whether no attempt or service exchange owns resources. */
        [[nodiscard]] virtual bool idle() const noexcept = 0;

        /** @return Bound rank in the complete overlay world. */
        [[nodiscard]] virtual int worldRank() const noexcept = 0;

        /** @return Complete overlay world size. */
        [[nodiscard]] virtual int worldSize() const noexcept = 0;
    };
} // namespace llaminar2
