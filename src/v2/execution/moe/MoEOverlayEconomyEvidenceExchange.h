/**
 * @file MoEOverlayEconomyEvidenceExchange.h
 * @brief All-rank migration-profile and service evidence exchange.
 *
 * Distributed profiling produces partial projection timings on every rank.
 * Dynamic placement is certified only after those rows are merged
 * conservatively and every rank receives the same pointer-free result. Service
 * measurements use the same principle: each participant's owning rank
 * contributes exact live prepared-engine totals, and the exchange rejects
 * missing, duplicated, or wrongly owned rows.
 */

#pragma once

#include "MoEOverlayInferenceInterferenceProbe.h"
#include "MoEOverlayMigrationMeasurementExchange.h"

#include <cstdint>
#include <string>
#include <vector>

namespace llaminar2
{
    /**
     * @brief One rank's partial timing rows for a real reversible profile wave.
     *
     * The distributed residency transport already provides the rank-symmetric
     * reservation and launch protocol.  This value therefore carries only the
     * physical evidence needed after the wave has completed; it deliberately
     * contains no inference workload or overlap handshake.
     */
    struct MoEOverlayMigrationProfileEvidence
    {
        std::uint64_t profile_sequence = 0;
        MoEOverlayMigrationMeasurementCoordinate coordinate;
        std::vector<MoEOverlayCompletedMigrationMeasurement>
            local_measurements;

        /** @return Whether identity and the reciprocal pair rows are complete. */
        [[nodiscard]] bool valid() const noexcept;
    };

    /** @brief Conservatively merged reciprocal rows returned on every rank. */
    struct MoEOverlayMigrationProfileResult
    {
        std::vector<MoEOverlayCompletedMigrationMeasurement> measurements;

        /** @return Whether both complete directed observations are present. */
        [[nodiscard]] bool valid() const noexcept;
    };

    /** @brief Distributed calibration edge that must precede physical staging. */
    enum class MoEOverlayCalibrationReadinessKind : std::uint8_t
    {
        BaselineDeviceComplete = 1,
    };

    /**
     * @brief Exact all-rank identity proving baseline inference has quiesced.
     *
     * Every rank publishes this only after its local production graph terminal.
     * The private economy lane rejects any phase, coordinate, sequence, or
     * workload mismatch before a rank may reserve migration resources.
     */
    struct MoEOverlayCalibrationReadiness
    {
        MoEOverlayCalibrationReadinessKind kind =
            MoEOverlayCalibrationReadinessKind::BaselineDeviceComplete;
        std::uint64_t calibration_sequence = 0;
        MoEOverlayMigrationMeasurementCoordinate coordinate;
        ExpertHistogramSource source = ExpertHistogramSource::SyntheticTest;
        MoEOverlayInferenceWorkloadIdentity workload;

        /** @return Whether this names one complete production baseline edge. */
        [[nodiscard]] bool valid() const noexcept;

        /** @brief Compare the complete rank-symmetric readiness identity. */
        bool operator==(const MoEOverlayCalibrationReadiness &) const = default;
    };

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
     * @brief One rank's typed vote in a service-evidence readiness round.
     *
     * The numeric order is intentional: an MPI minimum reduction propagates a
     * topology-wide stop ahead of incomplete evidence, and incomplete evidence
     * ahead of readiness.  This lets every rank execute the same collective
     * sequence without a rank-local boolean deciding whether that sequence
     * exists.
     */
    enum class MoEOverlayServiceEvidenceReadiness : std::int32_t
    {
        Stopping = 0,        ///< At least one rank is closing certification.
        AwaitingEvidence = 1, ///< At least one rank lacks a required sample.
        Ready = 2,           ///< This rank retained a complete local snapshot.
    };

    /**
     * @brief Pure deterministic global reducers used by MPI and protocol tests.
     */
    class MoEOverlayEconomyEvidenceMerger final
    {
    public:
        /**
         * @brief Merge one rank-ordered real transfer-profile observation.
         * @throws std::invalid_argument For identity divergence or incomplete
         *         projection timing evidence.
         */
        [[nodiscard]] static MoEOverlayMigrationProfileResult
        mergeMigrationProfile(
            const std::vector<MoEOverlayMigrationProfileEvidence> &ranks);

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
         * @param production_topology Exact retained-layer graph reachability.
         * @return Canonically participant/layer-ordered complete totals.
         * Reachable phases may be coherently empty in individual equivalent
         * layers; the certifier subsequently requires economy-priced coverage
         * after exact-class pooling. Unreachable layer/phase coordinates must
         * remain empty here, including main-only phases on MTP sidecars.
         *
         * @throws std::invalid_argument For missing, duplicate, wrong-rank,
         *         malformed, overflowed, or unreachable-phase rows.
         */
        [[nodiscard]] static std::vector<
            MoEOverlayParticipantLayerServiceTotals> mergeService(
            const std::vector<std::vector<
                MoEOverlayParticipantLayerServiceTotals>> &rank_rows,
            const MoEExpertOwnerMap &owner_map,
            const ExpertHistogramProductionTopology &production_topology);
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

        /** @brief Begin one all-rank physical migration-profile exchange. */
        virtual bool beginMigrationProfile(
            const MoEOverlayMigrationProfileEvidence &local,
            std::string *error = nullptr) = 0;

        /** @brief Poll the active migration-profile exchange without waiting. */
        virtual MoEOverlayResidencyWaveProgress pollMigrationProfile(
            MoEOverlayMigrationProfileResult *result,
            std::string *error = nullptr) = 0;

        /** @brief Begin one all-rank calibration-attempt exchange. */
        virtual bool beginAttempt(
            const MoEOverlayCalibrationAttemptEvidence &local,
            std::string *error = nullptr) = 0;

        /** @brief Poll the active attempt once without blocking. */
        virtual MoEOverlayResidencyWaveProgress pollAttempt(
            MoEOverlayCalibrationAttemptResult *result,
            std::string *error = nullptr) = 0;

        /**
         * @brief Begin an exact all-rank calibration readiness exchange.
         *
         * This uses the existing private economy evidence lane. It is entered
         * only after local device completion and before any migration transport
         * reservation, so one fast rank cannot interfere with a slower rank's
         * still-running inference graph.
         */
        virtual bool beginCalibrationReadiness(
            const MoEOverlayCalibrationReadiness &local,
            std::string *error = nullptr) = 0;

        /**
         * @brief Poll the active calibration readiness edge without waiting.
         *
         * @return `Ready` when every rank sampled the same workload identity,
         *         `Deferred` when valid rank-local samples differ and the whole
         *         attempt must be re-armed, `Pending` while MPI is live, or
         *         `Failed` for malformed/authentication/progress errors.
         */
        virtual MoEOverlayResidencyWaveProgress pollCalibrationReadiness(
            std::string *error = nullptr) = 0;

        /**
         * @brief Begin one typed all-rank service-readiness round.
         *
         * Every rank enters every round, including ranks whose live sparse
         * traffic has not yet covered every required service class. Stopping
         * is a first-class result, so teardown cannot strand a peer in a later
         * readiness round. The round is the ordering edge that prevents a ready
         * rank from entering the larger service all-gather alone.
         */
        virtual bool beginServiceReadiness(
            MoEOverlayServiceEvidenceReadiness local_readiness,
            std::string *error = nullptr) = 0;

        /**
         * @brief Poll the active readiness vote once without blocking.
         * @param global_readiness Receives the most conservative rank result.
         */
        virtual MoEOverlayResidencyWaveProgress pollServiceReadiness(
            MoEOverlayServiceEvidenceReadiness *global_readiness,
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
