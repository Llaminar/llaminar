/**
 * @file MoEOverlayEconomyCertificationController.h
 * @brief Pollable installation of measured ExpertOverlay economics.
 *
 * Dynamic residency must not consume a histogram until both prepared-expert
 * service cost and real transfer cost are measured. This controller joins the
 * bounded startup transfer profile with service telemetry accumulated by
 * ordinary requests, expands only exact manifest-equivalent layer
 * representatives, composes the complete immutable profiles, and installs
 * them into the residency authority. Every operation is maintenance-owned and
 * non-blocking with respect to inference.
 */

#pragma once

#include "MoEOverlayEconomyCalibrationController.h"
#include "MoEOverlayEconomyProfileComposer.h"
#include "../InferenceMeasurementReadiness.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace llaminar2
{
    /** Destination that receives one complete immutable economy certificate. */
    enum class MoEOverlayEconomyCertificationTarget : std::uint8_t
    {
        /** Install into the legacy host-owned residency authority. */
        ResidencyAuthority,
        /** Retain profiles for publication to the sole device authority. */
        DetachedDeviceAuthority,
    };

    /** @brief Observable lifecycle of rank-local economy certification. */
    enum class MoEOverlayEconomyCertificationState : std::uint8_t
    {
        CalibratingMovement,
        AwaitingServiceEvidence,
        ExchangingServiceReadiness,
        ExchangingServiceEvidence,
        /** Profiles exist, but calibration-era routing demand is still draining. */
        RebasingRoutingEvidence,
        Complete,
        Failed,
        Stopped,
    };

    /** @brief Race-safe proof counters for one certification lifecycle. */
    struct MoEOverlayEconomyCertificationStats
    {
        std::uint64_t polls = 0;
        std::uint64_t movement_calibration_polls = 0;
        std::uint64_t service_snapshot_attempts = 0;
        std::uint64_t service_snapshot_contentions = 0;
        std::uint64_t incomplete_service_snapshots = 0;
        std::uint64_t profiles_composed = 0;
        std::uint64_t routing_evidence_rebase_polls = 0;
        std::uint64_t routing_evidence_rebases = 0;
        std::uint64_t certifications_installed = 0;
        std::uint64_t detached_profiles_completed = 0;
        std::uint64_t fatal_failures = 0;
    };

    /**
     * @brief Joins live-path measurements into one authority certificate.
     *
     * A process-local topology must expose every participant through its local
     * registry. A distributed topology instead exchanges partial migration and
     * locally owned service rows through one private non-blocking lane. Both
     * forms install only a complete canonical matrix; silently treating a
     * rank-local row as global evidence is forbidden.
     */
    class MoEOverlayEconomyCertificationController final
    {
    public:
        /** @brief Immutable owners and arithmetic policy for certification. */
        struct Config
        {
            std::shared_ptr<MoEOverlayEconomyCalibrationController>
                calibration;
            std::shared_ptr<MoEOverlayParticipantResidencyRegistry> registry;
            std::shared_ptr<const MoEOverlayEconomyCalibrationLayerCatalog>
                layer_catalog;
            std::shared_ptr<MoEOverlayResidencyAuthority> authority;
            /** Typed destination; detached mode never mutates host policy. */
            MoEOverlayEconomyCertificationTarget target =
                MoEOverlayEconomyCertificationTarget::ResidencyAuthority;
            MoERoutedExpertModelMetadata model_metadata;
            MoEOverlayMigrationEconomyPolicy economy_policy;
            /** Immutable phases reachable under this instance's MTP policy. */
            ExpertHistogramProductionSourceMask active_sources =
                kAllExpertHistogramProductionSources;
            /** Optional all-rank service and attempt evidence lane. */
            std::shared_ptr<IMoEOverlayEconomyEvidenceExchange>
                evidence_exchange;
            /** Minimum delay between incomplete distributed readiness rounds. */
            std::chrono::milliseconds service_readiness_retry_interval{100};
            std::string perf_device;
        };

        /**
         * @brief Validate local/distributed topology and retain async owners.
         * @throws std::invalid_argument For incomplete membership, an already
         *         certified authority, or geometrically inconsistent inputs.
         */
        explicit MoEOverlayEconomyCertificationController(Config config);

        MoEOverlayEconomyCertificationController(
            const MoEOverlayEconomyCertificationController &) = delete;
        MoEOverlayEconomyCertificationController &operator=(
            const MoEOverlayEconomyCertificationController &) = delete;

        /** @brief Advance at most one bounded calibration/certification edge. */
        void poll() noexcept;

        /** @brief Stop admission and asynchronously drain calibration ownership. */
        void requestStop() noexcept;

        /** @return Current release-published lifecycle state. */
        [[nodiscard]] MoEOverlayEconomyCertificationState state()
            const noexcept;

        /** @return Whether no fatal measurement or installation error occurred. */
        [[nodiscard]] bool healthy() const noexcept;

        /** @return Stable first fatal diagnostic, or an empty string. */
        [[nodiscard]] std::string failureMessage() const;

        /** @return Race-safe proof counters for the complete lifecycle. */
        [[nodiscard]] MoEOverlayEconomyCertificationStats stats()
            const noexcept;

        /**
         * @brief Snapshot whether the complete measured economy is installed.
         *
         * This accessor is passive and allocation-only diagnostic work.  It
         * never polls calibration, snapshots service counters, or publishes
         * policy state; the maintenance owner remains the sole progressor.
         */
        [[nodiscard]] InferenceMeasurementReadiness
        measurementReadiness() const;

        /**
         * @brief Import one immutable device-local service snapshot.
         *
         * All-GPU production graphs accumulate timing without touching host
         * endpoint cells. The maintenance worker calls this only after the
         * participant's mapped snapshot graph has completed. Certification
         * remains the sole evidence owner and rejects imports after distributed
         * readiness exchange has begun.
         *
         * @param participant_id Exact process-local participant identity.
         * @param rows Complete layer-ordered cumulative timing totals.
         * @param error Optional rejection diagnostic.
         * @return True when the registry accepted the immutable snapshot.
         */
        [[nodiscard]] bool importDeviceServiceMeasurements(
            int participant_id,
            const std::vector<MoEOverlayParticipantLayerServiceTotals> &rows,
            std::string *error = nullptr);

        /**
         * @brief Acquire the complete immutable profiles for a device publisher.
         *
         * The returned shared profile objects remain valid for the model
         * lifetime. Before @ref MoEOverlayEconomyCertificationState::Complete,
         * or when configured for host-authority installation, this method
         * returns no value. It never exposes partially composed evidence.
         */
        [[nodiscard]] std::optional<MoEOverlayCertifiedEconomyProfiles>
        detachedProfiles() const;

    private:
        /** @brief First missing measured coordinate in a coherent snapshot. */
        struct ServiceEvidenceCoverage
        {
            bool complete = false;
            int participant_id = -1;
            int representative_layer = -1;
            std::size_t source_index =
                kExpertHistogramProductionSourceCount;
        };

        /**
         * @brief Test measured coverage over exact layer-equivalence classes.
         * @param rows Canonical participant/layer rows to inspect.
         * @param participant_ids Exact sorted participant set represented.
         * @return False for coherent pending coverage; malformed rows throw.
         */
        [[nodiscard]] ServiceEvidenceCoverage serviceEvidenceCoverage(
            const std::vector<MoEOverlayParticipantLayerServiceTotals> &rows,
            const std::vector<int> &participant_ids) const;

        /**
         * @brief Enter one topology-wide typed service-readiness round.
         *
         * The private evidence lane is the round authority. Every rank submits
         * exactly one disposition in collective order; no rank-local coverage
         * decision may create or omit a round.
         *
         * @param local_readiness Complete, incomplete, or stopping disposition.
         * @throws std::logic_error For invalid state or lane ownership.
         * @throws std::runtime_error When the exchange cannot be started.
         */
        void beginServiceReadinessRound(
            MoEOverlayServiceEvidenceReadiness local_readiness);

        /** @brief Compose one complete canonical profile bundle. */
        void composeCompleteEvidence(
            std::vector<MoEOverlayParticipantLayerServiceTotals> rows);

        /**
         * @brief Release-publish successful terminal certification.
         *
         * Host profiles reach this edge only after calibration-era demand was
         * asynchronously drained and discarded. Detached device profiles
         * instead rely on the device controller's captured rebase epoch.
         */
        void markComplete();

        /**
         * @brief Publish the certified participant matrix and tier crossovers.
         *
         * Exact participant rows make every later placement decision
         * reproducible from PerfStats evidence. Tier-priority crossovers remain
         * diagnostics rather than rejection rules because measured service is
         * the live economy authority.
         *
         * @param profile Complete measured participant/tier/layer matrix.
         * @param plan Immutable integer-priority and capacity authority.
         */
        void recordCertifiedServiceEconomy(
            const MoERoutedTierServiceProfile &profile,
            const MoERoutedExpertPlacementPlan &plan) const;

        /** @brief Create a deterministic identity over exact raw service totals. */
        [[nodiscard]] std::string serviceMeasurementIdentity(
            const std::vector<MoEOverlayParticipantLayerServiceTotals> &rows)
            const;

        /** @brief Publish the first terminal diagnostic and stop progression. */
        void fail(std::string message) noexcept;

        Config config_;
        std::atomic<MoEOverlayEconomyCertificationState> state_{
            MoEOverlayEconomyCertificationState::CalibratingMovement};
        std::atomic<bool> healthy_{true};
        std::atomic<bool> stop_requested_{false};
        mutable std::mutex failure_mutex_;
        std::string failure_message_;
        /** Protects the one release-before-Complete detached publication. */
        mutable std::mutex detached_profiles_mutex_;
        std::optional<MoEOverlayCertifiedEconomyProfiles>
            detached_profiles_;
        /**
         * Complete host profiles withheld until the routing histogram rebase.
         *
         * The maintenance worker is the sole reader/writer. Keeping this value
         * separate from the authority makes it impossible for Dynamic policy
         * to consume the service-calibration request distribution.
         */
        std::optional<MoEOverlayCertifiedEconomyProfiles>
            pending_host_profiles_;
        std::optional<MoEOverlaySealedMigrationMeasurements>
            expanded_migration_measurements_;
        /** Snapshot retained across the readiness vote and service all-gather. */
        std::optional<std::vector<MoEOverlayParticipantLayerServiceTotals>>
            ready_local_service_evidence_;
        /** Last reported missing coordinate, suppressing maintenance-loop spam. */
        std::optional<std::array<int, 3>> last_reported_service_deficit_;
        /** Non-empty layer mask last printed for that missing coordinate. */
        std::string last_reported_service_observed_layers_;
        /** Earliest maintenance poll allowed to open the next evidence round. */
        std::chrono::steady_clock::time_point
            next_service_readiness_attempt_{};

        std::atomic<std::uint64_t> polls_{0};
        std::atomic<std::uint64_t> movement_calibration_polls_{0};
        std::atomic<std::uint64_t> service_snapshot_attempts_{0};
        std::atomic<std::uint64_t> service_snapshot_contentions_{0};
        std::atomic<std::uint64_t> incomplete_service_snapshots_{0};
        std::atomic<std::uint64_t> profiles_composed_{0};
        std::atomic<std::uint64_t> routing_evidence_rebase_polls_{0};
        std::atomic<std::uint64_t> routing_evidence_rebases_{0};
        std::atomic<std::uint64_t> certifications_installed_{0};
        std::atomic<std::uint64_t> detached_profiles_completed_{0};
        std::atomic<std::uint64_t> fatal_failures_{0};
    };
} // namespace llaminar2
