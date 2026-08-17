/**
 * @file MoEOverlayEconomyCertificationController.h
 * @brief Pollable installation of measured ExpertOverlay economics.
 *
 * Dynamic residency must not consume a histogram until both prepared-expert
 * service cost and real transfer/interference cost are measured. This
 * controller joins those independent evidence streams, expands only exact
 * manifest-equivalent layer representatives, composes the complete immutable
 * profiles, and installs them into the residency authority. Every operation is
 * maintenance-owned and non-blocking with respect to inference.
 */

#pragma once

#include "MoEOverlayEconomyCalibrationController.h"
#include "MoEOverlayEconomyProfileComposer.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace llaminar2
{
    /** @brief Observable lifecycle of rank-local economy certification. */
    enum class MoEOverlayEconomyCertificationState : std::uint8_t
    {
        CalibratingMovement,
        AwaitingServiceEvidence,
        ExchangingServiceReadiness,
        ExchangingServiceEvidence,
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
        std::uint64_t certifications_installed = 0;
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
            MoERoutedExpertModelMetadata model_metadata;
            MoEOverlayMigrationEconomyPolicy economy_policy;
            /** Immutable phases reachable under this instance's MTP policy. */
            ExpertHistogramProductionSourceMask active_sources =
                kAllExpertHistogramProductionSources;
            /** Optional all-rank service and attempt evidence lane. */
            std::shared_ptr<IMoEOverlayEconomyEvidenceExchange>
                evidence_exchange;
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

        /** @brief Compose and install one complete canonical service matrix. */
        void installCompleteEvidence(
            std::vector<MoEOverlayParticipantLayerServiceTotals> rows);

        /**
         * @brief Publish exact phase/backend crossovers without rejecting them.
         * @param profile Complete measured tier/layer service matrix.
         * @param plan Immutable integer-priority and capacity authority.
         */
        void recordServicePriorityCrossovers(
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
        std::optional<MoEOverlaySealedMigrationMeasurements>
            expanded_migration_measurements_;
        /** Snapshot retained across the readiness vote and service all-gather. */
        std::optional<std::vector<MoEOverlayParticipantLayerServiceTotals>>
            ready_local_service_evidence_;
        /** Last reported missing coordinate, suppressing maintenance-loop spam. */
        std::optional<std::array<int, 3>> last_reported_service_deficit_;

        std::atomic<std::uint64_t> polls_{0};
        std::atomic<std::uint64_t> movement_calibration_polls_{0};
        std::atomic<std::uint64_t> service_snapshot_attempts_{0};
        std::atomic<std::uint64_t> service_snapshot_contentions_{0};
        std::atomic<std::uint64_t> incomplete_service_snapshots_{0};
        std::atomic<std::uint64_t> profiles_composed_{0};
        std::atomic<std::uint64_t> certifications_installed_{0};
        std::atomic<std::uint64_t> fatal_failures_{0};
    };
} // namespace llaminar2
