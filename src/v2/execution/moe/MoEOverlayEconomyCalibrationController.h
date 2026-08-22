/**
 * @file MoEOverlayEconomyCalibrationController.h
 * @brief Bounded real-transfer profiler for ExpertOverlay movement economy.
 *
 * Startup profiling exercises already-materialized production transfer lanes
 * with reversible reciprocal expert swaps. It never runs synthetic inference,
 * pairs request latencies, or publishes a candidate residency epoch. A small
 * robust corpus is measured over representative weight geometries and physical
 * endpoint pairs; live inference service telemetry is collected separately.
 */

#pragma once

#include "MoEOverlayEconomyCalibrationPlanner.h"
#include "MoEOverlayEconomyEvidenceExchange.h"
#include "MoEOverlayMigrationMeasurementExchange.h"

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
    /** @brief Observable lifecycle of the bounded transfer profiler. */
    enum class MoEOverlayEconomyCalibrationState
    {
        StartWave,
        AwaitConcurrentWave,
        AbortCleanup,
        AwaitEvidenceExchange,
        Complete,
        Failed,
        Stopped,

        /* Retained spellings keep diagnostics source-compatible while the
         * obsolete inference-pair tests are removed. They are never live. */
        ArmBaseline,
        AwaitBaseline,
        AwaitBaselineReadiness,
        AwaitConcurrentInference,
        AwaitLateConcurrentSample,
    };

    /** @brief Race-safe proof counters for bounded transfer profiling. */
    struct MoEOverlayEconomyCalibrationStats
    {
        std::uint64_t polls = 0;
        std::uint64_t wave_start_attempts = 0;
        std::uint64_t waves_started = 0;
        std::uint64_t waves_deferred = 0;
        std::uint64_t waves_completed = 0;
        std::uint64_t waves_aborted = 0;
        std::uint64_t evidence_exchanges_started = 0;
        std::uint64_t evidence_exchanges_completed = 0;
        std::uint64_t accepted_pairs = 0;
        std::uint64_t fatal_failures = 0;

        /* Zero-only legacy fields remain during diagnostic-reader migration. */
        std::uint64_t baseline_arms = 0;
        std::uint64_t baseline_samples = 0;
        std::uint64_t baseline_readiness_started = 0;
        std::uint64_t baseline_readiness_completed = 0;
        std::uint64_t baseline_readiness_retries = 0;
        std::uint64_t concurrent_arms = 0;
        std::uint64_t concurrent_launch_wait_polls = 0;
        std::uint64_t concurrent_launches_during_inference = 0;
        std::uint64_t concurrent_launch_misses = 0;
        std::uint64_t concurrent_samples = 0;
        std::uint64_t exact_overlap_samples = 0;
        std::uint64_t partial_overlap_rejections = 0;
        std::uint64_t workload_pair_rejections = 0;
    };

    /**
     * @brief Pollable owner of real, non-publishable transfer profile waves.
     *
     * One maintenance worker owns all methods except the documented atomic
     * snapshots. Every wave is staged through the production transport and
     * then aborted, proving preparation/repack/DMA without changing residency.
     */
    class MoEOverlayEconomyCalibrationController final
    {
    public:
        /** @brief Immutable production dependencies for one bounded profile. */
        struct Config
        {
            std::shared_ptr<const MoEOverlayEconomyCalibrationPlanner> planner;
            std::shared_ptr<MoEOverlayMigrationMeasurementLedger> ledger;
            std::shared_ptr<MoEOverlayMigrationMeasurementJournal> journal;
            std::shared_ptr<IMoEOverlayResidencyTransport> transport;
            /** Optional all-rank merger for distributed partial timings. */
            std::shared_ptr<IMoEOverlayEconomyEvidenceExchange>
                evidence_exchange;
            std::string perf_device;
        };

        /**
         * @brief Validate the finite reciprocal-pair corpus and retain owners.
         * @throws std::invalid_argument For incomplete dependencies, topology,
         *         or a busy distributed evidence lane.
         */
        explicit MoEOverlayEconomyCalibrationController(Config config);

        MoEOverlayEconomyCalibrationController(
            const MoEOverlayEconomyCalibrationController &) = delete;
        MoEOverlayEconomyCalibrationController &operator=(
            const MoEOverlayEconomyCalibrationController &) = delete;

        /** @brief Advance at most one non-blocking transfer/profile edge. */
        void poll() noexcept;

        /** @brief Stop admission and asynchronously reap owned work. */
        void requestStop() noexcept;

        /** @return Current release-published profiler state. */
        [[nodiscard]] MoEOverlayEconomyCalibrationState state() const noexcept;

        /** @return Whether no physical profiling/protocol error occurred. */
        [[nodiscard]] bool healthy() const noexcept;

        /** @return Stable first fatal diagnostic, or an empty string. */
        [[nodiscard]] std::string failureMessage() const;

        /** @return Race-safe copy of physical profiling counters. */
        [[nodiscard]] MoEOverlayEconomyCalibrationStats stats() const noexcept;

        /** @return Exact finite number of reciprocal profile waves. */
        [[nodiscard]] std::uint64_t expectedAcceptedPairs() const noexcept
        {
            return expected_profile_waves_;
        }

        /** @return False: transport profiling never requests inference work. */
        [[nodiscard]] bool awaitsInferenceSample() const noexcept
        {
            return false;
        }

        /** @return Zero because no inference-probe generation exists. */
        [[nodiscard]] std::uint64_t awaitedInferenceGeneration() const noexcept
        {
            return 0u;
        }

        /** @return SyntheticTest because transfer profiling has no phase. */
        [[nodiscard]] ExpertHistogramSource awaitedInferenceSource() const noexcept
        {
            return ExpertHistogramSource::SyntheticTest;
        }

        /** @return Immutable sealed profile rows after Complete, else null. */
        [[nodiscard]] const MoEOverlaySealedMigrationMeasurements *
        sealedMeasurements() const noexcept;

        /** @return Runtime service phases required by later certification. */
        [[nodiscard]] const ExpertHistogramProductionSourceMask &
        requiredSources() const noexcept
        {
            return config_.ledger->requiredSources();
        }

    private:
        /** @brief One reciprocal physical endpoint pair at one weight geometry. */
        struct PairJob
        {
            int first_participant = -1;
            int second_participant = -1;
            int layer = -1;
        };

        /** @return Stable directed coordinate naming the current reciprocal job. */
        [[nodiscard]] MoEOverlayMigrationMeasurementCoordinate
        currentCoordinate() const noexcept;

        /** @brief Reserve and launch the next reversible production transfer. */
        void startWave();

        /** @brief Poll physical readiness, then request reversible cleanup. */
        void pollWave();

        /** @brief Poll abort cleanup and collect rank-local timing rows. */
        void pollAbortCleanup();

        /** @brief Poll the all-rank timing merge for one completed wave. */
        void pollEvidenceExchange();

        /** @brief Record one complete reciprocal observation in the ledger. */
        void recordCompletedWave(
            const std::vector<MoEOverlayCompletedMigrationMeasurement> &rows);

        /** @brief Advance robust sample and topology job after one observation. */
        void advanceProfileWave();

        /** @brief Abort/reap active work before honoring a stop request. */
        void progressStop();

        /** @brief Enter abort cleanup and retain the first causal failure. */
        void beginAbort(std::string failure_after_cleanup = {});

        /** @brief Publish a terminal failure after safe ownership cleanup. */
        void fail(std::string message) noexcept;

        Config config_;
        std::vector<PairJob> jobs_;
        std::atomic<MoEOverlayEconomyCalibrationState> state_{
            MoEOverlayEconomyCalibrationState::StartWave};
        std::atomic<bool> healthy_{true};
        std::atomic<bool> stop_requested_{false};
        mutable std::mutex failure_mutex_;
        std::string failure_message_;

        std::size_t job_index_ = 0;
        std::uint64_t sample_index_ = 0;
        std::uint64_t next_profile_sequence_ = 0;
        std::uint64_t active_profile_sequence_ = 0;
        std::unique_ptr<IMoEOverlayResidencyWave> active_wave_;
        std::vector<MoEOverlayCompletedMigrationMeasurement> local_rows_;
        bool stage_completed_ = false;
        bool evidence_exchange_active_ = false;
        std::string failure_after_cleanup_;
        std::optional<MoEOverlaySealedMigrationMeasurements> sealed_;
        const std::chrono::steady_clock::time_point profiling_started_at_;

        std::atomic<std::uint64_t> polls_{0};
        std::atomic<std::uint64_t> wave_start_attempts_{0};
        std::atomic<std::uint64_t> waves_started_{0};
        std::atomic<std::uint64_t> waves_deferred_{0};
        std::atomic<std::uint64_t> waves_completed_{0};
        std::atomic<std::uint64_t> waves_aborted_{0};
        std::atomic<std::uint64_t> evidence_exchanges_started_{0};
        std::atomic<std::uint64_t> evidence_exchanges_completed_{0};
        std::atomic<std::uint64_t> accepted_pairs_{0};
        std::atomic<std::uint64_t> fatal_failures_{0};
        std::uint64_t expected_profile_waves_ = 0u;
    };
} // namespace llaminar2
