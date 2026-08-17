/**
 * @file MoEOverlayEconomyCalibrationController.h
 * @brief Non-blocking calibration state machine for overlay movement economy.
 *
 * The controller is polled by the existing maintenance worker. It pairs an
 * inference-only production invocation with the identical live workload while
 * a real non-publishable expert swap runs through the production transport.
 * Staging is always aborted after evidence capture; calibration can never
 * mutate the live residency epoch. No method waits for inference, a stream, or
 * a collective.
 */

#pragma once

#include "MoEOverlayEconomyCalibrationPlanner.h"
#include "MoEOverlayEconomyEvidenceExchange.h"
#include "MoEOverlayInferenceInterferenceProbe.h"
#include "MoEOverlayMigrationMeasurementExchange.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace llaminar2
{
    /** @brief Observable phase of one rank-local calibration controller. */
    enum class MoEOverlayEconomyCalibrationState
    {
        ArmBaseline,
        AwaitBaseline,
        StartWave,
        AwaitConcurrentInference,
        AwaitConcurrentWave,
        AbortCleanup,
        AwaitLateConcurrentSample,
        AwaitEvidenceExchange,
        Complete,
        Failed,
        Stopped,
    };

    /** @brief Race-safe proof counters for calibration progress and overlap. */
    struct MoEOverlayEconomyCalibrationStats
    {
        std::uint64_t polls = 0;
        std::uint64_t baseline_arms = 0;
        std::uint64_t baseline_samples = 0;
        std::uint64_t wave_start_attempts = 0;
        std::uint64_t waves_started = 0;
        std::uint64_t waves_deferred = 0;
        std::uint64_t concurrent_arms = 0;
        std::uint64_t concurrent_launch_wait_polls = 0;
        std::uint64_t concurrent_launches_during_inference = 0;
        std::uint64_t concurrent_launch_misses = 0;
        std::uint64_t concurrent_samples = 0;
        std::uint64_t exact_overlap_samples = 0;
        std::uint64_t partial_overlap_rejections = 0;
        std::uint64_t workload_pair_rejections = 0;
        std::uint64_t waves_aborted = 0;
        std::uint64_t accepted_pairs = 0;
        std::uint64_t fatal_failures = 0;
    };

    /**
     * @brief Pollable owner of rank-local real-transport calibration attempts.
     *
     * A process-local deployment seals complete rows directly. A distributed
     * deployment supplies explicit partial rows from the production transport
     * and joins them through its private all-rank evidence lane. The controller
     * owns its active wave and retains it through asynchronous abort cleanup;
     * neither form can publish a calibration candidate as live residency.
     */
    class MoEOverlayEconomyCalibrationController final
    {
    public:
        /** @brief Immutable production dependencies for one calibration run. */
        struct Config
        {
            std::shared_ptr<const MoEOverlayEconomyCalibrationPlanner> planner;
            std::shared_ptr<MoEOverlayMigrationMeasurementLedger> ledger;
            std::shared_ptr<MoEOverlayMigrationMeasurementJournal> journal;
            std::shared_ptr<MoEOverlayInferenceInterferenceProbe> probe;
            std::shared_ptr<IMoEOverlayResidencyTransport> transport;
            /** Optional all-rank merger for distributed partial evidence. */
            std::shared_ptr<IMoEOverlayEconomyEvidenceExchange>
                evidence_exchange;
            std::string perf_device;
        };

        /**
         * @brief Validate complete directed-pair geometry and retain owners.
         * @throws std::invalid_argument For null dependencies, incomplete
         *         reverse coordinates, or a non-idle probe.
         */
        explicit MoEOverlayEconomyCalibrationController(Config config);

        MoEOverlayEconomyCalibrationController(
            const MoEOverlayEconomyCalibrationController &) = delete;
        MoEOverlayEconomyCalibrationController &operator=(
            const MoEOverlayEconomyCalibrationController &) = delete;

        /** @brief Advance at most one bounded state-machine edge without waiting. */
        void poll() noexcept;

        /**
         * @brief Stop admitting calibration attempts and reap owned async work.
         *
         * The request is non-blocking. The maintenance owner continues calling
         * @ref poll until state becomes Stopped, Complete, or Failed.
         */
        void requestStop() noexcept;

        /** @return Current externally observable calibration phase. */
        [[nodiscard]] MoEOverlayEconomyCalibrationState state() const noexcept;

        /** @return Whether no fatal protocol or evidence error was observed. */
        [[nodiscard]] bool healthy() const noexcept;

        /** @return Stable first fatal diagnostic, or an empty string. */
        [[nodiscard]] std::string failureMessage() const;

        /** @return Race-safe copy of calibration proof counters. */
        [[nodiscard]] MoEOverlayEconomyCalibrationStats stats() const noexcept;

        /**
         * @brief Return the sealed robust corpus after Complete.
         * @return Pointer owned by this controller, or null before completion.
         */
        [[nodiscard]] const MoEOverlaySealedMigrationMeasurements *
        sealedMeasurements() const noexcept;

        /** @return Immutable runtime phases required by the measurement ledger. */
        [[nodiscard]] const ExpertHistogramProductionSourceMask &
        requiredSources() const noexcept
        {
            return config_.ledger->requiredSources();
        }

    private:
        /** @brief Canonical unordered participant pair at one measured layer. */
        struct PairJob
        {
            int first_participant = -1;
            int second_participant = -1;
            int layer = -1;
        };

        /** @brief Current production phase in stable histogram order. */
        [[nodiscard]] ExpertHistogramSource currentSource() const noexcept;

        /** @brief First directed coordinate used to identify a paired probe. */
        [[nodiscard]] MoEOverlayMigrationMeasurementCoordinate
        currentCoordinate() const noexcept;

        /** @brief Arm one unconstrained baseline workload sample. */
        void armBaseline();

        /** @brief Consume and validate the current baseline, when ready. */
        void awaitBaseline();

        /** @brief Build and begin one real non-publishable pair-swap wave. */
        void startWave();

        /** @brief Wait until the exact inference ticket owns the launch window. */
        void awaitConcurrentInference();

        /** @brief Reconstruct the immutable request currently owned by the probe. */
        [[nodiscard]] MoEOverlayInterferenceProbeRequest
        currentConcurrentRequest() const;

        /** @brief Poll transfer readiness and collect concurrent inference. */
        void awaitConcurrentWave();

        /** @brief Reap every asynchronous abort edge before evaluating evidence. */
        void pollAbortCleanup();

        /** @brief Wait only by polling for a sample that was already running. */
        void awaitLateConcurrentSample();

        /** @brief Consume a completed concurrent sample, if one is available. */
        bool tryConsumeConcurrentSample();

        /** @brief Poll one in-flight all-rank attempt evidence exchange. */
        void pollEvidenceExchange();

        /** @brief Validate overlap and publish or exchange one attempt. */
        void finalizeAttempt();

        /** @brief Record one accepted local or globally merged attempt. */
        void recordAcceptedAttempt(
            const std::vector<MoEOverlayCompletedMigrationMeasurement> &rows,
            std::uint64_t baseline_nanoseconds,
            std::uint64_t concurrent_nanoseconds);

        /** @brief Discard an unusable attempt and require a fresh paired baseline. */
        void retryAttempt();

        /** @brief Advance sample, phase, and pair indices after accepted evidence. */
        void advanceAcceptedPair();

        /** @brief Abort the active wave and retain its cleanup ownership. */
        void beginAbort(std::string failure_after_cleanup = {});

        /** @brief Retain a fatal error until no inference ticket owns the probe. */
        void deferFailureUntilProbeQuiescent(std::string message) noexcept;

        /** @brief Abort/reap active work and preserve any running inference ticket. */
        void progressStop();

        /** @brief Publish one terminal failure after all safe cleanup edges. */
        void fail(std::string message) noexcept;

        Config config_;
        std::vector<PairJob> jobs_;
        /** Runtime-reachable phases in canonical histogram order. */
        std::vector<ExpertHistogramSource> phases_;
        std::atomic<MoEOverlayEconomyCalibrationState> state_{
            MoEOverlayEconomyCalibrationState::ArmBaseline};
        std::atomic<bool> healthy_{true};
        std::atomic<bool> stop_requested_{false};
        mutable std::mutex failure_mutex_;
        std::string failure_message_;

        std::size_t job_index_ = 0;
        std::size_t phase_index_ = 0;
        std::uint64_t sample_index_ = 0;
        std::uint64_t calibration_sequence_ = 0;
        std::optional<MoEOverlayInterferenceProbeSample> baseline_sample_;
        std::optional<MoEOverlayInterferenceProbeSample> concurrent_sample_;
        std::optional<MoEOverlayResidencyWaveInterval> stage_interval_;
        std::optional<MoEOverlayCalibrationAttemptEvidence>
            pending_attempt_evidence_;
        std::unique_ptr<IMoEOverlayResidencyWave> active_wave_;
        std::vector<MoEOverlayCompletedMigrationMeasurement> local_rows_;
        bool stage_completed_ = false;
        bool stage_dispatch_started_ = false;
        std::string failure_after_cleanup_;
        std::optional<MoEOverlaySealedMigrationMeasurements> sealed_;

        std::atomic<std::uint64_t> polls_{0};
        std::atomic<std::uint64_t> baseline_arms_{0};
        std::atomic<std::uint64_t> baseline_samples_{0};
        std::atomic<std::uint64_t> wave_start_attempts_{0};
        std::atomic<std::uint64_t> waves_started_{0};
        std::atomic<std::uint64_t> waves_deferred_{0};
        std::atomic<std::uint64_t> concurrent_arms_{0};
        std::atomic<std::uint64_t> concurrent_launch_wait_polls_{0};
        std::atomic<std::uint64_t> concurrent_launches_during_inference_{0};
        std::atomic<std::uint64_t> concurrent_launch_misses_{0};
        std::atomic<std::uint64_t> concurrent_samples_{0};
        std::atomic<std::uint64_t> exact_overlap_samples_{0};
        std::atomic<std::uint64_t> partial_overlap_rejections_{0};
        std::atomic<std::uint64_t> workload_pair_rejections_{0};
        std::atomic<std::uint64_t> waves_aborted_{0};
        std::atomic<std::uint64_t> accepted_pairs_{0};
        std::atomic<std::uint64_t> fatal_failures_{0};
    };
} // namespace llaminar2
