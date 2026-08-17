/**
 * @file MoEOverlayResidencyMaintenanceService.h
 * @brief Background driver for transactional ExpertOverlay tier migration.
 *
 * Inference threads only publish routing evidence and acquire immutable
 * residency tickets. This service owns the host maintenance thread that
 * rotates completed histogram windows, retries shadow-capacity backpressure,
 * polls exact transfer/publication events, and retires old banks. No method on
 * the inference path waits for this worker.
 */

#pragma once

#include "MoEOverlayEconomyCertificationController.h"
#include "MoEOverlayHistogramPublisher.h"
#include "MoEOverlayResidencyAuthority.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace llaminar2
{
    /** @brief Observable lifecycle of the ExpertOverlay maintenance worker. */
    enum class MoEOverlayMaintenanceState
    {
        Starting,   ///< Worker created but not yet inside its poll loop.
        CertifyingEconomy, ///< Real service/movement evidence is still sealing.
        Waiting,    ///< No full histogram window or migration wave is ready.
        DrainingEvidence, ///< Device histogram banks are being copied/reset.
        PublishingEvidence, ///< Coordinator is sending one frozen window.
        ReceivingEvidence, ///< Peer is awaiting authoritative routing evidence.
        Deferred,   ///< Frozen transaction awaits destination shadow capacity.
        Staging,    ///< Preparation and transfer events remain in flight.
        Committing, ///< Inactive participant banks are being published.
        Draining,   ///< Shutdown rejected new work and is reaping resources.
        Failed,     ///< Fatal protocol or transport error stopped proposals.
        Stopped,    ///< Every wave, abort, and retirement has quiesced.
    };

    /** @brief Race-safe counters for one maintenance-service lifetime. */
    struct MoEOverlayResidencyMaintenanceStats
    {
        uint64_t worker_starts = 0;       ///< Background worker entries.
        uint64_t poll_iterations = 0;     ///< Non-blocking authority polls.
        uint64_t notifications = 0;       ///< Explicit early-wake notifications.
        uint64_t economy_certification_polls = 0; ///< Setup evidence progress.
        uint64_t economy_certifications = 0; ///< Immutable installs observed.
        uint64_t proposals = 0;           ///< Frozen histogram/static proposals.
        uint64_t deferred_attempts = 0;   ///< Backpressured stage attempts.
        uint64_t waves_started = 0;       ///< Async migration waves begun.
        uint64_t committed_waves = 0;     ///< Candidate epochs published.
        uint64_t dynamic_no_movement = 0; ///< Dynamic windows needing no move.
        uint64_t static_no_movement = 0;  ///< Explicit static immobility checks.
        uint64_t histogram_windows_published = 0; ///< Coordinator send completions.
        uint64_t histogram_windows_received = 0; ///< Authenticated peer windows.
        uint64_t histogram_receives_rearmed = 0; ///< Next-generation peer Irecvs.
        uint64_t fatal_failures = 0;      ///< Terminal service failures.
    };

    /**
     * @brief Sole background scheduler for one overlay residency authority.
     *
     * The service retains both authority and transport so their streams,
     * events, source pins, and inactive slots outlive every in-flight wave. A
     * deferred transaction is retained verbatim: later routing evidence lands
     * in the next RCU histogram bank and cannot mutate or replace the candidate
     * being retried.
     *
     * Destruction requests shutdown and drains by event polling. It never calls
     * a device/stream synchronize operation, but it deliberately waits for the
     * maintenance thread to release every owned asynchronous resource. Runner
     * teardown must therefore stop ticket admission before destroying this
     * service.
     */
    class MoEOverlayResidencyMaintenanceService final
    {
    public:
        /** @brief Construction dependencies and idle polling economy policy. */
        struct Config
        {
            /** Single publication authority shared with inference dispatch. */
            std::shared_ptr<MoEOverlayResidencyAuthority> authority;
            /** Persistent event-driven physical migration transport. */
            std::shared_ptr<IMoEOverlayResidencyTransport> transport;
            /**
             * Required for an initially uncertified dynamic local authority.
             * The maintenance worker polls it to completion before it rotates
             * or publishes any histogram window.
             */
            std::shared_ptr<MoEOverlayEconomyCertificationController>
                economy_certification;
            /**
             * Optional authoritative frozen-window publication lane.
             *
             * When present, this service is in distributed mode. Only its
             * coordinator consults the process-local histogram; peers derive
             * proposals exclusively from authenticated published windows.
             */
            std::shared_ptr<IMoEOverlayHistogramPublisher>
                histogram_publisher;
            /**
             * Maximum delay before the worker polls without a notification.
             * This is a responsiveness policy, not a correctness timeout.
             */
            std::chrono::microseconds idle_poll_interval{
                std::chrono::milliseconds(2)};
            /** PerfStats device/topology label for maintenance evidence. */
            std::string perf_device;
        };

        /**
         * @brief Validate dependencies and immediately start the worker.
         * @throws std::invalid_argument for null dependencies or bad cadence.
         */
        explicit MoEOverlayResidencyMaintenanceService(Config config);

        /** @brief Request shutdown and drain all event-owned resources. */
        ~MoEOverlayResidencyMaintenanceService();

        MoEOverlayResidencyMaintenanceService(
            const MoEOverlayResidencyMaintenanceService &) = delete;
        MoEOverlayResidencyMaintenanceService &operator=(
            const MoEOverlayResidencyMaintenanceService &) = delete;

        /**
         * @brief Wake the worker after routing evidence or capacity changes.
         *
         * Notification is optional because the worker also polls at the
         * configured cadence. The method never waits and is safe on inference
         * and destination-slot release paths.
         */
        void notifyMaintenanceProgress() noexcept;

        /**
         * @brief Stop accepting proposals and drain active/retiring waves.
         *
         * This method is idempotent. It joins only the maintenance host thread;
         * GPU and network completion is observed through non-blocking polls.
         */
        void stopAndDrain();

        /** @return Current worker lifecycle state. */
        [[nodiscard]] MoEOverlayMaintenanceState state() const noexcept;

        /** @return Whether no fatal protocol or transport failure was observed. */
        [[nodiscard]] bool healthy() const noexcept;

        /** @return Stable copy of the first fatal diagnostic, or empty string. */
        [[nodiscard]] std::string failureMessage() const;

        /** @return Race-safe copy of service counters. */
        [[nodiscard]] MoEOverlayResidencyMaintenanceStats stats() const noexcept;

    private:
        /** @brief Worker entry that catches all failures and owns state progress. */
        void run(std::stop_token stop_token) noexcept;

        /** @brief Perform one non-blocking maintenance iteration. */
        void pollOnce(bool allow_new_proposal);

        /** @brief Start or retry the exact retained transaction. */
        void tryBeginRetainedTransaction();

        /** @brief Progress coordinator publication or peer reception once. */
        void progressDistributedProposal();

        /** @brief Derive and retain one exact coordinator-published proposal. */
        void retainDistributedProposal(
            std::shared_ptr<const DecodeExpertHistogramWindow> window);

        /** @brief Interpret progress for the authority-owned active wave. */
        void handleActiveWaveResult(
            const MoEOverlayResidencyApplyResult &result);

        /** @brief Record the first terminal failure and disable proposals. */
        void fail(std::string message) noexcept;

        /** @brief Export one low-frequency service counter to PerfStats. */
        void recordPerfCounter(const char *name, double value = 1.0) const;

        Config config_;
        std::jthread worker_;

        std::atomic<MoEOverlayMaintenanceState> state_{
            MoEOverlayMaintenanceState::Starting};
        std::atomic<bool> healthy_{true};
        /** Serializes first-failure publication before `healthy_` becomes false. */
        std::atomic<bool> failure_recorded_{false};
        std::atomic<bool> shutdown_requested_{false};
        std::mutex shutdown_mutex_;

        std::mutex wake_mutex_;
        std::condition_variable wake_cv_;
        bool wake_requested_ = false;

        mutable std::mutex failure_mutex_;
        std::string failure_message_;

        /* These transaction fields are owned exclusively by `worker_`. */
        MoEOverlayResidencyTransaction retained_transaction_;
        bool has_retained_transaction_ = false;
        bool active_wave_ = false;
        bool static_check_complete_ = false;
        /** Coordinator-owned window retained until its MPI sends complete. */
        std::shared_ptr<const DecodeExpertHistogramWindow>
            publishing_histogram_window_;
        /** True while coordinator sends or the peer's next Irecv is active. */
        bool histogram_exchange_active_ = false;
        /** Worker-local edge detector for the one immutable certificate. */
        bool economy_certification_observed_ = false;
        std::atomic<uint64_t> worker_starts_{0};
        std::atomic<uint64_t> poll_iterations_{0};
        std::atomic<uint64_t> notifications_{0};
        std::atomic<uint64_t> economy_certification_polls_{0};
        std::atomic<uint64_t> economy_certifications_{0};
        std::atomic<uint64_t> proposals_{0};
        std::atomic<uint64_t> deferred_attempts_{0};
        std::atomic<uint64_t> waves_started_{0};
        std::atomic<uint64_t> committed_waves_{0};
        std::atomic<uint64_t> dynamic_no_movement_{0};
        std::atomic<uint64_t> static_no_movement_{0};
        std::atomic<uint64_t> histogram_windows_published_{0};
        std::atomic<uint64_t> histogram_windows_received_{0};
        std::atomic<uint64_t> histogram_receives_rearmed_{0};
        std::atomic<uint64_t> fatal_failures_{0};
    };

} // namespace llaminar2
