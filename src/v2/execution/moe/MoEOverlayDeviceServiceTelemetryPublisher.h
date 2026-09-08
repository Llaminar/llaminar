/**
 * @file MoEOverlayDeviceServiceTelemetryPublisher.h
 * @brief Async GPU service observations for a host-resident overlay authority.
 *
 * GPU routed-expert stages accumulate timing in device-local cells. A
 * host-resident ExpertOverlay authority cannot certify its economy from those
 * cells until a finite maintenance graph publishes them into mapped pages.
 * This service owns exactly that observation edge: it has no histogram,
 * placement, migration, or epoch-policy authority.
 */

#pragma once

#include "MoEOverlayDeviceControllerRuntimeBinding.h"
#include "MoEOverlayParticipantResidency.h"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace llaminar2
{
    /** One coherent cumulative snapshot delivered to economy certification. */
    struct MoEOverlayDeviceServiceTelemetrySnapshot
    {
        int participant_id = -1;
        std::uint64_t publication_generation = 0u;
        std::uint32_t valid_sample_count = 0u;
        std::uint32_t armed_sample_count = 0u;
        std::uint64_t begun_sample_count = 0u;
        std::vector<MoEOverlayParticipantLayerServiceTotals> rows;

        /** @return Whether this record names one nonempty coherent publication. */
        [[nodiscard]] bool valid() const noexcept
        {
            return participant_id >= 0 && publication_generation != 0u &&
                   !rows.empty();
        }
    };

    /** @brief Race-safe evidence for one publisher lifetime. */
    struct MoEOverlayDeviceServiceTelemetryPublisherStats
    {
        std::uint64_t boundary_submissions = 0u;
        std::uint64_t boundary_deferrals = 0u;
        std::uint64_t publication_launches = 0u;
        std::uint64_t publication_completions = 0u;
        std::uint64_t snapshots_delivered = 0u;
        std::uint64_t fatal_failures = 0u;
    };

    /**
     * @brief Captured, event-polled publication for every local GPU endpoint.
     *
     * Each endpoint owns a dedicated non-null maintenance stream, terminal
     * event, mapped page, and retained single-kernel graph. Polling first asks
     * the model executor to append the exact inference-terminal dependency;
     * only then is the publication graph launched. Both the worker submission
     * and GPU terminal are polled without blocking the inference caller.
     */
    class MoEOverlayDeviceServiceTelemetryPublisher final
    {
    public:
        /** Immutable topology and low-frequency observation cadence. */
        struct Config
        {
            std::vector<MoEOverlayDeviceControllerRuntimeBinding>
                runtime_bindings;
            /** Prevent an idle maintenance loop from launching duplicate views. */
            std::chrono::milliseconds minimum_snapshot_interval{100};
            std::string perf_device;
        };

        /**
         * @brief Materialize mapped pages and retained graphs at model setup.
         * @throws std::invalid_argument for empty, duplicate, or incomplete bindings.
         * @throws std::runtime_error when a backend resource cannot be retained.
         */
        explicit MoEOverlayDeviceServiceTelemetryPublisher(Config config);

        /** @brief Drain tiny retained publications and release setup resources. */
        ~MoEOverlayDeviceServiceTelemetryPublisher();

        MoEOverlayDeviceServiceTelemetryPublisher(
            const MoEOverlayDeviceServiceTelemetryPublisher &) = delete;
        MoEOverlayDeviceServiceTelemetryPublisher &operator=(
            const MoEOverlayDeviceServiceTelemetryPublisher &) = delete;

        /**
         * @brief Advance each endpoint by at most one nonblocking state edge.
         *
         * Ready snapshots are appended to @p output. An empty successful poll
         * means that an inference boundary, worker submission, GPU event, or
         * cadence delay is still pending. The method never waits for an event
         * or synchronizes a stream.
         *
         * @param output Receives zero or more complete local GPU snapshots.
         * @param error Optional first fatal diagnostic.
         * @return False only after a terminal protocol/backend failure.
         */
        [[nodiscard]] bool poll(
            std::vector<MoEOverlayDeviceServiceTelemetrySnapshot> *output,
            std::string *error = nullptr) noexcept;

        /** @brief Stop admitting new publications; already submitted work drains. */
        void requestStop() noexcept;

        /** @return Whether no terminal publication failure occurred. */
        [[nodiscard]] bool healthy() const noexcept;

        /** @return Stable first fatal diagnostic, or an empty string. */
        [[nodiscard]] std::string failureMessage() const;

        /** @return Race-safe publication counters. */
        [[nodiscard]] MoEOverlayDeviceServiceTelemetryPublisherStats stats()
            const noexcept;

        /** @return Number of process-local GPU publications retained at setup. */
        [[nodiscard]] std::size_t endpointCount() const noexcept;

    private:
        struct Endpoint;

        /** @brief Allocate one endpoint's mapped page, stream, event, and graph. */
        void materializeEndpoint(Endpoint &endpoint);

        /** @brief Release one endpoint after any submitted terminal is complete. */
        void releaseEndpoint(Endpoint &endpoint) noexcept;

        /** @brief Latch the first fatal diagnostic and reject new publication. */
        [[nodiscard]] bool fail(std::string message) noexcept;

        Config config_;
        std::vector<std::unique_ptr<Endpoint>> endpoints_;
        std::atomic<bool> stop_requested_{false};
        std::atomic<bool> healthy_{true};
        mutable std::mutex failure_mutex_;
        std::string failure_message_;
        std::atomic<std::uint64_t> boundary_submissions_{0u};
        std::atomic<std::uint64_t> boundary_deferrals_{0u};
        std::atomic<std::uint64_t> publication_launches_{0u};
        std::atomic<std::uint64_t> publication_completions_{0u};
        std::atomic<std::uint64_t> snapshots_delivered_{0u};
        std::atomic<std::uint64_t> fatal_failures_{0u};
    };
} // namespace llaminar2
