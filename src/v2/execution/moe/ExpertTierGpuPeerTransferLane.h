/**
 * @file ExpertTierGpuPeerTransferLane.h
 * @brief Persistent event-polled packed transfer within one GPU backend.
 *
 * CUDA-to-CUDA and ROCm-to-ROCm tiers share the separated NativeVNNI layout,
 * so migration is a direct asynchronous blob copy.  This lane owns a named
 * destination auxiliary stream and one reusable completion event.  It joins an
 * exact source producer event when necessary, submits all separated regions,
 * and exposes readiness only through non-blocking event queries.
 */

#pragma once

#include "ExpertTierSourceReadiness.h"
#include "ExpertTierTransferMeasurement.h"
#include "GPUExpertTransfer.h"
#include "../../backends/DeviceId.h"
#include "../../transfer/TransferEngine.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>

namespace llaminar2
{
    class IBackend;
    class IWorkerGPUContext;

    /** @brief Host-visible lifecycle of one same-backend GPU transfer. */
    enum class ExpertTierGpuPeerTransferProgress : std::uint8_t
    {
        Idle,
        Pending,
        Ready,
        Failed,
    };

    /** @brief Cumulative proof counters for a persistent peer lane. */
    struct ExpertTierGpuPeerTransferLaneStats
    {
        std::uint64_t transfers_started = 0;
        std::uint64_t transfers_completed = 0;
        std::uint64_t bytes_submitted = 0;
        std::uint64_t pending_event_polls = 0;
        std::uint64_t failed_transfers = 0;
        std::uint64_t producer_event_waits = 0;
        std::uint64_t published_bank_sources = 0;
        std::uint64_t inference_stream_waits = 0;
        std::uint64_t blocking_synchronizations = 0;
        /** Most recent exact completed-transfer timing evidence. */
        ExpertTierProjectionTransferMeasurement last_measurement;
        /** Timing API failures; non-zero invalidates economy certification. */
        std::uint64_t timing_measurement_failures = 0;
    };

    /**
     * @brief Reusable direct-copy lane for CUDA/CUDA or ROCm/ROCm tiers.
     *
     * The destination device owns the stream and completion event.  Source and
     * destination storage are reserved before start, and the retained RCU bank
     * or exact producer event prevents source reclamation.  No inference stream
     * ever waits on this lane.
     */
    class ExpertTierGpuPeerTransferLane final
    {
    public:
        /** @brief Immutable endpoints and evidence identity. */
        struct Config
        {
            DeviceId source_device;
            DeviceId destination_device;
            /** Exact destination participant/cycle background stream. */
            PersistentTransferExecutionLane execution;
            std::string lane_name;
            std::string perf_device;
            /** Collect timing-event evidence for economy certification. */
            bool collect_timing_measurements = false;
        };

        /**
         * @brief Validate and retain one same-backend directed edge.
         * @throws std::invalid_argument For non-GPU, cross-backend, or unnamed
         *         topology.
         */
        explicit ExpertTierGpuPeerTransferLane(Config config);

        /** @brief Release the event only after its submitted work is quiescent. */
        ~ExpertTierGpuPeerTransferLane();

        ExpertTierGpuPeerTransferLane(
            const ExpertTierGpuPeerTransferLane &) = delete;
        ExpertTierGpuPeerTransferLane &operator=(
            const ExpertTierGpuPeerTransferLane &) = delete;

        /**
         * @brief Bind the pooled destination stream and create reusable events.
         * @param error Optional exact setup failure.
         * @return True when the complete persistent lane exists.
         */
        bool materialize(std::string *error = nullptr) noexcept;

        /**
         * @brief Submit one complete byte-compatible packed projection copy.
         * @param source Immutable separated source arrays.
         * @param destination Preallocated inactive destination arrays.
         * @param source_readiness Exact event or installed-bank authority.
         * @param error Optional exact submission failure.
         * @return True when submitted work has a destination completion fence.
         */
        bool start(
            const GpuExpertPackedDescriptor &source,
            const GpuExpertPackedDescriptor &destination,
            const ExpertTierSourceReadiness &source_readiness,
            std::string *error = nullptr) noexcept;

        /**
         * @brief Submit one raw FP16/BF16/FP32 projection blob copy.
         * @param source Immutable contiguous source device bytes.
         * @param destination Preallocated contiguous destination device bytes.
         * @param bytes Exact byte-identical projection size.
         * @param source_readiness Exact event or installed-bank authority.
         * @param error Optional exact submission failure.
         * @return True when submitted work has a destination completion fence.
         */
        bool startContiguous(
            const void *source,
            void *destination,
            std::size_t bytes,
            const ExpertTierSourceReadiness &source_readiness,
            std::string *error = nullptr) noexcept;

        /** @brief Query the completion event once without blocking. */
        ExpertTierGpuPeerTransferProgress poll(
            std::string *error = nullptr) noexcept;

        /** @return Current lifecycle without querying the device. */
        [[nodiscard]] ExpertTierGpuPeerTransferProgress progress() const noexcept
        {
            return progress_;
        }

        /** @return Whether stream/context/event ownership is complete. */
        [[nodiscard]] bool materialized() const noexcept;

        /** @return Whether no runtime work can still reference this lane. */
        [[nodiscard]] bool quiescent() const noexcept
        {
            return !work_may_be_in_flight_;
        }

        /** @return Cumulative non-blocking movement evidence. */
        [[nodiscard]] ExpertTierGpuPeerTransferLaneStats stats() const noexcept
        {
            return stats_;
        }

        /** @return Destination-owned completion event after submission. */
        [[nodiscard]] void *destinationReadyEvent() const noexcept
        {
            return completion_event_;
        }

        /** @return Exact directed source endpoint. */
        [[nodiscard]] DeviceId sourceDevice() const noexcept
        {
            return config_.source_device;
        }

        /** @return Exact directed destination endpoint. */
        [[nodiscard]] DeviceId destinationDevice() const noexcept
        {
            return config_.destination_device;
        }

    private:
        /** Record one stable terminal failure and its PerfStats evidence. */
        ExpertTierGpuPeerTransferProgress fail(
            const std::string &message,
            std::string *error) noexcept;

        /** @brief Submit up to four byte regions through one ordered event edge. */
        bool startRegions(
            const GPUExpertPointers &source,
            const GPUExpertPointers &destination,
            std::size_t payload_bytes,
            std::size_t scales_bytes,
            std::size_t mins_bytes,
            std::size_t emins_bytes,
            const ExpertTierSourceReadiness &source_readiness,
            std::string *error) noexcept;

        /** Release resources through the destination runtime without waiting. */
        void releaseQuiescentResources() noexcept;

        Config config_;
        IBackend *backend_ = nullptr;
        IWorkerGPUContext *source_context_ = nullptr;
        IWorkerGPUContext *destination_context_ = nullptr;
        int destination_ordinal_ = -1;
        void *transfer_stream_ = nullptr;
        void *completion_event_ = nullptr;
        void *timing_start_event_ = nullptr;
        void *timing_stop_event_ = nullptr;
        ExpertTierGpuPeerTransferProgress progress_ =
            ExpertTierGpuPeerTransferProgress::Idle;
        bool work_may_be_in_flight_ = false;
        bool fail_after_event_ = false;
        std::size_t submitted_bytes_ = 0;
        std::string failure_;
        std::chrono::steady_clock::time_point transfer_started_at_{};
        ExpertTierGpuPeerTransferLaneStats stats_;
    };
} // namespace llaminar2
