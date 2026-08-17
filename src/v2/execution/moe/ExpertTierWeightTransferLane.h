/**
 * @file ExpertTierWeightTransferLane.h
 * @brief Persistent event-polled GPU/CPU ExpertOverlay transfer lane.
 *
 * Cross-tier movement must continue while inference executes against an older
 * immutable residency epoch. This class owns the staging resources for one GPU
 * endpoint and advances conversion plus DMA in bounded chunks on a named
 * auxiliary stream. The maintenance worker observes progress with event
 * queries; no inference stream ever waits for this lane.
 */

#pragma once

#include "ExpertTierSourceReadiness.h"
#include "ExpertTierTransferMeasurement.h"
#include "ExpertTierWeightDeviceLayout.h"
#include "../../backends/DeviceId.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

namespace llaminar2
{
    class IBackend;
    class IWorkerGPUContext;

    /** @brief Host-visible lifecycle of one background tier transfer. */
    enum class ExpertTierWeightTransferProgress
    {
        Idle,    ///< No transfer has been submitted on this lane.
        Pending, ///< A conversion/DMA chunk or a later chunk remains outstanding.
        Ready,   ///< Every destination byte is complete and safe to publish.
        Failed,  ///< Submission or event observation failed fatally.
    };

    /** @brief Allocation-free counters retained across uses of one lane. */
    struct ExpertTierWeightTransferLaneStats
    {
        std::uint64_t transfers_started = 0;
        std::uint64_t transfers_completed = 0;
        std::uint64_t chunks_submitted = 0;
        std::uint64_t bytes_submitted = 0;
        std::uint64_t pending_event_polls = 0;
        std::uint64_t failed_transfers = 0;
        std::uint64_t inference_stream_waits = 0;
        std::uint64_t blocking_synchronizations = 0;
        /** Most recent exact completed-transfer timing evidence. */
        ExpertTierProjectionTransferMeasurement last_measurement;
        /** Timing API failures; non-zero makes economy certification invalid. */
        std::uint64_t timing_measurement_failures = 0;
    };

    /**
     * @brief One pre-materialized conversion and DMA lane for a GPU endpoint.
     *
     * The lane supports both directions of a CPU edge:
     *
     * - GPU to CPU: the source GPU converts directly into final CPU-format
     *   bytes in device staging storage, then DMA writes a pinned host chunk;
     * - CPU to GPU: pinned CPU-format bytes DMA into device staging storage,
     *   then the destination GPU converts them into its inactive packed arrays.
     *
     * Only one transfer may occupy a lane at a time. A scheduler obtains true
     * parallelism by materializing several named lanes before inference starts.
     */
    class ExpertTierWeightTransferLane final
    {
    public:
        /** @brief Immutable topology and capacity of one persistent lane. */
        struct Config
        {
            DeviceId device;
            std::size_t staging_capacity_bytes = 0;
            std::string lane_name;
            std::string perf_device;
            /** Collect timing-event evidence for economy certification. */
            bool collect_timing_measurements = false;
        };

        /** @brief Store lane identity without allocating or submitting work. */
        explicit ExpertTierWeightTransferLane(Config config);

        /**
         * @brief Release persistent resources after the lane is quiescent.
         *
         * Destruction never synchronizes. Destroying a pending lane is a fatal
         * lifecycle error because silently blocking here could pause inference.
         */
        ~ExpertTierWeightTransferLane();

        ExpertTierWeightTransferLane(
            const ExpertTierWeightTransferLane &) = delete;
        ExpertTierWeightTransferLane &operator=(
            const ExpertTierWeightTransferLane &) = delete;

        /**
         * @brief Allocate the stream, event, device chunk, and pinned chunk.
         * @param error Optional exact construction failure.
         * @return Whether every persistent resource now exists.
         *
         * Call during topology construction, never during inference or graph
         * replay. Repeated calls validate the original binding and allocate
         * nothing.
         */
        bool materialize(std::string *error = nullptr) noexcept;

        /**
         * @brief Start GPU-side repack followed by asynchronous D2H streaming.
         * @param layout Valid GPU-to-CPU stream layout.
         * @param source Exact immutable separated GPU source arrays.
         * @param final_cpu_bytes Already allocated final CPU execution storage.
         * @param source_readiness Exact producer event or installed-bank proof.
         * @param error Optional exact start failure.
         * @return Whether the first chunk and completion event were enqueued.
         */
        bool startGpuToCpu(
            const ExpertTierWeightDeviceLayout &layout,
            const ExpertTierGpuConstProjectionView &source,
            std::span<std::uint8_t> final_cpu_bytes,
            const ExpertTierSourceReadiness &source_readiness,
            std::string *error = nullptr) noexcept;

        /**
         * @brief Start asynchronous H2D streaming followed by GPU-side repack.
         * @param layout Valid CPU-to-GPU stream layout.
         * @param cpu_bytes Immutable final CPU execution storage.
         * @param destination Preallocated inactive separated GPU arrays.
         * @param error Optional exact start failure.
         * @return Whether the first chunk and completion event were enqueued.
         */
        bool startCpuToGpu(
            const ExpertTierWeightDeviceLayout &layout,
            std::span<const std::uint8_t> cpu_bytes,
            const ExpertTierGpuMutableProjectionView &destination,
            std::string *error = nullptr) noexcept;

        /**
         * @brief Advance the state machine using a non-blocking event query.
         * @param error Optional exact query or enqueue failure.
         * @return Current transfer progress after at most one completed chunk.
         *
         * A `Pending` result is normal. The caller should perform other
         * maintenance work and poll again; it must never substitute a stream or
         * event synchronization.
         */
        ExpertTierWeightTransferProgress poll(
            std::string *error = nullptr) noexcept;

        /** @brief Return the current transfer lifecycle without querying a device. */
        [[nodiscard]] ExpertTierWeightTransferProgress progress() const noexcept
        {
            return progress_;
        }

        /** @brief Return whether persistent resources match the configured lane. */
        [[nodiscard]] bool materialized() const noexcept;

        /**
         * @brief Return whether no runtime operation can still touch lane storage.
         * @return True when destruction or pool reuse is safe without waiting.
         *
         * This is a lifecycle query, not a success result. A failed transfer can
         * be quiescent after its completion fence is observed; conversely, a
         * failure whose fence could not be established remains non-quiescent and
         * must fail fatally rather than guess that reclamation is safe.
         */
        [[nodiscard]] bool quiescent() const noexcept
        {
            return !work_may_be_in_flight_;
        }

        /** @brief Return cumulative proof counters for integration assertions. */
        [[nodiscard]] ExpertTierWeightTransferLaneStats stats() const noexcept
        {
            return stats_;
        }

        /** @brief Return the GPU endpoint that owns conversion and DMA. */
        [[nodiscard]] DeviceId device() const noexcept { return config_.device; }

    private:
        enum class Direction
        {
            None,
            GpuToCpu,
            CpuToGpu,
        };

        /** @brief Record a stable failure diagnostic and exported counter. */
        ExpertTierWeightTransferProgress fail(
            const std::string &message,
            std::string *error) noexcept;

        /** @brief Enqueue the next bounded chunk and its reusable ready event. */
        bool enqueueNextChunk(std::string *error) noexcept;

        /** @brief Dispatch the backend-specific GPU-to-CPU conversion kernel. */
        bool launchGpuToCpuChunk(
            std::uint32_t first_unit,
            std::uint32_t unit_count,
            std::size_t bytes) noexcept;

        /** @brief Dispatch the backend-specific CPU-to-GPU conversion kernel. */
        bool launchCpuToGpuChunk(
            std::uint32_t first_unit,
            std::uint32_t unit_count,
            std::size_t bytes) noexcept;

        /** @brief Publish PerfStats evidence for one submitted chunk. */
        void recordSubmittedChunk(std::size_t bytes) noexcept;

        /** @brief Publish terminal latency and direction evidence. */
        void recordCompletion() noexcept;

        /** @brief Accumulate the completed chunk's device timing interval. */
        bool collectCompletedChunkTiming(std::string *error) noexcept;

        /** @brief Add one bounded host-copy duration without throwing. */
        void recordHostCopyDuration(
            std::chrono::steady_clock::duration duration) noexcept;

        /** @brief Free resources only after every recorded event is ready. */
        void releaseQuiescentResources() noexcept;

        Config config_;
        IBackend *backend_ = nullptr;
        IWorkerGPUContext *gpu_context_ = nullptr;
        int device_ordinal_ = -1;
        void *transfer_stream_ = nullptr;
        void *chunk_ready_event_ = nullptr;
        void *chunk_timing_start_event_ = nullptr;
        void *chunk_timing_stop_event_ = nullptr;
        std::uint8_t *device_chunk_ = nullptr;
        std::uint8_t *pinned_chunk_ = nullptr;

        Direction direction_ = Direction::None;
        ExpertTierWeightTransferProgress progress_ =
            ExpertTierWeightTransferProgress::Idle;
        ExpertTierWeightDeviceLayout layout_;
        ExpertTierGpuConstProjectionView gpu_source_;
        ExpertTierGpuMutableProjectionView gpu_destination_;
        std::span<std::uint8_t> cpu_destination_;
        std::span<const std::uint8_t> cpu_source_;
        std::uint32_t completed_units_ = 0;
        std::uint32_t in_flight_units_ = 0;
        std::size_t in_flight_bytes_ = 0;
        bool work_may_be_in_flight_ = false;
        bool fail_after_in_flight_event_ = false;
        bool in_flight_timing_valid_ = false;
        std::chrono::steady_clock::time_point transfer_started_at_{};
        std::uint64_t transfer_device_nanoseconds_ = 0;
        std::uint64_t transfer_host_nanoseconds_ = 0;
        std::uint64_t transfer_bytes_ = 0;
        ExpertTierWeightTransferLaneStats stats_;
        std::string failure_;
    };
} // namespace llaminar2
