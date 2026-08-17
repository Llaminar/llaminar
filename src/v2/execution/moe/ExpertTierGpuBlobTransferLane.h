/**
 * @file ExpertTierGpuBlobTransferLane.h
 * @brief Persistent, event-polled packed expert transfer between CUDA and ROCm.
 *
 * CUDA and ROCm consume the same separated NativeVNNI expert layout, so a
 * heterogeneous GPU edge must preserve the packed bytes instead of decoding
 * and repacking them. The two runtimes cannot portably wait on each other's
 * events or copy directly between their allocations. This lane therefore uses
 * a bounded, double-buffered host relay: the source runtime performs D2H into
 * source-owned pinned memory, a maintenance worker copies completed bytes into
 * destination-owned pinned memory, and the destination runtime performs H2D.
 * Every device transition is event-polled and all resources are materialized
 * before inference begins.
 */

#pragma once

#include "ExpertTierSourceReadiness.h"
#include "ExpertTierTransferMeasurement.h"
#include "GPUExpertTransfer.h"
#include "../../backends/DeviceId.h"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace llaminar2
{
    class IBackend;
    class IWorkerGPUContext;

    /** @brief Named separated-array region carried by one packed blob chunk. */
    enum class ExpertTierGpuBlobRegion : std::uint8_t
    {
        Payload = 0, ///< Quantized NativeVNNI payload bytes.
        Scales = 1,  ///< FP16 per-block scale bytes.
        Mins = 2,    ///< Optional FP16 per-block minimum bytes.
        Emins = 3,   ///< Optional effective-minimum metadata bytes.
    };

    /**
     * @brief One contiguous chunk that never crosses a separated-array boundary.
     *
     * Keeping a chunk inside one region lets both DMA submissions use an exact
     * pointer and byte range without constructing a temporary packed envelope.
     */
    struct ExpertTierGpuBlobChunk
    {
        ExpertTierGpuBlobRegion region = ExpertTierGpuBlobRegion::Payload;
        std::size_t region_offset = 0;
        std::size_t bytes = 0;
        std::uint64_t sequence = 0;

        /** @brief Return whether the chunk names a non-empty bounded operation. */
        [[nodiscard]] constexpr bool valid() const noexcept
        {
            return bytes != 0;
        }
    };

    /**
     * @brief Allocation-free chunk cursor for a separated packed expert blob.
     *
     * The cursor is device-free and is shared by unit tests and the live lane.
     * It emits payload, scales, optional minima, then optional effective minima
     * in a deterministic order. A chunk never straddles two arrays, which makes
     * retry, byte accounting, and destination pointer validation unambiguous.
     */
    class ExpertTierGpuBlobChunkProtocol final
    {
    public:
        /**
         * @brief Construct a reusable cursor with one fixed staging capacity.
         * @param chunk_capacity_bytes Maximum bytes in any emitted chunk.
         * @throws std::invalid_argument When the capacity is zero.
         */
        explicit ExpertTierGpuBlobChunkProtocol(
            std::size_t chunk_capacity_bytes);

        /**
         * @brief Bind compatible source and destination descriptors.
         * @param source Immutable source descriptor.
         * @param destination Writable destination descriptor expressed through
         *        the common descriptor type.
         * @param error Optional exact validation failure.
         * @return Whether the cursor was reset to the first non-empty region.
         */
        bool begin(
            const GpuExpertPackedDescriptor &source,
            const GpuExpertPackedDescriptor &destination,
            std::string *error = nullptr) noexcept;

        /**
         * @brief Reset the cursor for one contiguous floating projection.
         * @param bytes Exact source and destination byte size.
         * @param error Optional exact validation failure.
         * @return Whether region zero now covers the complete projection.
         */
        bool beginContiguous(
            std::size_t bytes,
            std::string *error = nullptr) noexcept;

        /**
         * @brief Take the next deterministic chunk and advance submission state.
         * @return The next chunk, or `std::nullopt` after every byte was issued.
         */
        [[nodiscard]] std::optional<ExpertTierGpuBlobChunk> takeNext() noexcept;

        /** @brief Return whether every descriptor byte has been issued. */
        [[nodiscard]] bool exhausted() const noexcept;

        /** @brief Return the sum of all descriptor region sizes. */
        [[nodiscard]] std::size_t totalBytes() const noexcept
        {
            return total_bytes_;
        }

        /** @brief Return bytes already assigned to chunks. */
        [[nodiscard]] std::size_t submittedBytes() const noexcept
        {
            return submitted_bytes_;
        }

        /** @brief Return the immutable maximum chunk size. */
        [[nodiscard]] std::size_t chunkCapacityBytes() const noexcept
        {
            return chunk_capacity_bytes_;
        }

    private:
        /** @brief Advance past empty or completely emitted separated regions. */
        void seekNextNonEmptyRegion() noexcept;

        std::size_t chunk_capacity_bytes_ = 0;
        std::array<std::size_t, 4> region_bytes_{};
        std::size_t region_index_ = 0;
        std::size_t region_offset_ = 0;
        std::size_t total_bytes_ = 0;
        std::size_t submitted_bytes_ = 0;
        std::uint64_t next_sequence_ = 0;
        bool begun_ = false;
    };

    /** @brief Host-visible lifecycle of one heterogeneous GPU blob transfer. */
    enum class ExpertTierGpuBlobTransferProgress : std::uint8_t
    {
        Idle,    ///< No transfer has been submitted.
        Pending, ///< At least one DMA or later chunk remains outstanding.
        Ready,   ///< Every destination byte is complete and safe to publish.
        Failed,  ///< Submission or event observation failed fatally.
    };

    /** @brief Cumulative proof counters for one persistent blob-transfer lane. */
    struct ExpertTierGpuBlobTransferLaneStats
    {
        std::uint64_t transfers_started = 0;
        std::uint64_t transfers_completed = 0;
        std::uint64_t chunks_submitted = 0;
        std::uint64_t chunks_completed = 0;
        std::uint64_t bytes_submitted = 0;
        std::uint64_t source_d2h_submissions = 0;
        std::uint64_t destination_h2d_submissions = 0;
        std::uint64_t host_relay_copies = 0;
        std::uint64_t host_relay_bytes = 0;
        std::uint64_t pending_event_polls = 0;
        std::uint64_t failed_transfers = 0;
        std::uint64_t maximum_in_flight_chunks = 0;
        std::uint64_t inference_stream_waits = 0;
        std::uint64_t blocking_synchronizations = 0;
        /** Most recent exact completed-transfer timing evidence. */
        ExpertTierProjectionTransferMeasurement last_measurement;
        /** Timing API failures; non-zero invalidates economy certification. */
        std::uint64_t timing_measurement_failures = 0;
    };

    /**
     * @brief Double-buffered CUDA-to-ROCm or ROCm-to-CUDA packed blob lane.
     *
     * The source and destination descriptors must be byte-compatible and must
     * refer to different GPU backend types. Source and destination streams are
     * context-owned auxiliary streams. The caller supplies the exact event that
     * made the immutable source bank ready; the source transfer stream waits on
     * it, but no inference stream ever waits on migration.
     *
     * `poll()` is intended to run on the background residency-maintenance
     * worker. It performs only non-blocking event queries plus at most two
     * bounded host-to-host relay copies per call.
     */
    class ExpertTierGpuBlobTransferLane final
    {
    public:
        /** @brief Immutable endpoints, staging capacity, and evidence identity. */
        struct Config
        {
            DeviceId source_device;
            DeviceId destination_device;
            std::size_t staging_capacity_bytes = 0;
            std::string lane_name;
            std::string perf_device;
            /** Collect per-runtime timing-event evidence for certification. */
            bool collect_timing_measurements = false;
        };

        /**
         * @brief Store and validate heterogeneous lane topology.
         * @param config Exact endpoints and persistent capacity.
         * @throws std::invalid_argument For non-GPU, same-backend, zero-capacity,
         *         or unnamed lanes.
         */
        explicit ExpertTierGpuBlobTransferLane(Config config);

        /**
         * @brief Release persistent resources only after all work is quiescent.
         *
         * Destruction never synchronizes. Destroying a lane with unresolved DMA
         * terminates because silently waiting here would hide a scheduler bug.
         */
        ~ExpertTierGpuBlobTransferLane();

        ExpertTierGpuBlobTransferLane(
            const ExpertTierGpuBlobTransferLane &) = delete;
        ExpertTierGpuBlobTransferLane &operator=(
            const ExpertTierGpuBlobTransferLane &) = delete;

        /**
         * @brief Allocate both streams' events and both runtimes' pinned slots.
         * @param error Optional exact construction failure.
         * @return Whether the complete persistent resource set exists.
         *
         * This method belongs to topology construction, before inference or
         * graph capture. Repeated calls allocate nothing.
         */
        bool materialize(std::string *error = nullptr) noexcept;

        /**
         * @brief Start a byte-preserving heterogeneous packed expert transfer.
         * @param source Immutable source device regions.
         * @param destination Preallocated inactive destination device regions.
         * @param source_readiness Exact producer event or installed-bank proof.
         * @param error Optional exact start failure.
         * @return Whether initial source DMA chunks were enqueued.
         */
        bool start(
            const GpuExpertPackedDescriptor &source,
            const GpuExpertPackedDescriptor &destination,
            const ExpertTierSourceReadiness &source_readiness,
            std::string *error = nullptr) noexcept;

        /**
         * @brief Start a byte-preserving heterogeneous floating projection relay.
         * @param source Immutable contiguous source device bytes.
         * @param destination Preallocated contiguous destination device bytes.
         * @param bytes Exact byte-identical projection size.
         * @param source_readiness Exact producer event or installed-bank proof.
         * @param error Optional exact start failure.
         * @return Whether initial source DMA chunks were enqueued.
         */
        bool startContiguous(
            const void *source,
            void *destination,
            std::size_t bytes,
            const ExpertTierSourceReadiness &source_readiness,
            std::string *error = nullptr) noexcept;

        /**
         * @brief Advance ready slots without waiting for either GPU.
         * @param error Optional exact query or enqueue failure.
         * @return Current lifecycle after one bounded maintenance pass.
         */
        ExpertTierGpuBlobTransferProgress poll(
            std::string *error = nullptr) noexcept;

        /** @brief Return current lifecycle without querying either backend. */
        [[nodiscard]] ExpertTierGpuBlobTransferProgress progress() const noexcept
        {
            return progress_;
        }

        /** @brief Return whether every persistent resource is available. */
        [[nodiscard]] bool materialized() const noexcept;

        /**
         * @brief Return whether neither GPU runtime can still touch slot storage.
         * @return True only after every recorded slot event is ready and no work
         *         lost its completion fence.
         *
         * This query lets an aborted composite wave retain the lane until it is
         * safe to recycle. It never queries an event or synchronizes a stream.
         */
        [[nodiscard]] bool quiescent() const noexcept
        {
            return !hasInFlightWork() && !unfenced_work_;
        }

        /** @brief Return cumulative proof counters for integration assertions. */
        [[nodiscard]] ExpertTierGpuBlobTransferLaneStats stats() const noexcept
        {
            return stats_;
        }

        /** @brief Return the source endpoint owning D2H submissions. */
        [[nodiscard]] DeviceId sourceDevice() const noexcept
        {
            return config_.source_device;
        }

        /** @brief Return the destination endpoint owning H2D submissions. */
        [[nodiscard]] DeviceId destinationDevice() const noexcept
        {
            return config_.destination_device;
        }

        /**
         * @brief Return the last destination event after successful completion.
         * @return Destination-owned event, or nullptr before any H2D submission.
         *
         * The lane retains event ownership. At `Ready` the event has already
         * been observed complete and can be used as publication provenance.
         */
        [[nodiscard]] void *destinationReadyEvent() const noexcept
        {
            return last_destination_event_;
        }

    private:
        /** @brief State of one reusable double-buffer slot. */
        enum class SlotPhase : std::uint8_t
        {
            Idle,
            SourceDmaPending,
            DestinationDmaPending,
        };

        /** @brief Runtime-owned resources and current chunk for one slot. */
        struct Slot
        {
            std::uint8_t *source_pinned = nullptr;
            std::uint8_t *destination_pinned = nullptr;
            void *source_event = nullptr;
            void *destination_event = nullptr;
            void *source_timing_start_event = nullptr;
            void *source_timing_stop_event = nullptr;
            void *destination_timing_start_event = nullptr;
            void *destination_timing_stop_event = nullptr;
            ExpertTierGpuBlobChunk chunk;
            SlotPhase phase = SlotPhase::Idle;
            bool source_timing_valid = false;
            bool destination_timing_valid = false;
        };

        /** @brief Initialize common event-polled state after binding byte views. */
        bool beginTransfer(
            const ExpertTierSourceReadiness &source_readiness,
            std::string *error) noexcept;

        /** @brief Set a stable terminal failure once and publish its counter. */
        void requestFailure(
            const std::string &message,
            std::string *error) noexcept;

        /** @brief Return the exact source pointer for one chunk. */
        [[nodiscard]] const std::uint8_t *sourceChunkPointer(
            const ExpertTierGpuBlobChunk &chunk) const noexcept;

        /** @brief Return the exact destination pointer for one chunk. */
        [[nodiscard]] std::uint8_t *destinationChunkPointer(
            const ExpertTierGpuBlobChunk &chunk) const noexcept;

        /** @brief Fill every idle slot while unissued chunks remain. */
        bool enqueueAvailableSourceChunks(std::string *error) noexcept;

        /** @brief Submit one D2H chunk and record its source-owned event. */
        bool enqueueSourceChunk(
            Slot &slot,
            const ExpertTierGpuBlobChunk &chunk,
            std::string *error) noexcept;

        /** @brief Relay one completed source chunk and submit destination H2D. */
        bool relayAndEnqueueDestination(
            Slot &slot,
            std::string *error) noexcept;

        /** @brief Publish PerfStats evidence for a complete transfer. */
        void recordCompletion() noexcept;

        /** @brief Accumulate one completed source-DMA timing interval. */
        bool collectSourceTiming(
            const Slot &slot,
            std::string *error) noexcept;

        /** @brief Accumulate one completed destination-DMA timing interval. */
        bool collectDestinationTiming(
            const Slot &slot,
            std::string *error) noexcept;

        /** @brief Return true when either runtime may still touch slot storage. */
        [[nodiscard]] bool hasInFlightWork() const noexcept;

        /** @brief Free only quiescent resources through their owning runtime. */
        void releaseQuiescentResources() noexcept;

        Config config_;
        ExpertTierGpuBlobChunkProtocol protocol_;
        IBackend *source_backend_ = nullptr;
        IBackend *destination_backend_ = nullptr;
        IWorkerGPUContext *source_context_ = nullptr;
        IWorkerGPUContext *destination_context_ = nullptr;
        int source_ordinal_ = -1;
        int destination_ordinal_ = -1;
        void *source_stream_ = nullptr;
        void *destination_stream_ = nullptr;
        std::array<Slot, 2> slots_{};
        GpuExpertPackedDescriptor source_;
        GpuExpertPackedDescriptor destination_;
        const std::uint8_t *contiguous_source_ = nullptr;
        std::uint8_t *contiguous_destination_ = nullptr;
        bool carries_contiguous_projection_ = false;
        ExpertTierGpuBlobTransferProgress progress_ =
            ExpertTierGpuBlobTransferProgress::Idle;
        std::size_t active_slots_ = 0;
        std::size_t completed_bytes_ = 0;
        void *last_destination_event_ = nullptr;
        bool failure_requested_ = false;
        bool failure_counted_ = false;
        bool unfenced_work_ = false;
        std::string failure_;
        std::chrono::steady_clock::time_point transfer_started_at_{};
        std::uint64_t transfer_device_nanoseconds_ = 0;
        std::uint64_t transfer_host_nanoseconds_ = 0;
        ExpertTierGpuBlobTransferLaneStats stats_;
    };
} // namespace llaminar2
