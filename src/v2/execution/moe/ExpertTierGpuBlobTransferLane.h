/**
 * @file ExpertTierGpuBlobTransferLane.h
 * @brief Persistent packed-expert relay through retained GPU progress epochs.
 *
 * CUDA and ROCm consume the same separated NativeVNNI expert layout, so a
 * GPU edge must preserve the packed bytes instead of decoding and repacking
 * them. Cross-runtime edges cannot portably share events or allocations, while
 * a same-runtime edge without driver-reported peer access must not rely on a
 * runtime's implicit `MemcpyPeerAsync` fallback. Both cases use this bounded,
 * double-buffered host relay. One retained source epoch writes into
 * source-mapped pages, a maintenance worker copies completed bytes into
 * destination-mapped pages, and one retained destination epoch copies them
 * into the inactive residency bank. Device executors submit those finite
 * epochs ahead of inference, so neither direction launches late behind an
 * already-resident inference graph.
 */

#pragma once

#include "ExpertTierSourceReadiness.h"
#include "ExpertTierTransferMeasurement.h"
#include "GPUExpertTransfer.h"
#include "../../backends/DeviceId.h"
#include "../../transfer/MappedTransferProgressEpoch.h"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace llaminar2
{
    class MappedHostTransferRegion;

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

    /** @brief Why a byte-compatible GPU edge uses the explicit host relay. */
    enum class ExpertTierGpuBlobRelayKind : std::uint8_t
    {
        CrossBackend, ///< CUDA/ROCm runtimes have no shared native peer domain.
        SameBackendWithoutPeerAccess, ///< Driver denied the exact directed edge.
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
        /** Source chunks submitted through latency-critical mapped writes. */
        std::uint64_t source_progress_kernel_submissions = 0;
        std::uint64_t destination_h2d_submissions = 0;
        /** Destination chunks submitted through retained mapped reads. */
        std::uint64_t destination_progress_kernel_submissions = 0;
        std::uint64_t host_relay_copies = 0;
        std::uint64_t host_relay_bytes = 0;
        std::uint64_t pending_event_polls = 0;
        std::uint64_t failed_transfers = 0;
        std::uint64_t maximum_in_flight_chunks = 0;
        std::uint64_t inference_stream_waits = 0;
        std::uint64_t blocking_synchronizations = 0;
        /** Most recent exact completed-transfer timing evidence. */
        ExpertTierProjectionTransferMeasurement last_measurement;
        /** Longest submit-to-observation interval for a source D2H chunk. */
        std::uint64_t last_max_source_dma_residence_nanoseconds = 0;
        /** Longest submit-to-observation interval for a destination H2D chunk. */
        std::uint64_t last_max_destination_dma_residence_nanoseconds = 0;
        /** Longest interval between maintenance polls during the last transfer. */
        std::uint64_t last_max_maintenance_poll_gap_nanoseconds = 0;
        /** Timing API failures; non-zero invalidates economy certification. */
        std::uint64_t timing_measurement_failures = 0;
    };

    /**
     * @brief Double-buffered host relay for a byte-compatible GPU edge.
     *
     * The source and destination descriptors must be byte-compatible and must
     * match the explicitly configured relay reason. Source and destination
     * progress epochs are shared by every physical relay lane on each endpoint
     * GPU. Each lane permanently leases two source and two destination command
     * slots. The caller must supply published-residency-bank readiness. A
     * producer event is rejected because a shared graph already in flight
     * could claim a newly published command before a dynamically inserted wait.
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
            /** Typed topology fact authorizing host relay instead of peer DMA. */
            ExpertTierGpuBlobRelayKind relay_kind =
                ExpertTierGpuBlobRelayKind::CrossBackend;
            std::size_t staging_capacity_bytes = 0;
            /** Source-device epoch shared across every physical relay lane. */
            std::shared_ptr<MappedTransferProgressEpoch>
                source_progress_epoch;
            /** Destination-device epoch shared across every physical relay lane. */
            std::shared_ptr<MappedTransferProgressEpoch>
                destination_progress_epoch;
            std::string lane_name;
            std::string perf_device;
            /** Collect per-runtime timing-event evidence for certification. */
            bool collect_timing_measurements = false;
        };

        /**
         * @brief Store and validate heterogeneous lane topology.
         * @param config Exact endpoints and persistent capacity.
         * @throws std::invalid_argument For non-GPU, contradictory topology,
         *         zero-capacity, or unnamed lanes.
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
         * @brief Lease epoch slots and allocate both mapped staging directions.
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
         * @brief Return whether neither retained epoch can touch slot storage.
         * @return True only after every published command was observed terminal.
         *
         * This query lets an aborted composite wave retain the lane until it is
         * safe to recycle. It never queries an event or synchronizes a stream.
         */
        [[nodiscard]] bool quiescent() const noexcept
        {
            return !hasInFlightWork();
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

    private:
        /** @brief State of one reusable double-buffer slot. */
        enum class SlotPhase : std::uint8_t
        {
            Idle,
            SourceProgressPending,
            DestinationProgressPending,
        };

        /** @brief Runtime-owned resources and current chunk for one slot. */
        struct Slot
        {
            /** Source-registered mapped pages written by asynchronous D2H DMA. */
            std::shared_ptr<MappedHostTransferRegion> source_mapped;
            /** Destination-registered mapped pages read by asynchronous H2D DMA. */
            std::shared_ptr<MappedHostTransferRegion> destination_mapped;
            /** Permanent source command identity from the shared source epoch. */
            MappedTransferProgressSlot source_progress;
            /** Permanent destination command identity from its shared epoch. */
            MappedTransferProgressSlot destination_progress;
            ExpertTierGpuBlobChunk chunk;
            /** Host submission time used to expose queue plus poll residence. */
            std::chrono::steady_clock::time_point source_submitted_at{};
            /** Host submission time used to expose queue plus poll residence. */
            std::chrono::steady_clock::time_point destination_submitted_at{};
            SlotPhase phase = SlotPhase::Idle;
        };

        /** @brief Initialize common event-polled state after binding byte views. */
        bool beginTransfer(
            const ExpertTierSourceReadiness &source_readiness,
            std::string *error) noexcept;

        /** @brief Prove every device-side projection address before publication. */
        bool validateBoundDeviceStorage(std::string *error) noexcept;

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

        /** @brief Publish one device-to-mapped command to the source epoch. */
        bool enqueueSourceChunk(
            Slot &slot,
            const ExpertTierGpuBlobChunk &chunk,
            std::string *error) noexcept;

        /** @brief Relay one completed chunk and publish mapped-to-device work. */
        bool relayAndEnqueueDestination(
            Slot &slot,
            std::string *error) noexcept;

        /** @brief Publish PerfStats evidence for a complete transfer. */
        void recordCompletion() noexcept;

        /** @brief Return true when either runtime may still touch slot storage. */
        [[nodiscard]] bool hasInFlightWork() const noexcept;

        /** @brief Free only quiescent resources through their owning runtime. */
        void releaseQuiescentResources() noexcept;

        Config config_;
        ExpertTierGpuBlobChunkProtocol protocol_;
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
        bool failure_requested_ = false;
        bool failure_counted_ = false;
        std::string failure_;
        std::chrono::steady_clock::time_point transfer_started_at_{};
        std::chrono::steady_clock::time_point last_poll_at_{};
        std::uint64_t transfer_device_nanoseconds_ = 0;
        std::uint64_t transfer_host_nanoseconds_ = 0;
        std::uint64_t max_source_dma_residence_nanoseconds_ = 0;
        std::uint64_t max_destination_dma_residence_nanoseconds_ = 0;
        std::uint64_t max_maintenance_poll_gap_nanoseconds_ = 0;
        ExpertTierGpuBlobTransferLaneStats stats_;
    };
} // namespace llaminar2
