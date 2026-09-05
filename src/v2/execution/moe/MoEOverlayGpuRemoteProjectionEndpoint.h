/**
 * @file MoEOverlayGpuRemoteProjectionEndpoint.h
 * @brief Event-polled GPU endpoints for cross-rank ExpertOverlay streaming.
 *
 * A remote projection is intentionally never materialized as a complete host
 * mirror. Source GPUs either repack one final CPU NativeVNNI unit range or copy
 * one separated NativeVNNI/contiguous floating GPU range into persistent pinned
 * storage. Destination GPUs copy each authenticated MPI chunk into GPU-owned
 * storage on a named auxiliary stream. The MPI data plane and device lane
 * advance through non-blocking polls, so inference continues on the retained
 * old residency epoch until the complete candidate bank is published.
 */

#pragma once

#include "ExpertTierSourceReadiness.h"
#include "MoEOverlayMPIRemoteProjectionTransport.h"
#include "../../transfer/TransferEngine.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>

namespace llaminar2
{
    class IBackend;
    class IWorkerGPUContext;

    /** Lifecycle of one bounded operation on a remote GPU staging lane. */
    enum class MoEOverlayGpuRemoteLaneProgress : std::uint8_t
    {
        Idle,    ///< The owner has not submitted a device chunk.
        Pending, ///< Conversion or DMA is protected by the lane event.
        Ready,   ///< The event completed and staging/final bytes are ready.
        Failed,  ///< Submission or event observation failed.
    };

    /** Cumulative proof counters for one persistent remote GPU lane. */
    struct MoEOverlayGpuRemoteProjectionLaneStats
    {
        std::uint64_t owners_acquired = 0;
        std::uint64_t chunks_submitted = 0;
        std::uint64_t chunks_completed = 0;
        std::uint64_t bytes_submitted = 0;
        std::uint64_t gpu_to_cpu_repack_chunks = 0;
        std::uint64_t cpu_to_gpu_repack_chunks = 0;
        std::uint64_t gpu_blob_read_chunks = 0;
        std::uint64_t gpu_blob_write_chunks = 0;
        std::uint64_t device_to_host_submissions = 0;
        std::uint64_t host_to_device_submissions = 0;
        std::uint64_t producer_event_waits = 0;
        std::uint64_t published_bank_sources = 0;
        std::uint64_t pending_event_polls = 0;
        std::uint64_t failures = 0;
        std::uint64_t inference_stream_waits = 0;
        std::uint64_t blocking_synchronizations = 0;
    };

    /**
     * @brief One model-time stream/event/device/pinned resource for a GPU rank.
     *
     * Endpoints acquire this lane for exactly one network chunk, then release it
     * after MPI or the destination event relinquishes staging ownership. The
     * physical fabric creates an admission-sized pool per GPU/projection/role
     * and assigns different endpoints to different lanes, so acquisition never
     * serializes independent operations within one admitted wave.
     */
    class MoEOverlayGpuRemoteProjectionLane final
    {
    public:
        /** Immutable GPU identity, staging BOM, and PerfStats labels. */
        struct Config
        {
            DeviceId device = DeviceId::invalid();
            /** Exclusive host/device staging region from one shared slab. */
            PersistentTransferStagingSlice staging;
            /** Exact participant/cycle stream shared across compatible work. */
            PersistentTransferExecutionLane execution;
            std::string lane_name;
            std::string perf_device;
        };

        /**
         * @brief Validate one exact GPU lane without allocating resources.
         * @param config Device, positive staging capacity, and stable identity.
         * @throws std::invalid_argument For an incomplete configuration.
         */
        explicit MoEOverlayGpuRemoteProjectionLane(Config config);

        /**
         * @brief Release resources only when no endpoint or event owns them.
         *
         * Destruction never synchronizes. An acquired or in-flight lane is a
         * fatal lifecycle error because silently draining it could pause live
         * inference and conceal a broken migration transaction.
         */
        ~MoEOverlayGpuRemoteProjectionLane();

        MoEOverlayGpuRemoteProjectionLane(
            const MoEOverlayGpuRemoteProjectionLane &) = delete;
        MoEOverlayGpuRemoteProjectionLane &operator=(
            const MoEOverlayGpuRemoteProjectionLane &) = delete;

        /**
         * @brief Bind the pooled stream/slab slice and create one lane event.
         * @param error Optional exact setup failure.
         * @return True when every persistent model-time resource exists.
         */
        bool materialize(std::string *error = nullptr) noexcept;

        /**
         * @brief Try to reserve the lane for one endpoint object.
         * @param owner Stable non-null endpoint address.
         * @return True when ownership was acquired. False indicates a violated
         *         admission/lifecycle invariant for a uniquely assigned lane.
         */
        bool tryAcquire(const void *owner) noexcept;

        /**
         * @brief Release one quiescent completed/aborted chunk reservation.
         * @param owner Exact current endpoint owner.
         * @param error Optional ownership or quiescence diagnostic.
         * @return True only when the lane is immediately reusable.
         */
        bool release(
            const void *owner,
            std::string *error = nullptr) noexcept;

        /**
         * @brief Join the exact producer edge before reading a source GPU bank.
         * @param owner Exact current endpoint owner.
         * @param readiness Producer event or retained published-bank proof.
         * @param error Optional ordering diagnostic.
         * @return True when the source edge is bound to this acquisition.
         */
        bool bindSourceReadiness(
            const void *owner,
            const ExpertTierSourceReadiness &readiness,
            std::string *error = nullptr) noexcept;

        /**
         * @brief Repack one source-GPU range and DMA final CPU bytes to pinned RAM.
         * @param owner Exact current endpoint owner.
         * @param layout Valid GPU-to-CPU conversion layout.
         * @param source Live separated source arrays.
         * @param first_unit First CPU NativeVNNI unit in this network chunk.
         * @param unit_count Number of complete units in this chunk.
         * @param error Optional launch/DMA diagnostic.
         * @return True when a reusable event fences all submitted work.
         */
        bool submitGpuToCpuRepack(
            const void *owner,
            const ExpertTierWeightDeviceLayout &layout,
            const ExpertTierGpuConstProjectionView &source,
            std::uint32_t first_unit,
            std::uint32_t unit_count,
            std::string *error = nullptr) noexcept;

        /**
         * @brief DMA one separated source-GPU blob range to pinned network RAM.
         * @param owner Exact current endpoint owner.
         * @param source Exact device range start.
         * @param bytes Bounded non-zero byte count.
         * @param error Optional DMA diagnostic.
         * @return True when a reusable event fences the D2H operation.
         */
        bool submitGpuBlobRead(
            const void *owner,
            const std::uint8_t *source,
            std::size_t bytes,
            std::string *error = nullptr) noexcept;

        /**
         * @brief DMA arriving CPU bytes and repack them into a destination GPU slot.
         * @param owner Exact current endpoint owner.
         * @param layout Valid CPU-to-GPU conversion layout.
         * @param destination Exact inactive separated destination arrays.
         * @param first_unit First final CPU unit represented by @p payload.
         * @param unit_count Complete unit count represented by @p payload.
         * @param payload Authenticated MPI receive bytes.
         * @param error Optional copy/repack diagnostic.
         * @return True when a reusable event fences H2D and conversion.
         */
        bool submitCpuToGpuRepack(
            const void *owner,
            const ExpertTierWeightDeviceLayout &layout,
            const ExpertTierGpuMutableProjectionView &destination,
            std::uint32_t first_unit,
            std::uint32_t unit_count,
            std::span<const std::uint8_t> payload,
            std::string *error = nullptr) noexcept;

        /**
         * @brief DMA one authenticated MPI blob range into final GPU storage.
         * @param owner Exact current endpoint owner.
         * @param destination Exact final separated-array range start.
         * @param payload Authenticated bounded MPI bytes.
         * @param error Optional copy diagnostic.
         * @return True when a reusable event fences the H2D operation.
         */
        bool submitGpuBlobWrite(
            const void *owner,
            std::uint8_t *destination,
            std::span<const std::uint8_t> payload,
            std::string *error = nullptr) noexcept;

        /**
         * @brief Query the exact completion event once without blocking.
         * @param owner Exact current endpoint owner.
         * @param error Optional event or deferred-submission failure.
         * @return Idle, Pending, Ready, or Failed for the current chunk.
         */
        MoEOverlayGpuRemoteLaneProgress poll(
            const void *owner,
            std::string *error = nullptr) noexcept;

        /**
         * @brief Borrow event-ready D2H bytes while this owner retains the lane.
         * @param owner Exact current endpoint owner.
         * @param bytes Exact completed D2H byte count.
         * @return Stable pinned span, or empty for an invalid lifecycle/range.
         */
        [[nodiscard]] std::span<const std::uint8_t> pinnedOutput(
            const void *owner,
            std::size_t bytes) const noexcept;

        /** @return Whether stream, event, and both staging buffers exist. */
        [[nodiscard]] bool materialized() const noexcept;

        /** @return Exact GPU device that owns stream and device storage. */
        [[nodiscard]] DeviceId device() const noexcept { return config_.device; }

        /** @return Immutable staging capacity used by the MPI endpoint. */
        [[nodiscard]] std::size_t stagingCapacityBytes() const noexcept
        {
            return config_.staging.sizeBytes();
        }

        /** @return Race-safe cumulative non-blocking-path evidence. */
        [[nodiscard]] MoEOverlayGpuRemoteProjectionLaneStats stats()
            const noexcept;

    private:
        /** Kind of device work currently protected by @ref completion_event_. */
        enum class OperationKind : std::uint8_t
        {
            None,
            GpuToCpuRepack,
            CpuToGpuRepack,
            GpuBlobRead,
            GpuBlobWrite,
        };

        /** @brief Validate exact owner and idle chunk lifecycle under the mutex. */
        bool canSubmitLocked(
            const void *owner,
            bool source_operation,
            std::string *error) noexcept;

        /** @brief Record one fence even when an earlier submission failed. */
        bool fenceSubmissionLocked(
            bool submitted,
            OperationKind kind,
            std::size_t bytes,
            std::string *error) noexcept;

        /** @brief Launch backend-specific GPU-to-CPU repack on the lane stream. */
        bool launchGpuToCpuLocked(
            const ExpertTierWeightDeviceLayout &layout,
            const ExpertTierGpuConstProjectionView &source,
            std::uint32_t first_unit,
            std::uint32_t unit_count) noexcept;

        /** @brief Launch backend-specific CPU-to-GPU repack on the lane stream. */
        bool launchCpuToGpuLocked(
            const ExpertTierWeightDeviceLayout &layout,
            const ExpertTierGpuMutableProjectionView &destination,
            std::uint32_t first_unit,
            std::uint32_t unit_count,
            std::size_t bytes) noexcept;

        /** @brief Export one low-frequency PerfStats submission record. */
        void recordSubmissionLocked(OperationKind kind, std::size_t bytes) noexcept;

        /** @brief Return the stable PerfStats tag for one operation kind. */
        [[nodiscard]] static const char *operationName(
            OperationKind kind) noexcept;

        /** @brief Free resources after the caller has proved quiescence. */
        void releaseResources() noexcept;

        Config config_;
        IBackend *backend_ = nullptr;
        IWorkerGPUContext *context_ = nullptr;
        int device_ordinal_ = -1;
        void *stream_ = nullptr;
        void *completion_event_ = nullptr;
        std::uint8_t *device_chunk_ = nullptr;
        std::uint8_t *pinned_chunk_ = nullptr;

        mutable std::mutex mutex_;
        const void *owner_ = nullptr;
        MoEOverlayGpuRemoteLaneProgress progress_ =
            MoEOverlayGpuRemoteLaneProgress::Idle;
        OperationKind operation_kind_ = OperationKind::None;
        std::size_t operation_bytes_ = 0;
        bool source_readiness_bound_ = false;
        bool work_may_be_in_flight_ = false;
        bool fail_after_event_ = false;
        bool unfenced_work_ = false;
        std::string failure_;
        MoEOverlayGpuRemoteProjectionLaneStats stats_;
    };

    /**
     * @brief Source endpoint streaming a live GPU projection directly to MPI.
     *
     * GPU-to-CPU edges expose repacked final CPU units; GPU-to-GPU edges expose
     * the current separated physical bytes.  The returned pinned payload remains
     * owned until @ref acknowledgeChunkSent proves MPI completed its send.
     */
    class MoEOverlayGpuRemoteProjectionSource final
        : public IMoEOverlayRemoteProjectionSourceEndpoint
    {
    public:
        /**
         * @brief Bind one authenticated manifest to a live retained GPU source.
         * @param manifest GPU-to-CPU unit stream or GPU-to-GPU blob stream.
         * @param lane Pre-materialized source lane for this GPU/projection role.
         * @param source Exact live separated GPU descriptor.
         * @param readiness Producer event or retained published-bank proof.
         * @param lifetime Non-null engine/bank lifetime protecting source arrays.
         * @throws std::invalid_argument For any mismatched topology or storage.
         */
        MoEOverlayGpuRemoteProjectionSource(
            MoEOverlayRemoteProjectionManifest manifest,
            std::shared_ptr<MoEOverlayGpuRemoteProjectionLane> lane,
            GpuExpertPackedDescriptor source,
            ExpertTierSourceReadiness readiness,
            std::shared_ptr<void> lifetime);

        /**
         * @brief Bind one raw floating manifest to a retained live GPU matrix.
         * @param manifest GPU-to-GPU contiguous floating blob contract.
         * @param lane Pre-materialized source lane for this GPU/projection role.
         * @param source Exact live FP16, BF16, or FP32 descriptor.
         * @param readiness Producer event or retained published-bank proof.
         * @param lifetime Non-null engine/bank lifetime protecting source bytes.
         * @throws std::invalid_argument For any mismatched topology or storage.
         */
        MoEOverlayGpuRemoteProjectionSource(
            MoEOverlayRemoteProjectionManifest manifest,
            std::shared_ptr<MoEOverlayGpuRemoteProjectionLane> lane,
            ContiguousFloatingPointWeightDescriptor source,
            ExpertTierSourceReadiness readiness,
            std::shared_ptr<void> lifetime);

        /** @brief Enforce explicit lane release before endpoint destruction. */
        ~MoEOverlayGpuRemoteProjectionSource() override;

        /** @return Immutable authenticated format sent before device work. */
        [[nodiscard]] const MoEOverlayRemoteProjectionManifest &manifest()
            const noexcept override
        {
            return manifest_;
        }

        /** @brief Acquire/submit/query one D2H network chunk without waiting. */
        MoEOverlayResidencyWaveProgress pollNextChunk(
            MoEOverlayRemoteProjectionChunkView *chunk,
            std::string *error = nullptr) noexcept override;

        /** @brief Release pinned bytes only after MPI completes the exact send. */
        bool acknowledgeChunkSent(
            const MoEOverlayRemoteProjectionChunkHeader &header,
            std::string *error = nullptr) noexcept override;

        /** @brief Stop producing chunks and begin event-aware cleanup. */
        void abort() noexcept override;

        /** @brief Query any submitted event, then release the source lane. */
        MoEOverlayResidencyWaveProgress pollAbort(
            std::string *error = nullptr) noexcept override;

    private:
        /** @brief Submit the next CPU-unit or GPU-region range after acquisition. */
        bool submitNextChunk(std::string *error) noexcept;

        /** @brief Advance over empty separated GPU regions. */
        void seekNextGpuRegion() noexcept;

        /** @brief Return the device pointer for the current GPU blob region. */
        [[nodiscard]] const std::uint8_t *currentGpuRegionPointer()
            const noexcept;

        /** @brief Release a quiescent acquired lane exactly once. */
        bool releaseLane(std::string *error) noexcept;

        MoEOverlayRemoteProjectionManifest manifest_;
        std::shared_ptr<MoEOverlayGpuRemoteProjectionLane> lane_;
        GpuExpertPackedDescriptor source_;
        ContiguousFloatingPointWeightDescriptor floating_source_;
        ExpertTierSourceReadiness readiness_;
        std::shared_ptr<void> lifetime_;
        std::optional<ExpertTierWeightDeviceLayout> cpu_layout_;
        MoEOverlayRemoteProjectionChunkHeader active_header_;
        std::uint64_t sequence_ = 0;
        std::uint64_t emitted_bytes_ = 0;
        std::size_t region_ = 0;
        std::uint64_t region_offset_ = 0;
        bool lane_owned_ = false;
        bool chunk_submitted_ = false;
        bool payload_outstanding_ = false;
        bool complete_ = false;
        bool aborted_ = false;
    };

    /** Descriptor and executable engine created for one received GPU format. */
    struct MoEOverlayGpuRemoteProjectionDestinationBinding
    {
        GpuExpertPackedDescriptor descriptor;
        ContiguousFloatingPointWeightDescriptor floating_descriptor;
        std::shared_ptr<ITensorGemm> engine;

        /**
         * @return Whether exactly one complete writable representation and its
         *         executable engine are retained.
         */
        [[nodiscard]] bool valid() const noexcept
        {
            return descriptor.valid() != floating_descriptor.valid() &&
                   engine != nullptr;
        }
    };

    /**
     * Model-specific factory binding an authenticated manifest to an inactive
     * GPU slot and the matching backend GEMM engine alias.
     */
    using MoEOverlayGpuRemoteProjectionDestinationFactory = std::function<bool(
        const MoEOverlayRemoteProjectionManifest &,
        MoEOverlayGpuRemoteProjectionDestinationBinding *,
        std::string *)>;

    /**
     * @brief Destination endpoint committing MPI chunks directly to a GPU slot.
     *
     * The model-specific factory runs after manifest authentication, so a GPU
     * destination can accept either canonical source bytes or a normalized live
     * representation while retaining the immutable GGUF arithmetic identity.
     */
    class MoEOverlayGpuRemoteProjectionDestination final
        : public IMoEOverlayRemoteProjectionDestinationEndpoint
    {
    public:
        /**
         * @brief Retain expected topology, lane, slot factory, and slot lifetime.
         * @param expected_identity Exact transaction projection expected here.
         * @param lane Pre-materialized destination lane for this GPU/role.
         * @param factory Authenticated format-to-slot/engine binder.
         * @param lifetime Non-null inactive-slot lease retained through publish.
         * @throws std::invalid_argument For incomplete ownership or topology.
         */
        MoEOverlayGpuRemoteProjectionDestination(
            MoEOverlayRemoteProjectionIdentity expected_identity,
            std::shared_ptr<MoEOverlayGpuRemoteProjectionLane> lane,
            MoEOverlayGpuRemoteProjectionDestinationFactory factory,
            std::shared_ptr<void> lifetime);

        /** @brief Enforce explicit lane release before endpoint destruction. */
        ~MoEOverlayGpuRemoteProjectionDestination() override;

        /** @brief Authenticate format and bind exact inactive GPU storage. */
        bool beginManifest(
            const MoEOverlayRemoteProjectionManifest &manifest,
            std::string *error = nullptr) noexcept override;

        /** @brief Validate and submit, or queue, one persistent MPI chunk span. */
        MoEOverlayResidencyWaveProgress beginChunk(
            const MoEOverlayRemoteProjectionChunkHeader &header,
            std::span<const std::uint8_t> payload,
            std::string *error = nullptr) noexcept override;

        /** @brief Acquire/submit/query one destination chunk without waiting. */
        MoEOverlayResidencyWaveProgress pollChunk(
            std::string *error = nullptr) noexcept override;

        /** @return Whether the final authenticated chunk reached GPU storage. */
        [[nodiscard]] bool complete() const noexcept override;

        /** @brief Confirm final readiness; publication is owned by the wrapper. */
        bool publishFinal(std::string *error = nullptr) noexcept override;

        /** @brief Poison the validator and begin event-aware cleanup. */
        void abort() noexcept override;

        /** @brief Query submitted work, then release the destination lane. */
        MoEOverlayResidencyWaveProgress pollAbort(
            std::string *error = nullptr) noexcept override;

        /** @return Exact executable alias created from the accepted manifest. */
        [[nodiscard]] std::shared_ptr<ITensorGemm> preparedEngine()
            const noexcept
        {
            return binding_.engine;
        }

    private:
        /** @brief Acquire a lane and submit the retained MPI payload when possible. */
        MoEOverlayResidencyWaveProgress startPendingChunk(
            std::string *error) noexcept;

        /** @brief Return exact mutable destination base for one blob region. */
        [[nodiscard]] std::uint8_t *gpuRegionPointer(
            std::uint8_t region) const noexcept;

        /** @brief Release a quiescent acquired lane exactly once. */
        bool releaseLane(std::string *error) noexcept;

        MoEOverlayRemoteProjectionIdentity expected_identity_;
        std::shared_ptr<MoEOverlayGpuRemoteProjectionLane> lane_;
        MoEOverlayGpuRemoteProjectionDestinationFactory factory_;
        std::shared_ptr<void> lifetime_;
        std::optional<MoEOverlayRemoteProjectionManifest> manifest_;
        std::optional<MoEOverlayRemoteProjectionChunkValidator> validator_;
        std::optional<ExpertTierWeightDeviceLayout> cpu_layout_;
        MoEOverlayGpuRemoteProjectionDestinationBinding binding_;
        MoEOverlayRemoteProjectionChunkHeader active_header_;
        std::span<const std::uint8_t> active_payload_;
        bool lane_owned_ = false;
        bool chunk_active_ = false;
        bool chunk_submitted_ = false;
        bool final_chunk_committed_ = false;
        bool aborted_ = false;
    };
} // namespace llaminar2
