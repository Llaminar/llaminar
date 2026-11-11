/**
 * @file CapturedTransferChannelProtocol.h
 * @brief Device-friendly ordering contract for one retained node-local message lane.
 *
 * A producer and consumer exchange exact byte messages through a setup-owned
 * mapped slot. Each endpoint owns its own monotonically increasing completion
 * word. The producer may reuse the slot only after the consumer acknowledges
 * its previous message. Neither endpoint resets a live word between requests;
 * graph replay therefore cannot mistake a previous request for fresh input.
 *
 * The pure decisions below are shared by host-only adversarial tests and GPU
 * lowering. They do not enqueue work, allocate storage, inspect a device from
 * the host, or constitute a second transfer authority. TransferEngine owns the
 * mapped allocation, exact endpoint aliases and explicit stream submission.
 * The caller must acquire the peer's completion word before reading its
 * descriptor and release its own completion word only after the payload work.
 */
#pragma once

#include <cstdint>
#include <cstddef>
#include <type_traits>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_TRANSFER_CHANNEL_HD __host__ __device__
#else
#define LLAMINAR_TRANSFER_CHANNEL_HD
#endif

namespace llaminar2
{
    /** @brief Ownership direction, independent of backend, ordinal or tier. */
    enum class CapturedTransferEndpoint : std::uint32_t { Producer, Consumer };

    /**
     * @brief Authority for a transaction's live extent, frozen when capturing.
     *
     * Fixed messages keep their original exact-size contract. Counted messages
     * read one producer-owned device scalar once at acquire; the consumer takes
     * that same extent from the release-published descriptor, never a second
     * independently computed count. Both retain a positive admitted capacity.
     */
    enum class CapturedTransferExtentSource : std::uint32_t
    {
        FixedMessage,
        ProducerDevice,
        ProducerPublication,
    };

    /** @return Whether an extent authority is legal for the selected endpoint. */
    [[nodiscard]] LLAMINAR_TRANSFER_CHANNEL_HD constexpr bool capturedTransferExtentMatchesRole(
        CapturedTransferExtentSource source, CapturedTransferEndpoint role) noexcept
    {
        switch (source)
        {
        case CapturedTransferExtentSource::FixedMessage:
            return role == CapturedTransferEndpoint::Producer || role == CapturedTransferEndpoint::Consumer;
        case CapturedTransferExtentSource::ProducerDevice:
            return role == CapturedTransferEndpoint::Producer;
        case CapturedTransferExtentSource::ProducerPublication:
            return role == CapturedTransferEndpoint::Consumer;
        }
        return false;
    }

    /**
     * @brief One producer-owned live slice within an immutable retained source region.
     *
     * Offsets use the binding's immutable coordinate origin, independent of
     * the mapped slot. The graph orders this whole descriptor before acquisition;
     * acquire snapshots both fields before waiting and retains them through
     * payload publication. Empty slices may name the source region's end.
     */
    struct CapturedTransferSourceSlice
    {
        std::uint64_t byte_offset = 0;
        std::uint64_t live_bytes = 0;
    };
    static_assert(sizeof(CapturedTransferSourceSlice) == 16);

    /** @brief Local lease state; a failed endpoint is never reusable. */
    enum class CapturedTransferPhase : std::uint32_t { Idle, Acquired, Publishing, Failed, Cancelled };

    /** @brief Semantic decision, independent of a backend enqueue result. */
    enum class CapturedTransferStatus : std::uint32_t
    {
        Ready,
        WaitForPeer,
        InvalidBinding,
        InvalidPhase,
        PeerAborted,
        EpochMismatch,
        MessageMismatch,
        EpochExhausted,
        TimedOut,
        Cancelled, ///< Original parent cancelled the bound invocation; channel is terminal.
    };

    /** Reserved terminal value releases a peer into failure, never into payload work. */
    inline constexpr std::uint64_t kCapturedTransferAbortEpoch = ~std::uint64_t{0};

    /**
     * @brief Exact captured message identity, not mutable model/sampler state.
     *
     * The key identifies a setup-frozen semantic role/geometry in the graph
     * family; byte count is the live physical extent, not allocated capacity.
     * A fixed binding names exact bytes. A counted binding names the maximum;
     * its acquired/publication message holds the actual extent (including zero).
     * Both endpoints retire exactly the same key and live extent.
     */
    struct CapturedTransferMessage
    {
        std::uint64_t key = 0;
        std::uint64_t bytes = 0;

        /** @return Whether a positive capture-time extent fits its admitted slot. */
        [[nodiscard]] LLAMINAR_TRANSFER_CHANNEL_HD constexpr bool fits(
            std::uint64_t capacity) const noexcept
        { return key != 0 && bytes != 0 && bytes <= capacity; }
    };

    /** @brief Explicit physical payload route; a peer request never selects mapped-host payload. */

    /** @brief Immutable mapped-lane identity; new storage receives a new identity. */
    struct CapturedTransferChannelIdentity
    {
        std::uint64_t nonce = 0;
        std::uint64_t capacity = 0;

        /** @return Whether both graph recordings name a real allocation contract. */
        [[nodiscard]] LLAMINAR_TRANSFER_CHANNEL_HD constexpr bool valid() const noexcept
        { return nonce != 0 && capacity != 0; }
    };

    /**
     * @brief Release-published endpoint record; exactly one endpoint writes it.
     *
     * After acquiring epoch N, a peer may read the corresponding descriptor.
     * Backpressure prevents that descriptor from changing until consumption.
     * The two records occupy separate cache lines in the mapped allocation.
     */
    struct alignas(64) CapturedTransferPublication
    {
        std::uint64_t epoch = 0;
        CapturedTransferMessage message;
    };

    /** @brief Endpoint-private lease; never a host shadow of GPU live state. */
    struct CapturedTransferCursor
    {
        std::uint64_t completed_epoch = 0;
        CapturedTransferPhase phase = CapturedTransferPhase::Idle;
        std::uint32_t completed_payload_blocks = 0; ///< GPU-only completion cohort; occupies former alignment padding.
        CapturedTransferMessage acquired_message;
        std::uint64_t acquired_source_offset = 0; ///< Producer-private snapshot; never a wire extent or host observation.
    };

    /** @brief Original admission outcome after the enclosing owner has proved native endpoint retirement. */
    enum class CapturedTransferInvocationRetirement { Completed, Cancelled };

    /** @brief Original immutable request identity borrowed from its existing parent admission. */
    struct CapturedTransferInvocation
    {
        std::uint64_t generation = 0;
        std::uint64_t invocation = 0;
        /** @return Whether this names a real original admission, never an unticketed observation. */
        [[nodiscard]] constexpr bool valid() const noexcept { return generation && invocation; }
        /** @return Exact original request equality, including its generation. */
        [[nodiscard]] constexpr bool operator==(const CapturedTransferInvocation &other) const noexcept
        { return generation == other.generation && invocation == other.invocation; }
        /** @return Whether either original request coordinate differs. */
        [[nodiscard]] constexpr bool operator!=(const CapturedTransferInvocation &other) const noexcept
        { return !(*this == other); }
    };

    /** @brief Setup identity, CPU-owned terminal cancellation and independent GPU release words. */
    struct alignas(64) CapturedTransferChannelControl
    {
        CapturedTransferChannelIdentity identity;
        std::uint64_t cancelled_nonce = 0; ///< Sole CPU parent publication; endpoints never clear or write it.
        CapturedTransferPublication producer;
        CapturedTransferPublication consumer;
    };

    // Cancellation occupies the identity cache line's existing alignment padding.
    // The shared memoryFor() BOM and physical producer/consumer offsets stay exact.
    static_assert(sizeof(CapturedTransferChannelControl) == 192);
    static_assert(offsetof(CapturedTransferChannelControl, producer) == 64);

    /** @brief Checked lowering of a device-owned logical row count; no second scalar or wire authority. */
    struct CapturedTransferRowBytes
    {
        bool valid;
        std::uint64_t bytes;
    };
    /** @brief Convert exactly the original live rows, rejecting negative, overflow and capacity excess.
     * @param rows Original signed INT32 device row count, including legitimate zero.
     * @param row_bytes Positive immutable physical row stride in bytes.
     * @param capacity Maximum captured message bytes, never the current live count.
     * @return Checked uint64 byte extent for the existing producer publication. */
    [[nodiscard]] LLAMINAR_TRANSFER_CHANNEL_HD constexpr CapturedTransferRowBytes capturedTransferRowBytes(
        std::int32_t rows, std::uint64_t row_bytes, std::uint64_t capacity) noexcept
    {
        if (rows < 0 || !row_bytes || !capacity || std::uint64_t(rows) > capacity / row_bytes)
            return {false, 0};
        return {true, std::uint64_t(rows) * row_bytes};
    }

    /**
     * @brief Exact aliases and immutable geometry passed to one CUDA/HIP boundary.
     *
     * TransferEngine must validate mapped bounds, endpoint registration and
     * cursor ownership before constructing this low-level kernel descriptor.
     * The timer bound is established during setup, never queried in capture.
     */
    struct CapturedTransferChannelDeviceBinding
    {
        CapturedTransferChannelControl *control = nullptr;
        CapturedTransferCursor *cursor = nullptr;
        CapturedTransferChannelIdentity expected;
        CapturedTransferMessage message;
        CapturedTransferEndpoint role = CapturedTransferEndpoint::Producer;
        std::uint64_t timeout_ticks = 0;
        CapturedTransferExtentSource extent_source = CapturedTransferExtentSource::FixedMessage;
        std::uint64_t *device_extent = nullptr; ///< Existing byte-count input/output, absent for typed row counts.
        const std::int32_t *device_rows = nullptr; ///< Retained local live-row input; never reinterpreted as uint64.
        std::uint64_t row_bytes = 0; ///< Immutable checked bytes per row; zero on existing byte-count paths.
        const CapturedTransferSourceSlice *source_slice = nullptr; ///< Optional producer descriptor in its original retained Tensor.
        std::uint64_t source_span_bytes = 0; ///< Immutable local source region, independent of maximum/live wire bytes.
        std::uint64_t source_offset_origin = 0; ///< Coordinate origin of that source region in the original descriptor.

        /** @return Whether immutable fields can name a bounded native operation. */
        [[nodiscard]] LLAMINAR_TRANSFER_CHANNEL_HD constexpr bool valid() const noexcept
        {
            return control && cursor && expected.valid() && message.fits(expected.capacity) && timeout_ticks != 0 &&
                capturedTransferExtentMatchesRole(extent_source, role) &&
                (device_rows ? !device_extent && row_bytes != 0 && message.bytes % row_bytes == 0 &&
                    extent_source != CapturedTransferExtentSource::FixedMessage && !source_slice
                    : row_bytes == 0 && ((extent_source == CapturedTransferExtentSource::FixedMessage) == (device_extent == nullptr))) &&
                (source_slice ? role == CapturedTransferEndpoint::Producer &&
                    extent_source == CapturedTransferExtentSource::ProducerDevice &&
                    device_extent == &source_slice->live_bytes && source_span_bytes != 0 && source_span_bytes >= message.bytes &&
                    source_offset_origin <= ~std::uint64_t{0} - source_span_bytes
                    : source_span_bytes == 0 && source_offset_origin == 0);
        }
    };

    /**
     * @brief Read-only device receipt borrowed from an actual retained consumer binding.
     *
     * This view adds no progress word and never reads GPU state on the host.
     * The original binding/channel owner must outlive every captured reader.
     * Native callers acquire the original mapped publication/cancellation words
     * before evaluating the same pure receipt law used by CPU replay.
     */
    class CapturedTransferReceiveCompletion final
    {
    public:
        /** @return Explicit absence for native phases that do not consume a payload receipt. */
        [[nodiscard]] LLAMINAR_TRANSFER_CHANNEL_HD static constexpr CapturedTransferReceiveCompletion notRequired() noexcept
        { return CapturedTransferReceiveCompletion{}; }
        /** @return Whether this names original positive-capacity consumer metadata. */
        [[nodiscard]] LLAMINAR_TRANSFER_CHANNEL_HD constexpr bool valid() const noexcept
        { return control_ && cursor_ && expected_.valid() && message_.fits(expected_.capacity); }
        /** @return Original mapped cancellation address; native system-acquire only. */
        [[nodiscard]] LLAMINAR_TRANSFER_CHANNEL_HD constexpr const std::uint64_t* cancellationAddress() const noexcept
        { return control_ ? &control_->cancelled_nonce : nullptr; }
        /** @return Original consumer publication, acquired on device after that endpoint completes. */
        [[nodiscard]] LLAMINAR_TRANSFER_CHANNEL_HD constexpr const CapturedTransferPublication* publication() const noexcept
        { return control_ ? &control_->consumer : nullptr; }
        /** @return Original endpoint-local cancellation disposition; native callers inspect this before a rare system acquire. */
        [[nodiscard]] LLAMINAR_TRANSFER_CHANNEL_HD constexpr bool hasCancelledCursor() const noexcept
        { return valid() && cursor_->phase == CapturedTransferPhase::Cancelled; }
        /**
         * @brief Recognize absorbing cancellation on this exact original receive endpoint.
         * @param cancellation System-acquired publication from this original physical channel.
         * @return True only for its absorbing Cancelled cursor and exact original nonce.
         * @note This claims neither that no payload was copied nor that model/owner retirement is proven.
         */
        [[nodiscard]] LLAMINAR_TRANSFER_CHANNEL_HD constexpr bool cancelledReceive(
            std::uint64_t cancellation) const noexcept
        {
            return valid() && control_->identity.nonce == expected_.nonce &&
                control_->identity.capacity == expected_.capacity && cancellation == expected_.nonce &&
                cursor_->phase == CapturedTransferPhase::Cancelled;
        }
        /**
         * @brief Validate actual completed payload identity rather than inferring success from byte count.
         * @param bytes Actual received bytes, including a legitimate zero.
         * @param cancellation System-acquired original channel cancellation publication.
         * @param observed_epoch Original publication epoch acquired on device, or recorded CPU input.
         * @param observed_message Descriptor read after that acquire, never a host GPU observation.
         * @return Ready only for the exact successful message/epoch; real prior faults outrank cancellation.
         */
        [[nodiscard]] LLAMINAR_TRANSFER_CHANNEL_HD constexpr CapturedTransferStatus status(
            std::uint64_t bytes, std::uint64_t cancellation, std::uint64_t observed_epoch,
            CapturedTransferMessage observed_message) const noexcept
        {
            if (!valid() || control_->identity.nonce != expected_.nonce || control_->identity.capacity != expected_.capacity)
                return CapturedTransferStatus::InvalidBinding;
            if (cursor_->phase == CapturedTransferPhase::Failed) return CapturedTransferStatus::InvalidPhase;
            if (cursor_->phase == CapturedTransferPhase::Cancelled)
                return cancellation == expected_.nonce ? CapturedTransferStatus::Cancelled : CapturedTransferStatus::InvalidBinding;
            if (cursor_->phase != CapturedTransferPhase::Idle) return CapturedTransferStatus::InvalidPhase;
            if (!cursor_->completed_epoch || cursor_->completed_epoch == kCapturedTransferAbortEpoch ||
                observed_epoch != cursor_->completed_epoch) return CapturedTransferStatus::EpochMismatch;
            if (observed_message.key != message_.key || observed_message.bytes != bytes || bytes > message_.bytes)
                return CapturedTransferStatus::MessageMismatch;
            if (cancellation)
                return cancellation == expected_.nonce ? CapturedTransferStatus::Cancelled : CapturedTransferStatus::InvalidBinding;
            return CapturedTransferStatus::Ready;
        }
    private:
        friend class TransferEngine;
        /** @brief Explicit unbound source-only argument; never accepted by an ACK launch. */
        LLAMINAR_TRANSFER_CHANNEL_HD constexpr CapturedTransferReceiveCompletion() noexcept = default;
        /** @brief Only TransferEngine projects an already authenticated original consumer binding. */
        explicit constexpr CapturedTransferReceiveCompletion(const CapturedTransferChannelDeviceBinding& original) noexcept
            : control_(original.control), cursor_(original.cursor), expected_(original.expected), message_(original.message) {}
        const CapturedTransferChannelControl* control_ = nullptr;
        const CapturedTransferCursor* cursor_ = nullptr;
        CapturedTransferChannelIdentity expected_;
        CapturedTransferMessage message_;
    };
    static_assert(std::is_trivially_copyable_v<CapturedTransferReceiveCompletion>);

    /** @brief Result of one pure transition; only Ready grants payload access. */
    struct CapturedTransferDecision
    {
        CapturedTransferStatus status = CapturedTransferStatus::InvalidBinding;
        std::uint64_t epoch = 0;

        /** @return Whether the caller may execute the already-captured byte operation. */
        [[nodiscard]] LLAMINAR_TRANSFER_CHANNEL_HD constexpr bool ready() const noexcept
        { return status == CapturedTransferStatus::Ready; }
    };

    /**
     * @brief Single-slot protocol shared by device execution and CPU-only tests.
     *
     * Acquire, payload, release is the whole lifecycle. There is no host-owned
     * request counter, reset round or secondary coordinator. Waiting leaves the
     * cursor unchanged. Every semantic failure is absorbing; the lowering must
     * publish AbortEpoch to wake a peer and surface the backend failure.
     */
    class CapturedTransferChannelProtocol final
    {
    public:
        /**
         * @brief Acquire the next slot generation after observing a peer release.
         * @param role This graph's fixed producer/consumer role.
         * @param identity Allocation identity embedded by this captured graph.
         * @param actual Setup-frozen header from the mapped channel.
         * @param cursor This endpoint's private state.
         * @param peer Snapshot acquired from the other endpoint's publication.
         * @param message Semantic role and exact size or positive maximum extent.
         * @param extent_source Immutable authority for this invocation's live size.
         * @param producer_bytes Producer device scalar sampled once, ignored by other sources.
         * @param producer_source_offset Optional offset sampled with that scalar from its original slice.
         * @param producer_source_span Positive immutable source region for a sliced producer; zero for ordinary bindings.
         * @param producer_source_origin Immutable coordinate origin; subtraction never widens the admitted region.
         * @param cancelled_nonce Original CPU parent's terminal publication, acquired with system scope.
         * @return Ready with one lease, WaitForPeer without mutation, or absorbing failure/cancellation.
         *
         * The producer waits for acknowledgement of its previous epoch. The
         * consumer waits for precisely one new producer epoch, then validates
         * the descriptor before reading any payload. An ahead-of-sequence
         * publication is a bug, not permission to skip a message.
         */
        [[nodiscard]] LLAMINAR_TRANSFER_CHANNEL_HD static constexpr CapturedTransferDecision acquire(
            CapturedTransferEndpoint role, CapturedTransferChannelIdentity identity,
            CapturedTransferChannelIdentity actual, CapturedTransferCursor &cursor,
            CapturedTransferPublication peer, CapturedTransferMessage message,
            CapturedTransferExtentSource extent_source = CapturedTransferExtentSource::FixedMessage,
            std::uint64_t producer_bytes = 0,
            std::uint64_t producer_source_offset = 0,
            std::uint64_t producer_source_span = 0,
            std::uint64_t producer_source_origin = 0,
            std::uint64_t cancelled_nonce = 0) noexcept
        {
            if (!identity.valid() || identity.nonce != actual.nonce || identity.capacity != actual.capacity ||
                !message.fits(identity.capacity) ||
                !capturedTransferExtentMatchesRole(extent_source, role) ||
                (extent_source == CapturedTransferExtentSource::ProducerDevice && producer_bytes > message.bytes) ||
                (producer_source_span ? role != CapturedTransferEndpoint::Producer ||
                    extent_source != CapturedTransferExtentSource::ProducerDevice || producer_source_span < message.bytes ||
                    producer_source_origin > ~std::uint64_t{0} - producer_source_span ||
                    producer_source_offset < producer_source_origin ||
                    producer_source_offset - producer_source_origin > producer_source_span ||
                    producer_bytes > producer_source_span - (producer_source_offset - producer_source_origin)
                    : producer_source_offset != 0 || producer_source_origin != 0))
                return fail(cursor, CapturedTransferStatus::InvalidBinding);
            if (cursor.phase == CapturedTransferPhase::Failed)
                return fail(cursor, CapturedTransferStatus::InvalidPhase);
            if (cursor.phase == CapturedTransferPhase::Cancelled)
                return cancelled_nonce == identity.nonce ? fail(cursor, CapturedTransferStatus::Cancelled)
                    : fail(cursor, CapturedTransferStatus::InvalidPhase);
            if (cursor.phase != CapturedTransferPhase::Idle)
                return fail(cursor, CapturedTransferStatus::InvalidPhase);
            if (cancelled_nonce)
                return fail(cursor, cancelled_nonce == identity.nonce
                    ? CapturedTransferStatus::Cancelled : CapturedTransferStatus::InvalidBinding);
            if (peer.epoch == kCapturedTransferAbortEpoch)
                return fail(cursor, CapturedTransferStatus::PeerAborted);
            if (cursor.completed_epoch >= kCapturedTransferAbortEpoch - 1)
                return fail(cursor, CapturedTransferStatus::EpochExhausted);

            const auto completed = cursor.completed_epoch;
            if (role == CapturedTransferEndpoint::Producer)
            {
                if (peer.epoch > completed || (completed > 0 && peer.epoch < completed - 1))
                    return fail(cursor, CapturedTransferStatus::EpochMismatch);
                if (peer.epoch != completed)
                    return {CapturedTransferStatus::WaitForPeer, 0};
                if (extent_source == CapturedTransferExtentSource::ProducerDevice)
                    message.bytes = producer_bytes;
            }
            else
            {
                if (peer.epoch < completed || peer.epoch > completed + 1)
                    return fail(cursor, CapturedTransferStatus::EpochMismatch);
                if (peer.epoch == completed)
                    return {CapturedTransferStatus::WaitForPeer, 0};
                if (peer.message.key != message.key ||
                    (extent_source == CapturedTransferExtentSource::FixedMessage
                        ? peer.message.bytes != message.bytes : peer.message.bytes > message.bytes))
                    return fail(cursor, CapturedTransferStatus::MessageMismatch);
                // This is the producer's authenticated extent, not a receiver
                // guess. An empty message still owns and retires one epoch.
                message = peer.message;
            }
            cursor.phase = CapturedTransferPhase::Acquired;
            cursor.acquired_message = message;
            cursor.acquired_source_offset = producer_source_offset - producer_source_origin;
            return {CapturedTransferStatus::Ready, completed + 1};
        }

        /**
         * @brief Retire an exact lease after its payload read/write completes.
         * @param cursor Endpoint-private owner that acquired the slot.
         * @param lease Epoch returned by the successful acquire, never a host clock.
         * @param publication Descriptor to release-publish after this transition.
         * @param cancelled_nonce Optional system-acquired original parent cancellation.
         * @param original_nonce Original physical identity for that cancellation.
         * @return Ready with the completed epoch, or absorbing failure/cancellation.
         * Original phase/epoch errors are validated before cancellation.
         *
         * The lowering writes descriptor fields before the system-release epoch
         * store. A consumer acknowledgement follows the copy into its own bank,
         * so the producer may overwrite the mapped slot without touching that
         * bank. Capture contains the entire acquire/copy/publication order.
         */
        [[nodiscard]] LLAMINAR_TRANSFER_CHANNEL_HD static constexpr CapturedTransferDecision publish(
            CapturedTransferCursor &cursor, std::uint64_t lease, CapturedTransferPublication &publication,
            std::uint64_t cancelled_nonce = 0,std::uint64_t original_nonce = 0) noexcept
        {
            if (cursor.phase != CapturedTransferPhase::Acquired)
                return fail(cursor, CapturedTransferStatus::InvalidPhase);
            if (lease == 0 || lease == kCapturedTransferAbortEpoch || lease != cursor.completed_epoch + 1 ||
                publication.epoch != cursor.completed_epoch)
                return fail(cursor, CapturedTransferStatus::EpochMismatch);
            if(cancelled_nonce)return fail(cursor,cancelled_nonce==original_nonce && original_nonce
                ?CapturedTransferStatus::Cancelled:CapturedTransferStatus::InvalidBinding);
            publication.message = cursor.acquired_message;
            publication.epoch = lease;
            cursor.completed_epoch = lease;
            cursor.phase = CapturedTransferPhase::Idle;
            cursor.acquired_message = {};
            cursor.acquired_source_offset = 0;
            return {CapturedTransferStatus::Ready, lease};
        }

        /** @brief Prepare a release locally, validating original faults before cancellation.
         * @param cursor Private snapshot of the original acquired endpoint, committed only on Ready.
         * @param publication Private snapshot of its own publication, committed only on Ready.
         * @param expected Frozen semantic message geometry from the original capture binding.
         * @param extent_source Original fixed/count authority, never a replacement live count.
         * @param cancelled_nonce System-acquired original parent terminal publication.
         * @param original_nonce This channel's frozen physical identity.
         * @return Original phase/message/epoch failure, explicit cancellation, or complete Ready proposal.
         * @note No positive publication or epoch advance escapes a rejected proposal.
         */
        [[nodiscard]] LLAMINAR_TRANSFER_CHANNEL_HD static constexpr CapturedTransferDecision preparePublication(
            CapturedTransferCursor& cursor,CapturedTransferPublication& publication,CapturedTransferMessage expected,
            CapturedTransferExtentSource extent_source,std::uint64_t cancelled_nonce,std::uint64_t original_nonce) noexcept
        {
            if(cursor.phase!=CapturedTransferPhase::Acquired)
                return fail(cursor,CapturedTransferStatus::InvalidPhase);
            if(cursor.acquired_message.key!=expected.key ||
               (extent_source==CapturedTransferExtentSource::FixedMessage
                   ?cursor.acquired_message.bytes!=expected.bytes:cursor.acquired_message.bytes>expected.bytes))
                return fail(cursor,CapturedTransferStatus::MessageMismatch);
            return publish(cursor,cursor.completed_epoch+1,publication,cancelled_nonce,original_nonce);
        }

        /** @brief Decide only an explicit cancellation; prior or concurrent real faults remain fatal.
         * @param phase Actual original endpoint phase, acquired atomically by native lowering.
         * @param status Actual operation outcome, never inferred from a cancellation flag.
         * @param matches_original Whether the original parent's terminal publication matches this channel.
         * @return True only for explicit cancellation with no earlier endpoint fault.
         */
        [[nodiscard]] LLAMINAR_TRANSFER_CHANNEL_HD static constexpr bool drainsCancellation(
            CapturedTransferPhase phase, CapturedTransferStatus status, bool matches_original) noexcept
        {
            return status == CapturedTransferStatus::Cancelled && matches_original &&
                (phase == CapturedTransferPhase::Idle || phase == CapturedTransferPhase::Acquired ||
                    phase == CapturedTransferPhase::Publishing || phase == CapturedTransferPhase::Cancelled);
        }

        /**
         * @brief Abort this owner without modifying the other endpoint's words.
         * @param cursor Endpoint-private owner being retired permanently.
         * @param publication Its own terminal record to system-release publish.
         * The caller invokes this on a semantic failure, deadline or shutdown
         * failure; it may not reset the channel afterward to conceal that fault.
         */
        LLAMINAR_TRANSFER_CHANNEL_HD static constexpr void abort(
            CapturedTransferCursor &cursor, CapturedTransferPublication &publication) noexcept
        {
            cursor.phase = CapturedTransferPhase::Failed;
            publication.epoch = kCapturedTransferAbortEpoch;
        }

    private:
        /** @brief Make every semantic fault absorbing without publishing a false success. */
        [[nodiscard]] LLAMINAR_TRANSFER_CHANNEL_HD static constexpr CapturedTransferDecision fail(
            CapturedTransferCursor &cursor, CapturedTransferStatus status) noexcept
        {
            cursor.phase = status == CapturedTransferStatus::Cancelled && cursor.phase != CapturedTransferPhase::Failed
                ? CapturedTransferPhase::Cancelled : CapturedTransferPhase::Failed;
            return {status, 0};
        }
    };

    static_assert(std::is_trivially_copyable_v<CapturedTransferPublication>);
    static_assert(std::is_trivially_copyable_v<CapturedTransferCursor>);
    static_assert(sizeof(CapturedTransferPublication) == 64);
    static_assert(sizeof(CapturedTransferCursor) == 40);
    static_assert(sizeof(CapturedTransferChannelControl) == 192);
}

#undef LLAMINAR_TRANSFER_CHANNEL_HD
