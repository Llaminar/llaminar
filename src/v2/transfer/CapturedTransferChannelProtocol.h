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

    /** @brief Local lease state; a failed endpoint is never reusable. */
    enum class CapturedTransferPhase : std::uint32_t { Idle, Acquired, Publishing, Failed };

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
    };

    /** Reserved terminal value releases a peer into failure, never into payload work. */
    inline constexpr std::uint64_t kCapturedTransferAbortEpoch = ~std::uint64_t{0};

    /**
     * @brief Exact captured message identity, not mutable model/sampler state.
     *
     * The key identifies a setup-frozen semantic role/geometry in the graph
     * family; byte count is the live physical extent, not allocated capacity.
     * Both endpoints must name the same key and extent for each transaction.
     */
    struct CapturedTransferMessage
    {
        std::uint64_t key = 0;
        std::uint64_t bytes = 0;

        /** @return Whether the immutable message fits its admitted slot. */
        [[nodiscard]] LLAMINAR_TRANSFER_CHANNEL_HD constexpr bool fits(
            std::uint64_t capacity) const noexcept
        { return key != 0 && bytes != 0 && bytes <= capacity; }
    };

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
        CapturedTransferMessage acquired_message;
    };

    /** @brief Setup-owned mapped header and independent release words. */
    struct alignas(64) CapturedTransferChannelControl
    {
        CapturedTransferChannelIdentity identity;
        CapturedTransferPublication producer;
        CapturedTransferPublication consumer;
    };

    /** @brief Backend-only lowering operation; callers submit a whole transfer through TransferEngine. */
    enum class CapturedTransferBoundaryOperation : std::uint32_t { Acquire, Publish };

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

        /** @return Whether immutable fields can name a bounded native operation. */
        [[nodiscard]] LLAMINAR_TRANSFER_CHANNEL_HD constexpr bool valid() const noexcept
        {
            return control && cursor && expected.valid() && message.fits(expected.capacity) && timeout_ticks != 0 &&
                (role == CapturedTransferEndpoint::Producer || role == CapturedTransferEndpoint::Consumer);
        }
    };

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
         * @param message Exact bytes and semantic graph role of this invocation.
         * @return Ready with one lease, WaitForPeer without mutation, or failure.
         *
         * The producer waits for acknowledgement of its previous epoch. The
         * consumer waits for precisely one new producer epoch, then validates
         * the descriptor before reading any payload. An ahead-of-sequence
         * publication is a bug, not permission to skip a message.
         */
        [[nodiscard]] LLAMINAR_TRANSFER_CHANNEL_HD static constexpr CapturedTransferDecision acquire(
            CapturedTransferEndpoint role, CapturedTransferChannelIdentity identity,
            CapturedTransferChannelIdentity actual, CapturedTransferCursor &cursor,
            CapturedTransferPublication peer, CapturedTransferMessage message) noexcept
        {
            if (!identity.valid() || identity.nonce != actual.nonce || identity.capacity != actual.capacity ||
                !message.fits(identity.capacity) ||
                (role != CapturedTransferEndpoint::Producer && role != CapturedTransferEndpoint::Consumer))
                return fail(cursor, CapturedTransferStatus::InvalidBinding);
            if (cursor.phase != CapturedTransferPhase::Idle)
                return fail(cursor, CapturedTransferStatus::InvalidPhase);
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
            }
            else
            {
                if (peer.epoch < completed || peer.epoch > completed + 1)
                    return fail(cursor, CapturedTransferStatus::EpochMismatch);
                if (peer.epoch == completed)
                    return {CapturedTransferStatus::WaitForPeer, 0};
                if (peer.message.key != message.key || peer.message.bytes != message.bytes)
                    return fail(cursor, CapturedTransferStatus::MessageMismatch);
            }
            cursor.phase = CapturedTransferPhase::Acquired;
            cursor.acquired_message = message;
            return {CapturedTransferStatus::Ready, completed + 1};
        }

        /**
         * @brief Retire an exact lease after its payload read/write completes.
         * @param cursor Endpoint-private owner that acquired the slot.
         * @param lease Epoch returned by the successful acquire, never a host clock.
         * @param publication Descriptor to release-publish after this transition.
         * @return Ready with the completed epoch, or absorbing failure.
         *
         * The lowering writes descriptor fields before the system-release epoch
         * store. A consumer acknowledgement follows the copy into its own bank,
         * so the producer may overwrite the mapped slot without touching that
         * bank. Capture contains the entire acquire/copy/publication order.
         */
        [[nodiscard]] LLAMINAR_TRANSFER_CHANNEL_HD static constexpr CapturedTransferDecision publish(
            CapturedTransferCursor &cursor, std::uint64_t lease, CapturedTransferPublication &publication) noexcept
        {
            if (cursor.phase != CapturedTransferPhase::Acquired)
                return fail(cursor, CapturedTransferStatus::InvalidPhase);
            if (lease == 0 || lease == kCapturedTransferAbortEpoch || lease != cursor.completed_epoch + 1 ||
                publication.epoch != cursor.completed_epoch)
                return fail(cursor, CapturedTransferStatus::EpochMismatch);
            publication.message = cursor.acquired_message;
            publication.epoch = lease;
            cursor.completed_epoch = lease;
            cursor.phase = CapturedTransferPhase::Idle;
            cursor.acquired_message = {};
            return {CapturedTransferStatus::Ready, lease};
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
            cursor.phase = CapturedTransferPhase::Failed;
            return {status, 0};
        }
    };

    static_assert(std::is_trivially_copyable_v<CapturedTransferPublication>);
    static_assert(std::is_trivially_copyable_v<CapturedTransferCursor>);
    static_assert(sizeof(CapturedTransferPublication) == 64);
    static_assert(sizeof(CapturedTransferChannelControl) == 192);
}

#undef LLAMINAR_TRANSFER_CHANNEL_HD
