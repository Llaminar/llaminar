/**
 * @file CapturedTransferChannel.h
 * @brief Admitted rank-local message storage and immutable capture bindings.
 *
 * TransferEngine alone constructs a channel or enqueues a complete message.
 * The explicitly selected payload slot has two GPU-private cursors. Cold setup observes each exact
 * original initialization stream before publication, creating no extra events. Captured bindings retain those owners, exact tensor/buffer storage,
 * and message geometry; neither binding nor channel exposes live protocol state
 * to the host. This is a same-process/node transport, not an MPI address space.
 */
#pragma once

#include "CapturedTransferChannelProtocol.h"
#include "TransferEngine.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "backends/GPUDeviceContextPool.h"
#include <optional>
#include <chrono>
#include <mutex>

#include <array>
#include <variant>

namespace llaminar2
{
    /** @brief Pure allocation geometry contributed to the canonical memory plan. */
    struct CapturedTransferChannelMemory
    {
        size_t mapped_host_bytes = 0; ///< Page-rounded control plus payload only in MappedHost mode.
        size_t cursor_bytes_per_device = 0; ///< Exact private cursor, on each GPU.
        size_t producer_payload_bytes = 0; ///< Native peer slab, absent for mapped-host payload.
    };

    /**
     * @brief One non-resettable, rank-local GPU producer/consumer channel.
     *
     * Bindings hold this owner until their graph family is retired. The enclosing
     * graph owner must join its final submitted work before releasing its graph
     * and bindings, just as for every other captured arena allocation. Destruction
     * releases mapped/cursor storage before its original worker and PMA claims and never synchronizes.
     */
    class CapturedTransferChannel final
    {
    public:
        /** @brief Retire exact physical owners after the graph family's final join. */
        ~CapturedTransferChannel();
        CapturedTransferChannel(const CapturedTransferChannel &) = delete;
        CapturedTransferChannel &operator=(const CapturedTransferChannel &) = delete;

        /** @return Immutable capacity available to one message, not its live length. */
        [[nodiscard]] size_t capacityBytes() const noexcept { return identity_.capacity; }
        /** @return Exact allocation identity required in the enclosing graph-cache key. */
        [[nodiscard]] CapturedTransferChannelIdentity identity() const noexcept { return identity_; }
        /** @return Frozen physical endpoint identity; never queries live device state.
         * @throws std::invalid_argument for an unknown endpoint role. */
        [[nodiscard]] DeviceId endpointDevice(CapturedTransferEndpoint role) const;
        /** @return Exact physical geometry used both in planning and materialization. */
        [[nodiscard]] static CapturedTransferChannelMemory memoryFor(size_t capacity);
        /** @brief Quote the exact explicitly selected allocation route; never inspect residual capacity.
         * @param capacity Positive maximum message bytes. @param storage Exact physical payload route.
         * @return One mapped control bank, native cursors, and the selected payload allocation.
         * @throws std::exception For invalid route, zero capacity or arithmetic overflow.
         */
        [[nodiscard]] static CapturedTransferChannelMemory memoryFor(size_t capacity, CapturedTransferPayloadStorage storage);
        /** @return Immutable physical route retained in every endpoint capture identity. */
        CapturedTransferPayloadStorage payloadStorage() const noexcept { return payload_storage_; }


    private:
        friend class TransferEngine;
        friend class CapturedTransferBinding;
        friend struct CapturedTransferChannelReplayProbe; ///< Test-only payload canary access after both endpoints' terminal joins; no production observation API.
        /** @brief Bind the existing original parent's immutable identity before native submissions.
         * @param original Nonzero original admission; never a channel-owned request counter.
         * @throws std::exception For stale endpoint generations or overlapping/terminal admission.
         */
        void beginInvocation(CapturedTransferInvocation original);
        /** @brief Retire admission only after every original native endpoint has joined.
         * @param original Exact original admission whose terminal joins the caller owns.
         * @return Truthful completed/cancelled outcome; cancellation stays absorbing and never permits reuse.
         * @throws std::logic_error For missing or foreign admission.
         */
        CapturedTransferInvocationRetirement finishInvocation(CapturedTransferInvocation original);
        /** @brief Publish terminal cancellation for this channel's exact active original admission.
         * @param original Existing parent's identity, including its immutable generation.
         * @return False for stale/foreign admission; true for idempotent terminal cancellation.
         * @note Original physical owners remain retained until every submitted endpoint joins.
         */
        bool cancelInvocation(CapturedTransferInvocation original);
        /** @brief Create only inside TransferEngine's all-or-nothing setup transaction. */
        CapturedTransferChannel() = default;

        /** @brief Private endpoint resources, declared in reverse retirement order. */
        struct Endpoint
        {
            std::optional<GPUContextProvenance> original; ///< Retires after its cursor and canonical claim.
            PhysicalMemoryAllocationLease lease; ///< Released after cursor storage.
            std::shared_ptr<DeviceTransferBuffer> cursor;
            IBackend *backend = nullptr; ///< Exact process-lifetime backend authority.
            DeviceId device = DeviceId::invalid();
            enum class Initialization {Unused,Pending,Ready};
            std::uint64_t runtime_generation=0;
            void* setup_stream=nullptr;
            Initialization initialization=Initialization::Unused;
            std::chrono::steady_clock::time_point initialization_deadline{};
            std::uint64_t timeout_ticks = 0; ///< Frozen native clock geometry.
        };
        /** @brief Cold-only bounded initialization join; replay cannot invoke this operation. */
        static void completeColdInitialization(Endpoint& endpoint);
        /** @brief Authenticate original ready generation/controlblock before accessing any frozen pointer. */
        void requireOriginalEndpoint(CapturedTransferEndpoint role)const;
        static constexpr size_t payload_offset = sizeof(CapturedTransferChannelControl);
        CapturedTransferChannelIdentity identity_;
        std::array<Endpoint, 2> endpoints_; ///< Original workers outlive mapped registration teardown.
        PhysicalMemoryAllocationLease mapped_lease_; ///< Outlives mapped storage.
        std::shared_ptr<MappedHostTransferRegion> mapped_;
        std::mutex invocation_mutex_; ///< Serializes CPU admission/cancellation only; never native progress.
        CapturedTransferInvocation original_invocation_{};
        bool invocation_active_ = false, terminal_cancelled_ = false;
        PhysicalMemoryAllocationLease peer_payload_lease_; ///< Released after native peer payload storage.
        std::shared_ptr<DeviceTransferBuffer> peer_payload_; ///< Producer-owned slab retained through both endpoint bindings.
        CapturedTransferPayloadStorage payload_storage_ = CapturedTransferPayloadStorage::MappedHost;

    };

    /**
     * @brief Unforgeable fixed-storage message recorded in a retained graph.
     *
     * Sharing a channel across graph families is legal only when their endpoint
     * operations do not overlap. The GPU cursor rejects overlapping acquisition;
     * a host flag is not the authority. Source production and destination use
     * belong to the same capture DAG/explicit event edges as the byte operation.
     */
    class CapturedTransferBinding final
    {
    public:
        /** @brief Share the same immutable physical/count owners without another native borrow. */
        CapturedTransferBinding(const CapturedTransferBinding &) = default;
        /** @brief Move the complete retained endpoint with its original allocation borrow. */
        CapturedTransferBinding(CapturedTransferBinding &&) noexcept = default;
        /**
         * @brief Replace a complete binding before retiring its former physical owners.
         * @param other Fully retained replacement, copied or moved before the swap.
         * @return This binding after complete ownership replacement.
         *
         * The temporary retires its allocation borrow before its final tensor
         * owner. Memberwise assignment could release that tensor while the old
         * borrow was still live, violating the allocator's free contract.
         */
        CapturedTransferBinding &operator=(CapturedTransferBinding other) noexcept
        {
            using std::swap;
            swap(channel_, other.channel_); swap(storage_, other.storage_);
            swap(native_, other.native_); swap(device_, other.device_);
            swap(device_bytes_, other.device_bytes_); swap(offset_, other.offset_);
            swap(extent_storage_, other.extent_storage_); swap(extent_offset_, other.extent_offset_);
            swap(extent_capture_, other.extent_capture_); swap(source_capture_, other.source_capture_);
            return *this;
        }
        /**
         * @return Complete immutable channel, payload, extent, slice and allocation-borrow capture identity.
         * @note This never dereferences device state or joins a producer. Enqueue separately
         * rejects original owner/backend/address drift before submitting the retained binding.
         */
        [[nodiscard]] std::uint64_t captureIdentity() const;
        /** @return Exact fixed extent or counted maximum embedded in the recorded operation. */
        [[nodiscard]] CapturedTransferMessage message() const noexcept { return native_.message; }
        /** @return Capture-frozen extent authority; never reads the live device count. */
        [[nodiscard]] CapturedTransferExtentSource extentSource() const noexcept { return native_.extent_source; }
        /** @return Whether this original endpoint carries a capture-frozen source slice descriptor. */
        [[nodiscard]] bool hasSourceSlice() const noexcept { return native_.source_slice != nullptr; }
        /** @return Physical device which must execute this endpoint. */
        [[nodiscard]] DeviceId device() const noexcept { return device_; }
        /** @return Producer or consumer, without consulting mutable GPU state. */
        [[nodiscard]] CapturedTransferEndpoint role() const noexcept { return native_.role; }

    private:
        friend class TransferEngine;
        /** @brief Bind only after validating storage identity, bounds and residency. */
        CapturedTransferBinding() = default;
        std::shared_ptr<CapturedTransferChannel> channel_;
        std::variant<std::shared_ptr<DeviceTransferBuffer>, std::shared_ptr<TensorBase>,
            std::shared_ptr<const WorkspaceBufferLease>> storage_;
        CapturedTransferChannelDeviceBinding native_;
        DeviceId device_ = DeviceId::invalid();
        void *device_bytes_ = nullptr; ///< Immutable physical subregion start.
        size_t offset_ = 0; ///< Revalidated against the retained physical owner.
        std::variant<std::monostate, std::shared_ptr<DeviceTransferBuffer>,
            std::shared_ptr<const WorkspaceBufferLease>, std::shared_ptr<TensorBase>> extent_storage_; ///< Exact count input/output owner.
        size_t extent_offset_ = 0; ///< Aligned uint64 within that retained device region.
        std::shared_ptr<const CapturedDeviceStorageLease> extent_capture_; ///< Original tensor allocation borrow; retires before its strong tensor owner.
        std::shared_ptr<const CapturedDeviceStorageLease> source_capture_; ///< Optional sliced Tensor source borrow, distinct from descriptor authority.
    };
}
