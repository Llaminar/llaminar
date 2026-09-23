/**
 * @file CapturedTransferChannel.h
 * @brief Admitted rank-local message storage and immutable capture bindings.
 *
 * TransferEngine alone constructs a channel or enqueues a complete message.
 * The mapped slot has two GPU-private cursors and one initialization event per
 * endpoint, joined once before setup returns. Captured bindings retain those owners, exact tensor/buffer storage,
 * and message geometry; neither binding nor channel exposes live protocol state
 * to the host. This is a same-process/node transport, not an MPI address space.
 */
#pragma once

#include "CapturedTransferChannelProtocol.h"
#include "TransferEngine.h"
#include "planning/PhysicalMemoryAuthority.h"

#include <array>
#include <variant>

namespace llaminar2
{
    /** @brief Pure allocation geometry contributed to the canonical memory plan. */
    struct CapturedTransferChannelMemory
    {
        size_t mapped_host_bytes = 0; ///< Page-rounded control plus one byte slot.
        size_t cursor_bytes_per_device = 0; ///< Exact private cursor, on each GPU.
    };

    /**
     * @brief One non-resettable, rank-local GPU producer/consumer channel.
     *
     * Bindings hold this owner until their graph family is retired. The enclosing
     * graph owner must join its final submitted work before releasing its graph
     * and bindings, just as for every other captured arena allocation. Destruction
     * releases events/storage before their PMA leases and never synchronizes.
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

    private:
        friend class TransferEngine;
        /** @brief Create only inside TransferEngine's all-or-nothing setup transaction. */
        CapturedTransferChannel() = default;

        /** @brief Private endpoint resources, declared in reverse retirement order. */
        struct Endpoint
        {
            PhysicalMemoryAllocationLease lease; ///< Released after cursor storage.
            std::shared_ptr<DeviceTransferBuffer> cursor;
            IBackend *backend = nullptr; ///< Exact process-lifetime backend authority.
            DeviceId device = DeviceId::invalid();
            void *initialized_event = nullptr; ///< Recorded once after setup memset.
            std::uint64_t timeout_ticks = 0; ///< Frozen native clock geometry.
        };
        static constexpr size_t payload_offset = sizeof(CapturedTransferChannelControl);
        CapturedTransferChannelIdentity identity_;
        PhysicalMemoryAllocationLease mapped_lease_; ///< Outlives mapped storage.
        std::shared_ptr<MappedHostTransferRegion> mapped_;
        std::array<Endpoint, 2> endpoints_;
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
        /** @return Immutable message geometry embedded in the recorded operation. */
        [[nodiscard]] CapturedTransferMessage message() const noexcept { return native_.message; }
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
    };
}
