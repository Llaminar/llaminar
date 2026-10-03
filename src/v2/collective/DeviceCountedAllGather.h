/**
 * @file DeviceCountedAllGather.h
 * @brief Frozen rank-local allgather topology with device-authored message lengths.
 *
 * One domain owns one channel per directed peer pair and one extent bank per
 * participant. Sequential layers/graph families share those owners. This is a
 * byte exchange, not a router, arithmetic policy, or live placement authority.
 * TransferEngine owns every message epoch and publication; PMA owns every byte.
 */
#pragma once

#include "backends/PeerAccessCoverage.h"
#include "transfer/CapturedTransferChannel.h"
#include <span>
#include <vector>

namespace llaminar2
{
    /** @brief Pure per-participant contribution to the domain's physical BOM. */
    struct DeviceCountedAllGatherMemory
    {
        std::size_t host_bytes = 0; ///< Outgoing mapped slots, charged once by each producer.
        std::size_t device_bytes = 0; ///< Incoming/outgoing cursors and one extent per source.
    };

    /**
     * @brief Complete, capture-bound participant of a counted byte collective.
     *
     * The local producer writes its extent before enqueue. Receivers obtain
     * their extents only from authenticated TransferEngine publication. Enqueue
     * submits all sends before any receives, preventing cyclic submission
     * dependencies. No local packet is sent back to its own producer.
     */
    class DeviceCountedAllGatherBinding final
    {
    public:
        /** @brief Enqueue the complete participant on its exact graph stream.
         * @param stream Non-null stream ordered after local packet production.
         * @throws std::invalid_argument For a missing stream or stale binding. */
        void enqueue(void *stream) const;
        /** @brief Enqueue on an acquired fork; its paired join publishes the receive bank.
         * @param input Exact retained producer owner and auxiliary-stream event proof.
         * @throws std::exception For a foreign fork or incomplete binding. */
        void enqueueAcquiredInput(const AcquiredDeviceTransferInput &input) const;
        /** @return Stable device extent for one source, never a host observation.
         * @param source Ordered domain participant.
         * @throws std::out_of_range For a foreign participant. */
        [[nodiscard]] std::uint64_t *extent(std::size_t source) const;

    private:
        friend class DeviceCountedAllGather;
        std::shared_ptr<class DeviceCountedAllGather> owner_;
        std::size_t participant_ = 0;
        std::vector<CapturedTransferBinding> sends_, receives_;
        ITensor *received_ = nullptr; ///< Borrowed canonical owner retained by receives_.
    };

    /**
     * @brief Shared physical fabric for sequential captured allgather invocations.
     *
     * Only homogeneous, rank-local domains with proven absent native P2P may
     * install this transport. Enabled/unknown P2P is never silently bypassed.
     * The caller joins submitted graphs before retiring their bindings/fabric;
     * destruction releases storage without synchronizing inference.
     */
    class DeviceCountedAllGather final : public std::enable_shared_from_this<DeviceCountedAllGather>
    {
    public:
        /** @brief Size exactly the physical owners materialized by create().
         * @param participants Distinct GPU endpoints, at least two.
         * @param packet_capacity Maximum bytes per producer, not live wire length.
         * @throws std::invalid_argument For empty geometry.
         * @throws std::overflow_error For an unrepresentable BOM. */
        [[nodiscard]] static DeviceCountedAllGatherMemory memoryFor(std::size_t participants, std::size_t packet_capacity);

        /** @brief Materialize an already admitted complete domain outside capture.
         * @param authority Sole physical admission/allocation authority.
         * @param host_device Rank-local first-touch backing resource.
         * @param devices Exact ordered homogeneous communicator membership.
         * @param streams Exact setup streams, one per device.
         * @param coverage Canonical driver-backed P2P coverage; must be None.
         * @param packet_capacity Maximum physical payload for every directed edge.
         * @throws std::invalid_argument For inconsistent topology, P2P, or streams. */
        [[nodiscard]] static std::shared_ptr<DeviceCountedAllGather> create(
            PhysicalMemoryAuthority &authority, DeviceId host_device,
            std::span<const DeviceId> devices, std::span<void *const> streams,
            PeerAccessCoverage coverage, std::size_t packet_capacity);

        /** @brief Bind existing participant-local arena banks to one semantic message.
         * @param participant Exact communicator coordinate, not physical ordinal.
         * @param message Frozen semantic identity and maximum payload for this graph.
         * @param local Producer bank; its existing bits are transmitted unchanged.
         * @param received Peer-major bank, excluding this participant's local packet.
         * @throws std::invalid_argument For foreign/aliased or insufficient storage.
         *
         * Receive slot for source s is s when s < participant, otherwise s-1.
         * The physical stride is message.bytes; acquired counts restrict reads.
         */
        [[nodiscard]] DeviceCountedAllGatherBinding bind(std::size_t participant,
            CapturedTransferMessage message, std::shared_ptr<ITensor> local, std::shared_ptr<ITensor> received);
        /** @return Immutable ordered GPU membership. */
        [[nodiscard]] const std::vector<DeviceId> &devices() const noexcept { return devices_; }
        /** @return Admitted per-message capacity, never the next replay's length. */
        [[nodiscard]] std::size_t capacityBytes() const noexcept { return capacity_; }
        /** @return Read-only device count address for a participant's acquired source packet.
         * @param participant Consumer coordinate; the local source names its producer count.
         * @param source Exact source coordinate in the same frozen domain.
         * @throws std::out_of_range For either foreign coordinate.
         *
         * This exposes an address, not a host value. Graph consumers must follow
         * the collective; explicit terminal diagnostics must first join execution.
         */
        [[nodiscard]] const std::uint64_t *extentAddress(std::size_t participant, std::size_t source) const;
        /** @return Stable device address written only by this participant's local producer.
         * @param participant Exact frozen membership coordinate.
         * @throws std::out_of_range For foreign membership.
         * This is a graph binding, never permission for host reads or writes. */
        [[nodiscard]] std::uint64_t *producerExtentAddress(std::size_t participant) const
        { return const_cast<std::uint64_t *>(extentAddress(participant, participant)); }

    private:
        friend class DeviceCountedAllGatherBinding;
        /** @brief Construct only through the complete setup transaction. */
        DeviceCountedAllGather() = default;
        /** @brief Extent storage retires before its canonical physical claim. */
        struct Extents
        {
            PhysicalMemoryAllocationLease claim;
            std::shared_ptr<DeviceTransferBuffer> storage;
        };
        std::size_t capacity_ = 0;
        std::vector<DeviceId> devices_;
        std::vector<Extents> extents_;
        std::vector<std::shared_ptr<CapturedTransferChannel>> channels_; ///< Row-major directed edges; diagonal empty.
    };
}
