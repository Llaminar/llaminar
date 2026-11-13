/**
 * @file DeviceCountedAllGather.h
 * @brief Frozen rank-local allgather topology with device-authored message lengths.
 *
 * One domain borrows its original NCCL/RCCL communicator and one extent/frame bank per
 * participant. Sequential layers/graph families share those owners. This is a
 * byte exchange, not a router, arithmetic policy, or live placement authority.
 * Native completion owns every byte receipt; TransferEngine owns event publication; PMA owns every byte.
 */
#pragma once

#include "backends/PeerAccessCoverage.h"
#include "execution/mtp/DeviceGenerationCollectiveReceipt.h"
#include "execution/mpi_orchestration/DeviceInventory.h"
#include "transfer/CapturedTransferChannel.h"
#include "NativeCountedAllGatherFrame.h"
#include "NativeLocalTPContextBorrow.h"
#include <span>
#include <utility>
#include <vector>

namespace llaminar2
{
    class ILocalTPContext;
    /** @brief First-class original rank-local native transport; hardware P2P remains native-owned. */
    enum class DeviceCountedAllGatherTransport : std::uint8_t { NativeCollective };

    /** @brief Pure per-participant contribution to the domain's physical BOM. */
    struct DeviceCountedAllGatherMemory
    {
        std::size_t host_bytes = 0; ///< Zero: native execution has no mapped payload/control slabs.
        std::size_t device_bytes = 0; ///< Actual-degree counts, invocation frame, native operands, receipts and counters.
    };

    /** @brief Exact original rank-ordered packet capacities for one unequal native allgather. */
    class DeviceCountedAllGatherPacketLayout final {
    public:
        /** @brief Freeze positive per-source extents and this actual member's packed receive geometry. */
        DeviceCountedAllGatherPacketLayout(std::span<const std::size_t> capacities, std::size_t participant);
        /** @return Maximum capacity of any original source, before any live data is known. */
        std::size_t maximumBytes() const noexcept { return maximum_bytes_; }
        /** @return Physical capacity of the local packet. */
        std::size_t localBytes() const noexcept { return capacities_[participant_]; }
        /** @return Combined physical capacity of all actual remote packets, excluding self. */
        std::size_t receivedBytes() const noexcept { return received_bytes_; }
        /** @return Actual original group member count. */
        std::size_t degree() const noexcept { return capacities_.size(); }
        /** @return This capture's original local coordinate. */
        std::size_t participant() const noexcept { return participant_; }
        /** @return Physical capacity for an original source. */
        std::size_t capacity(std::size_t source) const { return capacities_.at(source); }
        /** @return Source-ordered offset in this receiver's admitted bank; self has no slot. */
        std::size_t receiveOffset(std::size_t source) const;
    private:
        std::vector<std::size_t> capacities_, receive_offsets_;
        std::size_t participant_, maximum_bytes_ = 0, received_bytes_ = 0;
    };

    /**
     * @brief Immutable original peer routes and their canonical physical BOM.
     *
     * This is setup metadata, not a live allocation ledger. Complete original
     * membership admits the native transport and its O(N) completion metadata.
     * Incomplete and one-way observations cannot materialize any storage.
     */
    class DeviceCountedAllGatherTopology final
    {
    public:
        /** @brief Resolve ordered membership from the original cached rank observation.
         * @param inventory Complete original rank-local GPU/P2P observation.
         * @param devices Distinct homogeneous GPU endpoints, at least two.
         * @return Frozen channel routes retaining the observation used to authenticate peers.
         * @throws std::invalid_argument For incomplete, foreign or one-way peer facts.
         * @throws std::overflow_error For unrepresentable route geometry. */
        [[nodiscard]] static DeviceCountedAllGatherTopology resolve(
            const RankInventory &inventory, std::span<const DeviceId> devices);
        /** @return Exact original owner rank; never inferred from a device ordinal. */
        [[nodiscard]] int worldRank() const noexcept { return inventory_.rank; }
        /** @return Original ordered communicator membership. */
        [[nodiscard]] const std::vector<DeviceId> &devices() const noexcept { return devices_; }
        /** @return Original hardware observation retained for native topology admission. */
        [[nodiscard]] const RankInventory &inventory() const noexcept { return inventory_; }
        /** @return Complete/None, or mixed symmetric cliques; never a one-way edge. */
        [[nodiscard]] PeerAccessCoverage coverage() const noexcept { return coverage_; }
        /** @brief Read the immutable physical route for one non-diagonal edge.
         * @param producer Original producer coordinate. @param consumer Original consumer coordinate.
         * @return First-class original native collective transport.
         * @throws std::out_of_range For a foreign or diagonal coordinate. */
        [[nodiscard]] DeviceCountedAllGatherTransport payloadStorage(
            std::size_t producer, std::size_t consumer) const;
        /** @brief Quote this participant's exact allocation contribution for PMA.
         * @param participant Original communicator coordinate.
         * @param packet_capacity Admitted maximum, never the live wire extent.
         * @return Actual-degree device counts, invocation frame, operands, receipts and counters; no payload slabs.
         * @throws std::exception For invalid capacity/index or arithmetic overflow. */
        [[nodiscard]] DeviceCountedAllGatherMemory memoryFor(
            std::size_t participant, std::size_t packet_capacity) const;
        /** @brief Compare original route identity without comparing capacity observations.
         * @param other Another immutable planning/setup projection.
         * @return Whether owner, order and all physical routes agree. */
        [[nodiscard]] bool sameRoutes(const DeviceCountedAllGatherTopology &other) const noexcept;
    private:
        DeviceCountedAllGatherTopology() = default;
        RankInventory inventory_; ///< Frozen discovery metadata, never live accounting.
        std::vector<DeviceId> devices_;
        std::vector<DeviceCountedAllGatherTransport> routes_; ///< Row-major; diagonal unused.
        PeerAccessCoverage coverage_ = PeerAccessCoverage::None;
    };

    /**
     * @brief Complete, capture-bound participant of a counted byte collective.
     *
     * The local producer writes its extent before enqueue. Receivers obtain
     * their extents from original completed native broadcasts. Every participant
     * submits identical ordered roots, then validates receipts on device before
     * direct enqueue publishes its exact stream frontier. An acquired auxiliary
     * fork defers receive publication to its enclosing completion-event join.
     */
    class DeviceCountedAllGatherBinding final
    {
    public:
        /** @brief Empty binding owns neither allocation nor capture lease. */
        DeviceCountedAllGatherBinding() = default;
        /** @brief Retain the same complete tensor/lease lifetime bundle. */
        DeviceCountedAllGatherBinding(const DeviceCountedAllGatherBinding &) = default;
        /** @brief Transfer the complete bundle without releasing its captured allocations. */
        DeviceCountedAllGatherBinding(DeviceCountedAllGatherBinding &&) noexcept = default;
        /**
         * @brief Replace one complete lifetime bundle, including an empty reset.
         * @param replacement Copy or moved bundle retained before this one changes.
         * @return This binding after the replacement is installed.
         *
         * Memberwise assignment would release tensor owners before their old
         * capture leases. Exchange every member first; the temporary's normal
         * reverse destruction then retires leases before the captured owners.
         * The same transaction preserves copy and move self-assignment.
         */
        DeviceCountedAllGatherBinding &operator=(DeviceCountedAllGatherBinding replacement) noexcept
        {
            swap(replacement);
            return *this;
        }
        /** @brief Enqueue, validate and publish the receive bank on its exact graph stream.
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
        /** @brief Exchange whole bundles without destroying a resource mid-transaction.
         * @param other Complete replacement retaining every owner and its matching capture lease. */
        void swap(DeviceCountedAllGatherBinding &other) noexcept
        {
            owner_.swap(other.owner_);
            std::swap(participant_,other.participant_);
            local_owner_.swap(other.local_owner_);
            received_owner_.swap(other.received_owner_);
            local_capture_.swap(other.local_capture_);
            received_capture_.swap(other.received_capture_);
            std::swap(message_,other.message_);
            std::swap(receipt_, other.receipt_);
            packets_.swap(other.packets_);
        }
        std::shared_ptr<class DeviceCountedAllGather> owner_;
        std::size_t participant_ = 0;
        std::shared_ptr<ITensor> local_owner_, received_owner_;
        std::shared_ptr<const CapturedDeviceStorageLease> local_capture_, received_capture_;
        /** @brief Submit only after original storage and the exact direct/fork frontier are authenticated.
         * @param stream Exact bound participant stream. @param fork Optional existing acquired producer proof; never a manufactured event. */
        void enqueueOnStream(void *stream,const AcquiredDeviceTransferInput *fork) const;
        CapturedTransferMessage message_{};
        std::optional<DeviceGenerationCollectiveReceipt> receipt_;
        std::shared_ptr<const DeviceCountedAllGatherPacketLayout> packets_; ///< Present only for exact unequal banks.
    };

    /**
     * @brief Shared native metadata for sequential captured allgather invocations.
     *
     * Homogeneous rank-local domains use their original native NCCL/RCCL communicator.
     * Original peer observations remain native topology evidence, never a second route.
     * The caller joins submitted graphs before retiring their bindings/fabric;
     * destruction releases storage without synchronizing inference.
     */
    class DeviceCountedAllGather final : public std::enable_shared_from_this<DeviceCountedAllGather>
    {
    public:
        /** @brief Materialize an already admitted complete domain outside capture.
         * @param authority Sole physical admission/allocation authority.
         * @param original Borrowed original LocalTP; its Rank owner retains it through graph retirement.
         * @param host_device Rank-local first-touch backing resource.
         * @param topology Original authenticated order/routes shared by planning and setup.
         * @param streams Exact setup streams, one per device.
         * @param packet_capacity Maximum physical payload for every directed edge.
         * @throws std::invalid_argument For inconsistent topology, P2P, or streams. */
        [[nodiscard]] static std::shared_ptr<DeviceCountedAllGather> create(
            PhysicalMemoryAuthority &authority, ILocalTPContext &original, DeviceId host_device,
            const DeviceCountedAllGatherTopology &topology,
            std::span<void *const> streams, std::size_t packet_capacity);

        /** @brief Seed this producer's exact live extent during cold setup between retired graph families.
         * @param participant Original local producer coordinate, never a GPU ordinal.
         * @param bytes Live payload extent, including zero, bounded by the admitted packet capacity.
         * @param stream Original exact setup stream retained at creation.
         * @throws std::exception For foreign ownership, capture, bounds or failed upload completion.
         * This initializes device input; replay still uses native DevicePrefix and completion validation.
         */
        void initializeLocalExtent(std::size_t participant,std::uint64_t bytes,void *stream);

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
            CapturedTransferMessage message,
            std::shared_ptr<ITensor> local, std::shared_ptr<ITensor> received);
        /** @brief Bind source-specific physical packet banks with no receiver padding or self slot. */
        [[nodiscard]] DeviceCountedAllGatherBinding bindPackets(std::uint64_t messageKey,
            std::shared_ptr<const DeviceCountedAllGatherPacketLayout> packets,
            std::shared_ptr<ITensor> local, std::shared_ptr<ITensor> received,
            std::optional<DeviceGenerationCollectiveReceipt> receipt = {});
        /** @brief Inspect the actual admitted native transport, not a proposed hardware edge.
         * @param producer Exact communicator producer coordinate.
         * @param consumer Distinct exact consumer coordinate.
         * @return Original native transport; no runtime query or alternate path.
         * @throws std::out_of_range For a foreign or diagonal coordinate. */
        [[nodiscard]] DeviceCountedAllGatherTransport payloadStorage(
            std::size_t producer, std::size_t consumer) const;
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
        /** @brief Retain exact local and receive banks for either declared packet geometry. */
        DeviceCountedAllGatherBinding bindBanks(std::size_t participant, CapturedTransferMessage message,
            std::shared_ptr<ITensor> local, std::shared_ptr<ITensor> received,
            std::shared_ptr<const DeviceCountedAllGatherPacketLayout> packets);
        /** @brief Extent storage retires before its canonical physical claim. */
        struct Extents
        {
            PhysicalMemoryAllocationLease claim;
            std::shared_ptr<DeviceTransferBuffer> storage;
        };
        std::vector<void*> streams_; ///< Original setup streams; cold extent publication cannot substitute another stream.
        std::size_t capacity_ = 0;
        std::vector<DeviceId> devices_;
        std::vector<Extents> extents_;
        std::vector<NativeCompletionAuthority> authorities_; ///< Immutable original communicator identity, retained after host abort admission closes.
        NativeLocalTPContextBorrow original_; ///< Owner-issued revocation applies to installed and direct-created fabrics alike.
        /** @return The same original communicator, rejecting a retired borrow. */
        NativeLocalTPContextUse original() const;
    public:
        /** @brief Reject installation into another equal-membership communicator.
         * @param expected Actual original owner, compared by issued lifetime authority rather than device names. */
        void requireOriginalContext(const ILocalTPContext &expected)const;
        /** @return Terminal-only actual outgoing useful-byte counter for one root broadcast. */
        const unsigned long long *payloadBytesAddress(std::size_t participant, std::size_t source) const;
        /** @return Terminal-only original native receipt; callers must first join the complete graph. */
        const ncclLlaminarByteCompletion *completionAddress(std::size_t participant, std::size_t source) const;
    };
}
