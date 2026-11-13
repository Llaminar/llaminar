/**
 * @file CapturedTransferChannel.h
 * @brief Admitted rank-local message storage and immutable capture bindings.
 *
 * TransferEngine alone constructs a channel or enqueues a complete message.
 * The explicitly selected payload slot has two GPU-private cursors. Cold setup observes each exact
 * original initialization stream before publication, creating no extra events. Captured bindings retain those owners, exact tensor/buffer storage,
 * and message geometry; neither binding nor channel exposes live protocol state
 * to the host. Split original-rank endpoints share authenticated node-local pages;
 * peer virtual addresses and GPU contexts never cross that process boundary.
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
    struct ClusterInventory;
    class IMPIContext;

    /** @brief Exact GPU in its original MPI rank namespace; ordinal alone is not identity. */
    struct CapturedTransferRankGPU final
    {
        int rank;
        DeviceId device;
        bool operator==(const CapturedTransferRankGPU &) const = default;
    };
    /** @brief Positive original channel-allocation scope and semantic boundary key, never a request clock. */
    struct CapturedTransferRankChannelId final
    {
        std::uint64_t generation;
        std::uint64_t boundary;
        bool operator==(const CapturedTransferRankChannelId &) const = default;
    };
    /**
     * @brief Sealed original-rank roles and exact physical quote for one node-local channel.
     *
     * The declared host endpoint owns shared backing. Each rank owns only its
     * local GPU cursor. This value contains immutable topology, not another
     * reservation ledger or a process-local alias to the peer GPU.
     */
    class CapturedTransferRankPlan final
    {
    public:
        /** @brief Resolve distinct same-node original GPU owners without touching a backend.
         * @param inventory Existing complete discovery publication.
         * @param id Original enclosing session allocation scope and semantic boundary identity.
         * @param producer Original producer rank and its local GPU.
         * @param consumer Original consumer rank and its local GPU.
         * @param capacity Positive admitted maximum payload bytes.
         * @param host_endpoint Original physical host owner; Producer for forward, Consumer for reverse when EarlierDomain owns both.
         * @return An immutable role/UUID/node binding shared by quote and materialization.
         * @throws std::invalid_argument For unknown, remote-node or foreign GPU ownership.
         * @throws std::overflow_error For an unrepresentable channel allocation. */
        static CapturedTransferRankPlan fromInventory(const ClusterInventory &inventory,
            CapturedTransferRankChannelId id, CapturedTransferRankGPU producer,
            CapturedTransferRankGPU consumer, size_t capacity,
            CapturedTransferEndpoint host_endpoint = CapturedTransferEndpoint::Producer);
        /** @return Original role metadata; never acquires a foreign GPU context. */
        CapturedTransferRankGPU endpoint(CapturedTransferEndpoint role) const;
        /** @return This rank's sole endpoint, or no endpoint on an uninvolved original rank. */
        std::optional<CapturedTransferEndpoint> localRole(int rank) const noexcept;
        /** @return Endpoint that owns the one physical host allocation/first touch. */
        CapturedTransferEndpoint hostEndpoint() const noexcept { return host_endpoint_; }
        /** @return Original host allocation rank; reverse metadata may retain EarlierDomain ownership. */
        int hostRank() const { return endpoint(host_endpoint_).rank; }
        /** @return Original physical node, authenticated independently of MPI ordering. */
        int physicalNode() const noexcept { return node_; }
        /** @return Existing immutable model/boundary identity. */
        CapturedTransferRankChannelId originalId() const noexcept { return id_; }
        /** @return Maximum payload extent, not the bytes sent by a replay. */
        size_t capacityBytes() const noexcept { return capacity_; }
        /** @return Exact original metadata for collective setup agreement, without pointer serialization. */
        std::vector<std::uint8_t> serialize() const;
        /** @brief Quote only this endpoint's physical allocations through the original PMA builder.
         * @param role Actual original local role.
         * @param gpu Original local GPU resource matching that role.
         * @param host Original CPU resource iff role is hostEndpoint(); absent on the other role.
         * @param builder Canonical admission builder, not a second live ledger. */
        void contributeMemory(CapturedTransferEndpoint role, PhysicalMemoryResource gpu,
            std::optional<PhysicalMemoryResource> host, PhysicalMemoryPlanBuilder &builder) const;
        bool operator==(const CapturedTransferRankPlan &) const = default;
    private:
        CapturedTransferRankPlan(CapturedTransferRankChannelId id, CapturedTransferRankGPU producer,
            CapturedTransferRankGPU consumer, int node, size_t capacity, std::array<std::string, 2> uuids, CapturedTransferEndpoint host_endpoint)
            : id_(id), endpoints_{producer, consumer}, node_(node), capacity_(capacity), uuids_(std::move(uuids)), host_endpoint_(host_endpoint) {}
        CapturedTransferRankChannelId id_;
        std::array<CapturedTransferRankGPU, 2> endpoints_;
        int node_;
        size_t capacity_;
        std::array<std::string, 2> uuids_;
        CapturedTransferEndpoint host_endpoint_;
    };

    /**
     * @brief Original enclosing model cohort retaining actual distributed backing until explicit retirement.
     *
     * Channels own only their local GPU aliases/cursor. This session owns the
     * original pages/file and host-owner PMA lease across those channel borrows.
     * It performs no GPU work on creation and creates no communicator/ledger.
     */
    class CapturedTransferRankSession final
    {
    public:
        /** @brief Retain the original existing discovery context for one model cohort.
         * @param original Existing shared owner, never a reconstructed MPI wrapper.
         * @return A setup/retirement owner containing no physical allocation yet. */
        static std::shared_ptr<CapturedTransferRankSession> create(std::shared_ptr<IMPIContext> original);
        /** @brief Fatal if physical owners were abandoned; never performs MPI from destruction. */
        ~CapturedTransferRankSession();
        /**
         * @brief Retire after the enclosing model has destroyed ALL original local graph/alias owners.
         *
         * The existing global consensus first rejects live channel borrows,
         * then proves all original mappings/file descriptors closed before the
         * original allocator releases its one PMA charge. Call from original explicit model
         * shutdown, including failed setup; never a per-request hot operation.
         */
        void retireAfterOriginalGraphRelease();
        /** @brief Reject a reconstructed/foreign MPI ownership control block before cold composition.
         * @param original Same shared original context used to create this session.
         * @throws std::invalid_argument For a foreign, replaced or retired original owner. */
        void requireOriginalContext(const std::shared_ptr<IMPIContext> &original) const;
        /** @return Whether explicit original all-rank retirement has completed. */
        bool retired() const noexcept { return retired_; }
        /** @return Original setup scope minted by the existing allocation-identity authority, never a request clock. */
        std::uint64_t originalGeneration() const noexcept { return generation_; }
    private:
        friend class TransferEngine;
        struct Entry;
        CapturedTransferRankSession(std::shared_ptr<IMPIContext> original, std::uint64_t generation)
            : original_(std::move(original)), generation_(generation) {}
        std::shared_ptr<IMPIContext> original_;
        std::uint64_t generation_;
        std::vector<std::shared_ptr<Entry>> entries_; ///< Actual RAII allocations, never another byte total.
        bool retired_ = false;
    };

    /** @brief Pure allocation geometry contributed to the canonical memory plan. */
    struct CapturedTransferChannelMemory
    {
        size_t mapped_host_bytes = 0; ///< Page-rounded original mapped control and payload.
        size_t cursor_bytes_per_device = 0; ///< Exact private cursor, on each GPU.
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
        /** @return Original remote-role declaration, or null for the original same-process pair. */
        const CapturedTransferRankPlan *rankPlan() const noexcept { return rank_plan_ ? &*rank_plan_ : nullptr; }
        /** @return Exact allocation identity required in the enclosing graph-cache key. */
        [[nodiscard]] CapturedTransferChannelIdentity identity() const noexcept { return identity_; }
        /** @return Frozen physical endpoint identity; never queries live device state.
         * @throws std::invalid_argument for an unknown endpoint role. */
        [[nodiscard]] DeviceId endpointDevice(CapturedTransferEndpoint role) const;
        /** @return Exact physical geometry used both in planning and materialization. */
        [[nodiscard]] static CapturedTransferChannelMemory memoryFor(size_t capacity);

    private:
        friend class TransferEngine;
        friend class CapturedTransferBinding;
        friend class PipelineActivationFanout; ///< Closed cold projection of the original local consumer publication only.
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
        std::array<Endpoint, 2> endpoints_; ///< Only declared local workers own contexts/cursors.
        std::optional<CapturedTransferRankPlan> rank_plan_; ///< Original remote roles; no peer process addresses.
        std::optional<CapturedTransferEndpoint> rank_role_; ///< Sole local role; absent only for the original local pair.
        std::array<std::uint64_t, 3> backing_identity_{}; ///< Actual shared file device/inode/uid; no peer virtual address.
        PhysicalMemoryAllocationLease mapped_lease_; ///< Outlives mapped storage.
        std::shared_ptr<MappedHostTransferRegion> mapped_;
        std::mutex invocation_mutex_; ///< Serializes CPU admission/cancellation only; never native progress.
        CapturedTransferInvocation original_invocation_{};
        bool invocation_active_ = false, terminal_cancelled_ = false;

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
