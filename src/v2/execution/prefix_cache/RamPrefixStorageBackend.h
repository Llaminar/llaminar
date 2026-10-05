/**
 * @file RamPrefixStorageBackend.h
 * @brief Capacity-tier prefix storage with producer-owned payload initialization.
 *
 * GPU recurrent checkpoint sections are filled completely by their asynchronous
 * producer. Clearing them on the CPU first duplicates a potentially large write.
 * Attention padding and optional terminal sections retain zero initialization.
 * The handle's readiness event, not allocation, makes either kind readable.
 */

#pragma once

#include "execution/prefix_cache/PrefixStorageBackend.h"
#include "backends/DeviceId.h"
#include "planning/PhysicalMemoryAuthority.h"

#include <memory>
#include <string>
#include <unordered_map>

namespace llaminar2
{
    class PrefixHostArena;

    /**
     * @brief Bounded rank-local RAM tier for durable prefix payloads.
     *
     * Production construction reserves the complete configured capacity from
     * `PrefixHostTier` once. GPU tiers materialize one persistent pinned arena
     * before inference; handle aliases lease ranges until their producer and
     * all consumers retire. CPU vectors retain individual physical claims.
     * Neither cache eviction nor request reset frees GPU-tier backing.
     */
    class RamPrefixStorageBackend : public IPrefixStorageBackend
    {
    public:
        /**
         * @brief Construct an unaccounted CPU backend for focused unit tests.
         *
         * Production orchestration must use @ref create. This constructor is
         * also used by the disk-backend unit oracle while its transient staging
         * allocation is kept outside a live model topology.
         */
        explicit RamPrefixStorageBackend(size_t budget_bytes);

        /**
         * @brief Construct an unaccounted producer-specific test backend.
         * GPU materialization requires @ref create and canonical admission;
         * this constructor does not establish a usable GPU archive tier.
         * @param producer_device Exact intended producer identity.
         * @param budget_bytes Logical cache capacity.
         */
        RamPrefixStorageBackend(DeviceId producer_device, size_t budget_bytes);

        /**
         * @brief Create the production RAM tier under canonical admission.
         * @param producer_device Device whose copies consume this host storage.
         * @param budget_bytes Exact persistent host-tier capacity to reserve.
         * @param memory_authority Rank-local CPU/GPU allocation authority.
         * @param error Optional deterministic construction diagnostic.
         * @return Accounted backend, or nullptr without a partial reservation.
         */
        [[nodiscard]] static std::shared_ptr<RamPrefixStorageBackend> create(
            DeviceId producer_device,
            size_t budget_bytes,
            std::shared_ptr<PhysicalMemoryAuthority> memory_authority,
            std::string *error = nullptr);

        /** @return Whether logical capacity and PMA both admit @p bytes now. */
        bool canStore(size_t bytes) const override;

        /**
         * @return Logical and physical placement headroom, without blocking.
         * GPU ranges remain unavailable through aliases and unfinished DMA;
         * their persistent backing stays charged to PMA for the arena lifetime.
         */
        size_t availableAllocationBytes() const;

        /**
         * @brief Allocate every serialized section and bind its lifetime lease.
         * GPU recurrent bytes are unspecified until the complete producer write
         * is published; attention and optional terminal storage start zeroed.
         */
        PrefixBlockHandle allocate(const PrefixCacheKey &key,
                                   const PrefixPayloadLayout &layout) override;

        /**
         * @brief Allocate a RAM archive with the exact rejection reason.
         *
         * Prefix harvest is a request-completion boundary. An allocation
         * failure must identify whether its cause was logical capacity,
         * an outstanding physical lease, or backing-memory admission rather
         * than reporting all three as an indistinguishable invalid handle.
         * The ordinary storage interface delegates to this same operation.
         *
         * @param key Immutable archive identity.
         * @param layout Serialized payload geometry.
         * @param error Receives the failure reason, if requested.
         * @return A live archive handle or an invalid handle on rejection.
         */
        PrefixBlockHandle allocateWithDiagnostics(
            const PrefixCacheKey &key,
            const PrefixPayloadLayout &layout,
            std::string *error);

        /**
         * @brief Retire one cache key without invalidating outstanding handles.
         */
        bool release(const PrefixBlockHandle &handle) override;

        /**
         * @brief Attach model-specific host state to an allocated RAM block.
         *
         * Runtime state is serialized by the model graph after the common
         * payload geometry is known. The backend folds those bytes into the
         * same capacity budget and gives them a separate child lease so a
         * device-hot handle may retain exactly this shared host allocation.
         *
         * @param handle Existing live RAM allocation owned by this backend.
         * @param storage Newly serialized model runtime bytes.
         * @return true when capacity and accounting accepted the attachment.
         */
        bool attachModelRuntimeState(
            PrefixBlockHandle *handle,
            std::shared_ptr<std::vector<uint8_t>> storage);

        /** @return Configured logical capacity. */
        size_t budgetBytes() const { return budget_bytes_; }
        /** @return Bytes charged to keys currently installed in this backend. */
        size_t usedBytes() const { return used_bytes_; }
        /** @return Number of currently installed keys. */
        size_t allocationCount() const { return allocations_.size(); }

        /** @return Whether this instance owns production capacity authority. */
        bool accounted() const noexcept { return reservation_.valid(); }

    private:
        /** @brief Cache-key incarnation authenticated by shared payload ownership. */
        struct Allocation
        {
            size_t bytes = 0u;
            std::weak_ptr<void> owner;
            /** @return Whether a handle retains this exact allocation, not a reused address/key. */
            bool matches(const std::shared_ptr<void> &candidate) const noexcept
            {
                return candidate && !owner.expired() &&
                       !owner.owner_before(candidate) && !candidate.owner_before(owner);
            }
        };

        /** @return The first real payload owner, whose control block seals this incarnation. */
        static std::shared_ptr<void> payloadOwner(const PrefixBlockHandle &handle) noexcept;

        /** @brief Whether the archive producer writes every allocated byte. */
        enum class SectionWriteCoverage
        {
            Partial,  ///< Short attention blocks or absent optional terminal rows.
            Complete, ///< A recurrent checkpoint writes its entire serialized image.
        };

        /**
         * @brief Bind one payload section with its exact initialization contract.
         *
         * GPU sections alias one admitted arena range. Binding never calls a
         * native allocator or changes backing identity.
         * Complete GPU sections remain unreadable until their producer fills all
         * bytes and publishes the handle's readiness event. Partial GPU sections
         * are zeroed first so short/omitted payloads cannot expose stale bytes.
         * CPU vector storage retains its ordinary value initialization.
         *
         * @param bytes Serialized capacity of this one section.
         * @param coverage Complete producer overwrite or possibly partial payload.
         * @param pageable_owner Receives CPU-owned vector storage, when applicable.
         * @param pinned_owner Receives GPU-host allocation lifetime ownership.
         * @param payload Receives the stable address, never a readiness guarantee.
         * @param arena_payload Complete leased payload, empty for CPU vectors.
         * @param offset This section's checked offset inside that payload.
         * @return true when backing storage was allocated successfully.
         */
        bool allocateSection(
            size_t bytes,
            SectionWriteCoverage coverage,
            std::shared_ptr<std::vector<uint8_t>> *pageable_owner,
            std::shared_ptr<void> *pinned_owner,
            void **payload,
            const std::shared_ptr<void> &arena_payload,
            size_t offset) const;

        /**
         * @brief Claim one physical allocation from the reserved host tier.
         * @return Opaque shared RAII owner, or empty for an unaccounted test backend.
         * @throws std::logic_error when production capacity is exhausted.
         */
        [[nodiscard]] std::shared_ptr<void> claimHostAllocation(
            size_t bytes) const;

        DeviceId producer_device_ = DeviceId::cpu();
        size_t budget_bytes_ = 0;
        size_t used_bytes_ = 0;
        std::unordered_map<PrefixCacheKey, Allocation, PrefixCacheKeyHasher> allocations_;
        PhysicalMemoryOwnerReservation reservation_;
        std::shared_ptr<PrefixHostArena> arena_;
    };

} // namespace llaminar2
