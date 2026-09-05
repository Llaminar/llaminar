/**
 * @file RamPrefixStorageBackend.h
 * @brief Capacity-tier prefix storage in pageable CPU or pinned GPU-host RAM.
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

    /**
     * @brief Bounded rank-local RAM tier for durable prefix payloads.
     *
     * Production construction reserves the complete configured capacity from
     * `PrefixHostTier` once. Every pageable or pinned backing allocation then
     * carries a child lease in its PrefixBlockHandle, so an asynchronous
     * consumer can outlive cache eviction or backend teardown without making
     * the accounting authority report those bytes as free prematurely.
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
         * @param producer_device CPU or GPU whose backend supplies pinned RAM.
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

        /** @return Whether the logical tier can accept @p bytes immediately. */
        bool canStore(size_t bytes) const override;

        /**
         * @brief Allocate every serialized section and bind its lifetime lease.
         */
        PrefixBlockHandle allocate(const PrefixCacheKey &key,
                                   const PrefixPayloadLayout &layout) override;

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
        /**
         * @brief Allocate one zeroed payload section in the correct RAM class.
         *
         * GPU archives hard-require backend-pinned memory. There is no pageable
         * fallback because that would silently turn supposedly asynchronous
         * DMA into a blocking runtime staging operation.
         */
        bool allocateSection(
            size_t bytes,
            std::shared_ptr<std::vector<uint8_t>> *pageable_owner,
            std::shared_ptr<void> *pinned_owner,
            void **payload,
            const std::shared_ptr<PrefixPayloadReadiness> &readiness) const;

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
        std::unordered_map<PrefixCacheKey, size_t, PrefixCacheKeyHasher> allocations_;
        PhysicalMemoryOwnerReservation reservation_;
    };

} // namespace llaminar2
