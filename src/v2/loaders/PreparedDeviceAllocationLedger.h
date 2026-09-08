/**
 * @file PreparedDeviceAllocationLedger.h
 * @brief Exact weak-ownership ledger for persistent prepared GPU allocations.
 *
 * A model can create several pooled GPU allocations while preparing dense and
 * routed weights. Logical kernels alias those pools, and Dynamic ExpertOverlay
 * may additionally create runner-owned shadow pools that disappear at teardown.
 * This ledger records each model preparation allocation once without extending
 * its lifetime. A reusable-context seal can therefore sum only owners that are
 * still live after runner teardown, independent of allocator telemetry.
 */

#pragma once

#include "../backends/DeviceId.h"

#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace llaminar2
{
    /**
     * @brief Model-owned registry of persistent prepared-device allocations.
     *
     * Registration identity is the shared-pointer control block, not the raw
     * address. This remains collision-free if an allocator later reuses an
     * object's address. Weak entries never keep abandoned or runner-only pools
     * alive, while every prepared kernel retains the true allocation owner.
     */
    class PreparedDeviceAllocationLedger final
    {
    public:
        PreparedDeviceAllocationLedger() = default;
        PreparedDeviceAllocationLedger(
            const PreparedDeviceAllocationLedger &) = delete;
        PreparedDeviceAllocationLedger &operator=(
            const PreparedDeviceAllocationLedger &) = delete;

        /**
         * @brief Register one finalized persistent allocation.
         * @param device Exact GPU backend and ordinal.
         * @param owner Shared allocation owner retained by kernel views.
         * @param bytes Persistent bytes after all staging regions are released.
         *
         * Re-registering the same control block with identical facts is
         * idempotent. A conflicting device or byte count is a fatal ownership
         * defect rather than a reason to guess which value is authoritative.
         */
        void registerAllocation(
            DeviceId device,
            const std::shared_ptr<void> &owner,
            size_t bytes)
        {
            if (!device.is_gpu() || !owner || bytes == 0u)
            {
                throw std::invalid_argument(
                    "Prepared device allocation requires an exact GPU, owner, and positive byte size");
            }

            std::lock_guard<std::mutex> lock(mutex_);
            for (const auto &entry : entries_)
            {
                if (!sameOwner(entry.owner, owner))
                    continue;
                if (entry.device != device || entry.bytes != bytes)
                {
                    throw std::logic_error(
                        "Prepared device allocation owner was registered with conflicting device or byte geometry");
                }
                return;
            }
            entries_.push_back({
                .device = device,
                .bytes = bytes,
                .owner = owner,
            });
        }

        /**
         * @brief Sum allocations whose exact owners remain live on one GPU.
         * @param device Exact GPU backend and ordinal.
         * @return Checked byte total across unique live control blocks.
         */
        [[nodiscard]] size_t liveBytes(DeviceId device) const
        {
            if (!device.is_gpu())
            {
                throw std::invalid_argument(
                    "Prepared device byte accounting requires an exact GPU");
            }

            std::lock_guard<std::mutex> lock(mutex_);
            size_t bytes = 0u;
            for (const auto &entry : entries_)
            {
                if (entry.device != device || entry.owner.expired())
                    continue;
                if (entry.bytes >
                    std::numeric_limits<size_t>::max() - bytes)
                {
                    throw std::overflow_error(
                        "Prepared device allocation BOM overflows size_t");
                }
                bytes += entry.bytes;
            }
            return bytes;
        }

    private:
        /** @brief One weak, immutable allocation fact. */
        struct Entry
        {
            DeviceId device = DeviceId::invalid();
            size_t bytes = 0u;
            std::weak_ptr<void> owner;
        };

        /** @return Whether weak and strong handles share one control block. */
        static bool sameOwner(
            const std::weak_ptr<void> &left,
            const std::shared_ptr<void> &right) noexcept
        {
            return !left.owner_before(right) &&
                   !right.owner_before(left);
        }

        mutable std::mutex mutex_;
        std::vector<Entry> entries_;
    };
} // namespace llaminar2
