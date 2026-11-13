/**
 * @file ReleasedWeightLoadStaging.h
 * @brief Immutable setup receipt for a drained ordinary weight-upload slot.
 *
 * This value contains geometry and ownership, never capacity observations or
 * execution state. Only LoadOrchestrator can mint it after freeing its pinned
 * and device staging and releasing their PMA claims. Reallocation revokes the
 * producer lifetime identity. A resident native table may reuse one admitted
 * host slot, never derive a new staging size from the complete table extent.
 */
#pragma once
#include "planning/PhysicalMemoryAuthority.h"
#include <cstddef>
#include <memory>
#include <stdexcept>
namespace llaminar2
{
class LoadOrchestrator;
/** @brief Checked evidence of the original loader's resolved, now released slot. */
class ReleasedWeightLoadStaging final
{
public:
    /** @return Original GPU whose weight loader resolved this geometry. */
    DeviceId device() const noexcept { return device_; }
    /** @return Exact admitted host resource used by the original pinned ring. */
    DeviceId hostDevice() const noexcept { return host_device_; }
    /** @return One original logical pinned-slot capacity, never table capacity. */
    std::size_t slotBytes() const noexcept { return slot_bytes_; }
    /** @brief Reject foreign authority, expired producer or an unrepresentable native row. */
    void requireFor(const std::shared_ptr<PhysicalMemoryAuthority> &authority,
                    DeviceId device, std::size_t row_bytes) const
    {
        if (!authority || authority.get() != authority_.get() || device != device_ ||
            !device.is_gpu() || !host_device_.is_cpu() || !row_bytes ||
            row_bytes > slot_bytes_ || released_identity_.expired())
            throw std::logic_error("Native table upload requires its drained ordinary-loader staging receipt");
    }
private:
    friend class LoadOrchestrator;
    /** @brief Only the actual finalized load owner may authenticate a released slot. */
    ReleasedWeightLoadStaging(std::shared_ptr<PhysicalMemoryAuthority> authority,
        DeviceId device, DeviceId host_device, std::size_t slot_bytes,
        std::weak_ptr<const void> identity)
        : authority_(std::move(authority)), device_(device), host_device_(host_device),
          slot_bytes_(slot_bytes), released_identity_(std::move(identity)) {}
    std::shared_ptr<PhysicalMemoryAuthority> authority_;
    DeviceId device_, host_device_;
    std::size_t slot_bytes_;
    std::weak_ptr<const void> released_identity_;
};
}
