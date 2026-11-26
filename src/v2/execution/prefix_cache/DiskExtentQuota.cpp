/**
 * @file DiskExtentQuota.cpp
 * @brief Finite quota ledger implementation with exact retire semantics.
 */
#include "DiskExtentQuota.h"

namespace llaminar2
{
bool DiskExtentQuota::admit(DiskExtentKey extent)
{
    // A zero quota admits NOTHING. The subtraction below must never run when
    // the quota is zero: `quota_ - extent.bytes` would underflow to a huge
    // value and admit extents into a ledger with no capacity.
    if (!quota_ || !extent.bytes || contains(extent) || reserved_ > quota_ - extent.bytes)
        return false; // Zero/duplicate admission, or quota exhausted: no mutation.
    if (count_ == capacity()) return false;
    entries_[count_++] = extent;
    reserved_ += extent.bytes;
    return true;
}

bool DiskExtentQuota::retire(DiskExtentKey extent)
{
    for (std::size_t i = 0; i < count_; ++i)
    {
        if (entries_[i] != extent) continue;
        // Swap-remove: identity-exact retirement returns bytes exactly once.
        entries_[i] = entries_[count_ - 1];
        --count_;
        reserved_ -= extent.bytes;
        return true;
    }
    return false;
}

bool DiskExtentQuota::contains(DiskExtentKey extent) const
{
    for (std::size_t i = 0; i < count_; ++i)
        if (entries_[i] == extent) return true;
    return false;
}
} // namespace llaminar2
