/**
 * @file DiskExtentQuota.cpp
 * @brief Finite quota ledger implementation with exact retire semantics.
 */
#include "DiskExtentQuota.h"
#include <stdexcept>

namespace llaminar2
{
DiskExtentQuota::DiskExtentQuota(std::size_t quota_bytes, std::size_t journal_capacity)
    : quota_(quota_bytes), entries_(journal_capacity)
{
    // A zero-capacity journal could never accept an extent but also silently
    // change behavior: the admitted membership must be explicit and positive.
    if (journal_capacity == 0)
        throw std::invalid_argument("disk extent quota requires a positive admitted journal membership");
}

bool DiskExtentQuota::admit(DiskExtentKey extent)
{
    // CHECKED finite arithmetic: `quota_ - extent.bytes` must never
    // underflow, so refuse whenever the extent alone exceeds the quota
    // BEFORE the subtraction (0 <= quota_ < extent.bytes would otherwise
    // wrap to a huge value and admit extents beyond the bound).
    if (!extent.bytes || contains(extent) || extent.bytes > quota_ ||
        reserved_ > quota_ - extent.bytes)
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
