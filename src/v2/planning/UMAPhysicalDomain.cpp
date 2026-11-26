/**
 * @file UMAPhysicalDomain.cpp
 * @brief Single-charge ledger implementation.
 */
#include "UMAPhysicalDomain.h"

namespace llaminar2
{
bool UMAPhysicalDomainLedger::charge(UMAAllocationIdentity allocation)
{
    if (!allocation.bytes) return false;
    for (std::size_t i = 0; i < count_; ++i)
        if (entries_[i].allocation_id == allocation.allocation_id &&
            entries_[i].domain_id == allocation.domain_id)
            return false; // Alias of an already-charged allocation: no second charge.
    if (count_ == capacity()) return false;
    entries_[count_++] = allocation;
    charged_ += allocation.bytes;
    return true;
}

bool UMAPhysicalDomainLedger::release(UMAAllocationIdentity allocation)
{
    for (std::size_t i = 0; i < count_; ++i)
    {
        if (entries_[i] != allocation) continue;
        entries_[i] = entries_[count_ - 1];
        --count_;
        charged_ -= allocation.bytes;
        return true;
    }
    return false;
}

bool UMAStagingCharge::reserve(std::size_t bytes)
{
    if (!bytes || staged_ > max_ - bytes) return false;
    staged_ += bytes;
    return true;
}

bool UMAStagingCharge::release(std::size_t bytes)
{
    if (!bytes || bytes > staged_) return false;
    staged_ -= bytes;
    return true;
}
} // namespace llaminar2
