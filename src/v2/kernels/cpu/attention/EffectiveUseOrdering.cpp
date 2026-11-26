/**
 * @file EffectiveUseOrdering.cpp
 * @brief Bounded idempotent touch journal implementation.
 */
#include "EffectiveUseOrdering.h"
#include <stdexcept>

namespace llaminar2
{
bool EffectiveUseTracker::accept(EffectiveUseReceipt receipt)
{
    if (acknowledged(receipt) != 0) return false; // Idempotent duplicate.
    if (count_ == capacity()) throw std::logic_error("Effective-use journal exceeded its fixed capacity");
    entries_[count_++] = receipt;
    ++watermark_;
    return true;
}

std::uint64_t EffectiveUseTracker::acknowledged(EffectiveUseReceipt receipt) const
{
    std::uint64_t ordinal = 0;
    for (std::size_t i = 0; i < count_; ++i)
    {
        ++ordinal;
        if (entries_[i] == receipt) return ordinal;
    }
    return 0;
}
} // namespace llaminar2
