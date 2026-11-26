/**
 * @file EffectiveUseOrdering.cpp
 * @brief Bounded idempotent touch journal implementation.
 */
#include "EffectiveUseOrdering.h"
#include <stdexcept>

namespace llaminar2
{
void EffectiveUseGate::pin(EffectiveUseReceipt receipt) noexcept
{
    // A pin accompanies a NEW accepted touch. A duplicate while still pinned
    // is idempotent; a touch on the same incarnation AFTER its ordering point
    // is a fresh accepted use and re-pins (obsolete/foreign keys never match
    // because the receipt identity is exact).
    for (std::size_t i = 0; i < count_; ++i)
        if (states_[i].receipt == receipt)
        {
            if (!states_[i].pinned)
            {
                states_[i].pinned = true;
                ++pending_;
            }
            return;
        }
    if (count_ == kCapacity) return; // Fixed journal full: refuse (bounded).
    states_[count_++] = {receipt, true};
    ++pending_;
}

void EffectiveUseGate::acknowledge(EffectiveUseReceipt receipt) noexcept
{
    // Ordering point: release exactly one bounded pin for the accepted touch.
    for (std::size_t i = 0; i < count_; ++i)
        if (states_[i].receipt == receipt && states_[i].pinned)
        {
            states_[i].pinned = false;
            --pending_;
            return;
        }
    // Unknown/foreign/duplicate acknowledgment is idempotent.
}

bool EffectiveUseGate::mayEvict(EffectiveUseReceipt receipt) const noexcept
{
    // Ordained rule: no eviction may race ahead of an accepted touch. A
    // receipt is evictable only when it was accepted AND acknowledged; any
    // still-pinned incarnation blocks the gate (bounded fence).
    if (pending_ != 0) return false;
    for (std::size_t i = 0; i < count_; ++i)
        if (states_[i].receipt == receipt) return !states_[i].pinned;
    return false; // Foreign/obsolete receipt: not evictable through this gate.
}

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
