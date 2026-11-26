/**
 * @file EffectiveUseOrdering.cpp
 * @brief Membership-sized effective-use journal implementation.
 */
#include "EffectiveUseOrdering.h"

namespace llaminar2
{
EffectiveUseStatus EffectiveUseJournal::touch(EffectiveUseReceipt receipt) noexcept
{
    if (receipt.sequence == 0)
        return EffectiveUseStatus::Obsolete; // Fresh ordinals are nonzero.
    std::size_t free_slot = capacity_;
    for (std::size_t i = 0; i < capacity_; ++i)
    {
        auto &entry = entries_[i];
        if (!entry.live)
        {
            if (free_slot == capacity_) free_slot = i;
            continue;
        }
        if (entry.receipt.replica_incarnation != receipt.replica_incarnation)
            continue;
        // Same incarnation: an EXACT duplicate replays idempotently; a newer
        // ordinal refreshes (reclaims the older entry in place); an older
        // ordinal is obsolete and never refreshes a replacement.
        if (entry.receipt.sequence == receipt.sequence)
            return EffectiveUseStatus::Duplicate;
        if (entry.receipt.sequence > receipt.sequence)
            return EffectiveUseStatus::Obsolete;
        // Refresh: replace the entry in place.
        const bool was_pinned = entry.pinned;
        entry.receipt = receipt;
        entry.pinned = true;
        if (!was_pinned) ++pending_;
        if (receipt.sequence > watermark_) watermark_ = receipt.sequence;
        return EffectiveUseStatus::Refreshed;
    }
    if (free_slot == capacity_)
        return EffectiveUseStatus::JournalFull; // All live incarnations.
    entries_[free_slot] = Entry{receipt, true, true};
    ++pending_;
    if (receipt.sequence > watermark_) watermark_ = receipt.sequence;
    return EffectiveUseStatus::Accepted;
}

bool EffectiveUseJournal::acknowledge(EffectiveUseReceipt receipt) noexcept
{
    for (std::size_t i = 0; i < capacity_; ++i)
    {
        auto &entry = entries_[i];
        if (!entry.live || entry.receipt.replica_incarnation != receipt.replica_incarnation)
            continue;
        // Exact-ordinal match only: a delayed acknowledgement for an OLD
        // ordinal must never release a NEWER re-pinned touch.
        if (entry.receipt.sequence != receipt.sequence)
            return false;
        if (entry.pinned)
        {
            entry.pinned = false;
            --pending_;
            return true;
        }
        return true; // Already acknowledged: idempotent.
    }
    return false; // Unknown/foreign incarnation.
}

bool EffectiveUseJournal::retire(std::uint64_t incarnation) noexcept
{
    for (std::size_t i = 0; i < capacity_; ++i)
    {
        auto &entry = entries_[i];
        if (!entry.live || entry.receipt.replica_incarnation != incarnation)
            continue;
        if (entry.pinned) --pending_;
        entry = Entry{}; // Reclaim: capacity returns for the next incarnation.
        return true;
    }
    return false; // Nothing live for this incarnation (idempotent retire).
}

bool EffectiveUseJournal::mayEvict(std::uint64_t incarnation) const noexcept
{
    for (std::size_t i = 0; i < capacity_; ++i)
    {
        const auto &entry = entries_[i];
        if (!entry.live || entry.receipt.replica_incarnation != incarnation)
            continue;
        return !entry.pinned; // Accepted and acknowledged: evictable.
    }
    return false; // Unaccepted incarnation: never evictable through here.
}
} // namespace llaminar2
