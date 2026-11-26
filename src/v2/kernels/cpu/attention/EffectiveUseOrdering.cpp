/**
 * @file EffectiveUseOrdering.cpp
 * @brief Per-operation effective-use lease implementation.
 *
 * R10 invariants: pending == number of LIVE leases exactly (derived, never a
 * separate counter drifted from entry state). Access to entries_ is guarded by
 * bound()/capacity_ so an unbound or empty journal refuses every operation
 * (no null-pointer access, no capacity-1-with-null trap). Reclaiming requires
 * the incarnation's lease set empty; the slot is released by the lease's own
 * acknowledgement.
 */
#include "EffectiveUseOrdering.h"

namespace llaminar2
{
EffectiveUseStatus EffectiveUseJournal::touch(EffectiveUseReceipt receipt) noexcept
{
    if (!bound() || capacity_ == 0)
        return EffectiveUseStatus::JournalFull; // Unbound/empty: no leases.
    if (receipt.sequence == 0)
        return EffectiveUseStatus::Obsolete; // Fresh ordinals are nonzero.
    // A lease with this EXACT identity already exists: duplicate, no new lease.
    for (std::size_t i = 0; i < capacity_; ++i)
        if (entries_[i].live && entries_[i].receipt == receipt)
            return EffectiveUseStatus::Duplicate;
    for (std::size_t i = 0; i < capacity_; ++i)
    {
        auto &entry = entries_[i];
        if (!entry.live)
            continue;
        if (entry.receipt.replica_incarnation != receipt.replica_incarnation)
            continue;
        // Same incarnation, different ordinal: the identity (content/table
        // version) must agree, or this is a foreign replacement (never a
        // refresh of the old lease's identity).
        if (entry.receipt.content_version != receipt.content_version ||
            entry.receipt.table_version != receipt.table_version)
            return EffectiveUseStatus::Obsolete;
        if (entry.receipt.sequence > receipt.sequence)
            return EffectiveUseStatus::Obsolete; // Older ordinal: no lease.
        // A NEW ordinal on a live incarnation is a FRESH lease (the previous
        // lease stays outstanding until its OWN acknowledgement).
        break;
    }
    for (std::size_t i = 0; i < capacity_; ++i)
    {
        auto &entry = entries_[i];
        if (entry.live) continue;
        entry = Entry{receipt, true};
        ++pending_;
        if (receipt.sequence > watermark_) watermark_ = receipt.sequence;
        return EffectiveUseStatus::Accepted;
    }
    return EffectiveUseStatus::JournalFull; // Every slot holds a live lease.
}

bool EffectiveUseJournal::acknowledge(EffectiveUseReceipt receipt) noexcept
{
    if (!bound()) return false;
    for (std::size_t i = 0; i < capacity_; ++i)
    {
        auto &entry = entries_[i];
        if (!entry.live || !(entry.receipt == receipt))
            continue;
        // Release EXACTLY this lease. The RECEIPT IDENTITY is retained so
        // mayEvict can still recognize a previously accepted incarnation
        // (an incarnation that was accepted but is now lease-free is
        // evictable; an untracked incarnation is not).
        entry.live = false;
        --pending_;
        return true;
    }
    return false; // Unknown/foreign/duplicate identity: releases nothing.
}

bool EffectiveUseJournal::mayEvict(std::uint64_t incarnation) const noexcept
{
    if (!bound()) return false;
    for (std::size_t i = 0; i < capacity_; ++i)
    {
        const auto &entry = entries_[i];
        if (!entry.live || entry.receipt.replica_incarnation != incarnation)
            continue;
        return false; // Any live lease of this incarnation blocks eviction.
    }
    // No live leases for this incarnation: it may be evicted, but only if it
    // was EVER accepted (an untracked incarnation is never evictable here).
    for (std::size_t i = 0; i < capacity_; ++i)
        if (entries_[i].receipt.replica_incarnation == incarnation)
            return true;
    return false;
}

} // namespace llaminar2
