/**
 * @file EffectiveUseOrdering.cpp
 * @brief Membership-sized effective-use journal implementation.
 *
 * R9 semantics: each live incarnation holds ONE entry whose receipt stores
 * the FULL identity (content/table version, replica incarnation, highest
 * sequence), and a cumulative frontier: every touch ordinal <= the entry's
 * sequence is a live obligation until the caller acknowledging THE SAME FULL
 * IDENTITY (including the exact content/table versions) passes the frontier.
 * A wrong content/table at a matching incarnation+ordinal never acknowledges
 * (foreign receipt), a touch1..touch2 on the same replica leaves touch1
 * pending until ack2 with the same full identity, and retire(pinned) refuses.
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
        // SAME INCARNATION: the full content/table identity must also agree;
        // a different content generation under the same bank identity is a
        // foreign replacement and never refreshes this incarnation's receipt.
        if (entry.receipt.content_version != receipt.content_version ||
            entry.receipt.table_version != receipt.table_version)
            return EffectiveUseStatus::Obsolete;
        if (entry.receipt.sequence == receipt.sequence)
            return EffectiveUseStatus::Duplicate;
        if (entry.receipt.sequence > receipt.sequence)
            return EffectiveUseStatus::Obsolete;
        // Newer touch on the same live content: the frontier advances. The
        // OLDER touches remain obligations until the full frontier is
        // acknowledged (pinned_ until ack with this exact full identity).
        entry.receipt = receipt;
        ++pending_; // Each unacknowledged touch adds one obligation.
        if (receipt.sequence > watermark_) watermark_ = receipt.sequence;
        return EffectiveUseStatus::Refreshed;
    }
    if (free_slot == capacity_)
        return EffectiveUseStatus::JournalFull; // All live incarnations.
    entries_[free_slot] = Entry{receipt, 1, 1, true};
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
        // FULL IDENTITY: content/table versions must match the live entry, or
        // this is a foreign receipt that acknowledges nothing (R9 repro: the
        // wrong content/table at the same incarnation+ordinal must not ack).
        if (entry.receipt.content_version != receipt.content_version ||
            entry.receipt.table_version != receipt.table_version)
            return false;
        // The acknowledgement passes the frontier ONLY at the LATEST ordinal
        // with the exact full identity: a delayed older ordinal must never
        // release newer obligations, and a higher-than-frontier ordinal can
        // never acknowledge (foreign/future touch).
        if (entry.receipt.sequence != receipt.sequence)
            return false;
        if (entry.pinned_)
        {
            entry.pinned_ = 0; // Cumulative frontier acknowledged: release ALL.
            pending_ -= entry.obligations_;
            entry.obligations_ = 0;
            return true;
        }
        return true; // Idempotent ack of an already-acknowledged frontier.
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
        if (entry.pinned_ != 0)
            return false; // TRUE retirement: pending obligations refuse.
        entry = Entry{}; // Fully acknowledged: capacity returns.
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
        return entry.pinned_ == 0; // Whole frontier acknowledged.
    }
    return false; // Unaccepted incarnation: never evictable through here.
}
} // namespace llaminar2
