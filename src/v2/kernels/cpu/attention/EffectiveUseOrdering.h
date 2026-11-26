/**
 * @file EffectiveUseOrdering.h
 * @brief Per-operation effective-use LEASES with exact consumer ordering.
 *
 * Design §8.2: define effective use as successful compatible attachment, bind
 * the accepted touch to sealed content/version and replica incarnation, and
 * watermark accepted touches before any eviction decision. A bounded retention
 * pin/fence protects an unacknowledged accepted touch; once acknowledged,
 * ordinary recency governs.
 *
 * R10 model: EVERY touch is one independent lease with its own ordinal. A lease
 * is released ONLY by an acknowledgement carrying the EXACT full identity
 * (content/table version, replica incarnation, ordinal). There is no
 * cumulative-frontier shortcut: ack of lease N never proves lease M<N
 * finished, so it can never clear another consumer's obligation. Eviction is
 * permitted only when the incarnation has ZERO live leases. Retirement refuses
 * while any lease is live and otherwise reclaims the slot. The journal is
 * NON-OWNING: storage is admitted by the owner (PMA), never allocated here.
 * A receipt released by its only acknowledgement is automatically reclaimed.
 */
#pragma once
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

namespace llaminar2
{
/** @brief One accepted effective-use touch, incarnation-bound with a fresh ordinal. */
struct EffectiveUseReceipt final
{
    std::uint64_t content_version = 0;
    std::uint64_t table_version = 0;
    std::uint64_t replica_incarnation = 0;
    std::uint64_t sequence = 0; ///< Fresh successful-attachment ordinal, never zero.
    bool operator==(const EffectiveUseReceipt &) const = default;
};

/** @brief Typed touch outcome (never a silent drop). */
enum class EffectiveUseStatus
{
    Accepted,       ///< New accepted touch (issued a lease).
    Duplicate,      ///< Exact same ordinal replayed: no new lease.
    Obsolete,       ///< Older ordinal on a superseded identity: refused (no lease).
    JournalFull,    ///< Bounded journal is full (all leases live): refused typed.
};

/**
 * @brief Membership-sized per-operation lease journal.
 *
 * One slot per live lease. touch() issues a lease; acknowledge() releases the
 * EXACT lease (full identity + ordinal). Duplicate touch ordinals never issue
 * a second lease; older ordinals on the same incarnation are obsolete (a new
 * touch on a LIVE incarnation uses a fresh ordinal, which is a separate lease;
 * the old lease remains outstanding until its own acknowledgement). retired
 * incarnations release their slots only after every lease acknowledged.
 * The entry array is obtained by the owner; an empty/unbound journal refuses
 * every operation (no null access, no capacity1-with-null trap).
 */
class EffectiveUseJournal final
{
public:
    /** @brief One journal slot; storage is owner-admitted. */
    struct Entry
    {
        EffectiveUseReceipt receipt;  ///< Exact identity of this lease.
        bool live = false;            ///< Slot holds an outstanding lease.
    };
    /** @brief Empty/unbound journal: every operation refuses typed. */
    EffectiveUseJournal() noexcept = default;
    /** @brief Bind journal storage admitted by the owner (@p entries, nonzero). */
    explicit EffectiveUseJournal(std::span<Entry> entries) noexcept
        : entries_(entries.data()), capacity_(entries.size())
    {
        // CLEAN the extent: dirty/foreign entries must never be interpreted as
        // live leases (R10 dirty-storage control).
        for (std::size_t i = 0; i < entries.size(); ++i)
            entries[i] = Entry{};
    }
    /** @brief Bind admitted storage once, only while NO lease is live. */
    bool rebind(std::span<Entry> entries) noexcept
    {
        if (pending_ != 0)
            return false; // Active journal: forbid rebinding live leases.
        for (std::size_t i = 0; i < entries.size(); ++i)
            entries[i] = Entry{}; // Reclaimed storage is clean before use.
        entries_ = entries.data();
        capacity_ = entries.size();
        watermark_ = 0; // New storage begins a fresh history.
        return true;
    }
    EffectiveUseJournal(const EffectiveUseJournal &) = delete;
    EffectiveUseJournal &operator=(const EffectiveUseJournal &) = delete;
    /** @brief Reveal whether the journal is bound to admitted storage. */
    bool bound() const noexcept { return entries_ != nullptr; }

    /** @brief Issue one lease or refuse typed (no silent drop, no null access). */
    EffectiveUseStatus touch(EffectiveUseReceipt receipt) noexcept;
    /** @brief Release the EXACT lease; wrong identity releases nothing. */
    bool acknowledge(EffectiveUseReceipt receipt) noexcept;
    /**
     * @brief Retirement ordering point for one incarnation.
     * @return True when the incarnation had no live leases (its slot may
     * already be reclaimed by per-lease acknowledgement); false when a lease
     * is still live. An untracked incarnation returns false.
     * @note The slots are reclaimed by their own acknowledgement; retire is
     * the ordering CHECK (no retirement may precede every acknowledgement).
     */
    bool mayEvict(std::uint64_t incarnation) const noexcept;
    /** @return Highest accepted ordinal; eviction may not precede it. */
    std::uint64_t watermark() const noexcept { return watermark_; }
    /** @return Whether any incarnation holds a live lease. */
    bool hasPending() const noexcept { return pending_ != 0; }
    /** @return Admitted lease-slot capacity (0 when unbound). */
    std::size_t capacity() const noexcept { return capacity_; }
private:
    Entry *entries_ = nullptr;
    std::size_t capacity_ = 0;
    std::size_t pending_ = 0;
    std::uint64_t watermark_ = 0;
};
} // namespace llaminar2
