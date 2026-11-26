/**
 * @file EffectiveUseOrdering.h
 * @brief Effective-use receipts, membership-sized journal and eviction gate.
 *
 * Design §8.2: define effective use as successful compatible attachment, bind
 * the accepted touch to sealed content/version and replica incarnation, and
 * watermark accepted touches before any eviction decision. A bounded retention
 * pin/fence protects an unacknowledged accepted touch; once acknowledged,
 * ordinary recency governs. Obsolete touches never refresh a replacement, and
 * every fresh successful attachment mints a NEW ordinal (no-copy hits refresh).
 *
 * The journal is sized from ADMITTED LIVE MEMBERSHIP (one entry per replica
 * incarnation), not a magic constant: entries are replaced on refresh and
 * reclaimed on retirement, so the journal never accumulates an unbounded
 * lifetime history and never needs to throw as a lifetime limit. Capacity
 * refusal is TYPED (a status), never a silent drop.
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
    Accepted,       ///< New accepted touch (watermark advanced).
    Refreshed,      ///< Newer touch on an already-live incarnation (watermark advanced).
    Duplicate,      ///< Exact same ordinal replayed: idempotent, no advance.
    Obsolete,       ///< Older ordinal on a superseded incarnation: refused.
    JournalFull,    ///< Bounded journal is full (all live): refused typed.
};

/**
 * @brief Membership-sized effective-use journal with incarnation reclamation.
 *
 * One entry per replica incarnation; a newer touch replaces the older entry
 * (reclamation by refresh), a retired incarnation releases its entry, and a
 * full journal is reported typed. The sequence is the fresh attachment
 * ordinal minted by the owner, so duplicate/obsolete receipts are recognized
 * by exact ordinal and delayed acknowledgements cannot release a newer pin.
 *
 * The journal is NON-OWNING: its entry storage is admitted by the owner
 * through PhysicalMemoryAuthority (the same authority prices the table's
 * other metadata), so no hot-start allocation and no second accounting
 * ledger exist here.
 */
class EffectiveUseJournal final
{
public:
    /** @brief One journal slot; storage is owner-admitted. */
    struct Entry
    {
        EffectiveUseReceipt receipt;  ///< Full identity of the newest touch.
        std::size_t pinned_ = 0;      ///< Unacknowledged touches in the frontier.
        std::size_t obligations_ = 0; ///< Total obligations effectively released once.
        bool live = false;
    };
    /** @brief Empty journal; the owner must bind admitted storage before use. */
    EffectiveUseJournal() noexcept = default;
    /** @brief Bind journal storage admitted by the owner (@p entries, exact capacity). */
    explicit EffectiveUseJournal(std::span<Entry> entries) noexcept
        : entries_(entries.data()), capacity_(entries.size() ? entries.size() : 1)
    {
    }
    EffectiveUseJournal(const EffectiveUseJournal &) = delete;
    EffectiveUseJournal &operator=(const EffectiveUseJournal &) = delete;
    /** @brief Rebind admitted storage (owner-two-phase construction only). */
    void rebind(std::span<Entry> entries) noexcept
    {
        entries_ = entries.data();
        capacity_ = entries.size() ? entries.size() : 1;
        pending_ = 0;
        watermark_ = 0;
    }

    /** @brief Touch one incarnation; typed outcome; pins on new/refreshed. */
    EffectiveUseStatus touch(EffectiveUseReceipt receipt) noexcept;
    /** @brief Ordering point: release the pin of the EXACT touch ordinal. */
    bool acknowledge(EffectiveUseReceipt receipt) noexcept;
    /** @brief Retirement: reclaim the incarnation's entry (update its age). */
    bool retire(std::uint64_t incarnation) noexcept;
    /** @return Whether @p incarnation may be evicted (no pending pin). */
    bool mayEvict(std::uint64_t incarnation) const noexcept;
    /** @return Highest accepted ordinal; eviction may not precede it. */
    std::uint64_t watermark() const noexcept { return watermark_; }
    /** @return Whether any incarnation holds an unacknowledged pin. */
    bool hasPending() const noexcept { return pending_ != 0; }
    /** @return Admitted membership capacity. */
    std::size_t capacity() const noexcept { return capacity_; }
private:
    Entry *entries_ = nullptr;
    std::size_t capacity_;
    std::size_t pending_ = 0;
    std::uint64_t watermark_ = 0;
};
} // namespace llaminar2
