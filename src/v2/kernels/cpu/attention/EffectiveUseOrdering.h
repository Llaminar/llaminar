/**
 * @file EffectiveUseOrdering.h
 * @brief Bounded effective-use receipts and retention watermark for shared prefixes.
 *
 * Design §8.2: define effective use as successful compatible attachment, bind
 * the accepted touch to sealed content/version and replica incarnation, and
 * watermark accepted touches before any eviction decision. A bounded retention
 * pin/fence protects an unacknowledged accepted touch; once acknowledged,
 * ordinary recency governs. Duplicate touches are idempotent; obsolete touches
 * never refresh a replacement. This is the CPU frame/table subset of the
 * retention-value policy; tier/disk ownership and GPU no-copy receipts remain
 * separate (plan P3/P6).
 */
#pragma once
#include <cstddef>
#include <cstdint>
#include <span>

namespace llaminar2
{
/** @brief One accepted effective-use touch, idempotent and incarnation-bound. */
struct EffectiveUseReceipt final
{
    std::uint64_t content_version = 0;
    std::uint64_t table_version = 0;
    std::uint64_t replica_incarnation = 0;
    std::uint64_t sequence = 0;
    bool operator==(const EffectiveUseReceipt &) const = default;
};

/** @brief Bounded touch journal with acceptance and watermark state. */
class EffectiveUseTracker final
{
public:
    /**
     * @brief Record one idempotent accepted touch for an incarnation.
     * @param receipt Exact sealed content/version/replica identity.
     * @return True when the touch is new (advances the watermark), false for a duplicate.
     * @throws std::logic_error If the tracker exceeded its fixed capacity.
     */
    bool accept(EffectiveUseReceipt receipt);
    /** @return Whether this exact incarnation has been accepted, with its sequence. */
    std::uint64_t acknowledged(EffectiveUseReceipt receipt) const;
    /** @return Highest accepted watermark; eviction may not precede it. */
    std::uint64_t watermark() const noexcept { return watermark_; }
    /** @return Whether the tracker holds an unacknowledged bounded retention pin. */
    bool hasPendingPin() const noexcept { return pending_ != 0; }
    void pinForPendingUse() noexcept { ++pending_; }
    void releasePendingUse() noexcept { if (pending_) --pending_; }
    /** @return Fixed capacity; used for the authority BOM. */
    static constexpr std::size_t capacity() noexcept { return kCapacity; }
private:
    static constexpr std::size_t kCapacity = 64;
    EffectiveUseReceipt entries_[kCapacity] = {};
    std::size_t count_ = 0;
    std::uint64_t watermark_ = 0;
    std::size_t pending_ = 0;
};

/**
 * @brief Eviction ordering gate (§8.2): a pending accepted touch pins its
 * backing until acknowledged; after the ordering point ordinary recency
 * governs. Obsolete or foreign touches never refresh a replacement, so a
 * reordered receipt cannot defeat the watermark.
 */
class EffectiveUseGate final
{
public:
    /** @brief Notice a NEW accepted touch (bounded pin) for one incarnation. */
    void pin(EffectiveUseReceipt receipt) noexcept;
    /** @brief Ordering point: acknowledge the touch; the pin is released. */
    void acknowledge(EffectiveUseReceipt receipt) noexcept;
    /** @return Whether THIS incarnation may be evicted (no pending touch). */
    bool mayEvict(EffectiveUseReceipt receipt) const noexcept;
    /** @return Whether any incarnation still holds a bounded pin. */
    bool hasPending() const noexcept { return pending_ != 0; }
    /** @return Bounded journal capacity; one entry per accepted touch. */
    static constexpr std::size_t capacity() noexcept { return EffectiveUseTracker::capacity(); }
private:
    static constexpr std::size_t kCapacity = EffectiveUseTracker::capacity();
    struct State { EffectiveUseReceipt receipt; bool pinned = false; };
    State states_[kCapacity] = {};
    std::size_t count_ = 0;
    std::size_t pending_ = 0;
};
} // namespace llaminar2
