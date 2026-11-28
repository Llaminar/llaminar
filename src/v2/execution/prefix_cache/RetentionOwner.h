/**
 * @file RetentionOwner.h
 * @brief §8.4 client retention intent and tier ordering (one authority).
 *
 * Design §8.4 + plan P3: "Implement the design §8.4 tier ordering in the
 * same retention owner: typed intent (namespace, class, expiry) in bounded
 * admitted metadata; unexpired class, then value, then stable tie-break;
 * expiry refreshed only by effective use; shared pages at the highest
 * referencing class and moved once; tail-first, deficit-only demotion that
 * cascades only on real overflow and discards only from the lowest enabled
 * tier. Live state and protected continuations outrank every class."
 *
 * THIS ownership unit covers the RETENTION ORDERING authority: typed class
 * map (code-owned, one source), bounded admitted per-entry intent metadata,
 * ordinal expiry (minimum lifetime measured from the request start that wrote
 * or last effectively used the entry), effective-use-only refresh, the
 * three-level victim order (unexpired class longest first, then the §8.2
 * value score, then the stable older-effective-use sequence tie-break),
 * deficit-only selection restricted to idle entries (live/protected state
 * outrank every class and are never selected), and the shared-page class
 * aggregation (highest unexpired class of every referencing entry).
 *
 * The owner performs NO movement, NO page/byte accounting and NO quota: the
 * tier driver (composed route, later M) consumes the ordered victims and
 * applies tail-first/deficit-only cascade semantics; PMA and the store remain
 * the physical owners. Victim ordering is deterministic and device-free.
 */
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace llaminar2
{
/** @brief Ordered retention classes (ascending retention rank). */
enum class RetentionClass : std::uint32_t
{
    InMemory = 0, ///< Explicit short-lived scratch (P0 class rank 0).
    Default = 1,  ///< Absent-value class, value-score governed (rank between).
    TieredDay = 2 ///< 24h minimum lifetime contract (P0 class rank 2).
};

/** @brief §8.2 value score in one typed, totally-ordered struct.
 *
 * The documented total order used for victim ranking: higher score = more
 * valuable. Comparison is lexicographic on (recency, frequency, reuse
 * likelihood, avoided compute, footprint asc, restore cost ASC) — a larger
 * footprint reduces value, a smaller measured restore cost increases it. */
struct RetentionValueScore
{
    std::uint64_t recency = 0;
    std::uint64_t frequency = 0;
    std::uint64_t reuse_likelihood = 0;
    std::uint64_t avoided_compute = 0;
    std::uint64_t footprint = 0;
    std::uint64_t restore_cost = 0;
    bool operator<(const RetentionValueScore &o) const noexcept
    {
        if (recency != o.recency) return recency < o.recency;
        if (frequency != o.frequency) return frequency < o.frequency;
        if (reuse_likelihood != o.reuse_likelihood) return reuse_likelihood < o.reuse_likelihood;
        if (avoided_compute != o.avoided_compute) return avoided_compute < o.avoided_compute;
        if (footprint != o.footprint) return footprint < o.footprint;
        return restore_cost < o.restore_cost;
    }
    bool operator==(const RetentionValueScore &o) const noexcept = default;
};

/** @brief One §8.4 typed retention intent (class + minimum lifetime). */
struct RetentionIntent
{
    RetentionClass cls = RetentionClass::Default;
    /** @brief Minimum lifetime in ordinals from the writing request start;
     * zero = "no strong retention promise" (value-score governed only). */
    std::uint64_t min_lifetime_ordinals = 0;
};

/** @brief One entry's identity within a retention namespace. */
struct RetentionEntryKey
{
    std::uint64_t namespace_id = 0;
    std::uint64_t entry_id = 0;
    bool operator==(const RetentionEntryKey &) const = default;
};

/**
 * @brief Code-owned standard-value-to-class map (§8.4/P0 baseline).
 *
 * One source for the accepted values and their classes/lifetimes. A
 * recognized field with an unsupported value is REFUSED with a precise
 * diagnostic (never silently mapped); an absent value selects the code-owned
 * default class. No vendor extension is interpreted.
 */
class RetentionClassMap final
{
public:
    /**
     * @return The typed intent for a standard request value (absent == empty
     * string) or empty with a precise diagnostic for an unsupported value.
     */
    static std::optional<RetentionIntent> forValue(std::string_view value,
        std::string *refusal = nullptr);
};

/**
 * @brief Bounded admitted §8.4 retention metadata + deterministic victim order.
 *
 * The metadata is bounded admitted state (capacity, refuse beyond — never an
 * unbounded key map) priced by the caller in the BOM. Entries are either
 * protected/live (running requests, protected live continuations) or idle;
 * the owner NEVER selects a protected entry as a victim. Expiry uses
 * monotonic ORDINALS (deterministic, no wall clock): an entry expires at
 * write_ordinal + min_lifetime; an expired entry keeps its value score but
 * loses its class. Refresh happens ONLY on effective use (§8.2 ordered
 * receipt); a provisional peek never refreshes.
 */
class RetentionOwner final
{
public:
    /**
     * @param metadata_capacity Admitted entry-record membership (bounded).
     * @throws std::invalid_argument For a zero capacity (unrepresentable).
     */
    explicit RetentionOwner(std::size_t metadata_capacity);

    /**
     * @brief Admit one entry's typed retention intent as bounded metadata.
     * @param key Exact namespace/entry identity (no duplicate admission).
     * @param intent Typed class + minimum lifetime.
     * @param write_ordinal Ordinal at the writing request start.
     * @param value Value score (kept past expiry).
     * @param live Start-protected (running request writes are live).
     * @return True when admitted; false without mutation when the identity is
     * duplicate or the bounded metadata is full.
     */
    bool admit(RetentionEntryKey key, RetentionIntent intent,
        std::uint64_t write_ordinal, RetentionValueScore value,
        bool live = true);

    /** @brief §8.2 effective use: refresh the expiry under the same ordered
     * receipt (moves write_ordinal to the use ordinal). No-op for a
     * never-admitted key. @return True when the entry exists and refreshed. */
    bool refresh(RetentionEntryKey key, std::uint64_t use_ordinal);

    /** @brief Provisional inspection: NEVER refreshes; fetches the owner's
     * intent/score state without changing any memory. */
    std::optional<RetentionIntent> peek(RetentionEntryKey key,
        RetentionValueScore *score = nullptr) const;

    /** @brief Whether the entry is expired at the current ordinal (class is
     * lost when expired; the value score is retained). */
    bool expired(RetentionEntryKey key, std::uint64_t now) const;

    /** @brief Mark an entry live (protected) or idle (victimizable). */
    bool setLive(RetentionEntryKey key, bool live);

    /** @brief Remove one entry when the tier driver discards it. */
    bool remove(RetentionEntryKey key);

    /**
     * @brief Select at most `deficit` victims by the §8.4 order.
     *
     * Deficit-only: never more than the requested deficit; fewer only when
     * the idle set is smaller. Victim rank: (0) idle before protected (never
     * selected), (1) expired entries first (class lost, value kept), then
     * UNEXPIRED lowest class rank first (InMemory before Default before
     * TieredDay), (2) within one class lower value score first, (3) stable
     * tie-break OLDER effective-use sequence first. Live/protected entries
     * are never in the result.
     */
    std::vector<RetentionEntryKey> selectVictims(std::uint64_t now,
        std::size_t deficit) const;

    /**
     * @brief Shared-page class aggregation: the highest UNEXPIRED class of
     * every referencing entry (an expired reference never raises the class),
     * i.e. the class the shared page keeps while any reference holds it.
     */
    std::optional<RetentionClass> sharedClass(
        const std::vector<RetentionEntryKey> &referencing,
        std::uint64_t now) const;

    /** @return Admitted entry count (bounded metadata occupancy). */
    std::size_t count() const noexcept { return count_; }
    /** @return Admitted metadata capacity (typed membership, not a constant). */
    std::size_t capacity() const noexcept { return entries_.size(); }

private:
    struct Entry
    {
        RetentionEntryKey key;
        RetentionIntent intent;
        std::uint64_t write_ordinal = 0;
        std::uint64_t effective_use_seq = 0;
        RetentionValueScore value;
        bool live = true;
    };
    std::vector<Entry> entries_; ///< Admitted-membership-sized metadata.
    std::size_t count_ = 0;
    std::uint64_t next_use_seq_ = 0;
};
} // namespace llaminar2
