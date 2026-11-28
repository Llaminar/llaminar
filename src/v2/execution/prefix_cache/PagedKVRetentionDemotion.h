/**
 * @file PagedKVRetentionDemotion.h
 * @brief §8.4 demotion decision planner: tail-first, deficit-only, move-once.
 *
 * Plan P3: "tail-first, deficit-only demotion that cascades only on real
 * overflow and discards only from the lowest enabled tier ... Shared pages
 * take the highest unexpired class of any referencing entry and move once.
 * Live state and protected continuations outrank every class."
 *
 * THIS module is the DECISION DRIVER over real typed descriptors: given the
 * admitted RAM-tier page chains of idle prefixes (each page is its own extent
 * granularity), the retention owner's victim order and the composed route's
 * disk quota, it produces an exact demotion plan — which prefix, which TAIL
 * page (hit starts at token zero, so a partial demotion keeps the leading
 * pages resident and demotes the tail), the exact extent identity to move,
 * move-once deduplication for shared pages, and the cascade verdict. The
 * planner is deterministic and device-free; the mechanical byte movement
 * (snapshot -> store.put -> binding re-anchor) is the runner-side wiring M.
 *
 * The plan enforces: only the deficit (page-rounded by construction), victims
 * only from the retention owner's ordered idle set (live/protected never
 * selected), no partial plan (all-or-none: if the admitted disk quota cannot
 * hold the selected extent set, the plan refuses with a precise diagnostic),
 * and exactly one entry per PHYSICALLY shared page (deduped on the resolved
 * extent identity: namespace + digest + bytes, never a logical index) —
 * all-or-none output: any refusal publishes NO moves at all.
 */
#pragma once
#include "execution/prefix_cache/PagedKVArchiveStore.h"
#include "execution/prefix_cache/RetentionOwner.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace llaminar2
{
/** @brief One real RAM-tier page chain of an idle prefix. */
struct PagedPrefixTierEntry
{
    std::uint64_t namespace_id = 0;
    std::uint64_t prefix_id = 0;
    /** @brief Owner record in the retention owner (class, value, expiry). */
    RetentionEntryKey owner;
    /** @brief Contiguous page chain, HEAD first; page_index = logical page
     * ordinal. Demotion selects the TAIL pages first. */
    std::vector<std::uint64_t> page_indices;
    // NOTE: page_indices is the chain of page identities (logical ordinals);
    // the bytes live in the RAM pool and the extent identity is computed by
    // the caller at move time (the planner only needs the ordinal set).
};

/** @brief One planned tail-first page move with its exact extent identity. */
struct PagedDemotionMove
{
    std::uint64_t namespace_id = 0;
    std::uint64_t prefix_id = 0;
    std::uint64_t page_index = 0;
    /** @brief Exact admitted extent identity the move publishes under. */
    PagedKVArchiveExtentKey extent;
};

/** @brief One tail-first, deficit-only, move-once demotion plan. */
struct PagedDemotionPlan
{
    std::vector<PagedDemotionMove> moves; ///< In victim order, tails first.
    /** @brief Whether the deficit was NOT satisfiable against the admitted
     * route quota (real overflow): the caller must cascade to the next tier
     * or discard from the lowest enabled tier. A non-empty plan with
     * cascade=false exhausted the deficit exactly. */
    bool cascade_required = false;
    std::string refusal; ///< Nonempty exactly when no plan could be formed.

    /** @brief Whether the deficit is fully planned (or cascade is required). */
    bool ok() const noexcept { return !cascade_required && refusal.empty(); }
};

/**
 * @brief Plan the demotion of `deficit_pages` RAM pages.
 * @param entries Idle prefix chains (head first); live entries are ignored
 * (the retention owner is the live/exclusion authority — the caller passes
 * only idle chains and the owner still refuses live keys).
 * @param retention The §8.4 retention ordering owner (victim order; live
 * entries excluded; shared-class aggregation).
 * @param now Current ordinal for expiry evaluation.
 * @param deficit_pages Page-sized deficit (page granularity by construction).
 * @param can_admit Exact route admission predicate for one extent (the
 * composed disk store's quota). A refusal marks real overflow: the plan turns
 * cascade_required and carries no partial moves (all-or-none).
 * @return The exact plan (all-or-none) or a refusal.
 */
PagedDemotionPlan planTailFirstDemotion(
    const std::vector<PagedPrefixTierEntry> &entries,
    const RetentionOwner &retention, std::uint64_t now,
    std::size_t deficit_pages,
    const std::function<std::optional<PagedKVArchiveExtentKey>(
        std::uint64_t namespace_id, std::uint64_t prefix_id,
        std::uint64_t page_index)> &extent_for_page,
    const std::function<bool(const PagedKVArchiveExtentKey &)> &can_admit);

/**
 * @brief Deficit-only rounded page count from an exact byte deficit.
 *
 * A deficit is rounded UP to native page granularity (a partial page is never
 * split: the tail page is the atomic move unit).
 */
std::size_t roundDeficitToPages(std::size_t byte_deficit, std::size_t page_bytes);
} // namespace llaminar2
