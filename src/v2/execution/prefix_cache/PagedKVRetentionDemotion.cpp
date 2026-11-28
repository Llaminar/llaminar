/**
 * @file PagedKVRetentionDemotion.cpp
 * @brief §8.4 demotion decision planner implementation.
 */
#include "execution/prefix_cache/PagedKVRetentionDemotion.h"

namespace llaminar2
{
std::size_t roundDeficitToPages(std::size_t byte_deficit, std::size_t page_bytes)
{
    if (page_bytes == 0 || byte_deficit == 0) return 0;
    const auto pages = byte_deficit / page_bytes;
    return (byte_deficit % page_bytes == 0) ? pages : pages + 1;
}

PagedDemotionPlan planTailFirstDemotion(
    const std::vector<PagedPrefixTierEntry> &entries,
    const RetentionOwner &retention, std::uint64_t now,
    std::size_t deficit_pages,
    const std::function<std::optional<PagedKVArchiveExtentKey>(
        std::uint64_t namespace_id, std::uint64_t prefix_id,
        std::uint64_t page_index)> &extent_for_page,
    const std::function<bool(const PagedKVArchiveExtentKey &)> &can_admit)
{
    PagedDemotionPlan plan;
    if (deficit_pages == 0)
        return plan; // Zero deficit: nothing to do (deficit-only by rule).

    // Victim order comes FROM the retention owner (class -> value -> seq;
    // expired first; live/protected NEVER selected). Entries without an owner
    // record are NOT victimizable (no retention intent admitted for them).
    // The owner also returns at most the idle set; we request a generous view
    // and take the first `deficit_pages` distinct victims in order.
    const auto victims = retention.selectVictims(now, entries.size() + deficit_pages);
    std::vector<const PagedPrefixTierEntry *> ordered;
    for (const auto &victim : victims)
    {
        for (const auto &entry : entries)
        {
            if (entry.owner == victim)
            {
                ordered.push_back(&entry);
                break;
            }
        }
    }
    if (ordered.empty())
    {
        plan.refusal = "no idle prefix is victimizable under the retention owner's order";
        return plan;
    }

    // Tail-first: each victim demotes its LAST page first; leading pages stay
    // resident (a hit starts at token zero). Deficit-only: stop exactly at
    // deficit_pages. Move-once: a shared page appears at most once.
    std::size_t planned = 0;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> seen;
    for (const auto *entry : ordered)
    {
        for (auto it = entry->page_indices.rbegin();
             it != entry->page_indices.rend() && planned < deficit_pages;
             ++it)
        {
            const auto page = *it;
            bool duplicate = false;
            for (const auto &[seen_ns, seen_page] : seen)
                if (seen_ns == entry->namespace_id && seen_page == page)
                {
                    duplicate = true;
                    break;
                }
            if (duplicate) continue; // Shared page: move once.
            seen.emplace_back(entry->namespace_id, page);
            // The exact extent identity is supplied by the CALLER (the real
            // bytes live in the RAM pool; the wiring resolves the extent).
            const auto extent = extent_for_page(entry->namespace_id,
                entry->prefix_id, page);
            if (!extent.has_value())
            {
                // The page has no storable extent (never exported): it cannot
                // be demoted — the deficit is unsatisfiable against this tier.
                plan.refusal = "a selected tail page has no storable extent identity";
                return plan;
            }
            // REAL-OVERFLOW gate: the composed route's quota decides. A
            // refusal means the next tier is actually full (cascade required).
            if (!can_admit(*extent))
            {
                plan.cascade_required = true;
                plan.refusal = "the admitted disk quota cannot hold the selected extent (real overflow)";
                return plan;
            }
            plan.moves.push_back({entry->namespace_id, entry->prefix_id,
                page, *extent});
            ++planned;
        }
        if (planned >= deficit_pages) break;
    }
    if (planned < deficit_pages)
    {
        plan.refusal = "the idle prefix set cannot satisfy the deficit tail-first";
        return plan;
    }
    return plan;
}
} // namespace llaminar2
