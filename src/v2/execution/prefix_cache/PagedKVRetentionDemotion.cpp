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
    //
    // F1 (all-or-none OUTPUT): decisions are accumulated in a LOCAL vector and
    // published only after EVERY gate passed — a refusal anywhere (unresolved
    // extent, real overflow, unsatisfiable deficit) leaves the plan's moves
    // EMPTY, never a partial decision set.
    std::vector<PagedDemotionMove> decisions;
    std::size_t planned = 0;
    struct PhysicalIdentity
    {
        std::uint64_t namespace_id;
        std::uint64_t digest;
        std::size_t bytes;
    };
    std::vector<PhysicalIdentity> seen;
    for (const auto *entry : ordered)
    {
        for (auto it = entry->page_indices.rbegin();
             it != entry->page_indices.rend() && planned < deficit_pages;
             ++it)
        {
            const auto page = *it;
            // The exact extent identity is supplied by the CALLER (the real
            // bytes live in the RAM pool; the wiring resolves the extent).
            const auto extent = extent_for_page(entry->namespace_id,
                entry->prefix_id, page);
            if (!extent.has_value())
            {
                // The page has no storable extent (never exported): it cannot
                // be demoted — the deficit is unsatisfiable against this tier.
                plan.refusal = "a selected tail page has no storable extent identity";
                return plan; // decisions dropped: all-or-none.
            }
            // F2 (MOVE-ONCE on PHYSICAL identity): dedup on the RESOLVED
            // content identity (namespace + digest + bytes), never on the
            // logical (namespace, page_index) pair — two prefixes may own
            // different content at the same index and must BOTH move. A
            // genuinely shared page resolves to the SAME extent identity and
            // moves exactly once.
            bool duplicate = false;
            for (const auto &id : seen)
            {
                if (id.namespace_id == extent->namespace_id &&
                    id.digest == extent->digest && id.bytes == extent->bytes)
                {
                    duplicate = true;
                    break;
                }
            }
            if (duplicate) continue; // Physical shared page: move once.
            // REAL-OVERFLOW gate: the composed route's quota decides. A
            // refusal means the next tier is actually full (cascade required).
            if (!can_admit(*extent))
            {
                plan.cascade_required = true;
                plan.refusal = "the admitted disk quota cannot hold the selected extent (real overflow)";
                return plan; // decisions dropped: all-or-none.
            }
            seen.push_back({extent->namespace_id, extent->digest, extent->bytes});
            decisions.push_back({entry->namespace_id, entry->prefix_id,
                page, *extent});
            ++planned;
        }
        if (planned >= deficit_pages) break;
    }
    if (planned < deficit_pages)
    {
        plan.refusal = "the idle prefix set cannot satisfy the deficit tail-first";
        return plan; // decisions dropped: all-or-none.
    }
    plan.moves = std::move(decisions); // Publish only after every gate passed.
    return plan;
}
} // namespace llaminar2
