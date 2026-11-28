/**
 * @file PagedKVInstancePlan.cpp
 * @brief Single-candidate pooled-KV instance bundling implementation.
 */
#include "PagedKVInstancePlan.h"
#include "execution/prefix_cache/PagedKVArchiveStore.h"

namespace llaminar2
{
std::optional<PagedKVInstancePlan> planPagedKVInstance(
    const PagedKVTierConfig &config, const PagedKVPoolBudget &remaining,
    const PagedKVInstancePolicy &policy, DeviceId device,
    CPUStoragePlacement placement)
{
    std::string error;
    if (!config.valid(&error)) throw std::invalid_argument(error);
    if (policy.page_tokens <= 0 || !policy.bank_count || !policy.readers_per_bank ||
        !policy.request_records)
        throw std::invalid_argument("Paged KV instance policy requires positive exact geometry");
    if (remaining.frame_bytes == 0 || remaining.max_pages_per_request == 0)
        throw std::invalid_argument("Paged KV instance budget requires exact frame and per-request geometry");
    // One pooled-KV instance is ONE admitted consumer: table metadata and
    // pool frames both compete for the same remaining budget, so the typed
    // table metadata is priced FIRST and the pool frames come out of what is
    // left. That keeps the complete BOM inside the supplied budget (the
    // all-resource rule) and makes the metadata never cheaper than reality.
    CPUFP32PagedKVTableGeometry table_geometry{
        remaining.max_pages_per_request, policy.bank_count,
        policy.readers_per_bank, policy.page_tokens,
        0, 0};
    if (policy.bank_count > std::numeric_limits<std::uint64_t>::max() - policy.request_records)
        throw std::overflow_error("Paged KV instance issuance envelope overflows");
    table_geometry.issuance_budget =
        static_cast<std::uint64_t>(policy.bank_count) +
        static_cast<std::uint64_t>(policy.request_records);
    table_geometry.issuance_headroom = static_cast<std::uint64_t>(policy.bank_count);
    const auto table_bytes = CPUFP32PagedKVTable::physicalBytes(table_geometry, device, placement);
    if (table_bytes >= remaining.remaining_budget_bytes)
        return std::nullopt; // Metadata alone (or less than one frame) cannot fit.
    const std::size_t for_frames = remaining.remaining_budget_bytes - table_bytes;
    // The tier config supplies the pool cap intent; auto maximizes inside the
    // budget that remains AFTER the table metadata was priced.
    PagedKVPoolBudget pool_budget = remaining;
    pool_budget.remaining_budget_bytes = for_frames;
    pool_budget.request_records = policy.request_records;
    if (config.pool_budget_mode == PagedKVPoolBudgetMode::Cap)
    {
        pool_budget.capped = true;
        pool_budget.cap_bytes = config.pool_cap_bytes;
        // A cap that cannot even hold one full per-request set is NOT an
        // admitted instance: refuse the candidate up front rather than
        // constructing a geometry that violates its own frame bound.
        if (config.pool_cap_bytes / pool_budget.frame_bytes <
            pool_budget.max_pages_per_request)
            return std::nullopt;
    }
    auto pool = resolvePagedKVPoolGeometry(pool_budget);
    if (!pool || pool->geometry.frames < pool->geometry.max_pages_per_request)
        return std::nullopt; // No frame or under-cap per-request bound: no instance.
    if (config.disk_enabled)
    {
        // Composed route: the disk membership is sized from the PLANNED
        // geometry's typed need — one record per admitted frame, plus one
        // in-flight write, one pending replacement, one retained old
        // generation (old+new extents are both admitted until retired) and
        // one journal/rounding record. The quota must hold at least one
        // rounded extent of the admitted frame geometry (floor note: without
        // the archive header; a per-section put refuses precisely when its
        // exact serialized size crosses a block boundary). An authored cap
        // below the need, or a quota that cannot hold the smallest possible
        // section, fails closed here with a precise diagnostic (never a
        // silent boundary, never a default).
        const std::size_t membership_need =
            pool->geometry.frames + 4; // + in-flight + pending + old gen + rounding.
        if (config.disk_journal_membership < membership_need)
            throw std::invalid_argument("paged KV disk membership " +
                std::to_string(config.disk_journal_membership) +
                " is below the planned geometry's typed extent need (" +
                std::to_string(membership_need) + ")");
        const auto smallest_extent =
            pagedKVArchiveExtentBytes(pool->geometry.frame_bytes);
        if (config.disk_cap_bytes < smallest_extent)
            throw std::invalid_argument("paged KV disk quota " +
                std::to_string(config.disk_cap_bytes) +
                " cannot hold one rounded extent of the admitted frame geometry (" +
                std::to_string(smallest_extent) + " bytes)");
    }
    PagedKVInstancePlan plan;
    plan.pool_geometry = pool->geometry;
    plan.table_geometry = table_geometry;
    // The complete BOM is priced by the FACTORIES (exact object/allocator
    // overhead), never by the raw frame-product estimate: PMA charges what
    // the pool and table actually materialize, and admission must match it.
    const auto pool_bytes = PhysicalPagePool::physicalBytes(pool->geometry, device, placement);
    if (table_bytes > std::numeric_limits<std::size_t>::max() - pool_bytes)
        throw std::overflow_error("Paged KV instance complete BOM overflows");
    plan.total_bytes = pool_bytes + table_bytes;
    if (config.disk_enabled)
    {
        auto route = validatePagedKVDiskRoute(config);
        if (!route.valid)
            throw std::invalid_argument(route.refusal);
        plan.disk_route = route;
    }
    return plan;
}
} // namespace llaminar2
