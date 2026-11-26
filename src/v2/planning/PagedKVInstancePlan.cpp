/**
 * @file PagedKVInstancePlan.cpp
 * @brief Single-candidate pooled-KV instance bundling implementation.
 */
#include "PagedKVInstancePlan.h"

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
    PagedKVInstancePlan plan;
    plan.pool_geometry = pool->geometry;
    plan.table_geometry = table_geometry;
    if (table_bytes > std::numeric_limits<std::size_t>::max() - pool->budget_bytes)
        throw std::overflow_error("Paged KV instance complete BOM overflows");
    plan.total_bytes = pool->budget_bytes + table_bytes;
    return plan;
}
} // namespace llaminar2
