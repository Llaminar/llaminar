/**
 * @file PagedKVPoolGeometry.cpp
 * @brief Pure whole-frame pool-geometry resolution implementation.
 */
#include "PagedKVPoolGeometry.h"
#include <limits>

namespace llaminar2
{
std::optional<PagedKVPoolResolution> resolvePagedKVPoolGeometry(
    const PagedKVPoolBudget &budget)
{
    if (!budget.frame_bytes)
        throw std::invalid_argument("Paged KV pool geometry requires a positive frame byte cost");
    if (!budget.request_records || !budget.max_pages_per_request)
        throw std::invalid_argument("Paged KV pool geometry requires positive request/bound geometry");
    if (budget.capped && !budget.cap_bytes)
        throw std::invalid_argument("Paged KV pool explicit cap requires a positive byte budget");
    const std::size_t effective =
        budget.capped && budget.cap_bytes < budget.remaining_budget_bytes
            ? budget.cap_bytes
            : budget.remaining_budget_bytes;
    const std::size_t frames = effective / budget.frame_bytes;
    if (!frames) return std::nullopt;
    if (frames > std::numeric_limits<std::size_t>::max() / budget.frame_bytes)
        throw std::overflow_error("Paged KV pool geometry byte cost overflows");
    PhysicalPagePoolGeometry geometry{
        frames, budget.frame_bytes, budget.request_records, budget.max_pages_per_request};
    geometry.validate();
    PagedKVPoolResolution resolution;
    resolution.geometry = geometry;
    resolution.budget_bytes = frames * budget.frame_bytes;
    return resolution;
}
} // namespace llaminar2
