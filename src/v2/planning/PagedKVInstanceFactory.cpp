/**
 * @file PagedKVInstanceFactory.cpp
 * @brief Tier-intent-to-instance factory implementation.
 */
#include "PagedKVInstanceFactory.h"

namespace llaminar2
{
PagedKVInstanceOutcome admitConfiguredPagedKVInstance(
    std::shared_ptr<PhysicalMemoryAuthority> authority,
    const PagedKVTierConfig &tiers, const PagedKVPoolBudget &remaining,
    const PagedKVInstancePolicy &policy, DeviceId device,
    CPUStoragePlacement placement)
{
    std::string error;
    if (!tiers.valid(&error))
        throw std::invalid_argument(error);
    std::optional<PagedKVInstancePlan> plan;
    try
    {
        plan = planPagedKVInstance(tiers, remaining, policy, device, placement);
    }
    catch (const std::invalid_argument &e)
    {
        // Typed planner refusal (geometry, quota or composed-route admission
        // problem) surfaces as the PRECISE diagnostic, never a generic one.
        return PagedKVInstanceOutcome{std::nullopt, e.what()};
    }
    if (!plan)
        return PagedKVInstanceOutcome{std::nullopt,
            "the remaining budget cannot hold a pooled instance with a legal whole-frame pool"};
    std::string refusal;
    auto instance = PagedKVInstance::create(
        std::move(authority), *plan, device, placement, &refusal);
    if (!instance)
    {
        if (refusal.empty())
            refusal = "the physical authority refused the complete pooled instance BOM";
        return PagedKVInstanceOutcome{std::nullopt, std::move(refusal)};
    }
    return PagedKVInstanceOutcome{std::move(*instance), {}};
}
} // namespace llaminar2
