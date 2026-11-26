/**
 * @file PagedKVInstanceFactory.h
 * @brief One step from authored tier intent to a live admitted instance.
 *
 * The factory composes the typed tier config (PagedKVTierConfig), the exact
 * native geometry policy and the PMA-reported remaining budget into one
 * planned candidate and admits it through PagedKVInstance::create. It is the
 * ONLY path a caller uses to obtain a live pooled instance; it never performs
 * its own physical accounting, never reads host free memory and never
 * invents a default (the config is always explicit). A refusal returns the
 * precise diagnostic; no partial instance can exist.
 */
#pragma once
#include "execution/config/RuntimeConfig.h"
#include "planning/PagedKVInstance.h"

namespace llaminar2
{
/** @brief One exact factory outcome: live instance or precise refusal. */
struct PagedKVInstanceOutcome
{
    std::optional<PagedKVInstance> instance;
    std::string refusal; ///< Nonempty exactly when instance is empty.
    bool ok() const noexcept { return instance.has_value(); }
};

/**
 * @brief Admit one pooled instance from the authored tier intent.
 * @param authority Original PMA instance.
 * @param tiers Authored tier intents (pool auto/cap, host, disk).
 * @param remaining Budget input from the caller AFTER the complete non-pool
 * BOM (the value PMA reports; the factory never subtracts itself).
 * @param policy Exact geometry policy (page tokens, banks, reader envelope).
 * @param device Original CPU allocator identity.
 * @param placement Explicit first-touch policy.
 * @return The live instance, or an empty outcome with a precise refusal.
 * @throws std::invalid_argument For an invalid authored intent or policy.
 */
PagedKVInstanceOutcome admitConfiguredPagedKVInstance(
    std::shared_ptr<PhysicalMemoryAuthority> authority,
    const PagedKVTierConfig &tiers, const PagedKVPoolBudget &remaining,
    const PagedKVInstancePolicy &policy, DeviceId device,
    CPUStoragePlacement placement = CPUStoragePlacement::local());
} // namespace llaminar2
