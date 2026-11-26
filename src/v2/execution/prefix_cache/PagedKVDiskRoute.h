/**
 * @file PagedKVDiskRoute.h
 * @brief Typed disk-tier route admission (plan P3 §11.3).
 *
 * A configured disk tier is a FIRST-CLASS route: before any backend opens,
 * the route validator checks the typed quota against the admitted byte bound
 * and the configured directory against the engine-only namespace rules. An
 * impossible configured tier fails admission with a precise diagnostic — it
 * is never silently substituted by a system or network volume, and the
 * engine never purges files it does not own.
 *
 * This is a pure validation input: it does not allocate, does not touch the
 * filesystem and performs no physical accounting (PMA and the storage
 * backend own those). Callers pass the result to the backend opening path.
 */
#pragma once
#include "execution/config/RuntimeConfig.h"
#include "execution/prefix_cache/DiskExtentQuota.h"
#include <cstddef>
#include <string>

namespace llaminar2
{
/** @brief Validated disk-tier route decision. */
struct PagedKVDiskRoute
{
    bool valid = false;
    std::string refusal; ///< Nonempty exactly when invalid.
    std::size_t quota_bytes = 0; ///< Admitted finite quota (rounded, inclusive).
    /**
     * @brief Admitted extent-record membership for the route journal.
     *
     * Sized from the admitted geometry (per-admitted-page extent + in-flight
     * write + pending replacement + rounding), never a hard-coded constant:
     * the quota journal can always hold its own admitted extent set.
     */
    std::size_t admitted_extent_membership = 0;
};

/**
 * @brief Validate one typed disk-tier route.
 * @param tiers Paged tier intents (disk_enabled/quota/directory authoritative).
 * @return Valid with the exact admitted quota, or invalid with the precise
 * diagnostic (impossible quota, empty directory, disabled-with-requirements).
 */
PagedKVDiskRoute validatePagedKVDiskRoute(const PagedKVTierConfig &tiers);

/**
 * @brief Build a checked disk extent quota for the admitted route.
 *
 * The extent quota is the backend-facing finite quota ledger: every admitted
 * disk extent (payload + metadata + rounding) is charged here with exact
 * identity, so the route's asked and effective bounds are the same numbers.
 */
DiskExtentQuota makePagedKVDiskQuota(const PagedKVDiskRoute &route);
} // namespace llaminar2
