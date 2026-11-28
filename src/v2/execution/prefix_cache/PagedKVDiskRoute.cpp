/**
 * @file PagedKVDiskRoute.cpp
 * @brief Disk-tier route validation implementation.
 */
#include "PagedKVDiskRoute.h"

namespace llaminar2
{
PagedKVDiskRoute validatePagedKVDiskRoute(const PagedKVTierConfig &tiers)
{
    if (!tiers.disk_enabled)
        // Disabled: no route, no quota, one reserved journal record so the
        // off state is representable and every admit still refuses (quota 0).
        return PagedKVDiskRoute{true, {}, 0, 1, {}};
    std::string error;
    if (!tiers.valid(&error))
        return PagedKVDiskRoute{false, error, 0};
    // The quota is FINITE and includes metadata/rounding: zero quota or an
    // unrepresentable byte count is refused before the backend opens.
    if (tiers.disk_cap_bytes == 0)
        return PagedKVDiskRoute{false,
            "paged KV disk route requires a positive finite quota (metadata/rounding inclusive)"};
    if (tiers.disk_dir.empty())
        return PagedKVDiskRoute{false,
            "paged KV disk route requires a configured engine-owned directory"};
    // The configured directory is the EXACT storage route: no substitution,
    // no system/network default, never a silent fallback. The journal
    // membership is ADMITTED by the caller (typed extent geometry from the
    // planned pool); an enabled route without one is refused, never defaulted.
    if (tiers.disk_journal_membership == 0)
        return PagedKVDiskRoute{false,
            "paged KV disk route requires an admitted extent membership (typed journal cap)"};
    return PagedKVDiskRoute{true, {}, tiers.disk_cap_bytes,
        tiers.disk_journal_membership, tiers.disk_dir};
}

DiskExtentQuota makePagedKVDiskQuota(const PagedKVDiskRoute &route)
{
    // The journal membership is derived from the route's admitted extent
    // geometry (one record per admitted page plus metadata/rounding), never an
    // arbitrary constant: a disk tier that cannot hold its own admitted extent
    // set fails closed here, not at a silent 256-record boundary.
    const std::size_t membership = route.valid ? route.admitted_extent_membership : 1;
    return DiskExtentQuota(route.valid ? route.quota_bytes : 0, membership);
}
} // namespace llaminar2
