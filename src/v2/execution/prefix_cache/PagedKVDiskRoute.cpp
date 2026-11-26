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
        return PagedKVDiskRoute{true, {}, 0}; // Disabled: no route, no quota.
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
    // no system/network default, never a silent fallback.
    return PagedKVDiskRoute{true, {}, tiers.disk_cap_bytes};
}

DiskExtentQuota makePagedKVDiskQuota(const PagedKVDiskRoute &route)
{
    return DiskExtentQuota(route.valid ? route.quota_bytes : 0);
}
} // namespace llaminar2
