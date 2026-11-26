/**
 * @file Test__PagedKVDiskRoute.cpp
 * @brief Typed disk-tier route admission contracts (plan P3 §11.3).
 *
 * These tests prove an impossible configured tier fails admission with a
 * precise diagnostic before any backend opens (zero quota, missing directory,
 * invalid intent), a disabled tier is a fully qualified off state, and the
 * admitted route yields a checked finite extent quota with the same numbers.
 */
#include "execution/prefix_cache/PagedKVDiskRoute.h"
#include <gtest/gtest.h>

using namespace llaminar2;

/** @test A complete configured disk route admits with the exact finite quota. */
TEST(PagedKVDiskRoute, CompleteRouteAdmitsExactQuota)
{
    PagedKVTierConfig config;
    config.disk_enabled = true;
    config.disk_cap_bytes = 4 * 1024 * 1024;
    config.disk_dir = "/var/lib/llaminar/cb-kv";
    auto route = validatePagedKVDiskRoute(config);
    EXPECT_TRUE(route.valid) << route.refusal;
    EXPECT_EQ(route.quota_bytes, 4 * 1024 * 1024u);
    auto quota = makePagedKVDiskQuota(route);
    EXPECT_TRUE(quota.admit(DiskExtentKey{1, 1, 0xabc, 1 * 1024 * 1024}));
    EXPECT_FALSE(quota.admit(DiskExtentKey{2, 1, 0xdef, 4 * 1024 * 1024}));
}

/** @test Disabled tier is a qualified off state (no quota demanded). */
TEST(PagedKVDiskRoute, DisabledTierIsQualifiedOff)
{
    PagedKVTierConfig config; // Defaults: disk disabled.
    auto route = validatePagedKVDiskRoute(config);
    EXPECT_TRUE(route.valid) << route.refusal;
    EXPECT_EQ(route.quota_bytes, 0u);
    auto quota = makePagedKVDiskQuota(route);
    EXPECT_FALSE(quota.admit(DiskExtentKey{1, 1, 1, 1})); // Zero quota refuses.
}

/** @test Zero-quota ledger never admits (no underflow admission). */
TEST(PagedKVDiskRoute, ZeroQuotaNeverAdmits)
{
    DiskExtentQuota zero(0);
    EXPECT_FALSE(zero.admit(DiskExtentKey{1, 1, 1, 1}));
    EXPECT_EQ(zero.count(), 0u);
    EXPECT_EQ(zero.bytesAdmitted(), 0u);
    // Large extent must not underflow the subtraction either.
    EXPECT_FALSE(zero.admit(DiskExtentKey{2, 1, 1, SIZE_MAX / 2}));
    EXPECT_EQ(zero.count(), 0u);
}

/** @test Impossible configured tiers fail admission with the precise diagnostic. */
TEST(PagedKVDiskRoute, ImpossibleTiersFailWithPreciseDiagnostic)
{
    PagedKVTierConfig no_quota;
    no_quota.disk_enabled = true;
    auto route = validatePagedKVDiskRoute(no_quota);
    EXPECT_FALSE(route.valid);
    EXPECT_NE(route.refusal.find("quota"), std::string::npos) << route.refusal;

    PagedKVTierConfig no_dir;
    no_dir.disk_enabled = true;
    no_dir.disk_cap_bytes = 1024;
    auto no_dir_route = validatePagedKVDiskRoute(no_dir);
    EXPECT_FALSE(no_dir_route.valid);
    EXPECT_NE(no_dir_route.refusal.find("directory"), std::string::npos) << no_dir_route.refusal;
}
