/**
 * @file Test__PagedKVTierConfig.cpp
 * @brief Typed paged-KV tier intent validation (design §4.3 contract).
 *
 * These are constructed configuration contract tests: they verify that an
 * impossible tier combination is rejected before any instance is built, that
 * explicit caps require finite quotas, and that the pool/tier modes pass into
 * a resolved runtime config unchanged. They do not claim a live serving
 * route, UMA observation or disk durability (those remain P3/P6).
 */
#include "execution/config/RuntimeConfig.h"
#include <gtest/gtest.h>

using namespace llaminar2;

/** @test Default intent is legal: auto pool, host off, disk off. */
TEST(PagedKVTierConfig, DefaultIntentIsLegalAndExplicit)
{
    PagedKVTierConfig config;
    std::string error;
    EXPECT_TRUE(config.valid(&error)) << error;
    EXPECT_EQ(config.pool_budget_mode, PagedKVPoolBudgetMode::Auto);
    EXPECT_EQ(config.host_tier_mode, PagedKVHostTierMode::Off);
    EXPECT_FALSE(config.disk_enabled);
    EXPECT_EQ(pagedKVPoolBudgetModeToString(config.pool_budget_mode), std::string("auto"));
    EXPECT_EQ(pagedKVHostTierModeToString(config.host_tier_mode), std::string("off"));
}

/** @test Explicit caps require positive finite budgets; zero refuses typed. */
TEST(PagedKVTierConfig, ExplicitCapsRequirePositiveFiniteBudgets)
{
    PagedKVTierConfig cap_pool;
    cap_pool.pool_budget_mode = PagedKVPoolBudgetMode::Cap;
    EXPECT_FALSE(cap_pool.valid());
    cap_pool.pool_cap_bytes = 4096;
    EXPECT_TRUE(cap_pool.valid());

    PagedKVTierConfig cap_host;
    cap_host.host_tier_mode = PagedKVHostTierMode::Cap;
    EXPECT_FALSE(cap_host.valid());
    cap_host.host_cap_bytes = 8192;
    EXPECT_TRUE(cap_host.valid());
}

/** @test Disk enablement requires an explicit finite quota and configured directory. */
TEST(PagedKVTierConfig, DiskEnablementRequiresQuotaAndDirectory)
{
    PagedKVTierConfig config;
    config.disk_enabled = true;
    EXPECT_FALSE(config.valid()); // No quota or directory.
    config.disk_cap_bytes = 1ull << 30;
    EXPECT_FALSE(config.valid()); // Still no directory.
    config.disk_dir = "/var/lib/llaminar/kv";
    EXPECT_TRUE(config.valid());
    // A directory without a quota still refuses.
    config.disk_cap_bytes = 0;
    EXPECT_FALSE(config.valid());
}

/** @test Resolution fails closed on an invalid authored intent. */
TEST(PagedKVTierConfig, ResolutionFailsClosedOnInvalidIntent)
{
    PagedKVTierConfig invalid;
    invalid.disk_enabled = true;
    invalid.disk_cap_bytes = 0;
    EXPECT_THROW(RuntimeConfig::fromOrchestrationConfig(4096, 1, "fp32", "fp32",
        FusedAttentionBackend::JIT, RoutedExpertComputePolicy::Apportioned,
        MoEHotExpertCacheConfig{}, RoutedExpertPrefillRuntimeConfig{},
        MoERebalanceRuntimeConfig{}, {}, invalid), std::invalid_argument);
}

/** @test A complete typed intent survives resolution unchanged. */
TEST(PagedKVTierConfig, CompleteIntentSurvivesResolutionUnchanged)
{
    PagedKVTierConfig intent;
    intent.pool_budget_mode = PagedKVPoolBudgetMode::Cap;
    intent.pool_cap_bytes = 5 * 1024 * 1024;
    intent.host_tier_mode = PagedKVHostTierMode::Cap;
    intent.host_cap_bytes = 2 * 1024 * 1024;
    intent.disk_enabled = true;
    intent.disk_cap_bytes = 64 * 1024 * 1024;
    intent.disk_dir = "/tmp/llaminar-kv-tier";
    auto runtime = RuntimeConfig::fromOrchestrationConfig(4096, 2, "fp32", "fp32",
        FusedAttentionBackend::JIT, RoutedExpertComputePolicy::Apportioned,
        MoEHotExpertCacheConfig{}, RoutedExpertPrefillRuntimeConfig{},
        MoERebalanceRuntimeConfig{}, {}, intent);
    EXPECT_EQ(runtime.paged_kv_tiers.pool_cap_bytes, 5 * 1024 * 1024u);
    EXPECT_EQ(runtime.paged_kv_tiers.host_cap_bytes, 2 * 1024 * 1024u);
    EXPECT_EQ(runtime.paged_kv_tiers.disk_cap_bytes, 64 * 1024 * 1024u);
    EXPECT_TRUE(runtime.paged_kv_tiers.disk_enabled);
    EXPECT_EQ(runtime.paged_kv_tiers.disk_dir, std::string("/tmp/llaminar-kv-tier"));
}
