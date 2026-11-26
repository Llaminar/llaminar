/**
 * @file Test__PagedKVInstancePlan.cpp
 * @brief Bundled pooled-KV instance plan contracts (design §4.2/§4.3).
 *
 * These tests prove one candidate is complete or absent (no partial pool or
 * table readiness), that an explicit pool cap bounds the pool while the table
 * metadata remains priced within the same remaining budget, and that invalid
 * intents/policies fail typed before any plan can be built. PMA remains the
 * sole admission authority — the candidate is only a priced input.
 */
#include "planning/PagedKVInstancePlan.h"
#include <gtest/gtest.h>

using namespace llaminar2;

namespace
{
PagedKVPoolBudget budgetFor(std::size_t remaining = 4096 * 1024)
{
    PagedKVPoolBudget budget;
    budget.remaining_budget_bytes = remaining;
    budget.frame_bytes = 64 * 1024;
    budget.request_records = 4;
    budget.max_pages_per_request = 4;
    return budget;
}
}

/** @test Auto intent bundles pool + table whose complete BOM fits the budget. */
TEST(PagedKVInstancePlan, AutoIntentProducesCompleteCandidate)
{
    PagedKVTierConfig config; // Auto pool, host off, disk off (defaults).
    auto plan = planPagedKVInstance(config, budgetFor(),
        PagedKVInstancePolicy{8, 2, 4, 4}, DeviceId::cpu());
    ASSERT_TRUE(plan);
    EXPECT_GT(plan->pool_geometry.frames, 0u);
    EXPECT_EQ(plan->pool_geometry.frame_bytes, 64 * 1024u);
    EXPECT_EQ(plan->table_geometry.page_tokens, 8);
    EXPECT_EQ(plan->table_geometry.bank_count, 2u);
    EXPECT_EQ(plan->table_geometry.issuance_budget, 6u); // banks + records.
    EXPECT_EQ(plan->table_geometry.issuance_headroom, 2u);
    EXPECT_LE(plan->total_bytes, 4096 * 1024u);
    EXPECT_GT(plan->total_bytes, 0u);
    // Table metadata is priced inside the same remaining budget, so the pool
    // never over-admits: pool + table bytes stay within the supplied value.
    EXPECT_LE(plan->pool_geometry.frames * plan->pool_geometry.frame_bytes +
        CPUFP32PagedKVTable::physicalBytes(plan->table_geometry, DeviceId::cpu()),
        4096 * 1024u);
    // Both geometries are independently valid.
    EXPECT_NO_THROW(plan->pool_geometry.validate());
    EXPECT_NO_THROW(plan->table_geometry.validate());
}

/** @test An explicit cap bounds the pool frames without exceeding the budget. */
TEST(PagedKVInstancePlan, ExplicitCapBoundsPoolWithinBudget)
{
    PagedKVTierConfig config;
    config.pool_budget_mode = PagedKVPoolBudgetMode::Cap;
    config.pool_cap_bytes = 256 * 1024; // Exactly 4 frames at 64KiB.
    auto plan = planPagedKVInstance(config, budgetFor(),
        PagedKVInstancePolicy{8, 2, 4, 4}, DeviceId::cpu());
    ASSERT_TRUE(plan);
    EXPECT_EQ(plan->pool_geometry.frames * plan->pool_geometry.frame_bytes, 256 * 1024u);
    EXPECT_EQ(plan->pool_geometry.frames, 4u);
    // A cap below one full per-request set is refused as never-admitted.
    config.pool_cap_bytes = 128 * 1024; // 2 frames < bound 4.
    EXPECT_FALSE(planPagedKVInstance(config, budgetFor(),
        PagedKVInstancePolicy{8, 2, 4, 4}, DeviceId::cpu()));
}

/** @test Insufficient remaining budget yields NO candidate (no partial plan). */
TEST(PagedKVInstancePlan, InsufficientBudgetYieldsNoPartialPlan)
{
    PagedKVTierConfig config;
    // 64KiB remaining: metadata fits but no frame can, or per-request bound
    // exceeds the frame count: either way no candidate exists.
    auto plan = planPagedKVInstance(config, budgetFor(64 * 1024),
        PagedKVInstancePolicy{8, 2, 4, 4}, DeviceId::cpu());
    EXPECT_FALSE(plan);
}

/** @test Invalid intents and policies fail typed before any plan exists. */
TEST(PagedKVInstancePlan, InvalidIntentsFailTyped)
{
    PagedKVTierConfig disk_without_dir;
    disk_without_dir.disk_enabled = true; // No quota/dir: invalid intent.
    EXPECT_THROW(planPagedKVInstance(disk_without_dir, budgetFor(),
        PagedKVInstancePolicy{8, 2, 4, 4}, DeviceId::cpu()), std::invalid_argument);
    EXPECT_THROW(planPagedKVInstance(PagedKVTierConfig{}, budgetFor(),
        PagedKVInstancePolicy{0, 2, 4, 4}, DeviceId::cpu()), std::invalid_argument);
}
