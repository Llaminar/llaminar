/**
 * @file Test__PagedKVPoolGeometry.cpp
 * @brief Whole-frame pool geometry resolution contracts (design §4.2).
 *
 * Constructed contract tests: `auto` maximizes inside the supplied remaining
 * budget, an explicit cap bounds the same pool (never stealing execution
 * memory), unrepresentable/zero budgets refuse typed, and the pure resolver
 * performs no accounting of its own (the returned geometry is re-admitted by
 * PMA by the caller).
 */
#include "planning/PagedKVPoolGeometry.h"
#include <gtest/gtest.h>

using namespace llaminar2;

/** @test Auto maximizes whole frames; a trailing remainder is never spent. */
TEST(PagedKVPoolGeometry, AutoMaximizesWholeFramesOnly)
{
    PagedKVPoolBudget budget;
    budget.remaining_budget_bytes = 1000;
    budget.frame_bytes = 256;
    budget.request_records = 4;
    budget.max_pages_per_request = 3;
    auto resolution = resolvePagedKVPoolGeometry(budget);
    ASSERT_TRUE(resolution);
    EXPECT_EQ(resolution->geometry.frames, 3u); // 3*256 = 768 <= 1000; 4 would be 1024.
    EXPECT_EQ(resolution->budget_bytes, 768u);
}

/** @test An explicit cap bounds the same pool below `auto`; no steal, no overflow. */
TEST(PagedKVPoolGeometry, ExplicitCapBoundsTheSamePool)
{
    PagedKVPoolBudget budget;
    budget.remaining_budget_bytes = 4096;
    budget.frame_bytes = 256;
    budget.request_records = 4;
    budget.max_pages_per_request = 4;
    auto free = resolvePagedKVPoolGeometry(budget);
    ASSERT_TRUE(free);
    EXPECT_EQ(free->geometry.frames, 16u);
    budget.capped = true;
    budget.cap_bytes = 1024;
    auto capped = resolvePagedKVPoolGeometry(budget);
    ASSERT_TRUE(capped);
    EXPECT_EQ(capped->geometry.frames, 4u);
    EXPECT_EQ(capped->budget_bytes, 1024u);
    // A cap larger than the remaining budget never exceeds the admitted value.
    budget.cap_bytes = 8192;
    auto clamped = resolvePagedKVPoolGeometry(budget);
    ASSERT_TRUE(clamped);
    EXPECT_EQ(clamped->geometry.frames, 16u);
}

/** @test Budget, geometry and overflow rejections are typed and fail closed. */
TEST(PagedKVPoolGeometry, InvalidInputsFailClosed)
{
    PagedKVPoolBudget budget;
    budget.frame_bytes = 64;
    budget.request_records = 2;
    budget.max_pages_per_request = 1;
    EXPECT_FALSE(resolvePagedKVPoolGeometry(budget)); // Zero remaining budget.
    budget.remaining_budget_bytes = 63;
    EXPECT_FALSE(resolvePagedKVPoolGeometry(budget)); // Not enough for one whole frame.
    EXPECT_THROW(resolvePagedKVPoolGeometry([&] {
        auto broken = budget;
        broken.frame_bytes = 0;
        return broken;
    }()), std::invalid_argument);
    EXPECT_THROW(resolvePagedKVPoolGeometry([&] {
        auto broken = budget;
        broken.request_records = 0;
        return broken;
    }()), std::invalid_argument);
    EXPECT_THROW(resolvePagedKVPoolGeometry([&] {
        auto broken = budget;
        broken.capped = true;
        return broken;
    }()), std::invalid_argument);
}
