/**
 * @file Test__CPUQ16AttentionMath.cpp
 * @brief Device-free, exhaustive integer range proof for Q16 attention.
 *
 * Prove the full reduction bound independently in int64, including native
 * -32768 keys and sizes beyond current physical cache blocks. This test does
 * not construct tensors, launch workers, or execute an attention kernel.
 */
#include <gtest/gtest.h>
#include "kernels/cpu/attention/CPUQ16AttentionMath.h"
#include <cstdint>
#include <limits>

/** Every admitted size is safe, and its bound is maximal up to the scratch cap. */
TEST(CPUQ16AttentionMath, EveryPositiveRepresentableReductionIsSafe)
{
    constexpr std::int64_t limit = std::numeric_limits<std::int32_t>::max();
    for (int terms = 1; terms <= limit / 32768; ++terms)
    {
        const auto q = llaminar2::cpu::q16AttentionQueryLimit(terms);
        ASSERT_GT(q, 0);
        ASSERT_LE(q, 2047);
        ASSERT_LE(std::int64_t(q) * 32768 * terms, limit);
        if (q < 2047)
            ASSERT_GT(std::int64_t(q + 1) * 32768 * terms, limit);
    }
}

/** Physical block limits are compile-time values; invalid geometry is rejected. */
TEST(CPUQ16AttentionMath, PhysicalBlocksAndInvalidGeometry)
{
    using llaminar2::cpu::q16AttentionQueryLimit;
    static_assert(q16AttentionQueryLimit(32) == 2047);
    static_assert(q16AttentionQueryLimit(64) == 1023);
    static_assert(q16AttentionQueryLimit(128) == 511);
    EXPECT_THROW(q16AttentionQueryLimit(0), std::invalid_argument);
    EXPECT_THROW(q16AttentionQueryLimit(-1), std::invalid_argument);
    EXPECT_THROW(q16AttentionQueryLimit(65536), std::invalid_argument);
    EXPECT_THROW(q16AttentionQueryLimit(std::numeric_limits<int>::max()), std::invalid_argument);
}
