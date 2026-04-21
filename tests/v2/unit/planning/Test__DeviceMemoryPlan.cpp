/**
 * @file Test__DeviceMemoryPlan.cpp
 * @brief Tests for DeviceMemoryPlan struct methods: fits(), deficit(),
 *        remaining(), total_bytes(), summary().
 *
 * These methods are fundamental to the planning system but were only tested
 * indirectly through MemoryPlanner. This file provides direct coverage.
 */

#include <gtest/gtest.h>
#include "planning/MemoryPlan.h"
#include "backends/DeviceId.h"

using namespace llaminar2;

namespace
{

DeviceMemoryPlan makePlan(size_t weights_mb, size_t kv_mb, size_t act_mb,
                          size_t ws_mb, size_t free_mb,
                          size_t headroom_mb = 128)
{
    constexpr size_t MB = 1024ULL * 1024;
    DeviceMemoryPlan p;
    p.device = DeviceId::cuda(0);
    p.weight_bytes = weights_mb * MB;
    p.kv_cache_bytes = kv_mb * MB;
    p.activation_bytes = act_mb * MB;
    p.workspace_bytes = ws_mb * MB;
    p.device_total_bytes = free_mb * MB;
    p.device_free_bytes = free_mb * MB;
    p.headroom_bytes = headroom_mb * MB;
    return p;
}

} // anonymous namespace

TEST(Test__DeviceMemoryPlan, TotalBytes_SumsAllComponents)
{
    auto p = makePlan(100, 50, 30, 200, 1024);
    EXPECT_EQ(p.total_bytes(), (100 + 50 + 30 + 200) * 1024ULL * 1024);
}

TEST(Test__DeviceMemoryPlan, Fits_TrueWhenUnderBudget)
{
    // Total = 380 MB + 128 MB headroom = 508 MB, free = 1024 MB
    auto p = makePlan(100, 50, 30, 200, 1024);
    EXPECT_TRUE(p.fits());
}

TEST(Test__DeviceMemoryPlan, Fits_FalseWhenOverBudget)
{
    // Total = 380 MB + 128 MB headroom = 508 MB, free = 400 MB
    auto p = makePlan(100, 50, 30, 200, 400);
    EXPECT_FALSE(p.fits());
}

TEST(Test__DeviceMemoryPlan, Fits_ExactBoundary)
{
    // Total = 380 MB + 128 MB headroom = 508 MB, free = 508 MB → exactly fits
    auto p = makePlan(100, 50, 30, 200, 508);
    EXPECT_TRUE(p.fits());
}

TEST(Test__DeviceMemoryPlan, Fits_OneByteShort)
{
    // One byte short of fitting
    constexpr size_t MB = 1024ULL * 1024;
    DeviceMemoryPlan p;
    p.device = DeviceId::cuda(0);
    p.weight_bytes = 100 * MB;
    p.kv_cache_bytes = 50 * MB;
    p.activation_bytes = 30 * MB;
    p.workspace_bytes = 200 * MB;
    p.device_total_bytes = 508 * MB;
    p.device_free_bytes = 508 * MB - 1;  // One byte short
    p.headroom_bytes = 128 * MB;
    EXPECT_FALSE(p.fits());
}

TEST(Test__DeviceMemoryPlan, Deficit_ZeroWhenFits)
{
    auto p = makePlan(100, 50, 30, 200, 1024);
    EXPECT_EQ(p.deficit(), 0u);
}

TEST(Test__DeviceMemoryPlan, Deficit_CorrectWhenOverBudget)
{
    // Total = 380 MB + 128 MB = 508 MB, free = 400 MB → deficit = 108 MB
    auto p = makePlan(100, 50, 30, 200, 400);
    EXPECT_EQ(p.deficit(), 108ULL * 1024 * 1024);
}

TEST(Test__DeviceMemoryPlan, Remaining_CorrectWhenFits)
{
    // Total = 380 MB + 128 MB = 508 MB, free = 1024 MB → remaining = 516 MB
    auto p = makePlan(100, 50, 30, 200, 1024);
    EXPECT_EQ(p.remaining(), 516ULL * 1024 * 1024);
}

TEST(Test__DeviceMemoryPlan, Remaining_ZeroWhenOverBudget)
{
    auto p = makePlan(100, 50, 30, 200, 400);
    EXPECT_EQ(p.remaining(), 0u);
}

TEST(Test__DeviceMemoryPlan, Summary_ContainsDeviceName)
{
    auto p = makePlan(100, 50, 30, 200, 1024);
    auto s = p.summary();
    EXPECT_NE(s.find("CUDA:0"), std::string::npos);
}

TEST(Test__DeviceMemoryPlan, Summary_ContainsOK_WhenFits)
{
    auto p = makePlan(100, 50, 30, 200, 1024);
    EXPECT_NE(p.summary().find("[OK]"), std::string::npos);
}

TEST(Test__DeviceMemoryPlan, Summary_ContainsOVER_WhenDoesNotFit)
{
    auto p = makePlan(100, 50, 30, 200, 400);
    EXPECT_NE(p.summary().find("[OVER"), std::string::npos);
}

TEST(Test__DeviceMemoryPlan, ZeroBytes_Fits)
{
    auto p = makePlan(0, 0, 0, 0, 256);
    EXPECT_TRUE(p.fits());
    EXPECT_EQ(p.total_bytes(), 0u);
    EXPECT_EQ(p.deficit(), 0u);
    EXPECT_EQ(p.remaining(), 128ULL * 1024 * 1024);  // free - headroom
}
