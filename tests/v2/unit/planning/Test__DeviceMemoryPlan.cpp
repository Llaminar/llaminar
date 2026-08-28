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
                          size_t ws_mb, size_t free_mb)
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
    return p;
}

} // anonymous namespace

TEST(Test__DeviceMemoryPlan, TotalBytes_SumsAllComponents)
{
    auto p = makePlan(100, 50, 30, 200, 1024);
    EXPECT_EQ(p.total_bytes(), (100 + 50 + 30 + 200) * 1024ULL * 1024);
}

TEST(Test__DeviceMemoryPlan, TotalBytesIncludesPersistentState)
{
    constexpr size_t MB = 1024ULL * 1024ULL;
    auto p = makePlan(100, 50, 30, 200, 1024);
    p.persistent_state_bytes = 75 * MB;

    EXPECT_EQ(
        p.total_bytes(),
        (100 + 50 + 75 + 30 + 200) * MB);
    EXPECT_NE(p.summary().find("state=75 MB"), std::string::npos);
}

TEST(Test__DeviceMemoryPlan, Fits_TrueWhenUnderBudget)
{
    // Exact concrete total = 380 MB, free = 1024 MB.
    auto p = makePlan(100, 50, 30, 200, 1024);
    EXPECT_TRUE(p.fits());
}

TEST(Test__DeviceMemoryPlan, Fits_FalseWhenOverBudget)
{
    // Exact concrete total = 380 MB, free = 379 MB.
    auto p = makePlan(100, 50, 30, 200, 379);
    EXPECT_FALSE(p.fits());
}

TEST(Test__DeviceMemoryPlan, Fits_ExactBoundary)
{
    // Exact concrete total = 380 MB, free = 380 MB.
    auto p = makePlan(100, 50, 30, 200, 380);
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
    p.device_total_bytes = 380 * MB;
    p.device_free_bytes = 380 * MB - 1;  // One byte short
    EXPECT_FALSE(p.fits());
}

TEST(Test__DeviceMemoryPlan, Deficit_ZeroWhenFits)
{
    auto p = makePlan(100, 50, 30, 200, 1024);
    EXPECT_EQ(p.deficit(), 0u);
}

TEST(Test__DeviceMemoryPlan, Deficit_CorrectWhenOverBudget)
{
    // Total = 380 MB, free = 300 MB → deficit = 80 MB.
    auto p = makePlan(100, 50, 30, 200, 300);
    EXPECT_EQ(p.deficit(), 80ULL * 1024 * 1024);
}

TEST(Test__DeviceMemoryPlan, Remaining_CorrectWhenFits)
{
    // Total = 380 MB, free = 1024 MB → remaining = 644 MB.
    auto p = makePlan(100, 50, 30, 200, 1024);
    EXPECT_EQ(p.remaining(), 644ULL * 1024 * 1024);
}

TEST(Test__DeviceMemoryPlan, Remaining_ZeroWhenOverBudget)
{
    auto p = makePlan(100, 50, 30, 200, 379);
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
    auto p = makePlan(100, 50, 30, 200, 379);
    EXPECT_NE(p.summary().find("[OVER"), std::string::npos);
}

TEST(Test__DeviceMemoryPlan, ZeroBytes_Fits)
{
    auto p = makePlan(0, 0, 0, 0, 256);
    EXPECT_TRUE(p.fits());
    EXPECT_EQ(p.total_bytes(), 0u);
    EXPECT_EQ(p.deficit(), 0u);
    EXPECT_EQ(p.remaining(), 256ULL * 1024 * 1024);
}
