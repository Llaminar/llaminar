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
                          size_t state_mb = 0)
{
    constexpr size_t MB = 1024ULL * 1024;
    PhysicalMemoryBOMBuilder builder({
        .world_rank = -1,
        .device = DeviceId::cuda(0),
        .total_bytes = free_mb * MB,
        .admission_available_bytes = free_mb * MB,
    });
    builder
        .add(
            PhysicalMemoryOwner::PrimaryModelWeights,
            weights_mb * MB)
        .add(PhysicalMemoryOwner::KVCache, kv_mb * MB)
        .add(
            PhysicalMemoryOwner::RecurrentLiveState,
            state_mb * MB)
        .add(PhysicalMemoryOwner::ActivationArena, act_mb * MB)
        .add(PhysicalMemoryOwner::ExecutionWorkspace, ws_mb * MB);
    return DeviceMemoryPlan(builder.build(), /*max_seq_len=*/0,
                            /*activation_seq_len=*/0);
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
    auto p = makePlan(100, 50, 30, 200, 1024, 75);

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
    PhysicalMemoryBOMBuilder builder({
        .world_rank = -1,
        .device = DeviceId::cuda(0),
        .total_bytes = 380 * MB,
        .admission_available_bytes = 380 * MB - 1,
    });
    builder
        .add(PhysicalMemoryOwner::PrimaryModelWeights, 100 * MB)
        .add(PhysicalMemoryOwner::KVCache, 50 * MB)
        .add(PhysicalMemoryOwner::ActivationArena, 30 * MB)
        .add(PhysicalMemoryOwner::ExecutionWorkspace, 200 * MB);
    DeviceMemoryPlan p(
        builder.build(), /*max_seq_len=*/0, /*activation_seq_len=*/0);
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
