/**
 * @file Test__WorkDistributor.cpp
 * @brief Unit tests for WorkDistributor
 * @author David Sanftenberg
 * @date December 2025
 */

#include <gtest/gtest.h>
#include "execution/WorkDistributor.h"
#include "backends/DeviceId.h"

using namespace llaminar2;

// =============================================================================
// Construction Tests
// =============================================================================

TEST(Test__WorkDistributor, DefaultConstruction)
{
    WorkDistributor dist(1, 0);

    EXPECT_EQ(dist.worldSize(), 1);
    EXPECT_EQ(dist.rank(), 0);
    EXPECT_EQ(dist.deviceCount(), 1);
    EXPECT_FALSE(dist.hasMultipleDevices());
}

TEST(Test__WorkDistributor, ConfigConstruction)
{
    WorkDistributor::Config config{
        .world_size = 4,
        .rank = 2,
        .devices = {0, 1},             // CPU + GPU
        .device_weights = {0.3f, 0.7f} // 30% CPU, 70% GPU
    };

    WorkDistributor dist(config);

    EXPECT_EQ(dist.worldSize(), 4);
    EXPECT_EQ(dist.rank(), 2);
    EXPECT_EQ(dist.deviceCount(), 2);
    EXPECT_TRUE(dist.hasMultipleDevices());
}

TEST(Test__WorkDistributor, InvalidConfigThrows)
{
    // Invalid world_size
    EXPECT_THROW(WorkDistributor(0, 0), std::invalid_argument);

    // Rank out of range
    EXPECT_THROW(WorkDistributor(2, 2), std::invalid_argument);
    EXPECT_THROW(WorkDistributor(2, -1), std::invalid_argument);

    // Mismatched weights
    WorkDistributor::Config bad_config{
        .world_size = 1,
        .rank = 0,
        .devices = {0, 1},
        .device_weights = {1.0f} // Only 1 weight for 2 devices
    };
    EXPECT_THROW({ WorkDistributor dist(bad_config); }, std::invalid_argument);

    // Negative weight
    WorkDistributor::Config neg_weight{
        .world_size = 1,
        .rank = 0,
        .devices = {0},
        .device_weights = {-1.0f}};
    EXPECT_THROW({ WorkDistributor dist(neg_weight); }, std::invalid_argument);
}

// =============================================================================
// Rank-Level Distribution Tests
// =============================================================================

TEST(Test__WorkDistributor, SingleRankGetsAll)
{
    WorkDistributor dist(1, 0);

    auto slice = dist.getRankSlice(1000);

    EXPECT_EQ(slice.start, 0);
    EXPECT_EQ(slice.end, 1000);
    EXPECT_EQ(slice.count, 1000);
    EXPECT_EQ(slice.owner, 0);
}

TEST(Test__WorkDistributor, TwoRanksEvenSplit)
{
    // Rank 0
    WorkDistributor dist0(2, 0);
    auto slice0 = dist0.getRankSlice(1000);
    EXPECT_EQ(slice0.start, 0);
    EXPECT_EQ(slice0.end, 500);
    EXPECT_EQ(slice0.count, 500);

    // Rank 1
    WorkDistributor dist1(2, 1);
    auto slice1 = dist1.getRankSlice(1000);
    EXPECT_EQ(slice1.start, 500);
    EXPECT_EQ(slice1.end, 1000);
    EXPECT_EQ(slice1.count, 500);
}

TEST(Test__WorkDistributor, UnevenSplitDistributesRemainder)
{
    // 1001 elements across 4 ranks: 251, 250, 250, 250
    // Remainder goes to first ranks

    size_t total = 1001;

    WorkDistributor dist0(4, 0);
    auto slice0 = dist0.getRankSlice(total);
    EXPECT_EQ(slice0.count, 251); // Gets 1 extra

    WorkDistributor dist1(4, 1);
    auto slice1 = dist1.getRankSlice(total);
    EXPECT_EQ(slice1.count, 250);

    WorkDistributor dist2(4, 2);
    auto slice2 = dist2.getRankSlice(total);
    EXPECT_EQ(slice2.count, 250);

    WorkDistributor dist3(4, 3);
    auto slice3 = dist3.getRankSlice(total);
    EXPECT_EQ(slice3.count, 250);

    // Verify complete coverage
    EXPECT_EQ(slice0.count + slice1.count + slice2.count + slice3.count, total);
}

TEST(Test__WorkDistributor, AllRankSlicesContiguous)
{
    WorkDistributor dist(4, 0);
    auto slices = dist.getAllRankSlices(4096);

    EXPECT_EQ(slices.size(), 4);

    // Check contiguity
    for (size_t i = 1; i < slices.size(); ++i)
    {
        EXPECT_EQ(slices[i].start, slices[i - 1].end)
            << "Gap between rank " << (i - 1) << " and " << i;
    }

    // Check complete coverage
    EXPECT_EQ(slices[0].start, 0);
    EXPECT_EQ(slices.back().end, 4096);
}

TEST(Test__WorkDistributor, ZeroElementsAllSlicesEmpty)
{
    WorkDistributor dist(4, 1);
    auto slice = dist.getRankSlice(0);

    EXPECT_EQ(slice.count, 0);
    EXPECT_TRUE(slice.empty());
}

TEST(Test__WorkDistributor, RankHasWork)
{
    WorkDistributor dist(4, 2);

    EXPECT_TRUE(dist.rankHasWork(100));
    EXPECT_FALSE(dist.rankHasWork(0));
}

// =============================================================================
// Device-Level Distribution Tests
// =============================================================================

TEST(Test__WorkDistributor, SingleDeviceGetsAll)
{
    WorkDistributor dist(2, 0, DeviceId::cpu()); // Single CPU device

    auto slice = dist.getDeviceSlice(500, DeviceId::cpu());

    EXPECT_EQ(slice.start, 0);
    EXPECT_EQ(slice.end, 500);
    EXPECT_EQ(slice.count, 500);
    EXPECT_EQ(slice.owner, -1); // Legacy index for CPU is -1
}

TEST(Test__WorkDistributor, TwoDevicesEqualWeight)
{
    WorkDistributor::Config config{
        .world_size = 1,
        .rank = 0,
        .devices = {0, 1}, // No weights = equal distribution
        .device_weights = {}};
    WorkDistributor dist(config);

    auto slices = dist.getAllDeviceSlices(1000);

    EXPECT_EQ(slices.size(), 2);
    EXPECT_EQ(slices[0].count, 500);
    EXPECT_EQ(slices[1].count, 500);
}

TEST(Test__WorkDistributor, WeightedDeviceDistribution)
{
    WorkDistributor::Config config{
        .world_size = 1,
        .rank = 0,
        .devices = {0, 1},
        .device_weights = {0.25f, 0.75f} // 25% CPU, 75% GPU
    };
    WorkDistributor dist(config);

    auto slices = dist.getAllDeviceSlices(1000);

    EXPECT_EQ(slices.size(), 2);
    EXPECT_EQ(slices[0].count, 250); // 25%
    EXPECT_EQ(slices[1].count, 750); // 75%
}

TEST(Test__WorkDistributor, DeviceSlicesContiguous)
{
    WorkDistributor::Config config{
        .world_size = 1,
        .rank = 0,
        .devices = {0, 1, 2},
        .device_weights = {}};
    WorkDistributor dist(config);

    auto slices = dist.getAllDeviceSlices(900);

    EXPECT_EQ(slices.size(), 3);
    EXPECT_EQ(slices[0].start, 0);
    EXPECT_EQ(slices[1].start, slices[0].end);
    EXPECT_EQ(slices[2].start, slices[1].end);
    EXPECT_EQ(slices[2].end, 900);
}

TEST(Test__WorkDistributor, GetDeviceForElement)
{
    WorkDistributor::Config config{
        .world_size = 1,
        .rank = 0,
        .devices = {0, 1},
        .device_weights = {} // Equal: 50-50
    };
    WorkDistributor dist(config);

    // Element 0-499 should be device 0
    EXPECT_EQ(dist.getDeviceForElement(0, 1000), 0);
    EXPECT_EQ(dist.getDeviceForElement(499, 1000), 0);

    // Element 500-999 should be device 1
    EXPECT_EQ(dist.getDeviceForElement(500, 1000), 1);
    EXPECT_EQ(dist.getDeviceForElement(999, 1000), 1);
}

// =============================================================================
// Hierarchical Distribution Tests
// =============================================================================

TEST(Test__WorkDistributor, HierarchicalDistribute)
{
    // 2 ranks, 2 GPU devices each (legacy convention: 1=GPU:0, 2=GPU:1)
    WorkDistributor::Config config{
        .world_size = 2,
        .rank = 0,
        .devices = {1, 2}, // Legacy: GPU:0, GPU:1
        .device_weights = {}};
    WorkDistributor dist(config);

    auto slices = dist.distribute(4000);

    // Should get 2 slices (one per device in this rank)
    EXPECT_EQ(slices.size(), 2);

    // Rank 0 gets elements 0-1999
    // Device GPU:0 gets 0-999, Device GPU:1 gets 1000-1999
    EXPECT_EQ(slices[0].rank, 0);
    EXPECT_EQ(slices[0].device.ordinal, 0); // GPU:0 ordinal
    EXPECT_EQ(slices[0].global_start, 0);
    EXPECT_EQ(slices[0].global_end, 1000);
    EXPECT_EQ(slices[0].local_count, 1000);

    EXPECT_EQ(slices[1].rank, 0);
    EXPECT_EQ(slices[1].device.ordinal, 1); // GPU:1 ordinal
    EXPECT_EQ(slices[1].global_start, 1000);
    EXPECT_EQ(slices[1].global_end, 2000);
    EXPECT_EQ(slices[1].local_count, 1000);
}

TEST(Test__WorkDistributor, PrimaryDeviceSlice)
{
    WorkDistributor::Config config{
        .world_size = 2,
        .rank = 1, // Second rank
        .devices = {0, 1},
        .device_weights = {}};
    WorkDistributor dist(config);

    auto primary = dist.getPrimaryDeviceSlice(4000);

    // Rank 1 gets elements 2000-3999
    // Primary device (0) gets first half
    EXPECT_EQ(primary.rank, 1);
    EXPECT_EQ(primary.device.ordinal, 0);
    EXPECT_EQ(primary.global_start, 2000);
    EXPECT_EQ(primary.global_end, 3000);
}

// =============================================================================
// Utility Tests
// =============================================================================

TEST(Test__WorkDistributor, EstimateMemoryPerDevice)
{
    WorkDistributor::Config config{
        .world_size = 2,
        .rank = 0,
        .devices = {0, 1},
        .device_weights = {0.25f, 0.75f}};
    WorkDistributor dist(config);

    // 4GB total, 2 ranks = 2GB per rank
    // Device 0 gets 25% = 500MB, Device 1 gets 75% = 1.5GB
    // Max per device = 1.5GB
    size_t bytes = 4ULL * 1024 * 1024 * 1024;
    size_t per_device = dist.estimateMemoryPerDevice(bytes);

    // Should be approximately 1.5GB (75% of 2GB rank portion)
    size_t expected = static_cast<size_t>(0.75f * (bytes / 2));
    EXPECT_NEAR(static_cast<double>(per_device), static_cast<double>(expected), 1024);
}

TEST(Test__WorkDistributor, GetElementCountsPerDevice)
{
    WorkDistributor::Config config{
        .world_size = 1,
        .rank = 0,
        .devices = {0, 1, 2},
        .device_weights = {0.5f, 0.3f, 0.2f}};
    WorkDistributor dist(config);

    auto counts = dist.getElementCountsPerDevice(1000);

    EXPECT_EQ(counts.size(), 3);
    EXPECT_EQ(counts[0], 500); // 50%
    EXPECT_EQ(counts[1], 300); // 30%
    EXPECT_EQ(counts[2], 200); // 20% (includes rounding remainder)
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST(Test__WorkDistributor, MoreRanksThanElements)
{
    WorkDistributor dist(100, 50); // 100 ranks, element count < 100

    auto slice = dist.getRankSlice(10);

    // Rank 50 should get nothing (only first 10 ranks get 1 element each)
    EXPECT_EQ(slice.count, 0);
    EXPECT_TRUE(slice.empty());
}

TEST(Test__WorkDistributor, SliceContainsMethod)
{
    WorkDistributor dist(2, 0);
    auto slice = dist.getRankSlice(1000); // 0-499

    EXPECT_TRUE(slice.contains(0));
    EXPECT_TRUE(slice.contains(250));
    EXPECT_TRUE(slice.contains(499));
    EXPECT_FALSE(slice.contains(500));
    EXPECT_FALSE(slice.contains(1000));
}
