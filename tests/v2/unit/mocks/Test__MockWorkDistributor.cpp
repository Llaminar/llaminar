/**
 * @file Test__MockWorkDistributor.cpp
 * @brief Unit tests for MockWorkDistributor
 *
 * Tests the mock work distributor implementation including:
 * - Basic configuration (rank, world_size, devices)
 * - Rank-level work distribution
 * - Device-level work distribution
 * - Hierarchical distribution
 * - MoE expert distribution
 * - Call tracking
 * - Builder pattern
 * - Preset factory methods
 *
 * @author David Sanftenberg
 * @date January 2026
 *
 * The custom entry owns one lexical native GPU lifetime without probing a
 * device. Native workers/caches retire before process exit and MPI finalization.
 */

#include "backends/GPUDeviceProcessLifetime.h"
#include <gtest/gtest.h>
#include "mocks/MockWorkDistributor.h"
#include <memory>
#include <vector>
#include <thread>

using namespace llaminar2;
using namespace llaminar2::test;

// =============================================================================
// Test Fixture
// =============================================================================

class Test__MockWorkDistributor : public ::testing::Test {
protected:
    void SetUp() override {
        // Default single-rank mock
        single_ = MockWorkDistributor::singleRank();
        
        // Multi-rank tensor parallel mock (rank 1 of 4)
        tensor_parallel_ = MockWorkDistributor::tensorParallel(1, 4);
        
        // Heterogeneous mock (CPU + 2 GPUs)
        hetero_ = MockWorkDistributor::heterogeneous({
            DeviceId::cpu(),
            DeviceId::cuda(0),
            DeviceId::cuda(1)
        });
    }
    
    std::shared_ptr<MockWorkDistributor> single_;
    std::shared_ptr<MockWorkDistributor> tensor_parallel_;
    std::shared_ptr<MockWorkDistributor> hetero_;
};

// =============================================================================
// Basic Configuration Tests
// =============================================================================

TEST_F(Test__MockWorkDistributor, Config_SingleRank) {
    EXPECT_EQ(single_->rank(), 0);
    EXPECT_EQ(single_->worldSize(), 1);
    EXPECT_EQ(single_->deviceCount(), 1);
    EXPECT_FALSE(single_->hasMultipleDevices());
    EXPECT_TRUE(single_->devices()[0].is_cpu());
}

TEST_F(Test__MockWorkDistributor, Config_TensorParallel) {
    EXPECT_EQ(tensor_parallel_->rank(), 1);
    EXPECT_EQ(tensor_parallel_->worldSize(), 4);
    EXPECT_EQ(tensor_parallel_->deviceCount(), 1);
    EXPECT_FALSE(tensor_parallel_->hasMultipleDevices());
}

TEST_F(Test__MockWorkDistributor, Config_Heterogeneous) {
    EXPECT_EQ(hetero_->rank(), 0);
    EXPECT_EQ(hetero_->worldSize(), 1);
    EXPECT_EQ(hetero_->deviceCount(), 3);
    EXPECT_TRUE(hetero_->hasMultipleDevices());
    
    const auto& devices = hetero_->devices();
    EXPECT_TRUE(devices[0].is_cpu());
    EXPECT_TRUE(devices[1].is_cuda());
    EXPECT_EQ(devices[1].ordinal, 0);
    EXPECT_TRUE(devices[2].is_cuda());
    EXPECT_EQ(devices[2].ordinal, 1);
}

// =============================================================================
// Construction Validation Tests
// =============================================================================

TEST_F(Test__MockWorkDistributor, Construction_InvalidRank_Throws) {
    EXPECT_THROW(MockWorkDistributor(-1, 4), std::invalid_argument);
    EXPECT_THROW(MockWorkDistributor(4, 4), std::invalid_argument);  // rank >= world_size
    EXPECT_THROW(MockWorkDistributor(10, 4), std::invalid_argument);
}

TEST_F(Test__MockWorkDistributor, Construction_InvalidWorldSize_Throws) {
    EXPECT_THROW(MockWorkDistributor(0, 0), std::invalid_argument);
    EXPECT_THROW(MockWorkDistributor(0, -1), std::invalid_argument);
}

TEST_F(Test__MockWorkDistributor, Construction_EmptyDevices_DefaultsToCPU) {
    MockWorkDistributor::Config config;
    config.rank = 0;
    config.world_size = 1;
    config.devices.clear();
    
    // Empty devices should default to CPU, not throw
    MockWorkDistributor mock(config);
    EXPECT_EQ(mock.deviceCount(), 1);
    EXPECT_TRUE(mock.devices()[0].is_cpu());
}

TEST_F(Test__MockWorkDistributor, Construction_MismatchedWeights_Throws) {
    // Note: This test validates that mismatched weights throw
    // when explicitly provided (not when empty/default)
    auto builder = MockWorkDistributor::Builder()
        .withRank(0)
        .withWorldSize(1)
        .withDevices({DeviceId::cpu(), DeviceId::cuda(0)})
        .withDeviceWeights({1.0f});  // Only 1 weight for 2 devices
    
    EXPECT_THROW(builder.build(), std::invalid_argument);
}

// =============================================================================
// Builder Pattern Tests
// =============================================================================

TEST_F(Test__MockWorkDistributor, Builder_BasicConfiguration) {
    auto mock = MockWorkDistributor::Builder()
        .withRank(2)
        .withWorldSize(8)
        .build();
    
    EXPECT_EQ(mock->rank(), 2);
    EXPECT_EQ(mock->worldSize(), 8);
}

TEST_F(Test__MockWorkDistributor, Builder_DeviceConfiguration) {
    // Use withDevices for explicit device list
    auto mock = MockWorkDistributor::Builder()
        .withRank(0)
        .withWorldSize(1)
        .withDevices({DeviceId::cpu(), DeviceId::cuda(0), DeviceId::cuda(1)})
        .build();
    
    EXPECT_EQ(mock->deviceCount(), 3);
    EXPECT_TRUE(mock->devices()[0].is_cpu());
    EXPECT_TRUE(mock->devices()[1].is_cuda());
    EXPECT_TRUE(mock->devices()[2].is_cuda());
}

TEST_F(Test__MockWorkDistributor, Builder_DeviceWeights) {
    auto mock = MockWorkDistributor::Builder()
        .withRank(0)
        .withWorldSize(1)
        .withDevices({DeviceId::cpu(), DeviceId::cuda(0)})
        .withDeviceWeights({0.25f, 0.75f})  // CPU gets 25%, GPU gets 75%
        .build();
    
    auto slices = mock->getAllDeviceSlices(100);
    EXPECT_EQ(slices.size(), 2);
    EXPECT_EQ(slices[0].count, 25);  // CPU: 25%
    EXPECT_EQ(slices[1].count, 75);  // GPU: 75%
}

TEST_F(Test__MockWorkDistributor, Builder_CustomRankSlice) {
    IWorkDistributor::WorkSlice custom{100, 200, 100, 0};
    
    auto mock = MockWorkDistributor::Builder()
        .withRank(0)
        .withWorldSize(4)
        .withCustomRankSlice(custom)
        .build();
    
    auto slice = mock->getRankSlice(1000);  // Would normally be 250 elements
    EXPECT_EQ(slice.start, 100);
    EXPECT_EQ(slice.end, 200);
    EXPECT_EQ(slice.count, 100);
}

TEST_F(Test__MockWorkDistributor, Builder_CustomDeviceSlices) {
    std::vector<IWorkDistributor::WorkSlice> custom = {
        {0, 30, 30, 0},
        {30, 100, 70, 1}
    };
    
    auto mock = MockWorkDistributor::Builder()
        .withDevices({DeviceId::cpu(), DeviceId::cuda(0)})
        .withCustomDeviceSlices(custom)
        .build();
    
    auto slices = mock->getAllDeviceSlices(100);
    EXPECT_EQ(slices[0].count, 30);
    EXPECT_EQ(slices[1].count, 70);
}

TEST_F(Test__MockWorkDistributor, Builder_CallTrackingToggle) {
    auto mock = MockWorkDistributor::Builder()
        .withCallTracking(false)
        .build();
    
    mock->getRankSlice(100);
    mock->getRankSlice(100);
    
    // Call tracking disabled - should remain 0
    EXPECT_EQ(mock->get_rank_slice_call_count(), 0);
}

// =============================================================================
// Preset Factory Method Tests
// =============================================================================

TEST_F(Test__MockWorkDistributor, Preset_SingleRank) {
    auto mock = MockWorkDistributor::singleRank();
    EXPECT_EQ(mock->rank(), 0);
    EXPECT_EQ(mock->worldSize(), 1);
    EXPECT_EQ(mock->deviceCount(), 1);
}

TEST_F(Test__MockWorkDistributor, Preset_TensorParallel) {
    for (int r = 0; r < 4; ++r) {
        auto mock = MockWorkDistributor::tensorParallel(r, 4);
        EXPECT_EQ(mock->rank(), r);
        EXPECT_EQ(mock->worldSize(), 4);
    }
}

TEST_F(Test__MockWorkDistributor, Preset_Heterogeneous) {
    auto mock = MockWorkDistributor::heterogeneous({
        DeviceId::cpu(),
        DeviceId::cuda(0)
    }, {0.3f, 0.7f});
    
    EXPECT_EQ(mock->deviceCount(), 2);
    
    auto slices = mock->getAllDeviceSlices(100);
    EXPECT_EQ(slices[0].count, 30);
    EXPECT_EQ(slices[1].count, 70);
}

TEST_F(Test__MockWorkDistributor, Preset_MultiGPU) {
    auto mock = MockWorkDistributor::multiGPU(4);
    EXPECT_EQ(mock->deviceCount(), 4);
    
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_TRUE(mock->devices()[i].is_cuda());
        EXPECT_EQ(mock->devices()[i].ordinal, static_cast<int>(i));
    }
}

TEST_F(Test__MockWorkDistributor, Preset_Distributed) {
    auto mock = MockWorkDistributor::distributed(2, 4, {
        DeviceId::cpu(),
        DeviceId::cuda(0)
    });
    
    EXPECT_EQ(mock->rank(), 2);
    EXPECT_EQ(mock->worldSize(), 4);
    EXPECT_EQ(mock->deviceCount(), 2);
}

// =============================================================================
// Rank-Level Distribution Tests
// =============================================================================

TEST_F(Test__MockWorkDistributor, RankSlice_SingleRank_GetsAll) {
    auto slice = single_->getRankSlice(1000);
    EXPECT_EQ(slice.start, 0);
    EXPECT_EQ(slice.end, 1000);
    EXPECT_EQ(slice.count, 1000);
    EXPECT_EQ(slice.owner, 0);
}

TEST_F(Test__MockWorkDistributor, RankSlice_EvenSplit) {
    // 100 elements across 4 ranks = 25 each
    for (int r = 0; r < 4; ++r) {
        auto mock = MockWorkDistributor::tensorParallel(r, 4);
        auto slice = mock->getRankSlice(100);
        EXPECT_EQ(slice.count, 25) << "Rank " << r;
        EXPECT_EQ(slice.owner, r);
    }
}

TEST_F(Test__MockWorkDistributor, RankSlice_UnevenSplit) {
    // 10 elements across 4 ranks = 3,3,2,2
    auto mock0 = MockWorkDistributor::tensorParallel(0, 4);
    auto mock1 = MockWorkDistributor::tensorParallel(1, 4);
    auto mock2 = MockWorkDistributor::tensorParallel(2, 4);
    auto mock3 = MockWorkDistributor::tensorParallel(3, 4);
    
    EXPECT_EQ(mock0->getRankSlice(10).count, 3);
    EXPECT_EQ(mock1->getRankSlice(10).count, 3);
    EXPECT_EQ(mock2->getRankSlice(10).count, 2);
    EXPECT_EQ(mock3->getRankSlice(10).count, 2);
}

TEST_F(Test__MockWorkDistributor, RankSlice_ContiguousSlices) {
    auto all_slices = tensor_parallel_->getAllRankSlices(100);
    EXPECT_EQ(all_slices.size(), 4);
    
    // Verify slices are contiguous
    size_t prev_end = 0;
    for (const auto& slice : all_slices) {
        EXPECT_EQ(slice.start, prev_end);
        EXPECT_EQ(slice.end, slice.start + slice.count);
        prev_end = slice.end;
    }
    EXPECT_EQ(prev_end, 100);
}

TEST_F(Test__MockWorkDistributor, RankSlice_LessThanRanks) {
    // 2 elements across 4 ranks
    auto mock2 = MockWorkDistributor::tensorParallel(2, 4);
    auto mock3 = MockWorkDistributor::tensorParallel(3, 4);
    
    EXPECT_EQ(mock2->getRankSlice(2).count, 0);
    EXPECT_EQ(mock3->getRankSlice(2).count, 0);
}

TEST_F(Test__MockWorkDistributor, RankHasWork_True) {
    EXPECT_TRUE(single_->rankHasWork(100));
    EXPECT_TRUE(tensor_parallel_->rankHasWork(100));
}

TEST_F(Test__MockWorkDistributor, RankHasWork_False) {
    // Rank 3 of 4 with only 2 elements
    auto mock = MockWorkDistributor::tensorParallel(3, 4);
    EXPECT_FALSE(mock->rankHasWork(2));
}

// =============================================================================
// Device-Level Distribution Tests
// =============================================================================

TEST_F(Test__MockWorkDistributor, DeviceSlice_SingleDevice_GetsAll) {
    auto slice = single_->getDeviceSlice(100, DeviceId::cpu());
    EXPECT_EQ(slice.count, 100);
}

TEST_F(Test__MockWorkDistributor, DeviceSlice_EvenSplit) {
    // 3 devices, 99 elements = 33 each
    auto slices = hetero_->getAllDeviceSlices(99);
    EXPECT_EQ(slices.size(), 3);
    EXPECT_EQ(slices[0].count, 33);
    EXPECT_EQ(slices[1].count, 33);
    EXPECT_EQ(slices[2].count, 33);
}

TEST_F(Test__MockWorkDistributor, DeviceSlice_UnevenSplit) {
    // 3 devices, 100 elements
    auto slices = hetero_->getAllDeviceSlices(100);
    
    // Should handle remainder
    size_t total = 0;
    for (const auto& s : slices) {
        total += s.count;
    }
    EXPECT_EQ(total, 100);
}

TEST_F(Test__MockWorkDistributor, DeviceSlice_WeightedDistribution) {
    auto mock = MockWorkDistributor::Builder()
        .withDevices({DeviceId::cpu(), DeviceId::cuda(0)})
        .withDeviceWeights({0.2f, 0.8f})
        .build();
    
    auto slices = mock->getAllDeviceSlices(1000);
    EXPECT_EQ(slices[0].count, 200);
    EXPECT_EQ(slices[1].count, 800);
}

TEST_F(Test__MockWorkDistributor, DeviceSlice_UnknownDevice_ReturnsEmpty) {
    auto slice = single_->getDeviceSlice(100, DeviceId::cuda(99));
    EXPECT_TRUE(slice.empty());
    EXPECT_EQ(slice.owner, -1);
}

TEST_F(Test__MockWorkDistributor, GetDeviceForElement_CorrectDevice) {
    // With 3 even devices and 99 elements:
    // Device 0: 0-32, Device 1: 33-65, Device 2: 66-98
    EXPECT_EQ(hetero_->getDeviceForElement(0, 99), 0);
    EXPECT_EQ(hetero_->getDeviceForElement(32, 99), 0);
    EXPECT_EQ(hetero_->getDeviceForElement(33, 99), 1);
    EXPECT_EQ(hetero_->getDeviceForElement(65, 99), 1);
    EXPECT_EQ(hetero_->getDeviceForElement(66, 99), 2);
    EXPECT_EQ(hetero_->getDeviceForElement(98, 99), 2);
}

// =============================================================================
// Hierarchical Distribution Tests
// =============================================================================

TEST_F(Test__MockWorkDistributor, Distribute_SingleRankSingleDevice) {
    auto slices = single_->distribute(1000);
    EXPECT_EQ(slices.size(), 1);
    
    const auto& s = slices[0];
    EXPECT_EQ(s.rank, 0);
    EXPECT_TRUE(s.device.is_cpu());
    EXPECT_EQ(s.global_start, 0);
    EXPECT_EQ(s.global_end, 1000);
    EXPECT_EQ(s.local_start, 0);
    EXPECT_EQ(s.local_count, 1000);
}

TEST_F(Test__MockWorkDistributor, Distribute_MultiRankSingleDevice) {
    // Rank 1 of 4 with 100 elements should get global 25-50
    auto slices = tensor_parallel_->distribute(100);
    EXPECT_EQ(slices.size(), 1);
    
    const auto& s = slices[0];
    EXPECT_EQ(s.rank, 1);
    EXPECT_EQ(s.global_start, 25);
    EXPECT_EQ(s.global_end, 50);
    EXPECT_EQ(s.local_count, 25);
}

TEST_F(Test__MockWorkDistributor, Distribute_Heterogeneous) {
    auto slices = hetero_->distribute(99);
    EXPECT_EQ(slices.size(), 3);
    
    // All should be on rank 0
    for (const auto& s : slices) {
        EXPECT_EQ(s.rank, 0);
    }
    
    // Global positions should cover [0, 99)
    EXPECT_EQ(slices[0].global_start, 0);
    EXPECT_EQ(slices[2].global_end, 99);
}

TEST_F(Test__MockWorkDistributor, PrimaryDeviceSlice_ReturnsFirst) {
    auto primary = hetero_->getPrimaryDeviceSlice(99);
    EXPECT_EQ(primary.rank, 0);
    EXPECT_TRUE(primary.device.is_cpu());
    EXPECT_EQ(primary.local_count, 33);
}

// =============================================================================
// MoE Expert Distribution Tests
// =============================================================================

TEST_F(Test__MockWorkDistributor, DistributeExperts_SingleDevice) {
    auto assignments = single_->distributeExperts(8);
    EXPECT_EQ(assignments.size(), 8);
    
    for (const auto& ea : assignments) {
        EXPECT_EQ(ea.rank, 0);
        EXPECT_TRUE(ea.device.is_cpu());
    }
}

TEST_F(Test__MockWorkDistributor, DistributeExperts_MultiDevice) {
    // 3 devices, 8 experts - round robin
    auto assignments = hetero_->distributeExperts(8);
    EXPECT_EQ(assignments.size(), 8);
    
    // Experts 0,3,6 on device 0; 1,4,7 on device 1; 2,5 on device 2
    int device0_count = 0, device1_count = 0, device2_count = 0;
    for (const auto& ea : assignments) {
        if (ea.device.is_cpu()) device0_count++;
        else if (ea.device.ordinal == 0) device1_count++;
        else if (ea.device.ordinal == 1) device2_count++;
    }
    
    // Should be roughly evenly distributed
    EXPECT_GE(device0_count, 2);
    EXPECT_GE(device1_count, 2);
    EXPECT_GE(device2_count, 2);
}

TEST_F(Test__MockWorkDistributor, RouteTokensToExperts_Basic) {
    // Create simple router output: 4 tokens, 2 experts
    // Token 0 prefers expert 0, Token 1 prefers expert 1, etc.
    std::vector<float> router_output = {
        0.9f, 0.1f,  // Token 0: expert 0
        0.1f, 0.9f,  // Token 1: expert 1
        0.8f, 0.2f,  // Token 2: expert 0
        0.2f, 0.8f   // Token 3: expert 1
    };
    
    auto assignments = single_->distributeExperts(2);
    auto routing = single_->routeTokensToExperts(
        router_output.data(), assignments, 1, 4, 2);
    
    // Should have 2 routing entries (one per expert)
    EXPECT_EQ(routing.size(), 2);
}

TEST_F(Test__MockWorkDistributor, GetExpertsForDevice_Static) {
    std::vector<IWorkDistributor::ExpertAssignment> assignments = {
        {0, DeviceId::cpu(), 0},
        {1, DeviceId::cuda(0), 0},
        {2, DeviceId::cpu(), 0},
        {3, DeviceId::cuda(0), 0}
    };
    
    auto cpu_experts = IWorkDistributor::getExpertsForDevice(assignments, DeviceId::cpu());
    auto gpu_experts = IWorkDistributor::getExpertsForDevice(assignments, DeviceId::cuda(0));
    
    EXPECT_EQ(cpu_experts.size(), 2);
    EXPECT_EQ(gpu_experts.size(), 2);
    EXPECT_EQ(cpu_experts[0], 0);
    EXPECT_EQ(cpu_experts[1], 2);
    EXPECT_EQ(gpu_experts[0], 1);
    EXPECT_EQ(gpu_experts[1], 3);
}

// =============================================================================
// Utility Method Tests
// =============================================================================

TEST_F(Test__MockWorkDistributor, EstimateMemoryPerDevice_Single) {
    EXPECT_EQ(single_->estimateMemoryPerDevice(1000), 1000);
}

TEST_F(Test__MockWorkDistributor, EstimateMemoryPerDevice_MultiRank) {
    // 4 ranks, 1 device each = 250 per device
    EXPECT_EQ(tensor_parallel_->estimateMemoryPerDevice(1000), 250);
}

TEST_F(Test__MockWorkDistributor, EstimateMemoryPerDevice_MultiDevice) {
    // 1 rank, 3 devices = 333 per device
    EXPECT_EQ(hetero_->estimateMemoryPerDevice(1000), 333);
}

TEST_F(Test__MockWorkDistributor, GetElementCountsPerDevice) {
    auto counts = hetero_->getElementCountsPerDevice(99);
    EXPECT_EQ(counts.size(), 3);
    
    size_t total = 0;
    for (size_t c : counts) {
        total += c;
    }
    EXPECT_EQ(total, 99);
}

// =============================================================================
// Call Tracking Tests
// =============================================================================

TEST_F(Test__MockWorkDistributor, CallTracking_InitiallyZero) {
    EXPECT_EQ(single_->get_rank_slice_call_count(), 0);
    EXPECT_EQ(single_->get_all_rank_slices_call_count(), 0);
    EXPECT_EQ(single_->rank_has_work_call_count(), 0);
    EXPECT_EQ(single_->get_device_slice_call_count(), 0);
    EXPECT_EQ(single_->get_all_device_slices_call_count(), 0);
    EXPECT_EQ(single_->total_call_count(), 0);
}

TEST_F(Test__MockWorkDistributor, CallTracking_RankSlice) {
    single_->getRankSlice(100);
    EXPECT_EQ(single_->get_rank_slice_call_count(), 1);
    
    single_->getRankSlice(200);
    single_->getRankSlice(300);
    EXPECT_EQ(single_->get_rank_slice_call_count(), 3);
}

TEST_F(Test__MockWorkDistributor, CallTracking_AllMethods) {
    single_->getRankSlice(100);
    single_->getAllRankSlices(100);
    single_->rankHasWork(100);
    single_->getDeviceSlice(100, DeviceId::cpu());
    single_->getAllDeviceSlices(100);
    single_->getDeviceForElement(50, 100);
    single_->distribute(100);
    single_->getPrimaryDeviceSlice(100);
    single_->distributeExperts(8);
    single_->estimateMemoryPerDevice(1000);
    single_->getElementCountsPerDevice(100);
    
    // Note: Some methods call others internally, so we verify >= 1
    EXPECT_GE(single_->get_rank_slice_call_count(), 1);
    EXPECT_GE(single_->get_all_rank_slices_call_count(), 1);
    EXPECT_GE(single_->rank_has_work_call_count(), 1);
    EXPECT_GE(single_->get_device_slice_call_count(), 1);
    // getAllDeviceSlices called multiple times internally
    EXPECT_GE(single_->get_all_device_slices_call_count(), 1);
    EXPECT_GE(single_->get_device_for_element_call_count(), 1);
    EXPECT_GE(single_->distribute_call_count(), 1);
    EXPECT_GE(single_->get_primary_device_slice_call_count(), 1);
    EXPECT_GE(single_->distribute_experts_call_count(), 1);
    EXPECT_GE(single_->estimate_memory_call_count(), 1);
    EXPECT_GE(single_->get_element_counts_call_count(), 1);
}

TEST_F(Test__MockWorkDistributor, CallTracking_Reset) {
    single_->getRankSlice(100);
    single_->getAllRankSlices(100);
    
    EXPECT_GT(single_->total_call_count(), 0);
    
    single_->reset_call_counts();
    
    EXPECT_EQ(single_->get_rank_slice_call_count(), 0);
    EXPECT_EQ(single_->get_all_rank_slices_call_count(), 0);
    EXPECT_EQ(single_->total_call_count(), 0);
}

// =============================================================================
// Runtime Override Tests
// =============================================================================

TEST_F(Test__MockWorkDistributor, RuntimeOverride_CustomRankSlice) {
    // Initially uses computed slice
    auto slice1 = single_->getRankSlice(100);
    EXPECT_EQ(slice1.count, 100);
    
    // Set custom slice
    single_->setCustomRankSlice({10, 20, 10, 0});
    auto slice2 = single_->getRankSlice(100);
    EXPECT_EQ(slice2.count, 10);
    
    // Clear override
    single_->clearCustomRankSlice();
    auto slice3 = single_->getRankSlice(100);
    EXPECT_EQ(slice3.count, 100);
}

TEST_F(Test__MockWorkDistributor, RuntimeOverride_CustomDeviceSlices) {
    auto slices1 = hetero_->getAllDeviceSlices(99);
    EXPECT_EQ(slices1[0].count, 33);
    
    hetero_->setCustomDeviceSlices({
        {0, 50, 50, 0},
        {50, 75, 25, 1},
        {75, 99, 24, 2}
    });
    
    auto slices2 = hetero_->getAllDeviceSlices(99);
    EXPECT_EQ(slices2[0].count, 50);
    EXPECT_EQ(slices2[1].count, 25);
    EXPECT_EQ(slices2[2].count, 24);
    
    hetero_->clearCustomDeviceSlices();
    auto slices3 = hetero_->getAllDeviceSlices(99);
    EXPECT_EQ(slices3[0].count, 33);
}

// =============================================================================
// Interface Compliance Tests
// =============================================================================

TEST_F(Test__MockWorkDistributor, InterfaceCompliance_CanUseAsIWorkDistributor) {
    std::shared_ptr<IWorkDistributor> iface = single_;
    
    EXPECT_EQ(iface->rank(), 0);
    EXPECT_EQ(iface->worldSize(), 1);
    
    auto slice = iface->getRankSlice(100);
    EXPECT_EQ(slice.count, 100);
}

TEST_F(Test__MockWorkDistributor, InterfaceCompliance_WorkSliceEquality) {
    IWorkDistributor::WorkSlice a{0, 100, 100, 0};
    IWorkDistributor::WorkSlice b{0, 100, 100, 0};
    IWorkDistributor::WorkSlice c{0, 50, 50, 0};
    
    EXPECT_EQ(a, b);
    EXPECT_FALSE(a == c);
}

TEST_F(Test__MockWorkDistributor, InterfaceCompliance_HierarchicalSliceEquality) {
    IWorkDistributor::HierarchicalSlice a{0, DeviceId::cpu(), 0, 100, 0, 100};
    IWorkDistributor::HierarchicalSlice b{0, DeviceId::cpu(), 0, 100, 0, 100};
    IWorkDistributor::HierarchicalSlice c{1, DeviceId::cpu(), 0, 100, 0, 100};
    
    EXPECT_EQ(a, b);
    EXPECT_FALSE(a == c);
}

// =============================================================================
// Thread Safety Tests
// =============================================================================

TEST_F(Test__MockWorkDistributor, ThreadSafety_ConcurrentCallTracking) {
    const int num_threads = 4;
    const int calls_per_thread = 100;
    
    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([this, calls_per_thread]() {
            for (int i = 0; i < calls_per_thread; ++i) {
                single_->getRankSlice(100);
            }
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    EXPECT_EQ(single_->get_rank_slice_call_count(),
              static_cast<size_t>(num_threads * calls_per_thread));
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST_F(Test__MockWorkDistributor, EdgeCase_ZeroElements) {
    auto slice = single_->getRankSlice(0);
    EXPECT_TRUE(slice.empty());
    EXPECT_EQ(slice.count, 0);
}

TEST_F(Test__MockWorkDistributor, EdgeCase_SingleElement) {
    // 1 element across 4 ranks - only rank 0 gets it
    auto mock0 = MockWorkDistributor::tensorParallel(0, 4);
    auto mock1 = MockWorkDistributor::tensorParallel(1, 4);
    
    EXPECT_EQ(mock0->getRankSlice(1).count, 1);
    EXPECT_EQ(mock1->getRankSlice(1).count, 0);
}

TEST_F(Test__MockWorkDistributor, EdgeCase_WorkSliceContains) {
    IWorkDistributor::WorkSlice slice{10, 20, 10, 0};
    
    EXPECT_FALSE(slice.contains(9));
    EXPECT_TRUE(slice.contains(10));
    EXPECT_TRUE(slice.contains(15));
    EXPECT_TRUE(slice.contains(19));
    EXPECT_FALSE(slice.contains(20));
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    llaminar2::GPUDeviceProcessLifetime gpu_process_lifetime;

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
