/**
 * @file Test__MockMPIContext.cpp
 * @brief Unit tests for MockMPIContext
 *
 * Tests the mock MPI context implementation including:
 * - Basic identity (rank, world_size, is_root)
 * - Work distribution
 * - Call tracking
 * - Failure injection
 * - Collective operation simulation
 *
 * @author David Sanftenberg
 * @date January 2026
 *
 * The custom entry owns one lexical native GPU lifetime without probing a
 * device. Native workers/caches retire before process exit and MPI finalization.
 */

#include "backends/GPUDeviceProcessLifetime.h"
#include <gtest/gtest.h>
#include "mocks/MockMPIContext.h"
#include <memory>
#include <vector>
#include <cmath>

using namespace llaminar2;
using namespace llaminar2::test;

// =============================================================================
// Test Fixture
// =============================================================================

class Test__MockMPIContext : public ::testing::Test {
protected:
    void SetUp() override {
        // Default single-rank context
        ctx_single_ = std::make_shared<MockMPIContext>(0, 1);
        
        // Multi-rank context (rank 1 of 4)
        ctx_multi_ = std::make_shared<MockMPIContext>(1, 4);
    }
    
    std::shared_ptr<MockMPIContext> ctx_single_;
    std::shared_ptr<MockMPIContext> ctx_multi_;
};

// =============================================================================
// Identity Tests
// =============================================================================

TEST_F(Test__MockMPIContext, Identity_SingleRank) {
    EXPECT_EQ(ctx_single_->rank(), 0);
    EXPECT_EQ(ctx_single_->world_size(), 1);
    EXPECT_TRUE(ctx_single_->is_root());
}

TEST_F(Test__MockMPIContext, Identity_MultiRank) {
    EXPECT_EQ(ctx_multi_->rank(), 1);
    EXPECT_EQ(ctx_multi_->world_size(), 4);
    EXPECT_FALSE(ctx_multi_->is_root());  // rank 1 is not root
}

TEST_F(Test__MockMPIContext, Identity_RootRank) {
    MockMPIContext root_ctx(0, 4);
    EXPECT_TRUE(root_ctx.is_root());
}

TEST_F(Test__MockMPIContext, Identity_NonRootRanks) {
    for (int r = 1; r < 4; ++r) {
        MockMPIContext ctx(r, 4);
        EXPECT_FALSE(ctx.is_root());
    }
}

// =============================================================================
// Construction Validation Tests
// =============================================================================

TEST_F(Test__MockMPIContext, Construction_InvalidRank_Throws) {
    EXPECT_THROW(MockMPIContext(-1, 4), std::invalid_argument);
    EXPECT_THROW(MockMPIContext(4, 4), std::invalid_argument);  // rank >= world_size
    EXPECT_THROW(MockMPIContext(10, 4), std::invalid_argument);
}

TEST_F(Test__MockMPIContext, Construction_InvalidWorldSize_Throws) {
    EXPECT_THROW(MockMPIContext(0, 0), std::invalid_argument);
    EXPECT_THROW(MockMPIContext(0, -1), std::invalid_argument);
}

TEST_F(Test__MockMPIContext, Construction_ConfigStruct) {
    MockMPIContext::Config config{};
    config.rank = 2;
    config.world_size = 8;
    config.track_calls = true;
    config.simulate_noop = true;
    MockMPIContext ctx(config);
    
    EXPECT_EQ(ctx.rank(), 2);
    EXPECT_EQ(ctx.world_size(), 8);
    EXPECT_FALSE(ctx.is_root());
}

// =============================================================================
// Work Distribution Tests
// =============================================================================

TEST_F(Test__MockMPIContext, WorkDistribution_EvenSplit) {
    // 100 elements across 4 ranks = 25 each
    MockMPIContext rank0(0, 4);
    MockMPIContext rank1(1, 4);
    MockMPIContext rank2(2, 4);
    MockMPIContext rank3(3, 4);
    
    auto [s0, c0] = rank0.get_local_slice(100);
    auto [s1, c1] = rank1.get_local_slice(100);
    auto [s2, c2] = rank2.get_local_slice(100);
    auto [s3, c3] = rank3.get_local_slice(100);
    
    EXPECT_EQ(s0, 0);  EXPECT_EQ(c0, 25);
    EXPECT_EQ(s1, 25); EXPECT_EQ(c1, 25);
    EXPECT_EQ(s2, 50); EXPECT_EQ(c2, 25);
    EXPECT_EQ(s3, 75); EXPECT_EQ(c3, 25);
    
    // Total should be 100
    EXPECT_EQ(c0 + c1 + c2 + c3, 100);
}

TEST_F(Test__MockMPIContext, WorkDistribution_UnevenSplit) {
    // 10 elements across 4 ranks = 3,3,2,2 (first ranks get extra)
    MockMPIContext rank0(0, 4);
    MockMPIContext rank1(1, 4);
    MockMPIContext rank2(2, 4);
    MockMPIContext rank3(3, 4);
    
    auto [s0, c0] = rank0.get_local_slice(10);
    auto [s1, c1] = rank1.get_local_slice(10);
    auto [s2, c2] = rank2.get_local_slice(10);
    auto [s3, c3] = rank3.get_local_slice(10);
    
    // First 2 ranks get 3 elements (10 / 4 = 2 remainder 2)
    EXPECT_EQ(c0, 3);
    EXPECT_EQ(c1, 3);
    EXPECT_EQ(c2, 2);
    EXPECT_EQ(c3, 2);
    
    // Total should be 10
    EXPECT_EQ(c0 + c1 + c2 + c3, 10);
    
    // Slices should be contiguous
    EXPECT_EQ(s0, 0);
    EXPECT_EQ(s1, s0 + c0);
    EXPECT_EQ(s2, s1 + c1);
    EXPECT_EQ(s3, s2 + c2);
}

TEST_F(Test__MockMPIContext, WorkDistribution_SingleRank) {
    auto [start, count] = ctx_single_->get_local_slice(1000);
    EXPECT_EQ(start, 0);
    EXPECT_EQ(count, 1000);  // Single rank gets everything
}

TEST_F(Test__MockMPIContext, WorkDistribution_LessThanRanks) {
    // 2 elements across 4 ranks
    MockMPIContext rank2(2, 4);
    MockMPIContext rank3(3, 4);
    
    auto [s2, c2] = rank2.get_local_slice(2);
    auto [s3, c3] = rank3.get_local_slice(2);
    
    // Last 2 ranks should get 0 elements
    EXPECT_EQ(c2, 0);
    EXPECT_EQ(c3, 0);
}

TEST_F(Test__MockMPIContext, DistributeRows_SameAsGetLocalSlice) {
    auto slice = ctx_multi_->get_local_slice(100);
    auto rows = ctx_multi_->distribute_rows(100);
    
    EXPECT_EQ(slice, rows);  // Should be identical
}

// =============================================================================
// Call Tracking Tests
// =============================================================================

TEST_F(Test__MockMPIContext, CallTracking_InitiallyZero) {
    EXPECT_EQ(ctx_single_->barrier_call_count(), 0);
    EXPECT_EQ(ctx_single_->allreduce_call_count(), 0);
    EXPECT_EQ(ctx_single_->broadcast_call_count(), 0);
    EXPECT_EQ(ctx_single_->allgather_call_count(), 0);
    EXPECT_EQ(ctx_single_->total_collective_calls(), 0);
}

TEST_F(Test__MockMPIContext, CallTracking_Barrier) {
    ctx_single_->barrier();
    EXPECT_EQ(ctx_single_->barrier_call_count(), 1);
    
    ctx_single_->barrier();
    ctx_single_->barrier();
    EXPECT_EQ(ctx_single_->barrier_call_count(), 3);
}

TEST_F(Test__MockMPIContext, CallTracking_AllReduce) {
    std::vector<float> data(100, 1.0f);
    std::vector<float> recv(100);
    
    ctx_single_->allreduce_sum(data.data(), recv.data(), data.size());
    EXPECT_EQ(ctx_single_->allreduce_call_count(), 1);
    
    ctx_single_->allreduce_sum_inplace(data.data(), data.size());
    EXPECT_EQ(ctx_single_->allreduce_call_count(), 2);
}

TEST_F(Test__MockMPIContext, CallTracking_Broadcast) {
    std::vector<float> data(50, 2.0f);
    
    ctx_single_->broadcast(data.data(), data.size(), 0);
    ctx_single_->broadcast(data.data(), data.size(), 0);
    
    EXPECT_EQ(ctx_single_->broadcast_call_count(), 2);
}

TEST_F(Test__MockMPIContext, CallTracking_AllGather) {
    std::vector<float> send(10, 1.0f);
    std::vector<float> recv(10);  // Single rank
    
    ctx_single_->allgather(send.data(), recv.data(), send.size());
    EXPECT_EQ(ctx_single_->allgather_call_count(), 1);
}

TEST_F(Test__MockMPIContext, CallTracking_TotalCollectiveCalls) {
    std::vector<float> data(10, 1.0f);
    
    ctx_single_->barrier();
    ctx_single_->allreduce_sum_inplace(data.data(), data.size());
    ctx_single_->broadcast(data.data(), data.size(), 0);
    
    EXPECT_EQ(ctx_single_->total_collective_calls(), 3);
}

TEST_F(Test__MockMPIContext, CallTracking_Reset) {
    std::vector<float> data(10, 1.0f);
    
    ctx_single_->barrier();
    ctx_single_->allreduce_sum_inplace(data.data(), data.size());
    
    EXPECT_GT(ctx_single_->total_collective_calls(), 0);
    
    ctx_single_->reset_call_counts();
    
    EXPECT_EQ(ctx_single_->barrier_call_count(), 0);
    EXPECT_EQ(ctx_single_->allreduce_call_count(), 0);
    EXPECT_EQ(ctx_single_->total_collective_calls(), 0);
}

// =============================================================================
// Collective Simulation Tests
// =============================================================================

TEST_F(Test__MockMPIContext, Simulation_AllReduce_CopiesData) {
    std::vector<float> send = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> recv(4, 0.0f);
    
    ctx_single_->allreduce_sum(send.data(), recv.data(), send.size());
    
    // In single-rank mock, output should equal input
    EXPECT_EQ(recv, send);
}

TEST_F(Test__MockMPIContext, Simulation_AllGather_ReplicatesData) {
    MockMPIContext ctx(0, 4);  // 4 ranks
    std::vector<float> send = {1.0f, 2.0f};
    std::vector<float> recv(8, 0.0f);  // 4 ranks * 2 elements
    
    ctx.allgather(send.data(), recv.data(), send.size());
    
    // Mock replicates send data for each rank
    EXPECT_EQ(recv[0], 1.0f);
    EXPECT_EQ(recv[1], 2.0f);
    EXPECT_EQ(recv[2], 1.0f);  // Rank 1's data (simulated)
    EXPECT_EQ(recv[3], 2.0f);
    EXPECT_EQ(recv[4], 1.0f);  // Rank 2's data (simulated)
    EXPECT_EQ(recv[5], 2.0f);
    EXPECT_EQ(recv[6], 1.0f);  // Rank 3's data (simulated)
    EXPECT_EQ(recv[7], 2.0f);
}

TEST_F(Test__MockMPIContext, Simulation_AllGatherBytes) {
    MockMPIContext ctx(0, 2);  // 2 ranks
    int32_t send_val = 42;
    std::vector<int32_t> recv(2, 0);
    
    ctx.allgather_bytes(&send_val, recv.data(), sizeof(int32_t));
    
    // Both positions should have the same value (simulated)
    EXPECT_EQ(recv[0], 42);
    EXPECT_EQ(recv[1], 42);
}

// =============================================================================
// Failure Injection Tests
// =============================================================================

TEST_F(Test__MockMPIContext, FailureInjection_Barrier) {
    MockMPIContext::Config config{};
    config.rank = 0;
    config.world_size = 2;
    config.barrier_should_fail = true;
    MockMPIContext ctx(config);
    
    EXPECT_THROW(ctx.barrier(), std::runtime_error);
}

TEST_F(Test__MockMPIContext, FailureInjection_AllReduce) {
    MockMPIContext::Config config{};
    config.rank = 0;
    config.world_size = 2;
    config.allreduce_should_fail = true;
    MockMPIContext ctx(config);
    
    std::vector<float> data(10, 1.0f);
    std::vector<float> recv(10);
    
    EXPECT_THROW(ctx.allreduce_sum(data.data(), recv.data(), data.size()), std::runtime_error);
    EXPECT_THROW(ctx.allreduce_sum_inplace(data.data(), data.size()), std::runtime_error);
}

TEST_F(Test__MockMPIContext, FailureInjection_RuntimeToggle) {
    // Start without failure injection
    ctx_single_->barrier();  // Should not throw
    
    // Enable failure injection
    ctx_single_->set_barrier_fails(true);
    EXPECT_THROW(ctx_single_->barrier(), std::runtime_error);
    
    // Disable failure injection
    ctx_single_->set_barrier_fails(false);
    ctx_single_->barrier();  // Should not throw again
}

TEST_F(Test__MockMPIContext, FailureInjection_AllReduceCorruption) {
    MockMPIContext::Config config{};
    config.rank = 0;
    config.world_size = 1;
    config.allreduce_corruption = 2.0f;  // Multiply results by 2
    MockMPIContext ctx(config);
    
    std::vector<float> send = {1.0f, 2.0f, 3.0f};
    std::vector<float> recv(3, 0.0f);
    
    ctx.allreduce_sum(send.data(), recv.data(), send.size());
    
    // Results should be corrupted (multiplied by 2)
    EXPECT_FLOAT_EQ(recv[0], 2.0f);
    EXPECT_FLOAT_EQ(recv[1], 4.0f);
    EXPECT_FLOAT_EQ(recv[2], 6.0f);
}

TEST_F(Test__MockMPIContext, FailureInjection_CorruptionInplace) {
    ctx_single_->set_allreduce_corruption(0.5f);  // Halve results
    
    std::vector<float> data = {10.0f, 20.0f};
    ctx_single_->allreduce_sum_inplace(data.data(), data.size());
    
    EXPECT_FLOAT_EQ(data[0], 5.0f);
    EXPECT_FLOAT_EQ(data[1], 10.0f);
}

// =============================================================================
// Quantized Operations Tests
// =============================================================================

TEST_F(Test__MockMPIContext, QuantizedOps_Q8_CallTracking) {
    // Note: Q8_1Block is defined in BlockStructures.h
    // Mock just tracks calls, doesn't actually process data
    ctx_single_->allreduce_q8_1_inplace(nullptr, 10);
    
    EXPECT_EQ(ctx_single_->allreduce_q8_call_count(), 1);
}

TEST_F(Test__MockMPIContext, QuantizedOps_Q16_CallTracking) {
    ctx_single_->allreduce_q16_1_inplace(nullptr, 10);
    
    EXPECT_EQ(ctx_single_->allreduce_q16_call_count(), 1);
}

TEST_F(Test__MockMPIContext, QuantizedOps_FP16_CallTracking) {
    ctx_single_->allreduce_fp16_inplace(nullptr, 10);
    
    EXPECT_EQ(ctx_single_->allreduce_fp16_call_count(), 1);
}

TEST_F(Test__MockMPIContext, QuantizedOps_BF16_CallTracking) {
    ctx_single_->allreduce_bf16_inplace(nullptr, 10);
    
    EXPECT_EQ(ctx_single_->allreduce_bf16_call_count(), 1);
}

// =============================================================================
// Interface Compliance Tests
// =============================================================================

TEST_F(Test__MockMPIContext, InterfaceCompliance_CanUseAsIMPIContext) {
    // Verify MockMPIContext can be used through IMPIContext pointer
    std::shared_ptr<IMPIContext> interface_ptr = ctx_single_;
    
    EXPECT_EQ(interface_ptr->rank(), 0);
    EXPECT_EQ(interface_ptr->world_size(), 1);
    EXPECT_TRUE(interface_ptr->is_root());
    
    // Barrier should work through interface
    interface_ptr->barrier();
}

TEST_F(Test__MockMPIContext, InterfaceCompliance_WorkDistributionThroughInterface) {
    std::shared_ptr<IMPIContext> iface = std::make_shared<MockMPIContext>(1, 4);
    
    auto [start, count] = iface->get_local_slice(100);
    
    EXPECT_EQ(start, 25);  // Rank 1 starts at element 25
    EXPECT_EQ(count, 25);  // Gets 25 elements
}

// =============================================================================
// Thread Safety Tests (basic)
// =============================================================================

TEST_F(Test__MockMPIContext, ThreadSafety_ConcurrentCallTracking) {
    const int num_threads = 4;
    const int calls_per_thread = 100;
    
    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([this, calls_per_thread]() {
            for (int i = 0; i < calls_per_thread; ++i) {
                ctx_single_->barrier();
            }
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    // All calls should be tracked
    EXPECT_EQ(ctx_single_->barrier_call_count(), 
              static_cast<size_t>(num_threads * calls_per_thread));
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    llaminar2::GPUDeviceProcessLifetime gpu_process_lifetime;

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
