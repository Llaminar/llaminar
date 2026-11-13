/**
 * @file Test__MockCollectiveContext.cpp
 * @brief Unit tests for MockCollectiveContext
 *
 * Tests the mock collective context implementation including:
 * - Basic identity (rank, world_size, requiresCollectives)
 * - Builder pattern configuration
 * - Presets (singleRankCPU, multiRankCPU, failingContext)
 * - Call tracking for allreduce, allgather, broadcast
 * - Failure injection
 * - Backend availability configuration
 * - Custom hooks
 *
 * @author David Sanftenberg
 * @date January 2026
 *
 * The custom entry owns one lexical native GPU lifetime without probing a
 * device. Native workers/caches retire before process exit and MPI finalization.
 */

#include "backends/GPUDeviceProcessLifetime.h"
#include <gtest/gtest.h>
#include "mocks/MockCollectiveContext.h"
#include <memory>
#include <vector>
#include <thread>

using namespace llaminar2;
using namespace llaminar2::test;

// =============================================================================
// Test Fixture
// =============================================================================

class Test__MockCollectiveContext : public ::testing::Test {
protected:
    void SetUp() override {
        // Default single-rank context
        ctx_single_ = MockCollectiveContext::singleRankCPU();
        
        // Multi-rank context (rank 1 of 4)
        ctx_multi_ = MockCollectiveContext::multiRankCPU(1, 4);
    }
    
    std::unique_ptr<MockCollectiveContext> ctx_single_;
    std::unique_ptr<MockCollectiveContext> ctx_multi_;
};

// =============================================================================
// Identity Tests
// =============================================================================

TEST_F(Test__MockCollectiveContext, Identity_SingleRank) {
    EXPECT_EQ(ctx_single_->rank(), 0);
    EXPECT_EQ(ctx_single_->worldSize(), 1);
    EXPECT_FALSE(ctx_single_->requiresCollectives());  // world_size == 1
}

TEST_F(Test__MockCollectiveContext, Identity_MultiRank) {
    EXPECT_EQ(ctx_multi_->rank(), 1);
    EXPECT_EQ(ctx_multi_->worldSize(), 4);
    EXPECT_TRUE(ctx_multi_->requiresCollectives());  // world_size > 1
}

TEST_F(Test__MockCollectiveContext, Identity_RootRank) {
    auto root_ctx = MockCollectiveContext::multiRankCPU(0, 4);
    EXPECT_EQ(root_ctx->rank(), 0);
    EXPECT_TRUE(root_ctx->requiresCollectives());
}

TEST_F(Test__MockCollectiveContext, Identity_NonRootRanks) {
    for (int r = 1; r < 4; ++r) {
        auto ctx = MockCollectiveContext::multiRankCPU(r, 4);
        EXPECT_EQ(ctx->rank(), r);
        EXPECT_TRUE(ctx->requiresCollectives());
    }
}

TEST_F(Test__MockCollectiveContext, Identity_LocalDevices) {
    EXPECT_EQ(ctx_single_->localDevices().size(), 1);
    EXPECT_EQ(ctx_single_->localDevices()[0].type, DeviceType::CPU);
}

// =============================================================================
// Construction Validation Tests
// =============================================================================

TEST_F(Test__MockCollectiveContext, Construction_InvalidRank_Throws) {
    EXPECT_THROW(MockCollectiveContext(-1, 4), std::invalid_argument);
    EXPECT_THROW(MockCollectiveContext(4, 4), std::invalid_argument);  // rank >= world_size
    EXPECT_THROW(MockCollectiveContext(10, 4), std::invalid_argument);
}

TEST_F(Test__MockCollectiveContext, Construction_InvalidWorldSize_Throws) {
    EXPECT_THROW(MockCollectiveContext(0, 0), std::invalid_argument);
    EXPECT_THROW(MockCollectiveContext(0, -1), std::invalid_argument);
}

TEST_F(Test__MockCollectiveContext, Construction_ConfigStruct) {
    MockCollectiveContext::Config config{};
    config.rank = 2;
    config.world_size = 8;
    config.local_devices = {DeviceId::cpu()};
    config.track_calls = true;
    MockCollectiveContext ctx(config);
    
    EXPECT_EQ(ctx.rank(), 2);
    EXPECT_EQ(ctx.worldSize(), 8);
    EXPECT_TRUE(ctx.requiresCollectives());
}

TEST_F(Test__MockCollectiveContext, Construction_DefaultDevice) {
    // If no devices specified, should default to CPU(0)
    MockCollectiveContext::Config config{};
    config.rank = 0;
    config.world_size = 1;
    MockCollectiveContext ctx(config);
    
    EXPECT_EQ(ctx.localDevices().size(), 1);
    EXPECT_EQ(ctx.localDevices()[0].type, DeviceType::CPU);
}

// =============================================================================
// Builder Pattern Tests
// =============================================================================

TEST_F(Test__MockCollectiveContext, Builder_BasicConfiguration) {
    auto ctx = MockCollectiveContext::Builder()
        .withRank(2)
        .withWorldSize(8)
        .withDevice(DeviceId::cpu())
        .build();
    
    EXPECT_EQ(ctx->rank(), 2);
    EXPECT_EQ(ctx->worldSize(), 8);
    EXPECT_EQ(ctx->localDevices().size(), 1);
}

TEST_F(Test__MockCollectiveContext, Builder_MultipleDevices) {
    auto ctx = MockCollectiveContext::Builder()
        .withRank(0)
        .withWorldSize(1)
        .withDevice(DeviceId::cpu())
        .withDevice(DeviceId::cuda(0))
        .withDevice(DeviceId::cuda(1))
        .build();
    
    EXPECT_EQ(ctx->localDevices().size(), 3);
    EXPECT_EQ(ctx->localDevices()[0].type, DeviceType::CPU);
    EXPECT_EQ(ctx->localDevices()[1].type, DeviceType::CUDA);
    EXPECT_EQ(ctx->localDevices()[2].type, DeviceType::CUDA);
}

TEST_F(Test__MockCollectiveContext, Builder_SetDevices) {
    std::vector<DeviceId> devices = {
        DeviceId::cpu(),
        DeviceId::cuda(0),
        DeviceId::cuda(1)
    };
    
    auto ctx = MockCollectiveContext::Builder()
        .withRank(0)
        .withWorldSize(1)
        .withDevices(devices)
        .build();
    
    EXPECT_EQ(ctx->localDevices().size(), 3);
}

TEST_F(Test__MockCollectiveContext, Builder_CallTracking) {
    auto ctx = MockCollectiveContext::Builder()
        .withRank(0)
        .withWorldSize(1)
        .withCallTracking(false)
        .build();
    
    EXPECT_FALSE(ctx->config().track_calls);
}

TEST_F(Test__MockCollectiveContext, Builder_FailureInjection) {
    auto ctx = MockCollectiveContext::Builder()
        .withRank(0)
        .withWorldSize(2)
        .withAllreduceFails(true)
        .withAllgatherFails(true)
        .withBroadcastFails(true)
        .build();
    
    EXPECT_TRUE(ctx->config().allreduce_should_fail);
    EXPECT_TRUE(ctx->config().allgather_should_fail);
    EXPECT_TRUE(ctx->config().broadcast_should_fail);
}

TEST_F(Test__MockCollectiveContext, Builder_BuildShared) {
    std::shared_ptr<MockCollectiveContext> ctx = MockCollectiveContext::Builder()
        .withRank(1)
        .withWorldSize(4)
        .buildShared();
    
    EXPECT_NE(ctx, nullptr);
    EXPECT_EQ(ctx->rank(), 1);
}

// =============================================================================
// Preset Tests
// =============================================================================

TEST_F(Test__MockCollectiveContext, Preset_SingleRankCPU) {
    auto ctx = MockCollectiveContext::singleRankCPU();
    
    EXPECT_EQ(ctx->rank(), 0);
    EXPECT_EQ(ctx->worldSize(), 1);
    EXPECT_FALSE(ctx->requiresCollectives());
    EXPECT_EQ(ctx->localDevices().size(), 1);
    EXPECT_EQ(ctx->localDevices()[0].type, DeviceType::CPU);
}

TEST_F(Test__MockCollectiveContext, Preset_MultiRankCPU) {
    auto ctx = MockCollectiveContext::multiRankCPU(2, 8);
    
    EXPECT_EQ(ctx->rank(), 2);
    EXPECT_EQ(ctx->worldSize(), 8);
    EXPECT_TRUE(ctx->requiresCollectives());
}

TEST_F(Test__MockCollectiveContext, Preset_FailingContext) {
    auto ctx = MockCollectiveContext::failingContext();
    
    EXPECT_TRUE(ctx->config().allreduce_should_fail);
    EXPECT_TRUE(ctx->config().allgather_should_fail);
    EXPECT_TRUE(ctx->config().broadcast_should_fail);
    
    // Verify operations fail
    EXPECT_FALSE(ctx->executeAllreduce(nullptr, 100, DeviceId::cpu()));
    EXPECT_FALSE(ctx->executeAllgather(nullptr, nullptr, 10, DeviceId::cpu()));
    EXPECT_FALSE(ctx->executeBroadcast(nullptr, 100, 0, DeviceId::cpu()));
}

TEST_F(Test__MockCollectiveContext, Preset_NoBackendsAvailable) {
    auto ctx = MockCollectiveContext::noBackendsAvailable();
    
    EXPECT_FALSE(ctx->isBackendAvailable(CollectiveBackendType::MPI));
    EXPECT_FALSE(ctx->isBackendAvailable(CollectiveBackendType::NCCL));
    EXPECT_FALSE(ctx->isBackendAvailable(CollectiveBackendType::RCCL));
    EXPECT_FALSE(ctx->isBackendAvailable(CollectiveBackendType::HOST));
    EXPECT_FALSE(ctx->isBackendAvailable(CollectiveBackendType::AUTO));
}

// =============================================================================
// Backend Availability Tests
// =============================================================================

TEST_F(Test__MockCollectiveContext, BackendAvailability_Default) {
    // Default: MPI and HOST available, NCCL and RCCL not
    EXPECT_TRUE(ctx_single_->isBackendAvailable(CollectiveBackendType::MPI));
    EXPECT_TRUE(ctx_single_->isBackendAvailable(CollectiveBackendType::HOST));
    EXPECT_FALSE(ctx_single_->isBackendAvailable(CollectiveBackendType::NCCL));
    EXPECT_FALSE(ctx_single_->isBackendAvailable(CollectiveBackendType::RCCL));
    EXPECT_TRUE(ctx_single_->isBackendAvailable(CollectiveBackendType::AUTO));
}

TEST_F(Test__MockCollectiveContext, BackendAvailability_ConfigureNCCL) {
    auto ctx = MockCollectiveContext::Builder()
        .withRank(0)
        .withWorldSize(1)
        .withNCCLAvailable(true)
        .build();
    
    EXPECT_TRUE(ctx->isBackendAvailable(CollectiveBackendType::NCCL));
}

TEST_F(Test__MockCollectiveContext, BackendAvailability_ConfigureRCCL) {
    auto ctx = MockCollectiveContext::Builder()
        .withRank(0)
        .withWorldSize(1)
        .withRCCLAvailable(true)
        .build();
    
    EXPECT_TRUE(ctx->isBackendAvailable(CollectiveBackendType::RCCL));
}

TEST_F(Test__MockCollectiveContext, BackendAvailability_RuntimeModification) {
    ctx_single_->set_mpi_available(false);
    EXPECT_FALSE(ctx_single_->isBackendAvailable(CollectiveBackendType::MPI));
    
    ctx_single_->set_nccl_available(true);
    EXPECT_TRUE(ctx_single_->isBackendAvailable(CollectiveBackendType::NCCL));
}

// =============================================================================
// Call Tracking Tests
// =============================================================================

TEST_F(Test__MockCollectiveContext, CallTracking_InitiallyZero) {
    EXPECT_EQ(ctx_single_->allreduce_call_count(), 0);
    EXPECT_EQ(ctx_single_->allgather_call_count(), 0);
    EXPECT_EQ(ctx_single_->broadcast_call_count(), 0);
    EXPECT_EQ(ctx_single_->total_collective_calls(), 0);
}

TEST_F(Test__MockCollectiveContext, CallTracking_AllReduce) {
    ctx_single_->executeAllreduce(nullptr, 100, DeviceId::cpu());
    EXPECT_EQ(ctx_single_->allreduce_call_count(), 1);
    
    ctx_single_->executeAllreduce(nullptr, 200, DeviceId::cpu());
    ctx_single_->executeAllreduce(nullptr, 300, DeviceId::cpu());
    EXPECT_EQ(ctx_single_->allreduce_call_count(), 3);
}

TEST_F(Test__MockCollectiveContext, CallTracking_AllGather) {
    ctx_single_->executeAllgather(nullptr, nullptr, 10, DeviceId::cpu());
    EXPECT_EQ(ctx_single_->allgather_call_count(), 1);
    
    ctx_single_->executeAllgather(nullptr, nullptr, 20, DeviceId::cpu());
    EXPECT_EQ(ctx_single_->allgather_call_count(), 2);
}

TEST_F(Test__MockCollectiveContext, CallTracking_Broadcast) {
    ctx_single_->executeBroadcast(nullptr, 50, 0, DeviceId::cpu());
    ctx_single_->executeBroadcast(nullptr, 50, 0, DeviceId::cpu());
    
    EXPECT_EQ(ctx_single_->broadcast_call_count(), 2);
}

TEST_F(Test__MockCollectiveContext, CallTracking_TotalCalls) {
    ctx_single_->executeAllreduce(nullptr, 100, DeviceId::cpu());
    ctx_single_->executeAllgather(nullptr, nullptr, 10, DeviceId::cpu());
    ctx_single_->executeBroadcast(nullptr, 50, 0, DeviceId::cpu());
    
    EXPECT_EQ(ctx_single_->total_collective_calls(), 3);
}

TEST_F(Test__MockCollectiveContext, CallTracking_Reset) {
    ctx_single_->executeAllreduce(nullptr, 100, DeviceId::cpu());
    ctx_single_->executeAllgather(nullptr, nullptr, 10, DeviceId::cpu());
    
    EXPECT_GT(ctx_single_->total_collective_calls(), 0);
    
    ctx_single_->reset_call_counts();
    
    EXPECT_EQ(ctx_single_->allreduce_call_count(), 0);
    EXPECT_EQ(ctx_single_->allgather_call_count(), 0);
    EXPECT_EQ(ctx_single_->broadcast_call_count(), 0);
    EXPECT_EQ(ctx_single_->total_collective_calls(), 0);
}

// =============================================================================
// Last Call Information Tests
// =============================================================================

TEST_F(Test__MockCollectiveContext, LastCall_AllReduce) {
    DeviceId device = DeviceId::cuda(1);
    ctx_single_->executeAllreduce(nullptr, 512, device, CollectiveOp::ALLREDUCE_MAX);
    
    EXPECT_EQ(ctx_single_->last_allreduce_op(), CollectiveOp::ALLREDUCE_MAX);
    EXPECT_EQ(ctx_single_->last_allreduce_count(), 512);
    EXPECT_EQ(ctx_single_->last_allreduce_device().type, DeviceType::CUDA);
    EXPECT_EQ(ctx_single_->last_allreduce_device().ordinal, 1);
}

TEST_F(Test__MockCollectiveContext, LastCall_AllGather) {
    DeviceId device = DeviceId::rocm(0);
    ctx_single_->executeAllgather(nullptr, nullptr, 128, device);
    
    EXPECT_EQ(ctx_single_->last_allgather_seq_len(), 128);
    EXPECT_EQ(ctx_single_->last_allgather_device().type, DeviceType::ROCm);
}

TEST_F(Test__MockCollectiveContext, LastCall_Broadcast) {
    DeviceId device = DeviceId::cpu();
    ctx_single_->executeBroadcast(nullptr, 1024, 2, device);
    
    EXPECT_EQ(ctx_single_->last_broadcast_count(), 1024);
    EXPECT_EQ(ctx_single_->last_broadcast_root(), 2);
    EXPECT_EQ(ctx_single_->last_broadcast_device().type, DeviceType::CPU);
}

// =============================================================================
// Failure Injection Tests
// =============================================================================

TEST_F(Test__MockCollectiveContext, FailureInjection_AllReduce) {
    ctx_single_->set_allreduce_fails(true);
    
    bool result = ctx_single_->executeAllreduce(nullptr, 100, DeviceId::cpu());
    EXPECT_FALSE(result);
    
    // Calls should still be tracked
    EXPECT_EQ(ctx_single_->allreduce_call_count(), 1);
}

TEST_F(Test__MockCollectiveContext, FailureInjection_AllGather) {
    ctx_single_->set_allgather_fails(true);
    
    bool result = ctx_single_->executeAllgather(nullptr, nullptr, 10, DeviceId::cpu());
    EXPECT_FALSE(result);
    
    EXPECT_EQ(ctx_single_->allgather_call_count(), 1);
}

TEST_F(Test__MockCollectiveContext, FailureInjection_Broadcast) {
    ctx_single_->set_broadcast_fails(true);
    
    bool result = ctx_single_->executeBroadcast(nullptr, 50, 0, DeviceId::cpu());
    EXPECT_FALSE(result);
    
    EXPECT_EQ(ctx_single_->broadcast_call_count(), 1);
}

TEST_F(Test__MockCollectiveContext, FailureInjection_RuntimeToggle) {
    // Start without failure
    EXPECT_TRUE(ctx_single_->executeAllreduce(nullptr, 100, DeviceId::cpu()));
    
    // Enable failure
    ctx_single_->set_allreduce_fails(true);
    EXPECT_FALSE(ctx_single_->executeAllreduce(nullptr, 100, DeviceId::cpu()));
    
    // Disable failure
    ctx_single_->set_allreduce_fails(false);
    EXPECT_TRUE(ctx_single_->executeAllreduce(nullptr, 100, DeviceId::cpu()));
}

// =============================================================================
// Custom Hook Tests
// =============================================================================

TEST_F(Test__MockCollectiveContext, CustomHook_AllReduce) {
    bool hook_called = false;
    size_t hook_count = 0;
    
    auto ctx = MockCollectiveContext::Builder()
        .withRank(0)
        .withWorldSize(2)
        .withAllreduceHook([&](ITensor*, size_t count, DeviceId, CollectiveOp) {
            hook_called = true;
            hook_count = count;
            return true;
        })
        .build();
    
    ctx->executeAllreduce(nullptr, 256, DeviceId::cpu());
    
    EXPECT_TRUE(hook_called);
    EXPECT_EQ(hook_count, 256);
}

TEST_F(Test__MockCollectiveContext, CustomHook_AllGather) {
    bool hook_called = false;
    
    auto ctx = MockCollectiveContext::Builder()
        .withRank(0)
        .withWorldSize(2)
        .withAllgatherHook([&](ITensor*, ITensor*, size_t, DeviceId) {
            hook_called = true;
            return true;
        })
        .build();
    
    ctx->executeAllgather(nullptr, nullptr, 10, DeviceId::cpu());
    
    EXPECT_TRUE(hook_called);
}

TEST_F(Test__MockCollectiveContext, CustomHook_Broadcast) {
    int hook_root = -1;
    
    auto ctx = MockCollectiveContext::Builder()
        .withRank(0)
        .withWorldSize(4)
        .withBroadcastHook([&](ITensor*, size_t, int root, DeviceId) {
            hook_root = root;
            return true;
        })
        .build();
    
    ctx->executeBroadcast(nullptr, 100, 3, DeviceId::cpu());
    
    EXPECT_EQ(hook_root, 3);
}

TEST_F(Test__MockCollectiveContext, CustomHook_ReturnsFalse) {
    auto ctx = MockCollectiveContext::Builder()
        .withRank(0)
        .withWorldSize(2)
        .withAllreduceHook([](ITensor*, size_t, DeviceId, CollectiveOp) {
            return false;  // Simulate failure
        })
        .build();
    
    EXPECT_FALSE(ctx->executeAllreduce(nullptr, 100, DeviceId::cpu()));
}

// =============================================================================
// Interface Compliance Tests
// =============================================================================

TEST_F(Test__MockCollectiveContext, InterfaceCompliance_CanUseAsICollectiveContext) {
    // Verify MockCollectiveContext can be used through ICollectiveContext pointer
    std::unique_ptr<ICollectiveContext> interface_ptr = MockCollectiveContext::singleRankCPU();
    
    EXPECT_EQ(interface_ptr->rank(), 0);
    EXPECT_EQ(interface_ptr->worldSize(), 1);
    EXPECT_FALSE(interface_ptr->requiresCollectives());
    
    // Operations should work through interface
    EXPECT_TRUE(interface_ptr->executeAllreduce(nullptr, 100, DeviceId::cpu()));
}

TEST_F(Test__MockCollectiveContext, InterfaceCompliance_NullRouterAndMPI) {
    // Mock doesn't have real router or MPI context
    EXPECT_EQ(ctx_single_->router(), nullptr);
    EXPECT_EQ(ctx_single_->mpiContext(), nullptr);
}

// =============================================================================
// Thread Safety Tests
// =============================================================================

TEST_F(Test__MockCollectiveContext, ThreadSafety_ConcurrentCallTracking) {
    const int num_threads = 4;
    const int calls_per_thread = 100;
    
    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([this, calls_per_thread]() {
            for (int i = 0; i < calls_per_thread; ++i) {
                ctx_single_->executeAllreduce(nullptr, 100, DeviceId::cpu());
            }
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    // All calls should be tracked
    EXPECT_EQ(ctx_single_->allreduce_call_count(), 
              static_cast<size_t>(num_threads * calls_per_thread));
}

// =============================================================================
// Description Tests
// =============================================================================

TEST_F(Test__MockCollectiveContext, Description_SingleRank) {
    std::string desc = ctx_single_->description();
    
    EXPECT_TRUE(desc.find("rank=0") != std::string::npos);
    EXPECT_TRUE(desc.find("world_size=1") != std::string::npos);
}

TEST_F(Test__MockCollectiveContext, Description_MultiRank) {
    std::string desc = ctx_multi_->description();
    
    EXPECT_TRUE(desc.find("rank=1") != std::string::npos);
    EXPECT_TRUE(desc.find("world_size=4") != std::string::npos);
}

TEST_F(Test__MockCollectiveContext, Description_IncludesCallCount) {
    ctx_single_->executeAllreduce(nullptr, 100, DeviceId::cpu());
    ctx_single_->executeAllreduce(nullptr, 100, DeviceId::cpu());
    
    std::string desc = ctx_single_->description();
    EXPECT_TRUE(desc.find("calls=2") != std::string::npos);
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    llaminar2::GPUDeviceProcessLifetime gpu_process_lifetime;

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
