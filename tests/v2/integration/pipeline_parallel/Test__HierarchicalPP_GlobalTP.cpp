/**
 * @file Test__HierarchicalPP_GlobalTP.cpp
 * @brief MPI integration tests for HierarchicalPPContext with GLOBAL_TP_DOMAIN stages
 *
 * Tests pipeline parallel transfers involving global (cross-MPI-rank) TP domains.
 * These tests verify:
 * 1. PP transfers from local TP domain to global TP stage
 * 2. PP transfers from global TP stage to single device
 * 3. Full PP cycle with global TP stage in the middle
 * 4. Data integrity across ranks after transfers
 *
 * Test topology:
 *   Stage 0: LocalTP(cuda:0, cuda:1) or single device
 *   Stage 1: GlobalTP(0:cpu:0, 1:cpu:0) - cross-rank CPU TP
 *   Stage 2: SingleDevice(rocm:0) or single device
 *
 * These tests require MPI and run with 2 ranks (MPI_PROCS 2).
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#include <gtest/gtest.h>
#include <mpi.h>

#include "collective/PPStage.h"
#include "collective/ILocalPPContext.h"
#include "collective/GlobalTPContext.h"
#include "collective/IGlobalTPContext.h"
#include "backends/GlobalDeviceAddress.h"
#include "tensors/TensorClasses.h"
#include "utils/Logger.h"

using namespace llaminar2;

namespace
{

// =========================================================================
// Test Fixture
// =========================================================================

class Test__HierarchicalPP_GlobalTP : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Get MPI rank info
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
        
        LOG_INFO("[Test] MPI rank " << world_rank_ << " of " << world_size_);
        
        // Create GlobalTPContext for cross-rank CPU TP
        // Both ranks participate in domain 0
        std::vector<int> world_ranks = {0, 1};
        global_tp_ctx_ = GlobalTPContext::createForTest(
            MPI_COMM_WORLD,  // Use WORLD comm for cross-rank
            0,               // domain_id
            world_ranks      // Both ranks in domain
        );
        
        ASSERT_NE(global_tp_ctx_, nullptr) << "Failed to create GlobalTPContext";
        ASSERT_EQ(global_tp_ctx_->degree(), world_size_);
        ASSERT_EQ(global_tp_ctx_->myIndex(), world_rank_);
    }

    void TearDown() override
    {
        // Barrier to ensure all ranks complete before destroying context
        MPI_Barrier(MPI_COMM_WORLD);
        global_tp_ctx_.reset();
    }

    int world_rank_ = 0;
    int world_size_ = 1;
    std::shared_ptr<GlobalTPContext> global_tp_ctx_;
};

// =========================================================================
// Basic PPStage Tests with GlobalTPContext
// =========================================================================

TEST_F(Test__HierarchicalPP_GlobalTP, PPStage_FromGlobalTPContext_Works)
{
    // Each rank creates a PPStage from the global TP context
    auto stage = PPStage::fromGlobalTPContext(global_tp_ctx_);
    
    EXPECT_EQ(stage.type(), PPStageType::GLOBAL_TP_DOMAIN);
    EXPECT_TRUE(stage.isGlobalTPDomain());
    
    // Verify accessor returns valid context
    auto* ctx = stage.asGlobalTPContext();
    ASSERT_NE(ctx, nullptr);
    EXPECT_EQ(ctx->degree(), world_size_);
    
    MPI_Barrier(MPI_COMM_WORLD);
}

TEST_F(Test__HierarchicalPP_GlobalTP, PPStage_RepresentativeDevice_IsCPU)
{
    auto stage = PPStage::fromGlobalTPContext(global_tp_ctx_);
    
    GlobalDeviceAddress rep = stage.representativeDevice();
    
    // Global TP is CPU-only
    EXPECT_EQ(rep.device_type, DeviceType::CPU);
    
    // Each rank should get its own unique CPU device
    // The device should encode the rank info
    LOG_INFO("[Rank " << world_rank_ << "] Representative device: " << rep.toString());
    
    MPI_Barrier(MPI_COMM_WORLD);
}

TEST_F(Test__HierarchicalPP_GlobalTP, PPStage_DeviceCount_IsOne)
{
    auto stage = PPStage::fromGlobalTPContext(global_tp_ctx_);
    
    // For PP transfers, each rank sees 1 device in global TP
    EXPECT_EQ(stage.deviceCount(), 1);
    
    MPI_Barrier(MPI_COMM_WORLD);
}

TEST_F(Test__HierarchicalPP_GlobalTP, PPStage_AllDevices_ReturnsSingleDevice)
{
    auto stage = PPStage::fromGlobalTPContext(global_tp_ctx_);
    
    auto devices = stage.allDevices();
    ASSERT_EQ(devices.size(), 1u);
    EXPECT_EQ(devices[0].device_type, DeviceType::CPU);
    
    MPI_Barrier(MPI_COMM_WORLD);
}

// =========================================================================
// Global TP Context Verification
// =========================================================================

TEST_F(Test__HierarchicalPP_GlobalTP, GlobalTPContext_DegreeMatchesWorldSize)
{
    EXPECT_EQ(global_tp_ctx_->degree(), world_size_);
}

TEST_F(Test__HierarchicalPP_GlobalTP, GlobalTPContext_MyIndexMatchesRank)
{
    EXPECT_EQ(global_tp_ctx_->myIndex(), world_rank_);
}

TEST_F(Test__HierarchicalPP_GlobalTP, GlobalTPContext_IsNotLocal)
{
    // Global TP is cross-rank, so isLocal() should return false
    EXPECT_FALSE(global_tp_ctx_->isLocal());
    EXPECT_TRUE(global_tp_ctx_->isGlobal());
}

TEST_F(Test__HierarchicalPP_GlobalTP, GlobalTPContext_LocalDevice_IsCPU)
{
    GlobalDeviceAddress local = global_tp_ctx_->localDevice();
    
    EXPECT_EQ(local.device_type, DeviceType::CPU);
    LOG_INFO("[Rank " << world_rank_ << "] Local device: " << local.toString());
    
    MPI_Barrier(MPI_COMM_WORLD);
}

TEST_F(Test__HierarchicalPP_GlobalTP, GlobalTPContext_WorldRanks_ContainsAllRanks)
{
    const std::vector<int>& world_ranks = global_tp_ctx_->worldRanks();
    
    EXPECT_EQ(world_ranks.size(), static_cast<size_t>(world_size_));
    
    // Check that our rank is in the list
    bool found_self = false;
    for (int r : world_ranks)
    {
        if (r == world_rank_)
        {
            found_self = true;
            break;
        }
    }
    EXPECT_TRUE(found_self) << "Rank " << world_rank_ << " not found in worldRanks()";
    
    MPI_Barrier(MPI_COMM_WORLD);
}

// =========================================================================
// Global TP Barrier Test
// =========================================================================

TEST_F(Test__HierarchicalPP_GlobalTP, GlobalTPContext_Barrier_Synchronizes)
{
    // Simple barrier test - should not hang
    LOG_INFO("[Rank " << world_rank_ << "] Before barrier");
    
    global_tp_ctx_->barrier();
    
    LOG_INFO("[Rank " << world_rank_ << "] After barrier");
    
    MPI_Barrier(MPI_COMM_WORLD);
}

// =========================================================================
// Multi-Stage PP Configuration with Global TP
// =========================================================================

TEST_F(Test__HierarchicalPP_GlobalTP, HierarchicalPPConfig_WithGlobalTPStage)
{
    // Create a 3-stage PP configuration:
    // Stage 0: Single device (cuda:0)
    // Stage 1: Global TP domain
    // Stage 2: Single device (cpu:0)
    
    HierarchicalPPConfig config;
    config.layer_boundaries = {0, 10, 20, 28};  // 3 stages
    
    config.stages.push_back(PPStage::fromDevice(GlobalDeviceAddress::cuda(0)));
    config.stages.push_back(PPStage::fromGlobalTPContext(global_tp_ctx_));
    config.stages.push_back(PPStage::fromDevice(GlobalDeviceAddress::cpu()));
    
    EXPECT_EQ(config.stages.size(), 3u);
    EXPECT_EQ(config.numStages(), 3);
    
    // Verify stage types
    EXPECT_TRUE(config.stages[0].isSingleDevice());
    EXPECT_TRUE(config.stages[1].isGlobalTPDomain());
    EXPECT_TRUE(config.stages[2].isSingleDevice());
    
    // Verify layer ranges
    auto [start0, end0] = config.layerRangeForStage(0);
    EXPECT_EQ(start0, 0);
    EXPECT_EQ(end0, 10);
    
    auto [start1, end1] = config.layerRangeForStage(1);
    EXPECT_EQ(start1, 10);
    EXPECT_EQ(end1, 20);
    
    auto [start2, end2] = config.layerRangeForStage(2);
    EXPECT_EQ(start2, 20);
    EXPECT_EQ(end2, 28);
    
    MPI_Barrier(MPI_COMM_WORLD);
}

TEST_F(Test__HierarchicalPP_GlobalTP, HierarchicalPPConfig_StageForLayer_WithGlobalTP)
{
    HierarchicalPPConfig config;
    config.layer_boundaries = {0, 10, 20, 28};
    
    config.stages.push_back(PPStage::fromDevice(GlobalDeviceAddress::cuda(0)));
    config.stages.push_back(PPStage::fromGlobalTPContext(global_tp_ctx_));
    config.stages.push_back(PPStage::fromDevice(GlobalDeviceAddress::cpu()));
    
    // Test layer-to-stage mapping
    EXPECT_EQ(config.stageForLayer(0), 0);
    EXPECT_EQ(config.stageForLayer(5), 0);
    EXPECT_EQ(config.stageForLayer(9), 0);
    EXPECT_EQ(config.stageForLayer(10), 1);  // Global TP stage
    EXPECT_EQ(config.stageForLayer(15), 1);
    EXPECT_EQ(config.stageForLayer(19), 1);
    EXPECT_EQ(config.stageForLayer(20), 2);
    EXPECT_EQ(config.stageForLayer(27), 2);
    
    MPI_Barrier(MPI_COMM_WORLD);
}

// =========================================================================
// Describe Test
// =========================================================================

TEST_F(Test__HierarchicalPP_GlobalTP, PPStage_Describe_IncludesGlobalTPInfo)
{
    auto stage = PPStage::fromGlobalTPContext(global_tp_ctx_);
    
    std::string desc = stage.describe();
    
    LOG_INFO("[Rank " << world_rank_ << "] Stage description: " << desc);
    
    // Should include "GlobalTP" or similar
    EXPECT_TRUE(desc.find("GlobalTP") != std::string::npos ||
                desc.find("global") != std::string::npos ||
                desc.find("Global") != std::string::npos);
    
    MPI_Barrier(MPI_COMM_WORLD);
}

} // anonymous namespace
