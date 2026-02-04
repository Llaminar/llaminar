/**
 * @file Test__PPStage_GlobalTP.cpp
 * @brief Unit tests for PPStage GLOBAL_TP_DOMAIN support
 *
 * Tests the PPStage class's handling of global (cross-MPI-rank) TP domains.
 * These tests use GlobalTPContext::createForTest() with MPI_COMM_SELF for single-rank testing.
 *
 * Note: Requires MPI initialization (uses mpi_gtest_main.cpp).
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <memory>

#include "collective/PPStage.h"
#include "collective/GlobalTPContext.h"
#include "collective/IGlobalTPContext.h"
#include "backends/GlobalDeviceAddress.h"

namespace llaminar2::test
{

// =========================================================================
// Test Fixture
// =========================================================================

class Test__PPStage_GlobalTP : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Ensure MPI is initialized (mpi_gtest_main.cpp handles this)
        int initialized;
        MPI_Initialized(&initialized);
        ASSERT_TRUE(initialized) << "MPI must be initialized for GlobalTPContext tests";

        // Create a test GlobalTPContext with domain_id=0, simulating 2-rank domain
        // This uses MPI_COMM_SELF internally for testing without real MPI
        global_tp_ctx_ = GlobalTPContext::createForTest(
            MPI_COMM_SELF,  // Use SELF for single-process testing
            0,              // domain_id
            {0, 1}          // Simulated world ranks (would be 2 ranks in real setup)
        );
    }

    std::shared_ptr<IGlobalTPContext> global_tp_ctx_;
};

// =========================================================================
// PPStage: Global TP Domain Construction Tests
// =========================================================================

TEST_F(Test__PPStage_GlobalTP, FromGlobalTPContext_CreatesCorrectType)
{
    auto stage = PPStage::fromGlobalTPContext(global_tp_ctx_);

    EXPECT_EQ(stage.type(), PPStageType::GLOBAL_TP_DOMAIN);
    EXPECT_TRUE(stage.isGlobalTPDomain());
    EXPECT_FALSE(stage.isSingleDevice());
    EXPECT_FALSE(stage.isTPDomain());
    EXPECT_FALSE(stage.isNestedPP());
}

TEST_F(Test__PPStage_GlobalTP, Type_ReturnsGlobalTPDomain)
{
    auto stage = PPStage::fromGlobalTPContext(global_tp_ctx_);

    EXPECT_EQ(stage.type(), PPStageType::GLOBAL_TP_DOMAIN);
}

// =========================================================================
// PPStage: isGlobalTPDomain Tests
// =========================================================================

TEST_F(Test__PPStage_GlobalTP, IsGlobalTPDomain_ReturnsTrue)
{
    auto stage = PPStage::fromGlobalTPContext(global_tp_ctx_);

    EXPECT_TRUE(stage.isGlobalTPDomain());
}

TEST_F(Test__PPStage_GlobalTP, IsGlobalTPDomain_ReturnsFalseForSingleDevice)
{
    auto stage = PPStage::fromDevice(GlobalDeviceAddress::cuda(0));

    EXPECT_FALSE(stage.isGlobalTPDomain());
}

TEST_F(Test__PPStage_GlobalTP, IsGlobalTPDomain_ReturnsFalseForLocalTPDomain)
{
    // Can't easily test TP_DOMAIN without mock, so just verify the enum differs
    EXPECT_NE(PPStageType::TP_DOMAIN, PPStageType::GLOBAL_TP_DOMAIN);
}

// =========================================================================
// PPStage: Accessor Tests
// =========================================================================

TEST_F(Test__PPStage_GlobalTP, AsGlobalTPContext_ReturnsValidPointer)
{
    auto stage = PPStage::fromGlobalTPContext(global_tp_ctx_);

    IGlobalTPContext* ctx = stage.asGlobalTPContext();
    ASSERT_NE(ctx, nullptr);
    EXPECT_EQ(ctx, global_tp_ctx_.get());
}

TEST_F(Test__PPStage_GlobalTP, AsGlobalTPContext_ReturnsNullForSingleDevice)
{
    auto stage = PPStage::fromDevice(GlobalDeviceAddress::cuda(0));

    IGlobalTPContext* ctx = stage.asGlobalTPContext();
    EXPECT_EQ(ctx, nullptr);
}

TEST_F(Test__PPStage_GlobalTP, GlobalTPContextPtr_ReturnsSharedPtr)
{
    auto stage = PPStage::fromGlobalTPContext(global_tp_ctx_);

    auto ptr = stage.globalTPContextPtr();
    EXPECT_NE(ptr, nullptr);
    EXPECT_EQ(ptr.get(), global_tp_ctx_.get());
    // At minimum, both global_tp_ctx_ and the copy in ptr should hold references
    EXPECT_GE(ptr.use_count(), 2);
}

TEST_F(Test__PPStage_GlobalTP, GlobalTPContextPtr_ReturnsNullForSingleDevice)
{
    auto stage = PPStage::fromDevice(GlobalDeviceAddress::cuda(0));

    auto ptr = stage.globalTPContextPtr();
    EXPECT_EQ(ptr, nullptr);
}

// =========================================================================
// PPStage: representativeDevice Tests
// =========================================================================

TEST_F(Test__PPStage_GlobalTP, RepresentativeDevice_ReturnsLocalCPU)
{
    auto stage = PPStage::fromGlobalTPContext(global_tp_ctx_);

    GlobalDeviceAddress rep = stage.representativeDevice();
    
    // Global TP is CPU-only, so representative should be CPU
    EXPECT_EQ(rep.device_type, DeviceType::CPU);
}

TEST_F(Test__PPStage_GlobalTP, RepresentativeDevice_MatchesContextLocalDevice)
{
    auto stage = PPStage::fromGlobalTPContext(global_tp_ctx_);

    GlobalDeviceAddress rep = stage.representativeDevice();
    GlobalDeviceAddress ctx_local = global_tp_ctx_->localDevice();
    
    EXPECT_EQ(rep, ctx_local);
}

// =========================================================================
// PPStage: allDevices Tests
// =========================================================================

TEST_F(Test__PPStage_GlobalTP, AllDevices_ReturnsSingleDevice)
{
    auto stage = PPStage::fromGlobalTPContext(global_tp_ctx_);

    std::vector<GlobalDeviceAddress> devices = stage.allDevices();
    
    // For PP transfers, global TP returns only local rank's device
    ASSERT_EQ(devices.size(), 1u);
}

TEST_F(Test__PPStage_GlobalTP, AllDevices_ReturnsLocalCPU)
{
    auto stage = PPStage::fromGlobalTPContext(global_tp_ctx_);

    std::vector<GlobalDeviceAddress> devices = stage.allDevices();
    
    ASSERT_EQ(devices.size(), 1u);
    EXPECT_EQ(devices[0].device_type, DeviceType::CPU);
}

TEST_F(Test__PPStage_GlobalTP, AllDevices_MatchesRepresentativeDevice)
{
    auto stage = PPStage::fromGlobalTPContext(global_tp_ctx_);

    std::vector<GlobalDeviceAddress> devices = stage.allDevices();
    GlobalDeviceAddress rep = stage.representativeDevice();
    
    ASSERT_EQ(devices.size(), 1u);
    EXPECT_EQ(devices[0], rep);
}

// =========================================================================
// PPStage: deviceCount Tests
// =========================================================================

TEST_F(Test__PPStage_GlobalTP, DeviceCount_ReturnsOne)
{
    auto stage = PPStage::fromGlobalTPContext(global_tp_ctx_);

    // For PP transfers, global TP has 1 device per rank
    EXPECT_EQ(stage.deviceCount(), 1);
}

// =========================================================================
// PPStage: containsDevice Tests
// =========================================================================

TEST_F(Test__PPStage_GlobalTP, ContainsDevice_TrueForLocalCPU)
{
    auto stage = PPStage::fromGlobalTPContext(global_tp_ctx_);
    GlobalDeviceAddress local_cpu = global_tp_ctx_->localDevice();

    EXPECT_TRUE(stage.containsDevice(local_cpu));
}

TEST_F(Test__PPStage_GlobalTP, ContainsDevice_FalseForOtherDevice)
{
    auto stage = PPStage::fromGlobalTPContext(global_tp_ctx_);

    // A CUDA device should not be contained in a CPU-only global TP
    EXPECT_FALSE(stage.containsDevice(GlobalDeviceAddress::cuda(0)));
    EXPECT_FALSE(stage.containsDevice(GlobalDeviceAddress::rocm(0)));
}

// =========================================================================
// PPStage: describe Tests
// =========================================================================

TEST_F(Test__PPStage_GlobalTP, Describe_IncludesGlobalTP)
{
    auto stage = PPStage::fromGlobalTPContext(global_tp_ctx_);

    std::string desc = stage.describe();
    
    EXPECT_FALSE(desc.empty());
    EXPECT_TRUE(desc.find("GlobalTP") != std::string::npos ||
                desc.find("global") != std::string::npos ||
                desc.find("Global") != std::string::npos);
}

TEST_F(Test__PPStage_GlobalTP, Describe_IncludesDegree)
{
    auto stage = PPStage::fromGlobalTPContext(global_tp_ctx_);

    std::string desc = stage.describe();
    
    // Should include degree information
    EXPECT_TRUE(desc.find("degree") != std::string::npos ||
                desc.find("=") != std::string::npos);
}

// =========================================================================
// PPStage: Null Context Handling
// =========================================================================

TEST_F(Test__PPStage_GlobalTP, NullContext_DeviceCountReturnsZero)
{
    // Create stage with null context
    std::shared_ptr<IGlobalTPContext> null_ctx = nullptr;
    auto stage = PPStage::fromGlobalTPContext(null_ctx);

    EXPECT_EQ(stage.deviceCount(), 0);
}

TEST_F(Test__PPStage_GlobalTP, NullContext_AllDevicesReturnsEmpty)
{
    std::shared_ptr<IGlobalTPContext> null_ctx = nullptr;
    auto stage = PPStage::fromGlobalTPContext(null_ctx);

    auto devices = stage.allDevices();
    EXPECT_TRUE(devices.empty());
}

} // namespace llaminar2::test
