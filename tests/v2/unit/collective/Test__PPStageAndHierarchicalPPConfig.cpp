/**
 * @file Test__PPStageAndHierarchicalPPConfig.cpp
 * @brief Unit tests for PPStage variant type and HierarchicalPPConfig
 *
 * Tests the hierarchical PP stage model including:
 * - PPStage construction for single devices, TP domains, and nested PP
 * - PPStage type queries and accessors
 * - PPStage::representativeDevice() and allDevices() methods
 * - HierarchicalPPConfig validation
 * - HierarchicalPPConfig layer boundary and stage queries
 *
 * These tests work WITHOUT real GPUs by using MockLocalTPContext.
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#include <gtest/gtest.h>
#include <memory>

#include "collective/PPStage.h"
#include "collective/ILocalPPContext.h"  // HierarchicalPPConfig
#include "backends/GlobalDeviceAddress.h"
#include "mocks/MockLocalTPContext.h"

namespace llaminar2::test
{

    // =========================================================================
    // Test Fixture
    // =========================================================================

    class Test__PPStageAndHierarchicalPPConfig : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // Create mock TP contexts for testing
            mock_tp_ctx_ = std::make_shared<MockLocalTPContext>();
            mock_tp_ctx_->setDevices({
                GlobalDeviceAddress::rocm(0),
                GlobalDeviceAddress::rocm(1)
            });
            mock_tp_ctx_->setBackend(CollectiveBackendType::RCCL);
        }

        std::shared_ptr<MockLocalTPContext> mock_tp_ctx_;
    };

    // =========================================================================
    // PPStage: Single Device Tests
    // =========================================================================

    TEST_F(Test__PPStageAndHierarchicalPPConfig, SingleDevice_Construction)
    {
        auto stage = PPStage::fromDevice(GlobalDeviceAddress::cuda(0));

        EXPECT_EQ(stage.type(), PPStageType::SINGLE_DEVICE);
        EXPECT_TRUE(stage.isSingleDevice());
        EXPECT_FALSE(stage.isTPDomain());
        EXPECT_FALSE(stage.isNestedPP());
    }

    TEST_F(Test__PPStageAndHierarchicalPPConfig, SingleDevice_DeviceAccess)
    {
        auto device = GlobalDeviceAddress::cuda(2);
        auto stage = PPStage::fromDevice(device);

        EXPECT_EQ(stage.device().device_type, DeviceType::CUDA);
        EXPECT_EQ(stage.device().device_ordinal, 2);
    }

    TEST_F(Test__PPStageAndHierarchicalPPConfig, SingleDevice_RepresentativeDevice)
    {
        auto device = GlobalDeviceAddress::rocm(1);
        auto stage = PPStage::fromDevice(device);

        auto representative = stage.representativeDevice();
        EXPECT_EQ(representative.device_type, DeviceType::ROCm);
        EXPECT_EQ(representative.device_ordinal, 1);
    }

    TEST_F(Test__PPStageAndHierarchicalPPConfig, SingleDevice_AllDevices)
    {
        auto device = GlobalDeviceAddress::cpu();
        auto stage = PPStage::fromDevice(device);

        auto all = stage.allDevices();
        ASSERT_EQ(all.size(), 1u);
        EXPECT_EQ(all[0].device_type, DeviceType::CPU);
    }

    TEST_F(Test__PPStageAndHierarchicalPPConfig, SingleDevice_Describe)
    {
        auto stage = PPStage::fromDevice(GlobalDeviceAddress::cuda(0));

        std::string desc = stage.describe();
        EXPECT_FALSE(desc.empty());
        // Should mention "single" or "device" or the device type
        EXPECT_TRUE(desc.find("cuda") != std::string::npos ||
                    desc.find("CUDA") != std::string::npos ||
                    desc.find("single") != std::string::npos ||
                    desc.find("Single") != std::string::npos);
    }

    // =========================================================================
    // PPStage: TP Domain Tests
    // =========================================================================

    TEST_F(Test__PPStageAndHierarchicalPPConfig, TPDomain_Construction)
    {
        auto stage = PPStage::fromTPContext(mock_tp_ctx_);

        EXPECT_EQ(stage.type(), PPStageType::TP_DOMAIN);
        EXPECT_FALSE(stage.isSingleDevice());
        EXPECT_TRUE(stage.isTPDomain());
        EXPECT_FALSE(stage.isNestedPP());
    }

    TEST_F(Test__PPStageAndHierarchicalPPConfig, TPDomain_TPContextAccess)
    {
        auto stage = PPStage::fromTPContext(mock_tp_ctx_);

        ILocalTPContext* ctx = stage.asTPContext();
        ASSERT_NE(ctx, nullptr);
        EXPECT_EQ(ctx->degree(), 2);
        EXPECT_EQ(ctx->backend(), CollectiveBackendType::RCCL);
    }

    TEST_F(Test__PPStageAndHierarchicalPPConfig, TPDomain_RepresentativeDevice)
    {
        // Representative device should be the first device in the TP domain
        auto stage = PPStage::fromTPContext(mock_tp_ctx_);

        auto representative = stage.representativeDevice();
        EXPECT_EQ(representative.device_type, DeviceType::ROCm);
        EXPECT_EQ(representative.device_ordinal, 0);  // First device
    }

    TEST_F(Test__PPStageAndHierarchicalPPConfig, TPDomain_AllDevices)
    {
        auto stage = PPStage::fromTPContext(mock_tp_ctx_);

        auto all = stage.allDevices();
        ASSERT_EQ(all.size(), 2u);
        EXPECT_EQ(all[0].device_type, DeviceType::ROCm);
        EXPECT_EQ(all[0].device_ordinal, 0);
        EXPECT_EQ(all[1].device_type, DeviceType::ROCm);
        EXPECT_EQ(all[1].device_ordinal, 1);
    }

    TEST_F(Test__PPStageAndHierarchicalPPConfig, TPDomain_Describe)
    {
        auto stage = PPStage::fromTPContext(mock_tp_ctx_);

        std::string desc = stage.describe();
        EXPECT_FALSE(desc.empty());
        // Should mention "TP" or "domain" or "parallel"
        EXPECT_TRUE(desc.find("TP") != std::string::npos ||
                    desc.find("domain") != std::string::npos ||
                    desc.find("tensor") != std::string::npos ||
                    desc.find("2") != std::string::npos);  // 2 devices
    }

    // =========================================================================
    // PPStage: Error Cases
    // =========================================================================

    TEST_F(Test__PPStageAndHierarchicalPPConfig, SingleDevice_AccessTPContextReturnsNull)
    {
        auto stage = PPStage::fromDevice(GlobalDeviceAddress::cuda(0));

        // asTPContext() returns nullptr for non-TP stages (doesn't throw)
        EXPECT_EQ(stage.asTPContext(), nullptr);
    }

    TEST_F(Test__PPStageAndHierarchicalPPConfig, TPDomain_AccessDeviceThrows)
    {
        auto stage = PPStage::fromTPContext(mock_tp_ctx_);

        EXPECT_THROW(stage.device(), std::logic_error);
    }

    TEST_F(Test__PPStageAndHierarchicalPPConfig, TPDomain_NullContextHandled)
    {
        // Creating with nullptr TP context should still work (edge case)
        std::shared_ptr<ILocalTPContext> null_ctx;
        auto stage = PPStage::fromTPContext(null_ctx);

        EXPECT_TRUE(stage.isTPDomain());
        EXPECT_EQ(stage.asTPContext(), nullptr);
    }

    // =========================================================================
    // HierarchicalPPConfig: Validation Tests
    // =========================================================================

    TEST_F(Test__PPStageAndHierarchicalPPConfig, Config_EmptyStagesInvalid)
    {
        HierarchicalPPConfig config;
        config.stages = {};
        config.layer_boundaries = {0, 24};

        EXPECT_FALSE(config.isValid());
    }

    TEST_F(Test__PPStageAndHierarchicalPPConfig, Config_MismatchedBoundariesInvalid)
    {
        HierarchicalPPConfig config;
        config.stages = {
            PPStage::fromDevice(GlobalDeviceAddress::cuda(0)),
            PPStage::fromDevice(GlobalDeviceAddress::cuda(1))
        };
        // 2 stages need 3 boundaries, but we only provide 2
        config.layer_boundaries = {0, 24};

        EXPECT_FALSE(config.isValid());
    }

    TEST_F(Test__PPStageAndHierarchicalPPConfig, Config_NonMonotonicBoundariesInvalid)
    {
        HierarchicalPPConfig config;
        config.stages = {
            PPStage::fromDevice(GlobalDeviceAddress::cuda(0)),
            PPStage::fromDevice(GlobalDeviceAddress::cuda(1))
        };
        // Boundaries not monotonically increasing
        config.layer_boundaries = {0, 20, 10};

        EXPECT_FALSE(config.isValid());
    }

    TEST_F(Test__PPStageAndHierarchicalPPConfig, Config_ValidSingleStage)
    {
        HierarchicalPPConfig config;
        config.stages = {PPStage::fromDevice(GlobalDeviceAddress::cuda(0))};
        config.layer_boundaries = {0, 24};

        EXPECT_TRUE(config.isValid());
        EXPECT_EQ(config.numStages(), 1);
    }

    TEST_F(Test__PPStageAndHierarchicalPPConfig, Config_ValidMultiStage)
    {
        HierarchicalPPConfig config;
        config.stages = {
            PPStage::fromTPContext(mock_tp_ctx_),           // Stage 0: TP domain
            PPStage::fromDevice(GlobalDeviceAddress::cuda(0)),  // Stage 1: single CUDA
            PPStage::fromDevice(GlobalDeviceAddress::cpu())     // Stage 2: CPU
        };
        config.layer_boundaries = {0, 14, 22, 24};

        EXPECT_TRUE(config.isValid());
        EXPECT_EQ(config.numStages(), 3);
    }

    TEST_F(Test__PPStageAndHierarchicalPPConfig, Config_EqualBoundariesValid)
    {
        // Equal adjacent boundaries (zero-width stage) should be valid
        HierarchicalPPConfig config;
        config.stages = {
            PPStage::fromDevice(GlobalDeviceAddress::cuda(0)),
            PPStage::fromDevice(GlobalDeviceAddress::cuda(1))
        };
        config.layer_boundaries = {0, 24, 24};  // Stage 1 has 0 layers

        EXPECT_TRUE(config.isValid());
    }

    // =========================================================================
    // HierarchicalPPConfig: Query Methods
    // =========================================================================

    TEST_F(Test__PPStageAndHierarchicalPPConfig, Config_LayerRangeForStage)
    {
        HierarchicalPPConfig config;
        config.stages = {
            PPStage::fromTPContext(mock_tp_ctx_),
            PPStage::fromDevice(GlobalDeviceAddress::cuda(0)),
            PPStage::fromDevice(GlobalDeviceAddress::cpu())
        };
        config.layer_boundaries = {0, 14, 22, 24};

        auto [start0, end0] = config.layerRangeForStage(0);
        EXPECT_EQ(start0, 0);
        EXPECT_EQ(end0, 14);

        auto [start1, end1] = config.layerRangeForStage(1);
        EXPECT_EQ(start1, 14);
        EXPECT_EQ(end1, 22);

        auto [start2, end2] = config.layerRangeForStage(2);
        EXPECT_EQ(start2, 22);
        EXPECT_EQ(end2, 24);
    }

    TEST_F(Test__PPStageAndHierarchicalPPConfig, Config_LayerRangeForInvalidStage)
    {
        HierarchicalPPConfig config;
        config.stages = {PPStage::fromDevice(GlobalDeviceAddress::cuda(0))};
        config.layer_boundaries = {0, 24};

        auto [start_neg, end_neg] = config.layerRangeForStage(-1);
        EXPECT_EQ(start_neg, -1);
        EXPECT_EQ(end_neg, -1);

        auto [start_oob, end_oob] = config.layerRangeForStage(5);
        EXPECT_EQ(start_oob, -1);
        EXPECT_EQ(end_oob, -1);
    }

    TEST_F(Test__PPStageAndHierarchicalPPConfig, Config_StageForLayer)
    {
        HierarchicalPPConfig config;
        config.stages = {
            PPStage::fromTPContext(mock_tp_ctx_),
            PPStage::fromDevice(GlobalDeviceAddress::cuda(0)),
            PPStage::fromDevice(GlobalDeviceAddress::cpu())
        };
        config.layer_boundaries = {0, 14, 22, 24};

        // Layers 0-13 → Stage 0
        EXPECT_EQ(config.stageForLayer(0), 0);
        EXPECT_EQ(config.stageForLayer(10), 0);
        EXPECT_EQ(config.stageForLayer(13), 0);

        // Layers 14-21 → Stage 1
        EXPECT_EQ(config.stageForLayer(14), 1);
        EXPECT_EQ(config.stageForLayer(18), 1);
        EXPECT_EQ(config.stageForLayer(21), 1);

        // Layers 22-23 → Stage 2
        EXPECT_EQ(config.stageForLayer(22), 2);
        EXPECT_EQ(config.stageForLayer(23), 2);
    }

    TEST_F(Test__PPStageAndHierarchicalPPConfig, Config_StageForOutOfBoundsLayer)
    {
        HierarchicalPPConfig config;
        config.stages = {PPStage::fromDevice(GlobalDeviceAddress::cuda(0))};
        config.layer_boundaries = {0, 24};

        EXPECT_EQ(config.stageForLayer(-1), -1);
        EXPECT_EQ(config.stageForLayer(24), -1);  // At boundary, not inside
        EXPECT_EQ(config.stageForLayer(100), -1);
    }

    // =========================================================================
    // HierarchicalPPConfig: representativeDeviceForStage
    // =========================================================================

    TEST_F(Test__PPStageAndHierarchicalPPConfig, Config_RepresentativeDeviceForStage)
    {
        HierarchicalPPConfig config;
        config.stages = {
            PPStage::fromTPContext(mock_tp_ctx_),                  // rocm:0, rocm:1
            PPStage::fromDevice(GlobalDeviceAddress::cuda(0)),
            PPStage::fromDevice(GlobalDeviceAddress::cpu())
        };
        config.layer_boundaries = {0, 14, 22, 24};

        // Stage 0: TP domain → representative is rocm:0 (first device)
        auto dev0 = config.stages[0].representativeDevice();
        EXPECT_EQ(dev0.device_type, DeviceType::ROCm);
        EXPECT_EQ(dev0.device_ordinal, 0);

        // Stage 1: Single device cuda:0
        auto dev1 = config.stages[1].representativeDevice();
        EXPECT_EQ(dev1.device_type, DeviceType::CUDA);
        EXPECT_EQ(dev1.device_ordinal, 0);

        // Stage 2: Single device cpu
        auto dev2 = config.stages[2].representativeDevice();
        EXPECT_EQ(dev2.device_type, DeviceType::CPU);
    }

    // =========================================================================
    // Mixed PPStage Types in Config
    // =========================================================================

    TEST_F(Test__PPStageAndHierarchicalPPConfig, Config_MixedStageTypes)
    {
        // Create a second mock TP context with different devices
        auto mock_tp_ctx_2 = std::make_shared<MockLocalTPContext>();
        mock_tp_ctx_2->setDevices({
            GlobalDeviceAddress::cuda(0),
            GlobalDeviceAddress::cuda(1)
        });
        mock_tp_ctx_2->setBackend(CollectiveBackendType::NCCL);

        HierarchicalPPConfig config;
        config.stages = {
            PPStage::fromTPContext(mock_tp_ctx_),     // ROCm TP domain
            PPStage::fromTPContext(mock_tp_ctx_2),    // CUDA TP domain
            PPStage::fromDevice(GlobalDeviceAddress::cpu())
        };
        config.layer_boundaries = {0, 10, 20, 24};

        EXPECT_TRUE(config.isValid());
        EXPECT_EQ(config.numStages(), 3);

        // Verify stage types
        EXPECT_TRUE(config.stages[0].isTPDomain());
        EXPECT_TRUE(config.stages[1].isTPDomain());
        EXPECT_TRUE(config.stages[2].isSingleDevice());

        // Verify representative devices
        auto dev0 = config.stages[0].representativeDevice();
        EXPECT_EQ(dev0.device_type, DeviceType::ROCm);

        auto dev1 = config.stages[1].representativeDevice();
        EXPECT_EQ(dev1.device_type, DeviceType::CUDA);

        auto dev2 = config.stages[2].representativeDevice();
        EXPECT_EQ(dev2.device_type, DeviceType::CPU);
    }

} // namespace llaminar2::test
