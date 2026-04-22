/**
 * @file Test__RankOrchestrator_PPConfig.cpp
 * @brief Unit tests for RankOrchestrator PP config validation
 *
 * Tests the Phase 1 PP config extensions:
 * - ParallelismMode enum and detection
 * - PPStageConfig validation
 * - Config::validate() for PP mode
 * - Mode detection (TP, PP, TP_PP)
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#include <gtest/gtest.h>
#include "execution/local_execution/orchestrators/RankOrchestrator.h"
#include "backends/GlobalDeviceAddress.h"

namespace llaminar2::test
{

    class Test__RankOrchestrator_PPConfig : public ::testing::Test
    {
    protected:
        using Config = RankOrchestrator::Config;
        using PPStageConfig = RankOrchestrator::PPStageConfig;
        using ParallelismMode = RankOrchestrator::ParallelismMode;
    };

    // =========================================================================
    // PPStageConfig Validation Tests
    // =========================================================================

    TEST_F(Test__RankOrchestrator_PPConfig, PPStageConfig_ValidSingleDevice)
    {
        PPStageConfig stage;
        stage.first_layer = 0;
        stage.last_layer = 12;
        stage.stage_devices = {GlobalDeviceAddress::cuda(0)};

        EXPECT_TRUE(stage.validate());
        EXPECT_EQ(stage.numLayers(), 12);
        EXPECT_FALSE(stage.isTPDomain());
    }

    TEST_F(Test__RankOrchestrator_PPConfig, PPStageConfig_ValidTPDomain)
    {
        PPStageConfig stage;
        stage.first_layer = 12;
        stage.last_layer = 24;
        stage.stage_devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};
        stage.tp_weights = {0.6f, 0.4f};

        EXPECT_TRUE(stage.validate());
        EXPECT_EQ(stage.numLayers(), 12);
        EXPECT_TRUE(stage.isTPDomain());
    }

    TEST_F(Test__RankOrchestrator_PPConfig, PPStageConfig_InvalidLayerRange)
    {
        PPStageConfig stage;
        stage.first_layer = 12;
        stage.last_layer = 6; // Invalid: last <= first
        stage.stage_devices = {GlobalDeviceAddress::cuda(0)};

        EXPECT_FALSE(stage.validate());
    }

    TEST_F(Test__RankOrchestrator_PPConfig, PPStageConfig_EmptyDevices)
    {
        PPStageConfig stage;
        stage.first_layer = 0;
        stage.last_layer = 12;
        stage.stage_devices = {}; // Invalid: no devices

        EXPECT_FALSE(stage.validate());
    }

    TEST_F(Test__RankOrchestrator_PPConfig, PPStageConfig_MismatchedWeights)
    {
        PPStageConfig stage;
        stage.first_layer = 0;
        stage.last_layer = 12;
        stage.stage_devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};
        stage.tp_weights = {0.5f}; // Invalid: wrong count

        EXPECT_FALSE(stage.validate());
    }

    // =========================================================================
    // ParallelismMode Detection Tests
    // =========================================================================

    TEST_F(Test__RankOrchestrator_PPConfig, DetectMode_TP)
    {
        Config config;
        config.devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};
        config.pp_stages.clear();

        EXPECT_EQ(config.detectMode(), ParallelismMode::TP);
        EXPECT_EQ(config.effectiveMode(), ParallelismMode::TP);
    }

    TEST_F(Test__RankOrchestrator_PPConfig, DetectMode_PP)
    {
        Config config;
        config.devices.clear();

        // Two single-device PP stages
        PPStageConfig stage0;
        stage0.first_layer = 0;
        stage0.last_layer = 12;
        stage0.stage_devices = {GlobalDeviceAddress::cuda(0)};

        PPStageConfig stage1;
        stage1.first_layer = 12;
        stage1.last_layer = 24;
        stage1.stage_devices = {GlobalDeviceAddress::cuda(1)};

        config.pp_stages = {stage0, stage1};

        EXPECT_EQ(config.detectMode(), ParallelismMode::PP);
        EXPECT_EQ(config.effectiveMode(), ParallelismMode::PP);
    }

    TEST_F(Test__RankOrchestrator_PPConfig, DetectMode_TP_PP)
    {
        Config config;
        config.devices.clear();

        // First stage is a TP domain (2 devices)
        PPStageConfig stage0;
        stage0.first_layer = 0;
        stage0.last_layer = 12;
        stage0.stage_devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};

        // Second stage is single device
        PPStageConfig stage1;
        stage1.first_layer = 12;
        stage1.last_layer = 24;
        stage1.stage_devices = {GlobalDeviceAddress::rocm(0)};

        config.pp_stages = {stage0, stage1};

        EXPECT_EQ(config.detectMode(), ParallelismMode::TP_PP);
        EXPECT_EQ(config.effectiveMode(), ParallelismMode::TP_PP);
    }

    TEST_F(Test__RankOrchestrator_PPConfig, EffectiveMode_AutoResolves)
    {
        Config config;
        config.mode = ParallelismMode::AUTO;
        config.devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};

        EXPECT_EQ(config.effectiveMode(), ParallelismMode::TP);
    }

    TEST_F(Test__RankOrchestrator_PPConfig, EffectiveMode_ExplicitOverridesAuto)
    {
        Config config;
        config.mode = ParallelismMode::TP;

        // Even with PP stages configured, explicit TP mode should be respected
        PPStageConfig stage0;
        stage0.first_layer = 0;
        stage0.last_layer = 12;
        stage0.stage_devices = {GlobalDeviceAddress::cuda(0)};
        config.pp_stages = {stage0};

        // effectiveMode returns explicit mode, not detected mode
        EXPECT_EQ(config.effectiveMode(), ParallelismMode::TP);
        EXPECT_EQ(config.detectMode(), ParallelismMode::PP);
    }

    // =========================================================================
    // Config::validate() Tests for TP Mode
    // =========================================================================

    TEST_F(Test__RankOrchestrator_PPConfig, Validate_TP_Valid)
    {
        Config config;
        config.devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};
        config.weights = {0.5f, 0.5f};

        EXPECT_TRUE(config.validate());
    }

    TEST_F(Test__RankOrchestrator_PPConfig, Validate_TP_NoDevices)
    {
        Config config;
        config.devices.clear();
        config.mode = ParallelismMode::TP;

        EXPECT_FALSE(config.validate());
    }

    TEST_F(Test__RankOrchestrator_PPConfig, Validate_TP_MismatchedWeights)
    {
        Config config;
        config.devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};
        config.weights = {0.5f}; // Wrong count

        EXPECT_FALSE(config.validate());
    }

    // =========================================================================
    // Config::validate() Tests for PP Mode
    // =========================================================================

    TEST_F(Test__RankOrchestrator_PPConfig, Validate_PP_Valid)
    {
        Config config;
        config.devices.clear();

        PPStageConfig stage0;
        stage0.first_layer = 0;
        stage0.last_layer = 12;
        stage0.stage_devices = {GlobalDeviceAddress::cuda(0)};

        PPStageConfig stage1;
        stage1.first_layer = 12;
        stage1.last_layer = 24;
        stage1.stage_devices = {GlobalDeviceAddress::cuda(1)};

        config.pp_stages = {stage0, stage1};

        EXPECT_TRUE(config.validate());
    }

    TEST_F(Test__RankOrchestrator_PPConfig, Validate_PP_LayerGap)
    {
        Config config;
        config.devices.clear();

        PPStageConfig stage0;
        stage0.first_layer = 0;
        stage0.last_layer = 10;
        stage0.stage_devices = {GlobalDeviceAddress::cuda(0)};

        PPStageConfig stage1;
        stage1.first_layer = 12; // Gap! Should be 10
        stage1.last_layer = 24;
        stage1.stage_devices = {GlobalDeviceAddress::cuda(1)};

        config.pp_stages = {stage0, stage1};

        EXPECT_FALSE(config.validate());
    }

    TEST_F(Test__RankOrchestrator_PPConfig, Validate_PP_InvalidStage)
    {
        Config config;
        config.devices.clear();

        PPStageConfig stage0;
        stage0.first_layer = 0;
        stage0.last_layer = 12;
        stage0.stage_devices.clear(); // Invalid stage - no devices

        config.pp_stages = {stage0};

        EXPECT_FALSE(config.validate());
    }

    // =========================================================================
    // Helper Method Tests
    // =========================================================================

    TEST_F(Test__RankOrchestrator_PPConfig, HasPP_True)
    {
        Config config;
        PPStageConfig stage;
        stage.first_layer = 0;
        stage.last_layer = 24;
        stage.stage_devices = {GlobalDeviceAddress::cuda(0)};
        config.pp_stages = {stage};

        EXPECT_TRUE(config.hasPP());
    }

    TEST_F(Test__RankOrchestrator_PPConfig, HasPP_False)
    {
        Config config;
        config.pp_stages.clear();

        EXPECT_FALSE(config.hasPP());
    }

    TEST_F(Test__RankOrchestrator_PPConfig, BuildLayerBoundaries_TwoStages)
    {
        Config config;

        PPStageConfig stage0;
        stage0.first_layer = 0;
        stage0.last_layer = 12;
        stage0.stage_devices = {GlobalDeviceAddress::cuda(0)};

        PPStageConfig stage1;
        stage1.first_layer = 12;
        stage1.last_layer = 24;
        stage1.stage_devices = {GlobalDeviceAddress::cuda(1)};

        config.pp_stages = {stage0, stage1};

        auto boundaries = config.buildLayerBoundaries();
        ASSERT_EQ(boundaries.size(), 3u);
        EXPECT_EQ(boundaries[0], 0);
        EXPECT_EQ(boundaries[1], 12);
        EXPECT_EQ(boundaries[2], 24);
    }

    TEST_F(Test__RankOrchestrator_PPConfig, BuildLayerBoundaries_Empty)
    {
        Config config;
        config.pp_stages.clear();

        auto boundaries = config.buildLayerBoundaries();
        EXPECT_TRUE(boundaries.empty());
    }

    TEST_F(Test__RankOrchestrator_PPConfig, GetNormalizedWeights_DefaultsToEqual)
    {
        Config config;
        config.devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};
        config.weights.clear();

        auto weights = config.getNormalizedWeights();
        ASSERT_EQ(weights.size(), 2u);
        EXPECT_FLOAT_EQ(weights[0], 0.5f);
        EXPECT_FLOAT_EQ(weights[1], 0.5f);
    }

} // namespace llaminar2::test
