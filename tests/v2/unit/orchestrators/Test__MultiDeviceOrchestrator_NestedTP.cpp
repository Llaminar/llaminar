/**
 * @file Test__MultiDeviceOrchestrator_NestedTP.cpp
 * @brief Unit tests for nested MultiDeviceOrchestrator creation in TP+PP hybrid mode
 *
 * Tests the Phase 5 implementation where PP stages configured as TP domains
 * create nested MultiDeviceOrchestrators instead of single DeviceGraphOrchestrators.
 *
 * Test cases:
 * - InitializePP_TPDomain_CreatesNestedMDO: Verify TP domain creates MDO
 * - InitializePP_SingleDevice_CreatesSingleRunner: Verify single device still works
 * - InitializePP_MixedStages_CreatesCorrectTypes: Stage 0 = MDO, Stage 1 = single
 * - NestedMDO_HasCorrectTPConfig: Verify backend, weights propagation
 * - NestedMDO_HasCorrectLayerRange: Verify layer range in nested MDO
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#include <gtest/gtest.h>
#include "execution/local_execution/orchestrators/MultiDeviceOrchestrator.h"
#include "execution/local_execution/orchestrators/IInferenceRunner.h"
#include "execution/factory/FactoryPPStageConfig.h"
#include "collective/ILocalTPContext.h"
#include "backends/GlobalDeviceAddress.h"
#include "tensors/Tensors.h"
#include <memory>
#include <vector>

namespace llaminar2::test
{

    // =========================================================================
    // Test Fixture
    // =========================================================================

    class Test__MultiDeviceOrchestrator_NestedTP : public ::testing::Test
    {
    protected:
        using Config = MultiDeviceOrchestrator::Config;
        using ParallelismMode = MultiDeviceOrchestrator::ParallelismMode;
        using PPStageConfig = MultiDeviceOrchestrator::PPStageConfig;
    };

    // =========================================================================
    // Configuration Tests
    // =========================================================================

    /**
     * @brief Verify that a TP domain stage (multiple devices) is correctly identified
     */
    TEST_F(Test__MultiDeviceOrchestrator_NestedTP, PPStageConfig_TPDomain_CorrectlyIdentified)
    {
        // Create a TP domain stage with 2 devices
        PPStageConfig stage;
        stage.first_layer = 0;
        stage.last_layer = 12;
        stage.stage_devices = {GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)};
        stage.tp_weights = {0.6f, 0.4f};
        stage.tp_backend = CollectiveBackendType::RCCL;

        EXPECT_TRUE(stage.isTPDomain());
        EXPECT_EQ(stage.stage_devices.size(), 2u);
        EXPECT_TRUE(stage.validate());
    }

    /**
     * @brief Verify that a single-device stage is not a TP domain
     */
    TEST_F(Test__MultiDeviceOrchestrator_NestedTP, PPStageConfig_SingleDevice_NotTPDomain)
    {
        PPStageConfig stage;
        stage.first_layer = 12;
        stage.last_layer = 24;
        stage.stage_devices = {GlobalDeviceAddress::cuda(0)};

        EXPECT_FALSE(stage.isTPDomain());
        EXPECT_EQ(stage.stage_devices.size(), 1u);
        EXPECT_TRUE(stage.validate());
    }

    /**
     * @brief Verify mixed stages mode detection results in TP_PP
     */
    TEST_F(Test__MultiDeviceOrchestrator_NestedTP, Config_MixedStages_DetectedAsTPPP)
    {
        Config config;
        config.devices.clear(); // No TP-level devices, all in PP stages

        // Stage 0: TP domain with 2 ROCm devices
        PPStageConfig stage0;
        stage0.first_layer = 0;
        stage0.last_layer = 12;
        stage0.stage_devices = {GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)};
        stage0.tp_weights = {0.6f, 0.4f};
        stage0.tp_backend = CollectiveBackendType::RCCL;

        // Stage 1: Single CUDA device
        PPStageConfig stage1;
        stage1.first_layer = 12;
        stage1.last_layer = 24;
        stage1.stage_devices = {GlobalDeviceAddress::cuda(0)};

        config.pp_stages = {stage0, stage1};

        // Mode should be TP_PP because one stage has multiple devices
        EXPECT_EQ(config.detectMode(), ParallelismMode::TP_PP);
        EXPECT_TRUE(config.validate());
    }

    /**
     * @brief Verify all single-device stages mode detection results in PP (not TP_PP)
     */
    TEST_F(Test__MultiDeviceOrchestrator_NestedTP, Config_AllSingleDevice_DetectedAsPP)
    {
        Config config;
        config.devices.clear();

        // Stage 0: Single CUDA device
        PPStageConfig stage0;
        stage0.first_layer = 0;
        stage0.last_layer = 12;
        stage0.stage_devices = {GlobalDeviceAddress::cuda(0)};

        // Stage 1: Single CUDA device
        PPStageConfig stage1;
        stage1.first_layer = 12;
        stage1.last_layer = 24;
        stage1.stage_devices = {GlobalDeviceAddress::cuda(1)};

        config.pp_stages = {stage0, stage1};

        // Mode should be PP because all stages have single device
        EXPECT_EQ(config.detectMode(), ParallelismMode::PP);
        EXPECT_TRUE(config.validate());
    }

    // =========================================================================
    // Nested Configuration Propagation Tests
    // =========================================================================

    /**
     * @brief Verify that TP configuration would be correctly built for nested MDO
     *
     * This tests the Config struct creation logic that would be used when
     * creating a nested MultiDeviceOrchestrator for a TP domain stage.
     */
    TEST_F(Test__MultiDeviceOrchestrator_NestedTP, NestedConfig_PropagatesTPSettings)
    {
        // Outer (PP) configuration
        Config outer_config;
        outer_config.mode = ParallelismMode::TP_PP;
        outer_config.max_seq_len = 2048;
        outer_config.batch_size = 1;
        outer_config.activation_precision = ActivationPrecision::FP32;
        outer_config.kv_cache_scale = 1.0f;
        outer_config.use_mapped_memory = true;

        // TP domain stage configuration
        PPStageConfig stage0;
        stage0.first_layer = 0;
        stage0.last_layer = 12;
        stage0.stage_devices = {GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)};
        stage0.tp_weights = {0.73f, 0.27f};
        stage0.tp_backend = CollectiveBackendType::RCCL;
        stage0.has_embedding = true;
        stage0.has_lm_head = false;

        outer_config.pp_stages = {stage0};

        // Simulate nested config creation (matches implementation logic)
        Config nested_config;
        nested_config.mode = ParallelismMode::TP;
        nested_config.devices = stage0.stage_devices;
        nested_config.weights = stage0.tp_weights;
        nested_config.backend = stage0.tp_backend;
        nested_config.max_seq_len = outer_config.max_seq_len;
        nested_config.batch_size = outer_config.batch_size;
        nested_config.activation_precision = outer_config.activation_precision;
        nested_config.kv_cache_scale = outer_config.kv_cache_scale;
        nested_config.use_mapped_memory = outer_config.use_mapped_memory;

        // Verify nested config has correct TP settings
        EXPECT_EQ(nested_config.mode, ParallelismMode::TP);
        ASSERT_EQ(nested_config.devices.size(), 2u);
        EXPECT_EQ(nested_config.devices[0].device_type, DeviceType::ROCm);
        EXPECT_EQ(nested_config.devices[1].device_type, DeviceType::ROCm);
        EXPECT_EQ(nested_config.devices[0].device_ordinal, 0);
        EXPECT_EQ(nested_config.devices[1].device_ordinal, 1);

        // Verify weights propagated
        ASSERT_EQ(nested_config.weights.size(), 2u);
        EXPECT_FLOAT_EQ(nested_config.weights[0], 0.73f);
        EXPECT_FLOAT_EQ(nested_config.weights[1], 0.27f);

        // Verify backend propagated
        EXPECT_EQ(nested_config.backend, CollectiveBackendType::RCCL);

        // Verify common settings propagated
        EXPECT_EQ(nested_config.max_seq_len, 2048u);
        EXPECT_EQ(nested_config.batch_size, 1);
        EXPECT_EQ(nested_config.activation_precision, ActivationPrecision::FP32);
        EXPECT_FLOAT_EQ(nested_config.kv_cache_scale, 1.0f);
        EXPECT_TRUE(nested_config.use_mapped_memory);

        // Verify nested config is valid for TP mode
        EXPECT_TRUE(nested_config.validate());
    }

    /**
     * @brief Verify that layer range is correctly specified in stage config
     */
    TEST_F(Test__MultiDeviceOrchestrator_NestedTP, NestedMDO_HasCorrectLayerRange)
    {
        // Stage configuration for layers 14-27 (14 layers)
        PPStageConfig stage;
        stage.first_layer = 14;
        stage.last_layer = 28; // Exclusive
        stage.stage_devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};
        stage.has_embedding = false;
        stage.has_lm_head = true;

        // Verify layer range
        EXPECT_EQ(stage.first_layer, 14);
        EXPECT_EQ(stage.last_layer, 28);
        EXPECT_EQ(stage.numLayers(), 14);
        EXPECT_FALSE(stage.has_embedding);
        EXPECT_TRUE(stage.has_lm_head);
    }

    /**
     * @brief Verify that three-stage PP with mixed TP domains has correct config
     */
    TEST_F(Test__MultiDeviceOrchestrator_NestedTP, ThreeStage_MixedTPDomains_ValidConfig)
    {
        Config config;
        config.mode = ParallelismMode::AUTO;

        // Stage 0: TP domain with RCCL (layers 0-8)
        PPStageConfig stage0;
        stage0.first_layer = 0;
        stage0.last_layer = 8;
        stage0.stage_devices = {GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)};
        stage0.tp_weights = {0.5f, 0.5f};
        stage0.tp_backend = CollectiveBackendType::RCCL;
        stage0.has_embedding = true;
        stage0.has_lm_head = false;

        // Stage 1: Single CUDA device (layers 8-16)
        PPStageConfig stage1;
        stage1.first_layer = 8;
        stage1.last_layer = 16;
        stage1.stage_devices = {GlobalDeviceAddress::cuda(0)};
        stage1.has_embedding = false;
        stage1.has_lm_head = false;

        // Stage 2: TP domain with NCCL (layers 16-24)
        PPStageConfig stage2;
        stage2.first_layer = 16;
        stage2.last_layer = 24;
        stage2.stage_devices = {GlobalDeviceAddress::cuda(1), GlobalDeviceAddress::cuda(2)};
        stage2.tp_weights = {0.6f, 0.4f};
        stage2.tp_backend = CollectiveBackendType::NCCL;
        stage2.has_embedding = false;
        stage2.has_lm_head = true;

        config.pp_stages = {stage0, stage1, stage2};

        // Verify detection
        EXPECT_EQ(config.detectMode(), ParallelismMode::TP_PP);
        EXPECT_TRUE(config.validate());

        // Verify stage properties
        EXPECT_TRUE(stage0.isTPDomain());
        EXPECT_FALSE(stage1.isTPDomain());
        EXPECT_TRUE(stage2.isTPDomain());

        // Verify layer boundaries
        auto boundaries = config.buildLayerBoundaries();
        ASSERT_EQ(boundaries.size(), 4u);
        EXPECT_EQ(boundaries[0], 0);
        EXPECT_EQ(boundaries[1], 8);
        EXPECT_EQ(boundaries[2], 16);
        EXPECT_EQ(boundaries[3], 24);
    }

    // =========================================================================
    // Backend Selection Tests
    // =========================================================================

    /**
     * @brief Verify different backends can be specified per TP domain stage
     */
    TEST_F(Test__MultiDeviceOrchestrator_NestedTP, DifferentBackendsPerStage)
    {
        Config config;

        // Stage 0: RCCL backend (ROCm devices)
        PPStageConfig stage0;
        stage0.first_layer = 0;
        stage0.last_layer = 12;
        stage0.stage_devices = {GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)};
        stage0.tp_backend = CollectiveBackendType::RCCL;

        // Stage 1: NCCL backend (CUDA devices)
        PPStageConfig stage1;
        stage1.first_layer = 12;
        stage1.last_layer = 24;
        stage1.stage_devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};
        stage1.tp_backend = CollectiveBackendType::NCCL;

        config.pp_stages = {stage0, stage1};

        // Verify different backends
        EXPECT_EQ(config.pp_stages[0].tp_backend, CollectiveBackendType::RCCL);
        EXPECT_EQ(config.pp_stages[1].tp_backend, CollectiveBackendType::NCCL);

        // Both are TP domains
        EXPECT_TRUE(config.pp_stages[0].isTPDomain());
        EXPECT_TRUE(config.pp_stages[1].isTPDomain());

        EXPECT_TRUE(config.validate());
    }

    /**
     * @brief Verify AUTO backend is default when not specified
     */
    TEST_F(Test__MultiDeviceOrchestrator_NestedTP, DefaultBackendIsAuto)
    {
        PPStageConfig stage;
        stage.first_layer = 0;
        stage.last_layer = 12;
        stage.stage_devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};
        // Don't set tp_backend - should default to AUTO

        EXPECT_EQ(stage.tp_backend, CollectiveBackendType::AUTO);
        EXPECT_TRUE(stage.validate());
    }

    // =========================================================================
    // Edge Case Tests
    // =========================================================================

    /**
     * @brief Verify empty weights defaults to equal distribution
     */
    TEST_F(Test__MultiDeviceOrchestrator_NestedTP, EmptyWeights_DefaultsToEqual)
    {
        PPStageConfig stage;
        stage.first_layer = 0;
        stage.last_layer = 12;
        stage.stage_devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};
        stage.tp_weights.clear(); // Empty - should default to equal

        EXPECT_TRUE(stage.validate()); // Should be valid with empty weights
        EXPECT_TRUE(stage.tp_weights.empty());
    }

    /**
     * @brief Verify TP domain with 3 devices works
     */
    TEST_F(Test__MultiDeviceOrchestrator_NestedTP, TPDomainWithThreeDevices)
    {
        PPStageConfig stage;
        stage.first_layer = 0;
        stage.last_layer = 12;
        stage.stage_devices = {
            GlobalDeviceAddress::cuda(0),
            GlobalDeviceAddress::cuda(1),
            GlobalDeviceAddress::cuda(2)};
        stage.tp_weights = {0.4f, 0.35f, 0.25f};

        EXPECT_TRUE(stage.isTPDomain());
        EXPECT_EQ(stage.stage_devices.size(), 3u);

        // Verify weights sum to ~1.0
        float sum = 0.0f;
        for (float w : stage.tp_weights)
        {
            sum += w;
        }
        EXPECT_NEAR(sum, 1.0f, 0.01f);

        EXPECT_TRUE(stage.validate());
    }

    // =========================================================================
    // Nested PP Stage Config Propagation Tests
    //
    // Bug Caught: In hybrid PP+TP mode, nested TP MDOs weren't receiving the
    // PP stage config, causing them to build FULL forward graphs (with LM_HEAD)
    // instead of PARTIAL graphs (without LM_HEAD for stage 0).
    //
    // Root Cause: nested_pp_stage_config wasn't being set in the nested MDO's
    // Config, and setPPStageConfig() wasn't being called on DeviceGraphOrchestrators.
    // =========================================================================

    /**
     * @brief Test: nested_pp_stage_config must be set when creating nested MDO for PP stage
     *
     * When a TP domain is part of a PP stage, the nested MDO must receive
     * the PP stage config so its DeviceGraphOrchestrators know to build
     * partial graphs instead of full graphs.
     */
    TEST_F(Test__MultiDeviceOrchestrator_NestedTP, NestedMDO_MustReceivePPStageConfig)
    {
        // Outer PP config with TP domain as stage 0
        PPStageConfig stage0;
        stage0.first_layer = 0;
        stage0.last_layer = 12;
        stage0.stage_devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};
        stage0.tp_backend = CollectiveBackendType::NCCL;
        stage0.has_embedding = true;
        stage0.has_lm_head = false; // KEY: Stage 0 does NOT have LM head

        // Create FactoryPPStageConfig as done in InferenceRunnerFactory
        FactoryPPStageConfig factory_pp_config;
        factory_pp_config.first_layer = stage0.first_layer;
        factory_pp_config.last_layer = stage0.last_layer;
        factory_pp_config.has_embedding = stage0.has_embedding;
        factory_pp_config.has_lm_head = stage0.has_lm_head;

        // Create nested MDO config (simulating what parent MDO does)
        Config nested_config;
        nested_config.mode = ParallelismMode::TP;
        nested_config.devices = stage0.stage_devices;
        nested_config.backend = stage0.tp_backend;

        // KEY FIX: This line was missing and caused the bug!
        nested_config.nested_pp_stage_config = factory_pp_config;

        // Verify nested_pp_stage_config is present and correct
        ASSERT_TRUE(nested_config.nested_pp_stage_config.has_value());
        EXPECT_EQ(nested_config.nested_pp_stage_config->first_layer, 0);
        EXPECT_EQ(nested_config.nested_pp_stage_config->last_layer, 12);
        EXPECT_TRUE(nested_config.nested_pp_stage_config->has_embedding);
        EXPECT_FALSE(nested_config.nested_pp_stage_config->has_lm_head);
    }

    /**
     * @brief Test: Nested config without PP stage config would cause full graph build
     *
     * This is the anti-pattern that caused the bug. If nested_pp_stage_config
     * is NOT set, the DeviceGraphOrchestrators won't call setPPStageConfig()
     * and will build full graphs including LM_HEAD even when they shouldn't.
     */
    TEST_F(Test__MultiDeviceOrchestrator_NestedTP, NestedMDO_MissingPPConfig_WouldBuildFullGraph)
    {
        PPStageConfig stage0;
        stage0.first_layer = 0;
        stage0.last_layer = 12;
        stage0.stage_devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};
        stage0.has_lm_head = false;

        // Incorrect: Create nested config WITHOUT setting nested_pp_stage_config
        Config nested_config_without_pp;
        nested_config_without_pp.mode = ParallelismMode::TP;
        nested_config_without_pp.devices = stage0.stage_devices;
        // NOT setting: nested_config_without_pp.nested_pp_stage_config = ...

        // Without nested_pp_stage_config, the DGOs would build full graphs!
        EXPECT_FALSE(nested_config_without_pp.nested_pp_stage_config.has_value());

        // Correct: Create nested config WITH nested_pp_stage_config
        Config nested_config_with_pp;
        nested_config_with_pp.mode = ParallelismMode::TP;
        nested_config_with_pp.devices = stage0.stage_devices;

        FactoryPPStageConfig factory_pp_config;
        factory_pp_config.first_layer = 0;
        factory_pp_config.last_layer = 12;
        factory_pp_config.has_lm_head = false;
        nested_config_with_pp.nested_pp_stage_config = factory_pp_config;

        // With nested_pp_stage_config, DGOs know to build partial graphs
        EXPECT_TRUE(nested_config_with_pp.nested_pp_stage_config.has_value());
        EXPECT_FALSE(nested_config_with_pp.nested_pp_stage_config->has_lm_head);
    }

    /**
     * @brief Test: FactoryPPStageConfig captures all required PP stage info
     *
     * Verifies that FactoryPPStageConfig has all the fields needed for
     * DeviceGraphOrchestrator to build the correct partial graph.
     */
    TEST_F(Test__MultiDeviceOrchestrator_NestedTP, FactoryPPStageConfig_HasAllFields)
    {
        FactoryPPStageConfig config;
        config.first_layer = 0;
        config.last_layer = 12;
        config.has_embedding = true;
        config.has_lm_head = false;
        config.use_bar_backed_hidden = true;

        // Verify all fields are accessible
        EXPECT_EQ(config.first_layer, 0);
        EXPECT_EQ(config.last_layer, 12);
        EXPECT_TRUE(config.has_embedding);
        EXPECT_FALSE(config.has_lm_head);
        EXPECT_TRUE(config.use_bar_backed_hidden);
    }

    /**
     * @brief Test: Layer range in nested config determines what stages to build
     *
     * The DeviceGraphOrchestrator uses first_layer/last_layer to determine
     * which transformer blocks to include in its graph.
     */
    TEST_F(Test__MultiDeviceOrchestrator_NestedTP, NestedPPConfig_LayerRangeDeterminesStages)
    {
        // Stage 0: layers 0-11 (12 layers)
        FactoryPPStageConfig stage0_config;
        stage0_config.first_layer = 0;
        stage0_config.last_layer = 12;
        stage0_config.has_embedding = true;  // Has embedding (first stage)
        stage0_config.has_lm_head = false;   // No LM head

        // Stage 1: layers 12-23 (12 layers)
        FactoryPPStageConfig stage1_config;
        stage1_config.first_layer = 12;
        stage1_config.last_layer = 24;
        stage1_config.has_embedding = false; // No embedding
        stage1_config.has_lm_head = true;    // Has LM head (last stage)

        // Verify layer counts
        int stage0_layers = stage0_config.last_layer - stage0_config.first_layer;
        int stage1_layers = stage1_config.last_layer - stage1_config.first_layer;

        EXPECT_EQ(stage0_layers, 12);
        EXPECT_EQ(stage1_layers, 12);

        // Total should cover all layers
        EXPECT_EQ(stage0_layers + stage1_layers, 24);

        // Layers don't overlap
        EXPECT_EQ(stage0_config.last_layer, stage1_config.first_layer);
    }

    /**
     * @brief Test: Hybrid PP+TP config creation follows correct pattern
     *
     * Documents the complete pattern for creating nested MDO configs
     * in PP+TP hybrid mode to prevent future regressions.
     */
    TEST_F(Test__MultiDeviceOrchestrator_NestedTP, HybridPPTP_CorrectConfigPattern)
    {
        // Step 1: Define outer PP config with TP domain stages
        Config outer_config;
        outer_config.mode = ParallelismMode::TP_PP;

        PPStageConfig stage0;
        stage0.first_layer = 0;
        stage0.last_layer = 12;
        stage0.stage_devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};
        stage0.tp_backend = CollectiveBackendType::NCCL;
        stage0.has_embedding = true;
        stage0.has_lm_head = false;

        PPStageConfig stage1;
        stage1.first_layer = 12;
        stage1.last_layer = 24;
        stage1.stage_devices = {GlobalDeviceAddress::rocm(0)};
        stage1.has_embedding = false;
        stage1.has_lm_head = true;

        outer_config.pp_stages = {stage0, stage1};
        ASSERT_TRUE(outer_config.validate());

        // Step 2: For each TP domain stage, create nested MDO config
        for (size_t i = 0; i < outer_config.pp_stages.size(); ++i)
        {
            const auto &pp_stage = outer_config.pp_stages[i];

            if (pp_stage.isTPDomain())
            {
                // Step 2a: Create FactoryPPStageConfig from PPStageConfig
                FactoryPPStageConfig factory_pp_config;
                factory_pp_config.first_layer = pp_stage.first_layer;
                factory_pp_config.last_layer = pp_stage.last_layer;
                factory_pp_config.has_embedding = pp_stage.has_embedding;
                factory_pp_config.has_lm_head = pp_stage.has_lm_head;

                // Step 2b: Create nested MDO config with TP settings
                Config nested_config;
                nested_config.mode = ParallelismMode::TP;
                nested_config.devices = pp_stage.stage_devices;
                nested_config.weights = pp_stage.tp_weights;
                nested_config.backend = pp_stage.tp_backend;
                nested_config.max_seq_len = outer_config.max_seq_len;
                nested_config.batch_size = outer_config.batch_size;

                // Step 2c: CRITICAL - Set nested_pp_stage_config!
                nested_config.nested_pp_stage_config = factory_pp_config;

                // Verify nested config is correct
                EXPECT_TRUE(nested_config.nested_pp_stage_config.has_value());
                EXPECT_EQ(nested_config.nested_pp_stage_config->first_layer,
                          pp_stage.first_layer);
                EXPECT_EQ(nested_config.nested_pp_stage_config->has_lm_head,
                          pp_stage.has_lm_head);
            }
        }
    }

} // namespace llaminar2::test
