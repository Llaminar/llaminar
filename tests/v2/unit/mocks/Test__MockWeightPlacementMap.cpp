/**
 * @file Test__MockWeightPlacementMap.cpp
 * @brief Unit tests for MockWeightPlacementMap
 *
 * Tests the mock weight placement map implementation including:
 * - Default construction and configuration
 * - Builder pattern for complex setups
 * - Tensor/layer/pattern device mapping
 * - Block-level (attention/FFN) mapping
 * - MoE expert mapping
 * - Call tracking
 * - Failure injection
 * - Preset scenarios
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include "mocks/MockWeightPlacementMap.h"
#include "execution/mpi_orchestration/PlacementPlan.h" // For PlacementPlan in applyPlan tests
#include <memory>
#include <vector>
#include <thread>

using namespace llaminar2;
using namespace llaminar2::test;

// =============================================================================
// Test Fixture
// =============================================================================

class Test__MockWeightPlacementMap : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Default CPU-only mock
        mock_cpu_ = std::make_shared<MockWeightPlacementMap>();

        // GPU mock
        mock_gpu_ = std::make_shared<MockWeightPlacementMap>(DeviceId::cuda(0));
    }

    std::shared_ptr<MockWeightPlacementMap> mock_cpu_;
    std::shared_ptr<MockWeightPlacementMap> mock_gpu_;
};

// =============================================================================
// Default Construction Tests
// =============================================================================

TEST_F(Test__MockWeightPlacementMap, DefaultConstruction_CPUDefault)
{
    MockWeightPlacementMap mock;
    EXPECT_TRUE(mock.defaultDevice().is_cpu());
    EXPECT_FALSE(mock.hasPlan());
    EXPECT_TRUE(mock.appliedStrategyName().empty());
}

TEST_F(Test__MockWeightPlacementMap, DefaultConstruction_NoMappings)
{
    MockWeightPlacementMap mock;
    EXPECT_EQ(mock.tensorMappingCount(), 0);
    EXPECT_EQ(mock.layerMappingCount(), 0);
}

TEST_F(Test__MockWeightPlacementMap, Construction_WithDeviceId)
{
    MockWeightPlacementMap mock(DeviceId::cuda(1));
    EXPECT_TRUE(mock.defaultDevice().is_cuda());
    EXPECT_EQ(mock.defaultDevice().ordinal, 1);
}

TEST_F(Test__MockWeightPlacementMap, Construction_WithConfig)
{
    MockWeightPlacementMap::Config config;
    config.default_device = DeviceId::cuda(2);
    config.applied_strategy_name = "test_strategy";
    config.has_plan = true;

    MockWeightPlacementMap mock(config);
    EXPECT_TRUE(mock.defaultDevice().is_cuda());
    EXPECT_EQ(mock.defaultDevice().ordinal, 2);
    EXPECT_TRUE(mock.hasPlan());
    EXPECT_EQ(mock.appliedStrategyName(), "test_strategy");
}

// =============================================================================
// Builder Pattern Tests
// =============================================================================

TEST_F(Test__MockWeightPlacementMap, Builder_DefaultDevice)
{
    auto mock = MockWeightPlacementMap::Builder()
                    .withDefaultDevice(DeviceId::cuda(0))
                    .build();

    EXPECT_TRUE(mock->defaultDevice().is_cuda());
    EXPECT_EQ(mock->defaultDevice().ordinal, 0);
}

TEST_F(Test__MockWeightPlacementMap, Builder_StrategyName)
{
    auto mock = MockWeightPlacementMap::Builder()
                    .withStrategyName("my_strategy")
                    .build();

    EXPECT_TRUE(mock->hasPlan());
    EXPECT_EQ(mock->appliedStrategyName(), "my_strategy");
}

TEST_F(Test__MockWeightPlacementMap, Builder_TensorMapping)
{
    auto mock = MockWeightPlacementMap::Builder()
                    .withTensorDevice("embed_tokens.weight", DeviceId::cuda(0))
                    .withTensorDevice("lm_head.weight", DeviceId::cuda(1))
                    .build();

    EXPECT_EQ(mock->tensorMappingCount(), 2);
    EXPECT_TRUE(mock->getDeviceForWeight("embed_tokens.weight").is_cuda());
    EXPECT_EQ(mock->getDeviceForWeight("lm_head.weight").ordinal, 1);
}

TEST_F(Test__MockWeightPlacementMap, Builder_LayerMapping)
{
    auto mock = MockWeightPlacementMap::Builder()
                    .withDefaultDevice(DeviceId::cpu())
                    .withLayerDevice(0, DeviceId::cuda(0))
                    .withLayerDevice(1, DeviceId::cuda(1))
                    .build();

    EXPECT_EQ(mock->layerMappingCount(), 2);
    EXPECT_TRUE(mock->getDeviceForWeight("blk.0.attn_q.weight", 0).is_cuda());
    EXPECT_EQ(mock->getDeviceForWeight("blk.1.ffn_gate.weight", 1).ordinal, 1);
}

TEST_F(Test__MockWeightPlacementMap, Builder_LayerRange)
{
    auto mock = MockWeightPlacementMap::Builder()
                    .withDefaultDevice(DeviceId::cpu())
                    .withLayerRange(0, 3, DeviceId::cuda(0))
                    .build();

    EXPECT_EQ(mock->layerMappingCount(), 4);
    for (int i = 0; i <= 3; ++i)
    {
        EXPECT_TRUE(mock->getDeviceForWeight("", i).is_cuda());
    }
    // Layer 4 should use default
    EXPECT_TRUE(mock->getDeviceForWeight("", 4).is_cpu());
}

TEST_F(Test__MockWeightPlacementMap, Builder_PatternMapping)
{
    auto mock = MockWeightPlacementMap::Builder()
                    .withDefaultDevice(DeviceId::cpu())
                    .withPatternDevice("attn", DeviceId::cuda(0))
                    .withPatternDevice("ffn", DeviceId::cuda(1))
                    .build();

    EXPECT_TRUE(mock->getDeviceForWeight("blk.5.attn_q.weight").is_cuda());
    EXPECT_EQ(mock->getDeviceForWeight("blk.5.attn_q.weight").ordinal, 0);
    EXPECT_TRUE(mock->getDeviceForWeight("blk.5.ffn_gate.weight").is_cuda());
    EXPECT_EQ(mock->getDeviceForWeight("blk.5.ffn_gate.weight").ordinal, 1);
}

TEST_F(Test__MockWeightPlacementMap, Builder_AttentionMapping)
{
    auto mock = MockWeightPlacementMap::Builder()
                    .withDefaultDevice(DeviceId::cpu())
                    .withAttentionDevice(0, DeviceId::cuda(0))
                    .build();

    EXPECT_TRUE(mock->getAttentionDevice(0).is_cuda());
    EXPECT_TRUE(mock->getAttentionDevice(1).is_cpu()); // Not mapped
}

TEST_F(Test__MockWeightPlacementMap, Builder_FFNMapping)
{
    auto mock = MockWeightPlacementMap::Builder()
                    .withDefaultDevice(DeviceId::cpu())
                    .withFFNDevice(2, DeviceId::cuda(0))
                    .build();

    EXPECT_TRUE(mock->getFFNDevice(2).is_cuda());
    EXPECT_TRUE(mock->getFFNDevice(0).is_cpu()); // Not mapped
}

TEST_F(Test__MockWeightPlacementMap, Builder_ChainingMultipleMappings)
{
    auto mock = MockWeightPlacementMap::Builder()
                    .withDefaultDevice(DeviceId::cpu())
                    .withTensorDevice("output.weight", DeviceId::cuda(0))
                    .withLayerDevice(0, DeviceId::cuda(1))
                    .withAttentionDevice(1, DeviceId::cuda(2))
                    .withFFNDevice(2, DeviceId::cuda(3))
                    .withStrategyName("complex_setup")
                    .build();

    EXPECT_EQ(mock->tensorMappingCount(), 1);
    EXPECT_EQ(mock->layerMappingCount(), 1);
    EXPECT_TRUE(mock->hasPlan());
}

TEST_F(Test__MockWeightPlacementMap, Builder_CallCountsResetAfterBuild)
{
    auto mock = MockWeightPlacementMap::Builder()
                    .withTensorDevice("test", DeviceId::cpu()) // This calls setTensorDevice
                    .withLayerDevice(0, DeviceId::cpu())       // This calls setLayerDevice
                    .build();

    // Call counts should be reset after build
    EXPECT_EQ(mock->setTensorDevice_call_count(), 0);
    EXPECT_EQ(mock->setLayerDevice_call_count(), 0);
}

// =============================================================================
// Preset Tests
// =============================================================================

TEST_F(Test__MockWeightPlacementMap, Preset_CPUOnly)
{
    auto mock = MockWeightPlacementMap::cpuOnlyPreset();

    EXPECT_TRUE(mock->defaultDevice().is_cpu());
    EXPECT_TRUE(mock->hasPlan());
    EXPECT_EQ(mock->appliedStrategyName(), "cpu_only");

    // All weights should be on CPU
    EXPECT_TRUE(mock->getDeviceForWeight("blk.0.attn_q.weight").is_cpu());
    EXPECT_TRUE(mock->getDeviceForWeight("embed_tokens.weight").is_cpu());
}

TEST_F(Test__MockWeightPlacementMap, Preset_SingleGPU)
{
    auto mock = MockWeightPlacementMap::singleGpuPreset();

    EXPECT_TRUE(mock->defaultDevice().is_cuda());
    EXPECT_EQ(mock->defaultDevice().ordinal, 0);
    EXPECT_EQ(mock->appliedStrategyName(), "single_gpu");
}

TEST_F(Test__MockWeightPlacementMap, Preset_MultiGPU)
{
    auto mock = MockWeightPlacementMap::multiGpuPreset(4);

    EXPECT_EQ(mock->appliedStrategyName(), "multi_gpu_even_split");

    // Even layers on GPU 0, odd layers on GPU 1
    EXPECT_EQ(mock->getDeviceForWeight("", 0).ordinal, 0);
    EXPECT_EQ(mock->getDeviceForWeight("", 1).ordinal, 1);
    EXPECT_EQ(mock->getDeviceForWeight("", 2).ordinal, 0);
    EXPECT_EQ(mock->getDeviceForWeight("", 3).ordinal, 1);
}

TEST_F(Test__MockWeightPlacementMap, Preset_CPUOffload)
{
    auto mock = MockWeightPlacementMap::cpuOffloadPreset(2, 8);

    EXPECT_EQ(mock->appliedStrategyName(), "cpu_offload");

    // First 2 layers on GPU
    EXPECT_TRUE(mock->getDeviceForWeight("", 0).is_cuda());
    EXPECT_TRUE(mock->getDeviceForWeight("", 1).is_cuda());

    // Rest on CPU
    EXPECT_TRUE(mock->getDeviceForWeight("", 2).is_cpu());
    EXPECT_TRUE(mock->getDeviceForWeight("", 7).is_cpu());
}

TEST_F(Test__MockWeightPlacementMap, Preset_Replicated)
{
    auto mock = MockWeightPlacementMap::replicatedPreset();
    EXPECT_EQ(mock->appliedStrategyName(), "replicated");
    EXPECT_TRUE(mock->defaultDevice().is_cpu());
}

TEST_F(Test__MockWeightPlacementMap, Preset_ColumnParallel)
{
    auto mock = MockWeightPlacementMap::columnParallelPreset();
    EXPECT_EQ(mock->appliedStrategyName(), "column_parallel");

    // QKV and gate/up are column parallel
    EXPECT_TRUE(mock->getDeviceForWeight("blk.0.attn_q.weight").is_cuda());
    EXPECT_TRUE(mock->getDeviceForWeight("blk.0.ffn_gate.weight").is_cuda());
}

TEST_F(Test__MockWeightPlacementMap, Preset_RowParallel)
{
    auto mock = MockWeightPlacementMap::rowParallelPreset();
    EXPECT_EQ(mock->appliedStrategyName(), "row_parallel");

    // Output and down are row parallel
    EXPECT_TRUE(mock->getDeviceForWeight("blk.0.attn_output.weight").is_cuda());
    EXPECT_TRUE(mock->getDeviceForWeight("blk.0.ffn_down.weight").is_cuda());
}

// =============================================================================
// Core Lookup Tests
// =============================================================================

TEST_F(Test__MockWeightPlacementMap, Lookup_DefaultDevice)
{
    EXPECT_TRUE(mock_cpu_->getDeviceForWeight("unknown_tensor").is_cpu());
    EXPECT_TRUE(mock_gpu_->getDeviceForWeight("unknown_tensor").is_cuda());
}

TEST_F(Test__MockWeightPlacementMap, Lookup_TensorPriority)
{
    // Tensor mapping should override layer and default
    mock_cpu_->setLayerDevice(0, DeviceId::cuda(1));
    mock_cpu_->setTensorDevice("blk.0.attn_q.weight", DeviceId::cuda(2));

    // Tensor mapping takes priority
    EXPECT_EQ(mock_cpu_->getDeviceForWeight("blk.0.attn_q.weight", 0).ordinal, 2);
}

TEST_F(Test__MockWeightPlacementMap, Lookup_LayerPriority)
{
    // Layer mapping should override default but not tensor
    mock_cpu_->setLayerDevice(0, DeviceId::cuda(1));

    EXPECT_EQ(mock_cpu_->getDeviceForWeight("blk.0.attn_q.weight", 0).ordinal, 1);
    // Unmapped layer should use default
    EXPECT_TRUE(mock_cpu_->getDeviceForWeight("blk.1.attn_q.weight", 1).is_cpu());
}

TEST_F(Test__MockWeightPlacementMap, Lookup_ExtractLayerFromTensorName)
{
    mock_cpu_->setLayerDevice(5, DeviceId::cuda(0));

    // Should extract layer 5 from tensor name even without explicit layer_idx
    EXPECT_TRUE(mock_cpu_->getDeviceForWeight("blk.5.attn_q.weight").is_cuda());
}

TEST_F(Test__MockWeightPlacementMap, Lookup_PatternMatching)
{
    mock_cpu_->setPatternDevice("embed", DeviceId::cuda(0));

    EXPECT_TRUE(mock_cpu_->getDeviceForWeight("embed_tokens.weight").is_cuda());
    EXPECT_TRUE(mock_cpu_->getDeviceForWeight("something_embed_something").is_cuda());
    EXPECT_TRUE(mock_cpu_->getDeviceForWeight("no_match").is_cpu());
}

TEST_F(Test__MockWeightPlacementMap, Lookup_DeviceInfo)
{
    mock_gpu_->setTensorDevice("test_tensor", DeviceId::cuda(1));

    WeightDeviceInfo info = mock_gpu_->getDeviceInfoForWeight("test_tensor");
    EXPECT_TRUE(info.prefill_device.is_cuda());
    EXPECT_EQ(info.prefill_device.ordinal, 1);
    EXPECT_FALSE(info.isDecodeDistributed());
}

// =============================================================================
// Block-Level Mapping Tests
// =============================================================================

TEST_F(Test__MockWeightPlacementMap, BlockLevel_AttentionDevice)
{
    mock_cpu_->setAttentionDevice(0, DeviceId::cuda(0));
    mock_cpu_->setAttentionDevice(1, DeviceId::cuda(1));

    EXPECT_TRUE(mock_cpu_->getAttentionDevice(0).is_cuda());
    EXPECT_EQ(mock_cpu_->getAttentionDevice(0).ordinal, 0);
    EXPECT_EQ(mock_cpu_->getAttentionDevice(1).ordinal, 1);
}

TEST_F(Test__MockWeightPlacementMap, BlockLevel_AttentionFallbackToLayer)
{
    mock_cpu_->setLayerDevice(0, DeviceId::cuda(0));

    // No explicit attention mapping, should fall back to layer
    EXPECT_TRUE(mock_cpu_->getAttentionDevice(0).is_cuda());
}

TEST_F(Test__MockWeightPlacementMap, BlockLevel_FFNDevice)
{
    mock_cpu_->setFFNDevice(0, DeviceId::cuda(0));
    mock_cpu_->setFFNDevice(1, DeviceId::cuda(1));

    EXPECT_TRUE(mock_cpu_->getFFNDevice(0).is_cuda());
    EXPECT_EQ(mock_cpu_->getFFNDevice(1).ordinal, 1);
}

// =============================================================================
// MoE Mapping Tests
// =============================================================================

TEST_F(Test__MockWeightPlacementMap, MoE_SharedExpertDevice)
{
    mock_cpu_->setSharedExpertDevice(0, DeviceId::cuda(0));
    mock_cpu_->setSharedExpertDevice(1, DeviceId::cuda(1));

    EXPECT_TRUE(mock_cpu_->getSharedExpertDevice(0).is_cuda());
    EXPECT_EQ(mock_cpu_->getSharedExpertDevice(1).ordinal, 1);

    // Unmapped expert should use default
    EXPECT_TRUE(mock_cpu_->getSharedExpertDevice(99).is_cpu());
}

TEST_F(Test__MockWeightPlacementMap, MoE_LocalExpertDevice)
{
    mock_cpu_->setLocalExpertDevice(0, 0, DeviceId::cuda(0));
    mock_cpu_->setLocalExpertDevice(0, 1, DeviceId::cuda(1));
    mock_cpu_->setLocalExpertDevice(1, 0, DeviceId::cuda(2));

    EXPECT_EQ(mock_cpu_->getLocalExpertDevice(0, 0).ordinal, 0);
    EXPECT_EQ(mock_cpu_->getLocalExpertDevice(0, 1).ordinal, 1);
    EXPECT_EQ(mock_cpu_->getLocalExpertDevice(1, 0).ordinal, 2);

    // Unmapped expert should use default
    EXPECT_TRUE(mock_cpu_->getLocalExpertDevice(99, 99).is_cpu());
}

TEST_F(Test__MockWeightPlacementMap, MoE_BuilderIntegration)
{
    auto mock = MockWeightPlacementMap::Builder()
                    .withDefaultDevice(DeviceId::cpu())
                    .withSharedExpertDevice(0, DeviceId::cuda(0))
                    .withLocalExpertDevice(0, 0, DeviceId::cuda(1))
                    .build();

    EXPECT_EQ(mock->getSharedExpertDevice(0).ordinal, 0);
    EXPECT_EQ(mock->getLocalExpertDevice(0, 0).ordinal, 1);
}

// =============================================================================
// Call Tracking Tests
// =============================================================================

TEST_F(Test__MockWeightPlacementMap, CallTracking_InitiallyZero)
{
    MockWeightPlacementMap mock;

    EXPECT_EQ(mock.getDeviceForWeight_call_count(), 0);
    EXPECT_EQ(mock.getDeviceInfoForWeight_call_count(), 0);
    EXPECT_EQ(mock.setTensorDevice_call_count(), 0);
    EXPECT_EQ(mock.setLayerDevice_call_count(), 0);
    EXPECT_EQ(mock.total_lookup_calls(), 0);
    EXPECT_EQ(mock.total_setter_calls(), 0);
}

TEST_F(Test__MockWeightPlacementMap, CallTracking_LookupCalls)
{
    mock_cpu_->getDeviceForWeight("tensor1");
    mock_cpu_->getDeviceForWeight("tensor2");
    mock_cpu_->getDeviceInfoForWeight("tensor3");

    // Note: getDeviceInfoForWeight internally calls getDeviceForWeight
    EXPECT_EQ(mock_cpu_->getDeviceForWeight_call_count(), 3); // 2 direct + 1 from getDeviceInfoForWeight
    EXPECT_EQ(mock_cpu_->getDeviceInfoForWeight_call_count(), 1);
    EXPECT_EQ(mock_cpu_->total_lookup_calls(), 4); // All lookup calls including internal ones
}

TEST_F(Test__MockWeightPlacementMap, CallTracking_SetterCalls)
{
    mock_cpu_->setTensorDevice("t1", DeviceId::cpu());
    mock_cpu_->setTensorDevice("t2", DeviceId::cpu());
    mock_cpu_->setLayerDevice(0, DeviceId::cpu());
    mock_cpu_->setLayerRange(1, 3, DeviceId::cpu());
    mock_cpu_->setPatternDevice("pattern", DeviceId::cpu());

    EXPECT_EQ(mock_cpu_->setTensorDevice_call_count(), 2);
    EXPECT_EQ(mock_cpu_->setLayerDevice_call_count(), 1);
    EXPECT_EQ(mock_cpu_->setLayerRange_call_count(), 1);
    EXPECT_EQ(mock_cpu_->setPatternDevice_call_count(), 1);
    EXPECT_EQ(mock_cpu_->total_setter_calls(), 5);
}

TEST_F(Test__MockWeightPlacementMap, CallTracking_BlockLevelCalls)
{
    mock_cpu_->setAttentionDevice(0, DeviceId::cpu());
    mock_cpu_->getAttentionDevice(0);
    mock_cpu_->getAttentionDevice(1);
    mock_cpu_->setFFNDevice(0, DeviceId::cpu());
    mock_cpu_->getFFNDevice(0);

    EXPECT_EQ(mock_cpu_->setAttentionDevice_call_count(), 1);
    EXPECT_EQ(mock_cpu_->getAttentionDevice_call_count(), 2);
    EXPECT_EQ(mock_cpu_->setFFNDevice_call_count(), 1);
    EXPECT_EQ(mock_cpu_->getFFNDevice_call_count(), 1);
}

TEST_F(Test__MockWeightPlacementMap, CallTracking_MoECalls)
{
    mock_cpu_->setSharedExpertDevice(0, DeviceId::cpu());
    mock_cpu_->getSharedExpertDevice(0);
    mock_cpu_->setLocalExpertDevice(0, 0, DeviceId::cpu());
    mock_cpu_->getLocalExpertDevice(0, 0);

    EXPECT_EQ(mock_cpu_->setSharedExpertDevice_call_count(), 1);
    EXPECT_EQ(mock_cpu_->getSharedExpertDevice_call_count(), 1);
    EXPECT_EQ(mock_cpu_->setLocalExpertDevice_call_count(), 1);
    EXPECT_EQ(mock_cpu_->getLocalExpertDevice_call_count(), 1);
}

TEST_F(Test__MockWeightPlacementMap, CallTracking_Reset)
{
    mock_cpu_->getDeviceForWeight("t1");
    mock_cpu_->setTensorDevice("t1", DeviceId::cpu());

    EXPECT_GT(mock_cpu_->total_lookup_calls(), 0);
    EXPECT_GT(mock_cpu_->total_setter_calls(), 0);

    mock_cpu_->reset_call_counts();

    EXPECT_EQ(mock_cpu_->total_lookup_calls(), 0);
    EXPECT_EQ(mock_cpu_->total_setter_calls(), 0);
}

// =============================================================================
// Plan Management Tests
// =============================================================================

TEST_F(Test__MockWeightPlacementMap, Plan_InitiallyNotApplied)
{
    EXPECT_FALSE(mock_cpu_->hasPlan());
    EXPECT_TRUE(mock_cpu_->appliedStrategyName().empty());
}

TEST_F(Test__MockWeightPlacementMap, Plan_ApplyPlan)
{
    // applyPlan doesn't actually parse the plan in mock - it just sets the flag
    PlacementPlan dummy_plan;
    mock_cpu_->applyPlan(dummy_plan);

    EXPECT_TRUE(mock_cpu_->hasPlan());
    EXPECT_EQ(mock_cpu_->applyPlan_call_count(), 1);
}

TEST_F(Test__MockWeightPlacementMap, Plan_Clear)
{
    mock_cpu_->setTensorDevice("t1", DeviceId::cuda(0));
    mock_cpu_->setLayerDevice(0, DeviceId::cuda(0));

    PlacementPlan dummy_plan;
    mock_cpu_->applyPlan(dummy_plan);

    mock_cpu_->clear();

    EXPECT_EQ(mock_cpu_->tensorMappingCount(), 0);
    EXPECT_EQ(mock_cpu_->layerMappingCount(), 0);
    EXPECT_FALSE(mock_cpu_->hasPlan());
    EXPECT_EQ(mock_cpu_->clear_call_count(), 1);
}

// =============================================================================
// Failure Injection Tests
// =============================================================================

TEST_F(Test__MockWeightPlacementMap, FailureInjection_LookupFails)
{
    MockWeightPlacementMap::Config config;
    config.lookup_should_fail = true;
    config.failure_message = "Test failure";
    MockWeightPlacementMap mock(config);

    EXPECT_THROW(mock.getDeviceForWeight("any_tensor"), std::runtime_error);
}

TEST_F(Test__MockWeightPlacementMap, FailureInjection_DeviceInfoFails)
{
    MockWeightPlacementMap::Config config;
    config.lookup_should_fail = true;
    MockWeightPlacementMap mock(config);

    EXPECT_THROW(mock.getDeviceInfoForWeight("any_tensor"), std::runtime_error);
}

TEST_F(Test__MockWeightPlacementMap, FailureInjection_RuntimeToggle)
{
    mock_cpu_->set_lookup_fails(true);
    EXPECT_THROW(mock_cpu_->getDeviceForWeight("tensor"), std::runtime_error);

    mock_cpu_->set_lookup_fails(false);
    EXPECT_NO_THROW(mock_cpu_->getDeviceForWeight("tensor"));
}

TEST_F(Test__MockWeightPlacementMap, FailureInjection_CustomMessage)
{
    mock_cpu_->set_lookup_fails(true);
    mock_cpu_->set_failure_message("Custom error message");

    try
    {
        mock_cpu_->getDeviceForWeight("tensor");
        FAIL() << "Expected exception";
    }
    catch (const std::runtime_error &e)
    {
        EXPECT_EQ(std::string(e.what()), "Custom error message");
    }
}

TEST_F(Test__MockWeightPlacementMap, FailureInjection_Builder)
{
    auto mock = MockWeightPlacementMap::Builder()
                    .withFailure(true, "Builder failure")
                    .build();

    EXPECT_THROW(mock->getDeviceForWeight("tensor"), std::runtime_error);
}

// =============================================================================
// Configuration Access Tests
// =============================================================================

TEST_F(Test__MockWeightPlacementMap, Config_ReadOnly)
{
    const auto &config = mock_cpu_->config();
    EXPECT_TRUE(config.default_device.is_cpu());
    EXPECT_TRUE(config.track_calls);
    EXPECT_FALSE(config.lookup_should_fail);
}

TEST_F(Test__MockWeightPlacementMap, Config_RuntimeModification)
{
    mock_cpu_->set_default_device(DeviceId::cuda(5));
    EXPECT_EQ(mock_cpu_->defaultDevice().ordinal, 5);
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST_F(Test__MockWeightPlacementMap, EdgeCase_EmptyTensorName)
{
    // Empty tensor name should return default device
    EXPECT_TRUE(mock_cpu_->getDeviceForWeight("").is_cpu());
}

TEST_F(Test__MockWeightPlacementMap, EdgeCase_NegativeLayerIndex)
{
    mock_cpu_->setLayerDevice(0, DeviceId::cuda(0));

    // Negative layer index should not match any layer mapping
    // and should try to extract from tensor name
    EXPECT_TRUE(mock_cpu_->getDeviceForWeight("no_layer_info", -1).is_cpu());
}

TEST_F(Test__MockWeightPlacementMap, EdgeCase_VeryLargeLayerIndex)
{
    // Large layer index should return default device
    EXPECT_TRUE(mock_cpu_->getDeviceForWeight("", 999999).is_cpu());
}

TEST_F(Test__MockWeightPlacementMap, EdgeCase_SpecialCharactersInTensorName)
{
    mock_cpu_->setTensorDevice("tensor.with.dots.and-dashes_and_underscores", DeviceId::cuda(0));
    EXPECT_TRUE(mock_cpu_->getDeviceForWeight("tensor.with.dots.and-dashes_and_underscores").is_cuda());
}

// =============================================================================
// Thread Safety Test (basic)
// =============================================================================

TEST_F(Test__MockWeightPlacementMap, ThreadSafety_ConcurrentLookups)
{
    // Simple test to ensure atomic counters work
    std::vector<std::thread> threads;
    const int num_threads = 4;
    const int lookups_per_thread = 100;

    for (int i = 0; i < num_threads; ++i)
    {
        threads.emplace_back([this, lookups_per_thread]()
                             {
            for (int j = 0; j < lookups_per_thread; ++j) {
                mock_cpu_->getDeviceForWeight("test_tensor");
            } });
    }

    for (auto &t : threads)
    {
        t.join();
    }

    EXPECT_EQ(mock_cpu_->getDeviceForWeight_call_count(),
              static_cast<size_t>(num_threads * lookups_per_thread));
}
