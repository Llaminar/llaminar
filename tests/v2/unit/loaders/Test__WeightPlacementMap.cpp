/**
 * @file Test__WeightPlacementMap.cpp
 * @brief Unit tests for WeightPlacementMap (Tensor-Level Device Placement - Phase 2)
 *
 * Tests block-level and MoE-specific device placement methods added for MoE readiness.
 */

#include "../../../../src/v2/loaders/WeightPlacementMap.h"
#include "../../../../src/v2/backends/DeviceId.h"
#include "../../../../src/v2/execution/PlacementPlan.h"
#include <gtest/gtest.h>
#include <memory>

using namespace llaminar2;

class WeightPlacementMapTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        map_ = std::make_shared<WeightPlacementMap>(DeviceId::cpu()); // CPU default
    }

    std::shared_ptr<WeightPlacementMap> map_;
};

// ========== Block-Level Placement Tests ==========

TEST_F(WeightPlacementMapTest, AttentionDevicePlacement)
{
    // Set attention block for layer 0 to GPU (device 0)
    map_->setAttentionDevice(0, DeviceId::cuda(0));

    // Verify attention tensors are on GPU
    EXPECT_EQ(map_->getDeviceForWeight("blk.0.attn_q.weight", 0), DeviceId::cuda(0));
    EXPECT_EQ(map_->getDeviceForWeight("blk.0.attn_k.weight", 0), DeviceId::cuda(0));
    EXPECT_EQ(map_->getDeviceForWeight("blk.0.attn_v.weight", 0), DeviceId::cuda(0));
    EXPECT_EQ(map_->getDeviceForWeight("blk.0.attn_output.weight", 0), DeviceId::cuda(0));
    EXPECT_EQ(map_->getDeviceForWeight("blk.0.attn_norm.weight", 0), DeviceId::cuda(0));

    // Verify getter returns correct device
    EXPECT_EQ(map_->getAttentionDevice(0), DeviceId::cuda(0));

    // FFN should still be on CPU (default)
    EXPECT_EQ(map_->getFFNDevice(0), DeviceId::cpu());
}

TEST_F(WeightPlacementMapTest, FFNDevicePlacement)
{
    // Set FFN block for layer 1 to GPU (device 0)
    map_->setFFNDevice(1, DeviceId::cuda(0));

    // Verify FFN tensors are on GPU
    EXPECT_EQ(map_->getDeviceForWeight("blk.1.ffn_gate.weight", 1), DeviceId::cuda(0));
    EXPECT_EQ(map_->getDeviceForWeight("blk.1.ffn_up.weight", 1), DeviceId::cuda(0));
    EXPECT_EQ(map_->getDeviceForWeight("blk.1.ffn_down.weight", 1), DeviceId::cuda(0));
    EXPECT_EQ(map_->getDeviceForWeight("blk.1.ffn_norm.weight", 1), DeviceId::cuda(0));

    // Verify getter returns correct device
    EXPECT_EQ(map_->getFFNDevice(1), DeviceId::cuda(0));

    // Attention should still be on CPU (default)
    EXPECT_EQ(map_->getAttentionDevice(1), DeviceId::cpu());
}

TEST_F(WeightPlacementMapTest, MixedAttentionFFNPlacement)
{
    // Layer 0: Attention on CPU, FFN on GPU
    map_->setAttentionDevice(0, DeviceId::cpu());
    map_->setFFNDevice(0, DeviceId::cuda(0));

    EXPECT_EQ(map_->getAttentionDevice(0), DeviceId::cpu());
    EXPECT_EQ(map_->getFFNDevice(0), DeviceId::cuda(0));

    // Layer 1: Attention on GPU, FFN on CPU
    map_->setAttentionDevice(1, DeviceId::cuda(0));
    map_->setFFNDevice(1, DeviceId::cpu());

    EXPECT_EQ(map_->getAttentionDevice(1), DeviceId::cuda(0));
    EXPECT_EQ(map_->getFFNDevice(1), DeviceId::cpu());
}

// ========== MoE-Specific Tests ==========

TEST_F(WeightPlacementMapTest, SharedExpertPlacement)
{
    // Place shared expert 0 on GPU 0
    map_->setSharedExpertDevice(0, DeviceId::cuda(0));
    EXPECT_EQ(map_->getSharedExpertDevice(0), DeviceId::cuda(0));

    // Verify pattern matching works for expert tensors
    EXPECT_EQ(map_->getDeviceForWeight("shared_expert.0.gate.weight", -1), DeviceId::cuda(0));
    EXPECT_EQ(map_->getDeviceForWeight("shared_expert.0.up.weight", -1), DeviceId::cuda(0));

    // Place shared expert 1 on GPU 1
    map_->setSharedExpertDevice(1, DeviceId::cuda(1));
    EXPECT_EQ(map_->getSharedExpertDevice(1), DeviceId::cuda(1));

    // Unset expert defaults to CPU
    EXPECT_EQ(map_->getSharedExpertDevice(99), DeviceId::cpu());
}

TEST_F(WeightPlacementMapTest, LocalExpertPlacement)
{
    // Layer 0, Expert 0 on GPU 0
    map_->setLocalExpertDevice(0, 0, DeviceId::cuda(0));
    EXPECT_EQ(map_->getLocalExpertDevice(0, 0), DeviceId::cuda(0));

    // Verify pattern matching
    EXPECT_EQ(map_->getDeviceForWeight("blk.0.expert.0.gate.weight", 0), DeviceId::cuda(0));

    // Layer 0, Expert 1 on GPU 1
    map_->setLocalExpertDevice(0, 1, DeviceId::cuda(1));
    EXPECT_EQ(map_->getLocalExpertDevice(0, 1), DeviceId::cuda(1));

    // Different layer, same expert index
    map_->setLocalExpertDevice(1, 0, DeviceId::cuda(1));
    EXPECT_EQ(map_->getLocalExpertDevice(1, 0), DeviceId::cuda(1));

    // Unset expert defaults to CPU
    EXPECT_EQ(map_->getLocalExpertDevice(5, 5), DeviceId::cpu());
}

TEST_F(WeightPlacementMapTest, MoEHeterogeneousPlacement)
{
    // Realistic MoE scenario:
    // - Attention: CPU (moderate size)
    // - Local FFN: CPU (moderate size)
    // - Shared Experts: GPU (large, reused)

    int n_layers = 24;

    // All attention on CPU
    for (int i = 0; i < n_layers; ++i)
    {
        map_->setAttentionDevice(i, DeviceId::cpu());
    }

    // All local FFN on CPU
    for (int i = 0; i < n_layers; ++i)
    {
        map_->setFFNDevice(i, DeviceId::cpu());
    }

    // Shared experts on GPU
    int n_experts = 8;
    for (int i = 0; i < n_experts; ++i)
    {
        map_->setSharedExpertDevice(i, DeviceId::cuda(0)); // GPU 0
    }

    // Verify placements
    EXPECT_EQ(map_->getAttentionDevice(0), DeviceId::cpu());
    EXPECT_EQ(map_->getFFNDevice(0), DeviceId::cpu());
    EXPECT_EQ(map_->getSharedExpertDevice(0), DeviceId::cuda(0));
    EXPECT_EQ(map_->getSharedExpertDevice(7), DeviceId::cuda(0));
}

// ========== Existing API Compatibility Tests ==========

TEST_F(WeightPlacementMapTest, LayerRangePlacement)
{
    // Set layers 0-11 to CPU, 12-23 to GPU
    map_->setLayerRange(0, 12, DeviceId::cpu());
    map_->setLayerRange(12, 24, DeviceId::cuda(0));

    EXPECT_EQ(map_->getAttentionDevice(5), DeviceId::cpu());
    EXPECT_EQ(map_->getAttentionDevice(15), DeviceId::cuda(0));
}

TEST_F(WeightPlacementMapTest, PatternBasedPlacement)
{
    // Set all embedding weights to GPU
    map_->setPatternDevice("*embedding*", DeviceId::cuda(0));

    EXPECT_EQ(map_->getDeviceForWeight("token_embedding.weight", -1), DeviceId::cuda(0));
    EXPECT_EQ(map_->getDeviceForWeight("position_embedding.weight", -1), DeviceId::cuda(0));

    // Attention weights still on default CPU
    EXPECT_EQ(map_->getDeviceForWeight("blk.0.attn_q.weight", 0), DeviceId::cpu());
}

TEST_F(WeightPlacementMapTest, ClearResetsAllMaps)
{
    map_->setAttentionDevice(0, DeviceId::cuda(0));
    map_->setFFNDevice(1, DeviceId::cuda(0));
    map_->setSharedExpertDevice(0, DeviceId::cuda(0));
    map_->setLocalExpertDevice(0, 0, DeviceId::cuda(0));

    map_->clear();

    // All should return default
    EXPECT_EQ(map_->getAttentionDevice(0), DeviceId::cpu());
    EXPECT_EQ(map_->getFFNDevice(1), DeviceId::cpu());
    EXPECT_EQ(map_->getSharedExpertDevice(0), DeviceId::cpu());
    EXPECT_EQ(map_->getLocalExpertDevice(0, 0), DeviceId::cpu());
}

// ========== WeightDeviceInfo and Phase-Aware Placement Tests ==========

TEST_F(WeightPlacementMapTest, WeightDeviceInfoDefaultConstruction)
{
    WeightDeviceInfo info;

    EXPECT_EQ(info.prefill_device, DeviceId::cpu());
    EXPECT_EQ(info.decode_devices.size(), 1);
    EXPECT_EQ(info.decode_devices[0], DeviceId::cpu());
    EXPECT_EQ(info.decode_fractions.size(), 1);
    EXPECT_FLOAT_EQ(info.decode_fractions[0], 1.0f);
    EXPECT_FALSE(info.cpu_decode_participation);
    EXPECT_FALSE(info.isDecodeDistributed());
    EXPECT_EQ(info.decodeDeviceCount(), 1);
}

TEST_F(WeightPlacementMapTest, WeightDeviceInfoSingleDeviceConstruction)
{
    // CPU device
    WeightDeviceInfo cpu_info(DeviceId::cpu());
    EXPECT_EQ(cpu_info.prefill_device, DeviceId::cpu());
    EXPECT_TRUE(cpu_info.cpu_decode_participation);
    EXPECT_FALSE(cpu_info.isDecodeDistributed());

    // GPU device
    WeightDeviceInfo gpu_info(DeviceId::cuda(0));
    EXPECT_EQ(gpu_info.prefill_device, DeviceId::cuda(0));
    EXPECT_FALSE(gpu_info.cpu_decode_participation);
    EXPECT_FALSE(gpu_info.isDecodeDistributed());
}

TEST_F(WeightPlacementMapTest, GetDeviceInfoForWeightBasic)
{
    // Set up basic layer placement
    map_->setLayerDevice(0, DeviceId::cuda(0));

    // Without decode info, should return single-device info matching prefill
    auto info = map_->getDeviceInfoForWeight("blk.0.attn_q.weight", 0);

    EXPECT_EQ(info.prefill_device, DeviceId::cuda(0));
    EXPECT_EQ(info.decode_devices.size(), 1);
    EXPECT_EQ(info.decode_devices[0], DeviceId::cuda(0));
    EXPECT_FALSE(info.cpu_decode_participation);
}

TEST_F(WeightPlacementMapTest, GetDeviceInfoForWeightExtractsLayerIndex)
{
    map_->setLayerDevice(5, DeviceId::cuda(1));

    // Don't provide layer_idx - should extract from tensor name
    auto info = map_->getDeviceInfoForWeight("blk.5.ffn_gate.weight");

    EXPECT_EQ(info.prefill_device, DeviceId::cuda(1));
}

TEST_F(WeightPlacementMapTest, GetDeviceInfoForWeightFallbackToDefault)
{
    // No explicit mappings - should return default device info
    auto info = map_->getDeviceInfoForWeight("some_random_tensor.weight");

    EXPECT_EQ(info.prefill_device, DeviceId::cpu()); // Default is CPU
    EXPECT_EQ(info.decode_devices.size(), 1);
    EXPECT_EQ(info.decode_devices[0], DeviceId::cpu());
}

// ========== PlacementPlan Integration with Decode Placement ==========

TEST_F(WeightPlacementMapTest, ApplyPlanWithDecodeDevices)
{
    // Create a plan with decode device info (simulating HybridOptimalPlacementStrategy output)
    PlacementPlan plan;
    plan.n_layers = 2;
    plan.world_size = 1;
    plan.strategy_name = "TestHybridStrategy";

    // Layer 0: GPU prefill, GPU+CPU decode (80%/20%)
    LayerPlacement layer0;
    layer0.layer_idx = 0;
    layer0.device = PlacementDevice::gpu(0);
    layer0.decode_devices = {PlacementDevice::gpu(0), PlacementDevice::cpu()};
    layer0.decode_weight_fractions = {0.8f, 0.2f};
    layer0.cpu_participates_in_decode = true;

    // Layer 1: GPU prefill, GPU only decode
    LayerPlacement layer1;
    layer1.layer_idx = 1;
    layer1.device = PlacementDevice::gpu(0);
    layer1.decode_devices = {PlacementDevice::gpu(0)};
    layer1.decode_weight_fractions = {1.0f};
    layer1.cpu_participates_in_decode = false;

    plan.layers = {layer0, layer1};

    map_->applyPlan(plan);

    // Verify layer 0 decode info
    auto info0 = map_->getDeviceInfoForWeight("blk.0.attn_q.weight", 0);
    EXPECT_EQ(info0.prefill_device, DeviceId::cuda(0));
    EXPECT_EQ(info0.decode_devices.size(), 2);
    EXPECT_EQ(info0.decode_devices[0], DeviceId::cuda(0));
    EXPECT_EQ(info0.decode_devices[1], DeviceId::cpu());
    EXPECT_FLOAT_EQ(info0.decode_fractions[0], 0.8f);
    EXPECT_FLOAT_EQ(info0.decode_fractions[1], 0.2f);
    EXPECT_TRUE(info0.cpu_decode_participation);
    EXPECT_TRUE(info0.isDecodeDistributed());

    // Verify layer 1 decode info
    auto info1 = map_->getDeviceInfoForWeight("blk.1.ffn_gate.weight", 1);
    EXPECT_EQ(info1.prefill_device, DeviceId::cuda(0));
    EXPECT_EQ(info1.decode_devices.size(), 1);
    EXPECT_EQ(info1.decode_devices[0], DeviceId::cuda(0));
    EXPECT_FALSE(info1.cpu_decode_participation);
    EXPECT_FALSE(info1.isDecodeDistributed());
}

TEST_F(WeightPlacementMapTest, ApplyPlanWithoutDecodeDevicesFallsBackToPrefill)
{
    // Create a plan WITHOUT decode device info (legacy strategy)
    PlacementPlan plan;
    plan.n_layers = 1;
    plan.world_size = 1;
    plan.strategy_name = "LegacyStrategy";

    LayerPlacement layer0;
    layer0.layer_idx = 0;
    layer0.device = PlacementDevice::gpu(0);
    // decode_devices left empty - should fallback to prefill device

    plan.layers = {layer0};

    map_->applyPlan(plan);

    auto info = map_->getDeviceInfoForWeight("blk.0.attn_q.weight", 0);
    EXPECT_EQ(info.prefill_device, DeviceId::cuda(0));
    EXPECT_EQ(info.decode_devices.size(), 1);
    EXPECT_EQ(info.decode_devices[0], DeviceId::cuda(0));
    EXPECT_FALSE(info.cpu_decode_participation);
}

TEST_F(WeightPlacementMapTest, ApplyPlanWithSplitAttentionFFN)
{
    // Plan with split attention/FFN placement
    PlacementPlan plan;
    plan.n_layers = 1;
    plan.world_size = 1;
    plan.strategy_name = "SplitStrategy";

    LayerPlacement layer0;
    layer0.layer_idx = 0;
    layer0.split_attention_ffn = true;
    layer0.attention_device = PlacementDevice::gpu(0);
    layer0.ffn_device = PlacementDevice::cpu();
    layer0.decode_devices = {PlacementDevice::gpu(0), PlacementDevice::cpu()};
    layer0.decode_weight_fractions = {0.7f, 0.3f};
    layer0.cpu_participates_in_decode = true;

    plan.layers = {layer0};

    map_->applyPlan(plan);

    // Attention should be on GPU (prefill)
    auto attn_info = map_->getDeviceInfoForWeight("blk.0.attn_q.weight", 0);
    // Note: prefill device comes from getDeviceForWeight which uses tensor name
    // Since we called setAttentionDevice, attn tensors should be on GPU
    EXPECT_EQ(map_->getAttentionDevice(0), DeviceId::cuda(0));

    // FFN should be on CPU (prefill)
    EXPECT_EQ(map_->getFFNDevice(0), DeviceId::cpu());

    // But decode info should still show distributed placement
    EXPECT_TRUE(attn_info.cpu_decode_participation);
    EXPECT_EQ(attn_info.decode_devices.size(), 2);
}

TEST_F(WeightPlacementMapTest, ClearResetsDecodeInfo)
{
    // Apply a plan with decode info
    PlacementPlan plan;
    plan.n_layers = 1;
    plan.world_size = 1;
    plan.strategy_name = "TestStrategy";

    LayerPlacement layer0;
    layer0.layer_idx = 0;
    layer0.device = PlacementDevice::gpu(0);
    layer0.decode_devices = {PlacementDevice::gpu(0), PlacementDevice::cpu()};
    layer0.decode_weight_fractions = {0.8f, 0.2f};
    layer0.cpu_participates_in_decode = true;

    plan.layers = {layer0};
    map_->applyPlan(plan);

    // Verify decode info exists
    auto info_before = map_->getDeviceInfoForWeight("blk.0.attn_q.weight", 0);
    EXPECT_TRUE(info_before.cpu_decode_participation);

    // Clear and verify decode info is gone
    map_->clear();

    // After clear, should get default info (single CPU device)
    auto info_after = map_->getDeviceInfoForWeight("blk.0.attn_q.weight", 0);
    EXPECT_EQ(info_after.prefill_device, DeviceId::cpu());
    EXPECT_EQ(info_after.decode_devices.size(), 1);
    EXPECT_FALSE(info_after.isDecodeDistributed());
}

TEST_F(WeightPlacementMapTest, BackwardCompatibilityWithGetDeviceForWeight)
{
    // Ensure getDeviceForWeight still works as before
    PlacementPlan plan;
    plan.n_layers = 2;
    plan.world_size = 1;
    plan.strategy_name = "TestStrategy";

    LayerPlacement layer0;
    layer0.layer_idx = 0;
    layer0.device = PlacementDevice::gpu(0);
    layer0.decode_devices = {PlacementDevice::gpu(0), PlacementDevice::cpu()};
    layer0.decode_weight_fractions = {0.8f, 0.2f};
    layer0.cpu_participates_in_decode = true;

    LayerPlacement layer1;
    layer1.layer_idx = 1;
    layer1.device = PlacementDevice::cpu();

    plan.layers = {layer0, layer1};
    map_->applyPlan(plan);

    // getDeviceForWeight should return prefill device (existing behavior unchanged)
    EXPECT_EQ(map_->getDeviceForWeight("blk.0.attn_q.weight", 0), DeviceId::cuda(0));
    EXPECT_EQ(map_->getDeviceForWeight("blk.1.attn_q.weight", 1), DeviceId::cpu());
}

// ============================================================================
// DeviceId Ordinal Tests - Regression tests for device ordinal propagation bug
// ============================================================================

/**
 * @brief Test that DeviceId correctly stores and reports ROCm ordinal
 *
 * Regression test for bug where WeightPreloader didn't propagate device ordinal.
 */
TEST_F(WeightPlacementMapTest, DeviceId_ROCmOrdinalStorage)
{
    DeviceId dev0 = DeviceId::rocm(0);
    DeviceId dev1 = DeviceId::rocm(1);
    DeviceId dev7 = DeviceId::rocm(7);

    EXPECT_TRUE(dev0.is_rocm());
    EXPECT_TRUE(dev1.is_rocm());
    EXPECT_TRUE(dev7.is_rocm());

    EXPECT_EQ(dev0.ordinal, 0);
    EXPECT_EQ(dev1.ordinal, 1);
    EXPECT_EQ(dev7.ordinal, 7);

    EXPECT_EQ(dev0.type, DeviceType::ROCm);
    EXPECT_EQ(dev1.type, DeviceType::ROCm);
    EXPECT_EQ(dev7.type, DeviceType::ROCm);
}

/**
 * @brief Test that DeviceId::cpu() has ordinal 0
 * (CPU uses ordinal 0, GPU devices use ordinal 0, 1, 2, ...)
 */
TEST_F(WeightPlacementMapTest, DeviceId_CPUHasOrdinalZero)
{
    DeviceId cpu = DeviceId::cpu();

    EXPECT_TRUE(cpu.is_cpu());
    EXPECT_FALSE(cpu.is_rocm());
    EXPECT_EQ(cpu.ordinal, 0);
    EXPECT_EQ(cpu.type, DeviceType::CPU);
}

/**
 * @brief Test that DeviceId equality considers ordinal
 *
 * Two DeviceIds with same type but different ordinal should NOT be equal.
 */
TEST_F(WeightPlacementMapTest, DeviceId_EqualityIncludesOrdinal)
{
    DeviceId rocm0 = DeviceId::rocm(0);
    DeviceId rocm1 = DeviceId::rocm(1);
    DeviceId rocm0_copy = DeviceId::rocm(0);

    // Same type, same ordinal -> equal
    EXPECT_EQ(rocm0, rocm0_copy);

    // Same type, different ordinal -> not equal
    EXPECT_NE(rocm0, rocm1);

    // Different type -> not equal
    DeviceId cpu = DeviceId::cpu();
    EXPECT_NE(rocm0, cpu);
}

/**
 * @brief Test placement map preserves ROCm device ordinal
 *
 * When setting a ROCm device with ordinal 1, getDeviceForWeight should return
 * a DeviceId with ordinal 1, not ordinal 0.
 */
TEST_F(WeightPlacementMapTest, PlacementMap_PreservesROCmOrdinal)
{
    // Set up placement map with ROCm device 1
    map_->setTensorDevice("blk.0.attn_q.weight", DeviceId::rocm(1));
    map_->setTensorDevice("blk.0.attn_k.weight", DeviceId::rocm(1));

    // Verify ordinal is preserved
    DeviceId device = map_->getDeviceForWeight("blk.0.attn_q.weight", 0);
    EXPECT_TRUE(device.is_rocm());
    EXPECT_EQ(device.ordinal, 1) << "ROCm ordinal should be preserved through placement map";
}

/**
 * @brief Test placement map with multiple GPU ordinals
 *
 * Simulates multi-GPU setup where different layers are on different GPUs.
 */
TEST_F(WeightPlacementMapTest, PlacementMap_MultipleGPUOrdinals)
{
    // Distribute weights across two GPUs
    map_->setTensorDevice("blk.0.attn_q.weight", DeviceId::rocm(0));
    map_->setTensorDevice("blk.1.attn_q.weight", DeviceId::rocm(1));

    DeviceId dev0 = map_->getDeviceForWeight("blk.0.attn_q.weight", 0);
    DeviceId dev1 = map_->getDeviceForWeight("blk.1.attn_q.weight", 1);

    EXPECT_EQ(dev0.ordinal, 0);
    EXPECT_EQ(dev1.ordinal, 1);
}

/**
 * @brief Test DeviceId::to_string() includes ordinal
 */
TEST_F(WeightPlacementMapTest, DeviceId_ToStringIncludesOrdinal)
{
    DeviceId rocm0 = DeviceId::rocm(0);
    DeviceId rocm1 = DeviceId::rocm(1);

    std::string str0 = rocm0.to_string();
    std::string str1 = rocm1.to_string();

    // Strings should be different (include ordinal)
    EXPECT_NE(str0, str1);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
