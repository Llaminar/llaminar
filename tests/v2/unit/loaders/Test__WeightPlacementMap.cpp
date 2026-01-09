/**
 * @file Test__WeightPlacementMap.cpp
 * @brief Unit tests for WeightPlacementMap (Tensor-Level Device Placement - Phase 2)
 *
 * Tests block-level and MoE-specific device placement methods added for MoE readiness.
 */

#include "../../../../src/v2/loaders/WeightPlacementMap.h"
#include "../../../../src/v2/backends/DeviceId.h"
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

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
