/**
 * @file Test__WeightPlacementMap.cpp
 * @brief Tests for WeightPlacementMap
 * @author David Sanftenberg
 * @date 2025-10-24
 */

#include <gtest/gtest.h>
#include "loaders/WeightPlacementMap.h"

using namespace llaminar2;

class Test__WeightPlacementMap : public ::testing::Test {
protected:
    void SetUp() override {
        // Default map uses device 0
        default_map_ = std::make_shared<WeightPlacementMap>(0);
    }

    std::shared_ptr<WeightPlacementMap> default_map_;
};

/**
 * Test: Default device fallback
 */
TEST_F(Test__WeightPlacementMap, DefaultDevice) {
    EXPECT_EQ(0, default_map_->defaultDevice());
    EXPECT_EQ(0, default_map_->getDeviceForWeight("any.tensor.name"));
    EXPECT_EQ(0, default_map_->getDeviceForWeight("blk.0.attn_q.weight"));
}

/**
 * Test: Explicit tensor mapping
 */
TEST_F(Test__WeightPlacementMap, ExplicitTensorMapping) {
    default_map_->setTensorDevice("blk.0.attn_q.weight", 1);
    default_map_->setTensorDevice("blk.0.attn_k.weight", 2);
    
    EXPECT_EQ(1, default_map_->getDeviceForWeight("blk.0.attn_q.weight"));
    EXPECT_EQ(2, default_map_->getDeviceForWeight("blk.0.attn_k.weight"));
    EXPECT_EQ(0, default_map_->getDeviceForWeight("blk.0.attn_v.weight")); // Not mapped, use default
    
    EXPECT_EQ(2, default_map_->tensorMappingCount());
}

/**
 * Test: Layer-based mapping
 */
TEST_F(Test__WeightPlacementMap, LayerMapping) {
    default_map_->setLayerDevice(0, 1);
    default_map_->setLayerDevice(1, 1);
    default_map_->setLayerDevice(2, 2);
    
    // With explicit layer_idx
    EXPECT_EQ(1, default_map_->getDeviceForWeight("blk.0.attn_q.weight", 0));
    EXPECT_EQ(1, default_map_->getDeviceForWeight("blk.1.attn_q.weight", 1));
    EXPECT_EQ(2, default_map_->getDeviceForWeight("blk.2.attn_q.weight", 2));
    
    EXPECT_EQ(3, default_map_->layerMappingCount());
}

/**
 * Test: Layer extraction from tensor name
 */
TEST_F(Test__WeightPlacementMap, LayerExtraction) {
    default_map_->setLayerDevice(0, 1);
    default_map_->setLayerDevice(5, 2);
    default_map_->setLayerDevice(12, 3);
    
    // Should extract layer index from name
    EXPECT_EQ(1, default_map_->getDeviceForWeight("blk.0.attn_q.weight"));
    EXPECT_EQ(1, default_map_->getDeviceForWeight("blk.0.ffn_gate.weight"));
    EXPECT_EQ(2, default_map_->getDeviceForWeight("blk.5.attn_k.weight"));
    EXPECT_EQ(3, default_map_->getDeviceForWeight("blk.12.mlp.down.weight"));
    
    // Non-layer tensors should use default
    EXPECT_EQ(0, default_map_->getDeviceForWeight("token_embd.weight"));
    EXPECT_EQ(0, default_map_->getDeviceForWeight("output.weight"));
}

/**
 * Test: Layer range mapping
 */
TEST_F(Test__WeightPlacementMap, LayerRange) {
    default_map_->setLayerRange(0, 3, 1);   // Layers 0-3 → GPU
    default_map_->setLayerRange(4, 7, 0);   // Layers 4-7 → CPU
    
    EXPECT_EQ(1, default_map_->getDeviceForWeight("blk.0.attn_q.weight"));
    EXPECT_EQ(1, default_map_->getDeviceForWeight("blk.3.attn_q.weight"));
    EXPECT_EQ(0, default_map_->getDeviceForWeight("blk.4.attn_q.weight"));
    EXPECT_EQ(0, default_map_->getDeviceForWeight("blk.7.attn_q.weight"));
}

/**
 * Test: Pattern-based mapping (prefix wildcard)
 */
TEST_F(Test__WeightPlacementMap, PatternPrefix) {
    default_map_->setPatternDevice("blk.0.*", 1);
    
    EXPECT_EQ(1, default_map_->getDeviceForWeight("blk.0.attn_q.weight"));
    EXPECT_EQ(1, default_map_->getDeviceForWeight("blk.0.ffn_gate.weight"));
    EXPECT_EQ(0, default_map_->getDeviceForWeight("blk.1.attn_q.weight")); // Different layer
}

/**
 * Test: Pattern-based mapping (suffix wildcard)
 */
TEST_F(Test__WeightPlacementMap, PatternSuffix) {
    default_map_->setPatternDevice("*embd*", 1);
    
    EXPECT_EQ(1, default_map_->getDeviceForWeight("token_embd.weight"));
    EXPECT_EQ(0, default_map_->getDeviceForWeight("output.weight"));
}

/**
 * Test: Pattern-based mapping (contains wildcard)
 */
TEST_F(Test__WeightPlacementMap, PatternContains) {
    default_map_->setPatternDevice("*attn*", 1);
    
    EXPECT_EQ(1, default_map_->getDeviceForWeight("blk.0.attn_q.weight"));
    EXPECT_EQ(1, default_map_->getDeviceForWeight("blk.5.attn_k.weight"));
    EXPECT_EQ(1, default_map_->getDeviceForWeight("model.layers.0.self_attn.q_proj"));
    EXPECT_EQ(0, default_map_->getDeviceForWeight("blk.0.ffn_gate.weight"));
}

/**
 * Test: Priority ordering (explicit > layer > pattern > default)
 */
TEST_F(Test__WeightPlacementMap, PriorityOrdering) {
    // Set up conflicting rules
    default_map_->setPatternDevice("*attn*", 1);         // Pattern: device 1
    default_map_->setLayerDevice(0, 2);                  // Layer 0: device 2
    default_map_->setTensorDevice("blk.0.attn_q.weight", 3); // Explicit: device 3
    
    // Explicit should win
    EXPECT_EQ(3, default_map_->getDeviceForWeight("blk.0.attn_q.weight"));
    
    // Layer should win over pattern
    EXPECT_EQ(2, default_map_->getDeviceForWeight("blk.0.attn_k.weight"));
    
    // Pattern should win over default
    EXPECT_EQ(1, default_map_->getDeviceForWeight("blk.5.attn_v.weight"));
    
    // Non-matching uses default
    EXPECT_EQ(0, default_map_->getDeviceForWeight("output.weight"));
}

/**
 * Test: Clear all mappings
 */
TEST_F(Test__WeightPlacementMap, ClearMappings) {
    default_map_->setTensorDevice("test", 1);
    default_map_->setLayerDevice(0, 2);
    default_map_->setPatternDevice("*attn*", 3);
    
    EXPECT_GT(default_map_->tensorMappingCount(), 0);
    EXPECT_GT(default_map_->layerMappingCount(), 0);
    
    default_map_->clear();
    
    EXPECT_EQ(0, default_map_->tensorMappingCount());
    EXPECT_EQ(0, default_map_->layerMappingCount());
    
    // Should fall back to default
    EXPECT_EQ(0, default_map_->getDeviceForWeight("test"));
    EXPECT_EQ(0, default_map_->getDeviceForWeight("blk.0.attn_q.weight"));
}

/**
 * Test: Realistic MoE scenario
 */
TEST_F(Test__WeightPlacementMap, MoEScenario) {
    // Default to CPU
    auto moe_map = std::make_shared<WeightPlacementMap>(0); // CPU
    
    // Shared components on GPU
    moe_map->setPatternDevice("*embd*", 1);
    moe_map->setPatternDevice("*output*", 1);
    moe_map->setPatternDevice("*gate*", 1);
    
    // Shared expert (expert 0) on GPU - use more specific pattern
    moe_map->setPatternDevice("*.expert.0.*", 1);
    
    // Sparse experts (1-7) on CPU (default)
    
    EXPECT_EQ(1, moe_map->getDeviceForWeight("token_embd.weight"));
    EXPECT_EQ(1, moe_map->getDeviceForWeight("blk.0.gate.weight"));
    EXPECT_EQ(1, moe_map->getDeviceForWeight("blk.0.expert.0.ffn_gate.weight"));
    EXPECT_EQ(1, moe_map->getDeviceForWeight("blk.0.expert.1.ffn_gate.weight")); // Matches *gate*
    EXPECT_EQ(1, moe_map->getDeviceForWeight("blk.0.expert.7.ffn_gate.weight")); // Matches *gate*
    EXPECT_EQ(1, moe_map->getDeviceForWeight("output.weight"));
}

/**
 * Test: Layer split scenario (first N layers on GPU)
 */
TEST_F(Test__WeightPlacementMap, LayerSplitScenario) {
    // Default to CPU
    auto split_map = std::make_shared<WeightPlacementMap>(0); // CPU
    
    // First 4 layers on GPU
    split_map->setLayerRange(0, 3, 1);
    
    // Embeddings and output on GPU
    split_map->setPatternDevice("*embd*", 1);
    split_map->setPatternDevice("output*", 1);
    
    EXPECT_EQ(1, split_map->getDeviceForWeight("token_embd.weight"));
    EXPECT_EQ(1, split_map->getDeviceForWeight("blk.0.attn_q.weight"));
    EXPECT_EQ(1, split_map->getDeviceForWeight("blk.3.ffn_down.weight"));
    EXPECT_EQ(0, split_map->getDeviceForWeight("blk.4.attn_q.weight"));
    EXPECT_EQ(0, split_map->getDeviceForWeight("blk.23.attn_q.weight"));
    EXPECT_EQ(1, split_map->getDeviceForWeight("output.weight"));
}

/**
 * Test: Different default device
 */
TEST_F(Test__WeightPlacementMap, CustomDefaultDevice) {
    auto gpu_map = std::make_shared<WeightPlacementMap>(1); // Default to GPU
    
    EXPECT_EQ(1, gpu_map->defaultDevice());
    EXPECT_EQ(1, gpu_map->getDeviceForWeight("any.tensor"));
    
    // Override specific tensors to CPU
    gpu_map->setPatternDevice("*large_expert*", 0);
    EXPECT_EQ(0, gpu_map->getDeviceForWeight("blk.0.large_expert.weight"));
    EXPECT_EQ(1, gpu_map->getDeviceForWeight("blk.0.small_expert.weight"));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
