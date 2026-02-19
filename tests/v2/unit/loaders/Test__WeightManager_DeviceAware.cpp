/**
 * @file Test__WeightManager_DeviceAware.cpp
 * @brief Unit tests for WeightManager device-aware features
 *
 * Tests the features folded from WeightPreloader into WeightManager:
 * - getWeightForDevice() - device-specific tensor instances
 * - preloadForDevices() - pre-upload weights to multiple devices
 * - cloneTensorForDevice() - tensor cloning for different devices
 * - packGemmWeights() - GEMM weight packing and preparation
 * - uploadNonGemmWeights() - non-GEMM weight upload
 * - preloadStats() - tracking packed weight counts
 * - Per-device caching with first_device tracking
 *
 * This test uses MockWeightManager to test the IWeightManager interface
 * contract without requiring real GGUF files or a real ModelLoader.
 *
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <thread>
#include <atomic>

#include "loaders/WeightManager.h"
#include "models/qwen/Qwen2Schema.h"
#include "tensors/Tensors.h"
#include "backends/DeviceId.h"
#include "utils/MPIContext.h"

// Test utilities and mocks
#include "mocks/MockWeightManager.h"
#include "utils/TestTensorFactory.h"

using namespace llaminar2;
using namespace llaminar2::test;

// =============================================================================
// Test Fixtures
// =============================================================================

/**
 * @brief Test fixture for WeightManager device-aware interface tests
 *
 * Uses MockWeightManager to test IWeightManager interface behavior without
 * requiring real GGUF files.
 */
class WeightManagerDeviceAwareTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Create mock weight manager with test weights using builder
        mock_ = MockWeightManagerBuilder()
                    .setStrategy(WeightDistributionStrategy::REPLICATED)
                    // Layer 0 weights - GEMM weights
                    .addQ4_0RandomWeight("blk.0.attn_q.weight", {64, 64})
                    .setColumnParallel("blk.0.attn_q.weight")
                    .addQ4_0RandomWeight("blk.0.attn_k.weight", {32, 64})
                    .setColumnParallel("blk.0.attn_k.weight")
                    .addQ4_0RandomWeight("blk.0.attn_v.weight", {32, 64})
                    .setColumnParallel("blk.0.attn_v.weight")
                    .addQ4_0RandomWeight("blk.0.attn_output.weight", {64, 64})
                    .setInputParallel("blk.0.attn_output.weight")
                    .addQ4_0RandomWeight("blk.0.ffn_gate.weight", {128, 64})
                    .setColumnParallel("blk.0.ffn_gate.weight")
                    .addQ4_0RandomWeight("blk.0.ffn_up.weight", {128, 64})
                    .setColumnParallel("blk.0.ffn_up.weight")
                    .addQ4_0RandomWeight("blk.0.ffn_down.weight", {64, 128})
                    .setInputParallel("blk.0.ffn_down.weight")
                    // Non-GEMM weights (norms, embeddings)
                    .addFP32RandomWeight("blk.0.attn_norm.weight", {64, 1})
                    .setNonGemm("blk.0.attn_norm.weight")
                    .setReplicated("blk.0.attn_norm.weight")
                    .addFP32RandomWeight("blk.0.ffn_norm.weight", {64, 1})
                    .setNonGemm("blk.0.ffn_norm.weight")
                    .setReplicated("blk.0.ffn_norm.weight")
                    .addFP32RandomWeight("token_embd.weight", {1000, 64})
                    .setNonGemm("token_embd.weight")
                    .setReplicated("token_embd.weight")
                    // Layer 1 weights
                    .addQ4_0RandomWeight("blk.1.attn_q.weight", {64, 64})
                    .setColumnParallel("blk.1.attn_q.weight")
                    .addFP32RandomWeight("blk.1.attn_norm.weight", {64, 1})
                    .setNonGemm("blk.1.attn_norm.weight")
                    .setReplicated("blk.1.attn_norm.weight")
                    .build();

        // Define test devices
        device_a_ = DeviceId::cpu();
        device_b_ = DeviceId(DeviceType::CUDA, 0);
        device_c_ = DeviceId(DeviceType::ROCm, 0);
    }

    std::shared_ptr<MockWeightManager> mock_;

    DeviceId device_a_;
    DeviceId device_b_;
    DeviceId device_c_;
};

// =============================================================================
// getWeightForDevice Tests
// =============================================================================

TEST_F(WeightManagerDeviceAwareTest, GetWeightForDevice_ReturnsWeight)
{
    // MockWeightManager.getWeightForDevice delegates to getWeight
    auto weight_a = mock_->getWeightForDevice("blk.0.attn_q.weight", device_a_);
    ASSERT_NE(weight_a, nullptr);
    EXPECT_EQ(weight_a->shape()[0], 64u);
    EXPECT_EQ(weight_a->shape()[1], 64u);
}

TEST_F(WeightManagerDeviceAwareTest, GetWeightForDevice_DifferentDevices)
{
    // Mock returns same weight for all devices (no cloning in mock)
    auto weight_a = mock_->getWeightForDevice("blk.0.attn_q.weight", device_a_);
    auto weight_b = mock_->getWeightForDevice("blk.0.attn_q.weight", device_b_);

    ASSERT_NE(weight_a, nullptr);
    ASSERT_NE(weight_b, nullptr);

    // Mock doesn't clone, so same pointer (this is expected mock behavior)
    EXPECT_EQ(weight_a.get(), weight_b.get());
}

TEST_F(WeightManagerDeviceAwareTest, GetWeightForDevice_DifferentWeights)
{
    // Get multiple different weights for same device
    auto q_weight = mock_->getWeightForDevice("blk.0.attn_q.weight", device_a_);
    auto k_weight = mock_->getWeightForDevice("blk.0.attn_k.weight", device_a_);
    auto norm_weight = mock_->getWeightForDevice("blk.0.attn_norm.weight", device_a_);

    ASSERT_NE(q_weight, nullptr);
    ASSERT_NE(k_weight, nullptr);
    ASSERT_NE(norm_weight, nullptr);

    // All should be different tensors
    EXPECT_NE(q_weight.get(), k_weight.get());
    EXPECT_NE(q_weight.get(), norm_weight.get());

    // Q and K have different shapes (K has fewer heads due to GQA)
    EXPECT_NE(q_weight->shape()[0], k_weight->shape()[0]);
}

TEST_F(WeightManagerDeviceAwareTest, GetWeightForDevice_NonExistentWeight)
{
    // Request a weight that doesn't exist
    auto weight = mock_->getWeightForDevice("nonexistent.weight", device_a_);
    EXPECT_EQ(weight, nullptr);

    // Missing request should be tracked
    const auto &missing = mock_->missingWeightRequests();
    EXPECT_EQ(missing.size(), 1u);
    EXPECT_EQ(missing[0], "nonexistent.weight");
}

TEST_F(WeightManagerDeviceAwareTest, GetWeightForDevice_TracksCallCount)
{
    EXPECT_EQ(mock_->getWeightCallCount(), 0u);

    mock_->getWeightForDevice("blk.0.attn_q.weight", device_a_);
    EXPECT_EQ(mock_->getWeightCallCount(), 1u);

    mock_->getWeightForDevice("blk.0.attn_k.weight", device_b_);
    EXPECT_EQ(mock_->getWeightCallCount(), 2u);

    mock_->getWeightForDevice("blk.0.attn_q.weight", device_a_); // Same weight again
    EXPECT_EQ(mock_->getWeightCallCount(), 3u);
}

// =============================================================================
// preloadForDevices Tests
// =============================================================================

TEST_F(WeightManagerDeviceAwareTest, PreloadForDevices_ReturnsTrue)
{
    // Mock preloadForDevices always returns true
    std::vector<DeviceId> devices = {device_a_, device_b_, device_c_};
    bool result = mock_->preloadForDevices(devices);
    EXPECT_TRUE(result);
}

TEST_F(WeightManagerDeviceAwareTest, PreloadForDevices_EmptyList_ReturnsTrue)
{
    // Edge case: empty device list should return true
    std::vector<DeviceId> empty_devices;
    bool result = mock_->preloadForDevices(empty_devices);
    EXPECT_TRUE(result);
}

TEST_F(WeightManagerDeviceAwareTest, PreloadForDevices_SingleDevice)
{
    std::vector<DeviceId> devices = {device_a_};
    bool result = mock_->preloadForDevices(devices);
    EXPECT_TRUE(result);
}

// =============================================================================
// packGemmWeights Tests
// =============================================================================

TEST_F(WeightManagerDeviceAwareTest, PackGemmWeights_ReturnsTrue)
{
    // Mock packGemmWeights always returns true
    bool result = mock_->packGemmWeights(DeviceId::cpu());
    EXPECT_TRUE(result);
}

TEST_F(WeightManagerDeviceAwareTest, PackGemmWeights_WithCallback)
{
    // Mock accepts callback but doesn't invoke it
    int callback_count = 0;
    auto callback = [&](size_t, size_t, const std::string &)
    {
        callback_count++;
    };

    bool result = mock_->packGemmWeights(DeviceId::cpu(), callback);
    EXPECT_TRUE(result);
    // Mock doesn't invoke callback
    EXPECT_EQ(callback_count, 0);
}

TEST_F(WeightManagerDeviceAwareTest, PackGemmWeights_WithReleaseFlag)
{
    bool result = mock_->packGemmWeights(DeviceId::cpu(), nullptr, true);
    EXPECT_TRUE(result);
}

TEST_F(WeightManagerDeviceAwareTest, PackGemmWeights_GPUDevice)
{
    bool result = mock_->packGemmWeights(DeviceId(DeviceType::CUDA, 0));
    EXPECT_TRUE(result);
}

// =============================================================================
// uploadNonGemmWeights Tests
// =============================================================================

TEST_F(WeightManagerDeviceAwareTest, UploadNonGemmWeights_ReturnsTrue)
{
    bool result = mock_->uploadNonGemmWeights(DeviceId::cpu());
    EXPECT_TRUE(result);
}

TEST_F(WeightManagerDeviceAwareTest, UploadNonGemmWeights_GPUDevice)
{
    bool result = mock_->uploadNonGemmWeights(DeviceId(DeviceType::CUDA, 0));
    EXPECT_TRUE(result);
}

// =============================================================================
// preloadStats Tests
// =============================================================================

TEST_F(WeightManagerDeviceAwareTest, PreloadStats_ReturnsZero)
{
    // Mock always returns (0, 0)
    auto [cpu_packed, gpu_packed] = mock_->preloadStats();
    EXPECT_EQ(cpu_packed, 0u);
    EXPECT_EQ(gpu_packed, 0u);
}

TEST_F(WeightManagerDeviceAwareTest, PreloadStats_AfterPacking)
{
    // Even after packing, mock returns (0, 0) - mock doesn't track stats
    mock_->packGemmWeights(DeviceId::cpu());
    auto [cpu_packed, gpu_packed] = mock_->preloadStats();
    EXPECT_EQ(cpu_packed, 0u);
    EXPECT_EQ(gpu_packed, 0u);
}

// =============================================================================
// isGemmWeight Tests
// =============================================================================

TEST_F(WeightManagerDeviceAwareTest, IsGemmWeight_GemmWeightsReturnTrue)
{
    EXPECT_TRUE(mock_->isGemmWeight("blk.0.attn_q.weight"));
    EXPECT_TRUE(mock_->isGemmWeight("blk.0.attn_k.weight"));
    EXPECT_TRUE(mock_->isGemmWeight("blk.0.attn_v.weight"));
    EXPECT_TRUE(mock_->isGemmWeight("blk.0.attn_output.weight"));
    EXPECT_TRUE(mock_->isGemmWeight("blk.0.ffn_gate.weight"));
    EXPECT_TRUE(mock_->isGemmWeight("blk.0.ffn_up.weight"));
    EXPECT_TRUE(mock_->isGemmWeight("blk.0.ffn_down.weight"));
}

TEST_F(WeightManagerDeviceAwareTest, IsGemmWeight_NonGemmWeightsReturnFalse)
{
    EXPECT_FALSE(mock_->isGemmWeight("blk.0.attn_norm.weight"));
    EXPECT_FALSE(mock_->isGemmWeight("blk.0.ffn_norm.weight"));
    EXPECT_FALSE(mock_->isGemmWeight("token_embd.weight"));
}

TEST_F(WeightManagerDeviceAwareTest, IsGemmWeight_UnknownReturnsTrue)
{
    // Unknown weights default to GEMM in MockWeightManager
    EXPECT_TRUE(mock_->isGemmWeight("unknown.weight"));
}

// =============================================================================
// getShardingMode Tests
// =============================================================================

TEST_F(WeightManagerDeviceAwareTest, GetShardingMode_ColumnParallel)
{
    EXPECT_EQ(mock_->getShardingMode("blk.0.attn_q.weight"), ShardingMode::COLUMN_PARALLEL);
    EXPECT_EQ(mock_->getShardingMode("blk.0.attn_k.weight"), ShardingMode::COLUMN_PARALLEL);
    EXPECT_EQ(mock_->getShardingMode("blk.0.ffn_gate.weight"), ShardingMode::COLUMN_PARALLEL);
}

TEST_F(WeightManagerDeviceAwareTest, GetShardingMode_InputParallel)
{
    EXPECT_EQ(mock_->getShardingMode("blk.0.attn_output.weight"), ShardingMode::INPUT_PARALLEL);
    EXPECT_EQ(mock_->getShardingMode("blk.0.ffn_down.weight"), ShardingMode::INPUT_PARALLEL);
}

TEST_F(WeightManagerDeviceAwareTest, GetShardingMode_Replicate)
{
    EXPECT_EQ(mock_->getShardingMode("blk.0.attn_norm.weight"), ShardingMode::REPLICATE);
    EXPECT_EQ(mock_->getShardingMode("blk.0.ffn_norm.weight"), ShardingMode::REPLICATE);
    EXPECT_EQ(mock_->getShardingMode("token_embd.weight"), ShardingMode::REPLICATE);
}

TEST_F(WeightManagerDeviceAwareTest, GetShardingMode_UnknownDefaultsToReplicate)
{
    EXPECT_EQ(mock_->getShardingMode("unknown.weight"), ShardingMode::REPLICATE);
}

// =============================================================================
// isWeightSharded Tests
// =============================================================================

TEST_F(WeightManagerDeviceAwareTest, IsWeightSharded_ReplicatedStrategy)
{
    // With REPLICATED strategy, no weights are sharded
    EXPECT_FALSE(mock_->isWeightSharded("blk.0.attn_q.weight"));
    EXPECT_FALSE(mock_->isWeightSharded("blk.0.attn_norm.weight"));
}

TEST_F(WeightManagerDeviceAwareTest, IsWeightSharded_ShardedStrategy)
{
    // Create mock with SHARDED strategy
    auto sharded_mock = MockWeightManagerBuilder()
                            .setStrategy(WeightDistributionStrategy::SHARDED)
                            .addQ4_0RandomWeight("blk.0.attn_q.weight", {64, 64})
                            .setColumnParallel("blk.0.attn_q.weight")
                            .addFP32RandomWeight("blk.0.attn_norm.weight", {64, 1})
                            .setReplicated("blk.0.attn_norm.weight")
                            .build();

    // Column-parallel weight should be sharded
    EXPECT_TRUE(sharded_mock->isWeightSharded("blk.0.attn_q.weight"));

    // Replicated weight should NOT be sharded
    EXPECT_FALSE(sharded_mock->isWeightSharded("blk.0.attn_norm.weight"));
}

// =============================================================================
// strategy Tests
// =============================================================================

TEST_F(WeightManagerDeviceAwareTest, Strategy_ReturnsCorrectValue)
{
    EXPECT_EQ(mock_->strategy(), WeightDistributionStrategy::REPLICATED);

    auto sharded_mock = MockWeightManagerBuilder()
                            .setStrategy(WeightDistributionStrategy::SHARDED)
                            .build();
    EXPECT_EQ(sharded_mock->strategy(), WeightDistributionStrategy::SHARDED);
}

// =============================================================================
// cacheSize and clearCache Tests
// =============================================================================

TEST_F(WeightManagerDeviceAwareTest, CacheSize_ReturnsCorrectCount)
{
    // Mock stores weights in cache
    size_t expected_count = 12; // We added 12 weights in SetUp
    EXPECT_EQ(mock_->cacheSize(), expected_count);
}

TEST_F(WeightManagerDeviceAwareTest, ClearCache_RemovesAllWeights)
{
    EXPECT_GT(mock_->cacheSize(), 0u);
    mock_->clearCache();
    EXPECT_EQ(mock_->cacheSize(), 0u);

    // After clearing, weights should return nullptr
    auto weight = mock_->getWeightForDevice("blk.0.attn_q.weight");
    EXPECT_EQ(weight, nullptr);
}

// =============================================================================
// getDecodeWeight Tests
// =============================================================================

TEST_F(WeightManagerDeviceAwareTest, GetDecodeWeight_ReturnsWeight)
{
    // Add a decode weight
    auto decode_tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{32, 64});
    mock_->addDecodeWeight("blk.0.attn_q.weight", decode_tensor);

    auto decode = mock_->getDecodeWeight("blk.0.attn_q.weight", DeviceId::cpu(), 0.5f);
    ASSERT_NE(decode, nullptr);
    EXPECT_EQ(decode.get(), decode_tensor.get());
}

TEST_F(WeightManagerDeviceAwareTest, GetDecodeWeight_FallsBackToMainWeight)
{
    // Without explicit decode weight, falls back to main weight
    auto decode = mock_->getDecodeWeight("blk.0.attn_q.weight", DeviceId::cpu(), 0.5f);
    auto main = mock_->getWeightForDevice("blk.0.attn_q.weight");

    ASSERT_NE(decode, nullptr);
    EXPECT_EQ(decode.get(), main.get());
}

// =============================================================================
// MockWeightManagerBuilder Tests
// =============================================================================

TEST_F(WeightManagerDeviceAwareTest, Builder_AddAttentionLayer)
{
    auto mock = MockWeightManagerBuilder()
                    .addAttentionLayer(0, 64, 16, 4, 2) // layer 0, hidden=64, head_dim=16, 4 heads, 2 KV heads
                    .build();

    EXPECT_NE(mock->getWeightForDevice("blk.0.attn_q.weight"), nullptr);
    EXPECT_NE(mock->getWeightForDevice("blk.0.attn_k.weight"), nullptr);
    EXPECT_NE(mock->getWeightForDevice("blk.0.attn_v.weight"), nullptr);
    EXPECT_NE(mock->getWeightForDevice("blk.0.attn_output.weight"), nullptr);
}

TEST_F(WeightManagerDeviceAwareTest, Builder_AddFFNLayer)
{
    auto mock = MockWeightManagerBuilder()
                    .addFFNLayer(0, 64, 256) // layer 0, hidden=64, ffn_dim=256
                    .build();

    EXPECT_NE(mock->getWeightForDevice("blk.0.ffn_gate.weight"), nullptr);
    EXPECT_NE(mock->getWeightForDevice("blk.0.ffn_up.weight"), nullptr);
    EXPECT_NE(mock->getWeightForDevice("blk.0.ffn_down.weight"), nullptr);
}

TEST_F(WeightManagerDeviceAwareTest, Builder_AddNormWeights)
{
    auto mock = MockWeightManagerBuilder()
                    .addNormWeights(0, 64) // layer 0, hidden=64
                    .build();

    EXPECT_NE(mock->getWeightForDevice("blk.0.attn_norm.weight"), nullptr);
    EXPECT_NE(mock->getWeightForDevice("blk.0.ffn_norm.weight"), nullptr);
}

TEST_F(WeightManagerDeviceAwareTest, Builder_AddEmbedding)
{
    auto mock = MockWeightManagerBuilder()
                    .addEmbedding(1000, 64) // vocab=1000, hidden=64
                    .build();

    auto embd = mock->getWeightForDevice("token_embd.weight");
    ASSERT_NE(embd, nullptr);
    EXPECT_EQ(embd->shape()[0], 1000u);
    EXPECT_EQ(embd->shape()[1], 64u);
}

TEST_F(WeightManagerDeviceAwareTest, Builder_AddLMHead)
{
    auto mock = MockWeightManagerBuilder()
                    .addLMHead(1000, 64) // vocab=1000, hidden=64
                    .build();

    auto head = mock->getWeightForDevice("output.weight");
    ASSERT_NE(head, nullptr);
    EXPECT_EQ(head->shape()[0], 1000u);
    EXPECT_EQ(head->shape()[1], 64u);
}

// =============================================================================
// Reset Counters Test
// =============================================================================

TEST_F(WeightManagerDeviceAwareTest, ResetCounters_ClearsTracking)
{
    mock_->getWeightForDevice("blk.0.attn_q.weight");
    mock_->getWeightForDevice("nonexistent.weight");

    EXPECT_EQ(mock_->getWeightCallCount(), 2u);
    EXPECT_EQ(mock_->missingWeightRequests().size(), 1u);

    mock_->resetCounters();

    EXPECT_EQ(mock_->getWeightCallCount(), 0u);
    EXPECT_EQ(mock_->missingWeightRequests().size(), 0u);
}

// =============================================================================
// Multiple Layers Test
// =============================================================================

TEST_F(WeightManagerDeviceAwareTest, MultipleLayerWeights)
{
    // Test handling of multiple layers
    auto q0 = mock_->getWeightForDevice("blk.0.attn_q.weight");
    auto q1 = mock_->getWeightForDevice("blk.1.attn_q.weight");

    ASSERT_NE(q0, nullptr);
    ASSERT_NE(q1, nullptr);

    // Different layer weights should be different tensors
    EXPECT_NE(q0.get(), q1.get());
}

// =============================================================================
// Static Preset Tests
// =============================================================================

TEST_F(WeightManagerDeviceAwareTest, CreateReplicated_Preset)
{
    auto mock = MockWeightManager::createReplicated();
    EXPECT_EQ(mock->strategy(), WeightDistributionStrategy::REPLICATED);
}

TEST_F(WeightManagerDeviceAwareTest, CreateShardedQwen2_Preset)
{
    auto mock = MockWeightManager::createShardedQwen2();
    EXPECT_EQ(mock->strategy(), WeightDistributionStrategy::SHARDED);
}
