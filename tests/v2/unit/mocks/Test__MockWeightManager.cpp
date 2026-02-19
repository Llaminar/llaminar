/**
 * @file Test__MockWeightManager.cpp
 * @brief Unit tests for MockWeightManager
 *
 * Tests the mock weight manager's ability to:
 * - Provide in-memory weights without GGUF files
 * - Support different distribution strategies (REPLICATED, SHARDED)
 * - Configure sharding modes per weight
 * - Track GEMM vs non-GEMM weights
 * - Use builder pattern for fluent configuration
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include "mocks/MockWeightManager.h"
#include "tensors/Tensors.h"

using namespace llaminar2;
using namespace llaminar2::test;

// =============================================================================
// BASIC CONSTRUCTION TESTS
// =============================================================================

TEST(Test__MockWeightManager, DefaultConstruction) {
    MockWeightManager mock;

    EXPECT_EQ(mock.strategy(), WeightDistributionStrategy::REPLICATED);
    EXPECT_EQ(mock.cacheSize(), 0);
    EXPECT_EQ(mock.decodeCacheSize(), 0);
    EXPECT_EQ(mock.getWeightCallCount(), 0);
}

TEST(Test__MockWeightManager, BuilderBasicConfiguration) {
    auto mock = MockWeightManagerBuilder()
        .setStrategy(WeightDistributionStrategy::SHARDED)
        .build();

    EXPECT_EQ(mock->strategy(), WeightDistributionStrategy::SHARDED);
}

// =============================================================================
// WEIGHT ADDITION TESTS
// =============================================================================

TEST(Test__MockWeightManager, AddFP32RandomWeight) {
    auto mock = MockWeightManagerBuilder()
        .addFP32RandomWeight("test_weight", {64, 128}, -1.0f, 1.0f, 42)
        .build();

    EXPECT_TRUE(mock->hasWeight("test_weight"));
    EXPECT_EQ(mock->cacheSize(), 1);

    auto weight = mock->getWeightForDevice("test_weight");
    ASSERT_NE(weight, nullptr);
    EXPECT_EQ(weight->rows(), 64);
    EXPECT_EQ(weight->cols(), 128);

    // Verify it's actually FP32
    auto* fp32 = dynamic_cast<FP32Tensor*>(weight.get());
    ASSERT_NE(fp32, nullptr);

    // Verify data is in range
    const float* data = fp32->data();
    for (size_t i = 0; i < 64 * 128; ++i) {
        EXPECT_GE(data[i], -1.0f);
        EXPECT_LE(data[i], 1.0f);
    }
}

TEST(Test__MockWeightManager, AddFP32ZerosWeight) {
    auto mock = MockWeightManagerBuilder()
        .addFP32ZerosWeight("zeros", {32, 64})
        .build();

    auto weight = mock->getWeightForDevice("zeros");
    ASSERT_NE(weight, nullptr);

    auto* fp32 = dynamic_cast<FP32Tensor*>(weight.get());
    ASSERT_NE(fp32, nullptr);

    const float* data = fp32->data();
    for (size_t i = 0; i < 32 * 64; ++i) {
        EXPECT_EQ(data[i], 0.0f);
    }
}

TEST(Test__MockWeightManager, AddFP32OnesWeight) {
    auto mock = MockWeightManagerBuilder()
        .addFP32OnesWeight("ones", {16, 32})
        .build();

    auto weight = mock->getWeightForDevice("ones");
    ASSERT_NE(weight, nullptr);

    auto* fp32 = dynamic_cast<FP32Tensor*>(weight.get());
    ASSERT_NE(fp32, nullptr);

    const float* data = fp32->data();
    for (size_t i = 0; i < 16 * 32; ++i) {
        EXPECT_EQ(data[i], 1.0f);
    }
}

TEST(Test__MockWeightManager, AddQ4_0RandomWeight) {
    auto mock = MockWeightManagerBuilder()
        .addQ4_0RandomWeight("q4_weight", {64, 128}, -1.0f, 1.0f, 42)
        .build();

    auto weight = mock->getWeightForDevice("q4_weight");
    ASSERT_NE(weight, nullptr);
    EXPECT_EQ(weight->rows(), 64);
    EXPECT_EQ(weight->cols(), 128);

    // Verify it's Q4_0
    auto* q4 = dynamic_cast<Q4_0Tensor*>(weight.get());
    ASSERT_NE(q4, nullptr);
}

TEST(Test__MockWeightManager, AddQ8_0RandomWeight) {
    auto mock = MockWeightManagerBuilder()
        .addQ8_0RandomWeight("q8_weight", {32, 64}, -0.5f, 0.5f, 123)
        .build();

    auto weight = mock->getWeightForDevice("q8_weight");
    ASSERT_NE(weight, nullptr);
    EXPECT_EQ(weight->rows(), 32);
    EXPECT_EQ(weight->cols(), 64);

    // Verify it's Q8_0
    auto* q8 = dynamic_cast<Q8_0Tensor*>(weight.get());
    ASSERT_NE(q8, nullptr);
}

TEST(Test__MockWeightManager, Add1DTensor) {
    // 1D tensors (like norm weights) should work with single dimension
    auto mock = MockWeightManagerBuilder()
        .addFP32OnesWeight("norm_weight", {896})
        .build();

    auto weight = mock->getWeightForDevice("norm_weight");
    ASSERT_NE(weight, nullptr);
    EXPECT_EQ(weight->rows(), 896);
    EXPECT_EQ(weight->cols(), 1);
}

// =============================================================================
// SHARDING MODE TESTS
// =============================================================================

TEST(Test__MockWeightManager, ShardingModeDefault) {
    auto mock = MockWeightManagerBuilder()
        .setStrategy(WeightDistributionStrategy::SHARDED)
        .addFP32RandomWeight("some_weight", {64, 64})
        .build();

    // Default should be REPLICATE
    EXPECT_EQ(mock->getShardingMode("some_weight"), ShardingMode::REPLICATE);
    EXPECT_FALSE(mock->isWeightSharded("some_weight"));
}

TEST(Test__MockWeightManager, ShardingModeColumnParallel) {
    auto mock = MockWeightManagerBuilder()
        .setStrategy(WeightDistributionStrategy::SHARDED)
        .addFP32RandomWeight("blk.0.attn_q.weight", {896, 896})
        .setColumnParallel("blk.0.attn_q.weight")
        .build();

    EXPECT_EQ(mock->getShardingMode("blk.0.attn_q.weight"), ShardingMode::COLUMN_PARALLEL);
    EXPECT_TRUE(mock->isWeightSharded("blk.0.attn_q.weight"));
}

TEST(Test__MockWeightManager, ShardingModeRowParallel) {
    auto mock = MockWeightManagerBuilder()
        .setStrategy(WeightDistributionStrategy::SHARDED)
        .addFP32RandomWeight("blk.0.attn_output.weight", {896, 896})
        .setRowParallel("blk.0.attn_output.weight")
        .build();

    EXPECT_EQ(mock->getShardingMode("blk.0.attn_output.weight"), ShardingMode::ROW_PARALLEL);
    EXPECT_TRUE(mock->isWeightSharded("blk.0.attn_output.weight"));
}

TEST(Test__MockWeightManager, ShardingModeInputParallel) {
    auto mock = MockWeightManagerBuilder()
        .setStrategy(WeightDistributionStrategy::SHARDED)
        .addFP32RandomWeight("blk.0.ffn_down.weight", {896, 4864})
        .setInputParallel("blk.0.ffn_down.weight")
        .build();

    EXPECT_EQ(mock->getShardingMode("blk.0.ffn_down.weight"), ShardingMode::INPUT_PARALLEL);
    EXPECT_TRUE(mock->isWeightSharded("blk.0.ffn_down.weight"));
}

TEST(Test__MockWeightManager, ShardingModeExplicitReplicate) {
    auto mock = MockWeightManagerBuilder()
        .setStrategy(WeightDistributionStrategy::SHARDED)
        .addFP32OnesWeight("blk.0.attn_norm.weight", {896})
        .setReplicated("blk.0.attn_norm.weight")
        .build();

    EXPECT_EQ(mock->getShardingMode("blk.0.attn_norm.weight"), ShardingMode::REPLICATE);
    EXPECT_FALSE(mock->isWeightSharded("blk.0.attn_norm.weight"));
}

TEST(Test__MockWeightManager, ShardingNotAppliedInReplicatedMode) {
    // Even if we set a sharding mode, REPLICATED strategy means no sharding
    auto mock = MockWeightManagerBuilder()
        .setStrategy(WeightDistributionStrategy::REPLICATED)
        .addFP32RandomWeight("blk.0.attn_q.weight", {896, 896})
        .setColumnParallel("blk.0.attn_q.weight")
        .build();

    // Mode is set, but isWeightSharded returns false because strategy is REPLICATED
    EXPECT_EQ(mock->getShardingMode("blk.0.attn_q.weight"), ShardingMode::COLUMN_PARALLEL);
    EXPECT_FALSE(mock->isWeightSharded("blk.0.attn_q.weight"));
}

// =============================================================================
// GEMM WEIGHT TESTS
// =============================================================================

TEST(Test__MockWeightManager, GemmWeightDefault) {
    auto mock = MockWeightManagerBuilder()
        .addFP32RandomWeight("some_weight", {64, 64})
        .build();

    // Default should be GEMM
    EXPECT_TRUE(mock->isGemmWeight("some_weight"));
}

TEST(Test__MockWeightManager, NonGemmWeight) {
    auto mock = MockWeightManagerBuilder()
        .addFP32OnesWeight("blk.0.attn_norm.weight", {896})
        .setNonGemm("blk.0.attn_norm.weight")
        .build();

    EXPECT_FALSE(mock->isGemmWeight("blk.0.attn_norm.weight"));
}

TEST(Test__MockWeightManager, SetGemmExplicitly) {
    auto mock = MockWeightManagerBuilder()
        .addFP32RandomWeight("some_weight", {64, 64})
        .setNonGemm("some_weight")  // First mark as non-GEMM
        .setGemm("some_weight")     // Then mark as GEMM
        .build();

    EXPECT_TRUE(mock->isGemmWeight("some_weight"));
}

// =============================================================================
// DECODE WEIGHT TESTS
// =============================================================================

TEST(Test__MockWeightManager, DecodeWeightDirect) {
    auto decode_tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{64, 64});

    auto mock = MockWeightManagerBuilder()
        .addFP32RandomWeight("blk.0.attn_q.weight", {896, 896})
        .addDecodeWeight("blk.0.attn_q.weight", decode_tensor)
        .build();

    auto decode = mock->getDecodeWeight("blk.0.attn_q.weight", DeviceId::cpu(), 0.2f);
    ASSERT_NE(decode, nullptr);
    EXPECT_EQ(decode.get(), decode_tensor.get());  // Same pointer
}

TEST(Test__MockWeightManager, DecodeWeightFallbackToMain) {
    auto mock = MockWeightManagerBuilder()
        .addFP32RandomWeight("blk.0.attn_q.weight", {896, 896})
        .build();

    // No decode weight set, should fall back to main weight
    auto decode = mock->getDecodeWeight("blk.0.attn_q.weight", DeviceId::cpu(), 0.2f);
    ASSERT_NE(decode, nullptr);

    auto main = mock->getWeightForDevice("blk.0.attn_q.weight");
    EXPECT_EQ(decode.get(), main.get());
}

TEST(Test__MockWeightManager, DecodeCacheClear) {
    auto decode_tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{64, 64});

    auto mock = MockWeightManagerBuilder()
        .addDecodeWeight("test", decode_tensor)
        .build();

    EXPECT_EQ(mock->decodeCacheSize(), 1);
    mock->clearDecodeCache();
    EXPECT_EQ(mock->decodeCacheSize(), 0);
}

// =============================================================================
// PRESET LAYER TESTS
// =============================================================================

TEST(Test__MockWeightManager, AddAttentionLayer) {
    // Use smaller dimensions for faster testing
    auto mock = MockWeightManagerBuilder()
        .setStrategy(WeightDistributionStrategy::SHARDED)
        .addAttentionLayer(0, 128, 32, 4, 2)  // Small test dimensions
        .build();

    // Check Q projection
    EXPECT_TRUE(mock->hasWeight("blk.0.attn_q.weight"));
    EXPECT_EQ(mock->getShardingMode("blk.0.attn_q.weight"), ShardingMode::COLUMN_PARALLEL);

    auto q = mock->getWeightForDevice("blk.0.attn_q.weight");
    ASSERT_NE(q, nullptr);
    EXPECT_EQ(q->rows(), 4 * 32);   // num_heads * head_dim = 128
    EXPECT_EQ(q->cols(), 128);      // hidden_dim

    // Check K projection
    EXPECT_TRUE(mock->hasWeight("blk.0.attn_k.weight"));
    EXPECT_EQ(mock->getShardingMode("blk.0.attn_k.weight"), ShardingMode::COLUMN_PARALLEL);

    auto k = mock->getWeightForDevice("blk.0.attn_k.weight");
    ASSERT_NE(k, nullptr);
    EXPECT_EQ(k->rows(), 2 * 32);   // num_kv_heads * head_dim = 64
    EXPECT_EQ(k->cols(), 128);

    // Check V projection
    EXPECT_TRUE(mock->hasWeight("blk.0.attn_v.weight"));
    EXPECT_EQ(mock->getShardingMode("blk.0.attn_v.weight"), ShardingMode::COLUMN_PARALLEL);

    // Check output projection
    EXPECT_TRUE(mock->hasWeight("blk.0.attn_output.weight"));
    EXPECT_EQ(mock->getShardingMode("blk.0.attn_output.weight"), ShardingMode::ROW_PARALLEL);

    auto wo = mock->getWeightForDevice("blk.0.attn_output.weight");
    ASSERT_NE(wo, nullptr);
    EXPECT_EQ(wo->rows(), 128);      // hidden_dim
    EXPECT_EQ(wo->cols(), 4 * 32);   // q_dim
}

TEST(Test__MockWeightManager, AddFFNLayer) {
    // Use smaller dimensions for faster testing
    auto mock = MockWeightManagerBuilder()
        .setStrategy(WeightDistributionStrategy::SHARDED)
        .addFFNLayer(0, 128, 512)  // Small test dimensions
        .build();

    // Check gate projection
    EXPECT_TRUE(mock->hasWeight("blk.0.ffn_gate.weight"));
    EXPECT_EQ(mock->getShardingMode("blk.0.ffn_gate.weight"), ShardingMode::COLUMN_PARALLEL);

    auto gate = mock->getWeightForDevice("blk.0.ffn_gate.weight");
    ASSERT_NE(gate, nullptr);
    EXPECT_EQ(gate->rows(), 512);
    EXPECT_EQ(gate->cols(), 128);

    // Check up projection
    EXPECT_TRUE(mock->hasWeight("blk.0.ffn_up.weight"));
    EXPECT_EQ(mock->getShardingMode("blk.0.ffn_up.weight"), ShardingMode::COLUMN_PARALLEL);

    // Check down projection
    EXPECT_TRUE(mock->hasWeight("blk.0.ffn_down.weight"));
    EXPECT_EQ(mock->getShardingMode("blk.0.ffn_down.weight"), ShardingMode::INPUT_PARALLEL);

    auto down = mock->getWeightForDevice("blk.0.ffn_down.weight");
    ASSERT_NE(down, nullptr);
    EXPECT_EQ(down->rows(), 128);
    EXPECT_EQ(down->cols(), 512);
}

TEST(Test__MockWeightManager, AddNormWeights) {
    auto mock = MockWeightManagerBuilder()
        .setStrategy(WeightDistributionStrategy::SHARDED)
        .addNormWeights(0, 128)  // Small test dimension
        .build();

    // Norms should be replicated and non-GEMM
    EXPECT_TRUE(mock->hasWeight("blk.0.attn_norm.weight"));
    EXPECT_EQ(mock->getShardingMode("blk.0.attn_norm.weight"), ShardingMode::REPLICATE);
    EXPECT_FALSE(mock->isGemmWeight("blk.0.attn_norm.weight"));

    EXPECT_TRUE(mock->hasWeight("blk.0.ffn_norm.weight"));
    EXPECT_EQ(mock->getShardingMode("blk.0.ffn_norm.weight"), ShardingMode::REPLICATE);
    EXPECT_FALSE(mock->isGemmWeight("blk.0.ffn_norm.weight"));
}

TEST(Test__MockWeightManager, AddEmbedding) {
    // Use smaller dimensions for faster testing
    auto mock = MockWeightManagerBuilder()
        .addEmbedding(1000, 128)  // Small vocab for testing
        .build();

    EXPECT_TRUE(mock->hasWeight("token_embd.weight"));
    EXPECT_EQ(mock->getShardingMode("token_embd.weight"), ShardingMode::REPLICATE);
    EXPECT_FALSE(mock->isGemmWeight("token_embd.weight"));

    auto embd = mock->getWeightForDevice("token_embd.weight");
    ASSERT_NE(embd, nullptr);
    EXPECT_EQ(embd->rows(), 1000);
    EXPECT_EQ(embd->cols(), 128);
}

TEST(Test__MockWeightManager, AddLMHead) {
    // Use smaller dimensions for faster testing
    auto mock = MockWeightManagerBuilder()
        .setStrategy(WeightDistributionStrategy::SHARDED)
        .addLMHead(1000, 128)
        .build();

    EXPECT_TRUE(mock->hasWeight("output.weight"));
    EXPECT_EQ(mock->getShardingMode("output.weight"), ShardingMode::COLUMN_PARALLEL);
    EXPECT_TRUE(mock->isGemmWeight("output.weight"));
}

// =============================================================================
// CALL TRACKING TESTS
// =============================================================================

TEST(Test__MockWeightManager, TrackGetWeightCalls) {
    auto mock = MockWeightManagerBuilder()
        .addFP32RandomWeight("weight1", {64, 64})
        .addFP32RandomWeight("weight2", {32, 32})
        .build();

    EXPECT_EQ(mock->getWeightCallCount(), 0);

    mock->getWeightForDevice("weight1");
    EXPECT_EQ(mock->getWeightCallCount(), 1);

    mock->getWeightForDevice("weight2");
    EXPECT_EQ(mock->getWeightCallCount(), 2);

    mock->getWeightForDevice("weight1");  // Same weight again
    EXPECT_EQ(mock->getWeightCallCount(), 3);
}

TEST(Test__MockWeightManager, TrackMissingRequests) {
    auto mock = MockWeightManagerBuilder()
        .addFP32RandomWeight("exists", {64, 64})
        .build();

    mock->getWeightForDevice("exists");
    mock->getWeightForDevice("does_not_exist");
    mock->getWeightForDevice("also_missing");

    auto& missing = mock->missingWeightRequests();
    EXPECT_EQ(missing.size(), 2);
    EXPECT_EQ(missing[0], "does_not_exist");
    EXPECT_EQ(missing[1], "also_missing");
}

TEST(Test__MockWeightManager, ResetCounters) {
    auto mock = MockWeightManagerBuilder()
        .addFP32RandomWeight("exists", {64, 64})
        .build();

    mock->getWeightForDevice("exists");
    mock->getWeightForDevice("missing");

    EXPECT_EQ(mock->getWeightCallCount(), 2);
    EXPECT_EQ(mock->missingWeightRequests().size(), 1);

    mock->resetCounters();

    EXPECT_EQ(mock->getWeightCallCount(), 0);
    EXPECT_EQ(mock->missingWeightRequests().size(), 0);
}

// =============================================================================
// CACHE MANAGEMENT TESTS
// =============================================================================

TEST(Test__MockWeightManager, CacheClear) {
    auto mock = MockWeightManagerBuilder()
        .addFP32RandomWeight("weight1", {64, 64})
        .addFP32RandomWeight("weight2", {32, 32})
        .build();

    EXPECT_EQ(mock->cacheSize(), 2);

    mock->clearCache();
    EXPECT_EQ(mock->cacheSize(), 0);

    // Weights are gone
    EXPECT_FALSE(mock->hasWeight("weight1"));
    EXPECT_FALSE(mock->hasWeight("weight2"));
}

TEST(Test__MockWeightManager, WeightNames) {
    auto mock = MockWeightManagerBuilder()
        .addFP32RandomWeight("alpha", {64, 64})
        .addFP32RandomWeight("beta", {32, 32})
        .addFP32RandomWeight("gamma", {16, 16})
        .build();

    auto names = mock->weightNames();
    EXPECT_EQ(names.size(), 3);

    // Check all names are present (order not guaranteed)
    std::set<std::string> name_set(names.begin(), names.end());
    EXPECT_TRUE(name_set.count("alpha"));
    EXPECT_TRUE(name_set.count("beta"));
    EXPECT_TRUE(name_set.count("gamma"));
}

// =============================================================================
// PRESET TESTS
// =============================================================================

TEST(Test__MockWeightManager, CreateReplicated) {
    auto mock = MockWeightManager::createReplicated();

    EXPECT_EQ(mock->strategy(), WeightDistributionStrategy::REPLICATED);
}

TEST(Test__MockWeightManager, CreateShardedQwen2) {
    auto mock = MockWeightManager::createShardedQwen2();

    EXPECT_EQ(mock->strategy(), WeightDistributionStrategy::SHARDED);
}

// =============================================================================
// INTERFACE COMPLIANCE TESTS
// =============================================================================

TEST(Test__MockWeightManager, ImplementsIWeightManager) {
    // Verify MockWeightManager can be used through IWeightManager interface
    std::shared_ptr<IWeightManager> interface = MockWeightManagerBuilder()
        .setStrategy(WeightDistributionStrategy::SHARDED)
        .addFP32RandomWeight("test", {64, 64})
        .setColumnParallel("test")
        .build();

    EXPECT_EQ(interface->strategy(), WeightDistributionStrategy::SHARDED);
    EXPECT_EQ(interface->cacheSize(), 1);
    EXPECT_TRUE(interface->isWeightSharded("test"));
    EXPECT_EQ(interface->getShardingMode("test"), ShardingMode::COLUMN_PARALLEL);
    EXPECT_TRUE(interface->isGemmWeight("test"));

    auto weight = interface->getWeightForDevice("test");
    ASSERT_NE(weight, nullptr);

    interface->clearCache();
    EXPECT_EQ(interface->cacheSize(), 0);
}

// =============================================================================
// COMPLEX SCENARIO TESTS
// =============================================================================

TEST(Test__MockWeightManager, FullTransformerLayer) {
    // Build a complete transformer layer with all weights (small dimensions for speed)
    auto mock = MockWeightManagerBuilder()
        .setStrategy(WeightDistributionStrategy::SHARDED)
        // Layer 0
        .addAttentionLayer(0, 128, 32, 4, 2)  // Small test dimensions
        .addFFNLayer(0, 128, 512)              // Small FFN
        .addNormWeights(0, 128)
        // Embedding and output
        .addEmbedding(1000, 128)               // Small vocab
        .addLMHead(1000, 128)
        .build();

    // Count weights: 4 attn + 3 ffn + 2 norms + 1 embd + 1 lm_head = 11
    EXPECT_EQ(mock->cacheSize(), 11);

    // Verify sharding is correct
    EXPECT_TRUE(mock->isWeightSharded("blk.0.attn_q.weight"));      // Column-parallel
    EXPECT_TRUE(mock->isWeightSharded("blk.0.attn_output.weight")); // Row-parallel
    EXPECT_TRUE(mock->isWeightSharded("blk.0.ffn_down.weight"));    // Input-parallel
    EXPECT_FALSE(mock->isWeightSharded("blk.0.attn_norm.weight"));  // Replicated
    EXPECT_FALSE(mock->isWeightSharded("token_embd.weight"));       // Replicated
    EXPECT_TRUE(mock->isWeightSharded("output.weight"));            // Column-parallel

    // Verify GEMM classification
    EXPECT_TRUE(mock->isGemmWeight("blk.0.attn_q.weight"));
    EXPECT_TRUE(mock->isGemmWeight("blk.0.ffn_gate.weight"));
    EXPECT_FALSE(mock->isGemmWeight("blk.0.attn_norm.weight"));
    EXPECT_FALSE(mock->isGemmWeight("token_embd.weight"));
}
