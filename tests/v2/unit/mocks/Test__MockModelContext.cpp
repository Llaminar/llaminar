/**
 * @file Test__MockModelContext.cpp
 * @brief Unit tests for MockModelContext
 *
 * Tests the mock model context's ability to:
 * - Compose MockModelLoader and MockWeightManager
 * - Support model presets (Qwen2-0.5B, Qwen2-7B, Llama-3-8B, Minimal)
 * - Configure model hyperparameters
 * - Add tensors and weights with sharding
 * - Use builder pattern for fluent configuration
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include "mocks/MockModelContext.h"
#include "tensors/Tensors.h"

using namespace llaminar2;
using namespace llaminar2::test;

// =============================================================================
// BASIC CONSTRUCTION TESTS
// =============================================================================

TEST(Test__MockModelContext, DefaultConstruction) {
    MockModelContext ctx;

    // Default values
    EXPECT_EQ(ctx.architecture(), "qwen2");
    EXPECT_EQ(ctx.blockCount(), 1);
    EXPECT_EQ(ctx.embeddingLength(), 128);
    EXPECT_EQ(ctx.headCount(), 4);
    EXPECT_EQ(ctx.headCountKV(), 2);
    EXPECT_EQ(ctx.path(), "mock_model.gguf");

    // Components should exist
    EXPECT_NE(ctx.loader(), nullptr);
    EXPECT_NE(ctx.weightManager(), nullptr);
}

TEST(Test__MockModelContext, BuilderBasicConfiguration) {
    auto ctx = MockModelContextBuilder()
        .setArchitecture("llama")
        .setBlockCount(32)
        .setEmbeddingLength(4096)
        .setHeadCount(32)
        .setHeadCountKV(8)
        .setVocabSize(128000)
        .setContextLength(8192)
        .build();

    EXPECT_EQ(ctx->architecture(), "llama");
    EXPECT_EQ(ctx->blockCount(), 32);
    EXPECT_EQ(ctx->embeddingLength(), 4096);
    EXPECT_EQ(ctx->headCount(), 32);
    EXPECT_EQ(ctx->headCountKV(), 8);
    EXPECT_EQ(ctx->vocabSize(), 128000);
    EXPECT_EQ(ctx->contextLength(), 8192);
}

TEST(Test__MockModelContext, SetPath) {
    auto ctx = MockModelContextBuilder()
        .setPath("/custom/path/model.gguf")
        .build();

    EXPECT_EQ(ctx->path(), "/custom/path/model.gguf");
}

// =============================================================================
// PRESET TESTS
// =============================================================================

TEST(Test__MockModelContext, PresetMinimal) {
    auto ctx = MockModelContext::createMinimal();

    // Presets only set metadata - no tensors by default (for fast tests)
    EXPECT_EQ(ctx->architecture(), "qwen2");
    EXPECT_EQ(ctx->blockCount(), 1);
    EXPECT_EQ(ctx->embeddingLength(), 128);
    EXPECT_EQ(ctx->headCount(), 4);
    EXPECT_EQ(ctx->headCountKV(), 2);
    EXPECT_EQ(ctx->vocabSize(), 1000);
    EXPECT_EQ(ctx->contextLength(), 512);
}

TEST(Test__MockModelContext, PresetMinimalWithTensors) {
    // Use builder pattern when you need actual tensors
    auto ctx = MockModelContextBuilder()
        .usePreset(ModelPreset::MINIMAL)
        .addEmbeddingLayer()
        .addTransformerLayer(0)
        .addOutputLayer()
        .build();

    EXPECT_EQ(ctx->architecture(), "qwen2");
    EXPECT_EQ(ctx->blockCount(), 1);

    // Should have basic weights when explicitly added
    EXPECT_TRUE(ctx->hasTensor("token_embd.weight"));
    EXPECT_TRUE(ctx->hasTensor("blk.0.attn_norm.weight"));
    EXPECT_TRUE(ctx->hasTensor("blk.0.attn_q.weight"));
    EXPECT_TRUE(ctx->hasTensor("output_norm.weight"));
    EXPECT_TRUE(ctx->hasTensor("output.weight"));
}

TEST(Test__MockModelContext, PresetQwen2_05B) {
    auto ctx = MockModelContext::createQwen2_05B();

    EXPECT_EQ(ctx->architecture(), "qwen2");
    EXPECT_EQ(ctx->blockCount(), 24);
    EXPECT_EQ(ctx->embeddingLength(), 896);
    EXPECT_EQ(ctx->headCount(), 14);
    EXPECT_EQ(ctx->headCountKV(), 2);
    EXPECT_EQ(ctx->vocabSize(), 151936);
    EXPECT_EQ(ctx->contextLength(), 32768);
}

TEST(Test__MockModelContext, PresetQwen2_7B) {
    auto ctx = MockModelContext::createQwen2_7B();

    EXPECT_EQ(ctx->architecture(), "qwen2");
    EXPECT_EQ(ctx->blockCount(), 28);
    EXPECT_EQ(ctx->embeddingLength(), 3584);
    EXPECT_EQ(ctx->headCount(), 28);
    EXPECT_EQ(ctx->headCountKV(), 4);
    EXPECT_EQ(ctx->vocabSize(), 152064);
}

TEST(Test__MockModelContext, PresetLlama3_8B) {
    auto ctx = MockModelContext::createLlama3_8B();

    EXPECT_EQ(ctx->architecture(), "llama");
    EXPECT_EQ(ctx->blockCount(), 32);
    EXPECT_EQ(ctx->embeddingLength(), 4096);
    EXPECT_EQ(ctx->headCount(), 32);
    EXPECT_EQ(ctx->headCountKV(), 8);
    EXPECT_EQ(ctx->vocabSize(), 128256);
}

TEST(Test__MockModelContext, PresetFromEnum) {
    auto ctx = MockModelContext::createFromPreset(ModelPreset::QWEN2_05B);

    EXPECT_EQ(ctx->architecture(), "qwen2");
    EXPECT_EQ(ctx->blockCount(), 24);
    EXPECT_EQ(ctx->embeddingLength(), 896);
}

// =============================================================================
// COMPONENT ACCESS TESTS
// =============================================================================

TEST(Test__MockModelContext, LoaderInterfaceAccess) {
    auto ctx = MockModelContext::createMinimal();

    auto loader = ctx->loader();
    ASSERT_NE(loader, nullptr);

    // Should be able to use interface methods
    EXPECT_TRUE(loader->isLoaded());
    EXPECT_EQ(loader->architecture(), "qwen2");
    EXPECT_EQ(loader->blockCount(), 1);
}

TEST(Test__MockModelContext, WeightManagerInterfaceAccess) {
    auto ctx = MockModelContextBuilder()
        .usePreset(ModelPreset::MINIMAL)
        .setStrategy(WeightDistributionStrategy::SHARDED)
        .addEmbeddingLayer()
        .build();

    auto wm = ctx->weightManager();
    ASSERT_NE(wm, nullptr);

    // Should be able to use interface methods
    EXPECT_EQ(wm->strategy(), WeightDistributionStrategy::SHARDED);
    EXPECT_GT(wm->cacheSize(), 0);
}

TEST(Test__MockModelContext, MockComponentAccess) {
    auto ctx = MockModelContext::createMinimal();

    // Access underlying mocks for test configuration
    MockModelLoader& loader = ctx->mockLoader();
    MockWeightManager& wm = ctx->mockWeightManager();

    // Can configure via mock APIs
    loader.setIntParam("custom_param", 42);
    wm.setStrategy(WeightDistributionStrategy::SHARDED);

    EXPECT_EQ(loader.getInt("custom_param"), 42);
    EXPECT_EQ(wm.strategy(), WeightDistributionStrategy::SHARDED);
}

// =============================================================================
// TENSOR/WEIGHT ADDITION TESTS
// =============================================================================

TEST(Test__MockModelContext, AddFP32RandomTensor) {
    auto ctx = MockModelContextBuilder()
        .addFP32RandomTensor("custom_weight", {64, 128}, -1.0f, 1.0f, 42)
        .build();

    // Should exist in loader
    EXPECT_TRUE(ctx->hasTensor("custom_weight"));

    // Should be retrievable as weight
    auto weight = ctx->getWeightForDevice("custom_weight");
    ASSERT_NE(weight, nullptr);
    EXPECT_EQ(weight->rows(), 64);
    EXPECT_EQ(weight->cols(), 128);

    // Verify it's FP32
    auto* fp32 = dynamic_cast<FP32Tensor*>(weight.get());
    ASSERT_NE(fp32, nullptr);
}

TEST(Test__MockModelContext, AddFP32ZerosTensor) {
    auto ctx = MockModelContextBuilder()
        .addFP32ZerosTensor("zeros", {32, 64})
        .build();

    auto weight = ctx->getWeightForDevice("zeros");
    ASSERT_NE(weight, nullptr);

    auto* fp32 = dynamic_cast<FP32Tensor*>(weight.get());
    ASSERT_NE(fp32, nullptr);

    // Verify all zeros
    const float* data = fp32->data();
    for (size_t i = 0; i < 32 * 64; ++i) {
        EXPECT_EQ(data[i], 0.0f);
    }
}

TEST(Test__MockModelContext, AddFP32OnesTensor) {
    auto ctx = MockModelContextBuilder()
        .addFP32OnesTensor("ones", {16, 32})
        .build();

    auto weight = ctx->getWeightForDevice("ones");
    ASSERT_NE(weight, nullptr);

    auto* fp32 = dynamic_cast<FP32Tensor*>(weight.get());
    ASSERT_NE(fp32, nullptr);

    // Verify all ones
    const float* data = fp32->data();
    for (size_t i = 0; i < 16 * 32; ++i) {
        EXPECT_EQ(data[i], 1.0f);
    }
}

TEST(Test__MockModelContext, AddQ4_0RandomTensor) {
    auto ctx = MockModelContextBuilder()
        .addQ4_0RandomTensor("q4_weight", {256, 256})
        .build();

    EXPECT_TRUE(ctx->hasTensor("q4_weight"));
    auto weight = ctx->getWeightForDevice("q4_weight");
    ASSERT_NE(weight, nullptr);

    // Should be Q4_0 tensor
    auto* q4 = dynamic_cast<Q4_0Tensor*>(weight.get());
    ASSERT_NE(q4, nullptr);
}

TEST(Test__MockModelContext, AddQ8_0RandomTensor) {
    auto ctx = MockModelContextBuilder()
        .addQ8_0RandomTensor("q8_weight", {128, 128})
        .build();

    EXPECT_TRUE(ctx->hasTensor("q8_weight"));
    auto weight = ctx->getWeightForDevice("q8_weight");
    ASSERT_NE(weight, nullptr);

    auto* q8 = dynamic_cast<Q8_0Tensor*>(weight.get());
    ASSERT_NE(q8, nullptr);
}

TEST(Test__MockModelContext, AddPrebuiltTensor) {
    // Create a custom tensor
    auto custom = std::make_shared<FP32Tensor>(std::vector<size_t>{100, 200});
    float* data = custom->mutable_data();
    for (size_t i = 0; i < 100 * 200; ++i) {
        data[i] = static_cast<float>(i) * 0.001f;
    }

    auto ctx = MockModelContextBuilder()
        .addTensor("custom_tensor", custom)
        .build();

    auto weight = ctx->getWeightForDevice("custom_tensor");
    ASSERT_NE(weight, nullptr);
    EXPECT_EQ(weight->rows(), 100);
    EXPECT_EQ(weight->cols(), 200);
}

// =============================================================================
// SHARDING CONFIGURATION TESTS
// =============================================================================

TEST(Test__MockModelContext, ShardingModeConfiguration) {
    auto ctx = MockModelContextBuilder()
        .setStrategy(WeightDistributionStrategy::SHARDED)
        .addFP32RandomTensor("attn_q", {896, 896})
        .setColumnParallel("attn_q")
        .addFP32RandomTensor("attn_output", {896, 896})
        .setRowParallel("attn_output")
        .addFP32RandomTensor("ffn_down", {896, 4864})
        .setInputParallel("ffn_down")
        .addFP32RandomTensor("norm", {896})
        .setReplicated("norm")
        .setNonGemm("norm")
        .build();

    auto wm = ctx->weightManager();

    EXPECT_TRUE(wm->isWeightSharded("attn_q"));
    EXPECT_EQ(wm->getShardingMode("attn_q"), ShardingMode::COLUMN_PARALLEL);

    EXPECT_TRUE(wm->isWeightSharded("attn_output"));
    EXPECT_EQ(wm->getShardingMode("attn_output"), ShardingMode::ROW_PARALLEL);

    EXPECT_TRUE(wm->isWeightSharded("ffn_down"));
    EXPECT_EQ(wm->getShardingMode("ffn_down"), ShardingMode::INPUT_PARALLEL);

    EXPECT_FALSE(wm->isWeightSharded("norm"));
    EXPECT_FALSE(wm->isGemmWeight("norm"));
}

TEST(Test__MockModelContext, AddShardedWeight) {
    auto tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{64, 64});

    auto ctx = MockModelContextBuilder()
        .setStrategy(WeightDistributionStrategy::SHARDED)
        .addShardedWeight("sharded_weight", tensor, ShardingMode::COLUMN_PARALLEL)
        .build();

    auto wm = ctx->weightManager();
    EXPECT_TRUE(wm->isWeightSharded("sharded_weight"));
    EXPECT_EQ(wm->getShardingMode("sharded_weight"), ShardingMode::COLUMN_PARALLEL);
}

// =============================================================================
// LAYER PRESET TESTS
// =============================================================================

TEST(Test__MockModelContext, AddEmbeddingLayer) {
    auto ctx = MockModelContextBuilder()
        .usePreset(ModelPreset::MINIMAL)
        .addEmbeddingLayer()
        .build();

    EXPECT_TRUE(ctx->hasTensor("token_embd.weight"));
    auto embd = ctx->getWeightForDevice("token_embd.weight");
    ASSERT_NE(embd, nullptr);

    // Minimal preset: vocab=1000, embedding=128
    EXPECT_EQ(embd->rows(), 1000);
    EXPECT_EQ(embd->cols(), 128);

    // Embeddings should be replicated and non-GEMM
    auto wm = ctx->weightManager();
    EXPECT_FALSE(wm->isWeightSharded("token_embd.weight"));
    EXPECT_FALSE(wm->isGemmWeight("token_embd.weight"));
}

TEST(Test__MockModelContext, AddTransformerLayer) {
    auto ctx = MockModelContextBuilder()
        .usePreset(ModelPreset::MINIMAL)
        .addTransformerLayer(0)
        .build();

    // Check attention weights
    EXPECT_TRUE(ctx->hasTensor("blk.0.attn_norm.weight"));
    EXPECT_TRUE(ctx->hasTensor("blk.0.attn_q.weight"));
    EXPECT_TRUE(ctx->hasTensor("blk.0.attn_k.weight"));
    EXPECT_TRUE(ctx->hasTensor("blk.0.attn_v.weight"));
    EXPECT_TRUE(ctx->hasTensor("blk.0.attn_output.weight"));

    // Check FFN weights
    EXPECT_TRUE(ctx->hasTensor("blk.0.ffn_norm.weight"));
    EXPECT_TRUE(ctx->hasTensor("blk.0.ffn_gate.weight"));
    EXPECT_TRUE(ctx->hasTensor("blk.0.ffn_up.weight"));
    EXPECT_TRUE(ctx->hasTensor("blk.0.ffn_down.weight"));

    // Check sharding modes
    auto wm = ctx->weightManager();
    EXPECT_EQ(wm->getShardingMode("blk.0.attn_q.weight"), ShardingMode::COLUMN_PARALLEL);
    EXPECT_EQ(wm->getShardingMode("blk.0.attn_output.weight"), ShardingMode::ROW_PARALLEL);
    EXPECT_EQ(wm->getShardingMode("blk.0.ffn_down.weight"), ShardingMode::INPUT_PARALLEL);
    EXPECT_FALSE(wm->isGemmWeight("blk.0.attn_norm.weight"));
}

TEST(Test__MockModelContext, AddOutputLayer) {
    auto ctx = MockModelContextBuilder()
        .usePreset(ModelPreset::MINIMAL)
        .addOutputLayer()
        .build();

    EXPECT_TRUE(ctx->hasTensor("output_norm.weight"));
    EXPECT_TRUE(ctx->hasTensor("output.weight"));

    auto wm = ctx->weightManager();
    EXPECT_FALSE(wm->isGemmWeight("output_norm.weight"));
    EXPECT_EQ(wm->getShardingMode("output.weight"), ShardingMode::COLUMN_PARALLEL);
}

TEST(Test__MockModelContext, AddAllLayers) {
    auto ctx = MockModelContextBuilder()
        .setBlockCount(3)
        .setEmbeddingLength(256)
        .setHeadCount(4)
        .setHeadCountKV(2)
        .setVocabSize(1000)
        .addAllLayers()
        .build();

    // Embedding layer
    EXPECT_TRUE(ctx->hasTensor("token_embd.weight"));

    // All 3 transformer layers
    for (int i = 0; i < 3; ++i) {
        std::string prefix = "blk." + std::to_string(i) + ".";
        EXPECT_TRUE(ctx->hasTensor(prefix + "attn_norm.weight"));
        EXPECT_TRUE(ctx->hasTensor(prefix + "attn_q.weight"));
        EXPECT_TRUE(ctx->hasTensor(prefix + "ffn_down.weight"));
    }

    // Output layer
    EXPECT_TRUE(ctx->hasTensor("output_norm.weight"));
    EXPECT_TRUE(ctx->hasTensor("output.weight"));
}

// =============================================================================
// CALL TRACKING TESTS
// =============================================================================

TEST(Test__MockModelContext, TrackGetWeightCalls) {
    auto ctx = MockModelContext::createMinimal();

    EXPECT_EQ(ctx->getWeightCallCount(), 0);

    ctx->getWeightForDevice("token_embd.weight");
    EXPECT_EQ(ctx->getWeightCallCount(), 1);

    ctx->getWeightForDevice("blk.0.attn_q.weight");
    ctx->getWeightForDevice("output.weight");
    EXPECT_EQ(ctx->getWeightCallCount(), 3);

    ctx->resetCounters();
    EXPECT_EQ(ctx->getWeightCallCount(), 0);
}

TEST(Test__MockModelContext, TrackLoadTensorCalls) {
    auto ctx = MockModelContext::createMinimal();

    EXPECT_EQ(ctx->loadTensorCallCount(), 0);

    // Load via loader interface
    ctx->loader()->loadTensor("token_embd.weight");
    EXPECT_EQ(ctx->loadTensorCallCount(), 1);

    ctx->resetCounters();
    EXPECT_EQ(ctx->loadTensorCallCount(), 0);
}

// =============================================================================
// INTEGRATION TESTS
// =============================================================================

TEST(Test__MockModelContext, FullPipelineSetup) {
    // Simulate setting up a complete inference context
    // Use MINIMAL preset to avoid allocating huge vocab-sized tensors
    auto ctx = MockModelContextBuilder()
        .usePreset(ModelPreset::MINIMAL)  // vocab=1000, dim=128 - fast
        .setStrategy(WeightDistributionStrategy::SHARDED)
        .addEmbeddingLayer()
        .addTransformerLayer(0)
        .addOutputLayer()
        .build();

    // Verify configuration
    EXPECT_EQ(ctx->architecture(), "qwen2");
    EXPECT_EQ(ctx->blockCount(), 1);
    EXPECT_EQ(ctx->embeddingLength(), 128);

    // Verify weight dimensions
    auto embd = ctx->getWeightForDevice("token_embd.weight");
    ASSERT_NE(embd, nullptr);
    EXPECT_EQ(embd->rows(), 1000);   // vocab_size (MINIMAL)
    EXPECT_EQ(embd->cols(), 128);    // embedding_length (MINIMAL)

    auto attn_q = ctx->getWeightForDevice("blk.0.attn_q.weight");
    ASSERT_NE(attn_q, nullptr);
    EXPECT_EQ(attn_q->rows(), 128);
    EXPECT_EQ(attn_q->cols(), 128);

    // Verify sharding
    auto wm = ctx->weightManager();
    EXPECT_TRUE(wm->isWeightSharded("blk.0.attn_q.weight"));
}

TEST(Test__MockModelContext, ComposedMocksAreConsistent) {
    auto ctx = MockModelContextBuilder()
        .setBlockCount(8)
        .setEmbeddingLength(512)
        .addFP32RandomTensor("test_weight", {512, 512})
        .build();

    // Both loader and weight manager should have the same tensor
    EXPECT_TRUE(ctx->loader()->hasTensor("test_weight"));

    auto from_loader = ctx->loader()->loadTensor("test_weight");
    auto from_wm = ctx->weightManager()->getWeightForDevice("test_weight");

    ASSERT_NE(from_loader, nullptr);
    ASSERT_NE(from_wm, nullptr);
    EXPECT_EQ(from_loader->rows(), from_wm->rows());
    EXPECT_EQ(from_loader->cols(), from_wm->cols());
}

TEST(Test__MockModelContext, CanModifyAfterBuild) {
    auto ctx = MockModelContextBuilder()
        .usePreset(ModelPreset::MINIMAL)
        .build();

    // Initial state
    EXPECT_EQ(ctx->blockCount(), 1);
    EXPECT_FALSE(ctx->hasTensor("new_weight"));

    // Modify via mock accessors
    ctx->setBlockCount(4);
    ctx->mockWeightManager().addFP32RandomWeight("new_weight", {64, 64});
    ctx->mockLoader().addFP32RandomTensor("new_weight", {64, 64});

    // Verify modifications
    EXPECT_EQ(ctx->blockCount(), 4);
    EXPECT_TRUE(ctx->hasTensor("new_weight"));
    EXPECT_NE(ctx->getWeightForDevice("new_weight"), nullptr);
}

// =============================================================================
// HYPERPARAMETER TESTS (RoPE, RMSNorm)
// =============================================================================

TEST(Test__MockModelContext, RopeAndNormConfig) {
    auto ctx = MockModelContextBuilder()
        .setRopeTheta(500000.0f)
        .setRmsNormEps(1e-5f)
        .build();

    // Access via loader interface
    auto loader = ctx->loader();
    EXPECT_FLOAT_EQ(loader->ropeTheta(), 500000.0f);
    EXPECT_FLOAT_EQ(loader->rmsNormEps(), 1e-5f);
}

TEST(Test__MockModelContext, PresetRopeConfig) {
    // Qwen2 uses rope_theta = 1000000
    auto qwen2 = MockModelContext::createQwen2_05B();
    EXPECT_FLOAT_EQ(qwen2->loader()->ropeTheta(), 1000000.0f);

    // Llama3 uses rope_theta = 500000
    auto llama = MockModelContext::createLlama3_8B();
    EXPECT_FLOAT_EQ(llama->loader()->ropeTheta(), 500000.0f);
}
