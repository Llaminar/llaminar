/**
 * @file Test__WeightManager_PPSafety.cpp
 * @brief Unit tests for WeightManager PP (pipeline parallel) safety fixes
 *
 * Tests verify:
 * 1. isWeightInLayerRange: tied embeddings check for LM head stage
 *    - Model WITH separate output.weight rejects token_embd.weight for LM head stage
 *    - Model WITHOUT output.weight (tied embeddings) allows token_embd.weight
 * 2. tryReleaseReclaimHostRawData: CPU-only host-resident tensor safety
 *    - CPU-only FP32 tensor is NOT released by releaseAllHostWeightData
 *    - Quantized tensor with no device copy is also not released
 * 3. Default WeightPrecision is NATIVE (not CONVERT_TO_FP32)
 *
 * All tests use MockModelLoader (no GGUF file I/O) and test through the public API.
 */

#include <gtest/gtest.h>
#include <memory>

#include "loaders/WeightManager.h"
#include "tensors/Tensors.h"
#include "mocks/MockModelLoader.h"

using namespace llaminar2;
using namespace llaminar2::test;

// ============================================================================
// TestableWeightManager — overrides getPreparedEmbeddingCount for test control
// ============================================================================

class PPSafetyTestableWeightManager : public WeightManager
{
    size_t mock_embedding_count_ = 0;

public:
    PPSafetyTestableWeightManager(IModelLoader &loader,
                                  WeightPrecision precision = WeightPrecision::NATIVE)
        : WeightManager(loader, nullptr, nullptr,
                        WeightDistributionStrategy::LAYER_PARTITIONED,
                        precision)
    {
        // Mark all lifecycle gates open so release decision logic is testable
        markMaterializationComplete();
        markDevicePreparationComplete();
        markGraphMaterializationComplete();
    }

    size_t getPreparedEmbeddingCount() const override
    {
        return mock_embedding_count_;
    }

    void setMockEmbeddingCount(size_t n) { mock_embedding_count_ = n; }
};

// ============================================================================
// Test Fixture
// ============================================================================

class Test__WeightManager_PPSafety : public ::testing::Test
{
protected:
    std::shared_ptr<MockModelLoader> mock_loader_;
};

// ============================================================================
// isWeightInLayerRange: Tied Embeddings Tests (tested via getWeightForDevice)
// ============================================================================

/**
 * @test LM head stage with separate output.weight rejects token_embd.weight
 *
 * When a model has a separate output.weight tensor (non-tied embeddings),
 * the LM head stage should NOT load token_embd.weight — it already has
 * output.weight for the LM head projection.
 */
TEST_F(Test__WeightManager_PPSafety, LMHeadStage_WithSeparateOutputWeight_RejectsEmbedding)
{
    mock_loader_ = MockModelLoader::createMinimal();
    mock_loader_->addQ4_0RandomTensor("token_embd.weight", {256, 64});
    mock_loader_->addQ8_0RandomTensor("output.weight", {256, 64});
    mock_loader_->addFP32RandomTensor("output_norm.weight", {64});

    PPSafetyTestableWeightManager wm(*mock_loader_);
    // LM head stage: layers [12,24), no embedding, has LM head
    wm.setLayerRange(12, 24, false, true);

    // output.weight exists → token_embd.weight should be rejected
    auto emb = wm.getWeightForDevice("token_embd.weight", DeviceId::cpu(), 0);
    EXPECT_EQ(emb, nullptr)
        << "LM head stage with separate output.weight must reject token_embd.weight";

    // output.weight and output_norm.weight should still be loaded
    auto out = wm.getWeightForDevice("output.weight", DeviceId::cpu(), 0);
    EXPECT_NE(out, nullptr) << "LM head stage must load output.weight";

    auto norm = wm.getWeightForDevice("output_norm.weight", DeviceId::cpu(), 0);
    EXPECT_NE(norm, nullptr) << "LM head stage must load output_norm.weight";
}

/**
 * @test LM head stage with tied embeddings (no output.weight) allows token_embd.weight
 *
 * When a model does NOT have a separate output.weight tensor, the model uses
 * tied embeddings where token_embd.weight serves as both embedding and LM head.
 */
TEST_F(Test__WeightManager_PPSafety, LMHeadStage_WithTiedEmbeddings_AllowsEmbedding)
{
    // Build a mock WITHOUT output.weight to simulate tied embeddings
    mock_loader_ = MockModelLoaderBuilder()
                       .setArchitecture("qwen2")
                       .setBlockCount(1)
                       .setEmbeddingLength(128)
                       .setHeadCount(4)
                       .setHeadCountKV(2)
                       .setVocabSize(1000)
                       .setContextLength(512)
                       .addEmbeddingLayer()
                       .addTransformerLayer(0)
                       .build();
    // Add output_norm but NOT output.weight — this simulates tied embeddings
    mock_loader_->addFP32RandomTensor("output_norm.weight", {128});

    PPSafetyTestableWeightManager wm(*mock_loader_);
    wm.setLayerRange(12, 24, false, true);

    // No output.weight → token_embd.weight should be allowed (tied)
    auto emb = wm.getWeightForDevice("token_embd.weight", DeviceId::cpu(), 0);
    EXPECT_NE(emb, nullptr)
        << "LM head stage with tied embeddings must allow token_embd.weight";
}

/**
 * @test Embedding stage allows token_embd.weight regardless of tied status
 */
TEST_F(Test__WeightManager_PPSafety, EmbeddingStage_AlwaysAllowsEmbedding)
{
    mock_loader_ = MockModelLoader::createMinimal();
    mock_loader_->addQ4_0RandomTensor("token_embd.weight", {256, 64});
    mock_loader_->addQ8_0RandomTensor("output.weight", {256, 64});

    PPSafetyTestableWeightManager wm(*mock_loader_);
    // Embedding stage: layers [0,12), has embedding, no LM head
    wm.setLayerRange(0, 12, true, false);

    auto emb = wm.getWeightForDevice("token_embd.weight", DeviceId::cpu(), 0);
    EXPECT_NE(emb, nullptr)
        << "Embedding stage must always allow token_embd.weight";
}

/**
 * @test Middle stage (no embedding, no LM head) rejects token_embd.weight
 */
TEST_F(Test__WeightManager_PPSafety, MiddleStage_RejectsEmbedding)
{
    mock_loader_ = MockModelLoader::createMinimal();
    mock_loader_->addQ4_0RandomTensor("token_embd.weight", {256, 64});

    PPSafetyTestableWeightManager wm(*mock_loader_);
    // Middle stage: layers [6,12), no embedding, no LM head
    wm.setLayerRange(6, 12, false, false);

    auto emb = wm.getWeightForDevice("token_embd.weight", DeviceId::cpu(), 0);
    EXPECT_EQ(emb, nullptr)
        << "Middle stage must reject token_embd.weight";
}

/**
 * @test LM head stage with separate output.weight still loads layer weights in range
 *
 * Verifies the tied embeddings fix doesn't break normal layer weight loading.
 */
TEST_F(Test__WeightManager_PPSafety, LMHeadStage_LoadsLayerWeightsInRange)
{
    mock_loader_ = MockModelLoader::createMinimal();
    mock_loader_->addQ4_0RandomTensor("token_embd.weight", {256, 64});
    mock_loader_->addQ8_0RandomTensor("output.weight", {256, 64});
    mock_loader_->addFP32RandomTensor("output_norm.weight", {64});
    mock_loader_->addQ4_0RandomTensor("blk.12.attn_q.weight", {64, 64});
    mock_loader_->addQ4_0RandomTensor("blk.11.attn_q.weight", {64, 64});

    PPSafetyTestableWeightManager wm(*mock_loader_);
    wm.setLayerRange(12, 24, false, true);

    // Layer 12 is in range [12,24)
    auto l12 = wm.getWeightForDevice("blk.12.attn_q.weight", DeviceId::cpu(), 0);
    EXPECT_NE(l12, nullptr) << "Layer 12 should be in range";

    // Layer 11 is outside range
    auto l11 = wm.getWeightForDevice("blk.11.attn_q.weight", DeviceId::cpu(), 0);
    EXPECT_EQ(l11, nullptr) << "Layer 11 should NOT be in range";
}

// ============================================================================
// Host Release Safety Tests
// ============================================================================

/**
 * @test CPU-only FP32 tensor is NOT released by releaseAllHostWeightData
 *
 * FP32 tensors on CPU are used live by FloatingPointGemmKernel (oneDNN).
 * Releasing their host data would cause a null pointer dereference during inference.
 */
TEST_F(Test__WeightManager_PPSafety, ReleaseAll_RetainsCPUOnlyFP32Tensor)
{
    mock_loader_ = MockModelLoader::createMinimal();
    mock_loader_->addFP32RandomTensor("test_weight", {64, 64});

    PPSafetyTestableWeightManager wm(*mock_loader_);
    wm.setMockEmbeddingCount(0); // No prepared embeddings

    auto tensor = wm.getWeightForDevice("test_weight", DeviceId::cpu(), 0);
    ASSERT_NE(tensor, nullptr);

    // CPU-only tensor: no device copy
    ASSERT_FALSE(tensor->deviceValid());

    // releaseAll should NOT release it — no safe device copy exists
    size_t released = wm.releaseAllHostWeightData();
    EXPECT_EQ(released, 0);
    EXPECT_FALSE(tensor->is_raw_data_released())
        << "CPU-only FP32 tensor must not be released";
    EXPECT_NE(tensor->data(), nullptr)
        << "Data pointer must remain valid after releaseAll";
}

/**
 * @test CPU-only quantized tensor is NOT released when no device copy exists
 */
TEST_F(Test__WeightManager_PPSafety, ReleaseAll_RetainsCPUOnlyQuantizedTensor)
{
    mock_loader_ = MockModelLoader::createMinimal();
    mock_loader_->addQ8_0RandomTensor("test_weight", {64, 64});

    PPSafetyTestableWeightManager wm(*mock_loader_);
    wm.setMockEmbeddingCount(0);

    auto tensor = wm.getWeightForDevice("test_weight", DeviceId::cpu(), 0);
    ASSERT_NE(tensor, nullptr);
    ASSERT_FALSE(tensor->deviceValid());

    size_t released = wm.releaseAllHostWeightData();
    EXPECT_EQ(released, 0);
    EXPECT_FALSE(tensor->is_raw_data_released())
        << "CPU-only quantized tensor must not be released without device copy";
}

/**
 * @test Host-resident tensor IS released when prepared embeddings exist
 *
 * Verifies the safety fix doesn't break the normal release path for
 * tensors that have been explicitly marked as host-resident (mmap-backed).
 */
TEST_F(Test__WeightManager_PPSafety, ReleaseAll_ReleasesHostResidentWithPrepared)
{
    mock_loader_ = MockModelLoader::createMinimal();
    mock_loader_->addQ8_0RandomTensor("test_weight", {64, 64});

    PPSafetyTestableWeightManager wm(*mock_loader_);
    wm.setMockEmbeddingCount(1); // Has prepared embeddings

    auto tensor = wm.getWeightForDevice("test_weight", DeviceId::cpu(), 0);
    ASSERT_NE(tensor, nullptr);
    tensor->setHostResident();

    size_t released = wm.releaseAllHostWeightData();
    EXPECT_GE(released, 1);
    EXPECT_TRUE(tensor->is_raw_data_released())
        << "Host-resident tensor with prepared embeddings should be released";
}

// ============================================================================
// Default Precision Tests
// ============================================================================

/**
 * @test Default WeightPrecision is NATIVE
 *
 * The WeightManager constructor's default parameter should be NATIVE,
 * not CONVERT_TO_FP32. Using FP32 by default causes FloatingPointGemmKernel
 * to be selected, which doesn't support fused SwiGLU and has host memory
 * lifecycle issues.
 */
TEST_F(Test__WeightManager_PPSafety, DefaultPrecision_IsNative)
{
    mock_loader_ = MockModelLoader::createMinimal();
    mock_loader_->addQ4_0RandomTensor("token_embd.weight", {256, 64});

    // Construct with all defaults (no explicit precision)
    WeightManager wm(*mock_loader_);

    // Load a quantized tensor — should remain quantized (NATIVE), not FP32
    auto tensor = wm.getWeightForDevice("token_embd.weight", DeviceId::cpu(), 0);
    ASSERT_NE(tensor, nullptr);

    // With NATIVE precision, Q4_0 tensor stays as Q4_0
    EXPECT_NE(tensor->native_type(), TensorType::FP32)
        << "Default precision should be NATIVE — Q4_0 tensor must not be converted to FP32";
}

/**
 * @test Explicit NATIVE precision keeps quantized format
 */
TEST_F(Test__WeightManager_PPSafety, ExplicitNative_KeepsQuantizedFormat)
{
    mock_loader_ = MockModelLoader::createMinimal();
    mock_loader_->addQ4_0RandomTensor("test_weight", {256, 64});

    WeightManager wm(*mock_loader_, nullptr, nullptr,
                     WeightDistributionStrategy::REPLICATED,
                     WeightPrecision::NATIVE);

    auto tensor = wm.getWeightForDevice("test_weight", DeviceId::cpu(), 0);
    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->native_type(), TensorType::Q4_0)
        << "NATIVE precision must preserve Q4_0 format";
}
