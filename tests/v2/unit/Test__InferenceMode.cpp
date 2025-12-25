/**
 * @file Test__InferenceMode.cpp
 * @brief Unit tests for InferenceMode
 * @author GitHub Copilot
 * @date December 2025
 *
 * Tests the centralized inference mode context that replaces scattered
 * use_hybrid_rope checks throughout the codebase.
 */

#include <gtest/gtest.h>
#include "../../../src/v2/execution/InferenceMode.h"
#include "../../../src/v2/models/qwen/Qwen2Graph.h"
#include "../../../src/v2/tensors/Tensors.h"

using namespace llaminar2;

// =============================================================================
// Test Fixture
// =============================================================================

class InferenceModeTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Create mock buffers for validation tests
        Q = std::make_unique<FP32Tensor>(std::vector<size_t>{9, 896});
        K = std::make_unique<FP32Tensor>(std::vector<size_t>{9, 128});
        V = std::make_unique<FP32Tensor>(std::vector<size_t>{9, 128});
        attn_output = std::make_unique<FP32Tensor>(std::vector<size_t>{9, 896});
        Q_rope = std::make_unique<FP32Tensor>(std::vector<size_t>{9, 896});
        K_rope = std::make_unique<FP32Tensor>(std::vector<size_t>{9, 128});
        V_dequant = std::make_unique<FP32Tensor>(std::vector<size_t>{9, 128});
    }

    std::unique_ptr<FP32Tensor> Q, K, V, attn_output;
    std::unique_ptr<FP32Tensor> Q_rope, K_rope, V_dequant;

    Qwen2ActivationBuffers makeBaseBuffers()
    {
        Qwen2ActivationBuffers buffers;
        buffers.Q = Q.get();
        buffers.K = K.get();
        buffers.V = V.get();
        buffers.attn_output = attn_output.get();
        return buffers;
    }

    Qwen2ActivationBuffers makeHybridBuffers()
    {
        auto buffers = makeBaseBuffers();
        buffers.Q_rope = Q_rope.get();
        buffers.K_rope = K_rope.get();
        buffers.V_dequant = V_dequant.get();
        return buffers;
    }
};

// =============================================================================
// Mode Identification Tests
// =============================================================================

TEST_F(InferenceModeTest, FP32Mode_Identification)
{
    InferenceMode mode(ActivationPrecision::FP32);

    EXPECT_TRUE(mode.isFP32());
    EXPECT_FALSE(mode.isQ8_1());
    EXPECT_FALSE(mode.isHybrid());
    EXPECT_EQ(mode.precision(), ActivationPrecision::FP32);
    EXPECT_EQ(mode.name(), "FP32");
}

TEST_F(InferenceModeTest, Q8_1Mode_Identification)
{
    InferenceMode mode(ActivationPrecision::Q8_1);

    EXPECT_FALSE(mode.isFP32());
    EXPECT_TRUE(mode.isQ8_1());
    EXPECT_FALSE(mode.isHybrid());
    EXPECT_EQ(mode.precision(), ActivationPrecision::Q8_1);
    EXPECT_EQ(mode.name(), "Q8_1");
}

TEST_F(InferenceModeTest, HybridMode_Identification)
{
    InferenceMode mode(ActivationPrecision::Hybrid);

    EXPECT_FALSE(mode.isFP32());
    EXPECT_FALSE(mode.isQ8_1());
    EXPECT_TRUE(mode.isHybrid());
    EXPECT_EQ(mode.precision(), ActivationPrecision::Hybrid);
    EXPECT_EQ(mode.name(), "Hybrid");
}

TEST_F(InferenceModeTest, FactoryMethods)
{
    EXPECT_TRUE(InferenceMode::FP32().isFP32());
    EXPECT_TRUE(InferenceMode::Q8_1().isQ8_1());
    EXPECT_TRUE(InferenceMode::Hybrid().isHybrid());
}

// =============================================================================
// Buffer Requirements Tests
// =============================================================================

TEST_F(InferenceModeTest, FP32Mode_NoExtraBuffers)
{
    InferenceMode mode = InferenceMode::FP32();

    EXPECT_FALSE(mode.needsQRope());
    EXPECT_FALSE(mode.needsKRope());
    EXPECT_FALSE(mode.needsVDequant());

    auto extras = mode.extraRequiredBuffers();
    EXPECT_TRUE(extras.empty());
}

TEST_F(InferenceModeTest, Q8_1Mode_NoExtraBuffers)
{
    InferenceMode mode = InferenceMode::Q8_1();

    EXPECT_FALSE(mode.needsQRope());
    EXPECT_FALSE(mode.needsKRope());
    EXPECT_FALSE(mode.needsVDequant());

    auto extras = mode.extraRequiredBuffers();
    EXPECT_TRUE(extras.empty());
}

TEST_F(InferenceModeTest, HybridMode_RequiresExtraBuffers)
{
    InferenceMode mode = InferenceMode::Hybrid();

    EXPECT_TRUE(mode.needsQRope());
    EXPECT_TRUE(mode.needsKRope());
    EXPECT_TRUE(mode.needsVDequant());

    auto extras = mode.extraRequiredBuffers();
    ASSERT_EQ(extras.size(), 3u);
    EXPECT_EQ(extras[0], "Q_rope");
    EXPECT_EQ(extras[1], "K_rope");
    EXPECT_EQ(extras[2], "V_dequant");
}

// =============================================================================
// Attention Strategy Tests
// =============================================================================

TEST_F(InferenceModeTest, FP32Mode_UsesDecomposedAttention)
{
    InferenceMode mode = InferenceMode::FP32();

    EXPECT_FALSE(mode.usesFusedQ8Attention());
    EXPECT_TRUE(mode.usesDecomposedFP32Attention());
    EXPECT_EQ(mode.attentionContextPrecision(), ActivationPrecision::FP32);
}

TEST_F(InferenceModeTest, Q8_1Mode_UsesFusedAttention)
{
    InferenceMode mode = InferenceMode::Q8_1();

    EXPECT_TRUE(mode.usesFusedQ8Attention());
    EXPECT_FALSE(mode.usesDecomposedFP32Attention());
    EXPECT_EQ(mode.attentionContextPrecision(), ActivationPrecision::FP32);
}

TEST_F(InferenceModeTest, HybridMode_UsesDecomposedAttention)
{
    InferenceMode mode = InferenceMode::Hybrid();

    EXPECT_FALSE(mode.usesFusedQ8Attention());
    EXPECT_TRUE(mode.usesDecomposedFP32Attention());
    EXPECT_EQ(mode.attentionContextPrecision(), ActivationPrecision::FP32);
}

// =============================================================================
// Buffer Validation Tests
// =============================================================================

TEST_F(InferenceModeTest, FP32Mode_ValidatesWithBaseBuffers)
{
    InferenceMode mode = InferenceMode::FP32();
    auto buffers = makeBaseBuffers();

    auto result = mode.validateBuffers(buffers);

    EXPECT_TRUE(result.valid);
    EXPECT_TRUE(result.missing_buffers.empty());
    EXPECT_TRUE(result.error_message.empty());
}

TEST_F(InferenceModeTest, Q8_1Mode_ValidatesWithBaseBuffers)
{
    InferenceMode mode = InferenceMode::Q8_1();
    auto buffers = makeBaseBuffers();

    auto result = mode.validateBuffers(buffers);

    EXPECT_TRUE(result.valid);
    EXPECT_TRUE(result.missing_buffers.empty());
}

TEST_F(InferenceModeTest, HybridMode_FailsWithBaseBuffers)
{
    InferenceMode mode = InferenceMode::Hybrid();
    auto buffers = makeBaseBuffers();

    auto result = mode.validateBuffers(buffers);

    EXPECT_FALSE(result.valid);
    ASSERT_EQ(result.missing_buffers.size(), 3u);
    EXPECT_EQ(result.missing_buffers[0], "Q_rope");
    EXPECT_EQ(result.missing_buffers[1], "K_rope");
    EXPECT_EQ(result.missing_buffers[2], "V_dequant");
    EXPECT_FALSE(result.error_message.empty());
}

TEST_F(InferenceModeTest, HybridMode_ValidatesWithHybridBuffers)
{
    InferenceMode mode = InferenceMode::Hybrid();
    auto buffers = makeHybridBuffers();

    auto result = mode.validateBuffers(buffers);

    EXPECT_TRUE(result.valid);
    EXPECT_TRUE(result.missing_buffers.empty());
}

TEST_F(InferenceModeTest, Validation_DetectsMissingCoreBuffers)
{
    InferenceMode mode = InferenceMode::FP32();
    Qwen2ActivationBuffers buffers; // All null

    auto result = mode.validateBuffers(buffers);

    EXPECT_FALSE(result.valid);
    EXPECT_EQ(result.missing_buffers.size(), 4u); // Q, K, V, attn_output
}

TEST_F(InferenceModeTest, ValidationResult_BoolConversion)
{
    InferenceMode mode = InferenceMode::FP32();

    auto valid_buffers = makeBaseBuffers();
    auto valid_result = mode.validateBuffers(valid_buffers);
    EXPECT_TRUE(static_cast<bool>(valid_result));

    Qwen2ActivationBuffers empty_buffers;
    auto invalid_result = mode.validateBuffers(empty_buffers);
    EXPECT_FALSE(static_cast<bool>(invalid_result));
}

// =============================================================================
// isHybridModeActive Helper Tests
// =============================================================================

TEST_F(InferenceModeTest, isHybridModeActive_TrueWhenHybridWithBuffers)
{
    InferenceMode mode = InferenceMode::Hybrid();
    auto buffers = makeHybridBuffers();

    EXPECT_TRUE(isHybridModeActive(mode, buffers));
}

TEST_F(InferenceModeTest, isHybridModeActive_FalseWhenHybridWithoutBuffers)
{
    InferenceMode mode = InferenceMode::Hybrid();
    auto buffers = makeBaseBuffers(); // Missing Q_rope, K_rope

    EXPECT_FALSE(isHybridModeActive(mode, buffers));
}

TEST_F(InferenceModeTest, isHybridModeActive_FalseWhenFP32)
{
    InferenceMode mode = InferenceMode::FP32();
    auto buffers = makeHybridBuffers();

    EXPECT_FALSE(isHybridModeActive(mode, buffers));
}

TEST_F(InferenceModeTest, isHybridModeActive_FalseWhenQ8_1)
{
    InferenceMode mode = InferenceMode::Q8_1();
    auto buffers = makeHybridBuffers();

    EXPECT_FALSE(isHybridModeActive(mode, buffers));
}

TEST_F(InferenceModeTest, isHybridModeActive_PartialBuffers)
{
    InferenceMode mode = InferenceMode::Hybrid();

    // Only Q_rope, missing K_rope
    auto buffers = makeBaseBuffers();
    buffers.Q_rope = Q_rope.get();

    EXPECT_FALSE(isHybridModeActive(mode, buffers));

    // Add K_rope -> now should be active (V_dequant not required for RoPE check)
    buffers.K_rope = K_rope.get();

    EXPECT_TRUE(isHybridModeActive(mode, buffers));
}
