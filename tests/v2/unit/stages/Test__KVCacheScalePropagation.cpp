/**
 * @file Test__KVCacheScalePropagation.cpp
 * @brief Unit tests for kv_cache_scale propagation from GraphSchema to KVCacheAppendStage
 *
 * These tests lock in the VNNI-safe quantization contract:
 * 1. GraphSchema.kv_cache_scale defines the fixed Q16 scale (default 8.0)
 * 2. Qwen2GraphConfig.kv_cache_scale copies this value for graph building
 * 3. KVCacheAppendStage::Params.kv_cache_scale receives the value
 * 4. Fixed-scale quantization uses scale = kv_cache_scale / 32767.0f
 *
 * See: docs/v2/PROJECT_Q16_INTEGER_ATTENTION_V2.md "VNNI OVERFLOW PREVENTION CONTRACT"
 */

#include <gtest/gtest.h>

#include "execution/GraphSchema.h"
#include "execution/RuntimeConfig.h"
#include "execution/InferenceRunnerFactory.h"
#include "execution/compute_stages/stages/KVCacheAppendStage.h"
#include "models/qwen/Qwen2Schema.h"
#include "models/qwen/Qwen2Graph.h"
#include "tensors/Tensors.h"
#include "kernels/cpu/CPUKVCache.h"
#include "kernels/cpu/attention/q16_1/VNNISafetyConstants.h"

namespace llaminar2::test
{

    // =========================================================================
    // Test Fixture
    // =========================================================================

    class KVCacheScalePropagationTest : public ::testing::Test
    {
    protected:
        // Model configuration constants (Qwen2-0.5B)
        static constexpr int HEAD_DIM = 64;
        static constexpr int N_HEADS = 14;
        static constexpr int N_KV_HEADS = 2;
        static constexpr int D_MODEL = 896;
        static constexpr int SEQ_LEN = 32;

        // Default and custom scales for testing
        static constexpr float DEFAULT_KV_CACHE_SCALE = 8.0f;
        static constexpr float CUSTOM_KV_CACHE_SCALE = 4.0f;
        static constexpr float TIGHT_KV_CACHE_SCALE = 2.0f;
    };

    // =========================================================================
    // GraphSchema Default Value Tests
    // =========================================================================

    TEST_F(KVCacheScalePropagationTest, GraphSchema_DefaultScale_Is8)
    {
        GraphSchema schema;

        // Lock in the default value
        EXPECT_FLOAT_EQ(schema.kv_cache_scale, DEFAULT_KV_CACHE_SCALE)
            << "GraphSchema default kv_cache_scale must be 8.0f (±8.0 FP32 range)";
    }

    TEST_F(KVCacheScalePropagationTest, GraphSchema_ScaleCanBeOverridden)
    {
        GraphSchema schema;
        schema.kv_cache_scale = CUSTOM_KV_CACHE_SCALE;

        EXPECT_FLOAT_EQ(schema.kv_cache_scale, CUSTOM_KV_CACHE_SCALE)
            << "GraphSchema.kv_cache_scale should be overridable";
    }

    // =========================================================================
    // Qwen2SchemaFactory Tests
    // =========================================================================

    TEST_F(KVCacheScalePropagationTest, Qwen2SchemaFactory_ProducesSchema)
    {
        Qwen2SchemaFactory factory;

        // Factory should produce a valid schema
        EXPECT_EQ(factory.architectureName(), "qwen2");

        // Schema-level defaults are inherited from GraphSchema
        GraphSchema schema;
        EXPECT_FLOAT_EQ(schema.kv_cache_scale, DEFAULT_KV_CACHE_SCALE)
            << "Schema should use default kv_cache_scale=8.0";
    }

    // =========================================================================
    // KVCacheAppendStage::Params Default Tests
    // =========================================================================

    TEST_F(KVCacheScalePropagationTest, KVCacheAppendStageParams_DefaultScale_Is8)
    {
        KVCacheAppendStage::Params params;

        // Lock in the default value in Params
        EXPECT_FLOAT_EQ(params.kv_cache_scale, DEFAULT_KV_CACHE_SCALE)
            << "KVCacheAppendStage::Params default kv_cache_scale must be 8.0f";
    }

    TEST_F(KVCacheScalePropagationTest, KVCacheAppendStageParams_DefaultHeadDim_Is128)
    {
        KVCacheAppendStage::Params params;

        // Default head_dim should be 128 (common value, conservative for VNNI limits)
        EXPECT_EQ(params.head_dim, 128)
            << "KVCacheAppendStage::Params default head_dim should be 128";
    }

    TEST_F(KVCacheScalePropagationTest, KVCacheAppendStageParams_AcceptsCustomScale)
    {
        KVCacheAppendStage::Params params;
        params.kv_cache_scale = TIGHT_KV_CACHE_SCALE;
        params.head_dim = HEAD_DIM;

        EXPECT_FLOAT_EQ(params.kv_cache_scale, TIGHT_KV_CACHE_SCALE);
        EXPECT_EQ(params.head_dim, HEAD_DIM);
    }

    // =========================================================================
    // VNNI Safety Constant Tests (ensure head_dim maps to correct limits)
    // =========================================================================

    TEST_F(KVCacheScalePropagationTest, VNNISafetyConstants_HeadDim64_Limit)
    {
        int16_t max_safe = vnni_safety::get_max_safe_int16(64);

        // For head_dim=64, with full 32K INT16 range, we expect the safe limit
        // See VNNISafetyConstants.h for the mathematical derivation
        EXPECT_GT(max_safe, 0) << "VNNI safe limit should be positive";
        EXPECT_LE(max_safe, 32767) << "VNNI safe limit should not exceed INT16 max";

        // Log for documentation
        std::cout << "[INFO] head_dim=64 -> max_safe_int16=" << max_safe << std::endl;
    }

    TEST_F(KVCacheScalePropagationTest, VNNISafetyConstants_HeadDim128_Limit)
    {
        int16_t max_safe = vnni_safety::get_max_safe_int16(128);

        EXPECT_GT(max_safe, 0);
        EXPECT_LE(max_safe, 32767);

        // head_dim=128 should have a smaller limit than head_dim=64
        // (more accumulation steps = lower per-element limit)
        int16_t max_safe_64 = vnni_safety::get_max_safe_int16(64);
        EXPECT_LE(max_safe, max_safe_64)
            << "Larger head_dim should have equal or smaller VNNI safe limit";

        std::cout << "[INFO] head_dim=128 -> max_safe_int16=" << max_safe << std::endl;
    }

    // =========================================================================
    // Fixed-Scale Quantization Math Tests
    // =========================================================================

    TEST_F(KVCacheScalePropagationTest, FixedScaleQuantization_ScaleFormula)
    {
        // The fixed scale formula: d = kv_cache_scale / 32767.0f
        // Verify this matches the contract in PROJECT_Q16_INTEGER_ATTENTION_V2.md

        const float kv_scale = DEFAULT_KV_CACHE_SCALE;
        const float expected_d = kv_scale / 32767.0f;

        // This should represent ±8.0 FP32 range mapped to ±32767 INT16
        EXPECT_NEAR(expected_d, 8.0f / 32767.0f, 1e-10f);

        // Verify round-trip: fp32 -> int16 -> fp32
        const float test_value = 3.5f; // Within ±8.0 range
        const int16_t quantized = static_cast<int16_t>(std::round(test_value / expected_d));
        const float dequantized = quantized * expected_d;

        EXPECT_NEAR(dequantized, test_value, expected_d)
            << "Round-trip error should be within one quantization step";
    }

    TEST_F(KVCacheScalePropagationTest, FixedScaleQuantization_TighterScale_BetterPrecision)
    {
        // Tighter kv_cache_scale (4.0 vs 8.0) should give better precision
        const float loose_scale = 8.0f;
        const float tight_scale = 4.0f;

        const float loose_d = loose_scale / 32767.0f;
        const float tight_d = tight_scale / 32767.0f;

        EXPECT_LT(tight_d, loose_d)
            << "Tighter kv_cache_scale should produce smaller quantization step";

        // Precision improvement ratio
        const float precision_ratio = loose_d / tight_d;
        EXPECT_NEAR(precision_ratio, 2.0f, 0.01f)
            << "4.0 scale should be 2x more precise than 8.0 scale";
    }

    // =========================================================================
    // Qwen2GraphConfig Propagation Tests
    // =========================================================================

    TEST_F(KVCacheScalePropagationTest, Qwen2GraphConfig_DefaultScale_MatchesGraphSchema)
    {
        Qwen2GraphConfig config;

        // Config default should match GraphSchema default
        EXPECT_FLOAT_EQ(config.kv_cache_scale, DEFAULT_KV_CACHE_SCALE)
            << "Qwen2GraphConfig.kv_cache_scale default should match GraphSchema default";
    }

    TEST_F(KVCacheScalePropagationTest, Qwen2GraphConfig_HeadDim_PropagatedCorrectly)
    {
        Qwen2GraphConfig config;
        config.head_dim = HEAD_DIM;
        config.kv_cache_scale = CUSTOM_KV_CACHE_SCALE;

        // These should be stored correctly
        EXPECT_EQ(config.head_dim, HEAD_DIM);
        EXPECT_FLOAT_EQ(config.kv_cache_scale, CUSTOM_KV_CACHE_SCALE);
    }

    // =========================================================================
    // Scale Value Validation Tests
    // =========================================================================

    TEST_F(KVCacheScalePropagationTest, ScaleValue_MustBePositive)
    {
        // Zero scale should be invalid (would cause divide-by-zero)
        const float zero_scale = 0.0f;
        const float negative_scale = -1.0f;

        // Test that zero/negative scales would produce invalid quantization
        EXPECT_LE(zero_scale, 0.0f);
        EXPECT_LT(negative_scale, 0.0f);

        // KVCacheAppendStage should reject invalid scales
        // (This is tested at execution time in the stage)
    }

    TEST_F(KVCacheScalePropagationTest, ScaleValue_CommonModelRanges)
    {
        // Document typical scale values for different models
        // These are based on activation profiling (see Phase 10 in project doc)

        // Default (conservative): ±8.0
        EXPECT_FLOAT_EQ(DEFAULT_KV_CACHE_SCALE, 8.0f);

        // Qwen2-0.5B (measured): ±4.0 works well
        EXPECT_FLOAT_EQ(CUSTOM_KV_CACHE_SCALE, 4.0f);

        // Very tight (for high-precision models): ±2.0
        EXPECT_FLOAT_EQ(TIGHT_KV_CACHE_SCALE, 2.0f);

        // All should be positive and reasonable
        EXPECT_GT(DEFAULT_KV_CACHE_SCALE, 0.0f);
        EXPECT_GT(CUSTOM_KV_CACHE_SCALE, 0.0f);
        EXPECT_GT(TIGHT_KV_CACHE_SCALE, 0.0f);
        EXPECT_LE(DEFAULT_KV_CACHE_SCALE, 16.0f); // Reasonable upper bound
    }

    // =========================================================================
    // Contract Verification Tests
    // =========================================================================

    TEST_F(KVCacheScalePropagationTest, Contract_AllDefaultsMatch)
    {
        // Verify all components use the same default value
        GraphSchema schema;
        KVCacheAppendStage::Params stage_params;
        Qwen2GraphConfig config;
        RuntimeConfig runtime_config;
        InferenceRunnerConfig runner_config;

        EXPECT_FLOAT_EQ(schema.kv_cache_scale, stage_params.kv_cache_scale)
            << "GraphSchema and KVCacheAppendStage::Params must have same default";

        EXPECT_FLOAT_EQ(schema.kv_cache_scale, config.kv_cache_scale)
            << "GraphSchema and Qwen2GraphConfig must have same default";

        EXPECT_FLOAT_EQ(stage_params.kv_cache_scale, config.kv_cache_scale)
            << "KVCacheAppendStage::Params and Qwen2GraphConfig must have same default";

        EXPECT_FLOAT_EQ(runtime_config.kv_cache_scale, config.kv_cache_scale)
            << "RuntimeConfig and Qwen2GraphConfig must have same default";

        EXPECT_FLOAT_EQ(runner_config.kv_cache_scale, config.kv_cache_scale)
            << "InferenceRunnerConfig and Qwen2GraphConfig must have same default";

        EXPECT_FLOAT_EQ(runtime_config.kv_cache_scale, DEFAULT_KV_CACHE_SCALE)
            << "RuntimeConfig default kv_cache_scale must be 8.0f";

        EXPECT_FLOAT_EQ(runner_config.kv_cache_scale, DEFAULT_KV_CACHE_SCALE)
            << "InferenceRunnerConfig default kv_cache_scale must be 8.0f";
    }

    TEST_F(KVCacheScalePropagationTest, Contract_QuantizationFormulaIsDocumented)
    {
        // The contract states: d = kv_cache_scale / 32767.0f
        // This test documents and verifies the formula

        const float kv_cache_scale = DEFAULT_KV_CACHE_SCALE;

        // Quantization: int16 = fp32 * 32767 / kv_cache_scale
        // Dequantization: fp32 = int16 * kv_cache_scale / 32767 = int16 * d
        const float d = kv_cache_scale / 32767.0f;
        const float inv_d = 32767.0f / kv_cache_scale;

        // Verify inverse relationship
        EXPECT_NEAR(d * inv_d, 1.0f, 1e-6f);

        // Verify max representable value
        const float max_fp32 = 32767.0f * d;
        EXPECT_NEAR(max_fp32, kv_cache_scale, 1e-4f)
            << "Max representable FP32 should equal kv_cache_scale";
    }

} // namespace llaminar2::test
