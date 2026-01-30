/**
 * @file Test__HybridPrecisionConfig.cpp
 * @brief Unit tests for HybridPrecisionConfig buffer precision resolution
 *
 * Tests the per-buffer precision configuration for Hybrid activation mode.
 */

#include <gtest/gtest.h>
#include "execution/config/HybridPrecisionConfig.h"
#include "execution/config/RuntimeConfig.h"

namespace llaminar2
{
    namespace test
    {

        class Test__HybridPrecisionConfig : public ::testing::Test
        {
        protected:
            HybridPrecisionConfig default_config_;

            void SetUp() override
            {
                default_config_ = HybridPrecisionConfig::defaultConfig();
            }
        };

        // =============================================================================
        // Default Configuration Tests
        // =============================================================================

        TEST_F(Test__HybridPrecisionConfig, DefaultConfig_QKV_GEMM_Output_Is_Q8_1)
        {
            auto prec = default_config_.getPrecision(HybridBufferType::QKV_GEMM_Output);
            EXPECT_EQ(prec, ActivationPrecision::Q8_1);
        }

        TEST_F(Test__HybridPrecisionConfig, DefaultConfig_Q_After_RoPE_Is_FP32)
        {
            auto prec = default_config_.getPrecision(HybridBufferType::Q_After_RoPE);
            EXPECT_EQ(prec, ActivationPrecision::FP32);
        }

        TEST_F(Test__HybridPrecisionConfig, DefaultConfig_K_After_RoPE_Is_FP32)
        {
            auto prec = default_config_.getPrecision(HybridBufferType::K_After_RoPE);
            EXPECT_EQ(prec, ActivationPrecision::FP32);
        }

        // NOTE: KV_Cache is currently FP32 (see HybridPrecisionConfig.h TODO)
        // The design intent is BF16, but FP32→BF16 conversion in KVCacheAppendStage is not yet implemented
        TEST_F(Test__HybridPrecisionConfig, DefaultConfig_KV_Cache_Is_FP32)
        {
            auto prec = default_config_.getPrecision(HybridBufferType::KV_Cache);
            EXPECT_EQ(prec, ActivationPrecision::FP32);
        }

        TEST_F(Test__HybridPrecisionConfig, DefaultConfig_Attention_Context_Is_FP32)
        {
            auto prec = default_config_.getPrecision(HybridBufferType::Attention_Context);
            EXPECT_EQ(prec, ActivationPrecision::FP32);
        }

        TEST_F(Test__HybridPrecisionConfig, DefaultConfig_Attention_Output_Is_FP32)
        {
            auto prec = default_config_.getPrecision(HybridBufferType::Attention_Output);
            EXPECT_EQ(prec, ActivationPrecision::FP32);
        }

        TEST_F(Test__HybridPrecisionConfig, DefaultConfig_FFN_Gate_Is_Q8_1)
        {
            auto prec = default_config_.getPrecision(HybridBufferType::FFN_Gate);
            EXPECT_EQ(prec, ActivationPrecision::Q8_1);
        }

        TEST_F(Test__HybridPrecisionConfig, DefaultConfig_FFN_Up_Is_Q8_1)
        {
            auto prec = default_config_.getPrecision(HybridBufferType::FFN_Up);
            EXPECT_EQ(prec, ActivationPrecision::Q8_1);
        }

        TEST_F(Test__HybridPrecisionConfig, DefaultConfig_FFN_Down_Is_FP32)
        {
            auto prec = default_config_.getPrecision(HybridBufferType::FFN_Down);
            EXPECT_EQ(prec, ActivationPrecision::FP32);
        }

        // =============================================================================
        // resolveBufferPrecision Tests
        // =============================================================================

        TEST_F(Test__HybridPrecisionConfig, ResolvePrecision_FP32Mode_ReturnsGlobalForAll)
        {
            // In FP32 mode, all buffers use FP32
            auto prec = resolveBufferPrecision(
                ActivationPrecision::FP32, HybridBufferType::QKV_GEMM_Output, nullptr);
            EXPECT_EQ(prec, ActivationPrecision::FP32);

            prec = resolveBufferPrecision(
                ActivationPrecision::FP32, HybridBufferType::FFN_Gate, nullptr);
            EXPECT_EQ(prec, ActivationPrecision::FP32);
        }

        TEST_F(Test__HybridPrecisionConfig, ResolvePrecision_Q8_1Mode_ReturnsGlobalForAll)
        {
            // In Q8_1 mode, intermediate buffers use Q8_1, but core buffers stay FP32
            auto prec = resolveBufferPrecision(
                ActivationPrecision::Q8_1, HybridBufferType::QKV_GEMM_Output, nullptr);
            EXPECT_EQ(prec, ActivationPrecision::Q8_1);

            // FFN_Down outputs to residual, so stays FP32 for numerical stability
            prec = resolveBufferPrecision(
                ActivationPrecision::Q8_1, HybridBufferType::FFN_Down, nullptr);
            EXPECT_EQ(prec, ActivationPrecision::FP32);

            // FFN_Gate is an intermediate, uses global precision
            prec = resolveBufferPrecision(
                ActivationPrecision::Q8_1, HybridBufferType::FFN_Gate, nullptr);
            EXPECT_EQ(prec, ActivationPrecision::Q8_1);
        }

        TEST_F(Test__HybridPrecisionConfig, ResolvePrecision_HybridMode_UsesPerBuffer)
        {
            // In Hybrid mode, uses per-buffer precision from config
            auto prec = resolveBufferPrecision(
                ActivationPrecision::Hybrid, HybridBufferType::QKV_GEMM_Output, &default_config_);
            EXPECT_EQ(prec, ActivationPrecision::Q8_1);

            prec = resolveBufferPrecision(
                ActivationPrecision::Hybrid, HybridBufferType::Q_After_RoPE, &default_config_);
            EXPECT_EQ(prec, ActivationPrecision::FP32);

            // NOTE: KV_Cache is currently FP32 (see HybridPrecisionConfig.h TODO)
            prec = resolveBufferPrecision(
                ActivationPrecision::Hybrid, HybridBufferType::KV_Cache, &default_config_);
            EXPECT_EQ(prec, ActivationPrecision::FP32);
        }

        TEST_F(Test__HybridPrecisionConfig, ResolvePrecision_HybridMode_NoConfig_UsesDefaultHybridConfig)
        {
            // Hybrid mode without config uses the default hybrid config internally
            // QKV_GEMM_Output should be Q8_1 (from default config)
            auto prec = resolveBufferPrecision(
                ActivationPrecision::Hybrid, HybridBufferType::QKV_GEMM_Output, nullptr);
            EXPECT_EQ(prec, ActivationPrecision::Q8_1);
        }

        // =============================================================================
        // requiresSeparateBuffer Tests
        // =============================================================================

        TEST_F(Test__HybridPrecisionConfig, RequiresSeparateBuffer_Q_After_RoPE)
        {
            // Q_After_RoPE needs separate buffer in Hybrid mode
            EXPECT_TRUE(HybridPrecisionConfig::requiresSeparateBuffer(HybridBufferType::Q_After_RoPE));
        }

        TEST_F(Test__HybridPrecisionConfig, RequiresSeparateBuffer_K_After_RoPE)
        {
            // K_After_RoPE needs separate buffer in Hybrid mode
            EXPECT_TRUE(HybridPrecisionConfig::requiresSeparateBuffer(HybridBufferType::K_After_RoPE));
        }

        TEST_F(Test__HybridPrecisionConfig, RequiresSeparateBuffer_Residual_DoesNot)
        {
            // Residual doesn't need separate buffer (always FP32)
            EXPECT_FALSE(HybridPrecisionConfig::requiresSeparateBuffer(HybridBufferType::Residual));
        }

        // =============================================================================
        // hybridBufferTypeToString Tests
        // =============================================================================

        TEST_F(Test__HybridPrecisionConfig, BufferTypeToString_Returns_NonEmpty)
        {
            const char *name = hybridBufferTypeToString(HybridBufferType::QKV_GEMM_Output);
            EXPECT_NE(name, nullptr);
            EXPECT_STRNE(name, "");
        }

        TEST_F(Test__HybridPrecisionConfig, BufferTypeToString_All_Types)
        {
            // Test all buffer types return non-null strings
            EXPECT_STRNE(hybridBufferTypeToString(HybridBufferType::Residual), "");
            EXPECT_STRNE(hybridBufferTypeToString(HybridBufferType::Normalized), "");
            EXPECT_STRNE(hybridBufferTypeToString(HybridBufferType::Hidden), "");
            EXPECT_STRNE(hybridBufferTypeToString(HybridBufferType::Logits), "");
            EXPECT_STRNE(hybridBufferTypeToString(HybridBufferType::QKV_GEMM_Output), "");
            EXPECT_STRNE(hybridBufferTypeToString(HybridBufferType::Q_After_RoPE), "");
            EXPECT_STRNE(hybridBufferTypeToString(HybridBufferType::K_After_RoPE), "");
            EXPECT_STRNE(hybridBufferTypeToString(HybridBufferType::KV_Cache), "");
            EXPECT_STRNE(hybridBufferTypeToString(HybridBufferType::Attention_Context), "");
            EXPECT_STRNE(hybridBufferTypeToString(HybridBufferType::Attention_Output), "");
            EXPECT_STRNE(hybridBufferTypeToString(HybridBufferType::FFN_Gate), "");
            EXPECT_STRNE(hybridBufferTypeToString(HybridBufferType::FFN_Up), "");
            EXPECT_STRNE(hybridBufferTypeToString(HybridBufferType::FFN_Down), "");
        }

        // =============================================================================
        // HybridQ16PrecisionConfig Tests
        // =============================================================================

        class Test__HybridQ16PrecisionConfig : public ::testing::Test
        {
        protected:
            HybridQ16PrecisionConfig default_config_;

            void SetUp() override
            {
                default_config_ = HybridQ16PrecisionConfig::defaultConfig();
            }
        };

        // --- Residual Stream Tests ---

        TEST_F(Test__HybridQ16PrecisionConfig, DefaultConfig_ResidualStream_Is_Q16_1)
        {
            auto prec = default_config_.getPrecision(HybridBufferType::ResidualStream);
            EXPECT_EQ(prec, ActivationPrecision::Q16_1);
        }

        TEST_F(Test__HybridQ16PrecisionConfig, DefaultConfig_Residual_Is_Q16_1)
        {
            auto prec = default_config_.getPrecision(HybridBufferType::Residual);
            EXPECT_EQ(prec, ActivationPrecision::Q16_1);
        }

        // --- QKV Path Tests ---

        TEST_F(Test__HybridQ16PrecisionConfig, DefaultConfig_QKV_GEMM_Output_Is_Q8_1)
        {
            auto prec = default_config_.getPrecision(HybridBufferType::QKV_GEMM_Output);
            EXPECT_EQ(prec, ActivationPrecision::Q8_1);
        }

        TEST_F(Test__HybridQ16PrecisionConfig, DefaultConfig_Q_After_RoPE_Is_Q16_1)
        {
            // Q16 fused attention kernel expects Q16_1 Q input
            auto prec = default_config_.getPrecision(HybridBufferType::Q_After_RoPE);
            EXPECT_EQ(prec, ActivationPrecision::Q16_1);
        }

        TEST_F(Test__HybridQ16PrecisionConfig, DefaultConfig_K_After_RoPE_Is_Q16_1)
        {
            // Q16 fused attention kernel expects Q16_1 K input
            auto prec = default_config_.getPrecision(HybridBufferType::K_After_RoPE);
            EXPECT_EQ(prec, ActivationPrecision::Q16_1);
        }

        TEST_F(Test__HybridQ16PrecisionConfig, DefaultConfig_KV_Cache_Is_Q16_1)
        {
            // Q16 fused attention kernel uses Q16_1 KV cache
            auto prec = default_config_.getPrecision(HybridBufferType::KV_Cache);
            EXPECT_EQ(prec, ActivationPrecision::Q16_1);
        }

        // --- Attention Tests ---

        TEST_F(Test__HybridQ16PrecisionConfig, DefaultConfig_Attention_Context_Is_Q16_1)
        {
            // For snapshots - fused kernel uses INT32 internally
            auto prec = default_config_.getPrecision(HybridBufferType::Attention_Context);
            EXPECT_EQ(prec, ActivationPrecision::Q16_1);
        }

        TEST_F(Test__HybridQ16PrecisionConfig, DefaultConfig_Attention_Output_Is_Q16_1)
        {
            // Fused kernel writes Q16_1 directly to residual
            auto prec = default_config_.getPrecision(HybridBufferType::Attention_Output);
            EXPECT_EQ(prec, ActivationPrecision::Q16_1);
        }

        // --- FFN Tests ---

        TEST_F(Test__HybridQ16PrecisionConfig, DefaultConfig_FFN_Gate_Is_Q8_1)
        {
            auto prec = default_config_.getPrecision(HybridBufferType::FFN_Gate);
            EXPECT_EQ(prec, ActivationPrecision::Q8_1);
        }

        TEST_F(Test__HybridQ16PrecisionConfig, DefaultConfig_FFN_Up_Is_Q8_1)
        {
            auto prec = default_config_.getPrecision(HybridBufferType::FFN_Up);
            EXPECT_EQ(prec, ActivationPrecision::Q8_1);
        }

        TEST_F(Test__HybridQ16PrecisionConfig, DefaultConfig_FFN_Down_Is_Q8_1)
        {
            // Added to Q16_1 residual
            auto prec = default_config_.getPrecision(HybridBufferType::FFN_Down);
            EXPECT_EQ(prec, ActivationPrecision::Q8_1);
        }

        // --- Core Buffer Tests ---

        TEST_F(Test__HybridQ16PrecisionConfig, DefaultConfig_Normalized_Is_FP32)
        {
            // RMSNorm output stays FP32 for GEMM input
            auto prec = default_config_.getPrecision(HybridBufferType::Normalized);
            EXPECT_EQ(prec, ActivationPrecision::FP32);
        }

        TEST_F(Test__HybridQ16PrecisionConfig, DefaultConfig_Hidden_Is_FP32)
        {
            auto prec = default_config_.getPrecision(HybridBufferType::Hidden);
            EXPECT_EQ(prec, ActivationPrecision::FP32);
        }

        TEST_F(Test__HybridQ16PrecisionConfig, DefaultConfig_Logits_Is_FP32)
        {
            auto prec = default_config_.getPrecision(HybridBufferType::Logits);
            EXPECT_EQ(prec, ActivationPrecision::FP32);
        }

        // --- resolveBufferPrecision HybridQ16 Mode Tests ---

        TEST_F(Test__HybridQ16PrecisionConfig, ResolvePrecision_HybridQ16Mode_Residual_Is_Q16_1)
        {
            auto prec = resolveBufferPrecision(
                ActivationPrecision::HybridQ16, HybridBufferType::Residual, nullptr);
            EXPECT_EQ(prec, ActivationPrecision::Q16_1);
        }

        TEST_F(Test__HybridQ16PrecisionConfig, ResolvePrecision_HybridQ16Mode_Q_After_RoPE_Is_Q16_1)
        {
            auto prec = resolveBufferPrecision(
                ActivationPrecision::HybridQ16, HybridBufferType::Q_After_RoPE, nullptr);
            EXPECT_EQ(prec, ActivationPrecision::Q16_1);
        }

        TEST_F(Test__HybridQ16PrecisionConfig, ResolvePrecision_HybridQ16Mode_K_After_RoPE_Is_Q16_1)
        {
            auto prec = resolveBufferPrecision(
                ActivationPrecision::HybridQ16, HybridBufferType::K_After_RoPE, nullptr);
            EXPECT_EQ(prec, ActivationPrecision::Q16_1);
        }

        TEST_F(Test__HybridQ16PrecisionConfig, ResolvePrecision_HybridQ16Mode_KV_Cache_Is_Q16_1)
        {
            auto prec = resolveBufferPrecision(
                ActivationPrecision::HybridQ16, HybridBufferType::KV_Cache, nullptr);
            EXPECT_EQ(prec, ActivationPrecision::Q16_1);
        }

        TEST_F(Test__HybridQ16PrecisionConfig, ResolvePrecision_HybridQ16Mode_Attention_Context_Is_Q16_1)
        {
            auto prec = resolveBufferPrecision(
                ActivationPrecision::HybridQ16, HybridBufferType::Attention_Context, nullptr);
            EXPECT_EQ(prec, ActivationPrecision::Q16_1);
        }

        TEST_F(Test__HybridQ16PrecisionConfig, ResolvePrecision_HybridQ16Mode_Attention_Output_Is_Q16_1)
        {
            auto prec = resolveBufferPrecision(
                ActivationPrecision::HybridQ16, HybridBufferType::Attention_Output, nullptr);
            EXPECT_EQ(prec, ActivationPrecision::Q16_1);
        }

        TEST_F(Test__HybridQ16PrecisionConfig, ResolvePrecision_HybridQ16Mode_FFN_Down_Is_Q8_1)
        {
            auto prec = resolveBufferPrecision(
                ActivationPrecision::HybridQ16, HybridBufferType::FFN_Down, nullptr);
            EXPECT_EQ(prec, ActivationPrecision::Q8_1);
        }

    } // namespace test
} // namespace llaminar2
