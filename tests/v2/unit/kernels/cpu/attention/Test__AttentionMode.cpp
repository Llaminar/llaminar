/**
 * @file Test__AttentionMode.cpp
 * @brief Unit tests for AttentionMode enum and detection function
 * @author David Sanftenberg
 *
 * Tests the AttentionMode enum used for kernel dispatch optimization.
 * Verifies correct mode detection based on batch_size, seq_len, and kv_len.
 */

#include <gtest/gtest.h>
#include "v2/tensors/TensorKernels.h"

using namespace llaminar2;

namespace
{
    /**
     * @brief Test fixture for AttentionMode tests
     */
    class Test__AttentionMode : public ::testing::Test
    {
    protected:
        void SetUp() override {}
        void TearDown() override {}
    };

    // ==========================================================================
    // Mode Detection Tests
    // ==========================================================================

    /**
     * @brief Test PREFILL mode detection
     *
     * PREFILL: seq_len > 1 and kv_len == seq_len (full prompt processing)
     */
    TEST_F(Test__AttentionMode, DetectPrefill)
    {
        // Single sequence, processing full prompt
        EXPECT_EQ(detect_attention_mode(1, 128, 128), AttentionMode::PREFILL);
        EXPECT_EQ(detect_attention_mode(1, 512, 512), AttentionMode::PREFILL);
        EXPECT_EQ(detect_attention_mode(1, 2, 2), AttentionMode::PREFILL);

        // Even with batch_size > 1, if seq_len > 1 and equals kv_len, it's prefill
        EXPECT_EQ(detect_attention_mode(4, 32, 32), AttentionMode::PREFILL);
    }

    /**
     * @brief Test DECODE mode detection
     *
     * DECODE: seq_len == 1, single sequence (autoregressive token generation)
     */
    TEST_F(Test__AttentionMode, DetectDecode)
    {
        // Single sequence, single new token, cache of various sizes
        EXPECT_EQ(detect_attention_mode(1, 1, 1), AttentionMode::DECODE);
        EXPECT_EQ(detect_attention_mode(1, 1, 128), AttentionMode::DECODE);
        EXPECT_EQ(detect_attention_mode(1, 1, 512), AttentionMode::DECODE);
        EXPECT_EQ(detect_attention_mode(1, 1, 4096), AttentionMode::DECODE);
    }

    /**
     * @brief Test BATCHED_DECODE mode detection
     *
     * BATCHED_DECODE: batch_size > 1 and seq_len == 1 (parallel decode)
     */
    TEST_F(Test__AttentionMode, DetectBatchedDecode)
    {
        // Multiple sequences, each decoding one token
        EXPECT_EQ(detect_attention_mode(2, 1, 128), AttentionMode::BATCHED_DECODE);
        EXPECT_EQ(detect_attention_mode(4, 1, 256), AttentionMode::BATCHED_DECODE);
        EXPECT_EQ(detect_attention_mode(32, 1, 512), AttentionMode::BATCHED_DECODE);
        EXPECT_EQ(detect_attention_mode(8, 1, 1024), AttentionMode::BATCHED_DECODE);
    }

    /**
     * @brief Test CHUNKED_PREFILL mode detection
     *
     * CHUNKED_PREFILL: seq_len > 1 but kv_len > seq_len (incremental prefill)
     */
    TEST_F(Test__AttentionMode, DetectChunkedPrefill)
    {
        // Processing chunk of prompt with existing cache
        EXPECT_EQ(detect_attention_mode(1, 64, 128), AttentionMode::CHUNKED_PREFILL);
        EXPECT_EQ(detect_attention_mode(1, 128, 512), AttentionMode::CHUNKED_PREFILL);
        EXPECT_EQ(detect_attention_mode(1, 32, 256), AttentionMode::CHUNKED_PREFILL);

        // Batch chunked prefill
        EXPECT_EQ(detect_attention_mode(4, 64, 128), AttentionMode::CHUNKED_PREFILL);
    }

    // ==========================================================================
    // Mode Name Tests
    // ==========================================================================

    /**
     * @brief Test attention_mode_name() returns correct strings
     */
    TEST_F(Test__AttentionMode, ModeNameStrings)
    {
        EXPECT_STREQ(attention_mode_name(AttentionMode::PREFILL), "PREFILL");
        EXPECT_STREQ(attention_mode_name(AttentionMode::DECODE), "DECODE");
        EXPECT_STREQ(attention_mode_name(AttentionMode::BATCHED_DECODE), "BATCHED_DECODE");
        EXPECT_STREQ(attention_mode_name(AttentionMode::CHUNKED_PREFILL), "CHUNKED_PREFILL");
    }

    // ==========================================================================
    // Edge Case Tests
    // ==========================================================================

    /**
     * @brief Test edge cases in mode detection
     */
    TEST_F(Test__AttentionMode, EdgeCases)
    {
        // Minimal prefill (2 tokens)
        EXPECT_EQ(detect_attention_mode(1, 2, 2), AttentionMode::PREFILL);

        // First decode step after prefill (kv_len = 1)
        EXPECT_EQ(detect_attention_mode(1, 1, 1), AttentionMode::DECODE);

        // Batched decode with just 2 sequences
        EXPECT_EQ(detect_attention_mode(2, 1, 1), AttentionMode::BATCHED_DECODE);

        // Large batch decode
        EXPECT_EQ(detect_attention_mode(128, 1, 4096), AttentionMode::BATCHED_DECODE);
    }

    /**
     * @brief Test that batch_size > 1 with seq_len > 1 uses correct mode
     *
     * When batch_size > 1 and seq_len > 1, the mode depends on seq_len vs kv_len:
     * - seq_len == kv_len: PREFILL (parallel prefill of multiple sequences)
     * - seq_len < kv_len: CHUNKED_PREFILL (incremental)
     */
    TEST_F(Test__AttentionMode, BatchPrefillModes)
    {
        // Batched prefill (multiple sequences, full prompt each)
        EXPECT_EQ(detect_attention_mode(4, 128, 128), AttentionMode::PREFILL);

        // Batched chunked prefill (multiple sequences, incremental)
        EXPECT_EQ(detect_attention_mode(4, 64, 256), AttentionMode::CHUNKED_PREFILL);
    }

    // ==========================================================================
    // Enum Value Tests
    // ==========================================================================

    /**
     * @brief Test that all enum values are distinct
     */
    TEST_F(Test__AttentionMode, EnumValuesDistinct)
    {
        EXPECT_NE(static_cast<int>(AttentionMode::PREFILL), static_cast<int>(AttentionMode::DECODE));
        EXPECT_NE(static_cast<int>(AttentionMode::PREFILL), static_cast<int>(AttentionMode::BATCHED_DECODE));
        EXPECT_NE(static_cast<int>(AttentionMode::PREFILL), static_cast<int>(AttentionMode::CHUNKED_PREFILL));
        EXPECT_NE(static_cast<int>(AttentionMode::DECODE), static_cast<int>(AttentionMode::BATCHED_DECODE));
        EXPECT_NE(static_cast<int>(AttentionMode::DECODE), static_cast<int>(AttentionMode::CHUNKED_PREFILL));
        EXPECT_NE(static_cast<int>(AttentionMode::BATCHED_DECODE), static_cast<int>(AttentionMode::CHUNKED_PREFILL));
    }

} // namespace
