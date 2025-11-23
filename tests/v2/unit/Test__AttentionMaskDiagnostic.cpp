/**
 * @file Test__AttentionMaskDiagnostic.cpp
 * @brief Minimal diagnostic test for attention mask construction
 * @author David Sanftenberg
 *
 * This test directly verifies attention mask utility functions work correctly
 * before testing them in the context of full attention computation.
 */

#include <gtest/gtest.h>
#include <vector>
#include <cstring>

#include "v2/pipelines/AttentionUtils.h"
#include "v2/tensors/Tensors.h"
#include "v2/utils/Logger.h"

using namespace llaminar2;

namespace
{
    constexpr float MASK_NEG_INF = -1e10f;

    bool is_masked(float value)
    {
        return value < -1e9f;
    }

    void print_mask(const char *label, const float *mask, int rows, int cols)
    {
        LOG_ERROR("Mask: " << label);
        for (int i = 0; i < rows; ++i)
        {
            std::stringstream ss;
            ss << "  Row " << i << ": [";
            for (int j = 0; j < cols; ++j)
            {
                if (j > 0)
                    ss << ", ";
                float val = mask[i * cols + j];
                if (is_masked(val))
                    ss << "-inf";
                else
                    ss << val;
            }
            ss << "]";
            LOG_ERROR(ss.str());
        }
    }
}

//==============================================================================
// Test 1: Simple causal mask (single sequence, no padding)
//==============================================================================
TEST(Test__AttentionMaskDiagnostic, SimpleCausalMask)
{
    const int seq_len = 4;
    std::vector<float> mask(seq_len * seq_len, 0.0f);

    // Create simple causal mask
    attention_utils::create_causal_mask(mask.data(), seq_len, 0 /* no window */);

    print_mask("Simple Causal (4x4)", mask.data(), seq_len, seq_len);

    // Verify structure:
    // Row 0: [0, -inf, -inf, -inf]  (position 0 can only see itself)
    // Row 1: [0, 0, -inf, -inf]     (position 1 can see 0,1)
    // Row 2: [0, 0, 0, -inf]        (position 2 can see 0,1,2)
    // Row 3: [0, 0, 0, 0]           (position 3 can see 0,1,2,3)

    EXPECT_FALSE(is_masked(mask[0 * seq_len + 0])) << "Pos 0 should see itself";
    EXPECT_TRUE(is_masked(mask[0 * seq_len + 1])) << "Pos 0 should NOT see future";

    EXPECT_FALSE(is_masked(mask[1 * seq_len + 0])) << "Pos 1 should see past";
    EXPECT_FALSE(is_masked(mask[1 * seq_len + 1])) << "Pos 1 should see itself";
    EXPECT_TRUE(is_masked(mask[1 * seq_len + 2])) << "Pos 1 should NOT see future";

    EXPECT_FALSE(is_masked(mask[3 * seq_len + 0])) << "Last pos should see all past";
    EXPECT_FALSE(is_masked(mask[3 * seq_len + 3])) << "Last pos should see itself";
}

//==============================================================================
// Test 2: Padding mask only (no causal, batch of 1)
//==============================================================================
TEST(Test__AttentionMaskDiagnostic, SimplePaddingMask)
{
    const int batch_size = 1;
    const int seq_len = 4;
    const int actual_length = 2; // Only 2 real tokens, 2 padding
    std::vector<int> lengths = {actual_length};

    std::vector<float> mask(batch_size * seq_len * seq_len, 0.0f);

    attention_utils::create_batch_padding_mask(
        mask.data(),
        batch_size,
        seq_len,
        lengths.data(),
        0 /* no window */
    );

    print_mask("Padding Only (batch=1, len=2/4)", mask.data(), seq_len, seq_len);

    // Expected structure (non-causal, so bidirectional attention):
    // Row 0 (real): [0, 0, -inf, -inf]  (can attend to 0,1 but not padding 2,3)
    // Row 1 (real): [0, 0, -inf, -inf]  (can attend to 0,1 but not padding 2,3)
    // Row 2 (pad):  [-inf, -inf, -inf, -inf]  (padding doesn't attend)
    // Row 3 (pad):  [-inf, -inf, -inf, -inf]  (padding doesn't attend)

    // Real tokens can attend to each other
    EXPECT_FALSE(is_masked(mask[0 * seq_len + 0])) << "Real token 0 should see token 0";
    EXPECT_FALSE(is_masked(mask[0 * seq_len + 1])) << "Real token 0 should see token 1";
    EXPECT_TRUE(is_masked(mask[0 * seq_len + 2])) << "Real token 0 should NOT see padding 2";
    EXPECT_TRUE(is_masked(mask[0 * seq_len + 3])) << "Real token 0 should NOT see padding 3";

    EXPECT_FALSE(is_masked(mask[1 * seq_len + 0])) << "Real token 1 should see token 0";
    EXPECT_FALSE(is_masked(mask[1 * seq_len + 1])) << "Real token 1 should see token 1";
    EXPECT_TRUE(is_masked(mask[1 * seq_len + 2])) << "Real token 1 should NOT see padding 2";

    // Padding tokens have all-masked rows
    EXPECT_TRUE(is_masked(mask[2 * seq_len + 0])) << "Padding row should be fully masked";
    EXPECT_TRUE(is_masked(mask[3 * seq_len + 0])) << "Padding row should be fully masked";
}

//==============================================================================
// Test 3: Batch padding mask (2 sequences, different lengths)
//==============================================================================
TEST(Test__AttentionMaskDiagnostic, BatchPaddingMask)
{
    const int batch_size = 2;
    const int seq_len = 4;
    std::vector<int> lengths = {4, 2}; // Seq0: 4 real, Seq1: 2 real + 2 padding

    std::vector<float> mask(batch_size * seq_len * seq_len, 0.0f);

    attention_utils::create_batch_padding_mask(
        mask.data(),
        batch_size,
        seq_len,
        lengths.data(),
        0 /* no window */
    );

    print_mask("Batch Padding (lengths=[4,2])", mask.data(), batch_size * seq_len, seq_len);

    // Seq0 (positions 0-3, no padding): All positions should be unmasked
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            EXPECT_FALSE(is_masked(mask[i * seq_len + j]))
                << "Seq0[" << i << "][" << j << "] should be unmasked (no padding)";
        }
    }

    // Seq1 (positions 4-7, actual_length=2):
    // Real tokens [4,5] should only attend to [4,5], not [6,7] (padding)
    const int seq1_offset = 4;
    EXPECT_FALSE(is_masked(mask[seq1_offset * seq_len + 0]))
        << "Seq1[0] should attend to position 0 (within local seq)";
    EXPECT_FALSE(is_masked(mask[seq1_offset * seq_len + 1]))
        << "Seq1[0] should attend to position 1 (within local seq)";
    EXPECT_TRUE(is_masked(mask[seq1_offset * seq_len + 2]))
        << "Seq1[0] should NOT attend to position 2 (padding)";
    EXPECT_TRUE(is_masked(mask[seq1_offset * seq_len + 3]))
        << "Seq1[0] should NOT attend to position 3 (padding)";
}

//==============================================================================
// Test 4: Combined causal + padding mask
//==============================================================================
TEST(Test__AttentionMaskDiagnostic, CombinedCausalPaddingMask)
{
    const int batch_size = 1;
    const int seq_len = 4;
    const int actual_length = 2;
    std::vector<int> lengths = {actual_length};

    std::vector<float> mask(batch_size * seq_len * seq_len, 0.0f);

    attention_utils::create_combined_batch_mask(
        mask.data(),
        batch_size,
        seq_len,
        lengths.data(),
        true, /* causal */
        0     /* no window */
    );

    print_mask("Combined Causal+Padding (len=2/4)", mask.data(), seq_len, seq_len);

    // Expected: Union of causal + padding masks
    // Row 0 (real): [0, -inf, -inf, -inf]  (causal: only self; padding: 2,3)
    // Row 1 (real): [0, 0, -inf, -inf]     (causal: 0,1; padding: 2,3)
    // Row 2 (pad):  [-inf, -inf, -inf, -inf]
    // Row 3 (pad):  [-inf, -inf, -inf, -inf]

    EXPECT_FALSE(is_masked(mask[0 * seq_len + 0])) << "Pos 0 should see itself";
    EXPECT_TRUE(is_masked(mask[0 * seq_len + 1])) << "Pos 0 should NOT see future (causal)";

    EXPECT_FALSE(is_masked(mask[1 * seq_len + 0])) << "Pos 1 should see past";
    EXPECT_FALSE(is_masked(mask[1 * seq_len + 1])) << "Pos 1 should see itself";
    EXPECT_TRUE(is_masked(mask[1 * seq_len + 2])) << "Pos 1 should NOT see padding";

    // Padding rows
    EXPECT_TRUE(is_masked(mask[2 * seq_len + 0])) << "Padding row fully masked";
    EXPECT_TRUE(is_masked(mask[3 * seq_len + 0])) << "Padding row fully masked";
}

//==============================================================================
// Main
//==============================================================================
int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
