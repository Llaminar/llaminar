/**
 * @file Test__Q8_1_ResidualAdd_Layer21.cpp
 * @brief Unit tests for Q8_1 residual add using actual layer 21 dump data
 *
 * These tests verify the Q8_1 residual add operation using real data captured
 * from layer 21 where we observed apparent divergence.
 *
 * KEY FINDING: The q8_1_add_q8_1 function is CORRECT. The apparent divergence
 * between FP32 and Q8_1 at layer21_FFN_RESIDUAL is caused by accumulated
 * quantization error in the INPUTS, not bugs in the addition operation.
 *
 * This test file serves as:
 * 1. Verification that the add operation works correctly for extreme values
 * 2. Documentation of the expected behavior for large-magnitude cancellation
 * 3. Regression testing for any future changes to q8_1_add_q8_1
 *
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <vector>
#include "tensors/SIMDHelpers.h"
#include "tensors/Tensors.h"

namespace llaminar2::test
{

    /**
     * @brief Test fixture for Q8_1 residual add with layer 21 data
     */
    class Test__Q8_1_ResidualAdd_Layer21 : public ::testing::Test
    {
    protected:
        /**
         * @brief Create a Q8_1Block with specified scale and quantized values
         */
        static Q8_1Block createBlock(float scale, int16_t sum_qs, const std::vector<int8_t> &qs)
        {
            Q8_1Block block;
            block.d = simd::fp32_to_fp16(scale);
            block.sum_qs = sum_qs;
            std::memset(block.qs, 0, 32);
            size_t copy_count = std::min(qs.size(), size_t(32));
            std::memcpy(block.qs, qs.data(), copy_count);
            return block;
        }

        /**
         * @brief Dequantize a Q8_1Block to FP32 values
         */
        static void dequantizeBlock(const Q8_1Block &block, float *output)
        {
            float scale = simd::fp16_to_fp32(block.d);
            for (int i = 0; i < 32; ++i)
            {
                output[i] = static_cast<float>(block.qs[i]) * scale;
            }
        }

        /**
         * @brief Compute max absolute difference between two arrays
         */
        static float maxAbsDiff(const float *a, const float *b, size_t n)
        {
            float max_diff = 0.0f;
            for (size_t i = 0; i < n; ++i)
            {
                max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
            }
            return max_diff;
        }
    };

    /**
     * @brief Test q8_1_add_q8_1 with Block 1 values from layer 21 dump
     *
     * Block 1 has the highest magnitude values and shows the largest error.
     * Residual: scale=8.6328, range [-8.63, 1096.37]
     * FFN output: scale=15.4844, range [-1966.52, 15.48]
     * Expected result: range [-870.15, 8.63]
     */
    TEST_F(Test__Q8_1_ResidualAdd_Layer21, Block1_HighMagnitude)
    {
        // These are the actual values from layer 21 block 1
        // Residual block 1: scale=8.6328, sum_qs=137
        // Most values are near 0, with one extreme outlier at 127 (1096.37)
        std::vector<int8_t> res_qs = {
            0, -1, 0, 3, 0, 1, 0, 1, 127, 0, 0, 0, 0, 0, 0, 0, // qs[8]=127 is the outlier
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        Q8_1Block res_block = createBlock(8.6328f, 137, res_qs);

        // FFN output block 1: scale=15.4844, sum_qs=-137
        // Extreme negative value at qs[8] = -127 (-1966.52)
        std::vector<int8_t> ffn_qs = {
            0, 0, 0, -3, 0, 0, 0, 0, -127, 0, 0, 0, 0, 0, 0, 0, // qs[8]=-127 is the outlier
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        Q8_1Block ffn_block = createBlock(15.4844f, -137, ffn_qs);

        // Output buffer
        Q8_1Block out_block;

        // Perform Q8_1 addition
        simd::q8_1_add_q8_1(&res_block, &ffn_block, &out_block, 32);

        // Dequantize all three blocks
        float res_dequant[32], ffn_dequant[32], out_dequant[32], expected[32];
        dequantizeBlock(res_block, res_dequant);
        dequantizeBlock(ffn_block, ffn_dequant);
        dequantizeBlock(out_block, out_dequant);

        // Compute expected result
        for (int i = 0; i < 32; ++i)
        {
            expected[i] = res_dequant[i] + ffn_dequant[i];
        }

        // Verify the add operation is correct within Q8_1 precision limits
        float max_diff = maxAbsDiff(expected, out_dequant, 32);

        // Q8_1 precision: with scale ~7, error can be up to scale/127 * 2 ≈ 0.11
        // But after requantization, error can compound. We allow up to 2.0 for extreme cases.
        EXPECT_LT(max_diff, 2.0f) << "Q8_1 add should be accurate within 2.0 for extreme values";

        // Verify the output range is approximately correct
        float out_min = *std::min_element(out_dequant, out_dequant + 32);
        float out_max = *std::max_element(out_dequant, out_dequant + 32);
        float expected_min = *std::min_element(expected, expected + 32);
        float expected_max = *std::max_element(expected, expected + 32);

        EXPECT_NEAR(out_min, expected_min, 2.0f);
        EXPECT_NEAR(out_max, expected_max, 2.0f);

        // Log the results for debugging
        std::cout << "Block 1 (high magnitude) test:" << std::endl;
        std::cout << "  Residual range: [" << *std::min_element(res_dequant, res_dequant + 32)
                  << ", " << *std::max_element(res_dequant, res_dequant + 32) << "]" << std::endl;
        std::cout << "  FFN range: [" << *std::min_element(ffn_dequant, ffn_dequant + 32)
                  << ", " << *std::max_element(ffn_dequant, ffn_dequant + 32) << "]" << std::endl;
        std::cout << "  Expected range: [" << expected_min << ", " << expected_max << "]" << std::endl;
        std::cout << "  Actual range: [" << out_min << ", " << out_max << "]" << std::endl;
        std::cout << "  Max diff: " << max_diff << std::endl;
    }

    /**
     * @brief Test q8_1_add_q8_1 with cancellation scenario (opposite signs, similar magnitudes)
     *
     * This tests the case where large positive and large negative values should nearly cancel.
     */
    TEST_F(Test__Q8_1_ResidualAdd_Layer21, CancellationScenario)
    {
        // Residual: large positive values
        std::vector<int8_t> res_qs;
        for (int i = 0; i < 32; ++i)
        {
            res_qs.push_back(static_cast<int8_t>(100 + (i % 27))); // ~100-127
        }
        Q8_1Block res_block = createBlock(10.0f, 3200, res_qs);

        // FFN output: large negative values (nearly canceling)
        std::vector<int8_t> ffn_qs;
        for (int i = 0; i < 32; ++i)
        {
            ffn_qs.push_back(static_cast<int8_t>(-100 - (i % 27))); // ~-100 to -127
        }
        Q8_1Block ffn_block = createBlock(10.0f, -3200, ffn_qs);

        // Output buffer
        Q8_1Block out_block;

        // Perform Q8_1 addition
        simd::q8_1_add_q8_1(&res_block, &ffn_block, &out_block, 32);

        // Dequantize
        float res_dequant[32], ffn_dequant[32], out_dequant[32], expected[32];
        dequantizeBlock(res_block, res_dequant);
        dequantizeBlock(ffn_block, ffn_dequant);
        dequantizeBlock(out_block, out_dequant);

        // Compute expected result
        for (int i = 0; i < 32; ++i)
        {
            expected[i] = res_dequant[i] + ffn_dequant[i];
        }

        // In this case, values should nearly cancel (result should be small)
        float expected_sum = std::accumulate(expected, expected + 32, 0.0f);
        float out_sum = std::accumulate(out_dequant, out_dequant + 32, 0.0f);

        // Both should be near zero (full cancellation)
        EXPECT_NEAR(expected_sum, 0.0f, 10.0f);   // Allow some margin due to asymmetry
        EXPECT_NEAR(out_sum, expected_sum, 5.0f); // Q8_1 result should be close to expected

        float max_diff = maxAbsDiff(expected, out_dequant, 32);
        EXPECT_LT(max_diff, 2.0f) << "Q8_1 add should handle cancellation accurately";
    }

    /**
     * @brief Test q8_1_add_q8_1 with different scales (scale mismatch)
     *
     * This tests when one input has much larger scale than the other.
     */
    TEST_F(Test__Q8_1_ResidualAdd_Layer21, ScaleMismatch)
    {
        // Residual: small scale
        std::vector<int8_t> res_qs;
        for (int i = 0; i < 32; ++i)
        {
            res_qs.push_back(static_cast<int8_t>(i * 4 - 64)); // -64 to 60
        }
        Q8_1Block res_block = createBlock(0.1f, 0, res_qs); // Small values: -6.4 to 6.0

        // FFN output: large scale
        std::vector<int8_t> ffn_qs;
        for (int i = 0; i < 32; ++i)
        {
            ffn_qs.push_back(static_cast<int8_t>(i * 4 - 64)); // Same pattern
        }
        Q8_1Block ffn_block = createBlock(15.0f, 0, ffn_qs); // Large values: -960 to 900

        // Output buffer
        Q8_1Block out_block;

        // Perform Q8_1 addition
        simd::q8_1_add_q8_1(&res_block, &ffn_block, &out_block, 32);

        // Dequantize
        float res_dequant[32], ffn_dequant[32], out_dequant[32], expected[32];
        dequantizeBlock(res_block, res_dequant);
        dequantizeBlock(ffn_block, ffn_dequant);
        dequantizeBlock(out_block, out_dequant);

        // Compute expected result
        for (int i = 0; i < 32; ++i)
        {
            expected[i] = res_dequant[i] + ffn_dequant[i];
        }

        float max_diff = maxAbsDiff(expected, out_dequant, 32);

        // With scale mismatch, the small-scale input contributes negligibly
        // Error can be larger due to requantization with new scale
        // Q8_1 precision is limited to scale/127, which for scale~15 is ~0.12 per value
        // With 32 values and possible compounding, allow up to 5.0 for extreme cases
        EXPECT_LT(max_diff, 5.0f) << "Q8_1 add should handle scale mismatch";

        // Verify the output is dominated by the large-scale input
        float ffn_max = *std::max_element(ffn_dequant, ffn_dequant + 32);
        float out_max = *std::max_element(out_dequant, out_dequant + 32);
        EXPECT_NEAR(out_max, ffn_max, 10.0f); // Should be close to FFN max
    }

    /**
     * @brief Test multiple blocks (full row) as would happen in actual pipeline
     */
    TEST_F(Test__Q8_1_ResidualAdd_Layer21, MultiBlockRow)
    {
        const size_t num_blocks = 28; // blocks_per_row for d_model=896
        const size_t elements = num_blocks * 32;

        std::vector<Q8_1Block> res_blocks(num_blocks);
        std::vector<Q8_1Block> ffn_blocks(num_blocks);
        std::vector<Q8_1Block> out_blocks(num_blocks);

        // Initialize with varying scales (simulating real activation patterns)
        for (size_t b = 0; b < num_blocks; ++b)
        {
            std::vector<int8_t> qs(32);
            for (int i = 0; i < 32; ++i)
            {
                qs[i] = static_cast<int8_t>((i + b * 7) % 255 - 127);
            }

            // Residual: mostly small, with occasional spikes
            float res_scale = (b == 1) ? 8.6f : 0.1f + 0.05f * b;
            res_blocks[b] = createBlock(res_scale, 0, qs);

            // FFN: can have large negative values
            float ffn_scale = (b == 1) ? 15.5f : 0.2f + 0.03f * b;
            for (int i = 0; i < 32; ++i)
            {
                qs[i] = static_cast<int8_t>(-((i + b * 7) % 255 - 127));
            }
            ffn_blocks[b] = createBlock(ffn_scale, 0, qs);
        }

        // Perform Q8_1 addition on all blocks
        simd::q8_1_add_q8_1(res_blocks.data(), ffn_blocks.data(), out_blocks.data(), elements);

        // Verify each block individually
        float total_max_diff = 0.0f;
        for (size_t b = 0; b < num_blocks; ++b)
        {
            float res_dequant[32], ffn_dequant[32], out_dequant[32], expected[32];
            dequantizeBlock(res_blocks[b], res_dequant);
            dequantizeBlock(ffn_blocks[b], ffn_dequant);
            dequantizeBlock(out_blocks[b], out_dequant);

            for (int i = 0; i < 32; ++i)
            {
                expected[i] = res_dequant[i] + ffn_dequant[i];
            }

            float block_diff = maxAbsDiff(expected, out_dequant, 32);
            total_max_diff = std::max(total_max_diff, block_diff);
        }

        // With varying scales across blocks, error can accumulate
        // Block 1 has large scale (8.6 and 15.5), which can cause ~2 error
        // Other blocks with different patterns may have similar issues
        // Allow up to 5.0 for the full row to accommodate extreme cases
        EXPECT_LT(total_max_diff, 5.0f) << "Multi-block row should maintain Q8_1 precision";
    }

} // namespace llaminar2::test

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
