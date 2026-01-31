/**
 * @file Test__CPUSoftmaxKernelT.cpp
 * @brief Unit tests for CPUSoftmaxKernelT (FP32/BF16/FP16/Q8_1 specializations)
 * @author David Sanftenberg
 *
 * Tests verify:
 * 1. Basic functionality for each precision
 * 2. Causal masking correctness
 * 3. Scale factor application
 * 4. Cross-precision consistency (all precisions produce similar probabilities)
 * 5. Q8_1 integer-aware path produces results comparable to FP32 reference
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include <vector>
#include <numeric>
#include <random>
#include <algorithm>

#include "kernels/cpu/ops/CPUSoftmaxKernelT.h"
#include "tensors/BlockStructures.h"
#include "tensors/FP16Utils.h"
#include "tensors/SIMDHelpers.h"

namespace llaminar2
{
    namespace
    {
        // Use shared conversion utilities from SIMDHelpers.h
        using simd::bf16_to_fp32;
        using simd::fp32_to_bf16;
        // FP16 utilities from FP16Utils.h are in llaminar2:: namespace

        // ============================================================================
        // Helper Functions
        // ============================================================================

        /**
         * @brief Reference FP32 softmax implementation
         */
        void reference_softmax_fp32(std::vector<float> &row, bool causal, int row_idx, float scale)
        {
            int cols = static_cast<int>(row.size());

            // Apply scale and find max (with causal masking)
            float max_val = -std::numeric_limits<float>::infinity();
            int valid_cols = causal ? std::min(row_idx + 1, cols) : cols;

            for (int j = 0; j < valid_cols; ++j)
            {
                row[j] *= scale;
                max_val = std::max(max_val, row[j]);
            }

            // Compute exp and sum
            float sum = 0.0f;
            for (int j = 0; j < valid_cols; ++j)
            {
                row[j] = std::exp(row[j] - max_val);
                sum += row[j];
            }

            // Normalize
            for (int j = 0; j < valid_cols; ++j)
            {
                row[j] /= sum;
            }

            // Zero out masked positions
            for (int j = valid_cols; j < cols; ++j)
            {
                row[j] = 0.0f;
            }
        }

        /**
         * @brief Compute cosine similarity between two vectors
         */
        float cosine_similarity(const std::vector<float> &a, const std::vector<float> &b)
        {
            if (a.size() != b.size() || a.empty())
                return 0.0f;

            float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
            for (size_t i = 0; i < a.size(); ++i)
            {
                dot += a[i] * b[i];
                norm_a += a[i] * a[i];
                norm_b += b[i] * b[i];
            }

            if (norm_a < 1e-12f || norm_b < 1e-12f)
                return 1.0f; // Both zero vectors

            return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
        }

        /**
         * @brief Check if softmax output is valid (sums to ~1, all >= 0)
         */
        bool is_valid_softmax(const std::vector<float> &row, float tol = 0.01f)
        {
            float sum = 0.0f;
            for (float v : row)
            {
                if (v < -1e-6f)
                    return false; // Negative probability
                sum += v;
            }
            return std::abs(sum - 1.0f) < tol;
        }

        /**
         * @brief Create Q8_1 block from FP32 values
         */
        Q8_1Block create_q8_1_block(const float *values)
        {
            Q8_1Block block;

            // Find max absolute value for scale
            float max_abs = 0.0f;
            for (int i = 0; i < 32; ++i)
            {
                max_abs = std::max(max_abs, std::abs(values[i]));
            }

            // Compute scale (d)
            float scale = max_abs / 127.0f;
            if (scale < 1e-10f)
                scale = 1e-10f;

            // Store scale as BF16
            block.d = fp32_to_bf16(scale);

            // Quantize and compute sum
            int32_t sum_qs = 0;
            for (int i = 0; i < 32; ++i)
            {
                int32_t q = static_cast<int32_t>(std::round(values[i] / scale));
                q = std::max(-128, std::min(127, q));
                block.qs[i] = static_cast<int8_t>(q);
                sum_qs += block.qs[i];
            }
            block.sum_qs = static_cast<int16_t>(sum_qs);

            return block;
        }

        /**
         * @brief Dequantize Q8_1 block to FP32
         */
        void dequantize_q8_1_block(const Q8_1Block &block, float *output)
        {
            float scale = bf16_to_fp32(block.d);
            for (int i = 0; i < 32; ++i)
            {
                output[i] = static_cast<float>(block.qs[i]) * scale;
            }
        }

    } // anonymous namespace

    // ============================================================================
    // FP32 Softmax Tests
    // ============================================================================

    TEST(CPUSoftmaxKernelT_FP32, BasicFunctionality)
    {
        CPUSoftmaxKernelT<ActivationPrecision::FP32> kernel;

        // Create test data
        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> expected = data;
        reference_softmax_fp32(expected, false, 0, 1.0f);

        // Apply kernel
        ASSERT_TRUE(kernel.apply_typed(data.data(), 1, 4, false, 1.0f));

        // Verify results
        for (size_t i = 0; i < data.size(); ++i)
        {
            EXPECT_NEAR(data[i], expected[i], 1e-5f) << "Mismatch at index " << i;
        }

        EXPECT_TRUE(is_valid_softmax(data));
    }

    TEST(CPUSoftmaxKernelT_FP32, CausalMasking)
    {
        CPUSoftmaxKernelT<ActivationPrecision::FP32> kernel;

        // 4 rows, 4 columns with causal masking
        std::vector<float> data(16);
        std::iota(data.begin(), data.end(), 1.0f);

        ASSERT_TRUE(kernel.apply_typed(data.data(), 4, 4, true, 1.0f));

        // Row 0: only position 0 valid -> should be 1.0
        EXPECT_NEAR(data[0], 1.0f, 1e-5f);
        EXPECT_NEAR(data[1], 0.0f, 1e-5f);
        EXPECT_NEAR(data[2], 0.0f, 1e-5f);
        EXPECT_NEAR(data[3], 0.0f, 1e-5f);

        // Row 1: positions 0,1 valid
        EXPECT_GT(data[4], 0.0f);
        EXPECT_GT(data[5], 0.0f);
        EXPECT_NEAR(data[6], 0.0f, 1e-5f);
        EXPECT_NEAR(data[7], 0.0f, 1e-5f);
        EXPECT_NEAR(data[4] + data[5], 1.0f, 0.01f);
    }

    TEST(CPUSoftmaxKernelT_FP32, ScaleFactor)
    {
        CPUSoftmaxKernelT<ActivationPrecision::FP32> kernel;

        std::vector<float> data1 = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> data2 = data1;

        // Apply with different scales
        ASSERT_TRUE(kernel.apply_typed(data1.data(), 1, 4, false, 1.0f));
        ASSERT_TRUE(kernel.apply_typed(data2.data(), 1, 4, false, 0.5f));

        // Both should be valid softmax distributions
        EXPECT_TRUE(is_valid_softmax(data1));
        EXPECT_TRUE(is_valid_softmax(data2));

        // Scale 0.5 should make distribution more uniform
        // (smaller difference between max and min probabilities)
        float range1 = *std::max_element(data1.begin(), data1.end()) - *std::min_element(data1.begin(), data1.end());
        float range2 = *std::max_element(data2.begin(), data2.end()) - *std::min_element(data2.begin(), data2.end());
        EXPECT_LT(range2, range1);
    }

    TEST(CPUSoftmaxKernelT_FP32, MultipleRows)
    {
        CPUSoftmaxKernelT<ActivationPrecision::FP32> kernel;

        const int rows = 8;
        const int cols = 64;
        std::vector<float> data(rows * cols);

        // Fill with random data
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
        for (auto &v : data)
            v = dist(gen);

        ASSERT_TRUE(kernel.apply_typed(data.data(), rows, cols, false, 1.0f));

        // Each row should be valid softmax
        for (int r = 0; r < rows; ++r)
        {
            std::vector<float> row(data.begin() + r * cols, data.begin() + (r + 1) * cols);
            EXPECT_TRUE(is_valid_softmax(row)) << "Row " << r << " is not valid softmax";
        }
    }

    // ============================================================================
    // BF16 Softmax Tests
    // ============================================================================

    TEST(CPUSoftmaxKernelT_BF16, BasicFunctionality)
    {
        CPUSoftmaxKernelT<ActivationPrecision::BF16> kernel;

        // Create test data
        std::vector<float> fp32_data = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<uint16_t> bf16_data(4);
        for (size_t i = 0; i < fp32_data.size(); ++i)
        {
            bf16_data[i] = fp32_to_bf16(fp32_data[i]);
        }

        // Apply kernel
        ASSERT_TRUE(kernel.apply_typed(bf16_data.data(), 1, 4, false, 1.0f));

        // Convert back to FP32 and verify
        std::vector<float> result(4);
        for (size_t i = 0; i < 4; ++i)
        {
            result[i] = bf16_to_fp32(bf16_data[i]);
        }

        EXPECT_TRUE(is_valid_softmax(result, 0.02f)); // Slightly larger tolerance for BF16
    }

    TEST(CPUSoftmaxKernelT_BF16, ConsistencyWithFP32)
    {
        CPUSoftmaxKernelT<ActivationPrecision::FP32> kernel_fp32;
        CPUSoftmaxKernelT<ActivationPrecision::BF16> kernel_bf16;

        // Create same input data
        std::vector<float> fp32_input = {-1.5f, 0.5f, 1.0f, 2.5f, -0.5f, 1.5f, 0.0f, 3.0f};
        std::vector<uint16_t> bf16_input(8);
        for (size_t i = 0; i < 8; ++i)
        {
            bf16_input[i] = fp32_to_bf16(fp32_input[i]);
        }

        // Apply kernels
        ASSERT_TRUE(kernel_fp32.apply_typed(fp32_input.data(), 1, 8, false, 1.0f));
        ASSERT_TRUE(kernel_bf16.apply_typed(bf16_input.data(), 1, 8, false, 1.0f));

        // Convert BF16 result to FP32
        std::vector<float> bf16_result(8);
        for (size_t i = 0; i < 8; ++i)
        {
            bf16_result[i] = bf16_to_fp32(bf16_input[i]);
        }

        // Check cosine similarity
        float sim = cosine_similarity(fp32_input, bf16_result);
        EXPECT_GT(sim, 0.99f) << "BF16 result diverges from FP32";
    }

    // ============================================================================
    // FP16 Softmax Tests
    // ============================================================================

    TEST(CPUSoftmaxKernelT_FP16, BasicFunctionality)
    {
        CPUSoftmaxKernelT<ActivationPrecision::FP16> kernel;

        // Create test data
        std::vector<float> fp32_data = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<uint16_t> fp16_data(4);
        for (size_t i = 0; i < fp32_data.size(); ++i)
        {
            fp16_data[i] = fp32_to_fp16(fp32_data[i]);
        }

        // Apply kernel
        ASSERT_TRUE(kernel.apply_typed(fp16_data.data(), 1, 4, false, 1.0f));

        // Convert back to FP32 and verify
        std::vector<float> result(4);
        for (size_t i = 0; i < 4; ++i)
        {
            result[i] = fp16_to_fp32(fp16_data[i]);
        }

        EXPECT_TRUE(is_valid_softmax(result, 0.02f)); // Slightly larger tolerance for FP16
    }

    TEST(CPUSoftmaxKernelT_FP16, ConsistencyWithFP32)
    {
        CPUSoftmaxKernelT<ActivationPrecision::FP32> kernel_fp32;
        CPUSoftmaxKernelT<ActivationPrecision::FP16> kernel_fp16;

        // Create same input data
        std::vector<float> fp32_input = {-1.5f, 0.5f, 1.0f, 2.5f, -0.5f, 1.5f, 0.0f, 3.0f};
        std::vector<uint16_t> fp16_input(8);
        for (size_t i = 0; i < 8; ++i)
        {
            fp16_input[i] = fp32_to_fp16(fp32_input[i]);
        }

        // Apply kernels
        ASSERT_TRUE(kernel_fp32.apply_typed(fp32_input.data(), 1, 8, false, 1.0f));
        ASSERT_TRUE(kernel_fp16.apply_typed(fp16_input.data(), 1, 8, false, 1.0f));

        // Convert FP16 result to FP32
        std::vector<float> fp16_result(8);
        for (size_t i = 0; i < 8; ++i)
        {
            fp16_result[i] = fp16_to_fp32(fp16_input[i]);
        }

        // Check cosine similarity
        float sim = cosine_similarity(fp32_input, fp16_result);
        EXPECT_GT(sim, 0.99f) << "FP16 result diverges from FP32";
    }

    // ============================================================================
    // Q8_1 Softmax Tests
    // ============================================================================

    TEST(CPUSoftmaxKernelT_Q8_1, BasicFunctionality)
    {
        CPUSoftmaxKernelT<ActivationPrecision::Q8_1> kernel;

        // Create test data: 1 row with 2 blocks (64 elements)
        std::vector<float> fp32_data(64);
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
        for (auto &v : fp32_data)
            v = dist(gen);

        // Make one element clearly the maximum
        fp32_data[32] = 5.0f;

        // Convert to Q8_1 blocks
        std::vector<Q8_1Block> blocks(2);
        blocks[0] = create_q8_1_block(fp32_data.data());
        blocks[1] = create_q8_1_block(fp32_data.data() + 32);

        // Apply kernel
        ASSERT_TRUE(kernel.apply_typed(blocks.data(), 1, 2, false, 1.0f));

        // Dequantize result
        std::vector<float> result(64);
        dequantize_q8_1_block(blocks[0], result.data());
        dequantize_q8_1_block(blocks[1], result.data() + 32);

        // Verify: output should be non-negative probabilities
        for (float v : result)
        {
            EXPECT_GE(v, -0.01f) << "Probability should be non-negative";
        }

        // Sum should be approximately 1
        float sum = std::accumulate(result.begin(), result.end(), 0.0f);
        EXPECT_NEAR(sum, 1.0f, 0.1f) << "Probabilities should sum to ~1";
    }

    TEST(CPUSoftmaxKernelT_Q8_1, ConsistencyWithFP32)
    {
        CPUSoftmaxKernelT<ActivationPrecision::FP32> kernel_fp32;
        CPUSoftmaxKernelT<ActivationPrecision::Q8_1> kernel_q8;

        // Create test data: 1 row with 2 blocks (64 elements)
        std::vector<float> fp32_data(64);
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
        for (auto &v : fp32_data)
            v = dist(gen);

        // Make distribution more peaky for better quantization
        fp32_data[32] = 5.0f;

        // Create Q8_1 version
        std::vector<Q8_1Block> blocks(2);
        blocks[0] = create_q8_1_block(fp32_data.data());
        blocks[1] = create_q8_1_block(fp32_data.data() + 32);

        // Dequantize Q8_1 to get actual input values (may differ due to quantization)
        std::vector<float> q8_input(64);
        dequantize_q8_1_block(blocks[0], q8_input.data());
        dequantize_q8_1_block(blocks[1], q8_input.data() + 32);

        // Apply FP32 kernel to dequantized values (ground truth)
        std::vector<float> fp32_result = q8_input;
        ASSERT_TRUE(kernel_fp32.apply_typed(fp32_result.data(), 1, 64, false, 1.0f));

        // Apply Q8_1 kernel
        ASSERT_TRUE(kernel_q8.apply_typed(blocks.data(), 1, 2, false, 1.0f));

        // Dequantize Q8_1 result
        std::vector<float> q8_result(64);
        dequantize_q8_1_block(blocks[0], q8_result.data());
        dequantize_q8_1_block(blocks[1], q8_result.data() + 32);

        // Compare results
        float sim = cosine_similarity(fp32_result, q8_result);
        EXPECT_GT(sim, 0.95f) << "Q8_1 result diverges significantly from FP32 reference";
    }

    TEST(CPUSoftmaxKernelT_Q8_1, CausalMasking)
    {
        CPUSoftmaxKernelT<ActivationPrecision::Q8_1> kernel;

        // Create test data: 2 rows with 2 blocks each (64 elements per row)
        const int rows = 2;
        const int blocks_per_row = 2;
        std::vector<Q8_1Block> blocks(rows * blocks_per_row);

        // Fill with test data
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        for (int r = 0; r < rows; ++r)
        {
            for (int b = 0; b < blocks_per_row; ++b)
            {
                float values[32];
                for (int i = 0; i < 32; ++i)
                {
                    values[i] = dist(gen);
                }
                blocks[r * blocks_per_row + b] = create_q8_1_block(values);
            }
        }

        // Apply with causal masking
        ASSERT_TRUE(kernel.apply_typed(blocks.data(), rows, blocks_per_row, true, 1.0f));

        // Dequantize results
        std::vector<float> result(rows * 64);
        for (int r = 0; r < rows; ++r)
        {
            dequantize_q8_1_block(blocks[r * blocks_per_row], result.data() + r * 64);
            dequantize_q8_1_block(blocks[r * blocks_per_row + 1], result.data() + r * 64 + 32);
        }

        // Row 0: only position 0 valid (approximately 1.0, rest ~0)
        float row0_sum = 0.0f;
        for (int j = 0; j < 64; ++j)
        {
            if (j > 0)
            {
                EXPECT_NEAR(result[j], 0.0f, 0.1f) << "Row 0, position " << j << " should be masked";
            }
            row0_sum += result[j];
        }
        EXPECT_NEAR(row0_sum, 1.0f, 0.15f) << "Row 0 probabilities should sum to ~1";
    }

    TEST(CPUSoftmaxKernelT_Q8_1, MultipleRowsStability)
    {
        CPUSoftmaxKernelT<ActivationPrecision::Q8_1> kernel;

        // Create larger test: 8 rows, 4 blocks per row (128 elements per row)
        const int rows = 8;
        const int blocks_per_row = 4;
        std::vector<Q8_1Block> blocks(rows * blocks_per_row);

        std::mt19937 gen(123);
        std::uniform_real_distribution<float> dist(-3.0f, 3.0f);

        for (int r = 0; r < rows; ++r)
        {
            for (int b = 0; b < blocks_per_row; ++b)
            {
                float values[32];
                for (int i = 0; i < 32; ++i)
                {
                    values[i] = dist(gen);
                }
                // Make one element dominant per row
                if (b == r % blocks_per_row)
                {
                    values[r % 32] = 8.0f;
                }
                blocks[r * blocks_per_row + b] = create_q8_1_block(values);
            }
        }

        // Apply kernel
        ASSERT_TRUE(kernel.apply_typed(blocks.data(), rows, blocks_per_row, false, 1.0f));

        // Verify each row
        for (int r = 0; r < rows; ++r)
        {
            float row_sum = 0.0f;
            for (int b = 0; b < blocks_per_row; ++b)
            {
                const Q8_1Block &block = blocks[r * blocks_per_row + b];
                float scale = bf16_to_fp32(block.d);
                for (int i = 0; i < 32; ++i)
                {
                    float prob = static_cast<float>(block.qs[i]) * scale;
                    EXPECT_GE(prob, -0.05f) << "Row " << r << " has negative probability";
                    row_sum += prob;
                }
            }
            EXPECT_NEAR(row_sum, 1.0f, 0.15f) << "Row " << r << " probabilities don't sum to 1";
        }
    }

    // ============================================================================
    // Cross-Precision Consistency Tests
    // ============================================================================

    TEST(CPUSoftmaxKernelT_CrossPrecision, AllPrecisionsProduceSimilarResults)
    {
        CPUSoftmaxKernelT<ActivationPrecision::FP32> kernel_fp32;
        CPUSoftmaxKernelT<ActivationPrecision::BF16> kernel_bf16;
        CPUSoftmaxKernelT<ActivationPrecision::FP16> kernel_fp16;
        CPUSoftmaxKernelT<ActivationPrecision::Q8_1> kernel_q8;

        // Create base FP32 data
        std::vector<float> base_data = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                                        -1.5f, 0.5f, 1.5f, 2.5f, -0.5f, 0.5f, 1.5f, 2.5f,
                                        0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f,
                                        -0.8f, -0.6f, -0.4f, -0.2f, 0.2f, 0.4f, 0.6f, 0.8f};

        // FP32 reference
        std::vector<float> fp32_data = base_data;
        ASSERT_TRUE(kernel_fp32.apply_typed(fp32_data.data(), 1, 32, false, 1.0f));

        // BF16
        std::vector<uint16_t> bf16_data(32);
        for (size_t i = 0; i < 32; ++i)
            bf16_data[i] = fp32_to_bf16(base_data[i]);
        ASSERT_TRUE(kernel_bf16.apply_typed(bf16_data.data(), 1, 32, false, 1.0f));
        std::vector<float> bf16_result(32);
        for (size_t i = 0; i < 32; ++i)
            bf16_result[i] = bf16_to_fp32(bf16_data[i]);

        // FP16
        std::vector<uint16_t> fp16_data(32);
        for (size_t i = 0; i < 32; ++i)
            fp16_data[i] = fp32_to_fp16(base_data[i]);
        ASSERT_TRUE(kernel_fp16.apply_typed(fp16_data.data(), 1, 32, false, 1.0f));
        std::vector<float> fp16_result(32);
        for (size_t i = 0; i < 32; ++i)
            fp16_result[i] = fp16_to_fp32(fp16_data[i]);

        // Q8_1 (single block of 32 elements)
        Q8_1Block q8_block = create_q8_1_block(base_data.data());
        ASSERT_TRUE(kernel_q8.apply_typed(&q8_block, 1, 1, false, 1.0f));
        std::vector<float> q8_result(32);
        dequantize_q8_1_block(q8_block, q8_result.data());

        // Compare all precisions against FP32 reference
        float bf16_sim = cosine_similarity(fp32_data, bf16_result);
        float fp16_sim = cosine_similarity(fp32_data, fp16_result);
        float q8_sim = cosine_similarity(fp32_data, q8_result);

        EXPECT_GT(bf16_sim, 0.99f) << "BF16 diverges from FP32";
        EXPECT_GT(fp16_sim, 0.99f) << "FP16 diverges from FP32";
        EXPECT_GT(q8_sim, 0.95f) << "Q8_1 diverges significantly from FP32";

        // All should produce valid softmax
        EXPECT_TRUE(is_valid_softmax(fp32_data)) << "FP32 not valid softmax";
        EXPECT_TRUE(is_valid_softmax(bf16_result, 0.02f)) << "BF16 not valid softmax";
        EXPECT_TRUE(is_valid_softmax(fp16_result, 0.02f)) << "FP16 not valid softmax";
        // Q8_1 has looser tolerance due to requantization
        float q8_sum = std::accumulate(q8_result.begin(), q8_result.end(), 0.0f);
        EXPECT_NEAR(q8_sum, 1.0f, 0.15f) << "Q8_1 probabilities don't sum to ~1";
    }

    // ============================================================================
    // Edge Cases
    // ============================================================================

    TEST(CPUSoftmaxKernelT_EdgeCases, EmptyInput)
    {
        CPUSoftmaxKernelT<ActivationPrecision::FP32> kernel_fp32;
        CPUSoftmaxKernelT<ActivationPrecision::Q8_1> kernel_q8;

        float fp32_data[1] = {0.0f};
        Q8_1Block q8_data[1] = {};

        // Empty should succeed (no-op)
        EXPECT_TRUE(kernel_fp32.apply_typed(fp32_data, 0, 4, false, 1.0f));
        EXPECT_TRUE(kernel_fp32.apply_typed(fp32_data, 1, 0, false, 1.0f));
        EXPECT_TRUE(kernel_q8.apply_typed(q8_data, 0, 2, false, 1.0f));
        EXPECT_TRUE(kernel_q8.apply_typed(q8_data, 1, 0, false, 1.0f));
    }

    TEST(CPUSoftmaxKernelT_EdgeCases, NullPointer)
    {
        CPUSoftmaxKernelT<ActivationPrecision::FP32> kernel_fp32;
        CPUSoftmaxKernelT<ActivationPrecision::Q8_1> kernel_q8;

        // Null pointers should fail
        EXPECT_FALSE(kernel_fp32.apply_typed(nullptr, 1, 4, false, 1.0f));
        EXPECT_FALSE(kernel_q8.apply_typed(nullptr, 1, 2, false, 1.0f));
    }

    TEST(CPUSoftmaxKernelT_EdgeCases, SingleElement)
    {
        CPUSoftmaxKernelT<ActivationPrecision::FP32> kernel;

        std::vector<float> data = {5.0f};
        ASSERT_TRUE(kernel.apply_typed(data.data(), 1, 1, false, 1.0f));

        // Single element softmax should be 1.0
        EXPECT_NEAR(data[0], 1.0f, 1e-5f);
    }

    TEST(CPUSoftmaxKernelT_EdgeCases, LargeValues)
    {
        CPUSoftmaxKernelT<ActivationPrecision::FP32> kernel;

        // Large values that could cause overflow without proper max subtraction
        std::vector<float> data = {100.0f, 101.0f, 102.0f, 103.0f};
        ASSERT_TRUE(kernel.apply_typed(data.data(), 1, 4, false, 1.0f));

        // Should still produce valid softmax (no inf/nan)
        EXPECT_TRUE(is_valid_softmax(data));
        for (float v : data)
        {
            EXPECT_FALSE(std::isnan(v));
            EXPECT_FALSE(std::isinf(v));
        }
    }

    TEST(CPUSoftmaxKernelT_EdgeCases, NegativeValues)
    {
        CPUSoftmaxKernelT<ActivationPrecision::FP32> kernel;

        std::vector<float> data = {-100.0f, -50.0f, -1.0f, 0.0f};
        ASSERT_TRUE(kernel.apply_typed(data.data(), 1, 4, false, 1.0f));

        // Should still produce valid softmax
        EXPECT_TRUE(is_valid_softmax(data));
    }

    // ============================================================================
    // Metadata and Interface Tests
    // ============================================================================

    TEST(CPUSoftmaxKernelT_Metadata, PrecisionNames)
    {
        EXPECT_STREQ(CPUSoftmaxKernelT<ActivationPrecision::FP32>::precision_name(), "FP32");
        EXPECT_STREQ(CPUSoftmaxKernelT<ActivationPrecision::BF16>::precision_name(), "BF16");
        EXPECT_STREQ(CPUSoftmaxKernelT<ActivationPrecision::FP16>::precision_name(), "FP16");
        EXPECT_STREQ(CPUSoftmaxKernelT<ActivationPrecision::Q8_1>::precision_name(), "Q8_1");
    }

    TEST(CPUSoftmaxKernelT_Metadata, CompressionRatios)
    {
        EXPECT_FLOAT_EQ(CPUSoftmaxKernelT<ActivationPrecision::FP32>::compression_ratio(), 1.0f);
        EXPECT_FLOAT_EQ(CPUSoftmaxKernelT<ActivationPrecision::BF16>::compression_ratio(), 2.0f);
        EXPECT_FLOAT_EQ(CPUSoftmaxKernelT<ActivationPrecision::FP16>::compression_ratio(), 2.0f);
        EXPECT_FLOAT_EQ(CPUSoftmaxKernelT<ActivationPrecision::Q8_1>::compression_ratio(), 4.0f);
    }

    TEST(CPUSoftmaxKernelT_Metadata, DeviceSupport)
    {
        CPUSoftmaxKernelT<ActivationPrecision::FP32> kernel_fp32;
        CPUSoftmaxKernelT<ActivationPrecision::Q8_1> kernel_q8;

        // CPU only (device -1)
        EXPECT_TRUE(kernel_fp32.supports_device(-1));
        EXPECT_FALSE(kernel_fp32.supports_device(0));
        EXPECT_FALSE(kernel_fp32.supports_device(1));

        EXPECT_TRUE(kernel_q8.supports_device(-1));
        EXPECT_FALSE(kernel_q8.supports_device(0));
    }

} // namespace llaminar2
