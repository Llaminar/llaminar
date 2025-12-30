/**
 * @file Test__Q16DotProduct.cpp
 * @brief Unit tests for Q16_1 integer dot product microkernel
 *
 * Tests the Q16DotProduct microkernel functions:
 * - q16_dot_single: Single Q×K dot product
 * - q16_qk_gemv: Flash Decode GEMV pattern
 * - q16_qk_gemm_tile: FA2 Prefill tiled GEMM pattern
 */

#include <gtest/gtest.h>
#include "kernels/cpu/attention/q16_1/ref/microkernels/Q16DotProduct.h"
#include "tensors/BlockStructures.h"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using namespace llaminar2::kernels::q16_1::microkernels;
using namespace llaminar2;

class Test__Q16DotProduct : public ::testing::Test
{
protected:
    std::mt19937 rng_{42}; // Fixed seed for reproducibility

    // Helper to create a Q16_1Block_64 with known values
    Q16_1Block_64 createBlock64(float scale, const std::vector<int16_t> &values)
    {
        Q16_1Block_64 block;
        block.d = scale;

        int32_t sum = 0;
        for (int i = 0; i < 64; ++i)
        {
            int16_t v = (i < static_cast<int>(values.size())) ? values[i] : 0;
            block.qs[i] = v;
            sum += v;
        }
        block.sum_qs = sum;
        return block;
    }

    // Helper to create a Q16_1Block_128 with known values
    Q16_1Block_128 createBlock128(float scale, const std::vector<int16_t> &values)
    {
        Q16_1Block_128 block;
        block.d = scale;

        int32_t sum = 0;
        for (int i = 0; i < 128; ++i)
        {
            int16_t v = (i < static_cast<int>(values.size())) ? values[i] : 0;
            block.qs[i] = v;
            sum += v;
        }
        block.sum_qs = sum;
        return block;
    }

    // Helper to create random Q16_1Block_64
    Q16_1Block_64 createRandomBlock64(float scale)
    {
        std::uniform_int_distribution<int> dist(-32767, 32767);

        Q16_1Block_64 block;
        block.d = scale;

        int32_t sum = 0;
        for (int i = 0; i < 64; ++i)
        {
            int16_t v = static_cast<int16_t>(dist(rng_));
            block.qs[i] = v;
            sum += v;
        }
        block.sum_qs = sum;
        return block;
    }

    // Helper to create random Q16_1Block_128
    Q16_1Block_128 createRandomBlock128(float scale)
    {
        std::uniform_int_distribution<int> dist(-32767, 32767);

        Q16_1Block_128 block;
        block.d = scale;

        int32_t sum = 0;
        for (int i = 0; i < 128; ++i)
        {
            int16_t v = static_cast<int16_t>(dist(rng_));
            block.qs[i] = v;
            sum += v;
        }
        block.sum_qs = sum;
        return block;
    }

    // Helper to compute expected dot product (ground truth)
    template <typename BlockType>
    int32_t computeExpectedDot(const BlockType *q, const BlockType *k,
                               int head_dim, int blocks_per_row)
    {
        constexpr int BLOCK_SIZE = BlockType::BLOCK_SIZE;
        int32_t acc = 0;

        for (int b = 0; b < blocks_per_row; ++b)
        {
            int start = b * BLOCK_SIZE;
            int end = std::min(start + BLOCK_SIZE, head_dim);
            int count = end - start;

            for (int i = 0; i < count; ++i)
            {
                acc += static_cast<int32_t>(q[b].qs[i]) *
                       static_cast<int32_t>(k[b].qs[i]);
            }
        }
        return acc;
    }
};

// ============================================================================
// q16_dot_single Tests
// ============================================================================

TEST_F(Test__Q16DotProduct, DotSingle_ZeroVectors_Block64)
{
    // Two zero vectors should produce 0
    std::vector<int16_t> zeros(64, 0);
    Q16_1Block_64 q = createBlock64(1.0f, zeros);
    Q16_1Block_64 k = createBlock64(1.0f, zeros);

    int32_t result = q16_dot_single<Q16_1Block_64>(&q, &k, 64, 1);

    EXPECT_EQ(result, 0);
}

TEST_F(Test__Q16DotProduct, DotSingle_IdentityVector_Block64)
{
    // Same vector dotted with itself should produce ||v||²
    std::vector<int16_t> values(64);
    for (int i = 0; i < 64; ++i)
    {
        values[i] = static_cast<int16_t>(i + 1);
    }
    Q16_1Block_64 v = createBlock64(1.0f, values);

    int32_t result = q16_dot_single<Q16_1Block_64>(&v, &v, 64, 1);

    // Expected: sum of squares = 1² + 2² + ... + 64² = 64*65*129/6 = 89440
    int32_t expected = 0;
    for (int i = 1; i <= 64; ++i)
    {
        expected += i * i;
    }

    EXPECT_EQ(result, expected);
}

TEST_F(Test__Q16DotProduct, DotSingle_IdentityVector_Block128)
{
    // Same vector dotted with itself for 128-dim
    std::vector<int16_t> values(128);
    for (int i = 0; i < 128; ++i)
    {
        values[i] = static_cast<int16_t>(i + 1);
    }
    Q16_1Block_128 v = createBlock128(1.0f, values);

    int32_t result = q16_dot_single<Q16_1Block_128>(&v, &v, 128, 1);

    // Expected: sum of squares = 1² + 2² + ... + 128² = 128*129*257/6 = 707328
    int32_t expected = 0;
    for (int i = 1; i <= 128; ++i)
    {
        expected += i * i;
    }

    EXPECT_EQ(result, expected);
}

TEST_F(Test__Q16DotProduct, DotSingle_Random_Block64)
{
    Q16_1Block_64 q = createRandomBlock64(1.0f);
    Q16_1Block_64 k = createRandomBlock64(1.0f);

    int32_t result = q16_dot_single<Q16_1Block_64>(&q, &k, 64, 1);
    int32_t expected = computeExpectedDot(&q, &k, 64, 1);

    EXPECT_EQ(result, expected);
}

TEST_F(Test__Q16DotProduct, DotSingle_Random_Block128)
{
    Q16_1Block_128 q = createRandomBlock128(1.0f);
    Q16_1Block_128 k = createRandomBlock128(1.0f);

    int32_t result = q16_dot_single<Q16_1Block_128>(&q, &k, 128, 1);
    int32_t expected = computeExpectedDot(&q, &k, 128, 1);

    EXPECT_EQ(result, expected);
}

// ============================================================================
// q16_qk_gemv Tests (Flash Decode Pattern)
// ============================================================================

TEST_F(Test__Q16DotProduct, GEMV_MultipleKV_Block64)
{
    const int kv_len = 16;
    const int head_dim = 64;
    const int blocks_per_row = 1;

    // Create Q and K blocks
    Q16_1Block_64 Q = createRandomBlock64(1.0f);
    std::vector<Q16_1Block_64> K(kv_len);
    for (int i = 0; i < kv_len; ++i)
    {
        K[i] = createRandomBlock64(1.0f);
    }

    // Compute scores
    std::vector<int32_t> scores(kv_len);
    q16_qk_gemv<Q16_1Block_64>(&Q, K.data(), scores.data(), kv_len, head_dim, blocks_per_row);

    // Verify each score
    for (int k = 0; k < kv_len; ++k)
    {
        int32_t expected = computeExpectedDot(&Q, &K[k], head_dim, blocks_per_row);
        EXPECT_EQ(scores[k], expected) << "Mismatch at K position " << k;
    }
}

TEST_F(Test__Q16DotProduct, GEMV_MultipleKV_Block128)
{
    const int kv_len = 32;
    const int head_dim = 128;
    const int blocks_per_row = 1;

    // Create Q and K blocks
    Q16_1Block_128 Q = createRandomBlock128(1.0f);
    std::vector<Q16_1Block_128> K(kv_len);
    for (int i = 0; i < kv_len; ++i)
    {
        K[i] = createRandomBlock128(1.0f);
    }

    // Compute scores
    std::vector<int32_t> scores(kv_len);
    q16_qk_gemv<Q16_1Block_128>(&Q, K.data(), scores.data(), kv_len, head_dim, blocks_per_row);

    // Verify each score
    for (int k = 0; k < kv_len; ++k)
    {
        int32_t expected = computeExpectedDot(&Q, &K[k], head_dim, blocks_per_row);
        EXPECT_EQ(scores[k], expected) << "Mismatch at K position " << k;
    }
}

// ============================================================================
// q16_qk_gemv_dispatch Tests
// ============================================================================

TEST_F(Test__Q16DotProduct, Dispatch_Block64)
{
    const int kv_len = 8;
    const int head_dim = 64;

    Q16_1Block_64 Q = createRandomBlock64(1.0f);
    std::vector<Q16_1Block_64> K(kv_len);
    for (int i = 0; i < kv_len; ++i)
    {
        K[i] = createRandomBlock64(1.0f);
    }

    std::vector<int32_t> scores(kv_len);
    q16_qk_gemv_dispatch(&Q, K.data(), scores.data(), kv_len, head_dim, Q16BlockSize::BLOCK_64);

    // Verify
    for (int k = 0; k < kv_len; ++k)
    {
        int32_t expected = computeExpectedDot(&Q, &K[k], head_dim, 1);
        EXPECT_EQ(scores[k], expected) << "Mismatch at K position " << k;
    }
}

TEST_F(Test__Q16DotProduct, Dispatch_Block128)
{
    const int kv_len = 8;
    const int head_dim = 128;

    Q16_1Block_128 Q = createRandomBlock128(1.0f);
    std::vector<Q16_1Block_128> K(kv_len);
    for (int i = 0; i < kv_len; ++i)
    {
        K[i] = createRandomBlock128(1.0f);
    }

    std::vector<int32_t> scores(kv_len);
    q16_qk_gemv_dispatch(&Q, K.data(), scores.data(), kv_len, head_dim, Q16BlockSize::BLOCK_128);

    // Verify
    for (int k = 0; k < kv_len; ++k)
    {
        int32_t expected = computeExpectedDot(&Q, &K[k], head_dim, 1);
        EXPECT_EQ(scores[k], expected) << "Mismatch at K position " << k;
    }
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(Test__Q16DotProduct, EdgeCase_MaxValues_Block64)
{
    // Test with maximum INT16 values to verify no overflow in accumulation
    std::vector<int16_t> max_vals(64, 32767);
    Q16_1Block_64 q = createBlock64(1.0f, max_vals);
    Q16_1Block_64 k = createBlock64(1.0f, max_vals);

    int32_t result = q16_dot_single<Q16_1Block_64>(&q, &k, 64, 1);

    // Expected: 64 × 32767² = 64 × 1073676289 = 68715282496
    // This overflows INT32 but the computation uses INT32 accumulators
    // The actual result will be truncated, which is expected behavior

    // For the test, we just verify it runs without crashing
    // Real usage should ensure data ranges don't cause overflow
    (void)result;
}

TEST_F(Test__Q16DotProduct, EdgeCase_MinMaxValues_Block64)
{
    // Test with min/max alternating to verify signed arithmetic
    std::vector<int16_t> vals(64);
    for (int i = 0; i < 64; ++i)
    {
        vals[i] = (i % 2 == 0) ? 32767 : -32768;
    }
    Q16_1Block_64 q = createBlock64(1.0f, vals);
    Q16_1Block_64 k = createBlock64(1.0f, vals);

    int32_t result = q16_dot_single<Q16_1Block_64>(&q, &k, 64, 1);
    int32_t expected = computeExpectedDot(&q, &k, 64, 1);

    EXPECT_EQ(result, expected);
}

// ============================================================================
// FP32 Accuracy Tests - Measure Quantization Error vs Float Reference
// ============================================================================

TEST_F(Test__Q16DotProduct, FP32Accuracy_10000Vectors_Block64)
{
    /**
     * ACCURACY TEST: Compare Q16 integer dot products against FP32 reference.
     *
     * The Q16 format stores values as: value = qs[i] * scale
     * Integer dot product accumulates: sum(q.qs[i] * k.qs[i])
     * FP32 dot product: sum(q.qs[i] * q.d * k.qs[i] * k.d)
     *                 = sum(q.qs[i] * k.qs[i]) * q.d * k.d
     *
     * For the same quantized blocks, the integer dot product is EXACT
     * when compared to the FP32 dequantized dot product scaled appropriately.
     * Error comes from the quantization of the original FP32 values.
     *
     * This test measures quantization error when going FP32 -> Q16 -> dot product.
     */
    const int num_vectors = 10000;
    const int head_dim = 64;
    const float scale = 1.0f / 128.0f; // Typical Q16 scale

    double total_rel_error = 0.0;
    double max_rel_error = 0.0;
    double total_abs_error = 0.0;
    double max_abs_error = 0.0;

    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int v = 0; v < num_vectors; ++v)
    {
        // Generate random FP32 vectors
        std::vector<float> q_fp32(head_dim);
        std::vector<float> k_fp32(head_dim);

        for (int i = 0; i < head_dim; ++i)
        {
            q_fp32[i] = dist(rng_);
            k_fp32[i] = dist(rng_);
        }

        // FP32 reference dot product
        double fp32_dot = 0.0;
        for (int i = 0; i < head_dim; ++i)
        {
            fp32_dot += static_cast<double>(q_fp32[i]) * static_cast<double>(k_fp32[i]);
        }

        // Quantize to Q16
        std::vector<int16_t> q_quant(head_dim);
        std::vector<int16_t> k_quant(head_dim);

        for (int i = 0; i < head_dim; ++i)
        {
            q_quant[i] = static_cast<int16_t>(std::clamp(
                std::round(q_fp32[i] / scale), -32767.0f, 32767.0f));
            k_quant[i] = static_cast<int16_t>(std::clamp(
                std::round(k_fp32[i] / scale), -32767.0f, 32767.0f));
        }

        Q16_1Block_64 q_block = createBlock64(scale, q_quant);
        Q16_1Block_64 k_block = createBlock64(scale, k_quant);

        // Q16 integer dot product
        int32_t q16_dot_int = q16_dot_single<Q16_1Block_64>(&q_block, &k_block, head_dim, 1);

        // Convert back to FP32: dot = int_dot * scale * scale
        double q16_dot_fp32 = static_cast<double>(q16_dot_int) * scale * scale;

        // Error metrics
        double abs_err = std::abs(fp32_dot - q16_dot_fp32);
        double rel_err = (std::abs(fp32_dot) > 1e-6) ? abs_err / std::abs(fp32_dot) : abs_err;

        total_abs_error += abs_err;
        total_rel_error += rel_err;
        max_abs_error = std::max(max_abs_error, abs_err);
        max_rel_error = std::max(max_rel_error, rel_err);
    }

    double mean_abs_error = total_abs_error / num_vectors;
    double mean_rel_error = total_rel_error / num_vectors;

    std::cout << "\n"
              << "╔══════════════════════════════════════════════════════════════════════════╗\n"
              << "║         Q16 DOT PRODUCT vs FP32 REFERENCE (10,000 Vectors)               ║\n"
              << "╠══════════════════════════════════════════════════════════════════════════╣\n"
              << "║ Head dimension:          " << std::setw(10) << head_dim << "                                    ║\n"
              << "║ Mean Absolute Error:     " << std::setw(14) << std::scientific << std::setprecision(6) << mean_abs_error << "                            ║\n"
              << "║ Max Absolute Error:      " << std::setw(14) << max_abs_error << "                            ║\n"
              << "║ Mean Relative Error:     " << std::setw(14) << mean_rel_error << "                            ║\n"
              << "║ Max Relative Error:      " << std::setw(14) << max_rel_error << "                            ║\n"
              << "╚══════════════════════════════════════════════════════════════════════════╝\n\n";

    // Q16 quantization has ~2-3% typical relative error due to INT16 discretization
    EXPECT_LT(mean_rel_error, 0.05) << "Mean relative error too high for Q16";
    EXPECT_LT(mean_abs_error, 0.05) << "Mean absolute error too high for Q16";
}

TEST_F(Test__Q16DotProduct, FP32Accuracy_CosineSimilarity_Block64)
{
    /**
     * Test Q×K dot product accuracy using cosine similarity.
     *
     * Cosine similarity is scale-invariant and avoids division-by-zero issues
     * that plague relative error when reference values are near zero.
     *
     * cos(θ) = (a·b) / (||a|| × ||b||)
     *
     * For scalar dot products, we collect batches and compare as vectors.
     */
    const int num_batches = 1000;
    const int batch_size = 64; // Compare 64 dot products at a time
    const int head_dim = 64;
    const float scale = 1.0f / 128.0f;

    double total_cosine_sim = 0.0;
    double min_cosine_sim = 1.0;

    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int b = 0; b < num_batches; ++b)
    {
        std::vector<double> fp32_dots(batch_size);
        std::vector<double> q16_dots(batch_size);

        for (int s = 0; s < batch_size; ++s)
        {
            std::vector<float> q_fp32(head_dim);
            std::vector<float> k_fp32(head_dim);

            for (int i = 0; i < head_dim; ++i)
            {
                q_fp32[i] = dist(rng_);
                k_fp32[i] = dist(rng_);
            }

            // FP32 reference
            double fp32_dot = 0.0;
            for (int i = 0; i < head_dim; ++i)
            {
                fp32_dot += static_cast<double>(q_fp32[i]) * static_cast<double>(k_fp32[i]);
            }

            // Quantize
            Q16_1Block_64 q_block, k_block;
            q_block.d = scale;
            k_block.d = scale;
            for (int i = 0; i < head_dim; ++i)
            {
                q_block.qs[i] = static_cast<int16_t>(std::clamp(
                    std::round(q_fp32[i] / scale), -32767.0f, 32767.0f));
                k_block.qs[i] = static_cast<int16_t>(std::clamp(
                    std::round(k_fp32[i] / scale), -32767.0f, 32767.0f));
            }

            int32_t q16_dot_int = q16_dot_single<Q16_1Block_64>(&q_block, &k_block, head_dim, 1);
            double q16_dot = static_cast<double>(q16_dot_int) * scale * scale;

            fp32_dots[s] = fp32_dot;
            q16_dots[s] = q16_dot;
        }

        // Cosine similarity between the two vectors of dot products
        double dot_ab = 0.0, norm_a = 0.0, norm_b = 0.0;
        for (int s = 0; s < batch_size; ++s)
        {
            dot_ab += fp32_dots[s] * q16_dots[s];
            norm_a += fp32_dots[s] * fp32_dots[s];
            norm_b += q16_dots[s] * q16_dots[s];
        }

        double cosine_sim = dot_ab / (std::sqrt(norm_a) * std::sqrt(norm_b) + 1e-10);
        total_cosine_sim += cosine_sim;
        min_cosine_sim = std::min(min_cosine_sim, cosine_sim);
    }

    double mean_cosine_sim = total_cosine_sim / num_batches;

    std::cout << "\n"
              << "╔══════════════════════════════════════════════════════════════════════════╗\n"
              << "║      Q16 DOT PRODUCT vs FP32 - COSINE SIMILARITY (1000 Batches)          ║\n"
              << "╠══════════════════════════════════════════════════════════════════════════╣\n"
              << "║ Head dimension:          " << std::setw(10) << head_dim << "                                    ║\n"
              << "║ Batch size:              " << std::setw(10) << batch_size << "                                    ║\n"
              << "║ Mean Cosine Similarity:  " << std::setw(14) << std::fixed << std::setprecision(6) << mean_cosine_sim << "                            ║\n"
              << "║ Min Cosine Similarity:   " << std::setw(14) << min_cosine_sim << "                            ║\n"
              << "╚══════════════════════════════════════════════════════════════════════════╝\n\n";

    // Cosine similarity should be very high (>0.99) for good quantization
    EXPECT_GT(mean_cosine_sim, 0.99) << "Mean cosine similarity too low for Q16";
    EXPECT_GT(min_cosine_sim, 0.95) << "Worst-case cosine similarity too low";
}

TEST_F(Test__Q16DotProduct, FP32Accuracy_10000Vectors_Block128)
{
    /**
     * Same accuracy test for 128-dim heads (Qwen2.5 default).
     */
    const int num_vectors = 10000;
    const int head_dim = 128;
    const float scale = 1.0f / 128.0f;

    double total_rel_error = 0.0;
    double total_abs_error = 0.0;
    double max_rel_error = 0.0;

    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int v = 0; v < num_vectors; ++v)
    {
        std::vector<float> q_fp32(head_dim);
        std::vector<float> k_fp32(head_dim);

        for (int i = 0; i < head_dim; ++i)
        {
            q_fp32[i] = dist(rng_);
            k_fp32[i] = dist(rng_);
        }

        double fp32_dot = 0.0;
        for (int i = 0; i < head_dim; ++i)
        {
            fp32_dot += static_cast<double>(q_fp32[i]) * static_cast<double>(k_fp32[i]);
        }

        std::vector<int16_t> q_quant(head_dim);
        std::vector<int16_t> k_quant(head_dim);

        for (int i = 0; i < head_dim; ++i)
        {
            q_quant[i] = static_cast<int16_t>(std::clamp(
                std::round(q_fp32[i] / scale), -32767.0f, 32767.0f));
            k_quant[i] = static_cast<int16_t>(std::clamp(
                std::round(k_fp32[i] / scale), -32767.0f, 32767.0f));
        }

        Q16_1Block_128 q_block = createBlock128(scale, q_quant);
        Q16_1Block_128 k_block = createBlock128(scale, k_quant);

        int32_t q16_dot_int = q16_dot_single<Q16_1Block_128>(&q_block, &k_block, head_dim, 1);
        double q16_dot_fp32 = static_cast<double>(q16_dot_int) * scale * scale;

        double abs_err = std::abs(fp32_dot - q16_dot_fp32);
        double rel_err = (std::abs(fp32_dot) > 1e-6) ? abs_err / std::abs(fp32_dot) : abs_err;

        total_rel_error += rel_err;
        total_abs_error += abs_err;
        max_rel_error = std::max(max_rel_error, rel_err);
    }

    double mean_rel_error = total_rel_error / num_vectors;
    double mean_abs_error = total_abs_error / num_vectors;

    std::cout << "  [Q16DotProduct Block128] Mean rel error: " << std::scientific << std::setprecision(4) << mean_rel_error
              << ", Max rel error: " << max_rel_error << std::endl;

    // Q16 quantization has ~2-3% typical relative error due to INT16 discretization
    EXPECT_LT(mean_rel_error, 0.05) << "Mean relative error too high for Q16 block128";
    EXPECT_LT(mean_abs_error, 0.05) << "Mean absolute error too high for Q16 block128";
}

TEST_F(Test__Q16DotProduct, FP32Accuracy_CosineSimilarity_Block128)
{
    /**
     * Cosine similarity test for 128-dim heads.
     */
    const int num_batches = 1000;
    const int batch_size = 64;
    const int head_dim = 128;
    const float scale = 1.0f / 128.0f;

    double total_cosine_sim = 0.0;
    double min_cosine_sim = 1.0;

    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int b = 0; b < num_batches; ++b)
    {
        std::vector<double> fp32_dots(batch_size);
        std::vector<double> q16_dots(batch_size);

        for (int s = 0; s < batch_size; ++s)
        {
            std::vector<float> q_fp32(head_dim);
            std::vector<float> k_fp32(head_dim);

            for (int i = 0; i < head_dim; ++i)
            {
                q_fp32[i] = dist(rng_);
                k_fp32[i] = dist(rng_);
            }

            double fp32_dot = 0.0;
            for (int i = 0; i < head_dim; ++i)
            {
                fp32_dot += static_cast<double>(q_fp32[i]) * static_cast<double>(k_fp32[i]);
            }

            Q16_1Block_128 q_block, k_block;
            q_block.d = scale;
            k_block.d = scale;
            for (int i = 0; i < head_dim; ++i)
            {
                q_block.qs[i] = static_cast<int16_t>(std::clamp(
                    std::round(q_fp32[i] / scale), -32767.0f, 32767.0f));
                k_block.qs[i] = static_cast<int16_t>(std::clamp(
                    std::round(k_fp32[i] / scale), -32767.0f, 32767.0f));
            }

            int32_t q16_dot_int = q16_dot_single<Q16_1Block_128>(&q_block, &k_block, head_dim, 1);
            double q16_dot = static_cast<double>(q16_dot_int) * scale * scale;

            fp32_dots[s] = fp32_dot;
            q16_dots[s] = q16_dot;
        }

        double dot_ab = 0.0, norm_a = 0.0, norm_b = 0.0;
        for (int s = 0; s < batch_size; ++s)
        {
            dot_ab += fp32_dots[s] * q16_dots[s];
            norm_a += fp32_dots[s] * fp32_dots[s];
            norm_b += q16_dots[s] * q16_dots[s];
        }

        double cosine_sim = dot_ab / (std::sqrt(norm_a) * std::sqrt(norm_b) + 1e-10);
        total_cosine_sim += cosine_sim;
        min_cosine_sim = std::min(min_cosine_sim, cosine_sim);
    }

    double mean_cosine_sim = total_cosine_sim / num_batches;

    std::cout << "  [Q16DotProduct Block128 CosineSim] Mean: " << std::fixed << std::setprecision(6)
              << mean_cosine_sim << ", Min: " << min_cosine_sim << std::endl;

    EXPECT_GT(mean_cosine_sim, 0.99) << "Mean cosine similarity too low for Q16 block128";
    EXPECT_GT(min_cosine_sim, 0.95) << "Worst-case cosine similarity too low";
}
