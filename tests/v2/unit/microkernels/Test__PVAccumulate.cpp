/**
 * @file Test__PVAccumulate.cpp
 * @brief Unit tests for Q16_1 P×V accumulation microkernel
 *
 * Tests the PVAccumulate microkernel functions:
 * - q16_pv_accumulate: Flash Decode weighted sum
 * - q16_pv_accumulate_add: Add to existing context
 * - q16_context_rescale: Online softmax rescaling
 */

#include <gtest/gtest.h>
#include "kernels/cpu/attention/q16_1/ref/microkernels/PVAccumulate.h"
#include "tensors/BlockStructures.h"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using namespace llaminar2::kernels::q16_1::microkernels;
using namespace llaminar2;

class Test__PVAccumulate : public ::testing::Test
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
        std::uniform_int_distribution<int> dist(-1000, 1000); // Smaller range to avoid overflow

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
        std::uniform_int_distribution<int> dist(-1000, 1000);

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

    // Helper to compute expected P×V accumulation (ground truth)
    template <typename BlockType>
    std::vector<int32_t> computeExpectedPV(
        const int16_t *weights,
        const BlockType *V,
        int kv_len,
        int head_dim,
        int blocks_per_row)
    {
        constexpr int BLOCK_SIZE = BlockType::BLOCK_SIZE;
        std::vector<int32_t> context(head_dim, 0);

        for (int k = 0; k < kv_len; ++k)
        {
            int16_t w = weights[k];
            if (w == 0)
                continue;

            for (int b = 0; b < blocks_per_row; ++b)
            {
                int start = b * BLOCK_SIZE;
                int end = std::min(start + BLOCK_SIZE, head_dim);
                int count = end - start;

                for (int i = 0; i < count; ++i)
                {
                    context[start + i] += static_cast<int32_t>(w) *
                                          static_cast<int32_t>(V[k * blocks_per_row + b].qs[i]);
                }
            }
        }
        return context;
    }
};

// ============================================================================
// q16_pv_accumulate Tests
// ============================================================================

TEST_F(Test__PVAccumulate, Accumulate_ZeroWeights_Block64)
{
    const int kv_len = 8;
    const int head_dim = 64;
    const int blocks_per_row = 1;

    // All zero weights
    std::vector<int16_t> weights(kv_len, 0);

    // Random V values
    std::vector<Q16_1Block_64> V(kv_len);
    for (int k = 0; k < kv_len; ++k)
    {
        V[k] = createRandomBlock64(1.0f);
    }

    // Compute
    std::vector<int32_t> context(head_dim);
    q16_pv_accumulate<Q16_1Block_64>(
        weights.data(), V.data(), context.data(),
        kv_len, head_dim, blocks_per_row);

    // Should be all zeros
    for (int d = 0; d < head_dim; ++d)
    {
        EXPECT_EQ(context[d], 0) << "Non-zero at dim " << d;
    }
}

TEST_F(Test__PVAccumulate, Accumulate_SingleWeight_Block64)
{
    const int kv_len = 8;
    const int head_dim = 64;
    const int blocks_per_row = 1;

    // Single non-zero weight at position 3
    std::vector<int16_t> weights(kv_len, 0);
    weights[3] = 100;

    // Known V values
    std::vector<Q16_1Block_64> V(kv_len);
    std::vector<int16_t> vals(64);
    for (int i = 0; i < 64; ++i)
    {
        vals[i] = static_cast<int16_t>(i + 1);
    }
    for (int k = 0; k < kv_len; ++k)
    {
        V[k] = createBlock64(1.0f, vals);
    }

    // Compute
    std::vector<int32_t> context(head_dim);
    q16_pv_accumulate<Q16_1Block_64>(
        weights.data(), V.data(), context.data(),
        kv_len, head_dim, blocks_per_row);

    // Expected: context[d] = 100 × (d+1)
    for (int d = 0; d < head_dim; ++d)
    {
        EXPECT_EQ(context[d], 100 * (d + 1)) << "Mismatch at dim " << d;
    }
}

TEST_F(Test__PVAccumulate, Accumulate_Random_Block64)
{
    const int kv_len = 16;
    const int head_dim = 64;
    const int blocks_per_row = 1;

    // Random weights in reasonable range
    std::uniform_int_distribution<int16_t> wdist(0, 1000);
    std::vector<int16_t> weights(kv_len);
    for (int k = 0; k < kv_len; ++k)
    {
        weights[k] = wdist(rng_);
    }

    // Random V
    std::vector<Q16_1Block_64> V(kv_len);
    for (int k = 0; k < kv_len; ++k)
    {
        V[k] = createRandomBlock64(1.0f);
    }

    // Compute
    std::vector<int32_t> context(head_dim);
    q16_pv_accumulate<Q16_1Block_64>(
        weights.data(), V.data(), context.data(),
        kv_len, head_dim, blocks_per_row);

    // Verify against reference
    auto expected = computeExpectedPV<Q16_1Block_64>(
        weights.data(), V.data(), kv_len, head_dim, blocks_per_row);

    for (int d = 0; d < head_dim; ++d)
    {
        EXPECT_EQ(context[d], expected[d]) << "Mismatch at dim " << d;
    }
}

TEST_F(Test__PVAccumulate, Accumulate_Random_Block128)
{
    const int kv_len = 32;
    const int head_dim = 128;
    const int blocks_per_row = 1;

    // Random weights
    std::uniform_int_distribution<int16_t> wdist(0, 500);
    std::vector<int16_t> weights(kv_len);
    for (int k = 0; k < kv_len; ++k)
    {
        weights[k] = wdist(rng_);
    }

    // Random V
    std::vector<Q16_1Block_128> V(kv_len);
    for (int k = 0; k < kv_len; ++k)
    {
        V[k] = createRandomBlock128(1.0f);
    }

    // Compute
    std::vector<int32_t> context(head_dim);
    q16_pv_accumulate<Q16_1Block_128>(
        weights.data(), V.data(), context.data(),
        kv_len, head_dim, blocks_per_row);

    // Verify against reference
    auto expected = computeExpectedPV<Q16_1Block_128>(
        weights.data(), V.data(), kv_len, head_dim, blocks_per_row);

    for (int d = 0; d < head_dim; ++d)
    {
        EXPECT_EQ(context[d], expected[d]) << "Mismatch at dim " << d;
    }
}

// ============================================================================
// q16_pv_accumulate_add Tests
// ============================================================================

TEST_F(Test__PVAccumulate, AccumulateAdd_PreservesExisting_Block64)
{
    const int kv_len = 4;
    const int head_dim = 64;
    const int blocks_per_row = 1;

    // Pre-initialize context with known values
    std::vector<int32_t> context(head_dim);
    for (int d = 0; d < head_dim; ++d)
    {
        context[d] = d * 1000;
    }

    // Zero weights - should not change context
    std::vector<int16_t> weights(kv_len, 0);
    std::vector<Q16_1Block_64> V(kv_len);
    for (int k = 0; k < kv_len; ++k)
    {
        V[k] = createRandomBlock64(1.0f);
    }

    q16_pv_accumulate_add<Q16_1Block_64>(
        weights.data(), V.data(), context.data(),
        kv_len, head_dim, blocks_per_row);

    // Should preserve original values
    for (int d = 0; d < head_dim; ++d)
    {
        EXPECT_EQ(context[d], d * 1000) << "Changed at dim " << d;
    }
}

TEST_F(Test__PVAccumulate, AccumulateAdd_AddsToExisting_Block64)
{
    const int kv_len = 4;
    const int head_dim = 64;
    const int blocks_per_row = 1;

    // Pre-initialize context
    std::vector<int32_t> context(head_dim, 100);

    // Single weight
    std::vector<int16_t> weights(kv_len, 0);
    weights[0] = 1;

    // V with values = index
    std::vector<Q16_1Block_64> V(kv_len);
    std::vector<int16_t> vals(64);
    for (int i = 0; i < 64; ++i)
    {
        vals[i] = static_cast<int16_t>(i);
    }
    for (int k = 0; k < kv_len; ++k)
    {
        V[k] = createBlock64(1.0f, vals);
    }

    q16_pv_accumulate_add<Q16_1Block_64>(
        weights.data(), V.data(), context.data(),
        kv_len, head_dim, blocks_per_row);

    // Expected: context[d] = 100 + 1 × d
    for (int d = 0; d < head_dim; ++d)
    {
        EXPECT_EQ(context[d], 100 + d) << "Mismatch at dim " << d;
    }
}

// ============================================================================
// q16_context_rescale Tests
// ============================================================================

TEST_F(Test__PVAccumulate, Rescale_NoOp)
{
    const int head_dim = 64;
    std::vector<int32_t> context(head_dim);
    for (int d = 0; d < head_dim; ++d)
    {
        context[d] = d * 100;
    }

    // scale_num=1, scale_shift=0 should be no-op
    q16_context_rescale(context.data(), head_dim, 1, 0);

    for (int d = 0; d < head_dim; ++d)
    {
        EXPECT_EQ(context[d], d * 100) << "Changed at dim " << d;
    }
}

TEST_F(Test__PVAccumulate, Rescale_HalfScale)
{
    const int head_dim = 64;
    std::vector<int32_t> context(head_dim);
    for (int d = 0; d < head_dim; ++d)
    {
        context[d] = 1000;
    }

    // scale = 1/2 → scale_num=1, scale_shift=1
    q16_context_rescale(context.data(), head_dim, 1, 1);

    // Expected: 1000 >> 1 = 500
    for (int d = 0; d < head_dim; ++d)
    {
        EXPECT_EQ(context[d], 500) << "Mismatch at dim " << d;
    }
}

TEST_F(Test__PVAccumulate, Rescale_ZeroScale)
{
    const int head_dim = 64;
    std::vector<int32_t> context(head_dim, 12345);

    // scale_num=0 should zero everything
    q16_context_rescale(context.data(), head_dim, 0, 0);

    for (int d = 0; d < head_dim; ++d)
    {
        EXPECT_EQ(context[d], 0) << "Not zeroed at dim " << d;
    }
}

TEST_F(Test__PVAccumulate, Rescale_WithMultiplier)
{
    const int head_dim = 8;
    std::vector<int32_t> context = {100, 200, 300, 400, 500, 600, 700, 800};

    // scale = 3/4 → scale_num=3, scale_shift=2
    q16_context_rescale(context.data(), head_dim, 3, 2);

    // Expected: (value × 3 + 2) >> 2 (with rounding)
    std::vector<int32_t> expected = {75, 150, 225, 300, 375, 450, 525, 600};

    for (int d = 0; d < head_dim; ++d)
    {
        EXPECT_EQ(context[d], expected[d]) << "Mismatch at dim " << d;
    }
}

// ============================================================================
// Dispatch Tests
// ============================================================================

TEST_F(Test__PVAccumulate, Dispatch_Block64)
{
    const int kv_len = 8;
    const int head_dim = 64;

    std::uniform_int_distribution<int16_t> wdist(0, 500);
    std::vector<int16_t> weights(kv_len);
    for (int k = 0; k < kv_len; ++k)
    {
        weights[k] = wdist(rng_);
    }

    std::vector<Q16_1Block_64> V(kv_len);
    for (int k = 0; k < kv_len; ++k)
    {
        V[k] = createRandomBlock64(1.0f);
    }

    std::vector<int32_t> context(head_dim);
    q16_pv_accumulate_dispatch(
        weights.data(), V.data(), context.data(),
        kv_len, head_dim, Q16BlockSize::BLOCK_64);

    auto expected = computeExpectedPV<Q16_1Block_64>(
        weights.data(), V.data(), kv_len, head_dim, 1);

    for (int d = 0; d < head_dim; ++d)
    {
        EXPECT_EQ(context[d], expected[d]) << "Mismatch at dim " << d;
    }
}

TEST_F(Test__PVAccumulate, Dispatch_Block128)
{
    const int kv_len = 8;
    const int head_dim = 128;

    std::uniform_int_distribution<int16_t> wdist(0, 500);
    std::vector<int16_t> weights(kv_len);
    for (int k = 0; k < kv_len; ++k)
    {
        weights[k] = wdist(rng_);
    }

    std::vector<Q16_1Block_128> V(kv_len);
    for (int k = 0; k < kv_len; ++k)
    {
        V[k] = createRandomBlock128(1.0f);
    }

    std::vector<int32_t> context(head_dim);
    q16_pv_accumulate_dispatch(
        weights.data(), V.data(), context.data(),
        kv_len, head_dim, Q16BlockSize::BLOCK_128);

    auto expected = computeExpectedPV<Q16_1Block_128>(
        weights.data(), V.data(), kv_len, head_dim, 1);

    for (int d = 0; d < head_dim; ++d)
    {
        EXPECT_EQ(context[d], expected[d]) << "Mismatch at dim " << d;
    }
}

// ============================================================================
// FP32 Accuracy Tests - P×V Accumulation Error Analysis
// ============================================================================

TEST_F(Test__PVAccumulate, FP32Accuracy_10000Vectors_Block64)
{
    /**
     * ACCURACY TEST: Compare Q16 P×V accumulation against FP32 reference.
     *
     * The P×V operation computes: context[d] = Σ_k weights[k] × V[k][d]
     *
     * In integer domain:
     *   context_int[d] = Σ_k weight_int[k] × v_int[k][d]
     *
     * FP32 equivalent:
     *   context_fp32[d] = Σ_k (weight_int[k]/w_scale) × (v_int[k][d] × v_scale)
     *                   = (v_scale / w_scale) × context_int[d]
     *
     * Error sources:
     * 1. Weight quantization (INT16 from softmax)
     * 2. V value quantization (INT16 in Q16_1 blocks)
     * 3. Accumulation (no error - exact integer math)
     */
    const int num_tests = 10000;
    const int head_dim = 64;
    const int kv_len = 64; // Typical decode KV cache length
    const float v_scale = 1.0f / 128.0f;
    const float w_scale = 1.0f / 32767.0f; // INT16 weight normalization

    double total_rel_error = 0.0;
    double max_rel_error = 0.0;
    int num_samples = 0;

    std::uniform_real_distribution<float> v_dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> w_dist(0.0f, 1.0f);

    for (int t = 0; t < num_tests; ++t)
    {
        // Generate random FP32 V values
        std::vector<std::vector<float>> V_fp32(kv_len, std::vector<float>(head_dim));
        for (int k = 0; k < kv_len; ++k)
        {
            for (int d = 0; d < head_dim; ++d)
            {
                V_fp32[k][d] = v_dist(rng_);
            }
        }

        // Generate random FP32 weights (softmax-like, sum to 1)
        std::vector<float> weights_fp32(kv_len);
        float w_sum = 0.0f;
        for (int k = 0; k < kv_len; ++k)
        {
            weights_fp32[k] = w_dist(rng_);
            w_sum += weights_fp32[k];
        }
        for (int k = 0; k < kv_len; ++k)
        {
            weights_fp32[k] /= w_sum;
        }

        // FP32 reference: context[d] = Σ_k weights[k] × V[k][d]
        std::vector<double> context_fp32(head_dim, 0.0);
        for (int k = 0; k < kv_len; ++k)
        {
            for (int d = 0; d < head_dim; ++d)
            {
                context_fp32[d] += weights_fp32[k] * V_fp32[k][d];
            }
        }

        // Quantize V to Q16 blocks
        std::vector<Q16_1Block_64> V_q16(kv_len);
        for (int k = 0; k < kv_len; ++k)
        {
            V_q16[k].d = v_scale;
            int32_t sum_qs = 0;
            for (int d = 0; d < head_dim; ++d)
            {
                int16_t v_int = static_cast<int16_t>(std::clamp(
                    std::round(V_fp32[k][d] / v_scale), -32767.0f, 32767.0f));
                V_q16[k].qs[d] = v_int;
                sum_qs += v_int;
            }
            V_q16[k].sum_qs = sum_qs;
        }

        // Quantize weights to INT16 (scale by 32767 for full range)
        std::vector<int16_t> weights_int(kv_len);
        for (int k = 0; k < kv_len; ++k)
        {
            weights_int[k] = static_cast<int16_t>(std::clamp(
                std::round(weights_fp32[k] * 32767.0f), 0.0f, 32767.0f));
        }

        // Q16 P×V accumulation
        std::vector<int32_t> context_q16(head_dim, 0);
        q16_pv_accumulate<Q16_1Block_64>(
            weights_int.data(), V_q16.data(), context_q16.data(),
            kv_len, head_dim, 1);

        // Convert back to FP32 for comparison
        // context_fp32 ≈ context_q16 × v_scale × w_scale
        for (int d = 0; d < head_dim; ++d)
        {
            double q16_context_fp32 = static_cast<double>(context_q16[d]) * v_scale * w_scale;
            double abs_err = std::abs(context_fp32[d] - q16_context_fp32);
            double rel_err = (std::abs(context_fp32[d]) > 1e-6)
                                 ? abs_err / std::abs(context_fp32[d])
                                 : abs_err;

            total_rel_error += rel_err;
            max_rel_error = std::max(max_rel_error, rel_err);
            num_samples++;
        }
    }

    double mean_rel_error = total_rel_error / num_samples;

    std::cout << "\n"
              << "╔══════════════════════════════════════════════════════════════════════════╗\n"
              << "║         Q16 P×V ACCUMULATE vs FP32 REFERENCE (10,000 Vectors)            ║\n"
              << "╠══════════════════════════════════════════════════════════════════════════╣\n"
              << "║ Head dimension:          " << std::setw(10) << head_dim << "                                    ║\n"
              << "║ KV cache length:         " << std::setw(10) << kv_len << "                                    ║\n"
              << "║ Total dim samples:       " << std::setw(10) << num_samples << "                                    ║\n"
              << "║ Mean Relative Error:     " << std::setw(14) << std::scientific << std::setprecision(6) << mean_rel_error << "                            ║\n"
              << "║ Max Relative Error:      " << std::setw(14) << max_rel_error << "                            ║\n"
              << "╚══════════════════════════════════════════════════════════════════════════╝\n\n";

    // P×V has error from both weight and V quantization
    EXPECT_LT(mean_rel_error, 0.05) << "Mean relative error too high for Q16 P×V";
}

TEST_F(Test__PVAccumulate, FP32Accuracy_CosineSimilarity)
{
    /**
     * Test P×V accumulate accuracy using cosine similarity.
     *
     * Cosine similarity measures directional accuracy, which is more meaningful
     * for attention context vectors than element-wise relative error.
     *
     * cos(θ) = (fp32_context · q16_context) / (||fp32|| × ||q16||)
     */
    const int num_tests = 1000;
    const int head_dim = 64;
    const int kv_len = 64;
    const float p_scale = 1.0f / 32768.0f; // Softmax weights scale
    const float v_scale = 1.0f / 128.0f;   // Value scale

    double total_cosine_sim = 0.0;
    double min_cosine_sim = 1.0;

    std::uniform_real_distribution<float> v_dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> p_dist(0.0f, 1.0f); // P is non-negative

    for (int t = 0; t < num_tests; ++t)
    {
        // Generate V cache (kv_len × head_dim)
        std::vector<std::vector<float>> V_fp32(kv_len, std::vector<float>(head_dim));
        for (int k = 0; k < kv_len; ++k)
        {
            for (int d = 0; d < head_dim; ++d)
            {
                V_fp32[k][d] = v_dist(rng_);
            }
        }

        // Generate softmax weights P (kv_len)
        std::vector<float> P_fp32(kv_len);
        float sum_p = 0.0f;
        for (int k = 0; k < kv_len; ++k)
        {
            P_fp32[k] = p_dist(rng_);
            sum_p += P_fp32[k];
        }
        for (int k = 0; k < kv_len; ++k)
        {
            P_fp32[k] /= sum_p; // Normalize to sum to 1
        }

        // FP32 reference: context[d] = Σ_k P[k] × V[k][d]
        std::vector<double> context_fp32(head_dim, 0.0);
        for (int d = 0; d < head_dim; ++d)
        {
            for (int k = 0; k < kv_len; ++k)
            {
                context_fp32[d] += P_fp32[k] * V_fp32[k][d];
            }
        }

        // Quantize P to INT16
        std::vector<int16_t> P_int16(kv_len);
        for (int k = 0; k < kv_len; ++k)
        {
            P_int16[k] = static_cast<int16_t>(std::clamp(
                std::round(P_fp32[k] / p_scale), 0.0f, 32767.0f));
        }

        // Quantize V to Q16_1 blocks
        std::vector<Q16_1Block_64> V_q16(kv_len);
        for (int k = 0; k < kv_len; ++k)
        {
            V_q16[k].d = v_scale;
            for (int d = 0; d < head_dim; ++d)
            {
                V_q16[k].qs[d] = static_cast<int16_t>(std::clamp(
                    std::round(V_fp32[k][d] / v_scale), -32767.0f, 32767.0f));
            }
        }

        // Q16 P×V accumulate - process all KV positions at once
        std::vector<int32_t> context_int32(head_dim, 0);
        q16_pv_accumulate<Q16_1Block_64>(
            P_int16.data(), V_q16.data(), context_int32.data(),
            kv_len, head_dim, 1 /* blocks_per_row */);

        // Convert to FP32
        std::vector<double> context_q16(head_dim);
        for (int d = 0; d < head_dim; ++d)
        {
            context_q16[d] = static_cast<double>(context_int32[d]) * p_scale * v_scale;
        }

        // Cosine similarity
        double dot_ab = 0.0, norm_a = 0.0, norm_b = 0.0;
        for (int d = 0; d < head_dim; ++d)
        {
            dot_ab += context_fp32[d] * context_q16[d];
            norm_a += context_fp32[d] * context_fp32[d];
            norm_b += context_q16[d] * context_q16[d];
        }

        double cosine_sim = dot_ab / (std::sqrt(norm_a) * std::sqrt(norm_b) + 1e-10);
        total_cosine_sim += cosine_sim;
        min_cosine_sim = std::min(min_cosine_sim, cosine_sim);
    }

    double mean_cosine_sim = total_cosine_sim / num_tests;

    std::cout << "\n"
              << "╔══════════════════════════════════════════════════════════════════════════╗\n"
              << "║       Q16 P×V ACCUMULATE - COSINE SIMILARITY (1000 Vectors)              ║\n"
              << "╠══════════════════════════════════════════════════════════════════════════╣\n"
              << "║ Head dimension:          " << std::setw(10) << head_dim << "                                    ║\n"
              << "║ KV cache length:         " << std::setw(10) << kv_len << "                                    ║\n"
              << "║ Mean Cosine Similarity:  " << std::setw(14) << std::fixed << std::setprecision(6) << mean_cosine_sim << "                            ║\n"
              << "║ Min Cosine Similarity:   " << std::setw(14) << min_cosine_sim << "                            ║\n"
              << "╚══════════════════════════════════════════════════════════════════════════╝\n\n";

    // Context vectors should be very similar in direction
    EXPECT_GT(mean_cosine_sim, 0.99) << "Mean cosine similarity too low for Q16 P×V";
    EXPECT_GT(min_cosine_sim, 0.95) << "Worst-case cosine similarity too low";
}

TEST_F(Test__PVAccumulate, FP32Accuracy_ContextRescaling)
{
    /**
     * Test accuracy of context rescaling (used in online softmax).
     *
     * When running max changes, we rescale: context_new = context_old × correction
     * where correction = exp(m_old - m_new) ≈ scale_num >> scale_shift
     *
     * The scale_num and scale_shift come from online_softmax_compute_rescale,
     * then q16_context_rescale applies them.
     */
    const int head_dim = 64;
    const int num_tests = 1000;

    double total_rel_error = 0.0;
    double max_rel_error = 0.0;

    std::uniform_int_distribution<int32_t> context_dist(-100000, 100000);
    std::uniform_int_distribution<int32_t> delta_dist(1, 1000); // Max increase

    // LUT parameters (same as OnlineSoftmaxState defaults)
    const int frac_bits = 11;
    const int lut_value_bits = 30;

    for (int t = 0; t < num_tests; ++t)
    {
        // Random context values
        std::vector<int32_t> context(head_dim);
        for (int d = 0; d < head_dim; ++d)
        {
            context[d] = context_dist(rng_);
        }

        // Random max increase (triggers rescaling)
        int32_t m_old = 5000;
        int32_t delta = delta_dist(rng_);
        int32_t m_new = m_old + delta;
        float alpha = 0.01f;

        // FP32 reference: context_new = context × exp(-delta × alpha)
        double correction_fp32 = std::exp(-static_cast<double>(delta) * alpha);
        std::vector<double> expected(head_dim);
        for (int d = 0; d < head_dim; ++d)
        {
            expected[d] = context[d] * correction_fp32;
        }

        // Compute scale_num and scale_shift using OnlineSoftmax helper
        // (We're testing the rescaling itself, not the compute_rescale function)
        // Simple approximation for this test:
        //   correction = 2^(-delta * alpha * log2e)
        //              = 2^(-negative_exponent)
        // For small corrections, use shift-based approximation
        double log2e = 1.4426950408889634;
        double neg_exp2_arg = delta * alpha * log2e; // Always >= 0 since delta > 0
        int32_t scale_shift = static_cast<int32_t>(neg_exp2_arg) + lut_value_bits;
        double fractional = neg_exp2_arg - static_cast<int32_t>(neg_exp2_arg);
        int32_t scale_num = static_cast<int32_t>((1 << lut_value_bits) * std::pow(2.0, -fractional));

        // Q16 rescaling
        std::vector<int32_t> context_rescaled = context; // Copy
        q16_context_rescale(context_rescaled.data(), head_dim, scale_num, scale_shift);

        // Compare
        for (int d = 0; d < head_dim; ++d)
        {
            double abs_err = std::abs(expected[d] - context_rescaled[d]);
            double rel_err = (std::abs(expected[d]) > 1.0)
                                 ? abs_err / std::abs(expected[d])
                                 : abs_err;

            total_rel_error += rel_err;
            max_rel_error = std::max(max_rel_error, rel_err);
        }
    }

    double mean_rel_error = total_rel_error / (num_tests * head_dim);

    std::cout << "  [PVAccumulate ContextRescale] Mean rel error: " << std::scientific << std::setprecision(4)
              << mean_rel_error << ", Max rel error: " << max_rel_error << std::endl;

    // Rescaling uses simple shift, expect decent accuracy
    // Max error can be high when reference is near zero
    EXPECT_LT(mean_rel_error, 0.1) << "Mean rescaling error too high";
}
