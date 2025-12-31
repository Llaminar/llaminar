/**
 * @file Test__Q16WoProjection.cpp
 * @brief Unit tests for Q16_1 Wo projection microkernel
 *
 * Tests the WoProjection microkernel functions:
 * - q16_context_normalize_to_int16: INT32 → INT16 normalization
 * - q16_wo_row_gemv: Single row projection
 * - q16_wo_projection: Full Wo projection (GEMV for decode)
 * - q16_quantize_to_q16_1: INT32 → Q16_1 quantization
 */

#include <gtest/gtest.h>
#include "kernels/cpu/attention/q16_1/ref/microkernels/WoProjection.h"
#include "tensors/BlockStructures.h"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using namespace llaminar2::kernels::q16_1::microkernels;
using namespace llaminar2;

class Test__Q16WoProjection : public ::testing::Test
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
        std::uniform_int_distribution<int> dist(-1000, 1000);

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
};

// ============================================================================
// q16_context_normalize_to_int16 Tests
// ============================================================================

TEST_F(Test__Q16WoProjection, Normalize_ZeroContext)
{
    const int num_elements = 64;
    std::vector<int32_t> context_int32(num_elements, 0);
    std::vector<int16_t> context_int16(num_elements);
    float scale;

    q16_context_normalize_to_int16(
        context_int32.data(), context_int16.data(), scale, num_elements);

    // All zeros should produce all zeros
    for (int i = 0; i < num_elements; ++i)
    {
        EXPECT_EQ(context_int16[i], 0) << "Non-zero at " << i;
    }
}

TEST_F(Test__Q16WoProjection, Normalize_MaxValue)
{
    const int num_elements = 64;
    std::vector<int32_t> context_int32(num_elements, 0);
    context_int32[0] = 100000; // Large value

    std::vector<int16_t> context_int16(num_elements);
    float scale;

    q16_context_normalize_to_int16(
        context_int32.data(), context_int16.data(), scale, num_elements);

    // The max value should map to 32767
    EXPECT_EQ(context_int16[0], 32767);
    // Scale should be 100000 / 32767 ≈ 3.05
    EXPECT_NEAR(scale, 100000.0f / 32767.0f, 0.01f);
}

TEST_F(Test__Q16WoProjection, Normalize_PreservesRatio)
{
    const int num_elements = 4;
    std::vector<int32_t> context_int32 = {10000, 20000, 30000, 40000};
    std::vector<int16_t> context_int16(num_elements);
    float scale;

    q16_context_normalize_to_int16(
        context_int32.data(), context_int16.data(), scale, num_elements);

    // Ratios should be preserved
    // max = 40000 maps to 32767, so scale = 40000/32767
    // Each value: v_int16 ≈ v_int32 / scale

    float ratio01 = static_cast<float>(context_int16[0]) / static_cast<float>(context_int16[1]);
    EXPECT_NEAR(ratio01, 0.5f, 0.01f); // 10000/20000 = 0.5

    float ratio23 = static_cast<float>(context_int16[2]) / static_cast<float>(context_int16[3]);
    EXPECT_NEAR(ratio23, 0.75f, 0.01f); // 30000/40000 = 0.75
}

TEST_F(Test__Q16WoProjection, Normalize_NegativeValues)
{
    const int num_elements = 4;
    std::vector<int32_t> context_int32 = {-10000, 20000, -30000, 40000};
    std::vector<int16_t> context_int16(num_elements);
    float scale;

    q16_context_normalize_to_int16(
        context_int32.data(), context_int16.data(), scale, num_elements);

    // Max abs = 40000
    EXPECT_EQ(context_int16[3], 32767); // Max positive
    EXPECT_LT(context_int16[2], 0);     // Negative preserved
    EXPECT_GT(context_int16[1], 0);     // Positive preserved
    EXPECT_LT(context_int16[0], 0);     // Negative preserved
}

// ============================================================================
// q16_wo_row_gemv Tests
// ============================================================================

TEST_F(Test__Q16WoProjection, RowGemv_ZeroInput_Block64)
{
    const int input_dim = 64;
    const int blocks_per_input = 1;

    std::vector<int16_t> context_int16(input_dim, 0);
    Q16_1Block_64 Wo_row = createRandomBlock64(1.0f);

    int32_t output;
    q16_wo_row_gemv<Q16_1Block_64>(
        context_int16.data(), &Wo_row, output, input_dim, blocks_per_input);

    EXPECT_EQ(output, 0);
}

TEST_F(Test__Q16WoProjection, RowGemv_SingleValue_Block64)
{
    const int input_dim = 64;
    const int blocks_per_input = 1;

    // Context with single 1 at position 0
    std::vector<int16_t> context_int16(input_dim, 0);
    context_int16[0] = 100;

    // Wo with value 50 at position 0
    std::vector<int16_t> wo_vals(64, 0);
    wo_vals[0] = 50;
    Q16_1Block_64 Wo_row = createBlock64(1.0f, wo_vals);

    int32_t output;
    q16_wo_row_gemv<Q16_1Block_64>(
        context_int16.data(), &Wo_row, output, input_dim, blocks_per_input);

    // Expected: 100 × 50 = 5000
    EXPECT_EQ(output, 5000);
}

TEST_F(Test__Q16WoProjection, RowGemv_MultipleValues_Block64)
{
    const int input_dim = 64;
    const int blocks_per_input = 1;

    // Sequential context values
    std::vector<int16_t> context_int16(input_dim);
    for (int i = 0; i < input_dim; ++i)
    {
        context_int16[i] = static_cast<int16_t>(i + 1);
    }

    // Wo = all ones
    std::vector<int16_t> wo_vals(64, 1);
    Q16_1Block_64 Wo_row = createBlock64(1.0f, wo_vals);

    int32_t output;
    q16_wo_row_gemv<Q16_1Block_64>(
        context_int16.data(), &Wo_row, output, input_dim, blocks_per_input);

    // Expected: sum(1..64) = 64×65/2 = 2080
    EXPECT_EQ(output, 2080);
}

// ============================================================================
// q16_quantize_to_q16_1 Tests
// ============================================================================

TEST_F(Test__Q16WoProjection, Quantize_ZeroValues_Block64)
{
    const int num_values = 64;
    const int blocks_per_output = 1;

    std::vector<int32_t> accumulators(num_values, 0);
    Q16_1Block_64 output;

    q16_quantize_to_q16_1<Q16_1Block_64>(
        accumulators.data(), &output, num_values, 1.0f, blocks_per_output);

    // All zeros
    for (int i = 0; i < 64; ++i)
    {
        EXPECT_EQ(output.qs[i], 0) << "Non-zero at " << i;
    }
}

TEST_F(Test__Q16WoProjection, Quantize_PositiveValues_Block64)
{
    const int num_values = 64;
    const int blocks_per_output = 1;

    // Accumulators with sequential values
    std::vector<int32_t> accumulators(num_values);
    for (int i = 0; i < num_values; ++i)
    {
        accumulators[i] = (i + 1) * 100;
    }

    Q16_1Block_64 output;

    q16_quantize_to_q16_1<Q16_1Block_64>(
        accumulators.data(), &output, num_values, 1.0f, blocks_per_output);

    // Max value is 6400 (64 × 100)
    // Scale should be ~6400/32767 ≈ 0.195
    // Largest value should quantize to ~32767

    // Check that values are proportional
    // output.qs[63] should be largest, output.qs[0] smallest
    EXPECT_GT(output.qs[63], output.qs[0]);
    EXPECT_GT(output.qs[63], 30000); // Should be close to max
}

// ============================================================================
// q16_wo_projection (Full GEMV) Tests
// ============================================================================

TEST_F(Test__Q16WoProjection, Projection_SmallModel_Block64)
{
    // Small test: d_model=64, input_dim=64, 1 block each
    const int d_model = 64;
    const int input_dim = 64;
    const int blocks_per_input = 1;
    const int blocks_per_output = 1;

    // Create context (from P×V accumulation)
    std::vector<int32_t> context_int32(input_dim);
    for (int i = 0; i < input_dim; ++i)
    {
        context_int32[i] = (i + 1) * 1000; // Known values
    }

    // Create Wo weight matrix [d_model × input_dim]
    // For simplicity, Wo[d, :] = d×ones → projection[d] = d × sum(context)
    std::vector<Q16_1Block_64> Wo(d_model * blocks_per_input);
    for (int d = 0; d < d_model; ++d)
    {
        std::vector<int16_t> row_vals(64, static_cast<int16_t>(d + 1));
        Wo[d] = createBlock64(1.0f, row_vals);
    }

    // Output
    std::vector<Q16_1Block_64> output(blocks_per_output);

    q16_wo_projection<Q16_1Block_64>(
        context_int32.data(),
        Wo.data(),
        output.data(),
        d_model,
        input_dim,
        blocks_per_input,
        blocks_per_output);

    // Output should be quantized Q16_1
    // Verify it's not all zeros
    bool all_zero = true;
    for (int i = 0; i < 64; ++i)
    {
        if (output[0].qs[i] != 0)
        {
            all_zero = false;
            break;
        }
    }
    EXPECT_FALSE(all_zero) << "Output is all zeros";

    // Verify scale is reasonable
    EXPECT_GT(output[0].d, 0.0f) << "Scale should be positive";
}

// ============================================================================
// Dispatch Tests
// ============================================================================

TEST_F(Test__Q16WoProjection, Dispatch_Block64)
{
    const int d_model = 64;
    const int input_dim = 64;

    std::vector<int32_t> context_int32(input_dim, 1000);
    std::vector<Q16_1Block_64> Wo(d_model);
    for (int d = 0; d < d_model; ++d)
    {
        Wo[d] = createRandomBlock64(0.01f);
    }

    std::vector<Q16_1Block_64> output(1);

    q16_wo_projection_dispatch(
        context_int32.data(),
        Wo.data(),
        output.data(),
        d_model, input_dim,
        Q16BlockSize::BLOCK_64);

    // Basic sanity check
    EXPECT_NE(output[0].d, 0.0f);
}

TEST_F(Test__Q16WoProjection, Dispatch_Block128)
{
    const int d_model = 128;
    const int input_dim = 128;

    std::vector<int32_t> context_int32(input_dim, 1000);
    std::vector<Q16_1Block_128> Wo(d_model);
    for (int d = 0; d < d_model; ++d)
    {
        Wo[d] = createRandomBlock128(0.01f);
    }

    std::vector<Q16_1Block_128> output(1);

    q16_wo_projection_dispatch(
        context_int32.data(),
        Wo.data(),
        output.data(),
        d_model, input_dim,
        Q16BlockSize::BLOCK_128);

    // Basic sanity check
    EXPECT_NE(output[0].d, 0.0f);
}

// ============================================================================
// FP32 Accuracy Tests - Wo Projection Error Analysis
// ============================================================================

TEST_F(Test__Q16WoProjection, FP32Accuracy_10000Projections_Block64)
{
    /**
     * ACCURACY TEST: Compare Q16 Wo projection against FP32 reference GEMV.
     *
     * The Wo projection computes: output[d] = Σ_i context[i] × Wo[d][i]
     *
     * In integer domain:
     *   output_int[d] = Σ_i context_int16[i] × wo_int16[d][i]
     *
     * FP32 equivalent:
     *   output_fp32[d] = Σ_i (context_int16[i] × context_scale) × (wo_int16[d][i] × wo_scale)
     *                  = context_scale × wo_scale × output_int[d]
     *
     * This test measures the end-to-end accuracy of the Wo projection path.
     */
    const int num_tests = 10000;
    const int d_model = 64;
    const int input_dim = 64; // Typically head_dim
    const float context_scale = 1.0f / 128.0f;
    const float wo_scale = 1.0f / 128.0f;
    const int blocks_per_input = 1; // Single block for 64-dim

    double total_rel_error = 0.0;
    double max_rel_error = 0.0;
    int num_samples = 0;

    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int t = 0; t < num_tests; ++t)
    {
        // Generate random FP32 context (attention output)
        std::vector<float> context_fp32(input_dim);
        for (int i = 0; i < input_dim; ++i)
        {
            context_fp32[i] = dist(rng_);
        }

        // Generate random FP32 Wo weights
        std::vector<std::vector<float>> Wo_fp32(d_model, std::vector<float>(input_dim));
        for (int d = 0; d < d_model; ++d)
        {
            for (int i = 0; i < input_dim; ++i)
            {
                Wo_fp32[d][i] = dist(rng_);
            }
        }

        // FP32 reference: output[d] = Σ_i context[i] × Wo[d][i]
        std::vector<double> output_fp32(d_model, 0.0);
        for (int d = 0; d < d_model; ++d)
        {
            for (int i = 0; i < input_dim; ++i)
            {
                output_fp32[d] += context_fp32[i] * Wo_fp32[d][i];
            }
        }

        // Quantize context to INT16
        std::vector<int16_t> context_int16(input_dim);
        for (int i = 0; i < input_dim; ++i)
        {
            context_int16[i] = static_cast<int16_t>(std::clamp(
                std::round(context_fp32[i] / context_scale), -32767.0f, 32767.0f));
        }

        // Quantize Wo to Q16_1 blocks (one block per output dim)
        std::vector<Q16_1Block_64> Wo_q16(d_model);
        for (int d = 0; d < d_model; ++d)
        {
            Wo_q16[d].d = wo_scale;
            int32_t sum_qs = 0;
            for (int i = 0; i < input_dim; ++i)
            {
                int16_t v = static_cast<int16_t>(std::clamp(
                    std::round(Wo_fp32[d][i] / wo_scale), -32767.0f, 32767.0f));
                Wo_q16[d].qs[i] = v;
                sum_qs += v;
            }
            Wo_q16[d].sum_qs = sum_qs;
        }

        // Q16 Wo projection (GEMV: [d_model, input_dim] × [input_dim] -> [d_model])
        std::vector<int32_t> output_int32(d_model, 0);
        for (int d = 0; d < d_model; ++d)
        {
            int32_t result = 0;
            q16_wo_row_gemv<Q16_1Block_64>(
                context_int16.data(), &Wo_q16[d], result, input_dim, blocks_per_input);
            output_int32[d] = result;
        }

        // Convert back to FP32 and compare
        for (int d = 0; d < d_model; ++d)
        {
            double q16_output_fp32 = static_cast<double>(output_int32[d]) * context_scale * wo_scale;

            double abs_err = std::abs(output_fp32[d] - q16_output_fp32);
            double rel_err = (std::abs(output_fp32[d]) > 1e-6)
                                 ? abs_err / std::abs(output_fp32[d])
                                 : abs_err;

            total_rel_error += rel_err;
            max_rel_error = std::max(max_rel_error, rel_err);
            num_samples++;
        }
    }

    double mean_rel_error = total_rel_error / num_samples;

    std::cout << "\n"
              << "╔══════════════════════════════════════════════════════════════════════════╗\n"
              << "║         Q16 Wo PROJECTION vs FP32 REFERENCE (10,000 Projections)         ║\n"
              << "╠══════════════════════════════════════════════════════════════════════════╣\n"
              << "║ Model dimension:         " << std::setw(10) << d_model << "                                    ║\n"
              << "║ Input dimension:         " << std::setw(10) << input_dim << "                                    ║\n"
              << "║ Total output samples:    " << std::setw(10) << num_samples << "                                    ║\n"
              << "║ Mean Relative Error:     " << std::setw(14) << std::scientific << std::setprecision(6) << mean_rel_error << "                            ║\n"
              << "║ Max Relative Error:      " << std::setw(14) << max_rel_error << "                            ║\n"
              << "╚══════════════════════════════════════════════════════════════════════════╝\n\n";

    // Wo projection has error from both context and weight quantization
    EXPECT_LT(mean_rel_error, 0.05) << "Mean relative error too high for Q16 Wo projection";
}

TEST_F(Test__Q16WoProjection, FP32Accuracy_CosineSimilarity)
{
    /**
     * Test Wo projection accuracy using cosine similarity.
     *
     * For each projection, we compare the output vector direction.
     * This is the most meaningful metric for neural network activations.
     */
    const int num_tests = 1000;
    const int d_model = 64;
    const int input_dim = 64;
    const float context_scale = 1.0f / 128.0f;
    const float wo_scale = 1.0f / 128.0f;
    const int blocks_per_input = 1;

    double total_cosine_sim = 0.0;
    double min_cosine_sim = 1.0;

    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int t = 0; t < num_tests; ++t)
    {
        // Generate random FP32 context (attention output)
        std::vector<float> context_fp32(input_dim);
        for (int i = 0; i < input_dim; ++i)
        {
            context_fp32[i] = dist(rng_);
        }

        // Generate random FP32 Wo weights
        std::vector<std::vector<float>> Wo_fp32(d_model, std::vector<float>(input_dim));
        for (int d = 0; d < d_model; ++d)
        {
            for (int i = 0; i < input_dim; ++i)
            {
                Wo_fp32[d][i] = dist(rng_);
            }
        }

        // FP32 reference: output[d] = Σ_i context[i] × Wo[d][i]
        std::vector<double> output_fp32(d_model, 0.0);
        for (int d = 0; d < d_model; ++d)
        {
            for (int i = 0; i < input_dim; ++i)
            {
                output_fp32[d] += context_fp32[i] * Wo_fp32[d][i];
            }
        }

        // Quantize context to INT16
        std::vector<int16_t> context_int16(input_dim);
        for (int i = 0; i < input_dim; ++i)
        {
            context_int16[i] = static_cast<int16_t>(std::clamp(
                std::round(context_fp32[i] / context_scale), -32767.0f, 32767.0f));
        }

        // Quantize Wo to Q16_1 blocks
        std::vector<Q16_1Block_64> Wo_q16(d_model);
        for (int d = 0; d < d_model; ++d)
        {
            Wo_q16[d].d = wo_scale;
            int32_t sum_qs = 0;
            for (int i = 0; i < input_dim; ++i)
            {
                int16_t v = static_cast<int16_t>(std::clamp(
                    std::round(Wo_fp32[d][i] / wo_scale), -32767.0f, 32767.0f));
                Wo_q16[d].qs[i] = v;
                sum_qs += v;
            }
            Wo_q16[d].sum_qs = sum_qs;
        }

        // Q16 Wo projection
        std::vector<double> output_q16(d_model);
        for (int d = 0; d < d_model; ++d)
        {
            int32_t result = 0;
            q16_wo_row_gemv<Q16_1Block_64>(
                context_int16.data(), &Wo_q16[d], result, input_dim, blocks_per_input);
            output_q16[d] = static_cast<double>(result) * context_scale * wo_scale;
        }

        // Cosine similarity
        double dot_ab = 0.0, norm_a = 0.0, norm_b = 0.0;
        for (int d = 0; d < d_model; ++d)
        {
            dot_ab += output_fp32[d] * output_q16[d];
            norm_a += output_fp32[d] * output_fp32[d];
            norm_b += output_q16[d] * output_q16[d];
        }

        double cosine_sim = dot_ab / (std::sqrt(norm_a) * std::sqrt(norm_b) + 1e-10);
        total_cosine_sim += cosine_sim;
        min_cosine_sim = std::min(min_cosine_sim, cosine_sim);
    }

    double mean_cosine_sim = total_cosine_sim / num_tests;

    std::cout << "\n"
              << "╔══════════════════════════════════════════════════════════════════════════╗\n"
              << "║       Q16 Wo PROJECTION - COSINE SIMILARITY (1000 Projections)           ║\n"
              << "╠══════════════════════════════════════════════════════════════════════════╣\n"
              << "║ Model dimension:         " << std::setw(10) << d_model << "                                    ║\n"
              << "║ Input dimension:         " << std::setw(10) << input_dim << "                                    ║\n"
              << "║ Mean Cosine Similarity:  " << std::setw(14) << std::fixed << std::setprecision(6) << mean_cosine_sim << "                            ║\n"
              << "║ Min Cosine Similarity:   " << std::setw(14) << min_cosine_sim << "                            ║\n"
              << "╚══════════════════════════════════════════════════════════════════════════╝\n\n";

    // Wo output vectors should be very similar in direction
    EXPECT_GT(mean_cosine_sim, 0.99) << "Mean cosine similarity too low for Q16 Wo";
    EXPECT_GT(min_cosine_sim, 0.95) << "Worst-case cosine similarity too low";
}

TEST_F(Test__Q16WoProjection, FP32Accuracy_NormalizationPreservation)
{
    /**
     * Test that INT32→INT16 normalization preserves relative magnitudes.
     *
     * The normalize function maps INT32 context to INT16 for Wo GEMV.
     * It should preserve the relative scale of values.
     */
    const int num_tests = 1000;
    const int input_dim = 64;

    double total_corr_error = 0.0;
    int num_samples = 0;

    std::uniform_int_distribution<int32_t> dist(-1000000, 1000000);

    for (int t = 0; t < num_tests; ++t)
    {
        // Random INT32 context
        std::vector<int32_t> context_int32(input_dim);
        for (int i = 0; i < input_dim; ++i)
        {
            context_int32[i] = dist(rng_);
        }

        // Normalize to INT16
        std::vector<int16_t> context_int16(input_dim);
        float scale = 0.0f;
        q16_context_normalize_to_int16(
            context_int32.data(), context_int16.data(), scale, input_dim);

        // Verify: original ≈ int16 × scale
        for (int i = 0; i < input_dim; ++i)
        {
            double reconstructed = static_cast<double>(context_int16[i]) * scale;
            double original = static_cast<double>(context_int32[i]);

            if (std::abs(original) > 1.0)
            {
                double rel_err = std::abs(reconstructed - original) / std::abs(original);
                total_corr_error += rel_err;
                num_samples++;
            }
        }
    }

    double mean_error = total_corr_error / num_samples;

    std::cout << "  [WoProjection Normalization] Mean reconstruction error: "
              << std::scientific << std::setprecision(4) << mean_error << std::endl;

    // Normalization should preserve values very accurately (just scaling)
    EXPECT_LT(mean_error, 0.01) << "Normalization introduces too much error";
}

// ============================================================================
// Cache-Aware Wo Tile Configuration Tests
// ============================================================================

/**
 * @brief Test that compute_wo_tile_config returns reasonable tile sizes
 */
TEST_F(Test__Q16WoProjection, WoTileConfig_SanityCheck)
{
    // Test Qwen-0.5B dimensions: d_model=896, input_dim=896
    {
        auto cfg = compute_wo_tile_config(896, 896, 1);
        EXPECT_GE(cfg.M_tile, 16) << "M_tile should be at least 16";
        EXPECT_LE(cfg.M_tile, 256) << "M_tile should be at most 256";
        EXPECT_GE(cfg.K_tile, 32) << "K_tile should be at least 32";
        EXPECT_LE(cfg.K_tile, 512) << "K_tile should be at most 512";
        EXPECT_EQ(cfg.N_tile, 1) << "N_tile should be 1 for decode";
        EXPECT_EQ(cfg.M_tile % 16, 0) << "M_tile should be multiple of 16";
        EXPECT_EQ(cfg.K_tile % 32, 0) << "K_tile should be multiple of 32";
    }

    // Test Qwen-7B dimensions: d_model=3584, input_dim=3584
    {
        auto cfg = compute_wo_tile_config(3584, 3584, 1);
        EXPECT_GE(cfg.M_tile, 16);
        EXPECT_GE(cfg.K_tile, 32);
        EXPECT_EQ(cfg.N_tile, 1);
    }

    // Test with batch (prefill)
    {
        auto cfg = compute_wo_tile_config(896, 896, 16);
        EXPECT_GE(cfg.N_tile, 1) << "N_tile should be at least 1";
        EXPECT_LE(cfg.N_tile, 16) << "N_tile should be at most batch_size";
    }
}

/**
 * @brief Test that tile sizes respect L1 cache constraints
 */
TEST_F(Test__Q16WoProjection, WoTileConfig_L1CacheConstraint)
{
    const auto &cache = cache_info();

    // Skip test if cache detection failed
    if (cache.l1_size == 0)
    {
        GTEST_SKIP() << "Cache detection not available";
    }

    for (int d_model : {896, 2048, 3584})
    {
        for (int input_dim : {896, 2048, 3584})
        {
            auto cfg = compute_wo_tile_config(d_model, input_dim, 1);

            // L1 working set (Wo tile + context tile) should fit in 50% of L1
            size_t l1_working = cfg.l1_working_set();
            size_t l1_target = cache.l1_size / 2;

            EXPECT_LE(l1_working, l1_target)
                << "L1 working set (" << l1_working << " bytes) exceeds 50% of L1 ("
                << l1_target << " bytes) for d_model=" << d_model
                << ", input_dim=" << input_dim;
        }
    }
}

/**
 * @brief Test default tile config matches legacy constants
 */
TEST_F(Test__Q16WoProjection, WoTileConfig_DefaultMatchesLegacy)
{
    auto cfg = default_wo_tile_config(896, 896, 1);

    EXPECT_EQ(cfg.M_tile, WO_M_TILE);
    EXPECT_EQ(cfg.K_tile, WO_K_TILE);
    EXPECT_EQ(cfg.N_tile, 1);
    EXPECT_EQ(cfg.d_model, 896);
    EXPECT_EQ(cfg.input_dim, 896);
}

/**
 * @brief Test that N_tile scales with batch size
 */
TEST_F(Test__Q16WoProjection, WoTileConfig_BatchScaling)
{
    const int d_model = 896;
    const int input_dim = 896;

    // Decode: N_tile should be 1
    {
        auto cfg = compute_wo_tile_config(d_model, input_dim, 1);
        EXPECT_EQ(cfg.N_tile, 1);
    }

    // Small batch: N_tile should grow
    {
        auto cfg_4 = compute_wo_tile_config(d_model, input_dim, 4);
        auto cfg_8 = compute_wo_tile_config(d_model, input_dim, 8);
        EXPECT_GE(cfg_4.N_tile, 1);
        EXPECT_GE(cfg_8.N_tile, cfg_4.N_tile) << "N_tile should grow with batch";
    }

    // Large batch: N_tile should be capped
    {
        auto cfg = compute_wo_tile_config(d_model, input_dim, 1024);
        EXPECT_LE(cfg.N_tile, 16) << "N_tile should be capped at 16";
    }
}

/**
 * @brief Test memory footprint calculations
 */
TEST_F(Test__Q16WoProjection, WoTileConfig_MemoryFootprint)
{
    auto cfg = compute_wo_tile_config(896, 896, 4);

    // Verify memory calculations
    size_t expected_wo_tile = static_cast<size_t>(cfg.M_tile) * cfg.K_tile * sizeof(int16_t);
    size_t expected_context_tile = static_cast<size_t>(cfg.K_tile) * sizeof(int16_t);
    size_t expected_accum_tile = static_cast<size_t>(cfg.M_tile) * cfg.N_tile * sizeof(int32_t);

    EXPECT_EQ(cfg.wo_tile_bytes(), expected_wo_tile);
    EXPECT_EQ(cfg.context_tile_bytes(), expected_context_tile);
    EXPECT_EQ(cfg.accumulator_tile_bytes(), expected_accum_tile);
    EXPECT_EQ(cfg.l1_working_set(), expected_wo_tile + expected_context_tile);
}
