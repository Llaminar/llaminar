/**
 * @file Test__Q8_1_Add_Extreme.cpp
 * @brief Comprehensive unit tests for q8_1_add_q8_1 SIMD primitive
 *
 * Tests extreme and edge cases to expose potential bugs in Q8_1 block addition:
 * - Catastrophic cancellation (values that nearly cancel out)
 * - Scale mismatches (vastly different magnitudes)
 * - Saturation (values at int8 limits)
 * - Numerical edge cases (denormals, very large, very small)
 * - Multi-block scenarios
 * - Realistic transformer activation patterns
 *
 * @author David Sanftenberg
 * @date 2025-12-11
 */

#include <gtest/gtest.h>
#include "v2/tensors/SIMDHelpers.h"
#include "v2/tensors/Tensors.h"
#include <vector>
#include <cmath>
#include <random>
#include <limits>
#include <numeric>
#include <iomanip>

using namespace llaminar2;

class Test__Q8_1_Add_Extreme : public ::testing::Test
{
protected:
    // Helper to quantize FP32 array to Q8_1 blocks
    void quantize_to_blocks(const std::vector<float> &fp32, std::vector<Q8_1Block> &blocks)
    {
        const size_t n_blocks = (fp32.size() + 31) / 32;
        blocks.resize(n_blocks);

        for (size_t blk = 0; blk < n_blocks; ++blk)
        {
            size_t offset = blk * 32;
            size_t valid = std::min<size_t>(32, fp32.size() - offset);
            simd::quantize_single_block_scalar(fp32.data() + offset, blocks[blk], valid);
        }
    }

    // Helper to dequantize Q8_1 blocks back to FP32
    void dequantize_blocks(const std::vector<Q8_1Block> &blocks, std::vector<float> &fp32, size_t num_elements)
    {
        fp32.resize(num_elements);

        for (size_t i = 0; i < num_elements; ++i)
        {
            size_t blk = i / 32;
            size_t idx = i % 32;
            float d = simd::fp16_to_fp32(blocks[blk].d);
            fp32[i] = d * static_cast<float>(blocks[blk].qs[idx]);
        }
    }

    // Compute FP32 reference addition
    std::vector<float> fp32_reference_add(const std::vector<float> &a, const std::vector<float> &b)
    {
        std::vector<float> result(a.size());
        for (size_t i = 0; i < a.size(); ++i)
        {
            result[i] = a[i] + b[i];
        }
        return result;
    }

    // Compute max absolute error
    double max_abs_error(const std::vector<float> &actual, const std::vector<float> &expected)
    {
        double max_err = 0.0;
        for (size_t i = 0; i < actual.size(); ++i)
        {
            max_err = std::max(max_err, static_cast<double>(std::abs(actual[i] - expected[i])));
        }
        return max_err;
    }

    // Compute relative L2 error
    double relative_l2_error(const std::vector<float> &actual, const std::vector<float> &expected)
    {
        double diff_sq = 0.0, ref_sq = 0.0;
        for (size_t i = 0; i < actual.size(); ++i)
        {
            double diff = actual[i] - expected[i];
            diff_sq += diff * diff;
            ref_sq += expected[i] * expected[i];
        }
        if (ref_sq < 1e-20)
            return diff_sq > 1e-20 ? 1.0 : 0.0;
        return std::sqrt(diff_sq / ref_sq);
    }

    // Compute cosine similarity
    double cosine_similarity(const std::vector<float> &a, const std::vector<float> &b)
    {
        double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
        for (size_t i = 0; i < a.size(); ++i)
        {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }
        if (norm_a < 1e-20 || norm_b < 1e-20)
            return 0.0;
        return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
    }

    // Print debug info for a block
    void debug_block(const char *name, const Q8_1Block &blk)
    {
        float d = simd::fp16_to_fp32(blk.d);
        std::cout << name << ": d=" << d << " sum_qs=" << blk.sum_qs << " qs=[";
        for (int i = 0; i < 8; ++i)
            std::cout << (int)blk.qs[i] << ",";
        std::cout << "...]" << std::endl;
    }
};

// =============================================================================
// CATASTROPHIC CANCELLATION TESTS
// These test scenarios where A + B ≈ 0, which can cause numerical issues
// =============================================================================

TEST_F(Test__Q8_1_Add_Extreme, ExactCancellation_SingleElement)
{
    // A = 100.0, B = -100.0 → Sum should be exactly 0
    Q8_1Block a, b, out;

    float fa[32] = {0};
    float fb[32] = {0};
    fa[0] = 100.0f;
    fb[0] = -100.0f;

    simd::quantize_single_block_scalar(fa, a, 32);
    simd::quantize_single_block_scalar(fb, b, 32);

    simd::q8_1_add_q8_1(&a, &b, &out, 32);

    // Dequantize and check
    float d = simd::fp16_to_fp32(out.d);
    float result = d * out.qs[0];

    // With exact cancellation, result should be very close to 0
    EXPECT_NEAR(result, 0.0f, 1.0f) << "Exact cancellation should produce ~0";
}

TEST_F(Test__Q8_1_Add_Extreme, NearCancellation_LargeValues)
{
    // A = 1000.0, B = -999.0 → Sum should be ~1.0
    // This is catastrophic cancellation: we lose precision
    Q8_1Block a, b, out;

    float fa[32] = {0};
    float fb[32] = {0};
    fa[0] = 1000.0f;
    fb[0] = -999.0f;

    simd::quantize_single_block_scalar(fa, a, 32);
    simd::quantize_single_block_scalar(fb, b, 32);

    // Debug output
    std::cout << "Near cancellation test:" << std::endl;
    debug_block("A", a);
    debug_block("B", b);

    simd::q8_1_add_q8_1(&a, &b, &out, 32);

    debug_block("Out", out);

    float d = simd::fp16_to_fp32(out.d);
    float result = d * out.qs[0];

    std::cout << "Expected: 1.0, Got: " << result << std::endl;

    // The quantization error here can be significant
    // A is quantized with scale ~7.87 (1000/127), so A[0] = 127*7.87 ≈ 1000
    // B is quantized with scale ~7.87 (999/127), so B[0] = -127*7.87 ≈ -999.7
    // Sum ≈ 0.3, but could vary due to rounding
    EXPECT_NEAR(result, 1.0f, 15.0f) << "Near cancellation has high error";
}

TEST_F(Test__Q8_1_Add_Extreme, FullBlockCancellation)
{
    // Every element cancels: A[i] = -B[i]
    Q8_1Block a, b, out;

    std::mt19937 gen(12345);
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);

    float fa[32], fb[32];
    for (int i = 0; i < 32; ++i)
    {
        fa[i] = dist(gen);
        fb[i] = -fa[i]; // Exact cancellation
    }

    simd::quantize_single_block_scalar(fa, a, 32);
    simd::quantize_single_block_scalar(fb, b, 32);

    simd::q8_1_add_q8_1(&a, &b, &out, 32);

    // All results should be near zero
    float d = simd::fp16_to_fp32(out.d);
    for (int i = 0; i < 32; ++i)
    {
        float result = d * out.qs[i];
        // Quantization error means we won't get exactly 0
        // Max error ≈ 2 * (max_abs / 127) since both A and B have quantization error
        float max_expected_error = 2.0f * (100.0f / 127.0f);
        EXPECT_NEAR(result, 0.0f, max_expected_error * 2)
            << "Full block cancellation at index " << i;
    }
}

TEST_F(Test__Q8_1_Add_Extreme, PartialCancellation_MixedMagnitudes)
{
    // Some elements cancel, others don't
    Q8_1Block a, b, out;

    float fa[32], fb[32], expected[32];

    // First half: near-cancellation
    for (int i = 0; i < 16; ++i)
    {
        fa[i] = 50.0f + i;
        fb[i] = -(50.0f + i) + 0.1f; // Nearly cancels
        expected[i] = 0.1f;
    }

    // Second half: additive
    for (int i = 16; i < 32; ++i)
    {
        fa[i] = 10.0f;
        fb[i] = 20.0f;
        expected[i] = 30.0f;
    }

    simd::quantize_single_block_scalar(fa, a, 32);
    simd::quantize_single_block_scalar(fb, b, 32);

    simd::q8_1_add_q8_1(&a, &b, &out, 32);

    float d = simd::fp16_to_fp32(out.d);

    // Check the additive part (should be more accurate)
    for (int i = 16; i < 32; ++i)
    {
        float result = d * out.qs[i];
        EXPECT_NEAR(result, expected[i], 5.0f)
            << "Additive part at index " << i;
    }
}

// =============================================================================
// SCALE MISMATCH TESTS
// Test when A and B have vastly different magnitudes
// =============================================================================

TEST_F(Test__Q8_1_Add_Extreme, ScaleMismatch_1000x)
{
    // A has values ~1000, B has values ~1
    Q8_1Block a, b, out;

    float fa[32], fb[32];
    for (int i = 0; i < 32; ++i)
    {
        fa[i] = 1000.0f + i * 10.0f;
        fb[i] = 1.0f + i * 0.01f;
    }

    simd::quantize_single_block_scalar(fa, a, 32);
    simd::quantize_single_block_scalar(fb, b, 32);

    std::cout << "Scale mismatch 1000x:" << std::endl;
    debug_block("A", a);
    debug_block("B", b);

    simd::q8_1_add_q8_1(&a, &b, &out, 32);

    debug_block("Out", out);

    // With 1000x scale difference, B's contribution may be lost to quantization
    float d = simd::fp16_to_fp32(out.d);
    float result_0 = d * out.qs[0];

    // A[0] = 1000, B[0] = 1, expected = 1001
    // But B quantized with its own scale (~0.01) has very low qs values
    // When dequantized with A's scale, B effectively becomes 0
    std::cout << "Expected: 1001, Got: " << result_0 << std::endl;

    // The smaller values in B should be swamped
    EXPECT_NEAR(result_0, 1001.0f, 50.0f);
}

TEST_F(Test__Q8_1_Add_Extreme, ScaleMismatch_1000000x)
{
    // Extreme: A = 1e6, B = 1
    Q8_1Block a, b, out;

    float fa[32] = {0}, fb[32] = {0};
    fa[0] = 1e6f;
    fb[0] = 1.0f;

    simd::quantize_single_block_scalar(fa, a, 32);
    simd::quantize_single_block_scalar(fb, b, 32);

    simd::q8_1_add_q8_1(&a, &b, &out, 32);

    float d = simd::fp16_to_fp32(out.d);
    float result = d * out.qs[0];

    // B should be completely lost
    EXPECT_NEAR(result, 1e6f, 1e4f);
}

TEST_F(Test__Q8_1_Add_Extreme, ScaleMismatch_Reversed)
{
    // A has small values, B has large values
    Q8_1Block a, b, out;

    float fa[32] = {0}, fb[32] = {0};
    fa[0] = 0.001f;
    fb[0] = 1000.0f;

    simd::quantize_single_block_scalar(fa, a, 32);
    simd::quantize_single_block_scalar(fb, b, 32);

    simd::q8_1_add_q8_1(&a, &b, &out, 32);

    float d = simd::fp16_to_fp32(out.d);
    float result = d * out.qs[0];

    // A should be lost
    EXPECT_NEAR(result, 1000.0f, 10.0f);
}

// =============================================================================
// SATURATION TESTS
// Test values at or near int8 limits
// =============================================================================

TEST_F(Test__Q8_1_Add_Extreme, Saturation_BothMaxPositive)
{
    // Both blocks have max positive values → should saturate
    Q8_1Block a, b, out;

    float fa[32], fb[32];
    for (int i = 0; i < 32; ++i)
    {
        fa[i] = 100.0f;
        fb[i] = 100.0f;
    }

    simd::quantize_single_block_scalar(fa, a, 32);
    simd::quantize_single_block_scalar(fb, b, 32);

    // Both should have qs[i] = 127
    EXPECT_EQ(a.qs[0], 127);
    EXPECT_EQ(b.qs[0], 127);

    simd::q8_1_add_q8_1(&a, &b, &out, 32);

    // Output should have larger scale and qs[i] = 127
    float d = simd::fp16_to_fp32(out.d);
    float result = d * out.qs[0];

    EXPECT_NEAR(result, 200.0f, 5.0f);
}

TEST_F(Test__Q8_1_Add_Extreme, Saturation_BothMaxNegative)
{
    Q8_1Block a, b, out;

    float fa[32], fb[32];
    for (int i = 0; i < 32; ++i)
    {
        fa[i] = -100.0f;
        fb[i] = -100.0f;
    }

    simd::quantize_single_block_scalar(fa, a, 32);
    simd::quantize_single_block_scalar(fb, b, 32);

    simd::q8_1_add_q8_1(&a, &b, &out, 32);

    float d = simd::fp16_to_fp32(out.d);
    float result = d * out.qs[0];

    EXPECT_NEAR(result, -200.0f, 5.0f);
}

TEST_F(Test__Q8_1_Add_Extreme, Saturation_MixedSigns)
{
    // A = +max, B = -max → cancellation
    Q8_1Block a, b, out;

    float fa[32], fb[32];
    for (int i = 0; i < 32; ++i)
    {
        fa[i] = 127.0f;
        fb[i] = -127.0f;
    }

    simd::quantize_single_block_scalar(fa, a, 32);
    simd::quantize_single_block_scalar(fb, b, 32);

    simd::q8_1_add_q8_1(&a, &b, &out, 32);

    // Should be near zero
    float d = simd::fp16_to_fp32(out.d);
    for (int i = 0; i < 32; ++i)
    {
        float result = d * out.qs[i];
        EXPECT_NEAR(result, 0.0f, 3.0f);
    }
}

// =============================================================================
// NUMERICAL EDGE CASES
// =============================================================================

TEST_F(Test__Q8_1_Add_Extreme, ZeroBlocks)
{
    // Both blocks are zero
    Q8_1Block a, b, out;

    float fa[32] = {0}, fb[32] = {0};

    simd::quantize_single_block_scalar(fa, a, 32);
    simd::quantize_single_block_scalar(fb, b, 32);

    EXPECT_EQ(simd::fp16_to_fp32(a.d), 0.0f);
    EXPECT_EQ(simd::fp16_to_fp32(b.d), 0.0f);

    simd::q8_1_add_q8_1(&a, &b, &out, 32);

    EXPECT_EQ(simd::fp16_to_fp32(out.d), 0.0f);
    EXPECT_EQ(out.sum_qs, 0);
}

TEST_F(Test__Q8_1_Add_Extreme, OneZeroBlock)
{
    // A is zero, B is not → should be B
    Q8_1Block a, b, out;

    float fa[32] = {0};
    float fb[32];
    for (int i = 0; i < 32; ++i)
        fb[i] = 10.0f + i;

    simd::quantize_single_block_scalar(fa, a, 32);
    simd::quantize_single_block_scalar(fb, b, 32);

    simd::q8_1_add_q8_1(&a, &b, &out, 32);

    // Result should match B
    float d_out = simd::fp16_to_fp32(out.d);
    float d_b = simd::fp16_to_fp32(b.d);

    for (int i = 0; i < 32; ++i)
    {
        float result = d_out * out.qs[i];
        float expected = d_b * b.qs[i];
        EXPECT_NEAR(result, expected, 1.0f);
    }
}

TEST_F(Test__Q8_1_Add_Extreme, VerySmallValues)
{
    // Values near zero
    Q8_1Block a, b, out;

    float fa[32], fb[32];
    for (int i = 0; i < 32; ++i)
    {
        fa[i] = 1e-5f * (i + 1);
        fb[i] = 2e-5f * (i + 1);
    }

    simd::quantize_single_block_scalar(fa, a, 32);
    simd::quantize_single_block_scalar(fb, b, 32);

    simd::q8_1_add_q8_1(&a, &b, &out, 32);

    float d = simd::fp16_to_fp32(out.d);
    for (int i = 0; i < 32; ++i)
    {
        float result = d * out.qs[i];
        float expected = 3e-5f * (i + 1);
        // Large relative tolerance for tiny values
        EXPECT_NEAR(result, expected, expected * 0.5f + 1e-6f);
    }
}

TEST_F(Test__Q8_1_Add_Extreme, VeryLargeValues)
{
    // Values near FP16 max
    Q8_1Block a, b, out;

    float fa[32], fb[32];
    float large = 60000.0f; // Near FP16 max (~65504)
    for (int i = 0; i < 32; ++i)
    {
        fa[i] = large;
        fb[i] = large;
    }

    simd::quantize_single_block_scalar(fa, a, 32);
    simd::quantize_single_block_scalar(fb, b, 32);

    simd::q8_1_add_q8_1(&a, &b, &out, 32);

    float d = simd::fp16_to_fp32(out.d);
    float result = d * out.qs[0];

    // May overflow FP16 scale, check behavior
    std::cout << "Large values test: d=" << d << " qs[0]=" << (int)out.qs[0]
              << " result=" << result << std::endl;

    // Either we get ~120000, or scale overflows
    EXPECT_TRUE(result > 100000.0f || std::isinf(result) || std::isnan(result));
}

TEST_F(Test__Q8_1_Add_Extreme, MixedPositiveNegative)
{
    // Alternating signs
    Q8_1Block a, b, out;

    float fa[32], fb[32];
    for (int i = 0; i < 32; ++i)
    {
        fa[i] = (i % 2 == 0) ? 50.0f : -50.0f;
        fb[i] = 25.0f;
    }

    simd::quantize_single_block_scalar(fa, a, 32);
    simd::quantize_single_block_scalar(fb, b, 32);

    simd::q8_1_add_q8_1(&a, &b, &out, 32);

    float d = simd::fp16_to_fp32(out.d);
    for (int i = 0; i < 32; ++i)
    {
        float result = d * out.qs[i];
        float expected = (i % 2 == 0) ? 75.0f : -25.0f;
        EXPECT_NEAR(result, expected, 3.0f) << "At index " << i;
    }
}

// =============================================================================
// MULTI-BLOCK TESTS
// =============================================================================

TEST_F(Test__Q8_1_Add_Extreme, MultiBlock_Sequential)
{
    // Test multiple blocks in sequence
    const size_t num_elements = 256; // 8 blocks

    std::vector<float> fa(num_elements), fb(num_elements);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-50.0f, 50.0f);

    for (size_t i = 0; i < num_elements; ++i)
    {
        fa[i] = dist(gen);
        fb[i] = dist(gen);
    }

    std::vector<Q8_1Block> blocks_a, blocks_b, blocks_out;
    quantize_to_blocks(fa, blocks_a);
    quantize_to_blocks(fb, blocks_b);
    blocks_out.resize(blocks_a.size());

    simd::q8_1_add_q8_1(blocks_a.data(), blocks_b.data(), blocks_out.data(), num_elements);

    // Check results
    std::vector<float> result;
    dequantize_blocks(blocks_out, result, num_elements);

    auto expected = fp32_reference_add(fa, fb);

    double cos_sim = cosine_similarity(result, expected);
    double rel_l2 = relative_l2_error(result, expected);

    std::cout << "Multi-block (256 elements):" << std::endl;
    std::cout << "  Cosine similarity: " << cos_sim << std::endl;
    std::cout << "  Relative L2 error: " << rel_l2 << std::endl;

    EXPECT_GT(cos_sim, 0.99) << "Cosine similarity too low";
    EXPECT_LT(rel_l2, 0.1) << "Relative L2 error too high";
}

TEST_F(Test__Q8_1_Add_Extreme, MultiBlock_VaryingScales)
{
    // Each block has different scale
    const size_t num_elements = 128; // 4 blocks

    std::vector<float> fa(num_elements), fb(num_elements);

    // Block 0: small values
    for (size_t i = 0; i < 32; ++i)
    {
        fa[i] = 0.1f;
        fb[i] = 0.2f;
    }
    // Block 1: medium values
    for (size_t i = 32; i < 64; ++i)
    {
        fa[i] = 10.0f;
        fb[i] = 20.0f;
    }
    // Block 2: large values
    for (size_t i = 64; i < 96; ++i)
    {
        fa[i] = 1000.0f;
        fb[i] = 2000.0f;
    }
    // Block 3: mixed signs
    for (size_t i = 96; i < 128; ++i)
    {
        fa[i] = 50.0f;
        fb[i] = -25.0f;
    }

    std::vector<Q8_1Block> blocks_a, blocks_b, blocks_out;
    quantize_to_blocks(fa, blocks_a);
    quantize_to_blocks(fb, blocks_b);
    blocks_out.resize(blocks_a.size());

    simd::q8_1_add_q8_1(blocks_a.data(), blocks_b.data(), blocks_out.data(), num_elements);

    std::vector<float> result;
    dequantize_blocks(blocks_out, result, num_elements);

    // Check each block's expected values
    EXPECT_NEAR(result[0], 0.3f, 0.1f);       // Block 0
    EXPECT_NEAR(result[32], 30.0f, 2.0f);     // Block 1
    EXPECT_NEAR(result[64], 3000.0f, 100.0f); // Block 2
    EXPECT_NEAR(result[96], 25.0f, 3.0f);     // Block 3
}

// =============================================================================
// REALISTIC TRANSFORMER PATTERNS
// =============================================================================

TEST_F(Test__Q8_1_Add_Extreme, TransformerResidual_SmallPerturbation)
{
    // Simulates: hidden = residual + small_attention_output
    // Residual is large, attention output is smaller
    const size_t d_model = 896;

    std::vector<float> residual(d_model), attention(d_model);
    std::mt19937 gen(123);
    std::normal_distribution<float> dist_res(0.0f, 10.0f); // Larger values
    std::normal_distribution<float> dist_attn(0.0f, 1.0f); // Smaller values

    for (size_t i = 0; i < d_model; ++i)
    {
        residual[i] = dist_res(gen);
        attention[i] = dist_attn(gen);
    }

    std::vector<Q8_1Block> blocks_res, blocks_attn, blocks_out;
    quantize_to_blocks(residual, blocks_res);
    quantize_to_blocks(attention, blocks_attn);
    blocks_out.resize(blocks_res.size());

    simd::q8_1_add_q8_1(blocks_res.data(), blocks_attn.data(), blocks_out.data(), d_model);

    std::vector<float> result;
    dequantize_blocks(blocks_out, result, d_model);

    auto expected = fp32_reference_add(residual, attention);

    double cos_sim = cosine_similarity(result, expected);
    double rel_l2 = relative_l2_error(result, expected);

    std::cout << "Transformer residual (small perturbation):" << std::endl;
    std::cout << "  Cosine similarity: " << cos_sim << std::endl;
    std::cout << "  Relative L2 error: " << rel_l2 << std::endl;

    EXPECT_GT(cos_sim, 0.98) << "Cosine similarity too low";
}

TEST_F(Test__Q8_1_Add_Extreme, TransformerResidual_LargePerturbation)
{
    // Simulates: hidden = residual + large_ffn_output
    // Both tensors have similar magnitude
    const size_t d_model = 896;

    std::vector<float> residual(d_model), ffn(d_model);
    std::mt19937 gen(456);
    std::normal_distribution<float> dist(0.0f, 10.0f);

    for (size_t i = 0; i < d_model; ++i)
    {
        residual[i] = dist(gen);
        ffn[i] = dist(gen);
    }

    std::vector<Q8_1Block> blocks_res, blocks_ffn, blocks_out;
    quantize_to_blocks(residual, blocks_res);
    quantize_to_blocks(ffn, blocks_ffn);
    blocks_out.resize(blocks_res.size());

    simd::q8_1_add_q8_1(blocks_res.data(), blocks_ffn.data(), blocks_out.data(), d_model);

    std::vector<float> result;
    dequantize_blocks(blocks_out, result, d_model);

    auto expected = fp32_reference_add(residual, ffn);

    double cos_sim = cosine_similarity(result, expected);
    double rel_l2 = relative_l2_error(result, expected);

    std::cout << "Transformer residual (large perturbation):" << std::endl;
    std::cout << "  Cosine similarity: " << cos_sim << std::endl;
    std::cout << "  Relative L2 error: " << rel_l2 << std::endl;

    EXPECT_GT(cos_sim, 0.98) << "Cosine similarity too low";
}

TEST_F(Test__Q8_1_Add_Extreme, TransformerResidual_Cancellation)
{
    // Worst case: FFN output nearly cancels residual
    // This can happen in certain attention patterns
    const size_t d_model = 896;

    std::vector<float> residual(d_model), ffn(d_model);
    std::mt19937 gen(789);
    std::normal_distribution<float> dist(0.0f, 10.0f);

    for (size_t i = 0; i < d_model; ++i)
    {
        residual[i] = dist(gen);
        // FFN nearly cancels residual with small perturbation
        ffn[i] = -residual[i] + dist(gen) * 0.1f;
    }

    std::vector<Q8_1Block> blocks_res, blocks_ffn, blocks_out;
    quantize_to_blocks(residual, blocks_res);
    quantize_to_blocks(ffn, blocks_ffn);
    blocks_out.resize(blocks_res.size());

    simd::q8_1_add_q8_1(blocks_res.data(), blocks_ffn.data(), blocks_out.data(), d_model);

    std::vector<float> result;
    dequantize_blocks(blocks_out, result, d_model);

    auto expected = fp32_reference_add(residual, ffn);

    double cos_sim = cosine_similarity(result, expected);
    double rel_l2 = relative_l2_error(result, expected);
    double max_err = max_abs_error(result, expected);

    std::cout << "Transformer residual (cancellation):" << std::endl;
    std::cout << "  Cosine similarity: " << cos_sim << std::endl;
    std::cout << "  Relative L2 error: " << rel_l2 << std::endl;
    std::cout << "  Max absolute error: " << max_err << std::endl;

    // This is the problematic case - cosine may be low
    // We're not asserting here, just measuring
    std::cout << "  (Note: Cancellation case may have poor cosine similarity)" << std::endl;
}

TEST_F(Test__Q8_1_Add_Extreme, Layer21_SimulatedPattern)
{
    // Simulate the layer 21 pattern from the divergence test
    // FFN_DOWN has range [-1850, 112], Residual has range [-90, 1523]
    const size_t d_model = 896;

    std::vector<float> residual(d_model), ffn_down(d_model);
    std::mt19937 gen(2112);

    // Residual: range [-90, 1523] - mostly positive with some negative
    std::normal_distribution<float> dist_res(700.0f, 400.0f);
    for (size_t i = 0; i < d_model; ++i)
    {
        residual[i] = std::clamp(dist_res(gen), -90.0f, 1523.0f);
    }

    // FFN_DOWN: range [-1850, 112] - mostly negative
    std::normal_distribution<float> dist_ffn(-800.0f, 500.0f);
    for (size_t i = 0; i < d_model; ++i)
    {
        ffn_down[i] = std::clamp(dist_ffn(gen), -1850.0f, 112.0f);
    }

    std::vector<Q8_1Block> blocks_res, blocks_ffn, blocks_out;
    quantize_to_blocks(residual, blocks_res);
    quantize_to_blocks(ffn_down, blocks_ffn);
    blocks_out.resize(blocks_res.size());

    simd::q8_1_add_q8_1(blocks_res.data(), blocks_ffn.data(), blocks_out.data(), d_model);

    std::vector<float> result;
    dequantize_blocks(blocks_out, result, d_model);

    auto expected = fp32_reference_add(residual, ffn_down);

    double cos_sim = cosine_similarity(result, expected);
    double rel_l2 = relative_l2_error(result, expected);
    double max_err = max_abs_error(result, expected);

    std::cout << "Layer 21 simulated pattern:" << std::endl;
    std::cout << "  Input residual range: ["
              << *std::min_element(residual.begin(), residual.end()) << ", "
              << *std::max_element(residual.begin(), residual.end()) << "]" << std::endl;
    std::cout << "  Input FFN_DOWN range: ["
              << *std::min_element(ffn_down.begin(), ffn_down.end()) << ", "
              << *std::max_element(ffn_down.begin(), ffn_down.end()) << "]" << std::endl;
    std::cout << "  Expected result range: ["
              << *std::min_element(expected.begin(), expected.end()) << ", "
              << *std::max_element(expected.begin(), expected.end()) << "]" << std::endl;
    std::cout << "  Actual result range: ["
              << *std::min_element(result.begin(), result.end()) << ", "
              << *std::max_element(result.begin(), result.end()) << "]" << std::endl;
    std::cout << "  Cosine similarity: " << cos_sim << std::endl;
    std::cout << "  Relative L2 error: " << rel_l2 << std::endl;
    std::cout << "  Max absolute error: " << max_err << std::endl;

    EXPECT_GT(cos_sim, 0.95) << "Layer 21 pattern should have good cosine similarity";
}

// =============================================================================
// SUM_QS VERIFICATION
// Test that sum_qs field is correctly computed
// =============================================================================

TEST_F(Test__Q8_1_Add_Extreme, SumQs_Verification)
{
    Q8_1Block a, b, out;

    float fa[32], fb[32];
    for (int i = 0; i < 32; ++i)
    {
        fa[i] = static_cast<float>(i);
        fb[i] = static_cast<float>(i * 2);
    }

    simd::quantize_single_block_scalar(fa, a, 32);
    simd::quantize_single_block_scalar(fb, b, 32);

    simd::q8_1_add_q8_1(&a, &b, &out, 32);

    // Verify sum_qs matches actual sum of qs
    int32_t computed_sum = 0;
    for (int i = 0; i < 32; ++i)
    {
        computed_sum += out.qs[i];
    }

    EXPECT_EQ(out.sum_qs, static_cast<int16_t>(computed_sum))
        << "sum_qs field should match sum of qs array";
}

TEST_F(Test__Q8_1_Add_Extreme, SumQs_AfterCancellation)
{
    Q8_1Block a, b, out;

    float fa[32], fb[32];
    for (int i = 0; i < 32; ++i)
    {
        fa[i] = 10.0f;
        fb[i] = -10.0f;
    }

    simd::quantize_single_block_scalar(fa, a, 32);
    simd::quantize_single_block_scalar(fb, b, 32);

    simd::q8_1_add_q8_1(&a, &b, &out, 32);

    // After cancellation, sum_qs should be near zero
    int32_t computed_sum = 0;
    for (int i = 0; i < 32; ++i)
    {
        computed_sum += out.qs[i];
    }

    EXPECT_EQ(out.sum_qs, static_cast<int16_t>(computed_sum));
    EXPECT_NEAR(static_cast<int>(out.sum_qs), 0, 64); // Some tolerance for rounding
}

// =============================================================================
// STRESS TESTS
// =============================================================================

TEST_F(Test__Q8_1_Add_Extreme, StressTest_LargeBuffer)
{
    // Test with a large buffer (like a full transformer layer)
    const size_t num_elements = 8192; // 256 blocks

    std::vector<float> fa(num_elements), fb(num_elements);
    std::mt19937 gen(99999);
    std::normal_distribution<float> dist(0.0f, 20.0f);

    for (size_t i = 0; i < num_elements; ++i)
    {
        fa[i] = dist(gen);
        fb[i] = dist(gen);
    }

    std::vector<Q8_1Block> blocks_a, blocks_b, blocks_out;
    quantize_to_blocks(fa, blocks_a);
    quantize_to_blocks(fb, blocks_b);
    blocks_out.resize(blocks_a.size());

    simd::q8_1_add_q8_1(blocks_a.data(), blocks_b.data(), blocks_out.data(), num_elements);

    std::vector<float> result;
    dequantize_blocks(blocks_out, result, num_elements);

    auto expected = fp32_reference_add(fa, fb);

    double cos_sim = cosine_similarity(result, expected);

    EXPECT_GT(cos_sim, 0.98) << "Large buffer should maintain good accuracy";
}

TEST_F(Test__Q8_1_Add_Extreme, StressTest_RepeatedAdditions)
{
    // Simulate multiple residual additions (like going through layers)
    Q8_1Block hidden, delta, temp;

    // Initialize hidden with some values
    float f_hidden[32];
    for (int i = 0; i < 32; ++i)
        f_hidden[i] = 10.0f;
    simd::quantize_single_block_scalar(f_hidden, hidden, 32);

    // Simulate 24 layer additions
    std::mt19937 gen(11111);
    std::normal_distribution<float> dist(-5.0f, 5.0f);

    float expected_sum = 10.0f * 32; // Initial sum

    for (int layer = 0; layer < 24; ++layer)
    {
        float f_delta[32];
        for (int i = 0; i < 32; ++i)
        {
            f_delta[i] = dist(gen);
            expected_sum += f_delta[i];
        }

        simd::quantize_single_block_scalar(f_delta, delta, 32);
        simd::q8_1_add_q8_1(&hidden, &delta, &temp, 32);

        // Copy result back to hidden
        hidden = temp;
    }

    // Check final sum is reasonable
    float d = simd::fp16_to_fp32(hidden.d);
    float actual_sum = 0.0f;
    for (int i = 0; i < 32; ++i)
    {
        actual_sum += d * hidden.qs[i];
    }

    std::cout << "After 24 additions: expected_sum=" << expected_sum
              << " actual_sum=" << actual_sum << std::endl;

    // Due to accumulated quantization error, allow large tolerance
    EXPECT_NEAR(actual_sum, expected_sum, std::abs(expected_sum) * 0.5f);
}
