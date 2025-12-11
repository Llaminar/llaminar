/**
 * @file Test__Q8_1_Add_Debug.cpp
 * @brief Debug test to investigate Q8_1 add behavior with actual layer 21 data patterns
 */

#include <gtest/gtest.h>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <algorithm>

// Include the Q8_1 block definition and SIMD helpers
#include "v2/tensors/SIMDHelpers.h"
#include "v2/tensors/Tensors.h"

using namespace llaminar2;

class Test__Q8_1_Add_Debug : public ::testing::Test
{
protected:
    static double cosine_similarity(const float *a, const float *b, size_t n)
    {
        double dot = 0, norm_a = 0, norm_b = 0;
        for (size_t i = 0; i < n; ++i)
        {
            dot += static_cast<double>(a[i]) * b[i];
            norm_a += static_cast<double>(a[i]) * a[i];
            norm_b += static_cast<double>(b[i]) * b[i];
        }
        return dot / (std::sqrt(norm_a) * std::sqrt(norm_b) + 1e-10);
    }

    // Simulate exact layer 21 pattern
    static void generate_layer21_pattern(
        std::vector<float> &attn_res,
        std::vector<float> &ffn_down,
        size_t n,
        std::mt19937 &rng)
    {
        // Layer 21 characteristics from PyTorch reference:
        // attn_res: mean=0.31, std=17.3, range=[-91, 1530]
        // ffn_down: mean=-0.21, std=15.9, range=[-1409, 85]
        // result: mean=0.10, std=2.4, range=[-7.6, 121]

        std::normal_distribution<float> attn_dist(0.3f, 17.0f);
        std::normal_distribution<float> noise_dist(0.0f, 2.0f);

        for (size_t i = 0; i < n; ++i)
        {
            // Generate attn_res with occasional large spikes
            attn_res[i] = attn_dist(rng);

            // Clip to realistic range with occasional large values
            if (rng() % 1000 < 5)
            {
                // 0.5% chance of large positive spike
                attn_res[i] = 1000.0f + (rng() % 530);
            }
            attn_res[i] = std::max(-91.0f, std::min(1530.0f, attn_res[i]));

            // Generate ffn_down that mostly cancels attn_res
            // This is the key pattern in layer 21
            ffn_down[i] = -attn_res[i] + noise_dist(rng);
            ffn_down[i] = std::max(-1409.0f, std::min(85.0f, ffn_down[i]));
        }
    }
};

TEST_F(Test__Q8_1_Add_Debug, Layer21Pattern_BlockByBlock)
{
    const size_t n_elements = 9 * 896; // layer 21 dimensions
    const size_t n_blocks = (n_elements + 31) / 32;

    std::vector<float> attn_res(n_elements);
    std::vector<float> ffn_down(n_elements);

    std::mt19937 rng(42);
    generate_layer21_pattern(attn_res, ffn_down, n_elements, rng);

    // Compute expected result
    std::vector<float> expected(n_elements);
    for (size_t i = 0; i < n_elements; ++i)
    {
        expected[i] = attn_res[i] + ffn_down[i];
    }

    // Quantize to Q8_1
    std::vector<Q8_1Block> attn_blocks(n_blocks);
    std::vector<Q8_1Block> ffn_blocks(n_blocks);
    std::vector<Q8_1Block> out_blocks(n_blocks);

    simd::quantize_fp32_to_q8_1_blocks(attn_res.data(), attn_blocks.data(), n_elements);
    simd::quantize_fp32_to_q8_1_blocks(ffn_down.data(), ffn_blocks.data(), n_elements);

    // Get FP32 views via dequantization
    // NOTE: Q8_1Block.d is uint16_t containing FP16 representation - must convert!
    std::vector<float> attn_dequant(n_elements);
    std::vector<float> ffn_dequant(n_elements);

    for (size_t b = 0; b < n_blocks; ++b)
    {
        const Q8_1Block &blk_a = attn_blocks[b];
        const Q8_1Block &blk_b = ffn_blocks[b];
        float scale_a = simd::fp16_to_fp32(blk_a.d);
        float scale_b = simd::fp16_to_fp32(blk_b.d);
        size_t start = b * 32;
        for (size_t i = 0; i < 32 && start + i < n_elements; ++i)
        {
            attn_dequant[start + i] = blk_a.qs[i] * scale_a;
            ffn_dequant[start + i] = blk_b.qs[i] * scale_b;
        }
    }

    double cos_attn = cosine_similarity(attn_res.data(), attn_dequant.data(), n_elements);
    double cos_ffn = cosine_similarity(ffn_down.data(), ffn_dequant.data(), n_elements);

    std::cout << "Input quantization quality:\n";
    std::cout << "  attn_res Q8_1 cosine: " << std::fixed << std::setprecision(6) << cos_attn << "\n";
    std::cout << "  ffn_down Q8_1 cosine: " << cos_ffn << "\n";

    EXPECT_GE(cos_attn, 0.999) << "attn_res quantization too lossy";
    EXPECT_GE(cos_ffn, 0.999) << "ffn_down quantization too lossy";

    // Perform Q8_1 addition
    simd::q8_1_add_q8_1(attn_blocks.data(), ffn_blocks.data(), out_blocks.data(), n_elements);

    // Dequantize result
    std::vector<float> result_dequant(n_elements);
    for (size_t b = 0; b < n_blocks; ++b)
    {
        const Q8_1Block &blk = out_blocks[b];
        float scale_out = simd::fp16_to_fp32(blk.d);
        size_t start = b * 32;
        for (size_t i = 0; i < 32 && start + i < n_elements; ++i)
        {
            result_dequant[start + i] = blk.qs[i] * scale_out;
        }
    }

    // Compare to expected
    double cos_result = cosine_similarity(expected.data(), result_dequant.data(), n_elements);

    // Also check: dequant inputs, add in FP32 (this is the BEST we can do)
    std::vector<float> naive_result(n_elements);
    for (size_t i = 0; i < n_elements; ++i)
    {
        naive_result[i] = attn_dequant[i] + ffn_dequant[i];
    }
    double cos_naive = cosine_similarity(expected.data(), naive_result.data(), n_elements);

    std::cout << "\nResult quality:\n";
    std::cout << "  Q8_1 add result cosine:   " << cos_result << "\n";
    std::cout << "  Naive (dequant+FP32 add): " << cos_naive << "\n";

    // Value range check
    float exp_min = *std::min_element(expected.begin(), expected.end());
    float exp_max = *std::max_element(expected.begin(), expected.end());
    float res_min = *std::min_element(result_dequant.begin(), result_dequant.end());
    float res_max = *std::max_element(result_dequant.begin(), result_dequant.end());

    std::cout << "\nValue ranges:\n";
    std::cout << "  Expected: [" << exp_min << ", " << exp_max << "]\n";
    std::cout << "  Q8_1 add: [" << res_min << ", " << res_max << "]\n";

    // The result should be close to naive (both suffer from input quantization)
    // Result should NOT be much worse than naive
    EXPECT_GE(cos_result, cos_naive - 0.01)
        << "Q8_1 add is significantly worse than naive dequant+add";

    // With cancellation, we expect cosine ~ 0.93-0.99 (not great but not catastrophic)
    EXPECT_GE(cos_result, 0.90)
        << "Catastrophic divergence in Q8_1 add";
}

TEST_F(Test__Q8_1_Add_Debug, InspectSingleBlock_WorstCase)
{
    // Create a single block with severe cancellation
    std::vector<float> a(32), b(32);

    // Block where a and b have large opposite values
    for (int i = 0; i < 32; ++i)
    {
        a[i] = 1000.0f + i * 10.0f; // Range: [1000, 1310]
        b[i] = -990.0f - i * 10.0f; // Range: [-990, -1300]
        // Result should be: [10, 10] (constant cancellation residue)
    }

    std::cout << "Input values:\n";
    std::cout << "  a[0]=" << a[0] << " a[31]=" << a[31] << "\n";
    std::cout << "  b[0]=" << b[0] << " b[31]=" << b[31] << "\n";

    Q8_1Block blk_a, blk_b, blk_out;
    simd::quantize_fp32_to_q8_1_blocks(a.data(), &blk_a, 32);
    simd::quantize_fp32_to_q8_1_blocks(b.data(), &blk_b, 32);

    // NOTE: blk_a.d is uint16_t containing FP16 representation, NOT float!
    // Must use fp16_to_fp32 to get actual scale value
    float scale_a = simd::fp16_to_fp32(blk_a.d);
    float scale_b = simd::fp16_to_fp32(blk_b.d);

    std::cout << "\nSingle block worst-case:\n";
    std::cout << "  a: scale=" << scale_a << " qs[0]=" << (int)blk_a.qs[0] << " qs[31]=" << (int)blk_a.qs[31] << "\n";
    std::cout << "  b: scale=" << scale_b << " qs[0]=" << (int)blk_b.qs[0] << " qs[31]=" << (int)blk_b.qs[31] << "\n";

    // Perform add
    simd::q8_1_add_q8_1(&blk_a, &blk_b, &blk_out, 32);

    float scale_out = simd::fp16_to_fp32(blk_out.d);
    std::cout << "  out: scale=" << scale_out << " qs[0]=" << (int)blk_out.qs[0] << " qs[31]=" << (int)blk_out.qs[31] << "\n";

    // Dequant and check
    float a_dequant[32], b_dequant[32], out_dequant[32];
    for (int i = 0; i < 32; ++i)
    {
        a_dequant[i] = blk_a.qs[i] * scale_a;
        b_dequant[i] = blk_b.qs[i] * scale_b;
        out_dequant[i] = blk_out.qs[i] * scale_out;
    }

    std::cout << "\n  Dequantized values [0..4]:\n";
    for (int i = 0; i < 5; ++i)
    {
        float expected = a[i] + b[i];
        float naive = a_dequant[i] + b_dequant[i];
        std::cout << "    [" << i << "]: expected=" << expected
                  << " naive=" << naive
                  << " q8_1=" << out_dequant[i]
                  << " err=" << std::abs(out_dequant[i] - expected) << "\n";
    }

    // Calculate overall error
    float total_err = 0;
    float max_err = 0;
    for (int i = 0; i < 32; ++i)
    {
        float expected = a[i] + b[i];
        float err = std::abs(out_dequant[i] - expected);
        total_err += err;
        max_err = std::max(max_err, err);
    }

    std::cout << "\n  Error stats: mean=" << total_err / 32 << " max=" << max_err << "\n";

    // The max error should be bounded by quantization error of both inputs
    // Scale of a is ~1310/127 ≈ 10.3, scale of b is ~1300/127 ≈ 10.2
    // Max quantization error per input ≈ scale/2 ≈ 5.1
    // Max error in sum ≈ 10.3 (both errors can compound)
    EXPECT_LE(max_err, 25.0f) << "Error exceeds expected quantization bound";
}
