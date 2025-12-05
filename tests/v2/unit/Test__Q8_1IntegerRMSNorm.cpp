/**
 * @file Test__Q8_1IntegerRMSNorm.cpp
 * @brief Unit tests for Q8_1 integer-space RMSNorm primitive
 *
 * Tests the correctness of computing RMSNorm mostly in integer space
 * for Q8_1 quantized tensors. Compares against FP32 reference.
 *
 * @author David Sanftenberg
 * @date 2025-12-04
 */

#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cstring>

#include "v2/kernels/cpu/primitives/RMSNormPrimitives.h"
#include "v2/tensors/BlockStructures.h"
#include "v2/tensors/SIMDHelpers.h"

using namespace llaminar2;
using namespace llaminar2::primitives;

class Q8_1IntegerRMSNormTest : public ::testing::Test
{
protected:
    std::mt19937 rng_{42};

    // Helper: FP16 <-> FP32 conversion
    static float fp16_to_fp32(uint16_t h)
    {
#if defined(__F16C__)
        return _cvtsh_ss(h);
#else
        uint32_t sign = (h & 0x8000U) << 16;
        uint32_t exp = (h & 0x7C00U) >> 10;
        uint32_t mantissa = h & 0x03FFU;
        uint32_t fp32_bits;
        if (exp == 0)
        {
            if (mantissa == 0)
                fp32_bits = sign;
            else
            {
                exp = 1;
                while ((mantissa & 0x0400U) == 0)
                {
                    mantissa <<= 1;
                    exp--;
                }
                mantissa &= 0x03FFU;
                fp32_bits = sign | ((exp + (127 - 15)) << 23) | (mantissa << 13);
            }
        }
        else if (exp == 0x1F)
        {
            fp32_bits = sign | 0x7F800000U | (mantissa << 13);
        }
        else
        {
            fp32_bits = sign | ((exp + (127 - 15)) << 23) | (mantissa << 13);
        }
        float result;
        std::memcpy(&result, &fp32_bits, sizeof(float));
        return result;
#endif
    }

    static uint16_t fp32_to_fp16(float f)
    {
#if defined(__F16C__)
        return _cvtss_sh(f, _MM_FROUND_TO_NEAREST_INT);
#else
        uint32_t bits;
        std::memcpy(&bits, &f, sizeof(float));
        uint32_t sign = (bits >> 16) & 0x8000U;
        int32_t exp = ((bits >> 23) & 0xFFU) - 127 + 15;
        uint32_t mantissa = (bits >> 13) & 0x03FFU;
        if (exp <= 0)
        {
            if (exp < -10)
                return sign;
            mantissa |= 0x0400U;
            mantissa >>= (1 - exp);
            return sign | mantissa;
        }
        else if (exp >= 0x1F)
        {
            return sign | 0x7C00U;
        }
        else
        {
            return sign | (exp << 10) | mantissa;
        }
#endif
    }

    // Helper: Create Q8_1 blocks from FP32 data
    void quantize_to_q8_1(const float *fp32, Q8_1Block *blocks, size_t n_elements)
    {
        size_t n_blocks = n_elements / 32;
        for (size_t b = 0; b < n_blocks; ++b)
        {
            const float *block_data = fp32 + b * 32;

            // Find max absolute value
            float max_abs = 0.0f;
            for (int i = 0; i < 32; ++i)
            {
                max_abs = std::max(max_abs, std::abs(block_data[i]));
            }

            // Compute scale
            float d = max_abs / 127.0f;
            float inv_d = (d > 0.0f) ? 127.0f / max_abs : 0.0f;
            blocks[b].d = fp32_to_fp16(d);

            // Quantize
            int32_t sum_qs = 0;
            for (int i = 0; i < 32; ++i)
            {
                int32_t q = static_cast<int32_t>(std::round(block_data[i] * inv_d));
                q = std::max(-127, std::min(127, q));
                blocks[b].qs[i] = static_cast<int8_t>(q);
                sum_qs += blocks[b].qs[i];
            }
            blocks[b].sum_qs = static_cast<int16_t>(sum_qs);
        }
    }

    // Helper: Dequantize Q8_1 blocks to FP32
    void dequantize_q8_1(const Q8_1Block *blocks, float *fp32, size_t n_elements)
    {
        size_t n_blocks = n_elements / 32;
        for (size_t b = 0; b < n_blocks; ++b)
        {
            float d = fp16_to_fp32(blocks[b].d);
            for (int i = 0; i < 32; ++i)
            {
                fp32[b * 32 + i] = d * static_cast<float>(blocks[b].qs[i]);
            }
        }
    }

    // Reference FP32 RMSNorm
    void rmsnorm_fp32_reference(const float *input, const float *gamma, float *output,
                                size_t n_elements, float epsilon)
    {
        // Compute sum of squares
        double sum_sq = 0.0;
        for (size_t i = 0; i < n_elements; ++i)
        {
            sum_sq += static_cast<double>(input[i]) * static_cast<double>(input[i]);
        }

        // Compute inverse RMS
        float inv_rms = 1.0f / std::sqrt(static_cast<float>(sum_sq / n_elements) + epsilon);

        // Apply normalization
        for (size_t i = 0; i < n_elements; ++i)
        {
            output[i] = input[i] * inv_rms * gamma[i];
        }
    }

    // Compare two Q8_1 block arrays
    struct ComparisonResult
    {
        float max_relative_error;
        float mean_relative_error;
        float cosine_similarity;
        bool scale_match;
    };

    ComparisonResult compare_q8_1_outputs(const Q8_1Block *a, const Q8_1Block *b,
                                          size_t blocks_per_row)
    {
        std::vector<float> fp32_a(blocks_per_row * 32);
        std::vector<float> fp32_b(blocks_per_row * 32);

        dequantize_q8_1(a, fp32_a.data(), blocks_per_row * 32);
        dequantize_q8_1(b, fp32_b.data(), blocks_per_row * 32);

        float max_rel_err = 0.0f;
        float sum_rel_err = 0.0f;
        double dot_ab = 0.0, dot_aa = 0.0, dot_bb = 0.0;

        for (size_t i = 0; i < fp32_a.size(); ++i)
        {
            float va = fp32_a[i];
            float vb = fp32_b[i];

            // Relative error
            float abs_max = std::max(std::abs(va), std::abs(vb));
            if (abs_max > 1e-6f)
            {
                float rel_err = std::abs(va - vb) / abs_max;
                max_rel_err = std::max(max_rel_err, rel_err);
                sum_rel_err += rel_err;
            }

            // Cosine similarity
            dot_ab += va * vb;
            dot_aa += va * va;
            dot_bb += vb * vb;
        }

        float cosine = static_cast<float>(dot_ab / (std::sqrt(dot_aa) * std::sqrt(dot_bb)));
        float mean_rel_err = sum_rel_err / fp32_a.size();

        // Check if scales are similar
        bool scale_match = true;
        for (size_t blk = 0; blk < blocks_per_row; ++blk)
        {
            float da = fp16_to_fp32(a[blk].d);
            float db = fp16_to_fp32(b[blk].d);
            float scale_err = std::abs(da - db) / std::max(std::abs(da), std::abs(db));
            if (scale_err > 0.1f)
                scale_match = false;
        }

        return {max_rel_err, mean_rel_err, cosine, scale_match};
    }
};

// ============================================================================
// Basic Correctness Tests
// ============================================================================

TEST_F(Q8_1IntegerRMSNormTest, BasicCorrectness_SingleBlock)
{
    const size_t cols = 32;
    const size_t blocks_per_row = 1;
    const float epsilon = 1e-6f;

    // Create test data
    std::vector<float> fp32_input(cols);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &v : fp32_input)
        v = dist(rng_);

    std::vector<float> gamma(cols, 1.0f);

    // Quantize input to Q8_1
    std::vector<Q8_1Block> q8_input(blocks_per_row);
    quantize_to_q8_1(fp32_input.data(), q8_input.data(), cols);

    // Run integer-space RMSNorm
    std::vector<Q8_1Block> q8_output_int(blocks_per_row);
    rmsnorm_q8_1_integer_row(q8_input.data(), gamma.data(), q8_output_int.data(),
                             blocks_per_row, epsilon);

    // Run reference: dequant -> FP32 RMSNorm -> requant
    std::vector<float> fp32_dequant(cols);
    dequantize_q8_1(q8_input.data(), fp32_dequant.data(), cols);

    std::vector<float> fp32_output(cols);
    rmsnorm_fp32_reference(fp32_dequant.data(), gamma.data(), fp32_output.data(), cols, epsilon);

    std::vector<Q8_1Block> q8_output_ref(blocks_per_row);
    quantize_to_q8_1(fp32_output.data(), q8_output_ref.data(), cols);

    // Compare
    auto result = compare_q8_1_outputs(q8_output_int.data(), q8_output_ref.data(), blocks_per_row);

    std::cout << "SingleBlock: max_rel_err=" << result.max_relative_error
              << " mean_rel_err=" << result.mean_relative_error
              << " cosine=" << result.cosine_similarity << std::endl;

    // Q8_1 has inherent quantization error, so we allow some tolerance
    EXPECT_LT(result.max_relative_error, 0.15f) << "Max relative error too high";
    EXPECT_GT(result.cosine_similarity, 0.99f) << "Cosine similarity too low";
}

TEST_F(Q8_1IntegerRMSNormTest, BasicCorrectness_MultiBlock)
{
    const size_t cols = 896; // Qwen 0.5B d_model
    const size_t blocks_per_row = cols / 32;
    const float epsilon = 1e-6f;

    // Create test data
    std::vector<float> fp32_input(cols);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &v : fp32_input)
        v = dist(rng_);

    std::vector<float> gamma(cols, 1.0f);

    // Quantize input to Q8_1
    std::vector<Q8_1Block> q8_input(blocks_per_row);
    quantize_to_q8_1(fp32_input.data(), q8_input.data(), cols);

    // Run integer-space RMSNorm
    std::vector<Q8_1Block> q8_output_int(blocks_per_row);
    rmsnorm_q8_1_integer_row(q8_input.data(), gamma.data(), q8_output_int.data(),
                             blocks_per_row, epsilon);

    // Run reference
    std::vector<float> fp32_dequant(cols);
    dequantize_q8_1(q8_input.data(), fp32_dequant.data(), cols);

    std::vector<float> fp32_output(cols);
    rmsnorm_fp32_reference(fp32_dequant.data(), gamma.data(), fp32_output.data(), cols, epsilon);

    std::vector<Q8_1Block> q8_output_ref(blocks_per_row);
    quantize_to_q8_1(fp32_output.data(), q8_output_ref.data(), cols);

    // Compare
    auto result = compare_q8_1_outputs(q8_output_int.data(), q8_output_ref.data(), blocks_per_row);

    std::cout << "MultiBlock (d_model=896): max_rel_err=" << result.max_relative_error
              << " mean_rel_err=" << result.mean_relative_error
              << " cosine=" << result.cosine_similarity << std::endl;

    EXPECT_LT(result.max_relative_error, 0.15f);
    EXPECT_GT(result.cosine_similarity, 0.99f);
}

TEST_F(Q8_1IntegerRMSNormTest, BasicCorrectness_Qwen7B_DModel)
{
    const size_t cols = 3584; // Qwen 7B d_model
    const size_t blocks_per_row = cols / 32;
    const float epsilon = 1e-6f;

    // Create test data
    std::vector<float> fp32_input(cols);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &v : fp32_input)
        v = dist(rng_);

    std::vector<float> gamma(cols);
    for (auto &v : gamma)
        v = dist(rng_) * 0.5f + 0.75f; // gamma in [0.25, 1.25]

    // Quantize input to Q8_1
    std::vector<Q8_1Block> q8_input(blocks_per_row);
    quantize_to_q8_1(fp32_input.data(), q8_input.data(), cols);

    // Run integer-space RMSNorm
    std::vector<Q8_1Block> q8_output_int(blocks_per_row);
    rmsnorm_q8_1_integer_row(q8_input.data(), gamma.data(), q8_output_int.data(),
                             blocks_per_row, epsilon);

    // Run reference
    std::vector<float> fp32_dequant(cols);
    dequantize_q8_1(q8_input.data(), fp32_dequant.data(), cols);

    std::vector<float> fp32_output(cols);
    rmsnorm_fp32_reference(fp32_dequant.data(), gamma.data(), fp32_output.data(), cols, epsilon);

    std::vector<Q8_1Block> q8_output_ref(blocks_per_row);
    quantize_to_q8_1(fp32_output.data(), q8_output_ref.data(), cols);

    // Compare
    auto result = compare_q8_1_outputs(q8_output_int.data(), q8_output_ref.data(), blocks_per_row);

    std::cout << "Qwen7B (d_model=3584): max_rel_err=" << result.max_relative_error
              << " mean_rel_err=" << result.mean_relative_error
              << " cosine=" << result.cosine_similarity << std::endl;

    EXPECT_LT(result.max_relative_error, 0.15f);
    EXPECT_GT(result.cosine_similarity, 0.99f);
}

TEST_F(Q8_1IntegerRMSNormTest, MultiRow_Parallel)
{
    const size_t rows = 128;
    const size_t cols = 3584;
    const size_t blocks_per_row = cols / 32;
    const float epsilon = 1e-6f;

    // Create test data
    std::vector<float> fp32_input(rows * cols);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &v : fp32_input)
        v = dist(rng_);

    std::vector<float> gamma(cols, 1.0f);

    // Quantize all rows
    std::vector<Q8_1Block> q8_input(rows * blocks_per_row);
    for (size_t r = 0; r < rows; ++r)
    {
        quantize_to_q8_1(fp32_input.data() + r * cols,
                         q8_input.data() + r * blocks_per_row, cols);
    }

    // Run integer-space RMSNorm (parallel)
    std::vector<Q8_1Block> q8_output_int(rows * blocks_per_row);
    rmsnorm_q8_1_integer(q8_input.data(), gamma.data(), q8_output_int.data(),
                         rows, blocks_per_row, epsilon);

    // Run reference for each row
    std::vector<Q8_1Block> q8_output_ref(rows * blocks_per_row);
    for (size_t r = 0; r < rows; ++r)
    {
        std::vector<float> fp32_dequant(cols);
        dequantize_q8_1(q8_input.data() + r * blocks_per_row, fp32_dequant.data(), cols);

        std::vector<float> fp32_output(cols);
        rmsnorm_fp32_reference(fp32_dequant.data(), gamma.data(), fp32_output.data(), cols, epsilon);

        quantize_to_q8_1(fp32_output.data(), q8_output_ref.data() + r * blocks_per_row, cols);
    }

    // Compare each row
    float total_max_err = 0.0f;
    float total_cosine = 0.0f;
    for (size_t r = 0; r < rows; ++r)
    {
        auto result = compare_q8_1_outputs(
            q8_output_int.data() + r * blocks_per_row,
            q8_output_ref.data() + r * blocks_per_row,
            blocks_per_row);
        total_max_err = std::max(total_max_err, result.max_relative_error);
        total_cosine += result.cosine_similarity;
    }

    float avg_cosine = total_cosine / rows;
    std::cout << "MultiRow (128x3584): max_rel_err=" << total_max_err
              << " avg_cosine=" << avg_cosine << std::endl;

    EXPECT_LT(total_max_err, 0.15f);
    EXPECT_GT(avg_cosine, 0.99f);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(Q8_1IntegerRMSNormTest, ZeroInput)
{
    const size_t cols = 128;
    const size_t blocks_per_row = cols / 32;
    const float epsilon = 1e-6f;

    // Zero input
    std::vector<float> fp32_input(cols, 0.0f);
    std::vector<float> gamma(cols, 1.0f);

    std::vector<Q8_1Block> q8_input(blocks_per_row);
    quantize_to_q8_1(fp32_input.data(), q8_input.data(), cols);

    std::vector<Q8_1Block> q8_output(blocks_per_row);
    rmsnorm_q8_1_integer_row(q8_input.data(), gamma.data(), q8_output.data(),
                             blocks_per_row, epsilon);

    // Output should be all zeros (or near-zero due to epsilon)
    std::vector<float> fp32_output(cols);
    dequantize_q8_1(q8_output.data(), fp32_output.data(), cols);

    float max_abs = 0.0f;
    for (float v : fp32_output)
    {
        max_abs = std::max(max_abs, std::abs(v));
    }

    // With epsilon, output should be small but not exactly zero
    EXPECT_LT(max_abs, 0.01f) << "Output should be near-zero for zero input";
}

TEST_F(Q8_1IntegerRMSNormTest, UniformInput)
{
    const size_t cols = 256;
    const size_t blocks_per_row = cols / 32;
    const float epsilon = 1e-6f;

    // Uniform input (all same value)
    std::vector<float> fp32_input(cols, 0.5f);
    std::vector<float> gamma(cols, 1.0f);

    std::vector<Q8_1Block> q8_input(blocks_per_row);
    quantize_to_q8_1(fp32_input.data(), q8_input.data(), cols);

    std::vector<Q8_1Block> q8_output(blocks_per_row);
    rmsnorm_q8_1_integer_row(q8_input.data(), gamma.data(), q8_output.data(),
                             blocks_per_row, epsilon);

    // Compare with reference
    std::vector<float> fp32_dequant(cols);
    dequantize_q8_1(q8_input.data(), fp32_dequant.data(), cols);

    std::vector<float> fp32_ref_output(cols);
    rmsnorm_fp32_reference(fp32_dequant.data(), gamma.data(), fp32_ref_output.data(), cols, epsilon);

    std::vector<Q8_1Block> q8_output_ref(blocks_per_row);
    quantize_to_q8_1(fp32_ref_output.data(), q8_output_ref.data(), cols);

    auto result = compare_q8_1_outputs(q8_output.data(), q8_output_ref.data(), blocks_per_row);

    std::cout << "UniformInput: cosine=" << result.cosine_similarity << std::endl;
    EXPECT_GT(result.cosine_similarity, 0.99f);
}

TEST_F(Q8_1IntegerRMSNormTest, LargeValues)
{
    const size_t cols = 128;
    const size_t blocks_per_row = cols / 32;
    const float epsilon = 1e-6f;

    // Large values that might stress quantization
    std::vector<float> fp32_input(cols);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (auto &v : fp32_input)
        v = dist(rng_);

    std::vector<float> gamma(cols, 1.0f);

    std::vector<Q8_1Block> q8_input(blocks_per_row);
    quantize_to_q8_1(fp32_input.data(), q8_input.data(), cols);

    std::vector<Q8_1Block> q8_output(blocks_per_row);
    rmsnorm_q8_1_integer_row(q8_input.data(), gamma.data(), q8_output.data(),
                             blocks_per_row, epsilon);

    // Reference
    std::vector<float> fp32_dequant(cols);
    dequantize_q8_1(q8_input.data(), fp32_dequant.data(), cols);

    std::vector<float> fp32_ref_output(cols);
    rmsnorm_fp32_reference(fp32_dequant.data(), gamma.data(), fp32_ref_output.data(), cols, epsilon);

    std::vector<Q8_1Block> q8_output_ref(blocks_per_row);
    quantize_to_q8_1(fp32_ref_output.data(), q8_output_ref.data(), cols);

    auto result = compare_q8_1_outputs(q8_output.data(), q8_output_ref.data(), blocks_per_row);

    std::cout << "LargeValues: cosine=" << result.cosine_similarity << std::endl;
    EXPECT_GT(result.cosine_similarity, 0.99f);
}

// =============================================================================
// Performance Comparison Tests
// =============================================================================

TEST_F(Q8_1IntegerRMSNormTest, PerformanceComparison_SingleRow)
{
    const size_t cols = 896; // Qwen 0.5B d_model
    const size_t blocks_per_row = cols / 32;
    const float epsilon = 1e-6f;
    const int warmup = 100;
    const int iterations = 10000;

    // Prepare input
    std::vector<float> fp32_input(cols);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &v : fp32_input)
        v = dist(rng_);

    std::vector<float> gamma(cols);
    for (auto &v : gamma)
        v = 0.9f + dist(rng_) * 0.2f;

    std::vector<Q8_1Block> q8_input(blocks_per_row);
    quantize_to_q8_1(fp32_input.data(), q8_input.data(), cols);

    std::vector<Q8_1Block> q8_output(blocks_per_row);
    alignas(64) std::vector<float> fp32_dequant(cols);
    alignas(64) std::vector<float> fp32_output(cols);
    std::vector<Q8_1Block> q8_output_simd(blocks_per_row);

    // Warmup Integer version
    for (int i = 0; i < warmup; ++i)
    {
        rmsnorm_q8_1_integer_row(q8_input.data(), gamma.data(), q8_output.data(),
                                 blocks_per_row, epsilon);
    }

    // Benchmark Integer version
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i)
    {
        rmsnorm_q8_1_integer_row(q8_input.data(), gamma.data(), q8_output.data(),
                                 blocks_per_row, epsilon);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double int_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iterations;

    // Warmup SIMD version (actual implementation from typed kernel)
    // dequant_q8_1_to_fp32 -> rmsnorm_fused_row_avx512 -> quantize_fp32_to_q8_1_blocks
    for (int i = 0; i < warmup; ++i)
    {
        simd::dequantize_q8_1_to_fp32(q8_input.data(), fp32_dequant.data(), cols);
        rmsnorm_fused_row_avx512(fp32_dequant.data(), gamma.data(), fp32_output.data(), cols, epsilon);
        simd::quantize_fp32_to_q8_1_blocks(fp32_output.data(), q8_output_simd.data(), cols);
    }

    // Benchmark SIMD version
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i)
    {
        simd::dequantize_q8_1_to_fp32(q8_input.data(), fp32_dequant.data(), cols);
        rmsnorm_fused_row_avx512(fp32_dequant.data(), gamma.data(), fp32_output.data(), cols, epsilon);
        simd::quantize_fp32_to_q8_1_blocks(fp32_output.data(), q8_output_simd.data(), cols);
    }
    t1 = std::chrono::high_resolution_clock::now();
    double simd_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iterations;

    double speedup = simd_us / int_us;
    std::cout << "\n=== Performance vs SIMD (Single Row, d_model=" << cols << ") ===" << std::endl;
    std::cout << "Integer-Space Q8_1: " << int_us << " us/row" << std::endl;
    std::cout << "SIMD (dequant->norm->quant): " << simd_us << " us/row" << std::endl;
    std::cout << "Speedup: " << speedup << "x" << std::endl;

    // Verify results match
    auto result = compare_q8_1_outputs(q8_output.data(), q8_output_simd.data(), blocks_per_row);
    std::cout << "Cosine similarity: " << result.cosine_similarity << std::endl;
    EXPECT_GT(result.cosine_similarity, 0.99f);
}

TEST_F(Q8_1IntegerRMSNormTest, PerformanceComparison_Batch)
{
    const size_t rows = 128;
    const size_t cols = 896; // Qwen 0.5B d_model
    const size_t blocks_per_row = cols / 32;
    const float epsilon = 1e-6f;
    const int warmup = 10;
    const int iterations = 1000;

    // Prepare input
    std::vector<float> fp32_input(rows * cols);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &v : fp32_input)
        v = dist(rng_);

    std::vector<float> gamma(cols);
    for (auto &v : gamma)
        v = 0.9f + dist(rng_) * 0.2f;

    std::vector<Q8_1Block> q8_input(rows * blocks_per_row);
    for (size_t r = 0; r < rows; ++r)
    {
        quantize_to_q8_1(fp32_input.data() + r * cols,
                         q8_input.data() + r * blocks_per_row, cols);
    }

    std::vector<Q8_1Block> q8_output(rows * blocks_per_row);
    std::vector<float> fp32_dequant(rows * cols);
    std::vector<float> fp32_output(rows * cols);
    std::vector<Q8_1Block> q8_output_fp32(rows * blocks_per_row);

    RMSNormExecOptions opts;
    opts.allow_parallel = true;
    opts.parallel_threshold_elems = 0; // Always parallel

    // Warmup Integer version
    for (int i = 0; i < warmup; ++i)
    {
        rmsnorm_q8_1_integer(q8_input.data(), gamma.data(), q8_output.data(),
                             rows, blocks_per_row, epsilon, opts);
    }

    // Benchmark Integer version
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i)
    {
        rmsnorm_q8_1_integer(q8_input.data(), gamma.data(), q8_output.data(),
                             rows, blocks_per_row, epsilon, opts);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double int_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iterations;

    // Warmup FP32 version (dequant -> rmsnorm -> quant)
    for (int i = 0; i < warmup; ++i)
    {
        for (size_t r = 0; r < rows; ++r)
        {
            dequantize_q8_1(q8_input.data() + r * blocks_per_row,
                            fp32_dequant.data() + r * cols, cols);
            rmsnorm_fp32_reference(fp32_dequant.data() + r * cols,
                                   gamma.data(),
                                   fp32_output.data() + r * cols, cols, epsilon);
            quantize_to_q8_1(fp32_output.data() + r * cols,
                             q8_output_fp32.data() + r * blocks_per_row, cols);
        }
    }

    // Benchmark FP32 version
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i)
    {
        for (size_t r = 0; r < rows; ++r)
        {
            dequantize_q8_1(q8_input.data() + r * blocks_per_row,
                            fp32_dequant.data() + r * cols, cols);
            rmsnorm_fp32_reference(fp32_dequant.data() + r * cols,
                                   gamma.data(),
                                   fp32_output.data() + r * cols, cols, epsilon);
            quantize_to_q8_1(fp32_output.data() + r * cols,
                             q8_output_fp32.data() + r * blocks_per_row, cols);
        }
    }
    t1 = std::chrono::high_resolution_clock::now();
    double fp32_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iterations;

    double speedup = fp32_ms / int_ms;
    std::cout << "\n=== Performance Comparison (Batch: " << rows << " rows, d_model=" << cols << ") ===" << std::endl;
    std::cout << "Integer-Space Q8_1: " << int_ms << " ms/batch" << std::endl;
    std::cout << "FP32 (dequant->norm->quant): " << fp32_ms << " ms/batch" << std::endl;
    std::cout << "Speedup: " << speedup << "x" << std::endl;

    // Note: We don't validate here since the FP32 version doesn't parallelize
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
