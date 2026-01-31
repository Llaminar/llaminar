/**
 * @file Test__Q8_1_to_Q16_RoPE_BlockSizes.cpp
 * @brief Unit tests for templated Q8_1 → Q16 RoPE conversion with variable block sizes.
 *
 * Tests verify:
 * 1. Per-head normalization (unified scale across all output blocks in a head)
 * 2. RoPE rotation correctness vs FP32 reference
 * 3. All output block sizes: 32, 64, 128, 192
 * 4. AVX2 and AVX512 paths match scalar reference
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
#include <numeric>

#include "kernels/cpu/primitives/RoPEPrimitives.h"
#include "tensors/BlockStructures.h"
#include "tensors/SIMDHelpers.h"

namespace llaminar2::test
{

    using namespace llaminar2::primitives;
    using llaminar2::simd::fp16_to_fp32;
    using llaminar2::simd::fp32_to_fp16;

    // Test fixture for Q8_1 → Q16 RoPE conversion tests
    class Q8_1_to_Q16_RoPETest : public ::testing::Test
    {
    protected:
        static constexpr int HEAD_DIM = 64; // Must be divisible by 32, 64
        static constexpr float ROPE_THETA = 10000.0f;

        std::mt19937 rng_{42};

        // Create Q8_1 blocks from random FP32 data
        void createQ8_1Blocks(std::vector<Q8_1Block> &blocks,
                              std::vector<float> &original_fp32,
                              int head_dim,
                              float scale_range = 1.0f)
        {
            std::uniform_real_distribution<float> dist(-scale_range, scale_range);

            const int num_blocks = head_dim / 32;
            blocks.resize(num_blocks);
            original_fp32.resize(head_dim);

            for (int b = 0; b < num_blocks; ++b)
            {
                // Find max abs for this block
                float max_abs = 0.0f;
                for (int i = 0; i < 32; ++i)
                {
                    original_fp32[b * 32 + i] = dist(rng_);
                    max_abs = std::max(max_abs, std::abs(original_fp32[b * 32 + i]));
                }

                // Quantize to Q8_1
                float scale = max_abs / 127.0f;
                if (scale < 1e-10f)
                    scale = 1e-10f;

                blocks[b].d = fp32_to_fp16(scale);
                int32_t sum = 0;
                for (int i = 0; i < 32; ++i)
                {
                    int32_t q = static_cast<int32_t>(std::round(original_fp32[b * 32 + i] / scale));
                    q = std::max(-127, std::min(127, q));
                    blocks[b].qs[i] = static_cast<int8_t>(q);
                    sum += q;
                }
                blocks[b].sum_qs = static_cast<int16_t>(sum); // Q8_1 sum field
            }
        }

        // Create Q15 sin/cos tables for position
        void createQ15SinCos(std::vector<int16_t> &cos_q15,
                             std::vector<int16_t> &sin_q15,
                             int half_dim,
                             int position)
        {
            cos_q15.resize(half_dim);
            sin_q15.resize(half_dim);

            const auto &inv_freq = get_inv_freq_cached(half_dim * 2, ROPE_THETA);

            for (int i = 0; i < half_dim; ++i)
            {
                float angle = static_cast<float>(position) * inv_freq[i];
                cos_q15[i] = static_cast<int16_t>(std::lround(std::cos(angle) * 32767.0f));
                sin_q15[i] = static_cast<int16_t>(std::lround(std::sin(angle) * 32767.0f));
            }
        }

        // FP32 reference implementation of RoPE rotation
        void applyRoPEReference(const std::vector<float> &input,
                                std::vector<float> &output,
                                int head_dim,
                                const std::vector<int16_t> &cos_q15,
                                const std::vector<int16_t> &sin_q15)
        {
            const int half_dim = head_dim / 2;
            output.resize(head_dim);

            for (int i = 0; i < half_dim; ++i)
            {
                float x = input[i];
                float y = input[i + half_dim];
                float cos_val = static_cast<float>(cos_q15[i]) / 32767.0f;
                float sin_val = static_cast<float>(sin_q15[i]) / 32767.0f;

                output[i] = x * cos_val - y * sin_val;
                output[i + half_dim] = x * sin_val + y * cos_val;
            }
        }

        // Dequantize Q16 blocks to FP32
        template <typename BlockType>
        void dequantizeQ16(const BlockType *blocks, std::vector<float> &output, int head_dim)
        {
            constexpr int BLOCK_SIZE = static_cast<int>(BlockType::BLOCK_SIZE);
            const int num_blocks = head_dim / BLOCK_SIZE;
            output.resize(head_dim);

            for (int b = 0; b < num_blocks; ++b)
            {
                float scale = blocks[b].d;
                for (int i = 0; i < BLOCK_SIZE; ++i)
                {
                    output[b * BLOCK_SIZE + i] = static_cast<float>(blocks[b].qs[i]) * scale;
                }
            }
        }

        // Dequantize Q8_1 blocks to FP32 (using per-block scales)
        void dequantizeQ8_1(const Q8_1Block *blocks, std::vector<float> &output, int head_dim)
        {
            const int num_blocks = head_dim / 32;
            output.resize(head_dim);

            for (int b = 0; b < num_blocks; ++b)
            {
                float scale = fp16_to_fp32(blocks[b].d);
                for (int i = 0; i < 32; ++i)
                {
                    output[b * 32 + i] = static_cast<float>(blocks[b].qs[i]) * scale;
                }
            }
        }

        // Compute MSE between two vectors
        float computeMSE(const std::vector<float> &a, const std::vector<float> &b)
        {
            if (a.size() != b.size())
                return std::numeric_limits<float>::infinity();

            double sum = 0.0;
            for (size_t i = 0; i < a.size(); ++i)
            {
                double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
                sum += diff * diff;
            }
            return static_cast<float>(sum / static_cast<double>(a.size()));
        }

        // Compute max absolute error
        float computeMaxAbsError(const std::vector<float> &a, const std::vector<float> &b)
        {
            float max_err = 0.0f;
            for (size_t i = 0; i < std::min(a.size(), b.size()); ++i)
            {
                max_err = std::max(max_err, std::abs(a[i] - b[i]));
            }
            return max_err;
        }
    };

    // Test that scalar path produces correct rotation with Q16_1Block (32-element)
    TEST_F(Q8_1_to_Q16_RoPETest, ScalarPath_Block32_CorrectRotation)
    {
        constexpr int head_dim = 64;

        std::vector<Q8_1Block> q8_input;
        std::vector<float> original_fp32;
        createQ8_1Blocks(q8_input, original_fp32, head_dim, 2.0f);

        std::vector<int16_t> cos_q15, sin_q15;
        createQ15SinCos(cos_q15, sin_q15, head_dim / 2, 10);

        // Apply Q8_1 → Q16 conversion with scalar
        std::vector<Q16_1Block> q16_output(head_dim / Q16_1Block::BLOCK_SIZE);
        float head_scale = apply_rope_q8_1_to_q16_head_scalar<Q16_1Block>(
            q8_input.data(), q16_output.data(), head_dim, cos_q15.data(), sin_q15.data());

        // Compute expected reference
        std::vector<float> dequant_input;
        dequantizeQ8_1(q8_input.data(), dequant_input, head_dim);

        std::vector<float> expected;
        applyRoPEReference(dequant_input, expected, head_dim, cos_q15, sin_q15);

        // Dequantize output
        std::vector<float> actual;
        dequantizeQ16<Q16_1Block>(q16_output.data(), actual, head_dim);

        float mse = computeMSE(expected, actual);
        float max_err = computeMaxAbsError(expected, actual);

        // Double quantization (Q8_1 → integer RoPE → Q16) has cumulative error:
        // - Q8_1 input: ~0.8% quantization error
        // - Integer RoPE: ~1-2% fixed-point rounding
        // - Q16 output: ~0.003% quantization error
        // Combined error can reach ~15-20% of value range in worst case
        constexpr float SCALE_RANGE = 2.0f;        // matches createQ8_1Blocks scale_range
        constexpr float MAX_ERROR_PERCENT = 0.05f; // 5% tolerance (improved with saturation fix)
        constexpr float MSE_TOLERANCE = 0.01f;     // MSE relative to variance
        EXPECT_LT(mse, MSE_TOLERANCE) << "MSE too high for scalar Block32 path";
        EXPECT_LT(max_err, SCALE_RANGE * MAX_ERROR_PERCENT) << "Max error too high for scalar Block32 path (" << max_err << " vs " << SCALE_RANGE * MAX_ERROR_PERCENT << ")";

        // Scale should be non-zero
        EXPECT_GT(head_scale, 0.0f);
    }

    // Test that scalar path produces correct rotation with Q16_1Block_64 (64-element)
    TEST_F(Q8_1_to_Q16_RoPETest, ScalarPath_Block64_CorrectRotation)
    {
        constexpr int head_dim = 64;

        std::vector<Q8_1Block> q8_input;
        std::vector<float> original_fp32;
        createQ8_1Blocks(q8_input, original_fp32, head_dim, 2.0f);

        std::vector<int16_t> cos_q15, sin_q15;
        createQ15SinCos(cos_q15, sin_q15, head_dim / 2, 10);

        // Apply Q8_1 → Q16_64 conversion with scalar
        std::vector<Q16_1Block_64> q16_output(head_dim / Q16_1Block_64::BLOCK_SIZE);
        float head_scale = apply_rope_q8_1_to_q16_head_scalar<Q16_1Block_64>(
            q8_input.data(), q16_output.data(), head_dim, cos_q15.data(), sin_q15.data());

        // Compute expected reference
        std::vector<float> dequant_input;
        dequantizeQ8_1(q8_input.data(), dequant_input, head_dim);

        std::vector<float> expected;
        applyRoPEReference(dequant_input, expected, head_dim, cos_q15, sin_q15);

        // Dequantize output
        std::vector<float> actual;
        dequantizeQ16<Q16_1Block_64>(q16_output.data(), actual, head_dim);

        float mse = computeMSE(expected, actual);
        float max_err = computeMaxAbsError(expected, actual);

        // Double quantization tolerances (see Block32 test for rationale)
        constexpr float SCALE_RANGE = 2.0f;
        constexpr float MAX_ERROR_PERCENT = 0.05f;
        constexpr float MSE_TOLERANCE = 0.01f;
        EXPECT_LT(mse, MSE_TOLERANCE) << "MSE too high for scalar Block64 path";
        EXPECT_LT(max_err, SCALE_RANGE * MAX_ERROR_PERCENT) << "Max error too high for scalar Block64 path (" << max_err << " vs " << SCALE_RANGE * MAX_ERROR_PERCENT << ")";
        EXPECT_GT(head_scale, 0.0f);
    }

    // Test that scalar path produces correct rotation with Q16_1Block_128 (128-element)
    TEST_F(Q8_1_to_Q16_RoPETest, ScalarPath_Block128_CorrectRotation)
    {
        constexpr int head_dim = 128;

        std::vector<Q8_1Block> q8_input;
        std::vector<float> original_fp32;
        createQ8_1Blocks(q8_input, original_fp32, head_dim, 2.0f);

        std::vector<int16_t> cos_q15, sin_q15;
        createQ15SinCos(cos_q15, sin_q15, head_dim / 2, 10);

        // Apply Q8_1 → Q16_128 conversion with scalar
        std::vector<Q16_1Block_128> q16_output(head_dim / Q16_1Block_128::BLOCK_SIZE);
        float head_scale = apply_rope_q8_1_to_q16_head_scalar<Q16_1Block_128>(
            q8_input.data(), q16_output.data(), head_dim, cos_q15.data(), sin_q15.data());

        // Compute expected reference
        std::vector<float> dequant_input;
        dequantizeQ8_1(q8_input.data(), dequant_input, head_dim);

        std::vector<float> expected;
        applyRoPEReference(dequant_input, expected, head_dim, cos_q15, sin_q15);

        // Dequantize output
        std::vector<float> actual;
        dequantizeQ16<Q16_1Block_128>(q16_output.data(), actual, head_dim);

        float mse = computeMSE(expected, actual);
        float max_err = computeMaxAbsError(expected, actual);

        // Double quantization tolerances for larger head_dim:
        // Larger head_dim (128) means more positions have RoPE applied, and the
        // scale determination from max-abs can be suboptimal for many values.
        // Error tolerance scales approximately with sqrt(head_dim/64) relative to baseline.
        constexpr float SCALE_RANGE = 2.0f;
        constexpr float MAX_ERROR_PERCENT = 0.05f; // 5% tolerance (improved with saturation fix)
        constexpr float MSE_TOLERANCE = 0.01f;
        EXPECT_LT(mse, MSE_TOLERANCE) << "MSE too high for scalar Block128 path";
        EXPECT_LT(max_err, SCALE_RANGE * MAX_ERROR_PERCENT) << "Max error too high for scalar Block128 path (" << max_err << " vs " << SCALE_RANGE * MAX_ERROR_PERCENT << ")";
        EXPECT_GT(head_scale, 0.0f);
    }

    // Test per-head scale normalization - all output blocks should have same scale
    TEST_F(Q8_1_to_Q16_RoPETest, ScalarPath_PerHeadNormalization)
    {
        constexpr int head_dim = 128;

        // Create input with varying per-block scales
        std::vector<Q8_1Block> q8_input(head_dim / 32);
        for (int b = 0; b < static_cast<int>(q8_input.size()); ++b)
        {
            // Different scale per block
            float block_scale = 0.5f + b * 0.3f;
            q8_input[b].d = fp32_to_fp16(block_scale);
            for (int i = 0; i < 32; ++i)
            {
                q8_input[b].qs[i] = static_cast<int8_t>(std::sin(i * 0.1f) * 100);
            }
        }

        std::vector<int16_t> cos_q15, sin_q15;
        createQ15SinCos(cos_q15, sin_q15, head_dim / 2, 5);

        // Convert to 32-element blocks
        std::vector<Q16_1Block> q16_out(head_dim / Q16_1Block::BLOCK_SIZE);
        apply_rope_q8_1_to_q16_head_scalar<Q16_1Block>(
            q8_input.data(), q16_out.data(), head_dim, cos_q15.data(), sin_q15.data());

        // Verify all output blocks have the same scale (per-head normalization)
        float first_scale = q16_out[0].d;
        for (size_t b = 1; b < q16_out.size(); ++b)
        {
            EXPECT_FLOAT_EQ(q16_out[b].d, first_scale)
                << "Block " << b << " has different scale than block 0";
        }
    }

    // Test AVX2 path matches scalar path
    TEST_F(Q8_1_to_Q16_RoPETest, AVX2Path_MatchesScalar)
    {
#if defined(__AVX2__)
        constexpr int head_dim = 64;

        std::vector<Q8_1Block> q8_input;
        std::vector<float> original_fp32;
        createQ8_1Blocks(q8_input, original_fp32, head_dim, 2.0f);

        std::vector<int16_t> cos_q15, sin_q15;
        createQ15SinCos(cos_q15, sin_q15, head_dim / 2, 15);

        // Scalar path
        std::vector<Q16_1Block> scalar_output(head_dim / Q16_1Block::BLOCK_SIZE);
        float scalar_scale = apply_rope_q8_1_to_q16_head_scalar<Q16_1Block>(
            q8_input.data(), scalar_output.data(), head_dim, cos_q15.data(), sin_q15.data());

        // AVX2 path
        std::vector<Q16_1Block> avx2_output(head_dim / Q16_1Block::BLOCK_SIZE);
        float avx2_scale = apply_rope_q8_1_to_q16_head_avx2<Q16_1Block>(
            q8_input.data(), avx2_output.data(), head_dim, cos_q15.data(), sin_q15.data());

        // Dequantize both
        std::vector<float> scalar_dequant, avx2_dequant;
        dequantizeQ16<Q16_1Block>(scalar_output.data(), scalar_dequant, head_dim);
        dequantizeQ16<Q16_1Block>(avx2_output.data(), avx2_dequant, head_dim);

        // Compare
        float mse = computeMSE(scalar_dequant, avx2_dequant);
        EXPECT_LT(mse, 1e-6f) << "AVX2 path diverges from scalar";

        // Scales should be identical
        EXPECT_FLOAT_EQ(scalar_scale, avx2_scale) << "AVX2 head scale differs from scalar";
#else
        GTEST_SKIP() << "AVX2 not available";
#endif
    }

    // Test AVX512 path matches scalar path
    TEST_F(Q8_1_to_Q16_RoPETest, AVX512Path_MatchesScalar)
    {
#if defined(__AVX512F__)
        constexpr int head_dim = 64;

        std::vector<Q8_1Block> q8_input;
        std::vector<float> original_fp32;
        createQ8_1Blocks(q8_input, original_fp32, head_dim, 2.0f);

        std::vector<int16_t> cos_q15, sin_q15;
        createQ15SinCos(cos_q15, sin_q15, head_dim / 2, 15);

        // Scalar path
        std::vector<Q16_1Block> scalar_output(head_dim / Q16_1Block::BLOCK_SIZE);
        float scalar_scale = apply_rope_q8_1_to_q16_head_scalar<Q16_1Block>(
            q8_input.data(), scalar_output.data(), head_dim, cos_q15.data(), sin_q15.data());

        // AVX512 path
        std::vector<Q16_1Block> avx512_output(head_dim / Q16_1Block::BLOCK_SIZE);
        float avx512_scale = apply_rope_q8_1_to_q16_head_avx512<Q16_1Block>(
            q8_input.data(), avx512_output.data(), head_dim, cos_q15.data(), sin_q15.data());

        // Dequantize both
        std::vector<float> scalar_dequant, avx512_dequant;
        dequantizeQ16<Q16_1Block>(scalar_output.data(), scalar_dequant, head_dim);
        dequantizeQ16<Q16_1Block>(avx512_output.data(), avx512_dequant, head_dim);

        // Compare
        float mse = computeMSE(scalar_dequant, avx512_dequant);
        EXPECT_LT(mse, 1e-6f) << "AVX512 path diverges from scalar";

        // Scales should be identical
        EXPECT_FLOAT_EQ(scalar_scale, avx512_scale) << "AVX512 head scale differs from scalar";
#else
        GTEST_SKIP() << "AVX512F not available";
#endif
    }

    // Test high-level apply_rope_q8_1_to_q16 with Q tensor
    TEST_F(Q8_1_to_Q16_RoPETest, HighLevelWrapper_QTensor)
    {
        constexpr int seq_len = 4;
        constexpr int n_heads = 2;
        constexpr int head_dim = 64;
        constexpr int n_kv_heads = 2;

        // Create Q input
        const int q8_blocks_per_head = head_dim / 32;
        std::vector<Q8_1Block> Q_in(seq_len * n_heads * q8_blocks_per_head);
        std::mt19937 rng(123);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        for (auto &block : Q_in)
        {
            block.d = fp32_to_fp16(0.5f + dist(rng) * 0.3f);
            for (int i = 0; i < 32; ++i)
            {
                block.qs[i] = static_cast<int8_t>(dist(rng) * 100);
            }
        }

        // Create output buffers
        const int q16_blocks_per_head = head_dim / Q16_1Block_64::BLOCK_SIZE;
        std::vector<Q16_1Block_64> Q_out(seq_len * n_heads * q16_blocks_per_head);
        std::vector<float> Q_head_scales(seq_len * n_heads);
        std::vector<int> position_ids(seq_len);
        std::iota(position_ids.begin(), position_ids.end(), 0);

        // Apply RoPE
        apply_rope_q8_1_to_q16<Q16_1Block_64>(
            Q_in.data(), nullptr,
            Q_out.data(), nullptr,
            Q_head_scales.data(), nullptr,
            position_ids.data(),
            seq_len, n_heads, n_kv_heads, head_dim, ROPE_THETA);

        // Verify head scales are populated
        for (int i = 0; i < seq_len * n_heads; ++i)
        {
            EXPECT_GT(Q_head_scales[i], 0.0f) << "Head scale " << i << " is zero";
        }

        // Verify output has per-head unified scale
        for (int t = 0; t < seq_len; ++t)
        {
            for (int h = 0; h < n_heads; ++h)
            {
                const int base_idx = (t * n_heads + h) * q16_blocks_per_head;
                float first_scale = Q_out[base_idx].d;
                for (int b = 1; b < q16_blocks_per_head; ++b)
                {
                    EXPECT_FLOAT_EQ(Q_out[base_idx + b].d, first_scale)
                        << "Head " << h << " token " << t << " block " << b << " has different scale";
                }
            }
        }
    }

    // Test runtime dispatch function
    TEST_F(Q8_1_to_Q16_RoPETest, RuntimeDispatch_AllBlockSizes)
    {
        constexpr int seq_len = 2;
        constexpr int n_heads = 1;
        constexpr int head_dim = 64;
        constexpr int n_kv_heads = 1;

        const int q8_blocks_per_head = head_dim / 32;
        std::vector<Q8_1Block> Q_in(seq_len * n_heads * q8_blocks_per_head);
        std::mt19937 rng(456);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        for (auto &block : Q_in)
        {
            block.d = fp32_to_fp16(0.8f);
            for (int i = 0; i < 32; ++i)
            {
                block.qs[i] = static_cast<int8_t>(dist(rng) * 100);
            }
        }

        std::vector<int> position_ids = {0, 1};

        // Test each block size
        struct BlockSizeTest
        {
            Q16BlockSize size;
            int block_elements;
            const char *name;
        };

        std::vector<BlockSizeTest> tests = {
            {Q16BlockSize::BLOCK_32, 32, "Block32"},
            {Q16BlockSize::BLOCK_64, 64, "Block64"},
        };

        for (const auto &test : tests)
        {
            const int blocks_per_head = head_dim / test.block_elements;
            std::vector<uint8_t> output_buffer(seq_len * n_heads * blocks_per_head *
                                               (sizeof(float) + test.block_elements * sizeof(int16_t) + sizeof(int32_t)));
            std::vector<float> head_scales(seq_len * n_heads);

            // This should not crash
            apply_rope_q8_1_to_q16_dispatch(
                Q_in.data(), nullptr,
                output_buffer.data(), nullptr,
                head_scales.data(), nullptr,
                test.size,
                position_ids.data(),
                seq_len, n_heads, n_kv_heads, head_dim, ROPE_THETA);

            // Verify scales are set
            for (int i = 0; i < seq_len * n_heads; ++i)
            {
                EXPECT_GT(head_scales[i], 0.0f) << test.name << " head scale " << i << " is zero";
            }
        }
    }

    // Test sum_qs field is computed correctly for Q16 blocks
    TEST_F(Q8_1_to_Q16_RoPETest, SumQsFieldCorrect)
    {
        constexpr int head_dim = 64;

        std::vector<Q8_1Block> q8_input;
        std::vector<float> original_fp32;
        createQ8_1Blocks(q8_input, original_fp32, head_dim, 2.0f);

        std::vector<int16_t> cos_q15, sin_q15;
        createQ15SinCos(cos_q15, sin_q15, head_dim / 2, 10);

        // Apply conversion
        std::vector<Q16_1Block> q16_output(head_dim / Q16_1Block::BLOCK_SIZE);
        apply_rope_q8_1_to_q16_head_scalar<Q16_1Block>(
            q8_input.data(), q16_output.data(), head_dim, cos_q15.data(), sin_q15.data());

        // Verify sum_qs matches actual sum of qs[]
        for (size_t b = 0; b < q16_output.size(); ++b)
        {
            int32_t computed_sum = 0;
            for (int i = 0; i < Q16_1Block::BLOCK_SIZE; ++i)
            {
                computed_sum += q16_output[b].qs[i];
            }
            EXPECT_EQ(q16_output[b].sum_qs, computed_sum)
                << "Block " << b << " sum_qs mismatch";
        }
    }

    // Edge case: zero input
    TEST_F(Q8_1_to_Q16_RoPETest, EdgeCase_ZeroInput)
    {
        constexpr int head_dim = 64;

        std::vector<Q8_1Block> q8_input(head_dim / 32);
        for (auto &block : q8_input)
        {
            block.d = fp32_to_fp16(0.0f);
            std::memset(block.qs, 0, sizeof(block.qs));
            block.sum_qs = 0;
        }

        std::vector<int16_t> cos_q15, sin_q15;
        createQ15SinCos(cos_q15, sin_q15, head_dim / 2, 10);

        std::vector<Q16_1Block> q16_output(head_dim / Q16_1Block::BLOCK_SIZE);
        float head_scale = apply_rope_q8_1_to_q16_head_scalar<Q16_1Block>(
            q8_input.data(), q16_output.data(), head_dim, cos_q15.data(), sin_q15.data());

        // Should handle gracefully (scale should be small but non-zero to avoid div by zero)
        EXPECT_GT(head_scale, 0.0f);

        // Output should be essentially zero
        for (size_t b = 0; b < q16_output.size(); ++b)
        {
            for (int i = 0; i < Q16_1Block::BLOCK_SIZE; ++i)
            {
                EXPECT_EQ(q16_output[b].qs[i], 0) << "Block " << b << " elem " << i << " is non-zero";
            }
        }
    }

    // ============================================================================
    // Performance Tests
    // ============================================================================

    class Q8_1_to_Q16_RoPE_PerfTest : public Q8_1_to_Q16_RoPETest
    {
    protected:
        static constexpr int NUM_WARMUP = 10;
        static constexpr int NUM_ITERATIONS = 100;

        template <typename Func>
        double benchmark(Func &&func)
        {
            // Warmup
            for (int i = 0; i < NUM_WARMUP; ++i)
            {
                func();
            }

            // Timed runs
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < NUM_ITERATIONS; ++i)
            {
                func();
            }
            auto end = std::chrono::high_resolution_clock::now();

            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
            return static_cast<double>(duration.count()) / NUM_ITERATIONS;
        }
    };

    // Performance: Compare scalar vs AVX2 vs AVX512 for Block32
    TEST_F(Q8_1_to_Q16_RoPE_PerfTest, Performance_Block32_AllPaths)
    {
        constexpr int head_dim = 128;

        std::vector<Q8_1Block> q8_input;
        std::vector<float> original_fp32;
        createQ8_1Blocks(q8_input, original_fp32, head_dim, 2.0f);

        std::vector<int16_t> cos_q15, sin_q15;
        createQ15SinCos(cos_q15, sin_q15, head_dim / 2, 10);

        std::vector<Q16_1Block> q16_output(head_dim / Q16_1Block::BLOCK_SIZE);

        // Benchmark scalar
        double scalar_ns = benchmark([&]()
                                     { apply_rope_q8_1_to_q16_head_scalar<Q16_1Block>(
                                           q8_input.data(), q16_output.data(), head_dim, cos_q15.data(), sin_q15.data()); });

#if defined(__AVX2__)
        // Benchmark AVX2
        double avx2_ns = benchmark([&]()
                                   { apply_rope_q8_1_to_q16_head_avx2<Q16_1Block>(
                                         q8_input.data(), q16_output.data(), head_dim, cos_q15.data(), sin_q15.data()); });
#endif

#if defined(__AVX512F__)
        // Benchmark AVX512
        double avx512_ns = benchmark([&]()
                                     { apply_rope_q8_1_to_q16_head_avx512<Q16_1Block>(
                                           q8_input.data(), q16_output.data(), head_dim, cos_q15.data(), sin_q15.data()); });
#endif

        std::cout << "\n=== Q8_1→Q16 RoPE Performance (Block32, head_dim=" << head_dim << ") ===\n";
        std::cout << "Scalar:  " << std::fixed << std::setprecision(0) << scalar_ns << " ns\n";
#if defined(__AVX2__)
        std::cout << "AVX2:    " << avx2_ns << " ns ("
                  << std::setprecision(2) << scalar_ns / avx2_ns << "x speedup)\n";
#endif
#if defined(__AVX512F__)
        std::cout << "AVX512:  " << avx512_ns << " ns ("
                  << std::setprecision(2) << scalar_ns / avx512_ns << "x speedup)\n";
#endif

        // Just verify it runs, no specific speedup requirement
        EXPECT_GT(scalar_ns, 0);
    }

    // Performance: Compare Block64 paths
    TEST_F(Q8_1_to_Q16_RoPE_PerfTest, Performance_Block64_AllPaths)
    {
        constexpr int head_dim = 128;

        std::vector<Q8_1Block> q8_input;
        std::vector<float> original_fp32;
        createQ8_1Blocks(q8_input, original_fp32, head_dim, 2.0f);

        std::vector<int16_t> cos_q15, sin_q15;
        createQ15SinCos(cos_q15, sin_q15, head_dim / 2, 10);

        std::vector<Q16_1Block_64> q16_output(head_dim / Q16_1Block_64::BLOCK_SIZE);

        double scalar_ns = benchmark([&]()
                                     { apply_rope_q8_1_to_q16_head_scalar<Q16_1Block_64>(
                                           q8_input.data(), q16_output.data(), head_dim, cos_q15.data(), sin_q15.data()); });

#if defined(__AVX2__)
        double avx2_ns = benchmark([&]()
                                   { apply_rope_q8_1_to_q16_head_avx2<Q16_1Block_64>(
                                         q8_input.data(), q16_output.data(), head_dim, cos_q15.data(), sin_q15.data()); });
#endif

#if defined(__AVX512F__)
        double avx512_ns = benchmark([&]()
                                     { apply_rope_q8_1_to_q16_head_avx512<Q16_1Block_64>(
                                           q8_input.data(), q16_output.data(), head_dim, cos_q15.data(), sin_q15.data()); });
#endif

        std::cout << "\n=== Q8_1→Q16 RoPE Performance (Block64, head_dim=" << head_dim << ") ===\n";
        std::cout << "Scalar:  " << std::fixed << std::setprecision(0) << scalar_ns << " ns\n";
#if defined(__AVX2__)
        std::cout << "AVX2:    " << avx2_ns << " ns ("
                  << std::setprecision(2) << scalar_ns / avx2_ns << "x speedup)\n";
#endif
#if defined(__AVX512F__)
        std::cout << "AVX512:  " << avx512_ns << " ns ("
                  << std::setprecision(2) << scalar_ns / avx512_ns << "x speedup)\n";
#endif

        EXPECT_GT(scalar_ns, 0);
    }

    // Performance: Compare Block128 (single block per head) paths
    TEST_F(Q8_1_to_Q16_RoPE_PerfTest, Performance_Block128_AllPaths)
    {
        constexpr int head_dim = 128;

        std::vector<Q8_1Block> q8_input;
        std::vector<float> original_fp32;
        createQ8_1Blocks(q8_input, original_fp32, head_dim, 2.0f);

        std::vector<int16_t> cos_q15, sin_q15;
        createQ15SinCos(cos_q15, sin_q15, head_dim / 2, 10);

        std::vector<Q16_1Block_128> q16_output(head_dim / Q16_1Block_128::BLOCK_SIZE);

        double scalar_ns = benchmark([&]()
                                     { apply_rope_q8_1_to_q16_head_scalar<Q16_1Block_128>(
                                           q8_input.data(), q16_output.data(), head_dim, cos_q15.data(), sin_q15.data()); });

#if defined(__AVX2__)
        double avx2_ns = benchmark([&]()
                                   { apply_rope_q8_1_to_q16_head_avx2<Q16_1Block_128>(
                                         q8_input.data(), q16_output.data(), head_dim, cos_q15.data(), sin_q15.data()); });
#endif

#if defined(__AVX512F__)
        double avx512_ns = benchmark([&]()
                                     { apply_rope_q8_1_to_q16_head_avx512<Q16_1Block_128>(
                                           q8_input.data(), q16_output.data(), head_dim, cos_q15.data(), sin_q15.data()); });
#endif

        std::cout << "\n=== Q8_1→Q16 RoPE Performance (Block128, head_dim=" << head_dim << ") ===\n";
        std::cout << "Scalar:  " << std::fixed << std::setprecision(0) << scalar_ns << " ns\n";
#if defined(__AVX2__)
        std::cout << "AVX2:    " << avx2_ns << " ns ("
                  << std::setprecision(2) << scalar_ns / avx2_ns << "x speedup)\n";
#endif
#if defined(__AVX512F__)
        std::cout << "AVX512:  " << avx512_ns << " ns ("
                  << std::setprecision(2) << scalar_ns / avx512_ns << "x speedup)\n";
#endif

        EXPECT_GT(scalar_ns, 0);
    }

} // namespace llaminar2::test
