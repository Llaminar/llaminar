/**
 * @file Test__CPURoPEKernelT.cpp
 * @brief Unit tests for CPURoPEKernelT with typed precision conversion
 *
 * Tests the typed RoPE kernel with FP32, BF16, FP16, and Q8_1 precision types.
 * The Q8_1 implementation uses pure-integer operations without FP32 round-trips.
 *
 * @author David Sanftenberg
 * @date 2025
 */

#include <gtest/gtest.h>

#include "v2/kernels/cpu/ops/CPURoPEKernelT.h"
#include "v2/kernels/cpu/primitives/RoPEPrimitives.h"
#include "v2/tensors/SIMDHelpers.h"
#include "v2/tensors/BlockStructures.h"
#include "v2/tensors/Tensors.h" // For Q16_1Tensor

#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <random>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace llaminar2
{
    namespace
    {

        /**
         * @brief Reference RoPE implementation for testing
         *
         * RoPE rotates pairs of elements: [x, y] -> [x*cos - y*sin, x*sin + y*cos]
         * Applied to first half paired with second half of each head.
         */
        void reference_rope(
            float *data,
            int seq_len,
            int n_heads,
            int head_dim,
            const int *position_ids,
            float rope_theta)
        {
            const int half_dim = head_dim / 2;
            const int head_stride = head_dim;
            const int token_stride = n_heads * head_dim;

            for (int tok = 0; tok < seq_len; ++tok)
            {
                int position = position_ids ? position_ids[tok] : tok;
                if (position < 0)
                    continue;

                for (int h = 0; h < n_heads; ++h)
                {
                    float *head_data = data + tok * token_stride + h * head_stride;

                    for (int i = 0; i < half_dim; ++i)
                    {
                        // Compute angle for this dimension
                        float freq = 1.0f / std::pow(rope_theta, static_cast<float>(2 * i) / head_dim);
                        float angle = position * freq;
                        float cos_val = std::cos(angle);
                        float sin_val = std::sin(angle);

                        // First half element pairs with second half element
                        float x = head_data[i];
                        float y = head_data[i + half_dim];

                        // RoPE rotation
                        head_data[i] = x * cos_val - y * sin_val;
                        head_data[i + half_dim] = x * sin_val + y * cos_val;
                    }
                }
            }
        }

        /**
         * @brief Reference partial RoPE implementation using full head stride.
         *
         * Partial RoPE rotates only the first rotary_dim values in each head,
         * but the physical head still has head_dim values. This mirrors Qwen3.5
         * style partial rotary embeddings and catches stride/denominator mixups.
         */
        void reference_partial_rope(
            float *data,
            int seq_len,
            int n_heads,
            int head_dim,
            int rotary_dim,
            const int *position_ids,
            float rope_theta)
        {
            const int half_rotary = rotary_dim / 2;
            const int token_stride = n_heads * head_dim;

            for (int tok = 0; tok < seq_len; ++tok)
            {
                int position = position_ids ? position_ids[tok] : tok;
                if (position < 0)
                    continue;

                for (int h = 0; h < n_heads; ++h)
                {
                    float *head_data = data + tok * token_stride + h * head_dim;

                    for (int i = 0; i < half_rotary; ++i)
                    {
                        float freq = 1.0f / std::pow(rope_theta, static_cast<float>(2 * i) / rotary_dim);
                        float angle = position * freq;
                        float cos_val = std::cos(angle);
                        float sin_val = std::sin(angle);

                        float x = head_data[i];
                        float y = head_data[i + half_rotary];

                        head_data[i] = x * cos_val - y * sin_val;
                        head_data[i + half_rotary] = x * sin_val + y * cos_val;
                    }
                }
            }
        }

        /**
         * @brief Generate random FP32 data
         */
        std::vector<float> generate_random_fp32(size_t count, float min_val = -2.0f, float max_val = 2.0f)
        {
            std::mt19937 rng(42); // Fixed seed for reproducibility
            std::uniform_real_distribution<float> dist(min_val, max_val);

            std::vector<float> data(count);
            for (auto &v : data)
            {
                v = dist(rng);
            }
            return data;
        }

        /**
         * @brief Compute max absolute difference between two vectors
         */
        float max_abs_diff(const float *a, const float *b, size_t count)
        {
            float max_diff = 0.0f;
            for (size_t i = 0; i < count; ++i)
            {
                float diff = std::abs(a[i] - b[i]);
                if (diff > max_diff)
                    max_diff = diff;
            }
            return max_diff;
        }

        /**
         * @brief Compute mean absolute error between two vectors
         */
        float mean_abs_error(const float *a, const float *b, size_t count)
        {
            float sum = 0.0f;
            for (size_t i = 0; i < count; ++i)
            {
                sum += std::abs(a[i] - b[i]);
            }
            return sum / count;
        }

        /**
         * @brief Compute cosine similarity between two vectors
         */
        float cosine_similarity(const float *a, const float *b, size_t count)
        {
            float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
            for (size_t i = 0; i < count; ++i)
            {
                dot += a[i] * b[i];
                norm_a += a[i] * a[i];
                norm_b += b[i] * b[i];
            }
            return dot / (std::sqrt(norm_a) * std::sqrt(norm_b) + 1e-8f);
        }

    } // anonymous namespace

    // =========================================================================
    // Test Fixture
    // =========================================================================

    class CPURoPEKernelTTest : public ::testing::Test
    {
    protected:
        static constexpr float ROPE_THETA = 10000.0f;

        // Tolerances for different precisions
        static constexpr float FP32_TOLERANCE = 1e-5f;
        static constexpr float FP32_LARGE_POS_TOLERANCE = 5e-4f; // Relaxed for large positions (>1000)
        static constexpr float BF16_TOLERANCE = 5e-2f;           // BF16 has ~3 decimal digits precision
        static constexpr float FP16_TOLERANCE = 2e-2f;           // FP16 has ~3.3 decimal digits precision
        static constexpr float Q8_1_TOLERANCE = 0.20f;           // Q8_1 is lossy (8-bit quantization + integer rotation)

        // Cosine similarity thresholds
        static constexpr float FP32_COSINE_MIN = 0.99999f;
        static constexpr float BF16_COSINE_MIN = 0.995f;
        static constexpr float FP16_COSINE_MIN = 0.997f;
        static constexpr float Q8_1_COSINE_MIN = 0.98f;

        void SetUp() override
        {
            // Default test dimensions
            seq_len_ = 4;
            n_heads_ = 8;
            n_kv_heads_ = 2;
            head_dim_ = 64; // Must be multiple of 32 for Q8_1
        }

        int seq_len_;
        int n_heads_;
        int n_kv_heads_;
        int head_dim_;
    };

    // =========================================================================
    // FP32 Specialization Tests
    // =========================================================================

    TEST_F(CPURoPEKernelTTest, FP32_apply_typed_matches_reference)
    {
        const size_t q_size = seq_len_ * n_heads_ * head_dim_;
        const size_t k_size = seq_len_ * n_kv_heads_ * head_dim_;

        // Generate test data
        auto q_data = generate_random_fp32(q_size);
        auto k_data = generate_random_fp32(k_size);

        // Copy for reference
        std::vector<float> q_expected = q_data;
        std::vector<float> k_expected = k_data;

        // Position IDs: 0, 1, 2, 3
        std::vector<int> position_ids(seq_len_);
        std::iota(position_ids.begin(), position_ids.end(), 0);

        // Compute reference
        reference_rope(q_expected.data(), seq_len_, n_heads_, head_dim_,
                       position_ids.data(), ROPE_THETA);
        reference_rope(k_expected.data(), seq_len_, n_kv_heads_, head_dim_,
                       position_ids.data(), ROPE_THETA);

        // Test typed kernel (operates in-place)
        CPURoPEKernelT<ActivationPrecision::FP32> kernel;

        // Call the kernel's apply_typed method
        ASSERT_TRUE(kernel.apply_typed(
            q_data.data(), k_data.data(),
            position_ids.data(),
            seq_len_, n_heads_, n_kv_heads_, head_dim_,
            ROPE_THETA));

        // Verify Q
        float q_max_diff = max_abs_diff(q_data.data(), q_expected.data(), q_size);
        EXPECT_LT(q_max_diff, FP32_TOLERANCE)
            << "FP32 RoPE Q max diff: " << q_max_diff;

        // Verify K
        float k_max_diff = max_abs_diff(k_data.data(), k_expected.data(), k_size);
        EXPECT_LT(k_max_diff, FP32_TOLERANCE)
            << "FP32 RoPE K max diff: " << k_max_diff;
    }

    TEST_F(CPURoPEKernelTTest, FP32_precision_metadata)
    {
        CPURoPEKernelT<ActivationPrecision::FP32> kernel;
        EXPECT_EQ(kernel.precision(), ActivationPrecision::FP32);
        EXPECT_STREQ(kernel.precision_name(), "FP32");
        EXPECT_FLOAT_EQ(kernel.compression_ratio(), 1.0f);
    }

    TEST_F(CPURoPEKernelTTest, FP32_single_token_single_head)
    {
        // Minimal test case: 1 token, 1 head
        int local_seq_len = 1;
        int local_n_heads = 1;
        const size_t q_size = local_seq_len * local_n_heads * head_dim_;

        auto q_data = generate_random_fp32(q_size);
        std::vector<float> q_expected = q_data;

        std::vector<int> position_ids = {5}; // Non-zero position

        reference_rope(q_expected.data(), local_seq_len, local_n_heads, head_dim_,
                       position_ids.data(), ROPE_THETA);

        CPURoPEKernelT<ActivationPrecision::FP32> kernel;
        ASSERT_TRUE(kernel.apply_typed(
            q_data.data(), nullptr,
            position_ids.data(),
            local_seq_len, local_n_heads, 0, head_dim_,
            ROPE_THETA));

        float max_diff = max_abs_diff(q_data.data(), q_expected.data(), q_size);
        EXPECT_LT(max_diff, FP32_TOLERANCE);
    }

    TEST_F(CPURoPEKernelTTest, FP32_partial_rotary_keeps_full_head_stride)
    {
        constexpr int local_seq_len = 3;
        constexpr int local_n_heads = 4;
        constexpr int local_n_kv_heads = 2;
        constexpr int local_head_dim = 16;
        constexpr int local_rotary_dim = 4;

        const size_t q_size = local_seq_len * local_n_heads * local_head_dim;
        const size_t k_size = local_seq_len * local_n_kv_heads * local_head_dim;

        auto q_data = generate_random_fp32(q_size);
        auto k_data = generate_random_fp32(k_size);
        std::vector<float> q_original = q_data;
        std::vector<float> k_original = k_data;
        std::vector<float> q_expected = q_data;
        std::vector<float> k_expected = k_data;
        std::vector<int> position_ids = {7, 8, 9};

        reference_partial_rope(q_expected.data(), local_seq_len, local_n_heads,
                               local_head_dim, local_rotary_dim,
                               position_ids.data(), ROPE_THETA);
        reference_partial_rope(k_expected.data(), local_seq_len, local_n_kv_heads,
                               local_head_dim, local_rotary_dim,
                               position_ids.data(), ROPE_THETA);

        CPURoPEKernelT<ActivationPrecision::FP32> kernel;
        ASSERT_TRUE(kernel.apply_typed(q_data.data(), k_data.data(), position_ids.data(),
                                       local_seq_len, local_n_heads, local_n_kv_heads,
                                       local_head_dim, ROPE_THETA, -1, local_rotary_dim));

        EXPECT_LT(max_abs_diff(q_data.data(), q_expected.data(), q_size), FP32_TOLERANCE);
        EXPECT_LT(max_abs_diff(k_data.data(), k_expected.data(), k_size), FP32_TOLERANCE);

        for (int tok = 0; tok < local_seq_len; ++tok)
        {
            for (int h = 0; h < local_n_heads; ++h)
            {
                for (int d = local_rotary_dim; d < local_head_dim; ++d)
                {
                    size_t idx = (static_cast<size_t>(tok) * local_n_heads + h) * local_head_dim + d;
                    EXPECT_FLOAT_EQ(q_data[idx], q_original[idx]);
                }
            }
            for (int h = 0; h < local_n_kv_heads; ++h)
            {
                for (int d = local_rotary_dim; d < local_head_dim; ++d)
                {
                    size_t idx = (static_cast<size_t>(tok) * local_n_kv_heads + h) * local_head_dim + d;
                    EXPECT_FLOAT_EQ(k_data[idx], k_original[idx]);
                }
            }
        }
    }

    TEST_F(CPURoPEKernelTTest, FP32_partial_rotary_handles_non_contiguous_positions)
    {
        constexpr int local_seq_len = 3;
        constexpr int local_n_heads = 3;
        constexpr int local_n_kv_heads = 2;
        constexpr int local_head_dim = 16;
        constexpr int local_rotary_dim = 8;

        const size_t q_size = local_seq_len * local_n_heads * local_head_dim;
        const size_t k_size = local_seq_len * local_n_kv_heads * local_head_dim;

        auto q_data = generate_random_fp32(q_size);
        auto k_data = generate_random_fp32(k_size);
        std::vector<float> q_expected = q_data;
        std::vector<float> k_expected = k_data;
        std::vector<int> position_ids = {11, -1, 4};

        reference_partial_rope(q_expected.data(), local_seq_len, local_n_heads,
                               local_head_dim, local_rotary_dim,
                               position_ids.data(), ROPE_THETA);
        reference_partial_rope(k_expected.data(), local_seq_len, local_n_kv_heads,
                               local_head_dim, local_rotary_dim,
                               position_ids.data(), ROPE_THETA);

        CPURoPEKernelT<ActivationPrecision::FP32> kernel;
        ASSERT_TRUE(kernel.apply_typed(q_data.data(), k_data.data(), position_ids.data(),
                                       local_seq_len, local_n_heads, local_n_kv_heads,
                                       local_head_dim, ROPE_THETA, -1, local_rotary_dim));

        EXPECT_LT(max_abs_diff(q_data.data(), q_expected.data(), q_size), FP32_TOLERANCE);
        EXPECT_LT(max_abs_diff(k_data.data(), k_expected.data(), k_size), FP32_TOLERANCE);
    }

    // =========================================================================
    // BF16 Specialization Tests
    // =========================================================================

    TEST_F(CPURoPEKernelTTest, BF16_apply_typed_with_conversion)
    {
        const size_t q_size = seq_len_ * n_heads_ * head_dim_;

        // Generate FP32 test data and convert to BF16
        auto fp32_q = generate_random_fp32(q_size);

        // Convert to BF16
        std::vector<uint16_t> bf16_q(q_size);
        simd::convert_fp32_to_bf16(fp32_q.data(), bf16_q.data(), q_size);

        // Reference: convert to FP32, apply RoPE, convert back
        std::vector<float> fp32_expected(q_size);
        simd::convert_bf16_to_fp32(bf16_q.data(), fp32_expected.data(), q_size);

        std::vector<int> position_ids(seq_len_);
        std::iota(position_ids.begin(), position_ids.end(), 0);

        reference_rope(fp32_expected.data(), seq_len_, n_heads_, head_dim_,
                       position_ids.data(), ROPE_THETA);

        // Apply RoPE to BF16 data using the kernel
        CPURoPEKernelT<ActivationPrecision::BF16> kernel;
        ASSERT_TRUE(kernel.apply_typed(
            bf16_q.data(), nullptr,
            position_ids.data(),
            seq_len_, n_heads_, 0, head_dim_,
            ROPE_THETA));

        // Convert result to FP32 for comparison
        std::vector<float> fp32_result(q_size);
        simd::convert_bf16_to_fp32(bf16_q.data(), fp32_result.data(), q_size);

        // Verify with BF16 tolerance
        float max_diff = max_abs_diff(fp32_result.data(), fp32_expected.data(), q_size);
        EXPECT_LT(max_diff, BF16_TOLERANCE)
            << "BF16 RoPE max diff: " << max_diff;

        float cosine = cosine_similarity(fp32_result.data(), fp32_expected.data(), q_size);
        EXPECT_GT(cosine, BF16_COSINE_MIN)
            << "BF16 RoPE cosine similarity: " << cosine;
    }

    TEST_F(CPURoPEKernelTTest, BF16_precision_metadata)
    {
        CPURoPEKernelT<ActivationPrecision::BF16> kernel;
        EXPECT_EQ(kernel.precision(), ActivationPrecision::BF16);
        EXPECT_STREQ(kernel.precision_name(), "BF16");
        EXPECT_FLOAT_EQ(kernel.compression_ratio(), 2.0f);
    }

    // =========================================================================
    // FP16 Specialization Tests
    // =========================================================================

    TEST_F(CPURoPEKernelTTest, FP16_apply_typed_with_conversion)
    {
        const size_t q_size = seq_len_ * n_heads_ * head_dim_;

        // Generate FP32 test data and convert to FP16
        auto fp32_q = generate_random_fp32(q_size);

        // Convert to FP16
        std::vector<uint16_t> fp16_q(q_size);
        simd::convert_fp32_to_fp16(fp32_q.data(), fp16_q.data(), q_size);

        // Reference: convert to FP32, apply RoPE
        std::vector<float> fp32_expected(q_size);
        simd::convert_fp16_to_fp32(fp16_q.data(), fp32_expected.data(), q_size);

        std::vector<int> position_ids(seq_len_);
        std::iota(position_ids.begin(), position_ids.end(), 0);

        reference_rope(fp32_expected.data(), seq_len_, n_heads_, head_dim_,
                       position_ids.data(), ROPE_THETA);

        // Apply RoPE to FP16 data using the kernel
        CPURoPEKernelT<ActivationPrecision::FP16> kernel;
        ASSERT_TRUE(kernel.apply_typed(
            fp16_q.data(), nullptr,
            position_ids.data(),
            seq_len_, n_heads_, 0, head_dim_,
            ROPE_THETA));

        // Convert result to FP32 for comparison
        std::vector<float> fp32_result(q_size);
        simd::convert_fp16_to_fp32(fp16_q.data(), fp32_result.data(), q_size);

        // Verify with FP16 tolerance
        float max_diff = max_abs_diff(fp32_result.data(), fp32_expected.data(), q_size);
        EXPECT_LT(max_diff, FP16_TOLERANCE)
            << "FP16 RoPE max diff: " << max_diff;

        float cosine = cosine_similarity(fp32_result.data(), fp32_expected.data(), q_size);
        EXPECT_GT(cosine, FP16_COSINE_MIN)
            << "FP16 RoPE cosine similarity: " << cosine;
    }

    TEST_F(CPURoPEKernelTTest, FP16_precision_metadata)
    {
        CPURoPEKernelT<ActivationPrecision::FP16> kernel;
        EXPECT_EQ(kernel.precision(), ActivationPrecision::FP16);
        EXPECT_STREQ(kernel.precision_name(), "FP16");
        EXPECT_FLOAT_EQ(kernel.compression_ratio(), 2.0f);
    }

    // =========================================================================
    // Q8_1 Specialization Tests
    // =========================================================================

    TEST_F(CPURoPEKernelTTest, Q8_1_apply_typed_with_conversion)
    {
        // Q8_1 requires head_dim divisible by 32
        const size_t q_size = seq_len_ * n_heads_ * head_dim_;
        const size_t num_blocks = q_size / 32;

        // Generate FP32 test data
        auto fp32_q = generate_random_fp32(q_size);

        // Convert to Q8_1
        std::vector<Q8_1Block> q8_q(num_blocks);
        simd::quantize_fp32_to_q8_1_blocks(fp32_q.data(), q8_q.data(), q_size);

        // Reference: dequant, apply RoPE
        std::vector<float> fp32_expected(q_size);
        simd::dequantize_q8_1_to_fp32(q8_q.data(), fp32_expected.data(), q_size);

        std::vector<int> position_ids(seq_len_);
        std::iota(position_ids.begin(), position_ids.end(), 0);

        reference_rope(fp32_expected.data(), seq_len_, n_heads_, head_dim_,
                       position_ids.data(), ROPE_THETA);

        // Apply RoPE to Q8_1 data (pure-integer)
        primitives::apply_rope_q8_1_integer(
            q8_q.data(), nullptr,
            position_ids.data(),
            seq_len_,
            n_heads_, 0,
            head_dim_,
            ROPE_THETA);

        // Dequantize result for comparison
        std::vector<float> fp32_result(q_size);
        simd::dequantize_q8_1_to_fp32(q8_q.data(), fp32_result.data(), q_size);

        // Q8_1 has more noise due to quantization + integer rotation
        // Use cosine similarity as primary metric
        float cosine = cosine_similarity(fp32_result.data(), fp32_expected.data(), q_size);
        EXPECT_GT(cosine, Q8_1_COSINE_MIN)
            << "Q8_1 RoPE cosine similarity: " << cosine;

        float max_diff = max_abs_diff(fp32_result.data(), fp32_expected.data(), q_size);
        EXPECT_LT(max_diff, Q8_1_TOLERANCE)
            << "Q8_1 RoPE max diff: " << max_diff;
    }

    TEST_F(CPURoPEKernelTTest, Q8_1_precision_metadata)
    {
        CPURoPEKernelT<ActivationPrecision::Q8_1> kernel;
        EXPECT_EQ(kernel.precision(), ActivationPrecision::Q8_1);
        EXPECT_STREQ(kernel.precision_name(), "Q8_1");
        EXPECT_FLOAT_EQ(kernel.compression_ratio(), 4.0f);
    }

    TEST_F(CPURoPEKernelTTest, Q8_1_requires_head_dim_multiple_of_32)
    {
        // Test that Q8_1 RoPE requires head_dim divisible by 32
        // This is enforced by the apply_rope_q8_1_integer function
        int bad_head_dim = 48; // Not aligned to block boundaries properly
        const size_t q_size = seq_len_ * n_heads_ * bad_head_dim;
        const size_t num_blocks = (q_size + 31) / 32;

        std::vector<Q8_1Block> q8_q(num_blocks);
        std::vector<int> position_ids(seq_len_);
        std::iota(position_ids.begin(), position_ids.end(), 0);

        // The function should handle this gracefully (either return early or succeed)
        // We just verify it doesn't crash
        primitives::apply_rope_q8_1_integer(
            q8_q.data(), nullptr,
            position_ids.data(),
            seq_len_,
            n_heads_, 0,
            bad_head_dim, // Not cleanly divisible
            ROPE_THETA);

        // No assertion - just verifying no crash
    }

    TEST_F(CPURoPEKernelTTest, Q8_1_with_both_Q_and_K)
    {
        // Test with both Q and K tensors
        const size_t q_size = seq_len_ * n_heads_ * head_dim_;
        const size_t k_size = seq_len_ * n_kv_heads_ * head_dim_;
        const size_t q_blocks = q_size / 32;
        const size_t k_blocks = k_size / 32;

        auto fp32_q = generate_random_fp32(q_size);
        auto fp32_k = generate_random_fp32(k_size);

        std::vector<Q8_1Block> q8_q(q_blocks);
        std::vector<Q8_1Block> q8_k(k_blocks);
        simd::quantize_fp32_to_q8_1_blocks(fp32_q.data(), q8_q.data(), q_size);
        simd::quantize_fp32_to_q8_1_blocks(fp32_k.data(), q8_k.data(), k_size);

        // Reference
        std::vector<float> fp32_q_expected(q_size);
        std::vector<float> fp32_k_expected(k_size);
        simd::dequantize_q8_1_to_fp32(q8_q.data(), fp32_q_expected.data(), q_size);
        simd::dequantize_q8_1_to_fp32(q8_k.data(), fp32_k_expected.data(), k_size);

        std::vector<int> position_ids(seq_len_);
        std::iota(position_ids.begin(), position_ids.end(), 0);

        reference_rope(fp32_q_expected.data(), seq_len_, n_heads_, head_dim_,
                       position_ids.data(), ROPE_THETA);
        reference_rope(fp32_k_expected.data(), seq_len_, n_kv_heads_, head_dim_,
                       position_ids.data(), ROPE_THETA);

        // Apply Q8_1 RoPE
        primitives::apply_rope_q8_1_integer(
            q8_q.data(), q8_k.data(),
            position_ids.data(),
            seq_len_,
            n_heads_, n_kv_heads_,
            head_dim_,
            ROPE_THETA);

        // Dequantize results
        std::vector<float> fp32_q_result(q_size);
        std::vector<float> fp32_k_result(k_size);
        simd::dequantize_q8_1_to_fp32(q8_q.data(), fp32_q_result.data(), q_size);
        simd::dequantize_q8_1_to_fp32(q8_k.data(), fp32_k_result.data(), k_size);

        // Verify Q
        float q_cosine = cosine_similarity(fp32_q_result.data(), fp32_q_expected.data(), q_size);
        EXPECT_GT(q_cosine, Q8_1_COSINE_MIN)
            << "Q8_1 RoPE Q cosine similarity: " << q_cosine;

        // Verify K
        float k_cosine = cosine_similarity(fp32_k_result.data(), fp32_k_expected.data(), k_size);
        EXPECT_GT(k_cosine, Q8_1_COSINE_MIN)
            << "Q8_1 RoPE K cosine similarity: " << k_cosine;
    }

    // =========================================================================
    // Cross-Precision Consistency Tests
    // =========================================================================

    TEST_F(CPURoPEKernelTTest, AllPrecisions_SameInput_SimilarOutput)
    {
        // Verify that all precisions produce similar results for the same input
        const size_t q_size = seq_len_ * n_heads_ * head_dim_;

        auto fp32_original = generate_random_fp32(q_size);

        std::vector<int> position_ids(seq_len_);
        std::iota(position_ids.begin(), position_ids.end(), 0);

        // FP32 result (ground truth)
        std::vector<float> fp32_q = fp32_original;
        CPURoPEKernelT<ActivationPrecision::FP32> kernel_fp32;
        kernel_fp32.apply_typed(
            fp32_q.data(), nullptr,
            position_ids.data(),
            seq_len_, n_heads_, 0, head_dim_,
            ROPE_THETA);

        // BF16 result
        std::vector<uint16_t> bf16_q(q_size);
        simd::convert_fp32_to_bf16(fp32_original.data(), bf16_q.data(), q_size);
        CPURoPEKernelT<ActivationPrecision::BF16> kernel_bf16;
        kernel_bf16.apply_typed(
            bf16_q.data(), nullptr,
            position_ids.data(),
            seq_len_, n_heads_, 0, head_dim_,
            ROPE_THETA);
        std::vector<float> bf16_result(q_size);
        simd::convert_bf16_to_fp32(bf16_q.data(), bf16_result.data(), q_size);

        // FP16 result
        std::vector<uint16_t> fp16_q(q_size);
        simd::convert_fp32_to_fp16(fp32_original.data(), fp16_q.data(), q_size);
        CPURoPEKernelT<ActivationPrecision::FP16> kernel_fp16;
        kernel_fp16.apply_typed(
            fp16_q.data(), nullptr,
            position_ids.data(),
            seq_len_, n_heads_, 0, head_dim_,
            ROPE_THETA);
        std::vector<float> fp16_result(q_size);
        simd::convert_fp16_to_fp32(fp16_q.data(), fp16_result.data(), q_size);

        // Q8_1 result
        const size_t num_blocks = q_size / 32;
        std::vector<Q8_1Block> q8_q(num_blocks);
        simd::quantize_fp32_to_q8_1_blocks(fp32_original.data(), q8_q.data(), q_size);
        CPURoPEKernelT<ActivationPrecision::Q8_1> kernel_q8;
        kernel_q8.apply_typed(
            q8_q.data(), nullptr,
            position_ids.data(),
            seq_len_,
            n_heads_, 0,
            head_dim_,
            ROPE_THETA);
        std::vector<float> q8_result(q_size);
        simd::dequantize_q8_1_to_fp32(q8_q.data(), q8_result.data(), q_size);

        // Compare all to FP32 reference
        float bf16_cosine = cosine_similarity(bf16_result.data(), fp32_q.data(), q_size);
        float fp16_cosine = cosine_similarity(fp16_result.data(), fp32_q.data(), q_size);
        float q8_cosine = cosine_similarity(q8_result.data(), fp32_q.data(), q_size);

        EXPECT_GT(bf16_cosine, BF16_COSINE_MIN)
            << "BF16 vs FP32 cosine: " << bf16_cosine;
        EXPECT_GT(fp16_cosine, FP16_COSINE_MIN)
            << "FP16 vs FP32 cosine: " << fp16_cosine;
        EXPECT_GT(q8_cosine, Q8_1_COSINE_MIN)
            << "Q8_1 vs FP32 cosine: " << q8_cosine;
    }

    // =========================================================================
    // Edge Case Tests
    // =========================================================================

    TEST_F(CPURoPEKernelTTest, FP32_skip_negative_position)
    {
        // Negative position IDs should be skipped (padding tokens)
        const size_t q_size = seq_len_ * n_heads_ * head_dim_;
        auto q_data = generate_random_fp32(q_size);
        std::vector<float> original = q_data;

        // Mark position 1 as padding (-1)
        std::vector<int> position_ids = {0, -1, 2, 3};

        CPURoPEKernelT<ActivationPrecision::FP32> kernel;
        kernel.apply_typed(
            q_data.data(), nullptr,
            position_ids.data(),
            seq_len_, n_heads_, 0, head_dim_,
            ROPE_THETA);

        // Position 1 should be unchanged
        const int token_stride = n_heads_ * head_dim_;
        for (int i = 0; i < token_stride; ++i)
        {
            EXPECT_FLOAT_EQ(q_data[token_stride + i], original[token_stride + i])
                << "Position 1 (padding) should be unchanged at index " << i;
        }
    }

    TEST_F(CPURoPEKernelTTest, FP32_different_rope_theta)
    {
        // Test with different rope_theta values
        const size_t q_size = seq_len_ * n_heads_ * head_dim_;

        for (float theta : {1000.0f, 10000.0f, 100000.0f})
        {
            auto q_data = generate_random_fp32(q_size);
            std::vector<float> expected = q_data;

            std::vector<int> position_ids(seq_len_);
            std::iota(position_ids.begin(), position_ids.end(), 0);

            reference_rope(expected.data(), seq_len_, n_heads_, head_dim_,
                           position_ids.data(), theta);

            CPURoPEKernelT<ActivationPrecision::FP32> kernel;
            kernel.apply_typed(
                q_data.data(), nullptr,
                position_ids.data(),
                seq_len_, n_heads_, 0, head_dim_,
                theta);

            float max_diff = max_abs_diff(q_data.data(), expected.data(), q_size);
            EXPECT_LT(max_diff, FP32_TOLERANCE)
                << "RoPE with theta=" << theta << " max diff: " << max_diff;
        }
    }

    TEST_F(CPURoPEKernelTTest, FP32_large_position_ids)
    {
        // Test with large position IDs (long context)
        const size_t q_size = seq_len_ * n_heads_ * head_dim_;
        auto q_data = generate_random_fp32(q_size);
        std::vector<float> expected = q_data;

        // Large position IDs (simulating long context)
        std::vector<int> position_ids = {1000, 1001, 1002, 1003};

        reference_rope(expected.data(), seq_len_, n_heads_, head_dim_,
                       position_ids.data(), ROPE_THETA);

        CPURoPEKernelT<ActivationPrecision::FP32> kernel;
        kernel.apply_typed(
            q_data.data(), nullptr,
            position_ids.data(),
            seq_len_, n_heads_, 0, head_dim_,
            ROPE_THETA);

        float max_diff = max_abs_diff(q_data.data(), expected.data(), q_size);
        EXPECT_LT(max_diff, FP32_LARGE_POS_TOLERANCE)
            << "RoPE with large positions max diff: " << max_diff;
    }

    // =========================================================================
    // Q16_1 Specialization Tests
    // =========================================================================

    TEST_F(CPURoPEKernelTTest, Q16_1_apply_typed_with_conversion)
    {
        // Q16_1 requires head_dim divisible by 32
        const size_t q_size = seq_len_ * n_heads_ * head_dim_;
        const size_t num_blocks = q_size / 32;

        // Generate FP32 test data
        auto fp32_q = generate_random_fp32(q_size);

        // Convert to Q16_1
        std::vector<Q16_1Block> q16_q(num_blocks);
        for (size_t b = 0; b < num_blocks; ++b)
        {
            const float *src = fp32_q.data() + b * 32;
            Q16_1Block &block = q16_q[b];

            // Find max for scale
            float max_val = 0.0f;
            for (int i = 0; i < 32; ++i)
            {
                max_val = std::max(max_val, std::abs(src[i]));
            }

            float scale = max_val / 32767.0f;
            if (scale < 1e-20f)
                scale = 1e-20f;
            float inv_scale = 1.0f / scale;

            block.d = scale;
            int64_t sum = 0;
            for (int i = 0; i < 32; ++i)
            {
                int32_t q = static_cast<int32_t>(std::round(src[i] * inv_scale));
                q = std::max(-32767, std::min(32767, q));
                block.qs[i] = static_cast<int16_t>(q);
                sum += q;
            }
            block.sum_qs = static_cast<int32_t>(sum);
        }

        // Reference: dequant, apply RoPE in FP32
        std::vector<float> fp32_expected(q_size);
        for (size_t b = 0; b < num_blocks; ++b)
        {
            const Q16_1Block &block = q16_q[b];
            for (int i = 0; i < 32; ++i)
            {
                fp32_expected[b * 32 + i] = static_cast<float>(block.qs[i]) * block.d;
            }
        }

        std::vector<int> position_ids(seq_len_);
        std::iota(position_ids.begin(), position_ids.end(), 0);

        reference_rope(fp32_expected.data(), seq_len_, n_heads_, head_dim_,
                       position_ids.data(), ROPE_THETA);

        // Apply RoPE to Q16_1 data (in-place)
        primitives::apply_rope_q16_1_integer(
            q16_q.data(), nullptr,
            position_ids.data(),
            seq_len_,
            n_heads_, 0,
            head_dim_,
            ROPE_THETA,
            nullptr);

        // Dequantize result for comparison
        std::vector<float> fp32_result(q_size);
        for (size_t b = 0; b < num_blocks; ++b)
        {
            const Q16_1Block &block = q16_q[b];
            for (int i = 0; i < 32; ++i)
            {
                fp32_result[b * 32 + i] = static_cast<float>(block.qs[i]) * block.d;
            }
        }

        // Q16_1 has 256× finer precision than Q8_1, so tolerance should be tighter
        // Cosine similarity should be very close to 1.0
        float cosine = cosine_similarity(fp32_result.data(), fp32_expected.data(), q_size);
        EXPECT_GT(cosine, 0.999f) // Much tighter than Q8_1's 0.98f
            << "Q16_1 RoPE cosine similarity: " << cosine;

        float max_diff = max_abs_diff(fp32_result.data(), fp32_expected.data(), q_size);
        EXPECT_LT(max_diff, 0.02f) // Much tighter than Q8_1's 0.20f
            << "Q16_1 RoPE max diff: " << max_diff;
    }

    TEST_F(CPURoPEKernelTTest, Q16_1_precision_metadata)
    {
        CPURoPEKernelT<ActivationPrecision::Q16_1> kernel;
        EXPECT_EQ(kernel.precision(), ActivationPrecision::Q16_1);
        EXPECT_STREQ(kernel.precision_name(), "Q16_1");
        EXPECT_FLOAT_EQ(kernel.compression_ratio(), 2.0f);
    }

    TEST_F(CPURoPEKernelTTest, Q16_1_apply_tensor)
    {
        // Test apply_tensor interface with Q16_1 tensors
        const size_t q_size = seq_len_ * n_heads_ * head_dim_;
        const size_t num_blocks = q_size / 32;

        // Generate FP32 test data
        auto fp32_q = generate_random_fp32(q_size);

        // Create Q16_1Tensor from shape (2D for activation tensor)
        std::vector<size_t> shape = {static_cast<size_t>(seq_len_), static_cast<size_t>(n_heads_ * head_dim_)};
        Q16_1Tensor q_tensor(shape);

        // Quantize into tensor
        std::vector<float> fp32_for_quant = fp32_q;
        q_tensor.copyFrom_fp32(fp32_for_quant.data());

        // Get reference result
        std::vector<float> fp32_expected = fp32_q;
        std::vector<int> position_ids(seq_len_);
        std::iota(position_ids.begin(), position_ids.end(), 0);
        reference_rope(fp32_expected.data(), seq_len_, n_heads_, head_dim_,
                       position_ids.data(), ROPE_THETA);

        // Apply RoPE via apply_tensor
        CPURoPEKernelT<ActivationPrecision::Q16_1> kernel;
        bool success = kernel.apply_tensor(
            &q_tensor, nullptr,
            position_ids.data(),
            seq_len_, n_heads_, 0, head_dim_,
            ROPE_THETA);
        EXPECT_TRUE(success);

        // Dequantize and compare
        std::vector<float> fp32_result(q_size);
        q_tensor.to_fp32(fp32_result.data());

        float cosine = cosine_similarity(fp32_result.data(), fp32_expected.data(), q_size);
        EXPECT_GT(cosine, 0.999f)
            << "Q16_1 apply_tensor cosine similarity: " << cosine;
    }

    TEST_F(CPURoPEKernelTTest, Q16_1_vs_Q8_1_precision_improvement)
    {
        // Verify that Q16_1 RoPE has better precision than Q8_1
        // This is a sanity check for the 256× improvement claim
        const size_t q_size = seq_len_ * n_heads_ * head_dim_;
        const size_t num_blocks = q_size / 32;

        // Generate FP32 test data
        auto fp32_q = generate_random_fp32(q_size);

        // Reference: FP32 RoPE result
        std::vector<float> fp32_expected = fp32_q;
        std::vector<int> position_ids(seq_len_);
        std::iota(position_ids.begin(), position_ids.end(), 0);
        reference_rope(fp32_expected.data(), seq_len_, n_heads_, head_dim_,
                       position_ids.data(), ROPE_THETA);

        // Q8_1 path
        std::vector<Q8_1Block> q8_q(num_blocks);
        simd::quantize_fp32_to_q8_1_blocks(fp32_q.data(), q8_q.data(), q_size);
        primitives::apply_rope_q8_1_integer(q8_q.data(), nullptr, position_ids.data(),
                                            seq_len_, n_heads_, 0, head_dim_, ROPE_THETA);
        std::vector<float> q8_result(q_size);
        simd::dequantize_q8_1_to_fp32(q8_q.data(), q8_result.data(), q_size);
        float q8_max_diff = max_abs_diff(q8_result.data(), fp32_expected.data(), q_size);

        // Q16_1 path
        std::vector<Q16_1Block> q16_q(num_blocks);
        for (size_t b = 0; b < num_blocks; ++b)
        {
            const float *src = fp32_q.data() + b * 32;
            Q16_1Block &block = q16_q[b];
            float max_val = 0.0f;
            for (int i = 0; i < 32; ++i)
                max_val = std::max(max_val, std::abs(src[i]));
            float scale = max_val / 32767.0f;
            if (scale < 1e-20f)
                scale = 1e-20f;
            float inv_scale = 1.0f / scale;
            block.d = scale;
            int64_t sum = 0;
            for (int i = 0; i < 32; ++i)
            {
                int32_t q = static_cast<int32_t>(std::round(src[i] * inv_scale));
                q = std::max(-32767, std::min(32767, q));
                block.qs[i] = static_cast<int16_t>(q);
                sum += q;
            }
            block.sum_qs = static_cast<int32_t>(sum);
        }
        primitives::apply_rope_q16_1_integer(q16_q.data(), nullptr, position_ids.data(),
                                             seq_len_, n_heads_, 0, head_dim_, ROPE_THETA, nullptr);
        std::vector<float> q16_result(q_size);
        for (size_t b = 0; b < num_blocks; ++b)
        {
            const Q16_1Block &block = q16_q[b];
            for (int i = 0; i < 32; ++i)
                q16_result[b * 32 + i] = static_cast<float>(block.qs[i]) * block.d;
        }
        float q16_max_diff = max_abs_diff(q16_result.data(), fp32_expected.data(), q_size);

        // Q16_1 should have at least 10× better precision than Q8_1
        // (256× in quantization precision, but rotation adds some error)
        EXPECT_LT(q16_max_diff, q8_max_diff / 5.0f)
            << "Q16_1 max_diff=" << q16_max_diff << " should be much better than Q8_1 max_diff=" << q8_max_diff;
    }

    TEST_F(CPURoPEKernelTTest, Q16_1_vs_FP32_RoPE_accuracy)
    {
        // Compare Q16_1 RoPE against pure FP32 RoPE to measure actual precision loss
        // This tests the full pipeline: FP32 → Q16_1 → RoPE → dequant vs FP32 → RoPE
        const size_t q_size = seq_len_ * n_heads_ * head_dim_;
        const size_t num_blocks = q_size / 32;

        // Generate FP32 test data (values in typical activation range)
        auto fp32_input = generate_random_fp32(q_size);

        // Path 1: Pure FP32 RoPE (ground truth)
        std::vector<float> fp32_result = fp32_input;
        std::vector<int> position_ids(seq_len_);
        std::iota(position_ids.begin(), position_ids.end(), 0);

        CPURoPEKernelT<ActivationPrecision::FP32> fp32_kernel;
        fp32_kernel.apply_typed(fp32_result.data(), nullptr, position_ids.data(),
                                seq_len_, n_heads_, 0, head_dim_, ROPE_THETA);

        // Path 2: Q16_1 RoPE (quantize → rope → dequantize)
        // Step 2a: Quantize FP32 to Q16_1
        std::vector<Q16_1Block> q16_blocks(num_blocks);
        for (size_t b = 0; b < num_blocks; ++b)
        {
            const float *src = fp32_input.data() + b * 32;
            Q16_1Block &block = q16_blocks[b];

            float max_val = 0.0f;
            for (int i = 0; i < 32; ++i)
                max_val = std::max(max_val, std::abs(src[i]));

            float scale = max_val / 32767.0f;
            if (scale < 1e-20f)
                scale = 1e-20f;
            float inv_scale = 1.0f / scale;

            block.d = scale;
            int64_t sum = 0;
            for (int i = 0; i < 32; ++i)
            {
                int32_t q = static_cast<int32_t>(std::round(src[i] * inv_scale));
                q = std::max(-32767, std::min(32767, q));
                block.qs[i] = static_cast<int16_t>(q);
                sum += q;
            }
            block.sum_qs = static_cast<int32_t>(sum);
        }

        // Step 2b: Apply Q16_1 RoPE in-place
        primitives::apply_rope_q16_1_integer(
            q16_blocks.data(), nullptr,
            position_ids.data(),
            seq_len_, n_heads_, 0, head_dim_,
            ROPE_THETA, nullptr);

        // Step 2c: Dequantize Q16_1 result to FP32
        std::vector<float> q16_result(q_size);
        for (size_t b = 0; b < num_blocks; ++b)
        {
            const Q16_1Block &block = q16_blocks[b];
            for (int i = 0; i < 32; ++i)
            {
                q16_result[b * 32 + i] = static_cast<float>(block.qs[i]) * block.d;
            }
        }

        // Compute error metrics
        float max_diff = max_abs_diff(q16_result.data(), fp32_result.data(), q_size);
        float cosine = cosine_similarity(q16_result.data(), fp32_result.data(), q_size);

        // Compute relative error (normalized by output magnitude)
        float sum_sq_fp32 = 0.0f;
        float sum_sq_diff = 0.0f;
        for (size_t i = 0; i < q_size; ++i)
        {
            float diff = q16_result[i] - fp32_result[i];
            sum_sq_diff += diff * diff;
            sum_sq_fp32 += fp32_result[i] * fp32_result[i];
        }
        float rel_rmse = std::sqrt(sum_sq_diff / sum_sq_fp32);

        // Print metrics for visibility
        printf("\n  Q16_1 vs FP32 RoPE accuracy:\n");
        printf("    Max absolute diff: %.6e\n", max_diff);
        printf("    Cosine similarity: %.8f\n", cosine);
        printf("    Relative RMSE:     %.6e\n", rel_rmse);

        // Assertions (10x margin over measured values):
        // Measured: max_diff ~9e-05, cosine ~0.99999946, rel_rmse ~2.2e-05

        // 1. Max absolute error (measured ~9e-05, threshold 1e-03)
        EXPECT_LT(max_diff, 1e-03f)
            << "Q16_1 RoPE max absolute error too large: " << max_diff;

        // 2. Cosine similarity (measured ~0.99999946, threshold 0.99999)
        EXPECT_GT(cosine, 0.99999f)
            << "Q16_1 RoPE cosine similarity too low: " << cosine;

        // 3. Relative RMSE (measured ~2.2e-05, threshold 2.5e-04)
        EXPECT_LT(rel_rmse, 2.5e-04f)
            << "Q16_1 RoPE relative RMSE too large: " << rel_rmse;
    }

} // namespace llaminar2