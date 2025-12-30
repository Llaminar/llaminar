/**
 * @file Test__CPURoPEKernelT_Q16_1_BlockSizes.cpp
 * @brief Unit tests for CPURoPEKernelT<Q16_1> with variable block sizes
 *
 * Tests the kernel-level API (apply_tensor) with all supported Q16 block sizes:
 * - 32-element blocks (Q16_1Block)
 * - 64-element blocks (Q16_1Block_64)
 * - 128-element blocks (Q16_1Block_128)
 *
 * Note: MLA architectures (DeepSeek V3, Kimi K2) should use separate
 * NOPE (128-dim) and ROPE (64-dim) tensors with their own scales.
 *
 * Each block size is tested for:
 * 1. Basic RoPE application (Q-only)
 * 2. Combined Q+K application
 * 3. Decode mode (seq_len=1)
 * 4. Padding handling (position_id=-1)
 * 5. Numerical accuracy vs FP32 reference
 *
 * @author David Sanftenberg
 * @date 2025
 */

#include <gtest/gtest.h>

#include "v2/kernels/cpu/ops/CPURoPEKernelT.h"
#include "v2/kernels/cpu/primitives/RoPEPrimitives.h"
#include "v2/tensors/BlockStructures.h"
#include "v2/tensors/Tensors.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace llaminar2
{
    namespace
    {
        /**
         * @brief Reference RoPE implementation for testing
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
            const int token_stride = n_heads * head_dim;

            for (int tok = 0; tok < seq_len; ++tok)
            {
                int position = position_ids ? position_ids[tok] : tok;
                if (position < 0)
                    continue;

                for (int h = 0; h < n_heads; ++h)
                {
                    float *head_data = data + tok * token_stride + h * head_dim;

                    for (int i = 0; i < half_dim; ++i)
                    {
                        float freq = 1.0f / std::pow(rope_theta, static_cast<float>(2 * i) / head_dim);
                        float angle = position * freq;
                        float cos_val = std::cos(angle);
                        float sin_val = std::sin(angle);

                        float x = head_data[i];
                        float y = head_data[i + half_dim];

                        head_data[i] = x * cos_val - y * sin_val;
                        head_data[i + half_dim] = x * sin_val + y * cos_val;
                    }
                }
            }
        }

        /**
         * @brief Generate random FP32 data with fixed seed for reproducibility
         */
        std::vector<float> generate_random_fp32(size_t count, int seed = 42)
        {
            std::mt19937 rng(seed);
            std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
            std::vector<float> data(count);
            for (auto &v : data)
                v = dist(rng);
            return data;
        }

        /**
         * @brief Compute cosine similarity between two vectors
         */
        float cosine_similarity(const float *a, const float *b, size_t n)
        {
            double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
            for (size_t i = 0; i < n; ++i)
            {
                dot += a[i] * b[i];
                norm_a += a[i] * a[i];
                norm_b += b[i] * b[i];
            }
            if (norm_a < 1e-12 || norm_b < 1e-12)
                return 0.0f;
            return static_cast<float>(dot / (std::sqrt(norm_a) * std::sqrt(norm_b)));
        }

        /**
         * @brief Compute max absolute difference
         */
        float max_abs_diff(const float *a, const float *b, size_t n)
        {
            float max_diff = 0.0f;
            for (size_t i = 0; i < n; ++i)
                max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
            return max_diff;
        }

    } // namespace

    // =======================================================================
    // Test Fixture
    // =======================================================================

    class CPURoPEKernelT_Q16_1_BlockSizesTest : public ::testing::Test
    {
    protected:
        static constexpr float ROPE_THETA = 10000.0f;
        static constexpr float COSINE_THRESHOLD = 0.998f; // Q16_1 should be very accurate

        void SetUp() override {}

        /**
         * @brief Test apply_tensor with a specific block size
         */
        template <Q16BlockSize BlockSize>
        void testApplyTensor_QOnly(int seq_len, int n_heads, int head_dim)
        {
            const size_t block_size_val = static_cast<size_t>(BlockSize);
            ASSERT_EQ(head_dim % block_size_val, 0)
                << "head_dim must be divisible by block size";

            const size_t q_size = seq_len * n_heads * head_dim;

            // Generate FP32 test data
            auto fp32_q = generate_random_fp32(q_size);

            // Create Q16_1Tensor with specified block size
            std::vector<size_t> shape = {static_cast<size_t>(seq_len),
                                         static_cast<size_t>(n_heads * head_dim)};
            Q16_1Tensor q_tensor(shape, BlockSize);

            EXPECT_EQ(q_tensor.q16_block_size(), BlockSize)
                << "Tensor should have correct block size";

            // Quantize into tensor
            q_tensor.copyFrom_fp32(fp32_q.data());

            // Get reference FP32 result
            std::vector<float> fp32_expected = fp32_q;
            std::vector<int> position_ids(seq_len);
            std::iota(position_ids.begin(), position_ids.end(), 0);
            reference_rope(fp32_expected.data(), seq_len, n_heads, head_dim,
                           position_ids.data(), ROPE_THETA);

            // Apply RoPE via kernel
            CPURoPEKernelT<ActivationPrecision::Q16_1> kernel;
            bool success = kernel.apply_tensor(
                &q_tensor, nullptr,
                position_ids.data(),
                seq_len, n_heads, 0, head_dim,
                ROPE_THETA);
            ASSERT_TRUE(success);

            // Dequantize and compare
            std::vector<float> fp32_result(q_size);
            q_tensor.to_fp32(fp32_result.data());

            float cosine = cosine_similarity(fp32_result.data(), fp32_expected.data(), q_size);
            EXPECT_GT(cosine, COSINE_THRESHOLD)
                << "BlockSize=" << block_size_val
                << " cosine similarity: " << cosine;

            float max_diff = max_abs_diff(fp32_result.data(), fp32_expected.data(), q_size);
            EXPECT_LT(max_diff, 0.1f)
                << "BlockSize=" << block_size_val
                << " max diff: " << max_diff;
        }

        /**
         * @brief Test apply_tensor with Q+K tensors
         */
        template <Q16BlockSize BlockSize>
        void testApplyTensor_QK(int seq_len, int n_heads, int n_kv_heads, int head_dim)
        {
            const size_t block_size_val = static_cast<size_t>(BlockSize);
            ASSERT_EQ(head_dim % block_size_val, 0)
                << "head_dim must be divisible by block size";

            const size_t q_size = seq_len * n_heads * head_dim;
            const size_t k_size = seq_len * n_kv_heads * head_dim;

            // Generate FP32 test data
            auto fp32_q = generate_random_fp32(q_size, 42);
            auto fp32_k = generate_random_fp32(k_size, 123);

            // Create tensors with specified block size
            std::vector<size_t> q_shape = {static_cast<size_t>(seq_len),
                                           static_cast<size_t>(n_heads * head_dim)};
            std::vector<size_t> k_shape = {static_cast<size_t>(seq_len),
                                           static_cast<size_t>(n_kv_heads * head_dim)};
            Q16_1Tensor q_tensor(q_shape, BlockSize);
            Q16_1Tensor k_tensor(k_shape, BlockSize);

            // Quantize
            q_tensor.copyFrom_fp32(fp32_q.data());
            k_tensor.copyFrom_fp32(fp32_k.data());

            // Get reference results
            std::vector<float> fp32_q_expected = fp32_q;
            std::vector<float> fp32_k_expected = fp32_k;
            std::vector<int> position_ids(seq_len);
            std::iota(position_ids.begin(), position_ids.end(), 0);
            reference_rope(fp32_q_expected.data(), seq_len, n_heads, head_dim,
                           position_ids.data(), ROPE_THETA);
            reference_rope(fp32_k_expected.data(), seq_len, n_kv_heads, head_dim,
                           position_ids.data(), ROPE_THETA);

            // Apply RoPE via kernel
            CPURoPEKernelT<ActivationPrecision::Q16_1> kernel;
            bool success = kernel.apply_tensor(
                &q_tensor, &k_tensor,
                position_ids.data(),
                seq_len, n_heads, n_kv_heads, head_dim,
                ROPE_THETA);
            ASSERT_TRUE(success);

            // Dequantize and compare Q
            std::vector<float> fp32_q_result(q_size);
            q_tensor.to_fp32(fp32_q_result.data());

            float q_cosine = cosine_similarity(fp32_q_result.data(), fp32_q_expected.data(), q_size);
            EXPECT_GT(q_cosine, COSINE_THRESHOLD)
                << "Q cosine (BlockSize=" << block_size_val << "): " << q_cosine;

            // Dequantize and compare K
            std::vector<float> fp32_k_result(k_size);
            k_tensor.to_fp32(fp32_k_result.data());

            float k_cosine = cosine_similarity(fp32_k_result.data(), fp32_k_expected.data(), k_size);
            EXPECT_GT(k_cosine, COSINE_THRESHOLD)
                << "K cosine (BlockSize=" << block_size_val << "): " << k_cosine;
        }

        /**
         * @brief Test decode mode (seq_len=1)
         */
        template <Q16BlockSize BlockSize>
        void testDecodeMode(int n_heads, int head_dim, int position)
        {
            const size_t block_size_val = static_cast<size_t>(BlockSize);
            ASSERT_EQ(head_dim % block_size_val, 0);

            const int seq_len = 1;
            const size_t q_size = n_heads * head_dim;

            auto fp32_q = generate_random_fp32(q_size, position); // Different seed per position

            std::vector<size_t> shape = {1, static_cast<size_t>(n_heads * head_dim)};
            Q16_1Tensor q_tensor(shape, BlockSize);
            q_tensor.copyFrom_fp32(fp32_q.data());

            // Reference
            std::vector<float> fp32_expected = fp32_q;
            std::vector<int> position_ids = {position};
            reference_rope(fp32_expected.data(), 1, n_heads, head_dim,
                           position_ids.data(), ROPE_THETA);

            // Apply RoPE
            CPURoPEKernelT<ActivationPrecision::Q16_1> kernel;
            bool success = kernel.apply_tensor(
                &q_tensor, nullptr,
                position_ids.data(),
                1, n_heads, 0, head_dim,
                ROPE_THETA);
            ASSERT_TRUE(success);

            // Compare
            std::vector<float> fp32_result(q_size);
            q_tensor.to_fp32(fp32_result.data());

            float cosine = cosine_similarity(fp32_result.data(), fp32_expected.data(), q_size);
            EXPECT_GT(cosine, COSINE_THRESHOLD)
                << "Decode position=" << position << " BlockSize=" << block_size_val;
        }

        /**
         * @brief Test padding handling (position_id=-1 should be skipped)
         */
        template <Q16BlockSize BlockSize>
        void testPaddingHandling(int n_heads, int head_dim)
        {
            const size_t block_size_val = static_cast<size_t>(BlockSize);
            ASSERT_EQ(head_dim % block_size_val, 0);

            const int seq_len = 4;
            const size_t q_size = seq_len * n_heads * head_dim;

            auto fp32_q = generate_random_fp32(q_size, 999);

            std::vector<size_t> shape = {static_cast<size_t>(seq_len),
                                         static_cast<size_t>(n_heads * head_dim)};
            Q16_1Tensor q_tensor(shape, BlockSize);
            q_tensor.copyFrom_fp32(fp32_q.data());

            // position_ids with padding: [0, 1, -1, -1]
            std::vector<int> position_ids = {0, 1, -1, -1};

            // Reference - only apply to positions 0 and 1
            std::vector<float> fp32_expected = fp32_q;
            reference_rope(fp32_expected.data(), seq_len, n_heads, head_dim,
                           position_ids.data(), ROPE_THETA);

            // Apply RoPE
            CPURoPEKernelT<ActivationPrecision::Q16_1> kernel;
            bool success = kernel.apply_tensor(
                &q_tensor, nullptr,
                position_ids.data(),
                seq_len, n_heads, 0, head_dim,
                ROPE_THETA);
            ASSERT_TRUE(success);

            // Compare
            std::vector<float> fp32_result(q_size);
            q_tensor.to_fp32(fp32_result.data());

            // Full result should match (including unchanged padding tokens)
            float cosine = cosine_similarity(fp32_result.data(), fp32_expected.data(), q_size);
            EXPECT_GT(cosine, COSINE_THRESHOLD)
                << "Padding test BlockSize=" << block_size_val;
        }
    };

    // =======================================================================
    // Block Size 32 Tests (Q16_1Block)
    // =======================================================================

    TEST_F(CPURoPEKernelT_Q16_1_BlockSizesTest, Block32_QOnly_Basic)
    {
        testApplyTensor_QOnly<Q16BlockSize::BLOCK_32>(2, 4, 64);
    }

    TEST_F(CPURoPEKernelT_Q16_1_BlockSizesTest, Block32_QOnly_LargeHeadDim)
    {
        testApplyTensor_QOnly<Q16BlockSize::BLOCK_32>(2, 4, 128);
    }

    TEST_F(CPURoPEKernelT_Q16_1_BlockSizesTest, Block32_QK)
    {
        testApplyTensor_QK<Q16BlockSize::BLOCK_32>(2, 4, 2, 64);
    }

    TEST_F(CPURoPEKernelT_Q16_1_BlockSizesTest, Block32_DecodeMode)
    {
        testDecodeMode<Q16BlockSize::BLOCK_32>(4, 64, 0);
        testDecodeMode<Q16BlockSize::BLOCK_32>(4, 64, 10);
        testDecodeMode<Q16BlockSize::BLOCK_32>(4, 64, 100);
    }

    TEST_F(CPURoPEKernelT_Q16_1_BlockSizesTest, Block32_Padding)
    {
        testPaddingHandling<Q16BlockSize::BLOCK_32>(4, 64);
    }

    // =======================================================================
    // Block Size 64 Tests (Q16_1Block_64)
    // =======================================================================

    TEST_F(CPURoPEKernelT_Q16_1_BlockSizesTest, Block64_QOnly_Basic)
    {
        testApplyTensor_QOnly<Q16BlockSize::BLOCK_64>(2, 4, 64);
    }

    TEST_F(CPURoPEKernelT_Q16_1_BlockSizesTest, Block64_QOnly_LargeHeadDim)
    {
        testApplyTensor_QOnly<Q16BlockSize::BLOCK_64>(2, 4, 128);
    }

    TEST_F(CPURoPEKernelT_Q16_1_BlockSizesTest, Block64_QK)
    {
        testApplyTensor_QK<Q16BlockSize::BLOCK_64>(2, 4, 2, 64);
    }

    TEST_F(CPURoPEKernelT_Q16_1_BlockSizesTest, Block64_DecodeMode)
    {
        testDecodeMode<Q16BlockSize::BLOCK_64>(4, 64, 0);
        testDecodeMode<Q16BlockSize::BLOCK_64>(4, 64, 10);
        testDecodeMode<Q16BlockSize::BLOCK_64>(4, 64, 100);
    }

    TEST_F(CPURoPEKernelT_Q16_1_BlockSizesTest, Block64_Padding)
    {
        testPaddingHandling<Q16BlockSize::BLOCK_64>(4, 64);
    }

    // =======================================================================
    // Block Size 128 Tests (Q16_1Block_128)
    // =======================================================================

    TEST_F(CPURoPEKernelT_Q16_1_BlockSizesTest, Block128_QOnly_Basic)
    {
        testApplyTensor_QOnly<Q16BlockSize::BLOCK_128>(2, 4, 128);
    }

    TEST_F(CPURoPEKernelT_Q16_1_BlockSizesTest, Block128_QOnly_LargeHeadDim)
    {
        testApplyTensor_QOnly<Q16BlockSize::BLOCK_128>(2, 4, 256);
    }

    TEST_F(CPURoPEKernelT_Q16_1_BlockSizesTest, Block128_QK)
    {
        testApplyTensor_QK<Q16BlockSize::BLOCK_128>(2, 4, 2, 128);
    }

    TEST_F(CPURoPEKernelT_Q16_1_BlockSizesTest, Block128_DecodeMode)
    {
        testDecodeMode<Q16BlockSize::BLOCK_128>(4, 128, 0);
        testDecodeMode<Q16BlockSize::BLOCK_128>(4, 128, 10);
        testDecodeMode<Q16BlockSize::BLOCK_128>(4, 128, 100);
    }

    TEST_F(CPURoPEKernelT_Q16_1_BlockSizesTest, Block128_Padding)
    {
        testPaddingHandling<Q16BlockSize::BLOCK_128>(4, 128);
    }

    // =======================================================================
    // Cross-block-size Tests
    // =======================================================================

    TEST_F(CPURoPEKernelT_Q16_1_BlockSizesTest, MismatchedBlockSizes_Fails)
    {
        // Q and K must have the same block size
        const int seq_len = 2;
        const int n_heads = 4;
        const int n_kv_heads = 2;
        const int head_dim = 64;

        std::vector<size_t> q_shape = {2, static_cast<size_t>(n_heads * head_dim)};
        std::vector<size_t> k_shape = {2, static_cast<size_t>(n_kv_heads * head_dim)};

        Q16_1Tensor q_tensor(q_shape, Q16BlockSize::BLOCK_32);
        Q16_1Tensor k_tensor(k_shape, Q16BlockSize::BLOCK_64);

        std::vector<int> position_ids = {0, 1};

        CPURoPEKernelT<ActivationPrecision::Q16_1> kernel;
        bool success = kernel.apply_tensor(
            &q_tensor, &k_tensor,
            position_ids.data(),
            seq_len, n_heads, n_kv_heads, head_dim,
            ROPE_THETA);

        EXPECT_FALSE(success) << "Should fail when Q and K have different block sizes";
    }

    TEST_F(CPURoPEKernelT_Q16_1_BlockSizesTest, InvalidHeadDim_Fails)
    {
        // head_dim must be divisible by block size
        // For BLOCK_64, head_dim=48 should fail
        std::vector<size_t> shape = {2, 192}; // 4 heads * 48 dim
        Q16_1Tensor q_tensor(shape, Q16BlockSize::BLOCK_64);

        std::vector<int> position_ids = {0, 1};

        CPURoPEKernelT<ActivationPrecision::Q16_1> kernel;
        bool success = kernel.apply_tensor(
            &q_tensor, nullptr,
            position_ids.data(),
            2, 4, 0, 48, // head_dim=48 not divisible by 64
            ROPE_THETA);

        EXPECT_FALSE(success) << "Should fail when head_dim not divisible by block size";
    }

    TEST_F(CPURoPEKernelT_Q16_1_BlockSizesTest, NullQ_Fails)
    {
        std::vector<int> position_ids = {0};
        CPURoPEKernelT<ActivationPrecision::Q16_1> kernel;
        bool success = kernel.apply_tensor(
            nullptr, nullptr,
            position_ids.data(),
            1, 4, 0, 64,
            ROPE_THETA);

        EXPECT_FALSE(success) << "Should fail with null Q tensor";
    }

    // =======================================================================
    // Precision Tests - Compare different block sizes
    // =======================================================================

    TEST_F(CPURoPEKernelT_Q16_1_BlockSizesTest, AllBlockSizes_SimilarAccuracy)
    {
        // All block sizes should achieve similar accuracy for same data
        // (within reasonable tolerance, as larger blocks may have slightly different quantization)
        const int seq_len = 2;
        const int n_heads = 4;
        const int head_dim = 384; // Divisible by all block sizes (32, 64, 128, 192)
        const size_t q_size = seq_len * n_heads * head_dim;

        auto fp32_q = generate_random_fp32(q_size, 12345);

        // Reference
        std::vector<float> fp32_expected = fp32_q;
        std::vector<int> position_ids(seq_len);
        std::iota(position_ids.begin(), position_ids.end(), 0);
        reference_rope(fp32_expected.data(), seq_len, n_heads, head_dim,
                       position_ids.data(), ROPE_THETA);

        // Test each block size
        std::vector<Q16BlockSize> block_sizes = {
            Q16BlockSize::BLOCK_32,
            Q16BlockSize::BLOCK_64,
            Q16BlockSize::BLOCK_128};

        for (auto bs : block_sizes)
        {
            std::vector<size_t> shape = {static_cast<size_t>(seq_len),
                                         static_cast<size_t>(n_heads * head_dim)};
            Q16_1Tensor q_tensor(shape, bs);
            q_tensor.copyFrom_fp32(fp32_q.data());

            CPURoPEKernelT<ActivationPrecision::Q16_1> kernel;
            bool success = kernel.apply_tensor(
                &q_tensor, nullptr,
                position_ids.data(),
                seq_len, n_heads, 0, head_dim,
                ROPE_THETA);
            ASSERT_TRUE(success) << "BlockSize=" << static_cast<size_t>(bs);

            std::vector<float> fp32_result(q_size);
            q_tensor.to_fp32(fp32_result.data());

            float cosine = cosine_similarity(fp32_result.data(), fp32_expected.data(), q_size);
            EXPECT_GT(cosine, 0.97f)
                << "BlockSize=" << static_cast<size_t>(bs) << " cosine=" << cosine;
        }
    }

} // namespace llaminar2
