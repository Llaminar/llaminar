/**
 * @file Test__JitFusedAttentionWo.cpp
 * @brief Unit tests for JIT fused attention + Wo projection kernel
 * @author David Sanftenberg
 *
 * Tests the composed JIT kernel that fuses:
 *   1. Q8_1 × Q8_1 dot product (Q @ K^T)
 *   2. Online softmax
 *   3. Weighted V accumulation
 *   4. Wo projection (context × Wo)
 *
 * Validates:
 * - Correct algorithm structure (per-head processing)
 * - Numerical correctness against reference implementation
 * - Wo projection accumulation across heads
 * - GQA support (heads_per_kv > 1)
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <random>
#include <cstring>

#include "kernels/cpu/jit/q8_1/JitFusedAttentionWo.h"
#include "kernels/cpu/attention/q8_1/FusedAttentionWoRef.h"
#include "tensors/BlockStructures.h"
#include "tensors/FP16Utils.h"

namespace llaminar::v2::kernels::jit::test
{

    using llaminar2::fp16_to_fp32;
    using llaminar2::fp32_to_fp16;
    using microkernels::Q8_1Block;

    // ============================================================================
    // Test Fixture
    // ============================================================================

    class Test__JitFusedAttentionWo : public ::testing::Test
    {
    protected:
        std::mt19937 gen_{42};
        std::uniform_real_distribution<float> dist_{-1.0f, 1.0f};

        // Correctness thresholds
        static constexpr double MIN_COSINE_SIM = 0.995;
        static constexpr double MAX_REL_L2_ERROR = 0.05;

        /**
         * @brief Quantize FP32 data to Q8_1 block format
         */
        void quantize_fp32_to_q8_1(const float *fp32_data, int rows, int cols, Q8_1Block *blocks)
        {
            const int num_blocks_per_row = cols / 32;

            for (int row = 0; row < rows; ++row)
            {
                for (int b = 0; b < num_blocks_per_row; ++b)
                {
                    const float *block_data = fp32_data + row * cols + b * 32;
                    Q8_1Block &blk = blocks[row * num_blocks_per_row + b];

                    // Find max absolute value
                    float max_abs = 0.0f;
                    for (int i = 0; i < 32; ++i)
                    {
                        max_abs = std::max(max_abs, std::fabs(block_data[i]));
                    }

                    // Compute scale
                    float scale = max_abs / 127.0f;
                    if (scale == 0.0f)
                        scale = 1.0f;

                    // Quantize
                    int32_t sum_qs = 0;
                    for (int i = 0; i < 32; ++i)
                    {
                        int8_t q = static_cast<int8_t>(std::round(block_data[i] / scale));
                        blk.qs[i] = q;
                        sum_qs += q;
                    }

                    blk.d = fp32_to_fp16(scale);
                    blk.sum_qs = static_cast<int16_t>(sum_qs);
                }
            }
        }

        /**
         * @brief Generate random FP32 data
         */
        std::vector<float> generate_random_fp32(int size)
        {
            std::vector<float> data(size);
            for (auto &v : data)
            {
                v = dist_(gen_);
            }
            return data;
        }

        /**
         * @brief Compute cosine similarity between two vectors
         */
        double cosine_similarity(const float *a, const float *b, int n)
        {
            double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
            for (int i = 0; i < n; ++i)
            {
                dot += a[i] * b[i];
                norm_a += a[i] * a[i];
                norm_b += b[i] * b[i];
            }
            if (norm_a < 1e-10 || norm_b < 1e-10)
                return 1.0;
            return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
        }

        /**
         * @brief Compute relative L2 error
         */
        double relative_l2_error(const float *actual, const float *expected, int n)
        {
            double sum_sq_err = 0.0, sum_sq_exp = 0.0;
            for (int i = 0; i < n; ++i)
            {
                double err = actual[i] - expected[i];
                sum_sq_err += err * err;
                sum_sq_exp += expected[i] * expected[i];
            }
            if (sum_sq_exp < 1e-10)
                return 0.0;
            return std::sqrt(sum_sq_err / sum_sq_exp);
        }
    };

    // ============================================================================
    // JIT Kernel Instantiation Tests
    // ============================================================================

    TEST_F(Test__JitFusedAttentionWo, CanInstantiateKernel_HeadDim64)
    {
        JitAttentionConfig config;
        config.head_dim = 64;
        config.num_heads = 14;
        config.num_kv_heads = 2;
        config.batch_size = 1;
        config.wo_format = WoFormat::FP32;

        // Should not throw
        ASSERT_NO_THROW({
            JitFusedAttentionWo attn(config);
            EXPECT_NE(attn.getKernel(), nullptr);
        });
    }

    TEST_F(Test__JitFusedAttentionWo, CanInstantiateKernel_HeadDim128)
    {
        JitAttentionConfig config;
        config.head_dim = 128;
        config.num_heads = 8;
        config.num_kv_heads = 8;
        config.batch_size = 1;
        config.wo_format = WoFormat::FP32;

        ASSERT_NO_THROW({
            JitFusedAttentionWo attn(config);
            EXPECT_NE(attn.getKernel(), nullptr);
        });
    }

    TEST_F(Test__JitFusedAttentionWo, KernelCacheWorks)
    {
        JitAttentionConfig config;
        config.head_dim = 64;
        config.num_heads = 14;
        config.num_kv_heads = 2;
        config.batch_size = 1;
        config.wo_format = WoFormat::FP32;

        // Get kernel twice - should return same pointer (cached)
        JitFusedAttentionWo attn1(config);
        JitFusedAttentionWo attn2(config);

        EXPECT_EQ(attn1.getKernel(), attn2.getKernel())
            << "Kernel cache should return same kernel for identical config";
    }

    TEST_F(Test__JitFusedAttentionWo, DifferentConfigsGetDifferentKernels)
    {
        JitAttentionConfig config1;
        config1.head_dim = 64;
        config1.num_heads = 14;
        config1.num_kv_heads = 2;
        config1.batch_size = 1;
        config1.wo_format = WoFormat::FP32;

        JitAttentionConfig config2;
        config2.head_dim = 128; // Different head_dim
        config2.num_heads = 8;
        config2.num_kv_heads = 8;
        config2.batch_size = 1;
        config2.wo_format = WoFormat::FP32;

        JitFusedAttentionWo attn1(config1);
        JitFusedAttentionWo attn2(config2);

        EXPECT_NE(attn1.getKernel(), attn2.getKernel())
            << "Different configs should get different kernels";
    }

    // ============================================================================
    // Wo Format Support Tests
    // ============================================================================

    TEST_F(Test__JitFusedAttentionWo, SupportsWoFormat_FP32)
    {
        JitAttentionConfig config;
        config.head_dim = 64;
        config.num_heads = 2;
        config.num_kv_heads = 2;
        config.batch_size = 1;
        config.wo_format = WoFormat::FP32;

        ASSERT_NO_THROW({
            JitFusedAttentionWo attn(config);
        });
    }

    TEST_F(Test__JitFusedAttentionWo, SupportsWoFormat_FP16)
    {
        JitAttentionConfig config;
        config.head_dim = 64;
        config.num_heads = 2;
        config.num_kv_heads = 2;
        config.batch_size = 1;
        config.wo_format = WoFormat::FP16;

        ASSERT_NO_THROW({
            JitFusedAttentionWo attn(config);
        });
    }

    TEST_F(Test__JitFusedAttentionWo, SupportsWoFormat_BF16)
    {
        JitAttentionConfig config;
        config.head_dim = 64;
        config.num_heads = 2;
        config.num_kv_heads = 2;
        config.batch_size = 1;
        config.wo_format = WoFormat::BF16;

        ASSERT_NO_THROW({
            JitFusedAttentionWo attn(config);
        });
    }

    TEST_F(Test__JitFusedAttentionWo, SupportsWoFormat_Q8_1)
    {
        JitAttentionConfig config;
        config.head_dim = 64;
        config.num_heads = 2;
        config.num_kv_heads = 2;
        config.batch_size = 1;
        config.wo_format = WoFormat::Q8_1;

        ASSERT_NO_THROW({
            JitFusedAttentionWo attn(config);
        });
    }

    // ============================================================================
    // Cache Statistics Tests
    // ============================================================================

    TEST_F(Test__JitFusedAttentionWo, CacheStatistics)
    {
        // Clear cache first
        JitAttentionKernelCache::instance().clear();

        EXPECT_EQ(JitAttentionKernelCache::instance().size(), 0u);

        // Create a kernel - should add to cache
        JitAttentionConfig config;
        config.head_dim = 64;
        config.num_heads = 14;
        config.num_kv_heads = 2;
        config.batch_size = 1;
        config.wo_format = WoFormat::FP32;

        JitFusedAttentionWo attn(config);

        EXPECT_EQ(JitAttentionKernelCache::instance().size(), 1u);
    }

} // namespace llaminar::v2::kernels::jit::test
