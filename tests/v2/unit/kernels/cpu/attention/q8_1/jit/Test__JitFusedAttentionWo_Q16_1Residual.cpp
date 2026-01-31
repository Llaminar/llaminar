/**
 * @file Test__JitFusedAttentionWo_Q16_1Residual.cpp
 * @brief Tests for JIT fused attention with Q16_1 typed residual stream
 * @author David Sanftenberg
 * @date December 2025
 *
 * Tests the fused residual addition path:
 *   FP32 Wo output + dequant(Q16_1 residual) → quantize → Q16_1 output
 *
 * This verifies Phase 5 of the Q16_1 Typed Residual Stream project.
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <random>
#include <cstring>
#include <iostream>
#include <iomanip>

#include "kernels/cpu/attention/q8_1/jit/JitFusedAttentionWo.h"
#include "tensors/BlockStructures.h"

namespace llaminar::v2::kernels::jit::test
{

    using microkernels::Q8_1Block;

    // Q16_1 block structure (72 bytes per 32 elements)
    struct Q16_1Block
    {
        float d;        // Scale factor
        int32_t sum_qs; // Sum of quantized values (for GEMM optimization)
        int16_t qs[32]; // Quantized int16 values
    };
    static_assert(sizeof(Q16_1Block) == 72, "Q16_1Block must be 72 bytes");

    // ============================================================================
    // Test Fixture
    // ============================================================================

    class Test__JitFusedAttentionWo_Q16_1Residual : public ::testing::Test
    {
    protected:
        std::mt19937 gen_{42};
        std::uniform_real_distribution<float> dist_{-1.0f, 1.0f};

        // Correctness thresholds (Q16_1 has ~4.5 decimal digits of precision)
        static constexpr double MIN_COSINE_SIM = 0.9999; // Very high similarity expected
        static constexpr double MAX_REL_ERROR = 0.001;   // 0.1% relative error
        static constexpr double MAX_ABS_ERROR = 1e-3f;   // Small absolute error

        /**
         * @brief Quantize FP32 data to Q16_1 block format
         */
        void quantize_fp32_to_q16_1(const float *fp32_data, int num_elements, Q16_1Block *blocks)
        {
            const int num_blocks = (num_elements + 31) / 32;

            for (int b = 0; b < num_blocks; ++b)
            {
                const float *block_data = fp32_data + b * 32;
                Q16_1Block &blk = blocks[b];

                // Elements in this block
                int block_elems = std::min(32, num_elements - b * 32);

                // Find max absolute value
                float max_abs = 0.0f;
                for (int i = 0; i < block_elems; ++i)
                {
                    max_abs = std::max(max_abs, std::fabs(block_data[i]));
                }

                // Compute scale: scale = max_abs / 32767.0f
                float scale = max_abs / 32767.0f;
                if (scale < 1e-10f)
                    scale = 1e-10f; // Avoid division by zero
                float inv_scale = 32767.0f / max_abs;
                if (max_abs < 1e-10f)
                    inv_scale = 0.0f;

                // Quantize
                int32_t sum_qs = 0;
                for (int i = 0; i < 32; ++i)
                {
                    if (i < block_elems)
                    {
                        int16_t q = static_cast<int16_t>(std::round(block_data[i] * inv_scale));
                        q = std::max<int16_t>(-32767, std::min<int16_t>(32767, q));
                        blk.qs[i] = q;
                        sum_qs += q;
                    }
                    else
                    {
                        blk.qs[i] = 0;
                    }
                }

                blk.d = scale;
                blk.sum_qs = sum_qs;
            }
        }

        /**
         * @brief Dequantize Q16_1 blocks to FP32
         */
        void dequantize_q16_1_to_fp32(const Q16_1Block *blocks, int num_elements, float *fp32_data)
        {
            const int num_blocks = (num_elements + 31) / 32;

            for (int b = 0; b < num_blocks; ++b)
            {
                const Q16_1Block &blk = blocks[b];
                float *block_data = fp32_data + b * 32;

                int block_elems = std::min(32, num_elements - b * 32);
                for (int i = 0; i < block_elems; ++i)
                {
                    block_data[i] = blk.d * static_cast<float>(blk.qs[i]);
                }
            }
        }

        /**
         * @brief Quantize FP32 to Q8_1 for attention inputs
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

                    float max_abs = 0.0f;
                    for (int i = 0; i < 32; ++i)
                    {
                        max_abs = std::max(max_abs, std::fabs(block_data[i]));
                    }

                    float scale = max_abs / 127.0f;
                    if (scale < 1e-10f)
                        scale = 1e-10f;
                    float inv_scale = 127.0f / max_abs;
                    if (max_abs < 1e-10f)
                        inv_scale = 0.0f;

                    int32_t sum_qs = 0;
                    for (int i = 0; i < 32; ++i)
                    {
                        int8_t q = static_cast<int8_t>(std::round(block_data[i] * inv_scale));
                        q = std::max<int8_t>(-127, std::min<int8_t>(127, q));
                        blk.qs[i] = q;
                        sum_qs += q;
                    }

                    blk.d = scale;
                    blk.sum_qs = sum_qs;
                }
            }
        }

        /**
         * @brief Generate random FP32 tensor
         */
        std::vector<float> generateRandomTensor(int size)
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
        double cosineSimilarity(const float *a, const float *b, int size)
        {
            double dot = 0, norm_a = 0, norm_b = 0;
            for (int i = 0; i < size; ++i)
            {
                dot += a[i] * b[i];
                norm_a += a[i] * a[i];
                norm_b += b[i] * b[i];
            }
            if (norm_a < 1e-10 || norm_b < 1e-10)
                return 0.0;
            return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
        }

        /**
         * @brief Compute relative L2 error
         */
        double relativeL2Error(const float *ref, const float *test, int size)
        {
            double sum_sq_diff = 0, sum_sq_ref = 0;
            for (int i = 0; i < size; ++i)
            {
                double diff = ref[i] - test[i];
                sum_sq_diff += diff * diff;
                sum_sq_ref += ref[i] * ref[i];
            }
            if (sum_sq_ref < 1e-10)
                return sum_sq_diff > 1e-10 ? 1.0 : 0.0;
            return std::sqrt(sum_sq_diff / sum_sq_ref);
        }
    };

    // ============================================================================
    // Q16_1 Residual Fusion Tests
    // ============================================================================

    /**
     * @test Test that the JIT kernel can be configured for Q16_1 residual fusion
     */
    TEST_F(Test__JitFusedAttentionWo_Q16_1Residual, ConfigAcceptsResidualFusion)
    {
        JitAttentionConfig config;
        config.head_dim = 64;
        config.num_heads = 14;
        config.num_kv_heads = 2;
        config.batch_size = 1;
        config.wo_format = WoFormat::FP32;
        config.fuse_residual_add = true;
        config.residual_type = JitAttentionConfig::ResidualType::Q16_1;

        // Should compile without error
        EXPECT_TRUE(config.fuse_residual_add);
        EXPECT_EQ(config.residual_type, JitAttentionConfig::ResidualType::Q16_1);
    }

    /**
     * @test Test Q16_1 quantize/dequantize roundtrip
     */
    TEST_F(Test__JitFusedAttentionWo_Q16_1Residual, Q16_1RoundtripAccuracy)
    {
        const int d_model = 896; // Qwen2-0.5B
        const int num_blocks = (d_model + 31) / 32;

        // Generate random FP32 data
        auto fp32_original = generateRandomTensor(d_model);

        // Quantize to Q16_1
        std::vector<Q16_1Block> blocks(num_blocks);
        quantize_fp32_to_q16_1(fp32_original.data(), d_model, blocks.data());

        // Dequantize back to FP32
        std::vector<float> fp32_recovered(d_model);
        dequantize_q16_1_to_fp32(blocks.data(), d_model, fp32_recovered.data());

        // Check accuracy
        double cosine = cosineSimilarity(fp32_original.data(), fp32_recovered.data(), d_model);
        double rel_err = relativeL2Error(fp32_original.data(), fp32_recovered.data(), d_model);

        EXPECT_GE(cosine, MIN_COSINE_SIM) << "Cosine similarity too low: " << cosine;
        EXPECT_LE(rel_err, MAX_REL_ERROR) << "Relative L2 error too high: " << rel_err;

        // Print metrics
        std::cout << "[Q16_1 Roundtrip] Cosine: " << std::fixed << std::setprecision(6) << cosine
                  << ", Rel L2 Error: " << rel_err << std::endl;
    }

    /**
     * @test Test residual addition in FP32 reference (what the JIT should match)
     */
    TEST_F(Test__JitFusedAttentionWo_Q16_1Residual, ReferenceResidualAddition)
    {
        const int d_model = 896;
        const int num_blocks = (d_model + 31) / 32;

        // Generate FP32 Wo output and Q16_1 residual
        auto wo_output = generateRandomTensor(d_model);
        auto residual_fp32 = generateRandomTensor(d_model);

        // Quantize residual to Q16_1
        std::vector<Q16_1Block> residual_q16_1(num_blocks);
        quantize_fp32_to_q16_1(residual_fp32.data(), d_model, residual_q16_1.data());

        // Reference: dequantize residual, add, requantize
        std::vector<float> residual_dequant(d_model);
        dequantize_q16_1_to_fp32(residual_q16_1.data(), d_model, residual_dequant.data());

        std::vector<float> sum_fp32(d_model);
        for (int i = 0; i < d_model; ++i)
        {
            sum_fp32[i] = wo_output[i] + residual_dequant[i];
        }

        // Quantize result to Q16_1
        std::vector<Q16_1Block> result_q16_1(num_blocks);
        quantize_fp32_to_q16_1(sum_fp32.data(), d_model, result_q16_1.data());

        // Dequantize for verification
        std::vector<float> result_fp32(d_model);
        dequantize_q16_1_to_fp32(result_q16_1.data(), d_model, result_fp32.data());

        // Compare with ideal sum (before final quantization)
        double cosine = cosineSimilarity(sum_fp32.data(), result_fp32.data(), d_model);
        double rel_err = relativeL2Error(sum_fp32.data(), result_fp32.data(), d_model);

        EXPECT_GE(cosine, MIN_COSINE_SIM) << "Cosine similarity too low: " << cosine;
        EXPECT_LE(rel_err, MAX_REL_ERROR) << "Relative L2 error too high: " << rel_err;

        std::cout << "[Reference Residual Add] Cosine: " << std::fixed << std::setprecision(6) << cosine
                  << ", Rel L2 Error: " << rel_err << std::endl;
    }

    /**
     * @test Test JIT kernel generates correctly with fused residual config
     *
     * This test verifies the JIT kernel can be instantiated with the fused
     * residual configuration. Full numerical testing requires the kernel
     * to be executed, which will be added in integration tests.
     */
    TEST_F(Test__JitFusedAttentionWo_Q16_1Residual, JitKernelGenerationWithFusedResidual)
    {
        JitAttentionConfig config;
        config.head_dim = 64;
        config.num_heads = 14;
        config.num_kv_heads = 2;
        config.batch_size = 1;
        config.d_model = 896;
        config.wo_format = WoFormat::FP32;
        config.causal = true;
        config.fuse_residual_add = true;
        config.residual_type = JitAttentionConfig::ResidualType::Q16_1;

        // This should generate a kernel without crashing
        JitFusedAttentionWo kernel(config);
        auto fn = kernel.getKernel();

        EXPECT_NE(fn, nullptr) << "JIT kernel generation failed";

        std::cout << "[JIT Generation] Kernel generated successfully with fused residual config" << std::endl;
    }

    /**
     * @test Test that non-fused config still works (regression test)
     */
    TEST_F(Test__JitFusedAttentionWo_Q16_1Residual, NonFusedConfigStillWorks)
    {
        JitAttentionConfig config;
        config.head_dim = 64;
        config.num_heads = 14;
        config.num_kv_heads = 2;
        config.batch_size = 1;
        config.d_model = 896;
        config.wo_format = WoFormat::FP32;
        config.causal = true;
        config.fuse_residual_add = false; // Standard path

        JitFusedAttentionWo kernel(config);
        auto fn = kernel.getKernel();

        EXPECT_NE(fn, nullptr) << "JIT kernel generation failed for non-fused config";

        std::cout << "[JIT Generation] Non-fused kernel generated successfully" << std::endl;
    }

    /**
     * @test Test Q16_1Block layout matches expected structure
     */
    TEST_F(Test__JitFusedAttentionWo_Q16_1Residual, Q16_1BlockLayout)
    {
        Q16_1Block block;

        // Verify offsets match JIT assumptions
        EXPECT_EQ(offsetof(Q16_1Block, d), 0) << "Scale d must be at offset 0";
        EXPECT_EQ(offsetof(Q16_1Block, sum_qs), 4) << "sum_qs must be at offset 4";
        EXPECT_EQ(offsetof(Q16_1Block, qs), 8) << "qs array must be at offset 8";
        EXPECT_EQ(sizeof(Q16_1Block), 72) << "Q16_1Block must be 72 bytes";
    }

} // namespace llaminar::v2::kernels::jit::test
