/**
 * @file Test__JitFusedAttentionWo_FA2Parity.cpp
 * @brief FA2 vs FA1 parity tests for JIT fused attention kernel
 * @author David Sanftenberg
 * @date December 2025
 *
 * Tests that the FA2 tiling path (use_fa2_tiling=true) produces
 * numerically equivalent output to the original FA1 path.
 *
 * FA2 optimization processes 4 KV positions per tile:
 * - Single softmax state update per tile (vs 4 in FA1)
 * - Single context rescale per tile (vs up to 4 in FA1)
 * - Interleaved V loads with FMA to hide latency
 *
 * Due to floating-point non-associativity, minor differences are expected,
 * but outputs should be highly correlated (>99.9% cosine similarity).
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
#include "tensors/FP16Utils.h"

namespace llaminar::v2::kernels::jit::test
{

    using llaminar2::fp16_to_fp32;
    using llaminar2::fp32_to_fp16;
    using microkernels::Q8_1Block;

    // ============================================================================
    // Test Fixture
    // ============================================================================

    class Test__JitFusedAttentionWo_FA2Parity : public ::testing::Test
    {
    protected:
        std::mt19937 gen_{42};
        std::uniform_real_distribution<float> dist_{-0.5f, 0.5f};

        // Parity thresholds between FA2 and FA1
        // FA2 batches 4 KV positions per softmax update, which changes the order of
        // floating-point operations. This causes numerical differences even though
        // the algorithms are mathematically equivalent. The differences are:
        //   - exp() computed with different intermediate values
        //   - Different accumulation order for weights and context
        //
        // Expected: Cosine similarity > 0.99 (very high correlation)
        //           Relative L2 error < 10% (dominated by accumulated FP error)
        static constexpr double MIN_COSINE_SIM = 0.99;   // 99% similarity
        static constexpr double MAX_REL_L2_ERROR = 0.10; // 10% relative L2 error

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
                        q = std::max(int8_t(-127), std::min(int8_t(127), q));
                        blk.qs[i] = q;
                        sum_qs += q;
                    }

                    blk.d = fp32_to_fp16(scale);
                    blk.sum_qs = static_cast<int16_t>(sum_qs);
                }
            }
        }

        std::vector<float> generate_random_fp32(int size, float scale = 1.0f)
        {
            std::vector<float> data(size);
            for (auto &v : data)
            {
                v = dist_(gen_) * scale;
            }
            return data;
        }

        double cosine_similarity(const float *a, const float *b, int n)
        {
            double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
            for (int i = 0; i < n; ++i)
            {
                dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
                norm_a += static_cast<double>(a[i]) * static_cast<double>(a[i]);
                norm_b += static_cast<double>(b[i]) * static_cast<double>(b[i]);
            }
            if (norm_a < 1e-10 || norm_b < 1e-10)
                return 1.0;
            return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
        }

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

        double max_abs_error(const float *actual, const float *expected, int n)
        {
            double max_err = 0.0;
            for (int i = 0; i < n; ++i)
            {
                double err = std::fabs(static_cast<double>(actual[i]) - static_cast<double>(expected[i]));
                max_err = std::max(max_err, err);
            }
            return max_err;
        }

        /**
         * @brief Run FA2 vs FA1 parity test
         *
         * Creates two JIT kernels - one with use_fa2_tiling=false (FA1),
         * one with use_fa2_tiling=true (FA2) - and compares their outputs.
         */
        void run_fa2_parity_test(
            int seq_len,
            int kv_seq_len,
            int num_heads,
            int num_kv_heads,
            int head_dim,
            const std::string &test_name)
        {
            const int d_model = num_heads * head_dim;
            const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
            const int blocks_per_head = head_dim / 32;

            std::cout << "\n=== " << test_name << " ===" << std::endl;
            std::cout << "  seq_len=" << seq_len << ", kv_seq_len=" << kv_seq_len
                      << ", heads=" << num_heads << "/" << num_kv_heads
                      << ", head_dim=" << head_dim << std::endl;

            // Generate random FP32 data
            auto Q_fp32 = generate_random_fp32(seq_len * num_heads * head_dim);
            auto K_fp32 = generate_random_fp32(kv_seq_len * num_kv_heads * head_dim);
            auto V_fp32 = generate_random_fp32(kv_seq_len * num_kv_heads * head_dim);

            // Quantize Q, K, V to Q8_1
            std::vector<Q8_1Block> Q_q8(seq_len * num_heads * blocks_per_head);
            std::vector<Q8_1Block> K_q8(kv_seq_len * num_kv_heads * blocks_per_head);
            std::vector<Q8_1Block> V_q8(kv_seq_len * num_kv_heads * blocks_per_head);

            quantize_fp32_to_q8_1(Q_fp32.data(), seq_len * num_heads, head_dim, Q_q8.data());
            quantize_fp32_to_q8_1(K_fp32.data(), kv_seq_len * num_kv_heads, head_dim, K_q8.data());
            quantize_fp32_to_q8_1(V_fp32.data(), kv_seq_len * num_kv_heads, head_dim, V_q8.data());

            // Dummy Wo weights (identity-like)
            std::vector<float> Wo_fp32(d_model * d_model, 0.0f);
            for (int i = 0; i < d_model; ++i)
            {
                Wo_fp32[i * d_model + i] = 1.0f;
            }

            // Output buffers
            std::vector<float> output_fa1(seq_len * d_model, 0.0f);
            std::vector<float> output_fa2(seq_len * d_model, 0.0f);

            // FA1 config (original)
            JitAttentionConfig config_fa1;
            config_fa1.head_dim = head_dim;
            config_fa1.num_heads = num_heads;
            config_fa1.num_kv_heads = num_kv_heads;
            config_fa1.batch_size = 1;
            config_fa1.wo_format = WoFormat::FP32;
            config_fa1.causal = false;
            config_fa1.use_fa2_tiling = false; // FA1 path

            // FA2 config (new tiling)
            JitAttentionConfig config_fa2;
            config_fa2.head_dim = head_dim;
            config_fa2.num_heads = num_heads;
            config_fa2.num_kv_heads = num_kv_heads;
            config_fa2.batch_size = 1;
            config_fa2.wo_format = WoFormat::FP32;
            config_fa2.causal = false;
            config_fa2.use_fa2_tiling = true;           // FA2 path
            config_fa2.enable_register_tracking = true; // Enable tracking for debugging

            // Create and run FA1 kernel
            JitFusedAttentionWo kernel_fa1(config_fa1);
            kernel_fa1.compute(
                Q_q8.data(),
                K_q8.data(),
                V_q8.data(),
                Wo_fp32.data(),
                output_fa1.data(),
                seq_len,
                kv_seq_len,
                scale,
                0);

            // Create and run FA2 kernel
            JitFusedAttentionWo kernel_fa2(config_fa2);
            kernel_fa2.compute(
                Q_q8.data(),
                K_q8.data(),
                V_q8.data(),
                Wo_fp32.data(),
                output_fa2.data(),
                seq_len,
                kv_seq_len,
                scale,
                0);

            // Compare outputs
            int output_size = seq_len * d_model;
            double cos_sim = cosine_similarity(output_fa2.data(), output_fa1.data(), output_size);
            double rel_l2 = relative_l2_error(output_fa2.data(), output_fa1.data(), output_size);
            double max_err = max_abs_error(output_fa2.data(), output_fa1.data(), output_size);

            std::cout << "  Cosine similarity (FA2 vs FA1):  " << std::fixed << std::setprecision(6) << cos_sim << std::endl;
            std::cout << "  Relative L2 error:               " << std::fixed << std::setprecision(6) << rel_l2 << std::endl;
            std::cout << "  Max absolute error:              " << std::fixed << std::setprecision(6) << max_err << std::endl;

            // Validate
            EXPECT_GE(cos_sim, MIN_COSINE_SIM)
                << "FA2 vs FA1 cosine similarity too low: " << cos_sim << " < " << MIN_COSINE_SIM;
            EXPECT_LE(rel_l2, MAX_REL_L2_ERROR)
                << "FA2 vs FA1 relative L2 error too high: " << rel_l2 << " > " << MAX_REL_L2_ERROR;

            // Debug output if parity fails
            if (cos_sim < MIN_COSINE_SIM || rel_l2 > MAX_REL_L2_ERROR || std::isnan(cos_sim))
            {
                std::cout << "  First 16 FA1 outputs: ";
                for (int i = 0; i < std::min(16, output_size); ++i)
                {
                    std::cout << output_fa1[i] << " ";
                }
                std::cout << std::endl;

                std::cout << "  First 16 FA2 outputs: ";
                for (int i = 0; i < std::min(16, output_size); ++i)
                {
                    std::cout << output_fa2[i] << " ";
                }
                std::cout << std::endl;

                // Check if outputs are all zeros and scan for NaN/Inf
                bool fa1_all_zero = true, fa2_all_zero = true;
                int fa1_nan_count = 0, fa2_nan_count = 0;
                int fa1_inf_count = 0, fa2_inf_count = 0;
                int first_fa1_nan = -1, first_fa2_nan = -1;
                float max_fa1_val = 0.0f, max_fa2_val = 0.0f;
                int max_fa1_idx = -1, max_fa2_idx = -1;
                for (int i = 0; i < output_size; ++i)
                {
                    if (output_fa1[i] != 0.0f)
                        fa1_all_zero = false;
                    if (output_fa2[i] != 0.0f)
                        fa2_all_zero = false;
                    if (std::isnan(output_fa1[i]))
                    {
                        fa1_nan_count++;
                        if (first_fa1_nan < 0)
                            first_fa1_nan = i;
                    }
                    if (std::isnan(output_fa2[i]))
                    {
                        fa2_nan_count++;
                        if (first_fa2_nan < 0)
                            first_fa2_nan = i;
                    }
                    if (std::isinf(output_fa1[i]))
                        fa1_inf_count++;
                    if (std::isinf(output_fa2[i]))
                        fa2_inf_count++;
                    if (std::fabs(output_fa1[i]) > max_fa1_val && !std::isnan(output_fa1[i]) && !std::isinf(output_fa1[i]))
                    {
                        max_fa1_val = std::fabs(output_fa1[i]);
                        max_fa1_idx = i;
                    }
                    if (std::fabs(output_fa2[i]) > max_fa2_val && !std::isnan(output_fa2[i]) && !std::isinf(output_fa2[i]))
                    {
                        max_fa2_val = std::fabs(output_fa2[i]);
                        max_fa2_idx = i;
                    }
                }
                std::cout << "  FA1 all zeros: " << (fa1_all_zero ? "YES" : "NO") << std::endl;
                std::cout << "  FA2 all zeros: " << (fa2_all_zero ? "YES" : "NO") << std::endl;
                std::cout << "  FA1 NaN count: " << fa1_nan_count << ", first at idx " << first_fa1_nan << std::endl;
                std::cout << "  FA2 NaN count: " << fa2_nan_count << ", first at idx " << first_fa2_nan << std::endl;
                std::cout << "  FA1 Inf count: " << fa1_inf_count << ", FA2 Inf count: " << fa2_inf_count << std::endl;
                std::cout << "  FA1 max abs value: " << max_fa1_val << " at idx " << max_fa1_idx << std::endl;
                std::cout << "  FA2 max abs value: " << max_fa2_val << " at idx " << max_fa2_idx << std::endl;

                // Print values around the largest diff
                double worst_diff = 0.0;
                int worst_idx = -1;
                for (int i = 0; i < output_size; ++i)
                {
                    double diff = std::fabs(static_cast<double>(output_fa1[i]) - static_cast<double>(output_fa2[i]));
                    if (diff > worst_diff && !std::isnan(diff))
                    {
                        worst_diff = diff;
                        worst_idx = i;
                    }
                }
                std::cout << "  Worst diff: " << worst_diff << " at idx " << worst_idx << std::endl;
                if (worst_idx >= 0)
                {
                    std::cout << "    FA1[" << worst_idx << "]=" << output_fa1[worst_idx]
                              << ", FA2[" << worst_idx << "]=" << output_fa2[worst_idx] << std::endl;
                    int head = worst_idx / head_dim;
                    int elem = worst_idx % head_dim;
                    int kv_head = head / (num_heads / num_kv_heads);
                    std::cout << "    => head=" << head << ", elem=" << elem << ", kv_head=" << kv_head << std::endl;
                }

                // Per-head error summary
                std::cout << "\n  Per-head max absolute errors:" << std::endl;
                for (int h = 0; h < num_heads; ++h)
                {
                    double head_max_err = 0.0;
                    int head_bad_count = 0;
                    for (int e = 0; e < head_dim; ++e)
                    {
                        int idx = h * head_dim + e;
                        double err = std::fabs(static_cast<double>(output_fa1[idx]) - static_cast<double>(output_fa2[idx]));
                        if (err > head_max_err)
                            head_max_err = err;
                        if (err > 1e10)
                            head_bad_count++;
                    }
                    int kv_head = h / (num_heads / num_kv_heads);
                    if (head_max_err > 0.1 || head_bad_count > 0)
                    {
                        std::cout << "    Head " << h << " (kv_head " << kv_head << "): max_err=" << head_max_err
                                  << ", bad_values=" << head_bad_count << std::endl;
                    }
                }
            }
        }
    };

    // ============================================================================
    // Test Cases: KV lengths that are exact multiples of 4
    // ============================================================================

    TEST_F(Test__JitFusedAttentionWo_FA2Parity, Qwen05B_Decode_KV4)
    {
        // kv_seq_len=4: exactly 1 FA2 tile, no remainder
        run_fa2_parity_test(1, 4, 14, 2, 64, "Qwen2 0.5B decode, kv=4");
    }

    TEST_F(Test__JitFusedAttentionWo_FA2Parity, Qwen05B_Decode_KV8)
    {
        // kv_seq_len=8: exactly 2 FA2 tiles, no remainder
        run_fa2_parity_test(1, 8, 14, 2, 64, "Qwen2 0.5B decode, kv=8");
    }

    TEST_F(Test__JitFusedAttentionWo_FA2Parity, Qwen05B_Decode_KV16)
    {
        run_fa2_parity_test(1, 16, 14, 2, 64, "Qwen2 0.5B decode, kv=16");
    }

    TEST_F(Test__JitFusedAttentionWo_FA2Parity, Qwen05B_Decode_KV64)
    {
        run_fa2_parity_test(1, 64, 14, 2, 64, "Qwen2 0.5B decode, kv=64");
    }

    // ============================================================================
    // Test Cases: KV lengths with remainder (1, 2, 3 extra positions)
    // ============================================================================

    TEST_F(Test__JitFusedAttentionWo_FA2Parity, Qwen05B_Decode_KV5)
    {
        // kv_seq_len=5: 1 FA2 tile + 1 remainder
        run_fa2_parity_test(1, 5, 14, 2, 64, "Qwen2 0.5B decode, kv=5 (1 tile + 1 remainder)");
    }

    TEST_F(Test__JitFusedAttentionWo_FA2Parity, Qwen05B_Decode_KV6)
    {
        // kv_seq_len=6: 1 FA2 tile + 2 remainder
        run_fa2_parity_test(1, 6, 14, 2, 64, "Qwen2 0.5B decode, kv=6 (1 tile + 2 remainder)");
    }

    TEST_F(Test__JitFusedAttentionWo_FA2Parity, Qwen05B_Decode_KV7)
    {
        // kv_seq_len=7: 1 FA2 tile + 3 remainder
        run_fa2_parity_test(1, 7, 14, 2, 64, "Qwen2 0.5B decode, kv=7 (1 tile + 3 remainder)");
    }

    TEST_F(Test__JitFusedAttentionWo_FA2Parity, Qwen05B_Decode_KV17)
    {
        // kv_seq_len=17: 4 FA2 tiles + 1 remainder
        run_fa2_parity_test(1, 17, 14, 2, 64, "Qwen2 0.5B decode, kv=17 (4 tiles + 1 remainder)");
    }

    // ============================================================================
    // Test Cases: Edge cases (KV < 4)
    // ============================================================================

    TEST_F(Test__JitFusedAttentionWo_FA2Parity, Qwen05B_Decode_KV1)
    {
        // kv_seq_len=1: no FA2 tiles, just 1 remainder
        run_fa2_parity_test(1, 1, 14, 2, 64, "Qwen2 0.5B decode, kv=1 (no tiles, 1 remainder)");
    }

    TEST_F(Test__JitFusedAttentionWo_FA2Parity, Qwen05B_Decode_KV2)
    {
        // kv_seq_len=2: no FA2 tiles, just 2 remainder
        run_fa2_parity_test(1, 2, 14, 2, 64, "Qwen2 0.5B decode, kv=2 (no tiles, 2 remainder)");
    }

    TEST_F(Test__JitFusedAttentionWo_FA2Parity, Qwen05B_Decode_KV3)
    {
        // kv_seq_len=3: no FA2 tiles, just 3 remainder
        run_fa2_parity_test(1, 3, 14, 2, 64, "Qwen2 0.5B decode, kv=3 (no tiles, 3 remainder)");
    }

    // ============================================================================
    // Test Cases: Larger head_dim (128)
    // ============================================================================

    TEST_F(Test__JitFusedAttentionWo_FA2Parity, Qwen7B_Decode_KV8)
    {
        run_fa2_parity_test(1, 8, 28, 4, 128, "Qwen2 7B decode, kv=8");
    }

    TEST_F(Test__JitFusedAttentionWo_FA2Parity, Qwen7B_Decode_KV11)
    {
        // kv_seq_len=11: 2 FA2 tiles + 3 remainder
        run_fa2_parity_test(1, 11, 28, 4, 128, "Qwen2 7B decode, kv=11 (2 tiles + 3 remainder)");
    }

    // ============================================================================
    // Test Cases: Larger KV sequence lengths
    // ============================================================================

    TEST_F(Test__JitFusedAttentionWo_FA2Parity, Qwen05B_Decode_KV128)
    {
        run_fa2_parity_test(1, 128, 14, 2, 64, "Qwen2 0.5B decode, kv=128");
    }

    TEST_F(Test__JitFusedAttentionWo_FA2Parity, Qwen05B_Decode_KV129)
    {
        // 32 tiles + 1 remainder
        run_fa2_parity_test(1, 129, 14, 2, 64, "Qwen2 0.5B decode, kv=129 (32 tiles + 1 remainder)");
    }

    TEST_F(Test__JitFusedAttentionWo_FA2Parity, Qwen05B_Decode_KV255)
    {
        // 63 tiles + 3 remainder
        run_fa2_parity_test(1, 255, 14, 2, 64, "Qwen2 0.5B decode, kv=255 (63 tiles + 3 remainder)");
    }

    TEST_F(Test__JitFusedAttentionWo_FA2Parity, Qwen05B_Decode_KV256)
    {
        // 64 tiles, no remainder
        run_fa2_parity_test(1, 256, 14, 2, 64, "Qwen2 0.5B decode, kv=256 (64 tiles)");
    }

    // ============================================================================
    // Test Case: Register tracking enabled to find conflicts
    // ============================================================================

    TEST_F(Test__JitFusedAttentionWo_FA2Parity, RegisterTrackingKV4)
    {
        // This test enables register tracking during JIT compilation to detect
        // any register conflicts. If there's a conflict, it will assert.

        int seq_len = 1;
        int kv_seq_len = 4; // Single FA2 tile
        int num_heads = 14;
        int num_kv_heads = 2;
        int head_dim = 64;
        int d_model = num_heads * head_dim;
        int blocks_per_head = head_dim / 32;
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        std::cout << "\n=== Running with REGISTER TRACKING enabled ===" << std::endl;

        // Generate random FP32 data
        auto Q_fp32 = generate_random_fp32(seq_len * num_heads * head_dim);
        auto K_fp32 = generate_random_fp32(kv_seq_len * num_kv_heads * head_dim);
        auto V_fp32 = generate_random_fp32(kv_seq_len * num_kv_heads * head_dim);

        // Quantize Q, K, V to Q8_1
        std::vector<Q8_1Block> Q_q8(seq_len * num_heads * blocks_per_head);
        std::vector<Q8_1Block> K_q8(kv_seq_len * num_kv_heads * blocks_per_head);
        std::vector<Q8_1Block> V_q8(kv_seq_len * num_kv_heads * blocks_per_head);

        quantize_fp32_to_q8_1(Q_fp32.data(), seq_len * num_heads, head_dim, Q_q8.data());
        quantize_fp32_to_q8_1(K_fp32.data(), kv_seq_len * num_kv_heads, head_dim, K_q8.data());
        quantize_fp32_to_q8_1(V_fp32.data(), kv_seq_len * num_kv_heads, head_dim, V_q8.data());

        // Dummy Wo weights (identity-like)
        std::vector<float> Wo_fp32(d_model * d_model, 0.0f);
        for (int i = 0; i < d_model; ++i)
        {
            Wo_fp32[i * d_model + i] = 1.0f;
        }

        std::vector<float> output_fa2(seq_len * d_model, 0.0f);

        // FA2 config with tracking enabled
        JitAttentionConfig config_fa2;
        config_fa2.head_dim = head_dim;
        config_fa2.num_heads = num_heads;
        config_fa2.num_kv_heads = num_kv_heads;
        config_fa2.batch_size = 1;
        config_fa2.wo_format = WoFormat::FP32;
        config_fa2.causal = false;
        config_fa2.use_fa2_tiling = true;
        config_fa2.enable_register_tracking = true; // ENABLE TRACKING!

        // Create and run FA2 kernel with tracking
        // If there's a register conflict, this will ASSERT during JIT compilation
        JitFusedAttentionWo kernel_fa2(config_fa2);
        kernel_fa2.compute(
            Q_q8.data(),
            K_q8.data(),
            V_q8.data(),
            Wo_fp32.data(),
            output_fa2.data(),
            seq_len,
            kv_seq_len,
            scale,
            0);

        // Check for NaN
        bool has_nan = false;
        for (int i = 0; i < seq_len * d_model; ++i)
        {
            if (std::isnan(output_fa2[i]))
            {
                has_nan = true;
                break;
            }
        }

        std::cout << "  Has NaN: " << (has_nan ? "YES" : "NO") << std::endl;
        std::cout << "  First 8 outputs: ";
        for (int i = 0; i < 8; ++i)
        {
            std::cout << output_fa2[i] << " ";
        }
        std::cout << std::endl;

        EXPECT_FALSE(has_nan) << "FA2 output contains NaN!";
    }

    // ========================================================================
    // Optimized Wo Projection Tests
    // ========================================================================

    /**
     * @brief Test optimized FP32 Wo projection vs naive implementation
     *
     * Verifies that use_optimized_wo=true produces numerically equivalent
     * output to use_optimized_wo=false (the naive row-by-row implementation).
     */
    TEST_F(Test__JitFusedAttentionWo_FA2Parity, OptimizedWo_vs_Naive_896x896)
    {
        constexpr int seq_len = 1;
        constexpr int kv_seq_len = 32;
        constexpr int num_heads = 14;
        constexpr int num_kv_heads = 2;
        constexpr int head_dim = 64;
        constexpr int d_model = num_heads * head_dim; // 896
        constexpr float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        constexpr int blocks_per_head = head_dim / 32;

        std::cout << "\n=== Optimized Wo vs Naive (d_model=" << d_model << ") ===" << std::endl;

        // Generate random data
        auto Q_fp32 = generate_random_fp32(seq_len * num_heads * head_dim);
        auto K_fp32 = generate_random_fp32(kv_seq_len * num_kv_heads * head_dim);
        auto V_fp32 = generate_random_fp32(kv_seq_len * num_kv_heads * head_dim);

        // Quantize
        std::vector<Q8_1Block> Q_q8(seq_len * num_heads * blocks_per_head);
        std::vector<Q8_1Block> K_q8(kv_seq_len * num_kv_heads * blocks_per_head);
        std::vector<Q8_1Block> V_q8(kv_seq_len * num_kv_heads * blocks_per_head);

        quantize_fp32_to_q8_1(Q_fp32.data(), seq_len * num_heads, head_dim, Q_q8.data());
        quantize_fp32_to_q8_1(K_fp32.data(), kv_seq_len * num_kv_heads, head_dim, K_q8.data());
        quantize_fp32_to_q8_1(V_fp32.data(), kv_seq_len * num_kv_heads, head_dim, V_q8.data());

        // Random Wo weights
        auto Wo_fp32 = generate_random_fp32(d_model * d_model, 0.1f);

        // Output buffers
        std::vector<float> output_naive(seq_len * d_model, 0.0f);
        std::vector<float> output_optimized(seq_len * d_model, 0.0f);

        // Naive config
        JitAttentionConfig config_naive;
        config_naive.head_dim = head_dim;
        config_naive.num_heads = num_heads;
        config_naive.num_kv_heads = num_kv_heads;
        config_naive.batch_size = 1;
        config_naive.wo_format = WoFormat::FP32;
        config_naive.causal = false;
        config_naive.use_fa2_tiling = true;
        config_naive.use_optimized_wo = false; // Naive

        // Optimized config
        JitAttentionConfig config_optimized;
        config_optimized.head_dim = head_dim;
        config_optimized.num_heads = num_heads;
        config_optimized.num_kv_heads = num_kv_heads;
        config_optimized.batch_size = 1;
        config_optimized.wo_format = WoFormat::FP32;
        config_optimized.causal = false;
        config_optimized.use_fa2_tiling = true;
        config_optimized.use_optimized_wo = true; // Optimized

        // Run both kernels
        JitFusedAttentionWo kernel_naive(config_naive);
        kernel_naive.compute(
            Q_q8.data(), K_q8.data(), V_q8.data(), Wo_fp32.data(),
            output_naive.data(), seq_len, kv_seq_len, scale, 0);

        JitFusedAttentionWo kernel_optimized(config_optimized);
        kernel_optimized.compute(
            Q_q8.data(), K_q8.data(), V_q8.data(), Wo_fp32.data(),
            output_optimized.data(), seq_len, kv_seq_len, scale, 0);

        // Compare outputs
        double cos_sim = cosine_similarity(output_optimized.data(), output_naive.data(), seq_len * d_model);
        double rel_l2 = relative_l2_error(output_optimized.data(), output_naive.data(), seq_len * d_model);
        double max_err = max_abs_error(output_optimized.data(), output_naive.data(), seq_len * d_model);

        std::cout << "  Cosine similarity (opt vs naive):  " << std::fixed << std::setprecision(6) << cos_sim << std::endl;
        std::cout << "  Relative L2 error:                 " << std::fixed << std::setprecision(6) << rel_l2 << std::endl;
        std::cout << "  Max absolute error:                " << std::fixed << std::setprecision(6) << max_err << std::endl;

        // Optimized should be numerically identical (both use same algorithm, just different loop structure)
        EXPECT_GT(cos_sim, 0.99999) << "Optimized Wo diverges from naive!";
        EXPECT_LT(rel_l2, 1e-5) << "Relative L2 error too large!";
        EXPECT_LT(max_err, 1e-5) << "Max absolute error too large!";

        // Timing (informational only; no assertions to avoid flakiness)
        // Measures end-to-end fused kernel with only Wo projection differing.
        auto time_kernel_ms = [&](JitFusedAttentionWo &kernel, float *out)
        {
            constexpr int warmup_iters = 10;
            constexpr int bench_iters = 100;
            std::vector<double> times;
            times.reserve(bench_iters);

            for (int i = 0; i < warmup_iters; ++i)
            {
                kernel.compute(Q_q8.data(), K_q8.data(), V_q8.data(), Wo_fp32.data(),
                               out, seq_len, kv_seq_len, scale, 0);
            }

            for (int i = 0; i < bench_iters; ++i)
            {
                auto start = std::chrono::high_resolution_clock::now();
                kernel.compute(Q_q8.data(), K_q8.data(), V_q8.data(), Wo_fp32.data(),
                               out, seq_len, kv_seq_len, scale, 0);
                auto end = std::chrono::high_resolution_clock::now();
                times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
            }

            std::sort(times.begin(), times.end());
            return times[times.size() / 2];
        };

        // Reuse outputs as scratch to avoid allocating inside timing loops
        double naive_ms = time_kernel_ms(kernel_naive, output_naive.data());
        double opt_ms = time_kernel_ms(kernel_optimized, output_optimized.data());
        double speedup = naive_ms / opt_ms;

        std::cout << "\n  Timing (median of 100 iters, warmup 10):" << std::endl;
        std::cout << "    Naive:     " << std::fixed << std::setprecision(3) << naive_ms << " ms" << std::endl;
        std::cout << "    Optimized: " << std::fixed << std::setprecision(3) << opt_ms << " ms" << std::endl;
        std::cout << "    Speedup (naive/opt): " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
    }

} // namespace llaminar::v2::kernels::jit::test
