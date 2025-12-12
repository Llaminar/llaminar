/**
 * @file Test__JitFusedAttentionWo_Correctness.cpp
 * @brief Numerical correctness tests for JIT fused attention + Wo kernel
 * @author David Sanftenberg
 * @date December 2025
 *
 * Comprehensive correctness tests comparing JIT kernel output against
 * the reference implementation (FusedAttentionWoRef).
 *
 * Test configurations cover real Qwen model dimensions:
 * - Qwen2 0.5B: 14 heads, 2 KV heads, head_dim=64, d_model=896
 * - Qwen2 1.5B: 12 heads, 2 KV heads, head_dim=64, d_model=768
 * - Qwen2 7B:   28 heads, 4 KV heads, head_dim=128, d_model=3584
 * - Qwen2 32B:  40 heads, 8 KV heads, head_dim=128, d_model=5120
 * - Qwen2 72B:  64 heads, 8 KV heads, head_dim=128, d_model=8192
 *
 * Sequence length tests:
 * - Single token decode (seq=1, common for inference)
 * - Short prefill (seq=8-32)
 * - Medium prefill (seq=128-256)
 * - Long prefill (seq=512-1024)
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <random>
#include <cstring>
#include <iostream>
#include <iomanip>

#include "kernels/cpu/jit/q8_1/JitFusedAttentionWo.h"
#include "kernels/cpu/attention/q8_1/FusedAttentionWoRef.h"
#include "tensors/BlockStructures.h"
#include "tensors/FP16Utils.h"

namespace llaminar::v2::kernels::jit::test
{

    using llaminar::v2::kernels::FusedAttentionWoParams;
    using llaminar::v2::kernels::FusedAttentionWoRef;
    using llaminar::v2::kernels::microkernels::WoWeightType;
    using llaminar2::fp16_to_fp32;
    using llaminar2::fp32_to_fp16;
    using microkernels::Q8_1Block;

    // ============================================================================
    // Test Fixture
    // ============================================================================

    class Test__JitFusedAttentionWo_Correctness : public ::testing::Test
    {
    protected:
        std::mt19937 gen_{42};
        std::uniform_real_distribution<float> dist_{-0.5f, 0.5f}; // Smaller range for numerical stability

        // Correctness thresholds
        // Note: Q8_1 quantization introduces noise, so we use relaxed thresholds
        static constexpr double MIN_COSINE_SIM = 0.990;        // 99% similarity
        static constexpr double MAX_REL_L2_ERROR = 0.10;       // 10% relative L2 error
        static constexpr double MAX_ABS_ERROR_PER_ELEM = 0.5f; // Per-element sanity check

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
                    if (scale < 1e-10f)
                        scale = 1e-10f; // Avoid division by zero
                    float inv_scale = 127.0f / max_abs;
                    if (max_abs < 1e-10f)
                        inv_scale = 0.0f;

                    // Quantize
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

        /**
         * @brief Generate random FP32 data with controlled magnitude
         */
        std::vector<float> generate_random_fp32(int size, float scale = 1.0f)
        {
            std::vector<float> data(size);
            for (auto &v : data)
            {
                v = dist_(gen_) * scale;
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
                dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
                norm_a += static_cast<double>(a[i]) * static_cast<double>(a[i]);
                norm_b += static_cast<double>(b[i]) * static_cast<double>(b[i]);
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

        /**
         * @brief Compute max absolute error
         */
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
         * @brief Run parity test between JIT and reference kernels
         *
         * @param seq_len Query sequence length
         * @param kv_seq_len KV sequence length
         * @param num_heads Number of Q heads
         * @param num_kv_heads Number of KV heads
         * @param head_dim Head dimension (64 or 128)
         * @param test_name Name for logging
         */
        void run_parity_test(
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
                      << ", head_dim=" << head_dim << ", d_model=" << d_model << std::endl;

            // Generate random FP32 data
            auto Q_fp32 = generate_random_fp32(seq_len * num_heads * head_dim);
            auto K_fp32 = generate_random_fp32(kv_seq_len * num_kv_heads * head_dim);
            auto V_fp32 = generate_random_fp32(kv_seq_len * num_kv_heads * head_dim);

            // Wo weights: [d_model, d_model] - FP32 for this test
            // Note: For fused attention, the context [num_heads, head_dim] is projected
            // through Wo [d_model, d_model] but currently JIT just outputs context directly
            // So we compare context outputs, not post-Wo outputs

            // Quantize Q, K, V to Q8_1
            std::vector<Q8_1Block> Q_q8(seq_len * num_heads * blocks_per_head);
            std::vector<Q8_1Block> K_q8(kv_seq_len * num_kv_heads * blocks_per_head);
            std::vector<Q8_1Block> V_q8(kv_seq_len * num_kv_heads * blocks_per_head);

            quantize_fp32_to_q8_1(Q_fp32.data(), seq_len * num_heads, head_dim, Q_q8.data());
            quantize_fp32_to_q8_1(K_fp32.data(), kv_seq_len * num_kv_heads, head_dim, K_q8.data());
            quantize_fp32_to_q8_1(V_fp32.data(), kv_seq_len * num_kv_heads, head_dim, V_q8.data());

            // Output buffers
            // Note: JIT currently outputs [seq_len, num_heads * head_dim] (context without Wo)
            std::vector<float> output_jit(seq_len * d_model, 0.0f);
            std::vector<float> output_ref(seq_len * d_model, 0.0f);

            // Create dummy Wo weights for reference (identity-like or simple pattern)
            // Since JIT doesn't apply Wo yet, we need to compare attention contexts
            // For now, use identity-like Wo so reference output ≈ attention context
            std::vector<float> Wo_fp32(d_model * d_model, 0.0f);
            // Create block-diagonal identity (each head's output maps to itself)
            for (int h = 0; h < num_heads; ++h)
            {
                for (int d = 0; d < head_dim; ++d)
                {
                    int row = h * head_dim + d;
                    // For identity-like projection, set Wo[row, row] = 1
                    // But Wo layout might be [out_dim, in_dim] or [in_dim, out_dim]
                    // Reference expects [d_model, num_heads * head_dim]
                    Wo_fp32[row * d_model + row] = 1.0f;
                }
            }

            // Run reference kernel
            FusedAttentionWoParams ref_params;
            ref_params.Q = Q_q8.data();
            ref_params.K = K_q8.data();
            ref_params.V = V_q8.data();
            ref_params.Wo = Wo_fp32.data();
            ref_params.wo_type = WoWeightType::FP32;
            ref_params.output = output_ref.data();
            ref_params.batch_size = 1;
            ref_params.seq_len = seq_len;
            ref_params.kv_seq_len = kv_seq_len;
            ref_params.num_heads = num_heads;
            ref_params.num_kv_heads = num_kv_heads;
            ref_params.head_dim = head_dim;
            ref_params.d_model = d_model;
            ref_params.scale = scale;
            ref_params.causal = false; // Non-causal for basic correctness
            ref_params.position_offset = 0;

            bool ref_ok = FusedAttentionWoRef::execute(ref_params);
            ASSERT_TRUE(ref_ok) << "Reference kernel failed";

            // Run JIT kernel
            JitAttentionConfig jit_config;
            jit_config.head_dim = head_dim;
            jit_config.num_heads = num_heads;
            jit_config.num_kv_heads = num_kv_heads;
            jit_config.batch_size = 1;
            jit_config.wo_format = WoFormat::FP32;

            JitFusedAttentionWo jit_kernel(jit_config);
            jit_kernel.compute(
                Q_q8.data(),
                K_q8.data(),
                V_q8.data(),
                Wo_fp32.data(), // Not used yet by JIT
                output_jit.data(),
                seq_len,
                kv_seq_len,
                scale);

            // Compare outputs
            int output_size = seq_len * d_model;
            double cos_sim = cosine_similarity(output_jit.data(), output_ref.data(), output_size);
            double rel_l2 = relative_l2_error(output_jit.data(), output_ref.data(), output_size);
            double max_err = max_abs_error(output_jit.data(), output_ref.data(), output_size);

            std::cout << "  Cosine similarity:  " << std::fixed << std::setprecision(6) << cos_sim << std::endl;
            std::cout << "  Relative L2 error:  " << std::fixed << std::setprecision(6) << rel_l2 << std::endl;
            std::cout << "  Max absolute error: " << std::fixed << std::setprecision(6) << max_err << std::endl;

            // Validate
            EXPECT_GE(cos_sim, MIN_COSINE_SIM)
                << "Cosine similarity too low: " << cos_sim << " < " << MIN_COSINE_SIM;
            EXPECT_LE(rel_l2, MAX_REL_L2_ERROR)
                << "Relative L2 error too high: " << rel_l2 << " > " << MAX_REL_L2_ERROR;

            // Debug: Print first few output values if error is high or NaN
            if (std::isnan(cos_sim) || cos_sim < MIN_COSINE_SIM || rel_l2 > MAX_REL_L2_ERROR)
            {
                std::cout << "  First 16 JIT outputs: ";
                for (int i = 0; i < std::min(16, output_size); ++i)
                {
                    std::cout << output_jit[i] << " ";
                }
                std::cout << std::endl;

                std::cout << "  First 16 REF outputs: ";
                for (int i = 0; i < std::min(16, output_size); ++i)
                {
                    std::cout << output_ref[i] << " ";
                }
                std::cout << std::endl;

                // Per-head comparison
                std::cout << "\n  Per-head mean errors:" << std::endl;
                for (int h = 0; h < num_heads; ++h)
                {
                    int head_start = h * head_dim;
                    double head_err = 0.0;
                    double max_head_err = 0.0;
                    int max_err_pos = 0;
                    for (int d = 0; d < head_dim; ++d)
                    {
                        double diff = std::fabs(output_jit[head_start + d] - output_ref[head_start + d]);
                        head_err += diff;
                        if (diff > max_head_err)
                        {
                            max_head_err = diff;
                            max_err_pos = d;
                        }
                    }
                    head_err /= head_dim;
                    std::cout << "    Head " << h << ": mean_abs_err=" << head_err
                              << ", max_err=" << max_head_err << " at pos " << max_err_pos
                              << " (jit=" << output_jit[head_start + max_err_pos]
                              << ", ref=" << output_ref[head_start + max_err_pos] << ")" << std::endl;
                }

                // Print outputs around head boundary
                if (num_heads >= 2)
                {
                    std::cout << "\n  Around head 0/1 boundary (positions 60-68):" << std::endl;
                    for (int i = head_dim - 4; i < head_dim + 4 && i < output_size; ++i)
                    {
                        std::cout << "    [" << i << "] jit=" << output_jit[i] << " ref=" << output_ref[i] << std::endl;
                    }
                }
            }
        }
    };

    // ============================================================================
    // Qwen2 0.5B Configuration Tests (14 heads, 2 KV heads, head_dim=64)
    // ============================================================================

    TEST_F(Test__JitFusedAttentionWo_Correctness, Qwen2_0_5B_Decode_SingleToken)
    {
        // Single token decode - most common inference pattern
        run_parity_test(
            /*seq_len=*/1,
            /*kv_seq_len=*/64, // Some context in KV cache
            /*num_heads=*/14,
            /*num_kv_heads=*/2,
            /*head_dim=*/64,
            "Qwen2 0.5B - Single Token Decode (seq=1, kv=64)");
    }

    TEST_F(Test__JitFusedAttentionWo_Correctness, Qwen2_0_5B_Decode_ShortKV)
    {
        run_parity_test(
            /*seq_len=*/1,
            /*kv_seq_len=*/8,
            /*num_heads=*/14,
            /*num_kv_heads=*/2,
            /*head_dim=*/64,
            "Qwen2 0.5B - Single Token Decode (seq=1, kv=8)");
    }

    TEST_F(Test__JitFusedAttentionWo_Correctness, Qwen2_0_5B_Decode_LongKV)
    {
        run_parity_test(
            /*seq_len=*/1,
            /*kv_seq_len=*/256,
            /*num_heads=*/14,
            /*num_kv_heads=*/2,
            /*head_dim=*/64,
            "Qwen2 0.5B - Single Token Decode (seq=1, kv=256)");
    }

    TEST_F(Test__JitFusedAttentionWo_Correctness, Qwen2_0_5B_Prefill_Short)
    {
        run_parity_test(
            /*seq_len=*/8,
            /*kv_seq_len=*/8,
            /*num_heads=*/14,
            /*num_kv_heads=*/2,
            /*head_dim=*/64,
            "Qwen2 0.5B - Short Prefill (seq=8)");
    }

    TEST_F(Test__JitFusedAttentionWo_Correctness, Qwen2_0_5B_Prefill_Medium)
    {
        run_parity_test(
            /*seq_len=*/32,
            /*kv_seq_len=*/32,
            /*num_heads=*/14,
            /*num_kv_heads=*/2,
            /*head_dim=*/64,
            "Qwen2 0.5B - Medium Prefill (seq=32)");
    }

    TEST_F(Test__JitFusedAttentionWo_Correctness, Qwen2_0_5B_Prefill_Long)
    {
        run_parity_test(
            /*seq_len=*/128,
            /*kv_seq_len=*/128,
            /*num_heads=*/14,
            /*num_kv_heads=*/2,
            /*head_dim=*/64,
            "Qwen2 0.5B - Long Prefill (seq=128)");
    }

    // ============================================================================
    // Qwen2 1.5B Configuration Tests (12 heads, 2 KV heads, head_dim=64)
    // ============================================================================

    TEST_F(Test__JitFusedAttentionWo_Correctness, Qwen2_1_5B_Decode_SingleToken)
    {
        run_parity_test(
            /*seq_len=*/1,
            /*kv_seq_len=*/64,
            /*num_heads=*/12,
            /*num_kv_heads=*/2,
            /*head_dim=*/64,
            "Qwen2 1.5B - Single Token Decode (seq=1, kv=64)");
    }

    TEST_F(Test__JitFusedAttentionWo_Correctness, Qwen2_1_5B_Prefill_Short)
    {
        run_parity_test(
            /*seq_len=*/16,
            /*kv_seq_len=*/16,
            /*num_heads=*/12,
            /*num_kv_heads=*/2,
            /*head_dim=*/64,
            "Qwen2 1.5B - Short Prefill (seq=16)");
    }

    // ============================================================================
    // Qwen2 7B Configuration Tests (28 heads, 4 KV heads, head_dim=128)
    // ============================================================================

    TEST_F(Test__JitFusedAttentionWo_Correctness, Qwen2_7B_Decode_SingleToken)
    {
        run_parity_test(
            /*seq_len=*/1,
            /*kv_seq_len=*/64,
            /*num_heads=*/28,
            /*num_kv_heads=*/4,
            /*head_dim=*/128,
            "Qwen2 7B - Single Token Decode (seq=1, kv=64)");
    }

    TEST_F(Test__JitFusedAttentionWo_Correctness, Qwen2_7B_Decode_LongKV)
    {
        run_parity_test(
            /*seq_len=*/1,
            /*kv_seq_len=*/256,
            /*num_heads=*/28,
            /*num_kv_heads=*/4,
            /*head_dim=*/128,
            "Qwen2 7B - Single Token Decode (seq=1, kv=256)");
    }

    TEST_F(Test__JitFusedAttentionWo_Correctness, Qwen2_7B_Prefill_Short)
    {
        run_parity_test(
            /*seq_len=*/8,
            /*kv_seq_len=*/8,
            /*num_heads=*/28,
            /*num_kv_heads=*/4,
            /*head_dim=*/128,
            "Qwen2 7B - Short Prefill (seq=8)");
    }

    TEST_F(Test__JitFusedAttentionWo_Correctness, Qwen2_7B_Prefill_Medium)
    {
        run_parity_test(
            /*seq_len=*/64,
            /*kv_seq_len=*/64,
            /*num_heads=*/28,
            /*num_kv_heads=*/4,
            /*head_dim=*/128,
            "Qwen2 7B - Medium Prefill (seq=64)");
    }

    // ============================================================================
    // Qwen2 32B Configuration Tests (40 heads, 8 KV heads, head_dim=128)
    // ============================================================================

    TEST_F(Test__JitFusedAttentionWo_Correctness, Qwen2_32B_Decode_SingleToken)
    {
        run_parity_test(
            /*seq_len=*/1,
            /*kv_seq_len=*/64,
            /*num_heads=*/40,
            /*num_kv_heads=*/8,
            /*head_dim=*/128,
            "Qwen2 32B - Single Token Decode (seq=1, kv=64)");
    }

    TEST_F(Test__JitFusedAttentionWo_Correctness, Qwen2_32B_Prefill_Short)
    {
        run_parity_test(
            /*seq_len=*/8,
            /*kv_seq_len=*/8,
            /*num_heads=*/40,
            /*num_kv_heads=*/8,
            /*head_dim=*/128,
            "Qwen2 32B - Short Prefill (seq=8)");
    }

    // ============================================================================
    // Qwen2 72B Configuration Tests (64 heads, 8 KV heads, head_dim=128)
    // ============================================================================

    TEST_F(Test__JitFusedAttentionWo_Correctness, Qwen2_72B_Decode_SingleToken)
    {
        run_parity_test(
            /*seq_len=*/1,
            /*kv_seq_len=*/64,
            /*num_heads=*/64,
            /*num_kv_heads=*/8,
            /*head_dim=*/128,
            "Qwen2 72B - Single Token Decode (seq=1, kv=64)");
    }

    // ============================================================================
    // MHA (Multi-Head Attention) Tests - 1:1 head ratio
    // ============================================================================

    TEST_F(Test__JitFusedAttentionWo_Correctness, MHA_SmallConfig)
    {
        run_parity_test(
            /*seq_len=*/4,
            /*kv_seq_len=*/4,
            /*num_heads=*/4,
            /*num_kv_heads=*/4, // MHA: same number of KV heads
            /*head_dim=*/64,
            "MHA Small - 4 heads, head_dim=64");
    }

    TEST_F(Test__JitFusedAttentionWo_Correctness, MHA_MediumConfig)
    {
        run_parity_test(
            /*seq_len=*/16,
            /*kv_seq_len=*/16,
            /*num_heads=*/8,
            /*num_kv_heads=*/8,
            /*head_dim=*/64,
            "MHA Medium - 8 heads, head_dim=64");
    }

    // ============================================================================
    // GQA Ratio Tests - Various group sizes
    // ============================================================================

    TEST_F(Test__JitFusedAttentionWo_Correctness, GQA_Ratio_2to1)
    {
        run_parity_test(
            /*seq_len=*/8,
            /*kv_seq_len=*/8,
            /*num_heads=*/8,
            /*num_kv_heads=*/4, // 2:1 ratio
            /*head_dim=*/64,
            "GQA 2:1 - 8 Q heads, 4 KV heads");
    }

    TEST_F(Test__JitFusedAttentionWo_Correctness, GQA_Ratio_4to1)
    {
        run_parity_test(
            /*seq_len=*/8,
            /*kv_seq_len=*/8,
            /*num_heads=*/16,
            /*num_kv_heads=*/4, // 4:1 ratio
            /*head_dim=*/64,
            "GQA 4:1 - 16 Q heads, 4 KV heads");
    }

    TEST_F(Test__JitFusedAttentionWo_Correctness, GQA_Ratio_7to1)
    {
        // Qwen 0.5B uses 7:1 ratio (14 Q heads, 2 KV heads)
        run_parity_test(
            /*seq_len=*/8,
            /*kv_seq_len=*/8,
            /*num_heads=*/14,
            /*num_kv_heads=*/2, // 7:1 ratio
            /*head_dim=*/64,
            "GQA 7:1 - 14 Q heads, 2 KV heads (Qwen 0.5B style)");
    }

    TEST_F(Test__JitFusedAttentionWo_Correctness, GQA_Ratio_8to1)
    {
        // Qwen 72B uses 8:1 ratio (64 Q heads, 8 KV heads)
        run_parity_test(
            /*seq_len=*/4,
            /*kv_seq_len=*/4,
            /*num_heads=*/64,
            /*num_kv_heads=*/8, // 8:1 ratio
            /*head_dim=*/128,
            "GQA 8:1 - 64 Q heads, 8 KV heads (Qwen 72B style)");
    }

    // ============================================================================
    // Edge Case Tests
    // ============================================================================

    TEST_F(Test__JitFusedAttentionWo_Correctness, EdgeCase_SingleHead)
    {
        // Minimal configuration
        run_parity_test(
            /*seq_len=*/1,
            /*kv_seq_len=*/4,
            /*num_heads=*/2,
            /*num_kv_heads=*/2,
            /*head_dim=*/64,
            "Edge Case - 2 heads (minimal)");
    }

    TEST_F(Test__JitFusedAttentionWo_Correctness, EdgeCase_SingleKV)
    {
        // Single KV position
        run_parity_test(
            /*seq_len=*/1,
            /*kv_seq_len=*/1,
            /*num_heads=*/4,
            /*num_kv_heads=*/2,
            /*head_dim=*/64,
            "Edge Case - Single KV position");
    }

    TEST_F(Test__JitFusedAttentionWo_Correctness, EdgeCase_LongSequence)
    {
        // Longer sequence - tests memory handling
        run_parity_test(
            /*seq_len=*/64,
            /*kv_seq_len=*/256,
            /*num_heads=*/8,
            /*num_kv_heads=*/2,
            /*head_dim=*/64,
            "Edge Case - Long sequence (seq=64, kv=256)");
    }

    // ============================================================================
    // Head Dimension Tests
    // ============================================================================

    TEST_F(Test__JitFusedAttentionWo_Correctness, HeadDim_64)
    {
        run_parity_test(
            /*seq_len=*/4,
            /*kv_seq_len=*/8,
            /*num_heads=*/4,
            /*num_kv_heads=*/2,
            /*head_dim=*/64,
            "Head Dimension - 64 (Qwen 0.5B/1.5B style)");
    }

    TEST_F(Test__JitFusedAttentionWo_Correctness, HeadDim_128)
    {
        run_parity_test(
            /*seq_len=*/4,
            /*kv_seq_len=*/8,
            /*num_heads=*/4,
            /*num_kv_heads=*/2,
            /*head_dim=*/128,
            "Head Dimension - 128 (Qwen 7B/32B/72B style)");
    }

    // ============================================================================
    // Stress Tests (Optional - can be slow)
    // ============================================================================

    TEST_F(Test__JitFusedAttentionWo_Correctness, DISABLED_Stress_VeryLongSequence)
    {
        // Very long sequence - disabled by default
        run_parity_test(
            /*seq_len=*/256,
            /*kv_seq_len=*/1024,
            /*num_heads=*/14,
            /*num_kv_heads=*/2,
            /*head_dim=*/64,
            "Stress - Very long sequence (seq=256, kv=1024)");
    }

    TEST_F(Test__JitFusedAttentionWo_Correctness, DISABLED_Stress_ManyHeads)
    {
        // Many heads - disabled by default
        run_parity_test(
            /*seq_len=*/16,
            /*kv_seq_len=*/64,
            /*num_heads=*/64,
            /*num_kv_heads=*/8,
            /*head_dim=*/128,
            "Stress - Many heads (64 Q heads, 8 KV heads)");
    }

    // Test with single KV position - eliminates online softmax error accumulation
    TEST_F(Test__JitFusedAttentionWo_Correctness, SingleKVPosition_MHA)
    {
        run_parity_test(
            /*seq_len=*/1,
            /*kv_seq_len=*/1, // Just 1 KV position
            /*num_heads=*/2,  // Simple MHA
            /*num_kv_heads=*/2,
            /*head_dim=*/64,
            "Single KV Position (MHA, heads=2, kv=1)");
    }

} // namespace llaminar::v2::kernels::jit::test
