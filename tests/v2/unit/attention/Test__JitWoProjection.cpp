/**
 * @file Test__JitWoProjection.cpp
 * @brief Unit tests for JIT Wo projection in fused attention kernel
 * @author David Sanftenberg
 * @date December 2025
 *
 * Tests the Wo projection implementation in JitFusedAttentionWo:
 *   - FP32 Wo projection
 *   - FP16 Wo projection
 *   - BF16 Wo projection
 *   - Q8_1 Wo projection
 *
 * Validates:
 * - Correct Wo projection (output = context * Wo^T)
 * - Parity with REFERENCE backend using same inputs
 * - Non-identity Wo weights (real projections, not pass-through)
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
#include "tensors/SIMDHelpers.h"

namespace llaminar::v2::kernels::jit::test
{

    using llaminar::v2::kernels::FusedAttentionWoParams;
    using llaminar::v2::kernels::FusedAttentionWoRef;
    using llaminar::v2::kernels::microkernels::WoWeightType;
    using llaminar2::fp16_to_fp32;
    using llaminar2::fp32_to_fp16;
    using llaminar2::simd::bf16_to_fp32;
    using llaminar2::simd::fp32_to_bf16;
    using microkernels::Q8_1Block;

    // ============================================================================
    // Test Fixture
    // ============================================================================

    class Test__JitWoProjection : public ::testing::Test
    {
    protected:
        std::mt19937 gen_{42};
        std::uniform_real_distribution<float> dist_{-0.5f, 0.5f};

        // Correctness thresholds - tighter now that JIT applies Wo projection
        static constexpr double MIN_COSINE_SIM = 0.990;
        static constexpr double MAX_REL_L2_ERROR = 0.10;
        static constexpr double MAX_ABS_ERROR_PER_ELEM = 0.5f;

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
                        scale = 1e-10f;
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
         * @brief Generate random FP32 data
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
         * @brief Convert FP32 to FP16
         */
        std::vector<uint16_t> fp32_to_fp16_vec(const std::vector<float> &fp32)
        {
            std::vector<uint16_t> fp16(fp32.size());
            for (size_t i = 0; i < fp32.size(); ++i)
            {
                fp16[i] = fp32_to_fp16(fp32[i]);
            }
            return fp16;
        }

        /**
         * @brief Convert FP32 to BF16
         */
        std::vector<uint16_t> fp32_to_bf16_vec(const std::vector<float> &fp32)
        {
            std::vector<uint16_t> bf16(fp32.size());
            for (size_t i = 0; i < fp32.size(); ++i)
            {
                bf16[i] = fp32_to_bf16(fp32[i]);
            }
            return bf16;
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
         * @brief Run JIT vs REFERENCE parity test with real Wo weights
         */
        void run_wo_parity_test(
            int seq_len,
            int kv_seq_len,
            int num_heads,
            int num_kv_heads,
            int head_dim,
            WoFormat wo_format,
            const std::string &test_name)
        {
            const int d_model = num_heads * head_dim;
            const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
            const int blocks_per_head = head_dim / 32;

            std::cout << "\n=== " << test_name << " ===" << std::endl;
            std::cout << "  seq_len=" << seq_len << ", kv_seq_len=" << kv_seq_len
                      << ", heads=" << num_heads << "/" << num_kv_heads
                      << ", head_dim=" << head_dim << ", d_model=" << d_model
                      << ", wo_format=" << static_cast<int>(wo_format) << std::endl;

            // Generate random FP32 data for Q, K, V
            auto Q_fp32 = generate_random_fp32(seq_len * num_heads * head_dim);
            auto K_fp32 = generate_random_fp32(kv_seq_len * num_kv_heads * head_dim);
            auto V_fp32 = generate_random_fp32(kv_seq_len * num_kv_heads * head_dim);

            // Generate random Wo weights (NOT identity!)
            // Wo layout: [d_model, d_model]
            auto Wo_fp32 = generate_random_fp32(d_model * d_model, 0.1f); // Smaller scale for stability

            // Quantize Q, K, V to Q8_1
            std::vector<Q8_1Block> Q_q8(seq_len * num_heads * blocks_per_head);
            std::vector<Q8_1Block> K_q8(kv_seq_len * num_kv_heads * blocks_per_head);
            std::vector<Q8_1Block> V_q8(kv_seq_len * num_kv_heads * blocks_per_head);

            quantize_fp32_to_q8_1(Q_fp32.data(), seq_len * num_heads, head_dim, Q_q8.data());
            quantize_fp32_to_q8_1(K_fp32.data(), kv_seq_len * num_kv_heads, head_dim, K_q8.data());
            quantize_fp32_to_q8_1(V_fp32.data(), kv_seq_len * num_kv_heads, head_dim, V_q8.data());

            // Convert Wo to target format
            std::vector<uint16_t> Wo_fp16;
            std::vector<uint16_t> Wo_bf16;
            std::vector<Q8_1Block> Wo_q8;
            const void *wo_data = nullptr;
            WoWeightType wo_type;

            switch (wo_format)
            {
            case WoFormat::FP32:
                wo_data = Wo_fp32.data();
                wo_type = WoWeightType::FP32;
                break;
            case WoFormat::FP16:
                Wo_fp16 = fp32_to_fp16_vec(Wo_fp32);
                wo_data = Wo_fp16.data();
                wo_type = WoWeightType::FP16;
                break;
            case WoFormat::BF16:
                Wo_bf16 = fp32_to_bf16_vec(Wo_fp32);
                wo_data = Wo_bf16.data();
                wo_type = WoWeightType::BF16;
                break;
            case WoFormat::Q8_1:
                // Quantize Wo to Q8_1: [d_model, d_model] -> [d_model * d_model / 32] blocks
                Wo_q8.resize(d_model * d_model / 32);
                quantize_fp32_to_q8_1(Wo_fp32.data(), d_model, d_model, Wo_q8.data());
                wo_data = Wo_q8.data();
                wo_type = WoWeightType::Q8_1;
                break;
            }

            // Output buffers
            std::vector<float> output_jit(seq_len * d_model, 0.0f);
            std::vector<float> output_ref(seq_len * d_model, 0.0f);

            // Run REFERENCE kernel
            FusedAttentionWoParams ref_params;
            ref_params.Q = Q_q8.data();
            ref_params.K = K_q8.data();
            ref_params.V = V_q8.data();
            ref_params.Wo = wo_data;
            ref_params.wo_type = wo_type;
            ref_params.output = output_ref.data();
            ref_params.batch_size = 1;
            ref_params.seq_len = seq_len;
            ref_params.kv_seq_len = kv_seq_len;
            ref_params.num_heads = num_heads;
            ref_params.num_kv_heads = num_kv_heads;
            ref_params.head_dim = head_dim;
            ref_params.d_model = d_model;
            ref_params.scale = scale;
            ref_params.causal = false;
            ref_params.position_offset = 0;

            bool ref_ok = FusedAttentionWoRef::execute(ref_params);
            ASSERT_TRUE(ref_ok) << "Reference kernel failed";

            // Run JIT kernel
            JitAttentionConfig jit_config;
            jit_config.head_dim = head_dim;
            jit_config.num_heads = num_heads;
            jit_config.num_kv_heads = num_kv_heads;
            jit_config.batch_size = 1;
            jit_config.wo_format = wo_format;
            jit_config.causal = false; // Non-causal for these tests

            JitFusedAttentionWo jit_kernel(jit_config);
            jit_kernel.compute(
                Q_q8.data(),
                K_q8.data(),
                V_q8.data(),
                wo_data,
                output_jit.data(),
                seq_len,
                kv_seq_len,
                scale,
                0); // position_offset = 0

            // Compare outputs
            int output_size = seq_len * d_model;
            double cos_sim = cosine_similarity(output_jit.data(), output_ref.data(), output_size);
            double rel_l2 = relative_l2_error(output_jit.data(), output_ref.data(), output_size);
            double max_err = max_abs_error(output_jit.data(), output_ref.data(), output_size);

            std::cout << "  Cosine similarity:  " << std::fixed << std::setprecision(6) << cos_sim << std::endl;
            std::cout << "  Relative L2 error:  " << std::fixed << std::setprecision(6) << rel_l2 << std::endl;
            std::cout << "  Max absolute error: " << std::fixed << std::setprecision(6) << max_err << std::endl;

            // Debug output on failure
            if (std::isnan(cos_sim) || cos_sim < MIN_COSINE_SIM || rel_l2 > MAX_REL_L2_ERROR)
            {
                std::cout << "\n  First 16 JIT outputs: ";
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
            }

            // Validate
            EXPECT_GE(cos_sim, MIN_COSINE_SIM)
                << "Cosine similarity too low: " << cos_sim << " < " << MIN_COSINE_SIM;
            EXPECT_LE(rel_l2, MAX_REL_L2_ERROR)
                << "Relative L2 error too high: " << rel_l2 << " > " << MAX_REL_L2_ERROR;
        }
    };

    // ============================================================================
    // FP32 Wo Projection Tests
    // ============================================================================

    TEST_F(Test__JitWoProjection, FP32_Wo_SingleToken_Qwen05B)
    {
        // Qwen2.5 0.5B: 14 heads, 2 KV heads, head_dim=64
        run_wo_parity_test(1, 8, 14, 2, 64, WoFormat::FP32, "FP32 Wo - Single Token Qwen 0.5B");
    }

    TEST_F(Test__JitWoProjection, FP32_Wo_ShortSequence_Qwen05B)
    {
        run_wo_parity_test(4, 16, 14, 2, 64, WoFormat::FP32, "FP32 Wo - Short Sequence Qwen 0.5B");
    }

    TEST_F(Test__JitWoProjection, FP32_Wo_MediumSequence_Qwen05B)
    {
        run_wo_parity_test(8, 32, 14, 2, 64, WoFormat::FP32, "FP32 Wo - Medium Sequence Qwen 0.5B");
    }

    // ============================================================================
    // FP16 Wo Projection Tests
    // ============================================================================

    TEST_F(Test__JitWoProjection, FP16_Wo_SingleToken_Qwen05B)
    {
        run_wo_parity_test(1, 8, 14, 2, 64, WoFormat::FP16, "FP16 Wo - Single Token Qwen 0.5B");
    }

    TEST_F(Test__JitWoProjection, FP16_Wo_ShortSequence_Qwen05B)
    {
        run_wo_parity_test(4, 16, 14, 2, 64, WoFormat::FP16, "FP16 Wo - Short Sequence Qwen 0.5B");
    }

    // ============================================================================
    // BF16 Wo Projection Tests
    // ============================================================================

    TEST_F(Test__JitWoProjection, BF16_Wo_SingleToken_Qwen05B)
    {
        run_wo_parity_test(1, 8, 14, 2, 64, WoFormat::BF16, "BF16 Wo - Single Token Qwen 0.5B");
    }

    TEST_F(Test__JitWoProjection, BF16_Wo_ShortSequence_Qwen05B)
    {
        run_wo_parity_test(4, 16, 14, 2, 64, WoFormat::BF16, "BF16 Wo - Short Sequence Qwen 0.5B");
    }

    // ============================================================================
    // Q8_1 Wo Projection Tests
    // ============================================================================

    TEST_F(Test__JitWoProjection, Q8_1_Wo_SingleToken_Qwen05B)
    {
        run_wo_parity_test(1, 8, 14, 2, 64, WoFormat::Q8_1, "Q8_1 Wo - Single Token Qwen 0.5B");
    }

    TEST_F(Test__JitWoProjection, Q8_1_Wo_ShortSequence_Qwen05B)
    {
        run_wo_parity_test(4, 16, 14, 2, 64, WoFormat::Q8_1, "Q8_1 Wo - Short Sequence Qwen 0.5B");
    }

    // ============================================================================
    // GQA Tests (different head ratios)
    // ============================================================================

    TEST_F(Test__JitWoProjection, FP32_Wo_GQA_7to1)
    {
        // 14 Q heads, 2 KV heads = 7:1 ratio
        run_wo_parity_test(2, 8, 14, 2, 64, WoFormat::FP32, "FP32 Wo - GQA 7:1");
    }

    TEST_F(Test__JitWoProjection, FP32_Wo_MHA)
    {
        // Equal Q and KV heads (standard MHA)
        run_wo_parity_test(2, 8, 8, 8, 64, WoFormat::FP32, "FP32 Wo - MHA 1:1");
    }

    // ============================================================================
    // Prefill/Batching Tests (seq_len_q > 1)
    // ============================================================================

    TEST_F(Test__JitWoProjection, FP32_Wo_Prefill_16)
    {
        // Prefill with 16 query positions
        run_wo_parity_test(16, 16, 14, 2, 64, WoFormat::FP32, "FP32 Wo - Prefill seq_len=16");
    }

    TEST_F(Test__JitWoProjection, FP32_Wo_Prefill_32)
    {
        // Prefill with 32 query positions
        run_wo_parity_test(32, 32, 14, 2, 64, WoFormat::FP32, "FP32 Wo - Prefill seq_len=32");
    }

    TEST_F(Test__JitWoProjection, FP32_Wo_Prefill_64)
    {
        // Prefill with 64 query positions (typical short prompt)
        run_wo_parity_test(64, 64, 14, 2, 64, WoFormat::FP32, "FP32 Wo - Prefill seq_len=64");
    }

    TEST_F(Test__JitWoProjection, FP16_Wo_Prefill_32)
    {
        // Prefill with FP16 Wo weights
        run_wo_parity_test(32, 32, 14, 2, 64, WoFormat::FP16, "FP16 Wo - Prefill seq_len=32");
    }

    TEST_F(Test__JitWoProjection, BF16_Wo_Prefill_32)
    {
        // Prefill with BF16 Wo weights
        run_wo_parity_test(32, 32, 14, 2, 64, WoFormat::BF16, "BF16 Wo - Prefill seq_len=32");
    }

    TEST_F(Test__JitWoProjection, Q8_1_Wo_Prefill_32)
    {
        // Prefill with Q8_1 Wo weights
        run_wo_parity_test(32, 32, 14, 2, 64, WoFormat::Q8_1, "Q8_1 Wo - Prefill seq_len=32");
    }

    TEST_F(Test__JitWoProjection, FP32_Wo_Prefill_128_LongContext)
    {
        // Longer context for realistic prefill scenario
        run_wo_parity_test(128, 128, 14, 2, 64, WoFormat::FP32, "FP32 Wo - Prefill seq_len=128");
    }

    // ============================================================================
    // Larger Model Configurations
    // ============================================================================

    TEST_F(Test__JitWoProjection, FP32_Wo_LargerHeadDim)
    {
        // head_dim=128 (like Qwen 7B+)
        run_wo_parity_test(1, 8, 8, 2, 128, WoFormat::FP32, "FP32 Wo - head_dim=128");
    }

    TEST_F(Test__JitWoProjection, FP32_Wo_LargerHeadDim_Prefill)
    {
        // head_dim=128 with prefill
        run_wo_parity_test(16, 16, 8, 2, 128, WoFormat::FP32, "FP32 Wo - head_dim=128 Prefill");
    }

    // ============================================================================
    // Causal Masking Tests
    // ============================================================================

    /**
     * @brief Run causal masking parity test between JIT and REFERENCE kernels
     *
     * For causal attention, query at position q can only attend to KV positions [0, q + position_offset].
     * This test verifies that JIT produces the same output as REFERENCE with causal=true.
     */
    void run_causal_parity_test(
        int seq_len,
        int kv_seq_len,
        int num_heads,
        int num_kv_heads,
        int head_dim,
        int position_offset,
        WoFormat wo_format,
        const std::string &test_name)
    {
        std::mt19937 gen{42};
        std::uniform_real_distribution<float> dist{-0.5f, 0.5f};

        const int d_model = num_heads * head_dim;
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        const int blocks_per_head = head_dim / 32;

        std::cout << "\n=== " << test_name << " ===" << std::endl;
        std::cout << "  seq_len=" << seq_len << ", kv_seq_len=" << kv_seq_len
                  << ", heads=" << num_heads << "/" << num_kv_heads
                  << ", head_dim=" << head_dim << ", position_offset=" << position_offset
                  << ", wo_format=" << static_cast<int>(wo_format) << std::endl;

        // Generate random FP32 data for Q, K, V
        std::vector<float> Q_fp32(seq_len * num_heads * head_dim);
        std::vector<float> K_fp32(kv_seq_len * num_kv_heads * head_dim);
        std::vector<float> V_fp32(kv_seq_len * num_kv_heads * head_dim);
        for (auto &v : Q_fp32)
            v = dist(gen);
        for (auto &v : K_fp32)
            v = dist(gen);
        for (auto &v : V_fp32)
            v = dist(gen);

        // Generate random Wo weights
        std::vector<float> Wo_fp32(d_model * d_model);
        for (auto &v : Wo_fp32)
            v = dist(gen) * 0.1f;

        // Quantize Q, K, V to Q8_1
        std::vector<Q8_1Block> Q_q8(seq_len * num_heads * blocks_per_head);
        std::vector<Q8_1Block> K_q8(kv_seq_len * num_kv_heads * blocks_per_head);
        std::vector<Q8_1Block> V_q8(kv_seq_len * num_kv_heads * blocks_per_head);

        auto quantize = [](const float *fp32_data, int rows, int cols, Q8_1Block *blocks)
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
                        max_abs = std::max(max_abs, std::fabs(block_data[i]));
                    float scale_val = max_abs / 127.0f;
                    if (scale_val < 1e-10f)
                        scale_val = 1e-10f;
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
                    blk.d = fp32_to_fp16(scale_val);
                    blk.sum_qs = static_cast<int16_t>(sum_qs);
                }
            }
        };

        quantize(Q_fp32.data(), seq_len * num_heads, head_dim, Q_q8.data());
        quantize(K_fp32.data(), kv_seq_len * num_kv_heads, head_dim, K_q8.data());
        quantize(V_fp32.data(), kv_seq_len * num_kv_heads, head_dim, V_q8.data());

        // Convert Wo to target format
        std::vector<uint16_t> Wo_fp16;
        std::vector<uint16_t> Wo_bf16;
        std::vector<Q8_1Block> Wo_q8;
        const void *wo_data = nullptr;
        WoWeightType wo_type;

        switch (wo_format)
        {
        case WoFormat::FP32:
            wo_data = Wo_fp32.data();
            wo_type = WoWeightType::FP32;
            break;
        case WoFormat::FP16:
            Wo_fp16.resize(Wo_fp32.size());
            for (size_t i = 0; i < Wo_fp32.size(); ++i)
                Wo_fp16[i] = fp32_to_fp16(Wo_fp32[i]);
            wo_data = Wo_fp16.data();
            wo_type = WoWeightType::FP16;
            break;
        case WoFormat::BF16:
            Wo_bf16.resize(Wo_fp32.size());
            for (size_t i = 0; i < Wo_fp32.size(); ++i)
                Wo_bf16[i] = fp32_to_bf16(Wo_fp32[i]);
            wo_data = Wo_bf16.data();
            wo_type = WoWeightType::BF16;
            break;
        case WoFormat::Q8_1:
            Wo_q8.resize(d_model * d_model / 32);
            quantize(Wo_fp32.data(), d_model, d_model, Wo_q8.data());
            wo_data = Wo_q8.data();
            wo_type = WoWeightType::Q8_1;
            break;
        }

        // Output buffers
        std::vector<float> output_jit(seq_len * d_model, 0.0f);
        std::vector<float> output_ref(seq_len * d_model, 0.0f);

        // Run REFERENCE kernel with causal=true
        FusedAttentionWoParams ref_params;
        ref_params.Q = Q_q8.data();
        ref_params.K = K_q8.data();
        ref_params.V = V_q8.data();
        ref_params.Wo = wo_data;
        ref_params.wo_type = wo_type;
        ref_params.output = output_ref.data();
        ref_params.batch_size = 1;
        ref_params.seq_len = seq_len;
        ref_params.kv_seq_len = kv_seq_len;
        ref_params.num_heads = num_heads;
        ref_params.num_kv_heads = num_kv_heads;
        ref_params.head_dim = head_dim;
        ref_params.d_model = d_model;
        ref_params.scale = scale;
        ref_params.causal = true; // Causal masking enabled
        ref_params.position_offset = position_offset;

        bool ref_ok = FusedAttentionWoRef::execute(ref_params);
        ASSERT_TRUE(ref_ok) << "Reference kernel failed";

        // Run JIT kernel with causal=true
        JitAttentionConfig jit_config;
        jit_config.head_dim = head_dim;
        jit_config.num_heads = num_heads;
        jit_config.num_kv_heads = num_kv_heads;
        jit_config.batch_size = 1;
        jit_config.wo_format = wo_format;
        jit_config.causal = true; // Causal masking enabled

        JitFusedAttentionWo jit_kernel(jit_config);
        jit_kernel.compute(
            Q_q8.data(),
            K_q8.data(),
            V_q8.data(),
            wo_data,
            output_jit.data(),
            seq_len,
            kv_seq_len,
            scale,
            position_offset);

        // Compare outputs
        int output_size = seq_len * d_model;
        double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
        for (int i = 0; i < output_size; ++i)
        {
            dot += output_jit[i] * output_ref[i];
            norm_a += output_jit[i] * output_jit[i];
            norm_b += output_ref[i] * output_ref[i];
        }
        double cos_sim = (norm_a > 1e-10 && norm_b > 1e-10) ? dot / (std::sqrt(norm_a) * std::sqrt(norm_b)) : 1.0;

        double sum_sq_err = 0.0, sum_sq_exp = 0.0;
        for (int i = 0; i < output_size; ++i)
        {
            double err = output_jit[i] - output_ref[i];
            sum_sq_err += err * err;
            sum_sq_exp += output_ref[i] * output_ref[i];
        }
        double rel_l2 = (sum_sq_exp > 1e-10) ? std::sqrt(sum_sq_err / sum_sq_exp) : 0.0;

        double max_err = 0.0;
        for (int i = 0; i < output_size; ++i)
        {
            max_err = std::max(max_err, static_cast<double>(std::fabs(output_jit[i] - output_ref[i])));
        }

        std::cout << "  Cosine similarity:  " << std::fixed << std::setprecision(6) << cos_sim << std::endl;
        std::cout << "  Relative L2 error:  " << std::fixed << std::setprecision(6) << rel_l2 << std::endl;
        std::cout << "  Max absolute error: " << std::fixed << std::setprecision(6) << max_err << std::endl;

        // Validate
        EXPECT_GE(cos_sim, 0.990) << "Cosine similarity too low for causal test";
        EXPECT_LE(rel_l2, 0.10) << "Relative L2 error too high for causal test";
    }

    TEST_F(Test__JitWoProjection, Causal_SingleToken_Decode)
    {
        // Single token decode: seq_len=1, position_offset shows where we are in KV cache
        // Query can only see KV positions [0, position_offset]
        run_causal_parity_test(1, 16, 14, 2, 64, 15, WoFormat::FP32, "Causal - Single Token Decode (pos=15)");
    }

    TEST_F(Test__JitWoProjection, Causal_SingleToken_Decode_Early)
    {
        // Early position in decode
        run_causal_parity_test(1, 16, 14, 2, 64, 3, WoFormat::FP32, "Causal - Single Token Decode (pos=3)");
    }

    TEST_F(Test__JitWoProjection, Causal_Prefill)
    {
        // Causal prefill: each query sees only previous positions
        // Query 0 sees KV[0], Query 1 sees KV[0:1], etc.
        run_causal_parity_test(8, 8, 14, 2, 64, 0, WoFormat::FP32, "Causal - Prefill seq_len=8");
    }

    TEST_F(Test__JitWoProjection, Causal_Prefill_Long)
    {
        // Longer causal prefill
        run_causal_parity_test(32, 32, 14, 2, 64, 0, WoFormat::FP32, "Causal - Prefill seq_len=32");
    }

    TEST_F(Test__JitWoProjection, Causal_LongContext_Decode)
    {
        // Decode after long prefill (lots of KV cache)
        run_causal_parity_test(1, 128, 14, 2, 64, 127, WoFormat::FP32, "Causal - Long Context Decode");
    }

    TEST_F(Test__JitWoProjection, Causal_MultiToken_Decode)
    {
        // Multi-token decode (speculative decoding scenario)
        // seq_len=4, position_offset=100 means queries at positions 100, 101, 102, 103
        run_causal_parity_test(4, 104, 14, 2, 64, 100, WoFormat::FP32, "Causal - Multi-Token Decode");
    }

    TEST_F(Test__JitWoProjection, Causal_Q8_1_Wo)
    {
        // Causal with Q8_1 Wo weights
        run_causal_parity_test(8, 16, 14, 2, 64, 8, WoFormat::Q8_1, "Causal - Q8_1 Wo");
    }

    TEST_F(Test__JitWoProjection, Causal_BF16_Wo)
    {
        // Causal with BF16 Wo weights
        run_causal_parity_test(8, 16, 14, 2, 64, 8, WoFormat::BF16, "Causal - BF16 Wo");
    }

} // namespace llaminar::v2::kernels::jit::test
