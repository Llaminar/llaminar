/**
 * @file Test__JitFusedAttentionWo_Q16_1Fused_vs_FP32.cpp
 * @brief Kernel Parity: JitFusedAttentionWo with Q16_1 Fused Residual vs FP32 Reference
 *
 * @category e2e/parity/kernels/attention
 * @tested   JitFusedAttentionWo with fuse_residual_add=true, residual_type=Q16_1
 * @reference FP32 decomposed attention + Wo projection + residual addition
 *
 * This test validates the Phase 5 Q16_1 Typed Residual fusion implementation
 * where the attention output + Wo projection + residual addition is fused
 * into a single JIT kernel, avoiding intermediate quantization.
 *
 * Flow being tested:
 *   JIT Kernel: Q/K/V → Attention → Context → Wo GEMM (FP32) → Add Q16_1 Residual → Q16_1 Output
 *
 * Reference:
 *   FP32: Q/K/V → Attention → Context → Wo GEMM → Add Residual → FP32 Output
 *
 * Comparison:
 *   Dequantize JIT Q16_1 output to FP32, compare against FP32 reference
 *
 * @author GitHub Copilot
 * @date December 2025
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>
#include <iostream>
#include <cstring>

#include "kernels/cpu/attention/q8_1/jit/JitFusedAttentionWo.h"
#include "tensors/BlockStructures.h"
#include "tensors/SIMDHelpers.h"
#include "utils/Logger.h"

namespace
{
    using llaminar::v2::kernels::jit::JitAttentionConfig;
    using llaminar::v2::kernels::jit::JitFusedAttentionWo;
    using llaminar::v2::kernels::jit::WoFormat;
    using llaminar2::Q16_1Block;
    using llaminar2::Q8_1Block;
    using llaminar2::simd::fp16_to_fp32;
    using llaminar2::simd::fp32_to_fp16;

    // ============================================================================
    // Q16_1 Block Constants
    // ============================================================================
    constexpr int Q16_1_BLOCK_SIZE = 32;   // Elements per block
    constexpr int Q16_1_BLOCK_BYTES = 72;  // 4 (d) + 4 (sum_qs) + 64 (qs[32])
    constexpr float Q16_1_QMAX = 32767.0f; // Max quantized value

    // ============================================================================
    // Utility Functions
    // ============================================================================

    double cosine_similarity(const float *a, const float *b, size_t n)
    {
        double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
            norm_a += static_cast<double>(a[i]) * static_cast<double>(a[i]);
            norm_b += static_cast<double>(b[i]) * static_cast<double>(b[i]);
        }
        if (norm_a < 1e-12 || norm_b < 1e-12)
            return 0.0;
        return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
    }

    double max_abs_diff(const float *a, const float *b, size_t n)
    {
        double max_diff = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            double diff = std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
            max_diff = std::max(max_diff, diff);
        }
        return max_diff;
    }

    double mean_abs_diff(const float *a, const float *b, size_t n)
    {
        double sum = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            sum += std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        }
        return sum / static_cast<double>(n);
    }

    void generate_random_data(float *data, size_t n, float scale, unsigned seed)
    {
        std::mt19937 gen(seed);
        std::normal_distribution<float> dist(0.0f, scale);
        for (size_t i = 0; i < n; ++i)
        {
            data[i] = dist(gen);
        }
    }

    void quantize_fp32_to_q8_1(const float *fp32, int num_rows, int row_dim, Q8_1Block *blocks)
    {
        int num_blocks = (row_dim + 31) / 32;
        for (int row = 0; row < num_rows; ++row)
        {
            for (int b = 0; b < num_blocks; ++b)
            {
                Q8_1Block &block = blocks[row * num_blocks + b];
                const float *src = fp32 + row * row_dim + b * 32;

                float max_abs = 0.0f;
                for (int i = 0; i < 32 && (b * 32 + i) < row_dim; ++i)
                {
                    max_abs = std::max(max_abs, std::abs(src[i]));
                }

                float d = max_abs / 127.0f;
                float inv_d = (d > 1e-10f) ? (1.0f / d) : 0.0f;
                block.d = fp32_to_fp16(d);

                int16_t sum_qs = 0;
                for (int i = 0; i < 32; ++i)
                {
                    float val = (b * 32 + i < row_dim) ? src[i] : 0.0f;
                    int8_t q = static_cast<int8_t>(std::round(std::clamp(val * inv_d, -127.0f, 127.0f)));
                    block.qs[i] = q;
                    sum_qs += q;
                }
                block.sum_qs = sum_qs;
            }
        }
    }

    /**
     * @brief Quantize FP32 data to Q16_1 format
     */
    void quantize_fp32_to_q16_1(const float *fp32, int num_elements, Q16_1Block *blocks)
    {
        int num_blocks = (num_elements + Q16_1_BLOCK_SIZE - 1) / Q16_1_BLOCK_SIZE;

        for (int b = 0; b < num_blocks; ++b)
        {
            Q16_1Block &block = blocks[b];
            const float *src = fp32 + b * Q16_1_BLOCK_SIZE;
            int block_elems = std::min(Q16_1_BLOCK_SIZE, num_elements - b * Q16_1_BLOCK_SIZE);

            // Find max_abs in block
            float max_abs = 0.0f;
            for (int i = 0; i < block_elems; ++i)
            {
                max_abs = std::max(max_abs, std::abs(src[i]));
            }

            // Compute scale = max_abs / 32767
            block.d = (max_abs > 1e-10f) ? (max_abs / Q16_1_QMAX) : 1e-10f;
            float inv_d = Q16_1_QMAX / (max_abs > 1e-10f ? max_abs : 1e-10f);

            // Quantize to int16
            int32_t sum_qs = 0;
            for (int i = 0; i < Q16_1_BLOCK_SIZE; ++i)
            {
                float val = (i < block_elems) ? src[i] : 0.0f;
                int16_t q = static_cast<int16_t>(std::round(std::clamp(val * inv_d, -Q16_1_QMAX, Q16_1_QMAX)));
                block.qs[i] = q;
                sum_qs += q;
            }
            block.sum_qs = sum_qs;
        }
    }

    /**
     * @brief Dequantize Q16_1 blocks to FP32
     */
    void dequantize_q16_1_to_fp32(const Q16_1Block *blocks, int num_elements, float *fp32)
    {
        int num_blocks = (num_elements + Q16_1_BLOCK_SIZE - 1) / Q16_1_BLOCK_SIZE;

        for (int b = 0; b < num_blocks; ++b)
        {
            const Q16_1Block &block = blocks[b];
            float *dst = fp32 + b * Q16_1_BLOCK_SIZE;
            int block_elems = std::min(Q16_1_BLOCK_SIZE, num_elements - b * Q16_1_BLOCK_SIZE);

            for (int i = 0; i < block_elems; ++i)
            {
                dst[i] = static_cast<float>(block.qs[i]) * block.d;
            }
        }
    }

    /**
     * @brief TRUE FP32 attention reference (no quantization)
     *
     * Implements: output = softmax(Q @ K^T * scale + mask) @ V
     */
    void compute_fp32_attention_reference(
        int seq_len,
        int kv_seq_len,
        int num_heads,
        int num_kv_heads,
        int head_dim,
        const float *Q_fp32,
        const float *K_fp32,
        const float *V_fp32,
        float scale,
        float *context_fp32, // [seq_len, num_heads * head_dim]
        bool causal)
    {
        int gqa_ratio = num_heads / num_kv_heads;

        for (int q_pos = 0; q_pos < seq_len; ++q_pos)
        {
            for (int h = 0; h < num_heads; ++h)
            {
                int kv_head = h / gqa_ratio;
                const float *Q_head = Q_fp32 + q_pos * num_heads * head_dim + h * head_dim;

                // Step 1: Compute Q @ K^T scores
                std::vector<float> scores(kv_seq_len);
                for (int k_pos = 0; k_pos < kv_seq_len; ++k_pos)
                {
                    const float *K_pos = K_fp32 + k_pos * num_kv_heads * head_dim + kv_head * head_dim;
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; ++d)
                    {
                        dot += Q_head[d] * K_pos[d];
                    }
                    scores[k_pos] = dot * scale;

                    if (causal && k_pos > q_pos)
                    {
                        scores[k_pos] = -std::numeric_limits<float>::infinity();
                    }
                }

                // Step 2: Softmax
                float max_score = *std::max_element(scores.begin(), scores.end());
                float sum_exp = 0.0f;
                for (int k = 0; k < kv_seq_len; ++k)
                {
                    scores[k] = std::exp(scores[k] - max_score);
                    sum_exp += scores[k];
                }
                for (int k = 0; k < kv_seq_len; ++k)
                {
                    scores[k] /= sum_exp;
                }

                // Step 3: scores @ V
                float *context_head = context_fp32 + q_pos * num_heads * head_dim + h * head_dim;
                for (int d = 0; d < head_dim; ++d)
                {
                    context_head[d] = 0.0f;
                }
                for (int k_pos = 0; k_pos < kv_seq_len; ++k_pos)
                {
                    const float *V_pos = V_fp32 + k_pos * num_kv_heads * head_dim + kv_head * head_dim;
                    for (int d = 0; d < head_dim; ++d)
                    {
                        context_head[d] += scores[k_pos] * V_pos[d];
                    }
                }
            }
        }
    }

    /**
     * @brief FP32 Wo projection reference
     *
     * Implements: output = context @ Wo^T
     * Where context is [seq_len, local_dim] and Wo is [d_model, local_dim]
     */
    void compute_fp32_wo_projection(
        int seq_len,
        int d_model,
        int local_dim,
        const float *context_fp32, // [seq_len, local_dim]
        const float *Wo_fp32,      // [d_model, local_dim]
        float *output_fp32)        // [seq_len, d_model]
    {
        for (int s = 0; s < seq_len; ++s)
        {
            const float *ctx_row = context_fp32 + s * local_dim;
            float *out_row = output_fp32 + s * d_model;

            for (int o = 0; o < d_model; ++o)
            {
                const float *wo_row = Wo_fp32 + o * local_dim;
                float sum = 0.0f;
                for (int k = 0; k < local_dim; ++k)
                {
                    sum += ctx_row[k] * wo_row[k];
                }
                out_row[o] = sum;
            }
        }
    }

    /**
     * @brief FP32 residual addition reference
     */
    void compute_fp32_residual_add(
        const float *wo_output_fp32,
        const float *residual_fp32,
        float *output_fp32,
        int num_elements)
    {
        for (int i = 0; i < num_elements; ++i)
        {
            output_fp32[i] = wo_output_fp32[i] + residual_fp32[i];
        }
    }

    // ============================================================================
    // Test Fixture
    // ============================================================================

    class Test__JitFusedAttentionWo_Q16_1Fused_vs_FP32 : public ::testing::Test
    {
    protected:
        // Thresholds for comparison
        // Q16_1 has much higher precision than Q8_1 (16-bit vs 8-bit quantization)
        // Expected: cosine > 0.999 for small data, > 0.995 for realistic data
        static constexpr double MIN_COSINE_EXCELLENT = 0.999;
        static constexpr double MIN_COSINE_GOOD = 0.995;
        static constexpr double MIN_COSINE_ACCEPTABLE = 0.99;

        void run_parity_test(
            int seq_len,
            int kv_seq_len,
            int num_heads,
            int num_kv_heads,
            int head_dim,
            float q_k_scale,
            float v_scale,
            float residual_scale,
            const std::string &test_name)
        {
            std::cout << "\n=== " << test_name << " ===" << std::endl;
            std::cout << "  Config: seq=" << seq_len << ", kv=" << kv_seq_len
                      << ", heads=" << num_heads << "/" << num_kv_heads
                      << ", head_dim=" << head_dim << std::endl;
            std::cout << "  Data scales: Q/K=±" << q_k_scale
                      << ", V=±" << v_scale
                      << ", residual=±" << residual_scale << std::endl;

            int d_model = num_heads * head_dim;
            int local_dim = num_heads * head_dim; // For non-sharded case
            float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

            // Generate FP32 data
            std::vector<float> Q_fp32(seq_len * num_heads * head_dim);
            std::vector<float> K_fp32(kv_seq_len * num_kv_heads * head_dim);
            std::vector<float> V_fp32(kv_seq_len * num_kv_heads * head_dim);
            std::vector<float> residual_fp32(seq_len * d_model);

            generate_random_data(Q_fp32.data(), Q_fp32.size(), q_k_scale, 100);
            generate_random_data(K_fp32.data(), K_fp32.size(), q_k_scale, 200);
            generate_random_data(V_fp32.data(), V_fp32.size(), v_scale, 300);
            generate_random_data(residual_fp32.data(), residual_fp32.size(), residual_scale, 400);

            // Print data statistics
            float q_max = *std::max_element(Q_fp32.begin(), Q_fp32.end());
            float q_min = *std::min_element(Q_fp32.begin(), Q_fp32.end());
            float res_max = *std::max_element(residual_fp32.begin(), residual_fp32.end());
            float res_min = *std::min_element(residual_fp32.begin(), residual_fp32.end());
            std::cout << "  Q range: [" << q_min << ", " << q_max << "]" << std::endl;
            std::cout << "  Residual range: [" << res_min << ", " << res_max << "]" << std::endl;

            // Quantize Q/K/V to Q8_1 for JIT kernel
            int blocks_per_head = (head_dim + 31) / 32;
            std::vector<Q8_1Block> Q_q8(seq_len * num_heads * blocks_per_head);
            std::vector<Q8_1Block> K_q8(kv_seq_len * num_kv_heads * blocks_per_head);
            std::vector<Q8_1Block> V_q8(kv_seq_len * num_kv_heads * blocks_per_head);

            quantize_fp32_to_q8_1(Q_fp32.data(), seq_len * num_heads, head_dim, Q_q8.data());
            quantize_fp32_to_q8_1(K_fp32.data(), kv_seq_len * num_kv_heads, head_dim, K_q8.data());
            quantize_fp32_to_q8_1(V_fp32.data(), kv_seq_len * num_kv_heads, head_dim, V_q8.data());

            // Quantize residual to Q16_1 for JIT kernel
            int q16_1_blocks_per_row = (d_model + Q16_1_BLOCK_SIZE - 1) / Q16_1_BLOCK_SIZE;
            std::vector<Q16_1Block> residual_q16_1(seq_len * q16_1_blocks_per_row);
            for (int s = 0; s < seq_len; ++s)
            {
                quantize_fp32_to_q16_1(
                    residual_fp32.data() + s * d_model,
                    d_model,
                    residual_q16_1.data() + s * q16_1_blocks_per_row);
            }

            // ═══════════════════════════════════════════════════════════════════════
            // FP32 Reference: Attention → Wo → Residual Add
            // ═══════════════════════════════════════════════════════════════════════

            // Step 1: FP32 attention context
            std::vector<float> context_fp32(seq_len * local_dim, 0.0f);
            compute_fp32_attention_reference(
                seq_len, kv_seq_len, num_heads, num_kv_heads, head_dim,
                Q_fp32.data(), K_fp32.data(), V_fp32.data(),
                scale, context_fp32.data(), false);

            // Step 2: FP32 Wo projection (identity for simplicity)
            std::vector<float> Wo_fp32(d_model * local_dim, 0.0f);
            for (int i = 0; i < std::min(d_model, local_dim); ++i)
            {
                Wo_fp32[i * local_dim + i] = 1.0f;
            }

            std::vector<float> wo_output_fp32(seq_len * d_model, 0.0f);
            compute_fp32_wo_projection(
                seq_len, d_model, local_dim,
                context_fp32.data(), Wo_fp32.data(),
                wo_output_fp32.data());

            // Step 3: FP32 residual add
            std::vector<float> reference_fp32(seq_len * d_model, 0.0f);
            compute_fp32_residual_add(
                wo_output_fp32.data(), residual_fp32.data(),
                reference_fp32.data(), seq_len * d_model);

            // ═══════════════════════════════════════════════════════════════════════
            // JIT Kernel: Fused Attention + Wo + Q16_1 Residual
            // ═══════════════════════════════════════════════════════════════════════

            // JIT kernel output is Q16_1 (same buffer as input residual in real use)
            std::vector<Q16_1Block> output_q16_1(seq_len * q16_1_blocks_per_row);

            // Copy input residual to output buffer (kernel reads residual, adds Wo, stores result)
            std::memcpy(output_q16_1.data(), residual_q16_1.data(),
                        output_q16_1.size() * sizeof(Q16_1Block));

            // Configure JIT kernel with Q16_1 residual fusion
            JitAttentionConfig jit_config;
            jit_config.head_dim = head_dim;
            jit_config.num_heads = num_heads;
            jit_config.num_kv_heads = num_kv_heads;
            jit_config.batch_size = 1;
            jit_config.d_model = d_model; // Required for Q16_1 residual fusion stride calculation
            jit_config.wo_format = WoFormat::FP32;
            jit_config.causal = false;
            jit_config.fuse_residual_add = true;
            jit_config.residual_type = JitAttentionConfig::ResidualType::Q16_1;

            JitFusedAttentionWo jit_kernel(jit_config);

            // Run JIT kernel - output goes to output_q16_1 buffer
            jit_kernel.compute(
                Q_q8.data(),
                K_q8.data(),
                V_q8.data(),
                Wo_fp32.data(),
                reinterpret_cast<float *>(output_q16_1.data()), // Q16_1 buffer
                seq_len,
                kv_seq_len,
                scale,
                0);

            // Dequantize JIT Q16_1 output to FP32 for comparison
            std::vector<float> jit_output_fp32(seq_len * d_model, 0.0f);
            for (int s = 0; s < seq_len; ++s)
            {
                dequantize_q16_1_to_fp32(
                    output_q16_1.data() + s * q16_1_blocks_per_row,
                    d_model,
                    jit_output_fp32.data() + s * d_model);
            }

            // ═══════════════════════════════════════════════════════════════════════
            // Compare Results
            // ═══════════════════════════════════════════════════════════════════════

            int output_size = seq_len * d_model;
            double cos_sim = cosine_similarity(jit_output_fp32.data(), reference_fp32.data(), output_size);
            double max_diff = max_abs_diff(jit_output_fp32.data(), reference_fp32.data(), output_size);
            double mean_diff = mean_abs_diff(jit_output_fp32.data(), reference_fp32.data(), output_size);

            std::cout << "  ────────────────────────────────────" << std::endl;
            std::cout << "  Cosine similarity: " << std::fixed << std::setprecision(6) << cos_sim;
            if (cos_sim >= MIN_COSINE_EXCELLENT)
                std::cout << " ✓ EXCELLENT";
            else if (cos_sim >= MIN_COSINE_GOOD)
                std::cout << " ✓ GOOD";
            else if (cos_sim >= MIN_COSINE_ACCEPTABLE)
                std::cout << " ✓ ACCEPTABLE";
            else
                std::cout << " ✗ DIVERGED";
            std::cout << std::endl;
            std::cout << "  Max abs difference: " << std::scientific << max_diff << std::endl;
            std::cout << "  Mean abs difference: " << std::scientific << mean_diff << std::endl;

            // Print first few outputs
            std::cout << "  First 8 FP32 ref: ";
            for (int i = 0; i < 8 && i < output_size; ++i)
                std::cout << std::fixed << std::setprecision(4) << reference_fp32[i] << " ";
            std::cout << std::endl;
            std::cout << "  First 8 JIT Q16_1: ";
            for (int i = 0; i < 8 && i < output_size; ++i)
                std::cout << std::fixed << std::setprecision(4) << jit_output_fp32[i] << " ";
            std::cout << std::endl;

            // Assert minimum acceptable similarity
            EXPECT_GE(cos_sim, MIN_COSINE_ACCEPTABLE)
                << "Q16_1 fused kernel diverged from FP32 reference";
        }
    };

    // ============================================================================
    // Tests with Small Data Scale
    // ============================================================================

    TEST_F(Test__JitFusedAttentionWo_Q16_1Fused_vs_FP32, SmallData_Decode_SingleQuery)
    {
        // Single query decode - tests emit_wo_projection_batched Q16_1 path
        run_parity_test(
            /*seq_len=*/1,
            /*kv_seq_len=*/16,
            /*num_heads=*/14,
            /*num_kv_heads=*/2,
            /*head_dim=*/64,
            /*q_k_scale=*/1.0f,
            /*v_scale=*/1.0f,
            /*residual_scale=*/1.0f,
            "SmallData_Decode_SingleQuery");
    }

    TEST_F(Test__JitFusedAttentionWo_Q16_1Fused_vs_FP32, SmallData_Prefill_Short)
    {
        // Short prefill sequence
        run_parity_test(
            /*seq_len=*/4,
            /*kv_seq_len=*/4,
            /*num_heads=*/14,
            /*num_kv_heads=*/2,
            /*head_dim=*/64,
            /*q_k_scale=*/1.0f,
            /*v_scale=*/1.0f,
            /*residual_scale=*/1.0f,
            "SmallData_Prefill_Short");
    }

    // ============================================================================
    // Tests with Realistic Data Magnitudes
    // ============================================================================

    TEST_F(Test__JitFusedAttentionWo_Q16_1Fused_vs_FP32, RealisticData_Decode)
    {
        // Realistic decode - larger KV cache, realistic value magnitudes
        run_parity_test(
            /*seq_len=*/1,
            /*kv_seq_len=*/64,
            /*num_heads=*/14,
            /*num_kv_heads=*/2,
            /*head_dim=*/64,
            /*q_k_scale=*/5.0f,
            /*v_scale=*/2.0f,
            /*residual_scale=*/3.0f,
            "RealisticData_Decode");
    }

    TEST_F(Test__JitFusedAttentionWo_Q16_1Fused_vs_FP32, RealisticData_Prefill)
    {
        run_parity_test(
            /*seq_len=*/8,
            /*kv_seq_len=*/8,
            /*num_heads=*/14,
            /*num_kv_heads=*/2,
            /*head_dim=*/64,
            /*q_k_scale=*/5.0f,
            /*v_scale=*/2.0f,
            /*residual_scale=*/3.0f,
            "RealisticData_Prefill");
    }

    // ============================================================================
    // Tests for Different Model Configs
    // ============================================================================

    TEST_F(Test__JitFusedAttentionWo_Q16_1Fused_vs_FP32, Qwen2_0_5B_Config)
    {
        // Qwen2.5-0.5B: 14 heads, 2 KV heads, head_dim=64
        run_parity_test(
            /*seq_len=*/1,
            /*kv_seq_len=*/32,
            /*num_heads=*/14,
            /*num_kv_heads=*/2,
            /*head_dim=*/64,
            /*q_k_scale=*/3.0f,
            /*v_scale=*/1.5f,
            /*residual_scale=*/2.0f,
            "Qwen2_0_5B_Config");
    }

    TEST_F(Test__JitFusedAttentionWo_Q16_1Fused_vs_FP32, Qwen2_7B_Config)
    {
        // Qwen2.5-7B: 28 heads, 4 KV heads, head_dim=128
        run_parity_test(
            /*seq_len=*/1,
            /*kv_seq_len=*/64,
            /*num_heads=*/28,
            /*num_kv_heads=*/4,
            /*head_dim=*/128,
            /*q_k_scale=*/3.0f,
            /*v_scale=*/1.5f,
            /*residual_scale=*/2.0f,
            "Qwen2_7B_Config");
    }

    // ============================================================================
    // Stress Test: Various Sequence Lengths
    // ============================================================================

    TEST_F(Test__JitFusedAttentionWo_Q16_1Fused_vs_FP32, MultipleSeqLens)
    {
        const std::vector<std::pair<int, int>> configs = {
            {1, 16},  // Decode start
            {1, 128}, // Decode mid-sequence
            {4, 4},   // Short prefill
            {16, 16}, // Medium prefill
        };

        for (const auto &[seq_len, kv_seq_len] : configs)
        {
            std::string name = "SeqLen_" + std::to_string(seq_len) + "_KV_" + std::to_string(kv_seq_len);
            run_parity_test(
                seq_len, kv_seq_len,
                /*num_heads=*/14,
                /*num_kv_heads=*/2,
                /*head_dim=*/64,
                /*q_k_scale=*/2.0f,
                /*v_scale=*/1.0f,
                /*residual_scale=*/1.5f,
                name);
        }
    }

} // namespace
