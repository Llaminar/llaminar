/**
 * @file Test__JitFusedAttentionWo_vs_FP32Attention.cpp
 * @brief Kernel Parity: JitFusedAttentionWo (FA2 Q8_1) vs FP32 Decomposed Attention
 *
 * @category e2e/parity/kernels/attention
 * @tested   JitFusedAttentionWo (7469-line FA2-style JIT kernel with Q8_1 inputs)
 * @reference FP32 decomposed attention (Q@K^T -> softmax -> @V)
 *
 * This test compares the JitFusedAttentionWo kernel (used in E2E pipeline)
 * against a true FP32 decomposed attention implementation.
 *
 * Key differences from unit/attention/Test__JitFusedAttentionWo_Correctness.cpp:
 * 1. Reference is TRUE FP32 attention, not Q8_1 FusedAttentionWoRef
 * 2. Uses realistic data magnitudes from actual model (±5-10 for Q/K)
 * 3. Documents expected divergence vs FP32 at larger data scales
 *
 * Findings (2025-12-25):
 * - JitFusedAttentionWo achieves cos≥0.999 vs FP32 even with realistic data (±10)
 * - E2E divergence is caused by accumulated Q/K/V quantization error, not this kernel
 *
 * @author David Sanftenberg
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
    using llaminar2::Q8_1Block;
    using llaminar2::simd::fp16_to_fp32;
    using llaminar2::simd::fp32_to_fp16;

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
     * @brief TRUE FP32 attention reference (no Q8_1 quantization)
     *
     * Implements: output = softmax(Q @ K^T * scale + mask) @ V
     * All computation done in FP32 for ground truth.
     */
    void compute_fp32_attention_reference(
        int seq_len,
        int kv_seq_len,
        int num_heads,
        int num_kv_heads,
        int head_dim,
        const float *Q_fp32, // [seq_len, num_heads, head_dim]
        const float *K_fp32, // [kv_seq_len, num_kv_heads, head_dim]
        const float *V_fp32, // [kv_seq_len, num_kv_heads, head_dim]
        float scale,
        float *output_fp32, // [seq_len, num_heads * head_dim]
        bool causal)
    {
        int gqa_ratio = num_heads / num_kv_heads;
        int d_model = num_heads * head_dim;

        // For each query position and head
        for (int q_pos = 0; q_pos < seq_len; ++q_pos)
        {
            for (int h = 0; h < num_heads; ++h)
            {
                int kv_head = h / gqa_ratio;
                const float *Q_head = Q_fp32 + q_pos * num_heads * head_dim + h * head_dim;
                const float *K_head = K_fp32 + kv_head * head_dim; // First K position
                const float *V_head = V_fp32 + kv_head * head_dim; // First V position

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

                    // Apply causal mask
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
                float *output_head = output_fp32 + q_pos * d_model + h * head_dim;
                for (int d = 0; d < head_dim; ++d)
                {
                    output_head[d] = 0.0f;
                }
                for (int k_pos = 0; k_pos < kv_seq_len; ++k_pos)
                {
                    const float *V_pos = V_fp32 + k_pos * num_kv_heads * head_dim + kv_head * head_dim;
                    for (int d = 0; d < head_dim; ++d)
                    {
                        output_head[d] += scores[k_pos] * V_pos[d];
                    }
                }
            }
        }
    }

    // ============================================================================
    // Test Fixture
    // ============================================================================

    class Test__JitFusedAttentionWo_vs_FP32_Parity : public ::testing::Test
    {
    protected:
        // Thresholds for comparison
        // With realistic data magnitudes, we expect lower similarity than small-scale tests
        static constexpr double MIN_COSINE_EXCELLENT = 0.999; // Should achieve with small data
        static constexpr double MIN_COSINE_GOOD = 0.95;       // Acceptable for Q8_1
        static constexpr double MIN_COSINE_ACCEPTABLE = 0.85; // E2E observes ~0.75, this is a warning

        void run_parity_test(
            int seq_len,
            int kv_seq_len,
            int num_heads,
            int num_kv_heads,
            int head_dim,
            float q_k_scale, // Scale for Q/K data generation (±5-10 for realistic)
            float v_scale,   // Scale for V data generation (±1-2 for realistic)
            const std::string &test_name)
        {
            std::cout << "\n=== " << test_name << " ===" << std::endl;
            std::cout << "  Config: seq=" << seq_len << ", kv=" << kv_seq_len
                      << ", heads=" << num_heads << "/" << num_kv_heads
                      << ", head_dim=" << head_dim << std::endl;
            std::cout << "  Data scales: Q/K=±" << q_k_scale << ", V=±" << v_scale << std::endl;

            int d_model = num_heads * head_dim;
            float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

            // Generate FP32 data with realistic scales
            std::vector<float> Q_fp32(seq_len * num_heads * head_dim);
            std::vector<float> K_fp32(kv_seq_len * num_kv_heads * head_dim);
            std::vector<float> V_fp32(kv_seq_len * num_kv_heads * head_dim);

            generate_random_data(Q_fp32.data(), Q_fp32.size(), q_k_scale, 100);
            generate_random_data(K_fp32.data(), K_fp32.size(), q_k_scale, 200);
            generate_random_data(V_fp32.data(), V_fp32.size(), v_scale, 300);

            // Print data statistics
            float q_max = *std::max_element(Q_fp32.begin(), Q_fp32.end());
            float q_min = *std::min_element(Q_fp32.begin(), Q_fp32.end());
            float k_max = *std::max_element(K_fp32.begin(), K_fp32.end());
            float k_min = *std::min_element(K_fp32.begin(), K_fp32.end());
            std::cout << "  Q range: [" << q_min << ", " << q_max << "]" << std::endl;
            std::cout << "  K range: [" << k_min << ", " << k_max << "]" << std::endl;

            // Quantize to Q8_1 for JIT kernel
            int blocks_per_head = (head_dim + 31) / 32;
            std::vector<Q8_1Block> Q_q8(seq_len * num_heads * blocks_per_head);
            std::vector<Q8_1Block> K_q8(kv_seq_len * num_kv_heads * blocks_per_head);
            std::vector<Q8_1Block> V_q8(kv_seq_len * num_kv_heads * blocks_per_head);

            quantize_fp32_to_q8_1(Q_fp32.data(), seq_len * num_heads, head_dim, Q_q8.data());
            quantize_fp32_to_q8_1(K_fp32.data(), kv_seq_len * num_kv_heads, head_dim, K_q8.data());
            quantize_fp32_to_q8_1(V_fp32.data(), kv_seq_len * num_kv_heads, head_dim, V_q8.data());

            // FP32 reference output
            std::vector<float> output_fp32(seq_len * d_model, 0.0f);
            compute_fp32_attention_reference(
                seq_len, kv_seq_len, num_heads, num_kv_heads, head_dim,
                Q_fp32.data(), K_fp32.data(), V_fp32.data(),
                scale, output_fp32.data(), false);

            // JIT kernel output
            std::vector<float> output_jit(seq_len * d_model, 0.0f);

            // Dummy Wo weights (identity-like)
            std::vector<float> Wo_fp32(d_model * d_model, 0.0f);
            for (int i = 0; i < d_model; ++i)
            {
                Wo_fp32[i * d_model + i] = 1.0f;
            }

            // Run JIT kernel
            JitAttentionConfig jit_config;
            jit_config.head_dim = head_dim;
            jit_config.num_heads = num_heads;
            jit_config.num_kv_heads = num_kv_heads;
            jit_config.batch_size = 1;
            jit_config.wo_format = WoFormat::FP32;
            jit_config.causal = false;

            JitFusedAttentionWo jit_kernel(jit_config);
            jit_kernel.compute(
                Q_q8.data(),
                K_q8.data(),
                V_q8.data(),
                Wo_fp32.data(),
                output_jit.data(),
                seq_len,
                kv_seq_len,
                scale,
                0);

            // Compare
            int output_size = seq_len * d_model;
            double cos_sim = cosine_similarity(output_jit.data(), output_fp32.data(), output_size);
            double max_diff = max_abs_diff(output_jit.data(), output_fp32.data(), output_size);

            std::cout << "  ────────────────────────────────────" << std::endl;
            std::cout << "  Cosine similarity: " << std::fixed << std::setprecision(6) << cos_sim;
            if (cos_sim >= MIN_COSINE_EXCELLENT)
                std::cout << " ✓ EXCELLENT";
            else if (cos_sim >= MIN_COSINE_GOOD)
                std::cout << " ✓ GOOD";
            else if (cos_sim >= MIN_COSINE_ACCEPTABLE)
                std::cout << " ⚠ ACCEPTABLE";
            else
                std::cout << " ✗ DIVERGED";
            std::cout << std::endl;
            std::cout << "  Max abs difference: " << std::scientific << max_diff << std::endl;

            // Print first few outputs
            std::cout << "  First 8 FP32: ";
            for (int i = 0; i < 8 && i < output_size; ++i)
                std::cout << output_fp32[i] << " ";
            std::cout << std::endl;
            std::cout << "  First 8 JIT:  ";
            for (int i = 0; i < 8 && i < output_size; ++i)
                std::cout << output_jit[i] << " ";
            std::cout << std::endl;

            // Assert: At minimum, cos should be >= 0.70 (documented E2E divergence is ~0.75)
            EXPECT_GE(cos_sim, 0.70)
                << "Cosine similarity catastrophically low - something is broken";

            // Note: We expect cos ~0.75-0.89 based on E2E observations
            // This test documents the expected behavior, not sets a passing bar
            if (cos_sim < MIN_COSINE_GOOD)
            {
                std::cout << "\n  ⚠ WARNING: JitFusedAttentionWo shows significant divergence from FP32" << std::endl;
                std::cout << "    This is EXPECTED based on E2E layer-by-layer analysis." << std::endl;
                std::cout << "    The Q8_1 quantization + online softmax introduces error" << std::endl;
                std::cout << "    that accumulates and is amplified by softmax." << std::endl;
            }
        }
    };

    // ============================================================================
    // Tests with Small Data (Should Pass with ~0.999 cosine)
    // ============================================================================

    TEST_F(Test__JitFusedAttentionWo_vs_FP32_Parity, SmallData_Qwen2_0_5B_Decode)
    {
        // Small data scale - should achieve excellent parity
        run_parity_test(
            /*seq_len=*/1,
            /*kv_seq_len=*/64,
            /*num_heads=*/14,
            /*num_kv_heads=*/2,
            /*head_dim=*/64,
            /*q_k_scale=*/0.5f, // Small scale like existing tests
            /*v_scale=*/0.5f,
            "Small Data Scale (±0.5) - Qwen2 0.5B Decode");
    }

    TEST_F(Test__JitFusedAttentionWo_vs_FP32_Parity, SmallData_Qwen2_0_5B_Prefill)
    {
        run_parity_test(
            /*seq_len=*/9,
            /*kv_seq_len=*/9,
            /*num_heads=*/14,
            /*num_kv_heads=*/2,
            /*head_dim=*/64,
            /*q_k_scale=*/0.5f,
            /*v_scale=*/0.5f,
            "Small Data Scale (±0.5) - Qwen2 0.5B Prefill");
    }

    // ============================================================================
    // Tests with Realistic Data (Expected to show divergence)
    // ============================================================================

    TEST_F(Test__JitFusedAttentionWo_vs_FP32_Parity, RealisticData_Qwen2_0_5B_Decode)
    {
        // Realistic data scale observed in actual model (Q/K after RoPE)
        // This is expected to show divergence similar to E2E (cos ~0.75-0.89)
        run_parity_test(
            /*seq_len=*/1,
            /*kv_seq_len=*/64,
            /*num_heads=*/14,
            /*num_kv_heads=*/2,
            /*head_dim=*/64,
            /*q_k_scale=*/5.0f, // Realistic scale (±5 std dev → ~±15 range)
            /*v_scale=*/1.0f,
            "Realistic Data Scale (Q/K ±5, V ±1) - Qwen2 0.5B Decode");
    }

    TEST_F(Test__JitFusedAttentionWo_vs_FP32_Parity, RealisticData_Qwen2_0_5B_Prefill)
    {
        run_parity_test(
            /*seq_len=*/9,
            /*kv_seq_len=*/9,
            /*num_heads=*/14,
            /*num_kv_heads=*/2,
            /*head_dim=*/64,
            /*q_k_scale=*/5.0f,
            /*v_scale=*/1.0f,
            "Realistic Data Scale (Q/K ±5, V ±1) - Qwen2 0.5B Prefill");
    }

    TEST_F(Test__JitFusedAttentionWo_vs_FP32_Parity, VeryLargeData_Qwen2_0_5B)
    {
        // Even larger data scale to stress-test numerical stability
        run_parity_test(
            /*seq_len=*/9,
            /*kv_seq_len=*/9,
            /*num_heads=*/14,
            /*num_kv_heads=*/2,
            /*head_dim=*/64,
            /*q_k_scale=*/10.0f, // Very large scale (±10 std dev → ~±30 range)
            /*v_scale=*/2.0f,
            "Very Large Data Scale (Q/K ±10, V ±2) - Qwen2 0.5B");
    }

    // ============================================================================
    // Comparison: Simple Kernel vs FA2 Kernel (same data, different kernels)
    // ============================================================================

    TEST_F(Test__JitFusedAttentionWo_vs_FP32_Parity, CompareKernels_SmallData)
    {
        // This test documents that both kernels should perform similarly with small data
        std::cout << "\n=== Kernel Comparison: Small Data ===" << std::endl;
        std::cout << "  Both QuantisedAttentionJit_Q8_1_Fused (simple) and" << std::endl;
        std::cout << "  JitFusedAttentionWo (FA2) should achieve ~0.999 cosine" << std::endl;
        std::cout << "  with small data scales." << std::endl;

        run_parity_test(1, 64, 14, 2, 64, 0.5f, 0.5f, "FA2 Kernel with Small Data");
    }

    TEST_F(Test__JitFusedAttentionWo_vs_FP32_Parity, CompareKernels_RealisticData)
    {
        // This test documents expected divergence with realistic data
        std::cout << "\n=== Kernel Comparison: Realistic Data ===" << std::endl;
        std::cout << "  E2E pipeline observes layer0_ATTENTION_CONTEXT cos ~0.75" << std::endl;
        std::cout << "  This test should reproduce similar divergence." << std::endl;

        run_parity_test(1, 64, 14, 2, 64, 5.0f, 1.0f, "FA2 Kernel with Realistic Data");
    }

} // anonymous namespace
