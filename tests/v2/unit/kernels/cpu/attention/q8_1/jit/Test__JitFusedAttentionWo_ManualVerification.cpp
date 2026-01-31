/**
 * @file Test__JitFusedAttentionWo_ManualVerification.cpp
 * @brief Manual verification tests for JIT fused attention kernel
 * @author David Sanftenberg
 * @date December 2025
 *
 * These tests verify the JIT kernel output against MANUALLY COMPUTED ground truth,
 * not against another C++ reference implementation. This catches bugs where both
 * the JIT kernel and reference implementation have the same error.
 *
 * Test approach:
 * 1. Create simple, known input patterns
 * 2. Manually compute expected attention scores and context vectors
 * 3. Run JIT kernel and compare against manual computation
 *
 * Key formulas verified:
 * - Attention score: (Q · K) / sqrt(head_dim)
 * - Softmax: exp(score - max) / sum(exp)
 * - Context: softmax_weights @ V
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <numeric>

#include "kernels/cpu/attention/q8_1/jit/JitFusedAttentionWo.h"
#include "tensors/BlockStructures.h"
#include "tensors/FP16Utils.h"

namespace llaminar::v2::kernels::jit::test
{

    using llaminar2::fp16_to_fp32;
    using llaminar2::fp32_to_fp16;
    using llaminar2::Q8_1Block;

    class Test__JitFusedAttentionWo_ManualVerification : public ::testing::Test
    {
    protected:
        /**
         * @brief Quantize a single 32-element block to Q8_1 format
         */
        Q8_1Block quantize_block(const float *data)
        {
            Q8_1Block blk;

            // Find max absolute value
            float max_abs = 0.0f;
            for (int i = 0; i < 32; ++i)
            {
                max_abs = std::max(max_abs, std::fabs(data[i]));
            }

            // Compute scale
            float scale = max_abs / 127.0f;
            if (scale < 1e-10f)
                scale = 1e-10f;
            float inv_scale = (max_abs > 1e-10f) ? (127.0f / max_abs) : 0.0f;

            // Quantize
            int32_t sum_qs = 0;
            for (int i = 0; i < 32; ++i)
            {
                int8_t q = static_cast<int8_t>(std::round(data[i] * inv_scale));
                q = std::max(int8_t(-127), std::min(int8_t(127), q));
                blk.qs[i] = q;
                sum_qs += q;
            }

            blk.d = fp32_to_fp16(scale);
            blk.sum_qs = static_cast<int16_t>(sum_qs);

            return blk;
        }

        /**
         * @brief Dequantize a Q8_1 block back to FP32
         */
        void dequantize_block(const Q8_1Block &blk, float *output)
        {
            float scale = fp16_to_fp32(blk.d);
            for (int i = 0; i < 32; ++i)
            {
                output[i] = static_cast<float>(blk.qs[i]) * scale;
            }
        }

        /**
         * @brief Compute FP32 dot product
         */
        float fp32_dot(const float *a, const float *b, int n)
        {
            double sum = 0.0;
            for (int i = 0; i < n; ++i)
            {
                sum += static_cast<double>(a[i]) * static_cast<double>(b[i]);
            }
            return static_cast<float>(sum);
        }

        /**
         * @brief Compute Q8_1 dot product (dequantize + FP32 dot)
         * This is ground truth - dequantize both operands and compute FP32 dot
         */
        float q8_1_dot_ground_truth(const Q8_1Block *q_blocks, const Q8_1Block *k_blocks, int num_blocks)
        {
            std::vector<float> q_fp32(num_blocks * 32);
            std::vector<float> k_fp32(num_blocks * 32);

            for (int b = 0; b < num_blocks; ++b)
            {
                dequantize_block(q_blocks[b], q_fp32.data() + b * 32);
                dequantize_block(k_blocks[b], k_fp32.data() + b * 32);
            }

            return fp32_dot(q_fp32.data(), k_fp32.data(), num_blocks * 32);
        }

        /**
         * @brief Compute softmax over a vector
         */
        std::vector<float> softmax(const std::vector<float> &scores)
        {
            float max_score = *std::max_element(scores.begin(), scores.end());

            std::vector<float> exp_scores(scores.size());
            float sum = 0.0f;
            for (size_t i = 0; i < scores.size(); ++i)
            {
                exp_scores[i] = std::exp(scores[i] - max_score);
                sum += exp_scores[i];
            }

            for (size_t i = 0; i < scores.size(); ++i)
            {
                exp_scores[i] /= sum;
            }

            return exp_scores;
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
         * @brief Print Q8_1 block details for debugging
         */
        void print_q8_1_block(const char *name, const Q8_1Block &blk)
        {
            float scale = fp16_to_fp32(blk.d);
            std::cout << "  " << name << ": d=" << scale << " sum_qs=" << blk.sum_qs;
            std::cout << " qs[0:4]=" << (int)blk.qs[0] << "," << (int)blk.qs[1]
                      << "," << (int)blk.qs[2] << "," << (int)blk.qs[3] << std::endl;
        }
    };

    // ============================================================================
    // Test: Single Head, Single KV Position, head_dim=64
    // ============================================================================

    /**
     * @brief Minimal test: 1 head, 1 query, 1 KV position
     *
     * With single KV position, softmax = 1.0 (100% weight to only position)
     * So context = V directly (weighted by 1.0)
     */
    TEST_F(Test__JitFusedAttentionWo_ManualVerification, SingleHeadSingleKV)
    {
        std::cout << "\n=== Single Head, Single KV Position ===" << std::endl;

        const int num_heads = 1;
        const int num_kv_heads = 1;
        const int head_dim = 64;
        const int seq_len = 1;
        const int kv_seq_len = 1;
        const int d_model = num_heads * head_dim;
        const int blocks_per_head = head_dim / 32;
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        // Create simple data: all 0.5 for Q, K, V
        std::vector<float> q_fp32(head_dim, 0.5f);
        std::vector<float> k_fp32(head_dim, 0.5f);
        std::vector<float> v_fp32(head_dim, 0.5f);

        // Quantize
        std::vector<Q8_1Block> Q(blocks_per_head);
        std::vector<Q8_1Block> K(blocks_per_head);
        std::vector<Q8_1Block> V(blocks_per_head);

        for (int b = 0; b < blocks_per_head; ++b)
        {
            Q[b] = quantize_block(q_fp32.data() + b * 32);
            K[b] = quantize_block(k_fp32.data() + b * 32);
            V[b] = quantize_block(v_fp32.data() + b * 32);
        }

        // ===== Manual computation =====

        // 1. Compute attention score: (Q · K) / sqrt(head_dim)
        float dot = q8_1_dot_ground_truth(Q.data(), K.data(), blocks_per_head);
        float score = dot * scale;

        std::cout << "  Manual Q·K = " << dot << std::endl;
        std::cout << "  Manual score = " << score << " (after scale " << scale << ")" << std::endl;

        // 2. Softmax with single position = 1.0
        float softmax_weight = 1.0f; // Only one KV position

        // 3. Context = softmax_weight * V = 1.0 * V = V
        std::vector<float> expected_context(head_dim);
        for (int b = 0; b < blocks_per_head; ++b)
        {
            dequantize_block(V[b], expected_context.data() + b * 32);
        }

        std::cout << "  Expected context[0:4] = " << expected_context[0] << ", " << expected_context[1]
                  << ", " << expected_context[2] << ", " << expected_context[3] << std::endl;

        // ===== Run JIT kernel =====

        // Create dummy Wo (identity)
        std::vector<float> Wo(d_model * d_model, 0.0f);
        for (int i = 0; i < d_model; ++i)
        {
            Wo[i * d_model + i] = 1.0f; // Identity matrix
        }

        // Output buffer
        std::vector<float> output_jit(d_model, 0.0f);

        // Configure JIT kernel
        JitAttentionConfig jit_config;
        jit_config.head_dim = head_dim;
        jit_config.num_heads = num_heads;
        jit_config.num_kv_heads = num_kv_heads;
        jit_config.batch_size = 1;
        jit_config.wo_format = WoFormat::FP32;
        jit_config.causal = false;

        JitFusedAttentionWo jit_kernel(jit_config);
        jit_kernel.compute(
            Q.data(),
            K.data(),
            V.data(),
            Wo.data(),
            output_jit.data(),
            seq_len,
            kv_seq_len,
            scale,
            0); // position_offset = 0

        std::cout << "  JIT output[0:4] = " << output_jit[0] << ", " << output_jit[1]
                  << ", " << output_jit[2] << ", " << output_jit[3] << std::endl;

        // ===== Compare =====
        double cos = cosine_similarity(output_jit.data(), expected_context.data(), head_dim);
        std::cout << "  Cosine similarity: " << cos << std::endl;

        // With single KV position, output should be exactly V (after Wo projection = identity)
        EXPECT_GT(cos, 0.99) << "JIT output should match expected context (V)";

        // Also check per-element
        for (int i = 0; i < 4; ++i)
        {
            EXPECT_NEAR(output_jit[i], expected_context[i], 0.1f)
                << "Element " << i << " mismatch";
        }
    }

    // ============================================================================
    // Test: Single Head, Two KV Positions (verify softmax)
    // ============================================================================

    /**
     * @brief Test with 2 KV positions to verify softmax computation
     *
     * If Q matches K[0] better than K[1], softmax should weight V[0] more heavily.
     */
    TEST_F(Test__JitFusedAttentionWo_ManualVerification, SingleHeadTwoKV)
    {
        std::cout << "\n=== Single Head, Two KV Positions ===" << std::endl;

        const int num_heads = 1;
        const int num_kv_heads = 1;
        const int head_dim = 64;
        const int seq_len = 1;
        const int kv_seq_len = 2;
        const int d_model = num_heads * head_dim;
        const int blocks_per_head = head_dim / 32;
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        // Create Q that matches K[0] more than K[1]
        std::vector<float> q_fp32(head_dim, 0.5f);
        std::vector<float> k0_fp32(head_dim, 0.5f);  // Matches Q
        std::vector<float> k1_fp32(head_dim, -0.5f); // Opposite of Q
        std::vector<float> v0_fp32(head_dim, 1.0f);  // Different values
        std::vector<float> v1_fp32(head_dim, -1.0f);

        // Quantize
        std::vector<Q8_1Block> Q(blocks_per_head);
        std::vector<Q8_1Block> K(2 * blocks_per_head); // 2 KV positions
        std::vector<Q8_1Block> V(2 * blocks_per_head);

        for (int b = 0; b < blocks_per_head; ++b)
        {
            Q[b] = quantize_block(q_fp32.data() + b * 32);
            K[b] = quantize_block(k0_fp32.data() + b * 32);
            K[blocks_per_head + b] = quantize_block(k1_fp32.data() + b * 32);
            V[b] = quantize_block(v0_fp32.data() + b * 32);
            V[blocks_per_head + b] = quantize_block(v1_fp32.data() + b * 32);
        }

        // ===== Manual computation =====

        // 1. Compute attention scores
        float dot0 = q8_1_dot_ground_truth(Q.data(), K.data(), blocks_per_head);
        float dot1 = q8_1_dot_ground_truth(Q.data(), K.data() + blocks_per_head, blocks_per_head);
        float score0 = dot0 * scale;
        float score1 = dot1 * scale;

        std::cout << "  Manual Q·K[0] = " << dot0 << " → score0 = " << score0 << std::endl;
        std::cout << "  Manual Q·K[1] = " << dot1 << " → score1 = " << score1 << std::endl;

        // 2. Softmax
        std::vector<float> scores = {score0, score1};
        std::vector<float> weights = softmax(scores);

        std::cout << "  Softmax weights: [" << weights[0] << ", " << weights[1] << "]" << std::endl;

        // Q matches K[0], so weight[0] should be > weight[1]
        EXPECT_GT(weights[0], weights[1]) << "Weight for matching K should be higher";

        // 3. Context = w0 * V[0] + w1 * V[1]
        std::vector<float> v0_deq(head_dim), v1_deq(head_dim);
        for (int b = 0; b < blocks_per_head; ++b)
        {
            dequantize_block(V[b], v0_deq.data() + b * 32);
            dequantize_block(V[blocks_per_head + b], v1_deq.data() + b * 32);
        }

        std::vector<float> expected_context(head_dim);
        for (int i = 0; i < head_dim; ++i)
        {
            expected_context[i] = weights[0] * v0_deq[i] + weights[1] * v1_deq[i];
        }

        std::cout << "  Expected context[0:4] = " << expected_context[0] << ", " << expected_context[1]
                  << ", " << expected_context[2] << ", " << expected_context[3] << std::endl;

        // ===== Run JIT kernel =====

        std::vector<float> Wo(d_model * d_model, 0.0f);
        for (int i = 0; i < d_model; ++i)
        {
            Wo[i * d_model + i] = 1.0f;
        }

        std::vector<float> output_jit(d_model, 0.0f);

        JitAttentionConfig jit_config;
        jit_config.head_dim = head_dim;
        jit_config.num_heads = num_heads;
        jit_config.num_kv_heads = num_kv_heads;
        jit_config.batch_size = 1;
        jit_config.wo_format = WoFormat::FP32;
        jit_config.causal = false;

        JitFusedAttentionWo jit_kernel(jit_config);
        jit_kernel.compute(
            Q.data(),
            K.data(),
            V.data(),
            Wo.data(),
            output_jit.data(),
            seq_len,
            kv_seq_len,
            scale,
            0);

        std::cout << "  JIT output[0:4] = " << output_jit[0] << ", " << output_jit[1]
                  << ", " << output_jit[2] << ", " << output_jit[3] << std::endl;

        // ===== Compare =====
        double cos = cosine_similarity(output_jit.data(), expected_context.data(), head_dim);
        std::cout << "  Cosine similarity: " << cos << std::endl;

        EXPECT_GT(cos, 0.95) << "JIT output should match expected context";
    }

    // ============================================================================
    // Test: Verify Q8_1 Dot Product Directly
    // ============================================================================

    /**
     * @brief Direct test of Q8_1 dot product computation in JIT
     *
     * We can't easily extract just the dot product from JIT, but we can verify
     * indirectly: with single KV position and V = constant, the attention score
     * doesn't matter (softmax = 1.0), so we test the score computation by looking
     * at the attention weights via multiple KV positions.
     */
    TEST_F(Test__JitFusedAttentionWo_ManualVerification, Q8_1DotProductAccuracy)
    {
        std::cout << "\n=== Q8_1 Dot Product Accuracy via Attention Weights ===" << std::endl;

        const int num_heads = 1;
        const int num_kv_heads = 1;
        const int head_dim = 64;
        const int seq_len = 1;
        const int kv_seq_len = 4; // More KV positions for better softmax differentiation
        const int d_model = num_heads * head_dim;
        const int blocks_per_head = head_dim / 32;
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        // Create Q and K patterns with known dot products
        // Q = [1, 1, 1, ..., 1] normalized
        // K[0] = [1, 0, 0, ..., 0] → dot = 1/sqrt(64)
        // K[1] = [0, 1, 0, ..., 0] → dot = 1/sqrt(64)
        // K[2] = [1, 1, 0, ..., 0] → dot = 2/sqrt(64)
        // K[3] = [0, 0, 1, 1, 0, ..., 0] → dot = 2/sqrt(64)

        std::vector<float> q_fp32(head_dim);
        float q_val = 1.0f / std::sqrt(static_cast<float>(head_dim));
        for (int i = 0; i < head_dim; ++i)
        {
            q_fp32[i] = q_val;
        }

        std::vector<std::vector<float>> k_fp32(kv_seq_len, std::vector<float>(head_dim, 0.0f));
        k_fp32[0][0] = 1.0f;
        k_fp32[1][1] = 1.0f;
        k_fp32[2][0] = 1.0f;
        k_fp32[2][1] = 1.0f;
        k_fp32[3][2] = 1.0f;
        k_fp32[3][3] = 1.0f;

        // V = distinct unit vectors for each position
        std::vector<std::vector<float>> v_fp32(kv_seq_len, std::vector<float>(head_dim, 0.0f));
        for (int kv = 0; kv < kv_seq_len; ++kv)
        {
            v_fp32[kv][kv * (head_dim / kv_seq_len)] = 1.0f; // Spread out
        }

        // Quantize
        std::vector<Q8_1Block> Q(blocks_per_head);
        std::vector<Q8_1Block> K(kv_seq_len * blocks_per_head);
        std::vector<Q8_1Block> V(kv_seq_len * blocks_per_head);

        for (int b = 0; b < blocks_per_head; ++b)
        {
            Q[b] = quantize_block(q_fp32.data() + b * 32);
        }

        for (int kv = 0; kv < kv_seq_len; ++kv)
        {
            for (int b = 0; b < blocks_per_head; ++b)
            {
                K[kv * blocks_per_head + b] = quantize_block(k_fp32[kv].data() + b * 32);
                V[kv * blocks_per_head + b] = quantize_block(v_fp32[kv].data() + b * 32);
            }
        }

        // ===== Manual computation =====

        std::vector<float> scores(kv_seq_len);
        for (int kv = 0; kv < kv_seq_len; ++kv)
        {
            float dot = q8_1_dot_ground_truth(Q.data(), K.data() + kv * blocks_per_head, blocks_per_head);
            scores[kv] = dot * scale;
            std::cout << "  Manual Q·K[" << kv << "] = " << dot << " → score = " << scores[kv] << std::endl;
        }

        std::vector<float> weights = softmax(scores);
        std::cout << "  Softmax weights: [";
        for (int kv = 0; kv < kv_seq_len; ++kv)
        {
            std::cout << weights[kv];
            if (kv < kv_seq_len - 1)
                std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        // K[2] and K[3] have higher dot products (2 elements vs 1)
        // So weights[2] and weights[3] should be higher
        EXPECT_GT(weights[2], weights[0]) << "Higher dot product should give higher weight";
        EXPECT_GT(weights[3], weights[1]) << "Higher dot product should give higher weight";

        // Compute expected context
        std::vector<float> expected_context(head_dim, 0.0f);
        for (int kv = 0; kv < kv_seq_len; ++kv)
        {
            std::vector<float> v_deq(head_dim);
            for (int b = 0; b < blocks_per_head; ++b)
            {
                dequantize_block(V[kv * blocks_per_head + b], v_deq.data() + b * 32);
            }
            for (int i = 0; i < head_dim; ++i)
            {
                expected_context[i] += weights[kv] * v_deq[i];
            }
        }

        // ===== Run JIT kernel =====

        std::vector<float> Wo(d_model * d_model, 0.0f);
        for (int i = 0; i < d_model; ++i)
        {
            Wo[i * d_model + i] = 1.0f;
        }

        std::vector<float> output_jit(d_model, 0.0f);

        JitAttentionConfig jit_config;
        jit_config.head_dim = head_dim;
        jit_config.num_heads = num_heads;
        jit_config.num_kv_heads = num_kv_heads;
        jit_config.batch_size = 1;
        jit_config.wo_format = WoFormat::FP32;
        jit_config.causal = false;

        JitFusedAttentionWo jit_kernel(jit_config);
        jit_kernel.compute(
            Q.data(),
            K.data(),
            V.data(),
            Wo.data(),
            output_jit.data(),
            seq_len,
            kv_seq_len,
            scale,
            0);

        // ===== Compare =====
        double cos = cosine_similarity(output_jit.data(), expected_context.data(), head_dim);
        std::cout << "  Cosine similarity: " << cos << std::endl;

        // Print difference if there's a mismatch
        if (cos < 0.95)
        {
            std::cout << "  === MISMATCH DETAILS ===" << std::endl;
            for (int i = 0; i < 8; ++i)
            {
                std::cout << "    [" << i << "] jit=" << output_jit[i]
                          << " expected=" << expected_context[i]
                          << " diff=" << (output_jit[i] - expected_context[i]) << std::endl;
            }
        }

        EXPECT_GT(cos, 0.95) << "JIT output should match manual computation";
    }

    // ============================================================================
    // Test: GQA Configuration (multiple Q heads, fewer KV heads)
    // ============================================================================

    TEST_F(Test__JitFusedAttentionWo_ManualVerification, GQABasic)
    {
        std::cout << "\n=== GQA: 2 Q heads, 1 KV head ===" << std::endl;

        const int num_heads = 2;
        const int num_kv_heads = 1;
        const int head_dim = 64;
        const int seq_len = 1;
        const int kv_seq_len = 1;
        const int d_model = num_heads * head_dim;
        const int blocks_per_head = head_dim / 32;
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        // Create different Q for each head, same K/V
        std::vector<float> q0_fp32(head_dim, 0.5f);
        std::vector<float> q1_fp32(head_dim, -0.5f); // Different from q0
        std::vector<float> k_fp32(head_dim, 0.5f);
        std::vector<float> v_fp32(head_dim, 1.0f);

        // Quantize
        std::vector<Q8_1Block> Q(num_heads * blocks_per_head);
        std::vector<Q8_1Block> K(blocks_per_head); // Only 1 KV head
        std::vector<Q8_1Block> V(blocks_per_head);

        for (int b = 0; b < blocks_per_head; ++b)
        {
            Q[b] = quantize_block(q0_fp32.data() + b * 32);
            Q[blocks_per_head + b] = quantize_block(q1_fp32.data() + b * 32);
            K[b] = quantize_block(k_fp32.data() + b * 32);
            V[b] = quantize_block(v_fp32.data() + b * 32);
        }

        // ===== Manual computation =====

        // With single KV, softmax = 1.0 for both heads
        // Context for both heads = V (same since they share KV)
        std::vector<float> expected_context(d_model);
        for (int b = 0; b < blocks_per_head; ++b)
        {
            dequantize_block(V[b], expected_context.data() + b * 32);
            dequantize_block(V[b], expected_context.data() + head_dim + b * 32);
        }

        std::cout << "  Expected context[0:4] (head 0) = " << expected_context[0] << ", "
                  << expected_context[1] << ", " << expected_context[2] << ", " << expected_context[3] << std::endl;
        std::cout << "  Expected context[64:68] (head 1) = " << expected_context[64] << ", "
                  << expected_context[65] << ", " << expected_context[66] << ", " << expected_context[67] << std::endl;

        // ===== Run JIT kernel =====

        std::vector<float> Wo(d_model * d_model, 0.0f);
        for (int i = 0; i < d_model; ++i)
        {
            Wo[i * d_model + i] = 1.0f;
        }

        std::vector<float> output_jit(d_model, 0.0f);

        JitAttentionConfig jit_config;
        jit_config.head_dim = head_dim;
        jit_config.num_heads = num_heads;
        jit_config.num_kv_heads = num_kv_heads;
        jit_config.batch_size = 1;
        jit_config.wo_format = WoFormat::FP32;
        jit_config.causal = false;

        JitFusedAttentionWo jit_kernel(jit_config);
        jit_kernel.compute(
            Q.data(),
            K.data(),
            V.data(),
            Wo.data(),
            output_jit.data(),
            seq_len,
            kv_seq_len,
            scale,
            0);

        std::cout << "  JIT output[0:4] (head 0) = " << output_jit[0] << ", "
                  << output_jit[1] << ", " << output_jit[2] << ", " << output_jit[3] << std::endl;
        std::cout << "  JIT output[64:68] (head 1) = " << output_jit[64] << ", "
                  << output_jit[65] << ", " << output_jit[66] << ", " << output_jit[67] << std::endl;

        // ===== Compare =====

        // Both heads should have same output (both use same V)
        double cos_head0 = cosine_similarity(output_jit.data(), expected_context.data(), head_dim);
        double cos_head1 = cosine_similarity(output_jit.data() + head_dim, expected_context.data() + head_dim, head_dim);

        std::cout << "  Cosine similarity (head 0): " << cos_head0 << std::endl;
        std::cout << "  Cosine similarity (head 1): " << cos_head1 << std::endl;

        EXPECT_GT(cos_head0, 0.95) << "Head 0 should match expected";
        EXPECT_GT(cos_head1, 0.95) << "Head 1 should match expected";
    }

} // namespace llaminar::v2::kernels::jit::test
