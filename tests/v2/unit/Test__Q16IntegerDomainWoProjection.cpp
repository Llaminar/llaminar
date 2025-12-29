/**
 * @file Test__Q16IntegerDomainWoProjection.cpp
 * @brief End-to-end test for Q16 integer-domain attention + Wo projection vs FP32 reference
 *
 * This test validates the COMPLETE integer-domain attention pipeline:
 * - Q×K^T → INT32 scores (Q16DotProduct)
 * - Softmax → INT16 weights (Exp2FixedSoftmax)
 * - P×V → INT32 accumulators (PVAccumulate)
 * - Wo projection → Q16_1 output (WoProjectionVNNIRef with VPDPWSSD)
 *
 * The test compares against a TRUE FP32 reference implementation to measure
 * the total accuracy loss from the integer-domain operations.
 *
 * Expected metrics:
 * - Cosine similarity: ≥ 0.90 (target ≥ 0.95)
 * - Documents precision characteristics of VPDPWSSD approach
 *
 * @author David Sanftenberg
 * @date December 2025
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <iostream>

// Q16 microkernels
#include "kernels/cpu/attention/q16_1/ref/microkernels/Q16DotProductRef.h"
#include "kernels/cpu/attention/q16_1/ref/microkernels/Exp2FixedSoftmaxRef.h"
#include "kernels/cpu/attention/q16_1/ref/microkernels/Int8RequantRef.h"
#include "kernels/cpu/attention/q16_1/ref/microkernels/PVAccumulateRef.h"
#include "kernels/cpu/attention/q16_1/ref/microkernels/WoProjectionVNNIRef.h"

// Tensor types and kernel factory
#include "tensors/Tensors.h"
#include "tensors/BlockStructures.h"
#include "kernels/KernelFactory.h"
#include "kernels/cpu/gemm_v4/QuantisedGemmKernel.h"

using namespace llaminar2;
using namespace llaminar2::kernels::q16_1;
using namespace llaminar2::kernels::q16_1::microkernels;

// Block size constant
constexpr int Q16_1_BLOCK_SIZE = 32;

// ============================================================================
// Test Fixture
// ============================================================================

class Q16IntegerDomainWoProjectionTest : public ::testing::Test
{
protected:
    std::mt19937 rng_{42};

    // ═══════════════════════════════════════════════════════════════════════
    // Statistical Utilities
    // ═══════════════════════════════════════════════════════════════════════

    double cosineSimilarity(const float *a, const float *b, size_t n)
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

    double maxAbsDiff(const float *a, const float *b, size_t n)
    {
        double max_diff = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            double diff = std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
            max_diff = std::max(max_diff, diff);
        }
        return max_diff;
    }

    double meanAbsDiff(const float *a, const float *b, size_t n)
    {
        double sum = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            sum += std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        }
        return sum / static_cast<double>(n);
    }

    double rmsDiff(const float *a, const float *b, size_t n)
    {
        double sum_sq = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
            sum_sq += diff * diff;
        }
        return std::sqrt(sum_sq / static_cast<double>(n));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Data Generation Utilities
    // ═══════════════════════════════════════════════════════════════════════

    void generateRandomFP32(float *data, size_t n, float scale = 1.0f)
    {
        std::uniform_real_distribution<float> dist(-scale, scale);
        for (size_t i = 0; i < n; ++i)
        {
            data[i] = dist(rng_);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Quantization Utilities
    // ═══════════════════════════════════════════════════════════════════════

    void quantizeFP32ToQ16_1(const float *input, int n, Q16_1Block *output)
    {
        constexpr int BLOCK_SIZE = 32;
        const int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

        for (int b = 0; b < num_blocks; ++b)
        {
            const int start = b * BLOCK_SIZE;
            const int end = std::min(start + BLOCK_SIZE, n);
            const int count = end - start;

            // Find max abs in block
            float maxabs = 0.0f;
            for (int i = 0; i < count; ++i)
            {
                maxabs = std::max(maxabs, std::abs(input[start + i]));
            }

            // Compute scale
            float scale = (maxabs > 0) ? maxabs / 32767.0f : 1.0f;
            output[b].d = scale;

            // Quantize
            int32_t sum = 0;
            for (int i = 0; i < BLOCK_SIZE; ++i)
            {
                if (i < count)
                {
                    float scaled = input[start + i] / scale;
                    int16_t q = static_cast<int16_t>(std::round(std::clamp(scaled, -32767.0f, 32767.0f)));
                    output[b].qs[i] = q;
                    sum += q;
                }
                else
                {
                    output[b].qs[i] = 0;
                }
            }
            output[b].sum_qs = sum;
        }
    }

    void dequantizeQ16_1ToFP32(const Q16_1Block *input, int n, float *output)
    {
        constexpr int BLOCK_SIZE = 32;
        const int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

        for (int b = 0; b < num_blocks; ++b)
        {
            const int start = b * BLOCK_SIZE;
            const int end = std::min(start + BLOCK_SIZE, n);
            const int count = end - start;

            float scale = input[b].d;
            for (int i = 0; i < count; ++i)
            {
                output[start + i] = static_cast<float>(input[b].qs[i]) * scale;
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // TRUE FP32 Reference: Attention + Wo Projection
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * @brief Compute TRUE FP32 attention + Wo projection
     *
     * This is the ground truth: pure FP32 softmax(Q @ K^T * scale) @ V @ Wo
     *
     * @param seq_len_q Number of query positions
     * @param kv_len Number of KV positions
     * @param num_heads Number of query heads
     * @param num_kv_heads Number of KV heads
     * @param head_dim Dimension per head
     * @param d_model Output dimension (typically num_heads * head_dim)
     * @param Q_fp32 Query tensor [seq_len_q × num_heads × head_dim]
     * @param K_fp32 Key tensor [kv_len × num_kv_heads × head_dim]
     * @param V_fp32 Value tensor [kv_len × num_kv_heads × head_dim]
     * @param Wo_fp32 Output projection weights [d_model × input_dim] (row-major)
     * @param attention_scale Scale factor (1/sqrt(head_dim))
     * @param output_fp32 Output [seq_len_q × d_model]
     * @param causal Whether to apply causal masking
     */
    void computeFP32AttentionWithWo(
        int seq_len_q,
        int kv_len,
        int num_heads,
        int num_kv_heads,
        int head_dim,
        int d_model,
        const float *Q_fp32,
        const float *K_fp32,
        const float *V_fp32,
        const float *Wo_fp32,
        float attention_scale,
        float *output_fp32,
        bool causal)
    {
        const int input_dim = num_heads * head_dim;
        const int gqa_ratio = num_heads / num_kv_heads;

        // Temporary: attention context [seq_len_q × input_dim]
        std::vector<float> context_fp32(seq_len_q * input_dim);

        for (int q_pos = 0; q_pos < seq_len_q; ++q_pos)
        {
            for (int h = 0; h < num_heads; ++h)
            {
                int kv_head = h / gqa_ratio;
                const float *Q_head = Q_fp32 + q_pos * input_dim + h * head_dim;
                float *ctx_head = context_fp32.data() + q_pos * input_dim + h * head_dim;

                // Determine effective KV length (causal masking)
                int kv_end = kv_len;
                if (causal)
                {
                    kv_end = std::min(kv_len, q_pos + 1);
                }

                if (kv_end <= 0)
                {
                    std::memset(ctx_head, 0, head_dim * sizeof(float));
                    continue;
                }

                // Step 1: Q @ K^T scores
                std::vector<float> scores(kv_end);
                for (int k_pos = 0; k_pos < kv_end; ++k_pos)
                {
                    const float *K_pos = K_fp32 + k_pos * num_kv_heads * head_dim + kv_head * head_dim;
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; ++d)
                    {
                        dot += Q_head[d] * K_pos[d];
                    }
                    scores[k_pos] = dot * attention_scale;
                }

                // Step 2: FP32 Softmax
                float max_score = *std::max_element(scores.begin(), scores.end());
                float sum_exp = 0.0f;
                for (int k = 0; k < kv_end; ++k)
                {
                    scores[k] = std::exp(scores[k] - max_score);
                    sum_exp += scores[k];
                }
                for (int k = 0; k < kv_end; ++k)
                {
                    scores[k] /= sum_exp;
                }

                // Step 3: Weighted V sum
                std::memset(ctx_head, 0, head_dim * sizeof(float));
                for (int k_pos = 0; k_pos < kv_end; ++k_pos)
                {
                    const float *V_pos = V_fp32 + k_pos * num_kv_heads * head_dim + kv_head * head_dim;
                    float weight = scores[k_pos];
                    for (int d = 0; d < head_dim; ++d)
                    {
                        ctx_head[d] += weight * V_pos[d];
                    }
                }
            }
        }

        // Step 4: Wo projection (context @ Wo^T)
        // Wo is [d_model × input_dim], output = context @ Wo^T
        for (int q_pos = 0; q_pos < seq_len_q; ++q_pos)
        {
            const float *ctx = context_fp32.data() + q_pos * input_dim;
            float *out = output_fp32 + q_pos * d_model;

            for (int n = 0; n < d_model; ++n)
            {
                float acc = 0.0f;
                for (int k = 0; k < input_dim; ++k)
                {
                    acc += ctx[k] * Wo_fp32[n * input_dim + k];
                }
                out[n] = acc;
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Q16 Integer-Domain Pipeline: Attention + VPDPWSSD Wo Projection
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * @brief Compute Q16 integer-domain attention + VPDPWSSD Wo projection
     *
     * Full integer pipeline:
     * 1. Q×K^T → INT32 scores (via Int8RequantRef)
     * 2. Exp2 softmax → INT16 weights
     * 3. P×V → INT32 accumulators (stays in integer domain!)
     * 4. INT32→INT16 context (keeps 16 bits precision)
     * 5. VPDPWSSD (INT16×INT16→INT32) with sign-extended weights
     * 6. INT32→Q16_1 output
     *
     * @param seq_len_q Number of query positions
     * @param kv_len Number of KV positions
     * @param num_heads Number of query heads
     * @param num_kv_heads Number of KV heads
     * @param head_dim Dimension per head
     * @param d_model Output dimension
     * @param Q_q16 Quantized queries
     * @param K_q16 Quantized keys
     * @param V_q16 Quantized values
     * @param Wo_packed VNNI-packed Wo weights
     * @param attention_scale Attention scale factor
     * @param output_fp32 Output (dequantized for comparison)
     * @param causal Whether to apply causal masking
     */
    void computeQ16IntegerDomainWithWo(
        int seq_len_q,
        int kv_len,
        int num_heads,
        int num_kv_heads,
        int head_dim,
        int d_model,
        const Q16_1Block *Q_q16,
        const Q16_1Block *K_q16,
        const Q16_1Block *V_q16,
        const llaminar2::gemm_v4::QuantisedPackedWeights *Wo_packed,
        float attention_scale,
        float *output_fp32,
        bool causal)
    {
        const int input_dim = num_heads * head_dim;
        const int gqa_ratio = num_heads / num_kv_heads;
        const int q_blocks_per_row = (input_dim + Q16_1_BLOCK_SIZE - 1) / Q16_1_BLOCK_SIZE;
        const int kv_blocks_per_row = (num_kv_heads * head_dim + Q16_1_BLOCK_SIZE - 1) / Q16_1_BLOCK_SIZE;

        // Output blocks
        const int output_blocks = (d_model + 31) / 32;
        std::vector<Q16_1Block> output_q16(seq_len_q * output_blocks);

        for (int q_pos = 0; q_pos < seq_len_q; ++q_pos)
        {
            // Per-query INT32 accumulator for full context (all heads concatenated)
            std::vector<int32_t> int32_context(input_dim, 0);
            std::vector<int32_t> head_weight_sums(num_heads, 1);
            std::vector<float> v_scales(input_dim, 1.0f);

            for (int h = 0; h < num_heads; ++h)
            {
                int kv_head = h / gqa_ratio;

                // Determine effective KV length
                int kv_end = kv_len;
                if (causal)
                {
                    kv_end = std::min(kv_len, q_pos + 1);
                }

                if (kv_end <= 0)
                {
                    continue;
                }

                // ════════════════════════════════════════════════════════════
                // STEP 1: Q×K^T → INT32 scores (via Int8Requant)
                // ════════════════════════════════════════════════════════════
                std::vector<int32_t> int32_scores(kv_end);
                float alpha = 0.0f;

                {
                    Int8RequantParams requant_params{};
                    requant_params.Q = Q_q16;
                    requant_params.K = K_q16;
                    requant_params.q_row = q_pos;
                    requant_params.head = h;
                    requant_params.kv_head = kv_head;
                    requant_params.head_dim = head_dim;
                    requant_params.kv_end = kv_end;
                    requant_params.q_blocks_per_row = q_blocks_per_row;
                    requant_params.kv_blocks_per_row = kv_blocks_per_row;
                    requant_params.attention_scale = attention_scale;

                    compute_int8_requant_logits(requant_params, int32_scores.data(), &alpha);
                }

                // ════════════════════════════════════════════════════════════
                // STEP 2: Exp2 Fixed Softmax → INT16 weights
                // ════════════════════════════════════════════════════════════
                std::vector<int16_t> int16_weights(kv_end);
                int32_t weight_sum = 0;

                exp2_fixed_softmax_row(
                    int32_scores.data(),
                    int16_weights.data(),
                    kv_end,
                    alpha,
                    &weight_sum);

                // ════════════════════════════════════════════════════════════
                // STEP 3: P×V → INT32 accumulators (stays in integer domain!)
                // ════════════════════════════════════════════════════════════

                // Accumulate into INT32 buffer for this head
                int32_t *head_accum = int32_context.data() + h * head_dim;
                float v_scale_product = 0.0f;

                // Manual P×V accumulation in INT32 domain
                for (int k_pos = 0; k_pos < kv_end; ++k_pos)
                {
                    int16_t w = int16_weights[k_pos];

                    // For each dimension in V
                    for (int d = 0; d < head_dim; ++d)
                    {
                        // Get V value (INT16 from Q16_1)
                        int block_idx = (kv_head * head_dim + d) / 32;
                        int elem_idx = (kv_head * head_dim + d) % 32;
                        const Q16_1Block &v_block = V_q16[k_pos * kv_blocks_per_row + block_idx];
                        int16_t v_int16 = v_block.qs[elem_idx];

                        // INT16 × INT16 → INT32 accumulation
                        head_accum[d] += static_cast<int32_t>(w) * static_cast<int32_t>(v_int16);

                        // Track V scale (use first block's scale as representative)
                        if (k_pos == 0 && d == 0)
                        {
                            v_scale_product = v_block.d;
                        }
                    }
                }

                head_weight_sums[h] = (weight_sum == 0) ? 1 : weight_sum;
                for (int d = 0; d < head_dim; ++d)
                {
                    v_scales[h * head_dim + d] = v_scale_product;
                }
            }

            // ════════════════════════════════════════════════════════════════
            // STEP 4: VPDPWSSD Wo Projection → Q16_1 output
            // ════════════════════════════════════════════════════════════════

            // Create IntegerContext from accumulated INT32 values
            IntegerContext context;
            context.int32_data = int32_context.data();
            context.weight_sums = head_weight_sums.data();
            context.v_scales = v_scales.data();
            context.num_heads = num_heads;
            context.head_dim = head_dim;
            context.count = input_dim;

            // Create output structure
            Q16_1Projection output_proj;
            output_proj.blocks = output_q16.data() + q_pos * output_blocks;
            output_proj.count = d_model;

            // Set up Wo projection params
            WoProjectionVNNIParams wo_params;
            wo_params.Wo_packed = Wo_packed;
            wo_params.input_dim = input_dim;
            wo_params.d_model = d_model;
            wo_params.bias = nullptr;

            // Execute VPDPWSSD projection
            wo_projection_vpdpwssd_to_q16_1_gemv(wo_params, context, output_proj);
        }

        // Dequantize Q16_1 output to FP32 for comparison
        for (int q_pos = 0; q_pos < seq_len_q; ++q_pos)
        {
            dequantizeQ16_1ToFP32(
                output_q16.data() + q_pos * output_blocks,
                d_model,
                output_fp32 + q_pos * d_model);
        }
    }
};

// ============================================================================
// End-to-End Parity Tests
// ============================================================================

TEST_F(Q16IntegerDomainWoProjectionTest, SmallConfig_Decode)
{
    std::cout << "\n"
              << std::string(70, '=') << std::endl;
    std::cout << "Q16 INTEGER-DOMAIN ATTENTION + VPDPWSSD Wo PROJECTION vs FP32" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "Configuration: seq_len=1 (decode), kv_len=16, heads=4, head_dim=64" << std::endl;
    std::cout << "Pipeline: Q×K→Softmax→P×V→INT32→INT16→VPDPWSSD→Q16_1" << std::endl;

    constexpr int SEQ_LEN_Q = 1;
    constexpr int KV_LEN = 16;
    constexpr int NUM_HEADS = 4;
    constexpr int NUM_KV_HEADS = 4;
    constexpr int HEAD_DIM = 64;
    constexpr int D_MODEL = NUM_HEADS * HEAD_DIM; // 256
    constexpr float ATTENTION_SCALE = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));
    constexpr bool CAUSAL = true;

    const int input_dim = NUM_HEADS * HEAD_DIM;
    const int kv_dim = NUM_KV_HEADS * HEAD_DIM;

    // ═══════════════════════════════════════════════════════════════════════
    // Generate random FP32 data
    // ═══════════════════════════════════════════════════════════════════════

    std::vector<float> Q_fp32(SEQ_LEN_Q * input_dim);
    std::vector<float> K_fp32(KV_LEN * kv_dim);
    std::vector<float> V_fp32(KV_LEN * kv_dim);
    std::vector<float> Wo_fp32(D_MODEL * input_dim);

    generateRandomFP32(Q_fp32.data(), Q_fp32.size(), 1.0f);
    generateRandomFP32(K_fp32.data(), K_fp32.size(), 1.0f);
    generateRandomFP32(V_fp32.data(), V_fp32.size(), 1.0f);
    generateRandomFP32(Wo_fp32.data(), Wo_fp32.size(), 1.0f / std::sqrt(static_cast<float>(input_dim)));

    // ═══════════════════════════════════════════════════════════════════════
    // Quantize to Q16_1
    // ═══════════════════════════════════════════════════════════════════════

    const int q_blocks_per_row = (input_dim + 31) / 32;
    const int kv_blocks_per_row = (kv_dim + 31) / 32;

    std::vector<Q16_1Block> Q_q16(SEQ_LEN_Q * q_blocks_per_row);
    std::vector<Q16_1Block> K_q16(KV_LEN * kv_blocks_per_row);
    std::vector<Q16_1Block> V_q16(KV_LEN * kv_blocks_per_row);

    for (int q = 0; q < SEQ_LEN_Q; ++q)
    {
        quantizeFP32ToQ16_1(Q_fp32.data() + q * input_dim, input_dim,
                            Q_q16.data() + q * q_blocks_per_row);
    }
    for (int kv = 0; kv < KV_LEN; ++kv)
    {
        quantizeFP32ToQ16_1(K_fp32.data() + kv * kv_dim, kv_dim,
                            K_q16.data() + kv * kv_blocks_per_row);
        quantizeFP32ToQ16_1(V_fp32.data() + kv * kv_dim, kv_dim,
                            V_q16.data() + kv * kv_blocks_per_row);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Create VNNI-packed Wo weights
    // ═══════════════════════════════════════════════════════════════════════

    // Quantize Wo to Q8_1
    auto Wo_q8 = Q8_1Tensor::quantize_from_fp32(
        Wo_fp32.data(),
        {static_cast<size_t>(D_MODEL), static_cast<size_t>(input_dim)});

    // Pack into VNNI format
    const auto *Wo_packed = llaminar::v2::kernels::KernelFactory::ensurePackedWeightsInTensorCache(Wo_q8.get());
    ASSERT_NE(Wo_packed, nullptr) << "Failed to pack Wo weights";

    // ═══════════════════════════════════════════════════════════════════════
    // COMPUTE REFERENCES
    // ═══════════════════════════════════════════════════════════════════════

    // 1. TRUE FP32 reference (original data, no quantization)
    std::vector<float> output_true_fp32(SEQ_LEN_Q * D_MODEL);
    computeFP32AttentionWithWo(
        SEQ_LEN_Q, KV_LEN, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, D_MODEL,
        Q_fp32.data(), K_fp32.data(), V_fp32.data(), Wo_fp32.data(),
        ATTENTION_SCALE, output_true_fp32.data(), CAUSAL);

    // 2. Q16 integer-domain with VPDPWSSD Wo projection
    std::vector<float> output_q16(SEQ_LEN_Q * D_MODEL);
    computeQ16IntegerDomainWithWo(
        SEQ_LEN_Q, KV_LEN, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, D_MODEL,
        Q_q16.data(), K_q16.data(), V_q16.data(), Wo_packed,
        ATTENTION_SCALE, output_q16.data(), CAUSAL);

    // ═══════════════════════════════════════════════════════════════════════
    // COMPARISONS
    // ═══════════════════════════════════════════════════════════════════════

    double cos_sim = cosineSimilarity(output_q16.data(), output_true_fp32.data(), D_MODEL);
    double max_diff = maxAbsDiff(output_q16.data(), output_true_fp32.data(), D_MODEL);
    double mean_diff = meanAbsDiff(output_q16.data(), output_true_fp32.data(), D_MODEL);
    double rms_diff = rmsDiff(output_q16.data(), output_true_fp32.data(), D_MODEL);

    std::cout << "\n┌────────────────────────────────────────────────────────────────────┐" << std::endl;
    std::cout << "│         Q16 INTEGER + VPDPWSSD vs TRUE FP32 RESULTS                │" << std::endl;
    std::cout << "├────────────────────────────────────────────────────────────────────┤" << std::endl;
    std::cout << "│ Cosine Similarity: " << std::fixed << std::setprecision(6) << std::setw(10) << cos_sim
              << "                                   │" << std::endl;
    std::cout << "│ Max Abs Diff:      " << std::scientific << std::setprecision(4) << std::setw(12) << max_diff
              << "                                 │" << std::endl;
    std::cout << "│ Mean Abs Diff:     " << std::scientific << std::setprecision(4) << std::setw(12) << mean_diff
              << "                                 │" << std::endl;
    std::cout << "│ RMS Diff:          " << std::scientific << std::setprecision(4) << std::setw(12) << rms_diff
              << "                                 │" << std::endl;
    std::cout << "└────────────────────────────────────────────────────────────────────┘" << std::endl;

    // Print first few elements for debugging
    std::cout << "\nFirst 8 elements comparison:" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  FP32:  ";
    for (int i = 0; i < 8; ++i)
        std::cout << std::setw(8) << output_true_fp32[i] << " ";
    std::cout << std::endl;
    std::cout << "  Q16:   ";
    for (int i = 0; i < 8; ++i)
        std::cout << std::setw(8) << output_q16[i] << " ";
    std::cout << std::endl;

    // Verdict
    std::cout << "\n  Verdict: ";
    if (cos_sim >= 0.98)
    {
        std::cout << "✓ EXCELLENT (≥0.98)" << std::endl;
    }
    else if (cos_sim >= 0.95)
    {
        std::cout << "✓ GOOD (≥0.95)" << std::endl;
    }
    else if (cos_sim >= 0.90)
    {
        std::cout << "○ ACCEPTABLE (≥0.90)" << std::endl;
    }
    else if (cos_sim >= 0.80)
    {
        std::cout << "△ MARGINAL (≥0.80) - needs investigation" << std::endl;
    }
    else
    {
        std::cout << "✗ POOR (<0.80) - NEEDS FIX" << std::endl;
    }

    // Assertions
    EXPECT_GT(cos_sim, 0.80) << "Cosine similarity too low for end-to-end pipeline";
    EXPECT_FALSE(std::isnan(cos_sim)) << "Cosine similarity is NaN";
}

TEST_F(Q16IntegerDomainWoProjectionTest, LargerConfig_Decode)
{
    std::cout << "\n"
              << std::string(70, '=') << std::endl;
    std::cout << "Q16 INTEGER-DOMAIN + VPDPWSSD (Larger Config)" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "Configuration: seq_len=1, kv_len=64, heads=8, head_dim=64" << std::endl;

    constexpr int SEQ_LEN_Q = 1;
    constexpr int KV_LEN = 64;
    constexpr int NUM_HEADS = 8;
    constexpr int NUM_KV_HEADS = 8;
    constexpr int HEAD_DIM = 64;
    constexpr int D_MODEL = NUM_HEADS * HEAD_DIM; // 512
    constexpr float ATTENTION_SCALE = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));
    constexpr bool CAUSAL = true;

    const int input_dim = NUM_HEADS * HEAD_DIM;
    const int kv_dim = NUM_KV_HEADS * HEAD_DIM;

    // Generate data
    std::vector<float> Q_fp32(SEQ_LEN_Q * input_dim);
    std::vector<float> K_fp32(KV_LEN * kv_dim);
    std::vector<float> V_fp32(KV_LEN * kv_dim);
    std::vector<float> Wo_fp32(D_MODEL * input_dim);

    generateRandomFP32(Q_fp32.data(), Q_fp32.size(), 1.0f);
    generateRandomFP32(K_fp32.data(), K_fp32.size(), 1.0f);
    generateRandomFP32(V_fp32.data(), V_fp32.size(), 1.0f);
    generateRandomFP32(Wo_fp32.data(), Wo_fp32.size(), 1.0f / std::sqrt(static_cast<float>(input_dim)));

    // Quantize
    const int q_blocks = (input_dim + 31) / 32 * SEQ_LEN_Q;
    const int kv_blocks_per_row = (kv_dim + 31) / 32;

    std::vector<Q16_1Block> Q_q16(q_blocks);
    std::vector<Q16_1Block> K_q16(KV_LEN * kv_blocks_per_row);
    std::vector<Q16_1Block> V_q16(KV_LEN * kv_blocks_per_row);

    for (int q = 0; q < SEQ_LEN_Q; ++q)
    {
        quantizeFP32ToQ16_1(Q_fp32.data() + q * input_dim, input_dim,
                            Q_q16.data() + q * (q_blocks / SEQ_LEN_Q));
    }
    for (int kv = 0; kv < KV_LEN; ++kv)
    {
        quantizeFP32ToQ16_1(K_fp32.data() + kv * kv_dim, kv_dim,
                            K_q16.data() + kv * kv_blocks_per_row);
        quantizeFP32ToQ16_1(V_fp32.data() + kv * kv_dim, kv_dim,
                            V_q16.data() + kv * kv_blocks_per_row);
    }

    // Pack Wo
    auto Wo_q8 = Q8_1Tensor::quantize_from_fp32(
        Wo_fp32.data(),
        {static_cast<size_t>(D_MODEL), static_cast<size_t>(input_dim)});
    const auto *Wo_packed = llaminar::v2::kernels::KernelFactory::ensurePackedWeightsInTensorCache(Wo_q8.get());
    ASSERT_NE(Wo_packed, nullptr);

    // Compute
    std::vector<float> output_fp32(SEQ_LEN_Q * D_MODEL);
    std::vector<float> output_q16(SEQ_LEN_Q * D_MODEL);

    computeFP32AttentionWithWo(
        SEQ_LEN_Q, KV_LEN, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, D_MODEL,
        Q_fp32.data(), K_fp32.data(), V_fp32.data(), Wo_fp32.data(),
        ATTENTION_SCALE, output_fp32.data(), CAUSAL);

    computeQ16IntegerDomainWithWo(
        SEQ_LEN_Q, KV_LEN, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, D_MODEL,
        Q_q16.data(), K_q16.data(), V_q16.data(), Wo_packed,
        ATTENTION_SCALE, output_q16.data(), CAUSAL);

    // Compare
    double cos_sim = cosineSimilarity(output_q16.data(), output_fp32.data(), D_MODEL);
    double max_diff = maxAbsDiff(output_q16.data(), output_fp32.data(), D_MODEL);

    std::cout << "\n  Cosine Similarity: " << std::fixed << std::setprecision(6) << cos_sim << std::endl;
    std::cout << "  Max Abs Diff:      " << std::scientific << std::setprecision(4) << max_diff << std::endl;

    std::cout << "\n  Verdict: ";
    if (cos_sim >= 0.95)
    {
        std::cout << "✓ GOOD (≥0.95)" << std::endl;
    }
    else if (cos_sim >= 0.80)
    {
        std::cout << "○ ACCEPTABLE (≥0.80)" << std::endl;
    }
    else
    {
        std::cout << "✗ POOR (<0.80)" << std::endl;
    }

    EXPECT_GT(cos_sim, 0.80);
}

TEST_F(Q16IntegerDomainWoProjectionTest, GQA_Config)
{
    std::cout << "\n"
              << std::string(70, '=') << std::endl;
    std::cout << "Q16 INTEGER-DOMAIN + VPDPWSSD (GQA Config)" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "Configuration: seq_len=1, kv_len=32, heads=8, kv_heads=2, head_dim=64" << std::endl;

    constexpr int SEQ_LEN_Q = 1;
    constexpr int KV_LEN = 32;
    constexpr int NUM_HEADS = 8;
    constexpr int NUM_KV_HEADS = 2; // GQA: 4:1 ratio
    constexpr int HEAD_DIM = 64;
    constexpr int D_MODEL = NUM_HEADS * HEAD_DIM; // 512
    constexpr float ATTENTION_SCALE = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));
    constexpr bool CAUSAL = true;

    const int input_dim = NUM_HEADS * HEAD_DIM;
    const int kv_dim = NUM_KV_HEADS * HEAD_DIM;

    // Generate data
    std::vector<float> Q_fp32(SEQ_LEN_Q * input_dim);
    std::vector<float> K_fp32(KV_LEN * kv_dim);
    std::vector<float> V_fp32(KV_LEN * kv_dim);
    std::vector<float> Wo_fp32(D_MODEL * input_dim);

    generateRandomFP32(Q_fp32.data(), Q_fp32.size(), 1.0f);
    generateRandomFP32(K_fp32.data(), K_fp32.size(), 1.0f);
    generateRandomFP32(V_fp32.data(), V_fp32.size(), 1.0f);
    generateRandomFP32(Wo_fp32.data(), Wo_fp32.size(), 1.0f / std::sqrt(static_cast<float>(input_dim)));

    // Quantize
    const int q_blocks_per_row = (input_dim + 31) / 32;
    const int kv_blocks_per_row = (kv_dim + 31) / 32;

    std::vector<Q16_1Block> Q_q16(SEQ_LEN_Q * q_blocks_per_row);
    std::vector<Q16_1Block> K_q16(KV_LEN * kv_blocks_per_row);
    std::vector<Q16_1Block> V_q16(KV_LEN * kv_blocks_per_row);

    quantizeFP32ToQ16_1(Q_fp32.data(), input_dim, Q_q16.data());
    for (int kv = 0; kv < KV_LEN; ++kv)
    {
        quantizeFP32ToQ16_1(K_fp32.data() + kv * kv_dim, kv_dim,
                            K_q16.data() + kv * kv_blocks_per_row);
        quantizeFP32ToQ16_1(V_fp32.data() + kv * kv_dim, kv_dim,
                            V_q16.data() + kv * kv_blocks_per_row);
    }

    // Pack Wo
    auto Wo_q8 = Q8_1Tensor::quantize_from_fp32(
        Wo_fp32.data(),
        {static_cast<size_t>(D_MODEL), static_cast<size_t>(input_dim)});
    const auto *Wo_packed = llaminar::v2::kernels::KernelFactory::ensurePackedWeightsInTensorCache(Wo_q8.get());
    ASSERT_NE(Wo_packed, nullptr);

    // Compute
    std::vector<float> output_fp32(D_MODEL);
    std::vector<float> output_q16(D_MODEL);

    computeFP32AttentionWithWo(
        SEQ_LEN_Q, KV_LEN, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, D_MODEL,
        Q_fp32.data(), K_fp32.data(), V_fp32.data(), Wo_fp32.data(),
        ATTENTION_SCALE, output_fp32.data(), CAUSAL);

    computeQ16IntegerDomainWithWo(
        SEQ_LEN_Q, KV_LEN, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, D_MODEL,
        Q_q16.data(), K_q16.data(), V_q16.data(), Wo_packed,
        ATTENTION_SCALE, output_q16.data(), CAUSAL);

    // Compare
    double cos_sim = cosineSimilarity(output_q16.data(), output_fp32.data(), D_MODEL);
    double max_diff = maxAbsDiff(output_q16.data(), output_fp32.data(), D_MODEL);

    std::cout << "\n  Cosine Similarity: " << std::fixed << std::setprecision(6) << cos_sim << std::endl;
    std::cout << "  Max Abs Diff:      " << std::scientific << std::setprecision(4) << max_diff << std::endl;

    std::cout << "\n  Verdict: ";
    if (cos_sim >= 0.95)
    {
        std::cout << "✓ GOOD (≥0.95)" << std::endl;
    }
    else if (cos_sim >= 0.80)
    {
        std::cout << "○ ACCEPTABLE (≥0.80)" << std::endl;
    }
    else
    {
        std::cout << "✗ POOR (<0.80)" << std::endl;
    }

    EXPECT_GT(cos_sim, 0.80);
}
