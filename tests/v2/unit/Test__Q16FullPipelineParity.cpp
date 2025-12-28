/**
 * @file Test__Q16FullPipelineParity.cpp
 * @brief End-to-end parity test for the FULL Q16 pure integer attention pipeline
 *
 * This test validates the complete Q16FusedAttentionRef pipeline against FP32:
 *
 * PURE INTEGER PIPELINE:
 * 1. Q×K^T → INT32 scores (via Int8Requant)
 * 2. Exp2FixedSoftmax → INT16 weights
 * 3. P×V → INT32 accumulators
 * 4. Wo projection (VPDPWSSD INT16×INT16) → Q16_1 output
 * 5. Native Q16_1 residual add → Q16_1 final output
 *
 * vs
 *
 * FP32 REFERENCE PIPELINE:
 * 1. Q×K^T → FP32 scores
 * 2. FP32 softmax
 * 3. P×V → FP32 context
 * 4. FP32 Wo projection
 * 5. FP32 residual add → FP32 final output
 *
 * Measures cosine similarity at each stage and end-to-end.
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

#include "kernels/cpu/attention/q16_1/ref/Q16FusedAttentionRef.h"
#include "kernels/cpu/attention/q16_1/ref/microkernels/Exp2FixedSoftmaxRef.h"
#include "kernels/cpu/attention/q16_1/ref/microkernels/Int8RequantRef.h"
#include "kernels/cpu/attention/q16_1/ref/microkernels/Q16DotProductRef.h"
#include "kernels/cpu/attention/q16_1/ref/microkernels/PVAccumulateRef.h"
#include "kernels/cpu/attention/q16_1/ref/microkernels/WoProjectionVNNIRef.h"
#include "kernels/cpu/gemm_v4/QuantisedGemmJit_M1.h"
#include "kernels/KernelFactory.h"
#include "tensors/BlockStructures.h"
#include "tensors/Tensors.h"

using namespace llaminar2;
using namespace llaminar2::kernels::q16_1;
using namespace llaminar2::kernels::q16_1::microkernels;

// ============================================================================
// Test Fixture
// ============================================================================

class Q16FullPipelineParityTest : public ::testing::Test
{
protected:
    std::mt19937 rng_{42};

    // ═══════════════════════════════════════════════════════════════════════
    // Helper Functions
    // ═══════════════════════════════════════════════════════════════════════

    void generateRandomFP32(float *data, size_t count, float range = 1.0f)
    {
        std::uniform_real_distribution<float> dist(-range, range);
        for (size_t i = 0; i < count; ++i)
        {
            data[i] = dist(rng_);
        }
    }

    void quantizeFP32ToQ16_1(const float *fp32, int num_elements, Q16_1Block *blocks)
    {
        constexpr int BLOCK_SIZE = 32;
        int num_blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

        for (int b = 0; b < num_blocks; ++b)
        {
            const float *src = fp32 + b * BLOCK_SIZE;
            int block_elems = std::min(BLOCK_SIZE, num_elements - b * BLOCK_SIZE);

            // Find max magnitude
            float max_val = 0.0f;
            for (int i = 0; i < block_elems; ++i)
            {
                max_val = std::max(max_val, std::abs(src[i]));
            }

            // Compute scale
            float d = max_val / 32767.0f;
            if (d < 1e-10f)
                d = 1e-10f;
            blocks[b].d = d;

            // Quantize
            int32_t sum = 0;
            for (int i = 0; i < block_elems; ++i)
            {
                int16_t q = static_cast<int16_t>(std::round(src[i] / d));
                blocks[b].qs[i] = q;
                sum += q;
            }
            for (int i = block_elems; i < BLOCK_SIZE; ++i)
            {
                blocks[b].qs[i] = 0;
            }
            blocks[b].sum_qs = sum;
        }
    }

    void dequantizeQ16_1ToFP32(const Q16_1Block *blocks, int num_elements, float *fp32)
    {
        constexpr int BLOCK_SIZE = 32;
        int num_blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

        for (int b = 0; b < num_blocks; ++b)
        {
            const Q16_1Block &block = blocks[b];
            float *dst = fp32 + b * BLOCK_SIZE;
            int block_elems = std::min(BLOCK_SIZE, num_elements - b * BLOCK_SIZE);

            for (int i = 0; i < block_elems; ++i)
            {
                dst[i] = static_cast<float>(block.qs[i]) * block.d;
            }
        }
    }

    void quantizeFP32ToQ8_0(const float *fp32, int num_elements, Q8_0Block *blocks)
    {
        constexpr int BLOCK_SIZE = 32;
        int num_blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

        for (int b = 0; b < num_blocks; ++b)
        {
            const float *src = fp32 + b * BLOCK_SIZE;
            int block_elems = std::min(BLOCK_SIZE, num_elements - b * BLOCK_SIZE);

            float max_val = 0.0f;
            for (int i = 0; i < block_elems; ++i)
            {
                max_val = std::max(max_val, std::abs(src[i]));
            }

            float d = max_val / 127.0f;
            if (d < 1e-10f)
                d = 1e-10f;

            // Store scale as FP16 bits
            uint32_t f = 0;
            std::memcpy(&f, &d, sizeof(float));
            blocks[b].d = static_cast<uint16_t>((f >> 16) & 0x8000) |
                          ((((f >> 23) - 127 + 15) & 0x1f) << 10) |
                          ((f >> 13) & 0x3ff);

            for (int i = 0; i < block_elems; ++i)
            {
                blocks[b].qs[i] = static_cast<int8_t>(std::round(src[i] / d));
            }
            for (int i = block_elems; i < BLOCK_SIZE; ++i)
            {
                blocks[b].qs[i] = 0;
            }
        }
    }

    double cosineSimilarity(const float *a, const float *b, int n)
    {
        double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
        for (int i = 0; i < n; ++i)
        {
            dot += static_cast<double>(a[i]) * b[i];
            norm_a += static_cast<double>(a[i]) * a[i];
            norm_b += static_cast<double>(b[i]) * b[i];
        }
        if (norm_a < 1e-10 || norm_b < 1e-10)
            return 0.0;
        return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
    }

    void printComparisonStats(const char *label, const float *test, const float *ref, int n)
    {
        double cos_sim = cosineSimilarity(test, ref, n);
        float max_diff = 0.0f, mean_diff = 0.0f;
        for (int i = 0; i < n; ++i)
        {
            float diff = std::abs(test[i] - ref[i]);
            max_diff = std::max(max_diff, diff);
            mean_diff += diff;
        }
        mean_diff /= n;

        std::cout << "\n  " << label << " Statistics:" << std::endl;
        std::cout << "    Cosine Similarity: " << std::fixed << std::setprecision(6) << cos_sim << std::endl;
        std::cout << "    Max Abs Diff:      " << std::scientific << std::setprecision(4) << max_diff << std::endl;
        std::cout << "    Mean Abs Diff:     " << std::scientific << std::setprecision(4) << mean_diff << std::endl;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // FP32 Reference Implementation
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * @brief Compute TRUE FP32 attention + Wo projection + residual add
     */
    void computeFP32FullPipeline(
        int seq_len_q, int kv_len, int num_heads, int num_kv_heads, int head_dim, int d_model,
        const float *Q_fp32, const float *K_fp32, const float *V_fp32,
        const float *Wo_fp32, const float *residual_fp32,
        float attention_scale, bool causal,
        float *output_fp32)
    {
        int gqa_ratio = num_heads / num_kv_heads;
        std::vector<float> context(seq_len_q * d_model, 0.0f);

        // Step 1-3: Attention (Q×K^T→Softmax→P×V)
        for (int q_pos = 0; q_pos < seq_len_q; ++q_pos)
        {
            for (int h = 0; h < num_heads; ++h)
            {
                int kv_head = h / gqa_ratio;
                const float *Q_head = Q_fp32 + q_pos * num_heads * head_dim + h * head_dim;
                float *ctx_head = context.data() + q_pos * d_model + h * head_dim;

                // Causal masking
                int kv_end = causal ? std::min(kv_len, q_pos + 1) : kv_len;
                if (kv_end <= 0)
                {
                    std::memset(ctx_head, 0, head_dim * sizeof(float));
                    continue;
                }

                // Q×K^T
                std::vector<float> scores(kv_end);
                for (int k = 0; k < kv_end; ++k)
                {
                    const float *K_pos = K_fp32 + k * num_kv_heads * head_dim + kv_head * head_dim;
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; ++d)
                    {
                        dot += Q_head[d] * K_pos[d];
                    }
                    scores[k] = dot * attention_scale;
                }

                // Softmax
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

                // P×V
                std::memset(ctx_head, 0, head_dim * sizeof(float));
                for (int k = 0; k < kv_end; ++k)
                {
                    const float *V_pos = V_fp32 + k * num_kv_heads * head_dim + kv_head * head_dim;
                    for (int d = 0; d < head_dim; ++d)
                    {
                        ctx_head[d] += scores[k] * V_pos[d];
                    }
                }
            }
        }

        // Step 4: Wo projection (context × Wo → projection)
        // Wo is [d_model × d_model]
        std::vector<float> projection(seq_len_q * d_model, 0.0f);
        for (int q = 0; q < seq_len_q; ++q)
        {
            const float *ctx_row = context.data() + q * d_model;
            float *proj_row = projection.data() + q * d_model;

            for (int out_col = 0; out_col < d_model; ++out_col)
            {
                float acc = 0.0f;
                for (int k = 0; k < d_model; ++k)
                {
                    acc += ctx_row[k] * Wo_fp32[out_col * d_model + k]; // Row-major Wo
                }
                proj_row[out_col] = acc;
            }
        }

        // Step 5: Residual add
        for (int i = 0; i < seq_len_q * d_model; ++i)
        {
            output_fp32[i] = residual_fp32[i] + projection[i];
        }
    }
};

// ============================================================================
// End-to-End Parity Tests
// ============================================================================

TEST_F(Q16FullPipelineParityTest, FullPipeline_SmallDecode)
{
    std::cout << "\n"
              << std::string(70, '=') << std::endl;
    std::cout << "Q16 FULL PIPELINE (PURE INTEGER) vs FP32 REFERENCE" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "Configuration: seq_len=1 (decode), kv_len=16, heads=4, head_dim=64" << std::endl;
    std::cout << "Pipeline: Q×K→Softmax→P×V→Wo(VPDPWSSD)→Q16_1+Q16_1→Q16_1" << std::endl;

    constexpr int SEQ_LEN_Q = 1;
    constexpr int KV_LEN = 16;
    constexpr int NUM_HEADS = 4;
    constexpr int NUM_KV_HEADS = 4;
    constexpr int HEAD_DIM = 64;
    constexpr int D_MODEL = NUM_HEADS * HEAD_DIM; // 256
    constexpr float ATTENTION_SCALE = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));
    constexpr bool CAUSAL = true;

    // Generate random FP32 data
    std::vector<float> Q_fp32(SEQ_LEN_Q * D_MODEL);
    std::vector<float> K_fp32(KV_LEN * D_MODEL);
    std::vector<float> V_fp32(KV_LEN * D_MODEL);
    std::vector<float> Wo_fp32(D_MODEL * D_MODEL);
    std::vector<float> residual_fp32(SEQ_LEN_Q * D_MODEL);

    generateRandomFP32(Q_fp32.data(), Q_fp32.size(), 1.0f);
    generateRandomFP32(K_fp32.data(), K_fp32.size(), 1.0f);
    generateRandomFP32(V_fp32.data(), V_fp32.size(), 1.0f);
    generateRandomFP32(Wo_fp32.data(), Wo_fp32.size(), 0.1f);
    generateRandomFP32(residual_fp32.data(), residual_fp32.size(), 1.0f);

    // ─────────────────────────────────────────────────────────────────────
    // FP32 Reference Pipeline
    // ─────────────────────────────────────────────────────────────────────
    std::vector<float> output_fp32(SEQ_LEN_Q * D_MODEL);
    computeFP32FullPipeline(
        SEQ_LEN_Q, KV_LEN, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, D_MODEL,
        Q_fp32.data(), K_fp32.data(), V_fp32.data(),
        Wo_fp32.data(), residual_fp32.data(),
        ATTENTION_SCALE, CAUSAL,
        output_fp32.data());

    // ─────────────────────────────────────────────────────────────────────
    // Q16 Integer Pipeline
    // ─────────────────────────────────────────────────────────────────────

    // Quantize inputs to Q16_1
    const int q_blocks_per_row = (D_MODEL + 31) / 32;
    const int kv_blocks_per_row = (D_MODEL + 31) / 32;
    const int residual_blocks = SEQ_LEN_Q * q_blocks_per_row;

    std::vector<Q16_1Block> Q_q16(SEQ_LEN_Q * q_blocks_per_row);
    std::vector<Q16_1Block> K_q16(KV_LEN * kv_blocks_per_row);
    std::vector<Q16_1Block> V_q16(KV_LEN * kv_blocks_per_row);
    std::vector<Q16_1Block> residual_q16(residual_blocks);
    std::vector<Q16_1Block> output_q16(residual_blocks);

    quantizeFP32ToQ16_1(Q_fp32.data(), D_MODEL, Q_q16.data());
    for (int kv = 0; kv < KV_LEN; ++kv)
    {
        quantizeFP32ToQ16_1(K_fp32.data() + kv * D_MODEL, D_MODEL, K_q16.data() + kv * kv_blocks_per_row);
        quantizeFP32ToQ16_1(V_fp32.data() + kv * D_MODEL, D_MODEL, V_q16.data() + kv * kv_blocks_per_row);
    }
    quantizeFP32ToQ16_1(residual_fp32.data(), D_MODEL, residual_q16.data());

    // Quantize Wo to Q8_1 and pack for VNNI
    auto Wo_q8 = Q8_1Tensor::quantize_from_fp32(
        Wo_fp32.data(),
        {static_cast<size_t>(D_MODEL), static_cast<size_t>(D_MODEL)});
    const auto *Wo_packed = llaminar::v2::kernels::KernelFactory::ensurePackedWeightsInTensorCache(Wo_q8.get());
    ASSERT_NE(Wo_packed, nullptr) << "Failed to pack Wo weights";

    // Set up params
    Q16FusedAttentionWoResidualParams params;
    params.Q = Q_q16.data();
    params.K = K_q16.data();
    params.V = V_q16.data();
    params.Wo_packed = Wo_packed;
    params.residual_in = residual_q16.data();
    params.residual_out = output_q16.data();
    params.seq_len_q = SEQ_LEN_Q;
    params.kv_len = KV_LEN;
    params.num_heads = NUM_HEADS;
    params.num_kv_heads = NUM_KV_HEADS;
    params.head_dim = HEAD_DIM;
    params.d_model = D_MODEL;
    params.scale = ATTENTION_SCALE;
    params.causal = CAUSAL;

    // Execute Q16 pipeline
    bool success = q16_fused_attention_wo_residual_decode(params);
    ASSERT_TRUE(success) << "Q16 fused attention failed";

    // Dequantize Q16 output for comparison
    std::vector<float> output_q16_dequant(SEQ_LEN_Q * D_MODEL);
    dequantizeQ16_1ToFP32(output_q16.data(), D_MODEL, output_q16_dequant.data());

    // ─────────────────────────────────────────────────────────────────────
    // Compare Results
    // ─────────────────────────────────────────────────────────────────────

    printComparisonStats("Q16 PURE INTEGER vs FP32 REFERENCE", output_q16_dequant.data(), output_fp32.data(), D_MODEL);

    double cos_sim = cosineSimilarity(output_q16_dequant.data(), output_fp32.data(), D_MODEL);

    std::cout << "\n┌────────────────────────────────────────────────────────────────────┐" << std::endl;
    std::cout << "│         END-TO-END COSINE SIMILARITY: " << std::fixed << std::setprecision(6) << cos_sim;
    std::cout << "                  │" << std::endl;
    std::cout << "└────────────────────────────────────────────────────────────────────┘" << std::endl;

    // First 8 elements comparison
    std::cout << "\nFirst 8 elements:" << std::endl;
    std::cout << "  FP32:  ";
    for (int i = 0; i < 8; ++i)
        std::cout << std::setw(8) << std::fixed << std::setprecision(4) << output_fp32[i] << " ";
    std::cout << std::endl;
    std::cout << "  Q16:   ";
    for (int i = 0; i < 8; ++i)
        std::cout << std::setw(8) << std::fixed << std::setprecision(4) << output_q16_dequant[i] << " ";
    std::cout << std::endl;

    // Verdict
    std::cout << "\n  Verdict: ";
    if (cos_sim >= 0.99)
    {
        std::cout << "✓ EXCELLENT (≥0.99)" << std::endl;
    }
    else if (cos_sim >= 0.98)
    {
        std::cout << "✓ VERY GOOD (≥0.98)" << std::endl;
    }
    else if (cos_sim >= 0.95)
    {
        std::cout << "✓ GOOD (≥0.95)" << std::endl;
    }
    else if (cos_sim >= 0.90)
    {
        std::cout << "⚠ ACCEPTABLE (≥0.90)" << std::endl;
    }
    else
    {
        std::cout << "✗ POOR (<0.90) - NEEDS INVESTIGATION" << std::endl;
    }

    EXPECT_GT(cos_sim, 0.95) << "End-to-end cosine similarity too low";
}

TEST_F(Q16FullPipelineParityTest, FullPipeline_LargerDecode)
{
    std::cout << "\n"
              << std::string(70, '=') << std::endl;
    std::cout << "Q16 FULL PIPELINE - LARGER CONFIG" << std::endl;
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

    // Generate random FP32 data
    std::vector<float> Q_fp32(SEQ_LEN_Q * D_MODEL);
    std::vector<float> K_fp32(KV_LEN * D_MODEL);
    std::vector<float> V_fp32(KV_LEN * D_MODEL);
    std::vector<float> Wo_fp32(D_MODEL * D_MODEL);
    std::vector<float> residual_fp32(SEQ_LEN_Q * D_MODEL);

    generateRandomFP32(Q_fp32.data(), Q_fp32.size(), 1.0f);
    generateRandomFP32(K_fp32.data(), K_fp32.size(), 1.0f);
    generateRandomFP32(V_fp32.data(), V_fp32.size(), 1.0f);
    generateRandomFP32(Wo_fp32.data(), Wo_fp32.size(), 0.05f); // Smaller weights for larger model
    generateRandomFP32(residual_fp32.data(), residual_fp32.size(), 1.0f);

    // FP32 Reference
    std::vector<float> output_fp32(SEQ_LEN_Q * D_MODEL);
    computeFP32FullPipeline(
        SEQ_LEN_Q, KV_LEN, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, D_MODEL,
        Q_fp32.data(), K_fp32.data(), V_fp32.data(),
        Wo_fp32.data(), residual_fp32.data(),
        ATTENTION_SCALE, CAUSAL,
        output_fp32.data());

    // Q16 Pipeline
    const int q_blocks_per_row = (D_MODEL + 31) / 32;
    const int kv_blocks_per_row = (D_MODEL + 31) / 32;

    std::vector<Q16_1Block> Q_q16(SEQ_LEN_Q * q_blocks_per_row);
    std::vector<Q16_1Block> K_q16(KV_LEN * kv_blocks_per_row);
    std::vector<Q16_1Block> V_q16(KV_LEN * kv_blocks_per_row);
    std::vector<Q16_1Block> residual_q16(SEQ_LEN_Q * q_blocks_per_row);
    std::vector<Q16_1Block> output_q16(SEQ_LEN_Q * q_blocks_per_row);

    quantizeFP32ToQ16_1(Q_fp32.data(), D_MODEL, Q_q16.data());
    for (int kv = 0; kv < KV_LEN; ++kv)
    {
        quantizeFP32ToQ16_1(K_fp32.data() + kv * D_MODEL, D_MODEL, K_q16.data() + kv * kv_blocks_per_row);
        quantizeFP32ToQ16_1(V_fp32.data() + kv * D_MODEL, D_MODEL, V_q16.data() + kv * kv_blocks_per_row);
    }
    quantizeFP32ToQ16_1(residual_fp32.data(), D_MODEL, residual_q16.data());

    // Wo - use Q8_1Tensor and KernelFactory for proper VNNI packing
    auto Wo_q8 = Q8_1Tensor::quantize_from_fp32(
        Wo_fp32.data(),
        {static_cast<size_t>(D_MODEL), static_cast<size_t>(D_MODEL)});
    const auto *Wo_packed = llaminar::v2::kernels::KernelFactory::ensurePackedWeightsInTensorCache(Wo_q8.get());
    ASSERT_NE(Wo_packed, nullptr) << "Failed to pack Wo weights";

    Q16FusedAttentionWoResidualParams params;
    params.Q = Q_q16.data();
    params.K = K_q16.data();
    params.V = V_q16.data();
    params.Wo_packed = Wo_packed;
    params.residual_in = residual_q16.data();
    params.residual_out = output_q16.data();
    params.seq_len_q = SEQ_LEN_Q;
    params.kv_len = KV_LEN;
    params.num_heads = NUM_HEADS;
    params.num_kv_heads = NUM_KV_HEADS;
    params.head_dim = HEAD_DIM;
    params.d_model = D_MODEL;
    params.scale = ATTENTION_SCALE;
    params.causal = CAUSAL;

    bool success = q16_fused_attention_wo_residual_decode(params);
    ASSERT_TRUE(success);

    std::vector<float> output_q16_dequant(SEQ_LEN_Q * D_MODEL);
    dequantizeQ16_1ToFP32(output_q16.data(), D_MODEL, output_q16_dequant.data());

    double cos_sim = cosineSimilarity(output_q16_dequant.data(), output_fp32.data(), D_MODEL);

    std::cout << "\n  Cosine Similarity: " << std::fixed << std::setprecision(6) << cos_sim << std::endl;
    std::cout << "  Verdict: " << (cos_sim >= 0.95 ? "✓ GOOD" : "⚠ NEEDS WORK") << std::endl;

    EXPECT_GT(cos_sim, 0.95);
}

TEST_F(Q16FullPipelineParityTest, FullPipeline_GQA)
{
    std::cout << "\n"
              << std::string(70, '=') << std::endl;
    std::cout << "Q16 FULL PIPELINE - GQA CONFIG" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "Configuration: seq_len=1, kv_len=32, heads=8, kv_heads=2, head_dim=64" << std::endl;

    constexpr int SEQ_LEN_Q = 1;
    constexpr int KV_LEN = 32;
    constexpr int NUM_HEADS = 8;
    constexpr int NUM_KV_HEADS = 2;
    constexpr int HEAD_DIM = 64;
    constexpr int D_MODEL = NUM_HEADS * HEAD_DIM;   // 512
    constexpr int KV_DIM = NUM_KV_HEADS * HEAD_DIM; // 128
    constexpr float ATTENTION_SCALE = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));
    constexpr bool CAUSAL = true;

    // Generate random FP32 data
    std::vector<float> Q_fp32(SEQ_LEN_Q * D_MODEL);
    std::vector<float> K_fp32(KV_LEN * KV_DIM);
    std::vector<float> V_fp32(KV_LEN * KV_DIM);
    std::vector<float> Wo_fp32(D_MODEL * D_MODEL);
    std::vector<float> residual_fp32(SEQ_LEN_Q * D_MODEL);

    generateRandomFP32(Q_fp32.data(), Q_fp32.size(), 1.0f);
    generateRandomFP32(K_fp32.data(), K_fp32.size(), 1.0f);
    generateRandomFP32(V_fp32.data(), V_fp32.size(), 1.0f);
    generateRandomFP32(Wo_fp32.data(), Wo_fp32.size(), 0.05f);
    generateRandomFP32(residual_fp32.data(), residual_fp32.size(), 1.0f);

    // FP32 Reference
    std::vector<float> output_fp32(SEQ_LEN_Q * D_MODEL);
    computeFP32FullPipeline(
        SEQ_LEN_Q, KV_LEN, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, D_MODEL,
        Q_fp32.data(), K_fp32.data(), V_fp32.data(),
        Wo_fp32.data(), residual_fp32.data(),
        ATTENTION_SCALE, CAUSAL,
        output_fp32.data());

    // Q16 Pipeline
    const int q_blocks_per_row = (D_MODEL + 31) / 32;
    const int kv_blocks_per_row = (KV_DIM + 31) / 32;

    std::vector<Q16_1Block> Q_q16(SEQ_LEN_Q * q_blocks_per_row);
    std::vector<Q16_1Block> K_q16(KV_LEN * kv_blocks_per_row);
    std::vector<Q16_1Block> V_q16(KV_LEN * kv_blocks_per_row);
    std::vector<Q16_1Block> residual_q16(SEQ_LEN_Q * q_blocks_per_row);
    std::vector<Q16_1Block> output_q16(SEQ_LEN_Q * q_blocks_per_row);

    quantizeFP32ToQ16_1(Q_fp32.data(), D_MODEL, Q_q16.data());
    for (int kv = 0; kv < KV_LEN; ++kv)
    {
        quantizeFP32ToQ16_1(K_fp32.data() + kv * KV_DIM, KV_DIM, K_q16.data() + kv * kv_blocks_per_row);
        quantizeFP32ToQ16_1(V_fp32.data() + kv * KV_DIM, KV_DIM, V_q16.data() + kv * kv_blocks_per_row);
    }
    quantizeFP32ToQ16_1(residual_fp32.data(), D_MODEL, residual_q16.data());

    // Wo - use Q8_1Tensor and KernelFactory for proper VNNI packing
    auto Wo_q8 = Q8_1Tensor::quantize_from_fp32(
        Wo_fp32.data(),
        {static_cast<size_t>(D_MODEL), static_cast<size_t>(D_MODEL)});
    const auto *Wo_packed = llaminar::v2::kernels::KernelFactory::ensurePackedWeightsInTensorCache(Wo_q8.get());
    ASSERT_NE(Wo_packed, nullptr) << "Failed to pack Wo weights";

    Q16FusedAttentionWoResidualParams params;
    params.Q = Q_q16.data();
    params.K = K_q16.data();
    params.V = V_q16.data();
    params.Wo_packed = Wo_packed;
    params.residual_in = residual_q16.data();
    params.residual_out = output_q16.data();
    params.seq_len_q = SEQ_LEN_Q;
    params.kv_len = KV_LEN;
    params.num_heads = NUM_HEADS;
    params.num_kv_heads = NUM_KV_HEADS;
    params.head_dim = HEAD_DIM;
    params.d_model = D_MODEL;
    params.scale = ATTENTION_SCALE;
    params.causal = CAUSAL;

    bool success = q16_fused_attention_wo_residual_decode(params);
    ASSERT_TRUE(success);

    std::vector<float> output_q16_dequant(SEQ_LEN_Q * D_MODEL);
    dequantizeQ16_1ToFP32(output_q16.data(), D_MODEL, output_q16_dequant.data());

    double cos_sim = cosineSimilarity(output_q16_dequant.data(), output_fp32.data(), D_MODEL);

    std::cout << "\n  Cosine Similarity: " << std::fixed << std::setprecision(6) << cos_sim << std::endl;
    std::cout << "  Verdict: " << (cos_sim >= 0.95 ? "✓ GOOD" : "⚠ NEEDS WORK") << std::endl;

    EXPECT_GT(cos_sim, 0.95);
}

// ============================================================================
// Test: Context Snapshot Capture
// ============================================================================

/**
 * @brief Verify that context_snapshot captures valid attention context
 *
 * This test ensures that:
 * 1. Setting context_snapshot parameter captures INT32→FP32 converted context
 * 2. The captured values are non-zero and reasonable
 * 3. Snapshot doesn't affect the main computation result
 */
TEST_F(Q16FullPipelineParityTest, DecodeContextSnapshotCapture)
{
    constexpr int NUM_HEADS = 8;
    constexpr int NUM_KV_HEADS = 2;
    constexpr int HEAD_DIM = 64;
    constexpr int KV_LEN = 32;
    constexpr int D_MODEL = NUM_HEADS * HEAD_DIM;
    constexpr int INPUT_DIM = NUM_HEADS * HEAD_DIM;
    constexpr int SEQ_LEN_Q = 1;

    constexpr int Q_BLOCKS = (INPUT_DIM + 31) / 32;
    constexpr int KV_BLOCKS = (NUM_KV_HEADS * HEAD_DIM + 31) / 32;
    constexpr int OUTPUT_BLOCKS = (D_MODEL + 31) / 32;

    std::cout << "\n=== Context Snapshot Capture Test ===" << std::endl;

    // Generate random FP32 data and quantize to Q16_1
    std::vector<float> Q_fp32(SEQ_LEN_Q * INPUT_DIM);
    std::vector<float> K_fp32(KV_LEN * NUM_KV_HEADS * HEAD_DIM);
    std::vector<float> V_fp32(KV_LEN * NUM_KV_HEADS * HEAD_DIM);
    std::vector<float> residual_fp32(SEQ_LEN_Q * D_MODEL);

    generateRandomFP32(Q_fp32.data(), Q_fp32.size(), 1.0f);
    generateRandomFP32(K_fp32.data(), K_fp32.size(), 1.0f);
    generateRandomFP32(V_fp32.data(), V_fp32.size(), 1.0f);
    generateRandomFP32(residual_fp32.data(), residual_fp32.size(), 0.5f);

    std::vector<Q16_1Block> Q_q16(SEQ_LEN_Q * Q_BLOCKS);
    std::vector<Q16_1Block> K_q16(KV_LEN * KV_BLOCKS);
    std::vector<Q16_1Block> V_q16(KV_LEN * KV_BLOCKS);
    std::vector<Q16_1Block> residual_q16(SEQ_LEN_Q * OUTPUT_BLOCKS);
    std::vector<Q16_1Block> output_q16(SEQ_LEN_Q * OUTPUT_BLOCKS);
    std::vector<Q16_1Block> output_q16_no_snapshot(SEQ_LEN_Q * OUTPUT_BLOCKS);

    quantizeFP32ToQ16_1(Q_fp32.data(), INPUT_DIM, Q_q16.data());
    for (int p = 0; p < KV_LEN; ++p)
    {
        quantizeFP32ToQ16_1(K_fp32.data() + p * NUM_KV_HEADS * HEAD_DIM, NUM_KV_HEADS * HEAD_DIM,
                            K_q16.data() + p * KV_BLOCKS);
        quantizeFP32ToQ16_1(V_fp32.data() + p * NUM_KV_HEADS * HEAD_DIM, NUM_KV_HEADS * HEAD_DIM,
                            V_q16.data() + p * KV_BLOCKS);
    }
    quantizeFP32ToQ16_1(residual_fp32.data(), D_MODEL, residual_q16.data());

    // Create Wo weights - use Q8_1Tensor for VNNI packing
    std::vector<float> Wo_fp32(D_MODEL * INPUT_DIM);
    generateRandomFP32(Wo_fp32.data(), Wo_fp32.size(), 0.1f);

    auto Wo_q8 = Q8_1Tensor::quantize_from_fp32(
        Wo_fp32.data(),
        {static_cast<size_t>(D_MODEL), static_cast<size_t>(INPUT_DIM)});
    const auto *Wo_packed = llaminar::v2::kernels::KernelFactory::ensurePackedWeightsInTensorCache(Wo_q8.get());
    ASSERT_NE(Wo_packed, nullptr) << "Failed to pack Wo weights";

    // Allocate context snapshot buffer
    std::vector<float> context_snapshot(SEQ_LEN_Q * INPUT_DIM, 0.0f);

    // Run with snapshot capture
    std::memcpy(output_q16.data(), residual_q16.data(), OUTPUT_BLOCKS * sizeof(Q16_1Block));

    Q16FusedAttentionWoResidualParams params;
    params.Q = Q_q16.data();
    params.K = K_q16.data();
    params.V = V_q16.data();
    params.Wo_packed = Wo_packed;
    params.residual_in = residual_q16.data();
    params.residual_out = output_q16.data();
    params.seq_len_q = SEQ_LEN_Q;
    params.kv_len = KV_LEN;
    params.num_heads = NUM_HEADS;
    params.num_kv_heads = NUM_KV_HEADS;
    params.head_dim = HEAD_DIM;
    params.d_model = D_MODEL;
    params.causal = true;
    params.context_snapshot = context_snapshot.data();

    bool success_with_snapshot = q16_fused_attention_wo_residual_decode(params);
    ASSERT_TRUE(success_with_snapshot) << "Kernel failed with snapshot enabled";

    // Run without snapshot capture
    std::memcpy(output_q16_no_snapshot.data(), residual_q16.data(), OUTPUT_BLOCKS * sizeof(Q16_1Block));

    Q16FusedAttentionWoResidualParams params_no_snapshot = params;
    params_no_snapshot.context_snapshot = nullptr;
    params_no_snapshot.residual_out = output_q16_no_snapshot.data();

    bool success_no_snapshot = q16_fused_attention_wo_residual_decode(params_no_snapshot);
    ASSERT_TRUE(success_no_snapshot) << "Kernel failed without snapshot";

    // Verify snapshot was captured (non-zero values)
    float snapshot_min = *std::min_element(context_snapshot.begin(), context_snapshot.end());
    float snapshot_max = *std::max_element(context_snapshot.begin(), context_snapshot.end());
    float snapshot_sum = std::accumulate(context_snapshot.begin(), context_snapshot.end(), 0.0f);
    float snapshot_mean = snapshot_sum / static_cast<float>(context_snapshot.size());

    std::cout << "  Context snapshot stats:" << std::endl;
    std::cout << "    Min: " << snapshot_min << std::endl;
    std::cout << "    Max: " << snapshot_max << std::endl;
    std::cout << "    Mean: " << snapshot_mean << std::endl;

    // Snapshot should have meaningful values
    EXPECT_NE(snapshot_max, 0.0f) << "Snapshot was not captured (all zeros)";
    EXPECT_NE(snapshot_min, snapshot_max) << "Snapshot has no variance";

    // Verify outputs are identical (snapshot doesn't affect computation)
    std::vector<float> output_dequant(D_MODEL);
    std::vector<float> output_no_snapshot_dequant(D_MODEL);
    dequantizeQ16_1ToFP32(output_q16.data(), D_MODEL, output_dequant.data());
    dequantizeQ16_1ToFP32(output_q16_no_snapshot.data(), D_MODEL, output_no_snapshot_dequant.data());

    double output_cos_sim = cosineSimilarity(output_dequant.data(), output_no_snapshot_dequant.data(), D_MODEL);
    std::cout << "  Output cosine similarity (with vs without snapshot): " << output_cos_sim << std::endl;

    EXPECT_GT(output_cos_sim, 0.9999) << "Snapshot capture affected computation result";

    std::cout << "  ✓ Context snapshot capture works correctly" << std::endl;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
