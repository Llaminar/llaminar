/**
 * @file Test__Q16AttentionContextParity.cpp
 * @brief Unit test to diagnose Q16 integer attention context vs FP32 reference
 *
 * This test isolates the attention context computation to identify where
 * numerical divergence occurs between the Q16 integer pipeline and FP32.
 *
 * Pipeline stages tested:
 * 1. Q×K^T dot products: Q16 Int8Requant vs FP32 dequant
 * 2. Softmax: Exp2FixedSoftmax vs FP32 softmax
 * 3. P×V weighted sum: INT32 accumulation vs FP32 accumulation
 * 4. Context output: INT32→FP32 conversion accuracy
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <random>
#include <numeric>
#include <iomanip>

#include "tensors/BlockStructures.h"
#include "kernels/cpu/attention/q16_1/ref/microkernels/Int8RequantRef.h"
#include "kernels/cpu/attention/q16_1/ref/microkernels/Exp2FixedSoftmaxRef.h"
#include "kernels/cpu/attention/q16_1/ref/microkernels/Q16DotProductRef.h"
#include "utils/Logger.h"

using namespace llaminar2;
using namespace llaminar2::kernels::q16_1;
using namespace llaminar2::kernels::q16_1::microkernels;

// =============================================================================
// Test Utilities
// =============================================================================

/**
 * @brief Compute cosine similarity between two arrays
 */
static double cosine_similarity(const float *a, const float *b, size_t n)
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

/**
 * @brief Compute max absolute difference
 */
static double max_abs_diff(const float *a, const float *b, size_t n)
{
    double max_diff = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        double diff = std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        if (diff > max_diff)
            max_diff = diff;
    }
    return max_diff;
}

/**
 * @brief Create Q16_1 blocks from FP32 data
 */
static void fp32_to_q16_1(const float *input, Q16_1Block *output, int n)
{
    constexpr int BLOCK_SIZE = Q16_1Block::BLOCK_SIZE;
    const int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int b = 0; b < num_blocks; ++b)
    {
        Q16_1Block &block = output[b];
        int start = b * BLOCK_SIZE;
        int end = std::min(start + BLOCK_SIZE, n);

        // Find max absolute value
        float amax = 0.0f;
        for (int i = start; i < end; ++i)
        {
            amax = std::max(amax, std::abs(input[i]));
        }

        // Compute scale
        const float scale = amax / 32767.0f;
        block.d = (scale > 0.0f) ? scale : 1.0f / 32767.0f;
        const float inv_scale = 1.0f / block.d;

        // Quantize and compute sum
        int32_t sum = 0;
        for (int i = 0; i < BLOCK_SIZE; ++i)
        {
            int idx = start + i;
            if (idx < end)
            {
                int32_t q = static_cast<int32_t>(std::round(input[idx] * inv_scale));
                q = std::max(-32767, std::min(32767, q));
                block.qs[i] = static_cast<int16_t>(q);
                sum += q;
            }
            else
            {
                block.qs[i] = 0;
            }
        }
        block.sum_qs = sum;
    }
}

/**
 * @brief Dequantize Q16_1 blocks to FP32
 */
static void q16_1_to_fp32(const Q16_1Block *input, float *output, int n)
{
    constexpr int BLOCK_SIZE = Q16_1Block::BLOCK_SIZE;
    const int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int b = 0; b < num_blocks; ++b)
    {
        const Q16_1Block &block = input[b];
        int start = b * BLOCK_SIZE;
        int end = std::min(start + BLOCK_SIZE, n);

        for (int i = 0; i < BLOCK_SIZE && start + i < end; ++i)
        {
            output[start + i] = static_cast<float>(block.qs[i]) * block.d;
        }
    }
}

/**
 * @brief FP32 reference implementation of softmax
 */
static void fp32_softmax(const float *input, float *output, int n)
{
    // Find max for numerical stability
    float max_val = input[0];
    for (int i = 1; i < n; ++i)
    {
        max_val = std::max(max_val, input[i]);
    }

    // Compute exp and sum
    float sum = 0.0f;
    for (int i = 0; i < n; ++i)
    {
        output[i] = std::exp(input[i] - max_val);
        sum += output[i];
    }

    // Normalize
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < n; ++i)
    {
        output[i] *= inv_sum;
    }
}

// =============================================================================
// Test Fixture
// =============================================================================

class Test__Q16AttentionContextParity : public ::testing::Test
{
protected:
    std::mt19937 rng_{42};

    void SetUp() override {}

    /**
     * @brief Generate random Q16_1 tensor with controlled magnitude
     */
    std::vector<Q16_1Block> generate_q16_tensor(int rows, int cols, float magnitude = 1.0f)
    {
        constexpr int BLOCK_SIZE = Q16_1Block::BLOCK_SIZE;
        const int blocks_per_row = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
        std::vector<Q16_1Block> blocks(rows * blocks_per_row);

        std::uniform_real_distribution<float> dist(-magnitude, magnitude);

        for (int r = 0; r < rows; ++r)
        {
            // Generate FP32 row first
            std::vector<float> row_fp32(cols);
            for (int c = 0; c < cols; ++c)
            {
                row_fp32[c] = dist(rng_);
            }

            // Convert to Q16_1
            fp32_to_q16_1(row_fp32.data(), blocks.data() + r * blocks_per_row, cols);
        }

        return blocks;
    }
};

// =============================================================================
// Test: Q×K Dot Product Comparison
// =============================================================================

TEST_F(Test__Q16AttentionContextParity, QKDotProduct_SingleHead)
{
    const int head_dim = 64;
    const int kv_len = 16;
    constexpr int BLOCK_SIZE = Q16_1Block::BLOCK_SIZE;
    const int blocks_per_head = (head_dim + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    LOG_INFO("═══════════════════════════════════════════════════════════════════");
    LOG_INFO("Testing Q×K dot product: head_dim=" << head_dim << ", kv_len=" << kv_len);
    LOG_INFO("═══════════════════════════════════════════════════════════════════");

    // Generate Q (single query) and K (kv_len keys)
    auto Q_blocks = generate_q16_tensor(1, head_dim, 1.0f);
    auto K_blocks = generate_q16_tensor(kv_len, head_dim, 1.0f);

    // Dequantize to FP32 for reference
    std::vector<float> Q_fp32(head_dim);
    q16_1_to_fp32(Q_blocks.data(), Q_fp32.data(), head_dim);

    std::vector<float> K_fp32(kv_len * head_dim);
    for (int k = 0; k < kv_len; ++k)
    {
        q16_1_to_fp32(K_blocks.data() + k * blocks_per_head,
                      K_fp32.data() + k * head_dim, head_dim);
    }

    // Compute FP32 reference scores
    std::vector<float> fp32_scores(kv_len);
    for (int k = 0; k < kv_len; ++k)
    {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d)
        {
            dot += Q_fp32[d] * K_fp32[k * head_dim + d];
        }
        fp32_scores[k] = dot * scale;
    }

    // Compute Q16 Int8Requant scores
    std::vector<int32_t> int32_scores(kv_len);
    float alpha = 0.0f;

    Int8RequantParams params{};
    params.Q = Q_blocks.data();
    params.K = K_blocks.data();
    params.q_row = 0;
    params.head = 0;
    params.kv_head = 0;
    params.head_dim = head_dim;
    params.kv_end = kv_len;
    params.q_blocks_per_row = blocks_per_head;
    params.kv_blocks_per_row = blocks_per_head;
    params.attention_scale = scale;

    compute_int8_requant_logits(params, int32_scores.data(), &alpha);

    // Convert INT32 scores to FP32 for comparison
    std::vector<float> int32_as_fp32(kv_len);
    for (int k = 0; k < kv_len; ++k)
    {
        // INT32 scores are already scaled by alpha
        int32_as_fp32[k] = static_cast<float>(int32_scores[k]) * alpha;
    }

    // Compare
    double cos_sim = cosine_similarity(fp32_scores.data(), int32_as_fp32.data(), kv_len);
    double max_diff = max_abs_diff(fp32_scores.data(), int32_as_fp32.data(), kv_len);

    LOG_INFO("Q×K Scores comparison:");
    LOG_INFO("  Cosine similarity: " << std::fixed << std::setprecision(6) << cos_sim);
    LOG_INFO("  Max abs diff:      " << std::scientific << max_diff);
    LOG_INFO("  Alpha factor:      " << alpha);

    // Print first few scores
    LOG_INFO("\n  Sample scores (FP32 ref vs INT32*alpha):");
    for (int k = 0; k < std::min(5, kv_len); ++k)
    {
        LOG_INFO("    [" << k << "] FP32=" << std::fixed << std::setprecision(4) << fp32_scores[k]
                         << ", INT32=" << int32_scores[k] << " → " << int32_as_fp32[k]
                         << " (diff=" << std::abs(fp32_scores[k] - int32_as_fp32[k]) << ")");
    }

    EXPECT_GT(cos_sim, 0.99) << "Q×K scores should have >0.99 cosine similarity";
}

// =============================================================================
// Test: Softmax Comparison
// =============================================================================

TEST_F(Test__Q16AttentionContextParity, Softmax_Exp2Fixed_vs_FP32)
{
    const int kv_len = 64;

    LOG_INFO("═══════════════════════════════════════════════════════════════════");
    LOG_INFO("Testing Softmax: Exp2FixedSoftmax vs FP32 reference");
    LOG_INFO("═══════════════════════════════════════════════════════════════════");

    // Generate random FP32 scores (typical attention score range)
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    std::vector<float> fp32_scores(kv_len);
    for (int k = 0; k < kv_len; ++k)
    {
        fp32_scores[k] = dist(rng_);
    }

    // FP32 reference softmax
    std::vector<float> fp32_weights(kv_len);
    fp32_softmax(fp32_scores.data(), fp32_weights.data(), kv_len);

    // Convert FP32 scores to INT32 (simulate Int8Requant output)
    // Assume alpha = 1/128 for simplicity (typical for Q8 requant)
    const float alpha = 1.0f / 128.0f;
    std::vector<int32_t> int32_scores(kv_len);
    for (int k = 0; k < kv_len; ++k)
    {
        int32_scores[k] = static_cast<int32_t>(std::round(fp32_scores[k] / alpha));
    }

    // Exp2FixedSoftmax
    std::vector<int16_t> int16_weights(kv_len);
    int32_t weight_sum = 0;
    exp2_fixed_softmax_row(int32_scores.data(), int16_weights.data(), kv_len, alpha, &weight_sum);

    // Convert INT16 weights to FP32 for comparison
    std::vector<float> int16_as_fp32(kv_len);
    for (int k = 0; k < kv_len; ++k)
    {
        // INT16 weights are [0, 32767], sum should be ~32767
        int16_as_fp32[k] = static_cast<float>(int16_weights[k]) / 32767.0f;
    }

    // Normalize by actual weight sum
    float int16_sum = 0.0f;
    for (int k = 0; k < kv_len; ++k)
    {
        int16_sum += int16_as_fp32[k];
    }
    if (int16_sum > 0.0f)
    {
        for (int k = 0; k < kv_len; ++k)
        {
            int16_as_fp32[k] /= int16_sum;
        }
    }

    // Compare
    double cos_sim = cosine_similarity(fp32_weights.data(), int16_as_fp32.data(), kv_len);
    double max_diff = max_abs_diff(fp32_weights.data(), int16_as_fp32.data(), kv_len);

    LOG_INFO("Softmax weights comparison:");
    LOG_INFO("  Cosine similarity: " << std::fixed << std::setprecision(6) << cos_sim);
    LOG_INFO("  Max abs diff:      " << std::scientific << max_diff);
    LOG_INFO("  Weight sum (INT32): " << weight_sum);

    // Print first few weights
    LOG_INFO("\n  Sample weights (FP32 ref vs INT16/32767):");
    for (int k = 0; k < std::min(5, kv_len); ++k)
    {
        LOG_INFO("    [" << k << "] FP32=" << std::fixed << std::setprecision(6) << fp32_weights[k]
                         << ", INT16=" << int16_weights[k] << " → " << int16_as_fp32[k]
                         << " (diff=" << std::abs(fp32_weights[k] - int16_as_fp32[k]) << ")");
    }

    EXPECT_GT(cos_sim, 0.99) << "Softmax weights should have >0.99 cosine similarity";
}

// =============================================================================
// Test: Full Attention Context (P×V) Comparison
// =============================================================================

TEST_F(Test__Q16AttentionContextParity, AttentionContext_SingleHead)
{
    const int head_dim = 64;
    const int kv_len = 16;
    constexpr int BLOCK_SIZE = Q16_1Block::BLOCK_SIZE;
    const int blocks_per_head = (head_dim + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    LOG_INFO("═══════════════════════════════════════════════════════════════════");
    LOG_INFO("Testing Full Attention Context: head_dim=" << head_dim << ", kv_len=" << kv_len);
    LOG_INFO("═══════════════════════════════════════════════════════════════════");

    // Generate Q, K, V
    auto Q_blocks = generate_q16_tensor(1, head_dim, 1.0f);
    auto K_blocks = generate_q16_tensor(kv_len, head_dim, 1.0f);
    auto V_blocks = generate_q16_tensor(kv_len, head_dim, 1.0f);

    // ═══════════════════════════════════════════════════════════════════════
    // FP32 Reference Pipeline
    // ═══════════════════════════════════════════════════════════════════════

    // Dequantize Q, K, V to FP32
    std::vector<float> Q_fp32(head_dim);
    q16_1_to_fp32(Q_blocks.data(), Q_fp32.data(), head_dim);

    std::vector<float> K_fp32(kv_len * head_dim);
    std::vector<float> V_fp32(kv_len * head_dim);
    for (int k = 0; k < kv_len; ++k)
    {
        q16_1_to_fp32(K_blocks.data() + k * blocks_per_head,
                      K_fp32.data() + k * head_dim, head_dim);
        q16_1_to_fp32(V_blocks.data() + k * blocks_per_head,
                      V_fp32.data() + k * head_dim, head_dim);
    }

    // FP32 Q×K scores
    std::vector<float> fp32_scores(kv_len);
    for (int k = 0; k < kv_len; ++k)
    {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d)
        {
            dot += Q_fp32[d] * K_fp32[k * head_dim + d];
        }
        fp32_scores[k] = dot * scale;
    }

    // FP32 softmax
    std::vector<float> fp32_weights(kv_len);
    fp32_softmax(fp32_scores.data(), fp32_weights.data(), kv_len);

    // FP32 weighted V sum → context
    std::vector<float> fp32_context(head_dim, 0.0f);
    for (int k = 0; k < kv_len; ++k)
    {
        for (int d = 0; d < head_dim; ++d)
        {
            fp32_context[d] += fp32_weights[k] * V_fp32[k * head_dim + d];
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Q16 Integer Pipeline
    // ═══════════════════════════════════════════════════════════════════════

    // Q×K via Int8Requant
    std::vector<int32_t> int32_scores(kv_len);
    float alpha = 0.0f;
    {
        Int8RequantParams params{};
        params.Q = Q_blocks.data();
        params.K = K_blocks.data();
        params.q_row = 0;
        params.head = 0;
        params.kv_head = 0;
        params.head_dim = head_dim;
        params.kv_end = kv_len;
        params.q_blocks_per_row = blocks_per_head;
        params.kv_blocks_per_row = blocks_per_head;
        params.attention_scale = scale;

        compute_int8_requant_logits(params, int32_scores.data(), &alpha);
    }

    // Exp2FixedSoftmax
    std::vector<int16_t> int16_weights(kv_len);
    int32_t weight_sum = 0;
    exp2_fixed_softmax_row(int32_scores.data(), int16_weights.data(), kv_len, alpha, &weight_sum);

    // INT32 P×V accumulation
    std::vector<int32_t> int32_context(head_dim, 0);
    float v_scale_sum = 0.0f;
    int v_scale_count = 0;

    for (int k = 0; k < kv_len; ++k)
    {
        int16_t w = int16_weights[k];
        if (w == 0)
            continue;

        for (int d = 0; d < head_dim; ++d)
        {
            int block_idx = d / BLOCK_SIZE;
            int elem_idx = d % BLOCK_SIZE;

            const Q16_1Block &v_block = V_blocks[k * blocks_per_head + block_idx];
            int16_t v_val = v_block.qs[elem_idx];

            // INT16 × INT16 → INT32
            int32_context[d] += static_cast<int32_t>(w) * static_cast<int32_t>(v_val);

            if (d == 0)
            {
                v_scale_sum += v_block.d;
                v_scale_count++;
            }
        }
    }

    float avg_v_scale = (v_scale_count > 0) ? v_scale_sum / v_scale_count : 1.0f;

    // Current INT32 → FP32 conversion (as in Q16FusedAttentionRef.cpp)
    std::vector<float> q16_context_current(head_dim);
    float inv_weight_sum = 1.0f / static_cast<float>(weight_sum);
    for (int d = 0; d < head_dim; ++d)
    {
        q16_context_current[d] = static_cast<float>(int32_context[d]) * inv_weight_sum * avg_v_scale;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Alternative: Per-element V scale correction
    // ═══════════════════════════════════════════════════════════════════════
    std::vector<float> q16_context_corrected(head_dim);
    for (int d = 0; d < head_dim; ++d)
    {
        int block_idx = d / BLOCK_SIZE;

        // Compute weighted V scale for this element
        float weighted_v_scale = 0.0f;
        float total_weight = 0.0f;
        for (int k = 0; k < kv_len; ++k)
        {
            float w = static_cast<float>(int16_weights[k]) / 32767.0f;
            if (w > 0.0f)
            {
                const Q16_1Block &v_block = V_blocks[k * blocks_per_head + block_idx];
                weighted_v_scale += w * v_block.d;
                total_weight += w;
            }
        }
        if (total_weight > 0.0f)
        {
            weighted_v_scale /= total_weight;
        }
        else
        {
            weighted_v_scale = avg_v_scale;
        }

        // INT32 value = sum(w_k * v_k_int16) where w_k in [0, 32767]
        // FP32 should be: sum(w_k/32767 * v_k_int16 * v_scale_k)
        // Our INT32: sum(w_k * v_k_int16)
        // So divide by weight_sum (≈32767) and multiply by v_scale
        q16_context_corrected[d] = static_cast<float>(int32_context[d]) * inv_weight_sum * weighted_v_scale;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Compare Results
    // ═══════════════════════════════════════════════════════════════════════

    double cos_current = cosine_similarity(fp32_context.data(), q16_context_current.data(), head_dim);
    double cos_corrected = cosine_similarity(fp32_context.data(), q16_context_corrected.data(), head_dim);
    double max_diff_current = max_abs_diff(fp32_context.data(), q16_context_current.data(), head_dim);
    double max_diff_corrected = max_abs_diff(fp32_context.data(), q16_context_corrected.data(), head_dim);

    LOG_INFO("\nContext comparison:");
    LOG_INFO("  Current formula (avg v_scale):");
    LOG_INFO("    Cosine similarity: " << std::fixed << std::setprecision(6) << cos_current);
    LOG_INFO("    Max abs diff:      " << std::scientific << max_diff_current);

    LOG_INFO("  Corrected formula (per-block v_scale):");
    LOG_INFO("    Cosine similarity: " << std::fixed << std::setprecision(6) << cos_corrected);
    LOG_INFO("    Max abs diff:      " << std::scientific << max_diff_corrected);

    LOG_INFO("\n  Sample context values:");
    for (int d = 0; d < std::min(8, head_dim); ++d)
    {
        LOG_INFO("    [" << d << "] FP32=" << std::fixed << std::setprecision(6) << fp32_context[d]
                         << ", Current=" << q16_context_current[d]
                         << ", Corrected=" << q16_context_corrected[d]);
    }

    // The issue is that neither formula properly accounts for:
    // 1. V elements have different scales per block
    // 2. The INT32 accumulator has weight_sum * v_int16 product
    // 3. We need: context[d] = sum_k(w_k * v_k[d] * v_scale_k[d]) / sum_k(w_k)
    //           = sum_k(w_k/weight_sum * v_k_int16 * v_scale_k[d])

    EXPECT_GT(cos_current, 0.8) << "Current formula should have reasonable similarity";
    LOG_INFO("\n[DIAGNOSIS] If cos_current < 0.9, the INT32→FP32 conversion is the issue.");
    LOG_INFO("[DIAGNOSIS] If cos_current > 0.99 but parity test fails, issue is elsewhere.");
}

// =============================================================================
// Test: Diagnose the exact conversion formula needed
// =============================================================================

TEST_F(Test__Q16AttentionContextParity, ConversionFormulaDiagnosis)
{
    const int head_dim = 64;
    const int kv_len = 8;
    constexpr int BLOCK_SIZE = Q16_1Block::BLOCK_SIZE;
    const int blocks_per_head = (head_dim + BLOCK_SIZE - 1) / BLOCK_SIZE;

    LOG_INFO("═══════════════════════════════════════════════════════════════════");
    LOG_INFO("Diagnosing INT32→FP32 context conversion formula");
    LOG_INFO("═══════════════════════════════════════════════════════════════════");

    // Generate V with known values for easier debugging
    std::vector<Q16_1Block> V_blocks(kv_len * blocks_per_head);
    std::vector<float> V_fp32(kv_len * head_dim);

    // Fill with simple values: V[k][d] = k * 0.1 + d * 0.01
    for (int k = 0; k < kv_len; ++k)
    {
        std::vector<float> v_row(head_dim);
        for (int d = 0; d < head_dim; ++d)
        {
            v_row[d] = k * 0.1f + d * 0.01f;
            V_fp32[k * head_dim + d] = v_row[d];
        }
        fp32_to_q16_1(v_row.data(), V_blocks.data() + k * blocks_per_head, head_dim);
    }

    // Uniform weights for simplicity
    std::vector<int16_t> int16_weights(kv_len);
    int32_t weight_sum = 0;
    for (int k = 0; k < kv_len; ++k)
    {
        int16_weights[k] = 32767 / kv_len; // Uniform weights
        weight_sum += int16_weights[k];
    }

    // FP32 reference: context = sum(w_k * V_k) / sum(w_k)
    std::vector<float> fp32_context(head_dim, 0.0f);
    float fp32_weight_sum = 0.0f;
    for (int k = 0; k < kv_len; ++k)
    {
        float w = static_cast<float>(int16_weights[k]) / 32767.0f;
        fp32_weight_sum += w;
        for (int d = 0; d < head_dim; ++d)
        {
            fp32_context[d] += w * V_fp32[k * head_dim + d];
        }
    }
    for (int d = 0; d < head_dim; ++d)
    {
        fp32_context[d] /= fp32_weight_sum;
    }

    // INT32 accumulation: int32_ctx[d] = sum(w_k_int16 * v_k_int16[d])
    std::vector<int32_t> int32_context(head_dim, 0);
    for (int k = 0; k < kv_len; ++k)
    {
        int16_t w = int16_weights[k];
        for (int d = 0; d < head_dim; ++d)
        {
            int block_idx = d / BLOCK_SIZE;
            int elem_idx = d % BLOCK_SIZE;
            const Q16_1Block &v_block = V_blocks[k * blocks_per_head + block_idx];
            int32_context[d] += static_cast<int32_t>(w) * static_cast<int32_t>(v_block.qs[elem_idx]);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // CORRECT FORMULA:
    //
    // The INT32 accumulator contains: sum(w_k_int16 * v_k_int16)
    // We want: context[d] = sum(w_k * V_k[d]) / sum(w_k)
    //                     = sum((w_k_int16/32767) * (v_k_int16 * v_scale_k)) / sum(w_k_int16/32767)
    //
    // For uniform v_scale:
    // context[d] = sum(w_k_int16 * v_k_int16) * v_scale / sum(w_k_int16)
    //            = int32_context[d] * v_scale / weight_sum
    //
    // For per-block v_scale (correct):
    // This requires knowing which V elements contributed with which weights.
    // ═══════════════════════════════════════════════════════════════════════

    // Formula 1: Single average v_scale
    float avg_v_scale = 0.0f;
    for (int k = 0; k < kv_len; ++k)
    {
        for (int b = 0; b < blocks_per_head; ++b)
        {
            avg_v_scale += V_blocks[k * blocks_per_head + b].d;
        }
    }
    avg_v_scale /= (kv_len * blocks_per_head);

    std::vector<float> ctx_formula1(head_dim);
    for (int d = 0; d < head_dim; ++d)
    {
        ctx_formula1[d] = static_cast<float>(int32_context[d]) / weight_sum * avg_v_scale;
    }

    // Formula 2: Per-position v_scale (still wrong for multi-block)
    std::vector<float> ctx_formula2(head_dim);
    for (int d = 0; d < head_dim; ++d)
    {
        int block_idx = d / BLOCK_SIZE;
        float block_scale = 0.0f;
        for (int k = 0; k < kv_len; ++k)
        {
            block_scale += V_blocks[k * blocks_per_head + block_idx].d;
        }
        block_scale /= kv_len;
        ctx_formula2[d] = static_cast<float>(int32_context[d]) / weight_sum * block_scale;
    }

    // Formula 3: Weighted per-block v_scale (CORRECT)
    std::vector<float> ctx_formula3(head_dim);
    for (int d = 0; d < head_dim; ++d)
    {
        int block_idx = d / BLOCK_SIZE;

        // The INT32 accumulator is: sum(w_k * v_k_int16)
        // We need to reconstruct: sum(w_k * v_k_fp32) / sum(w_k)
        //                       = sum(w_k * v_k_int16 * v_scale_k) / sum(w_k)
        //
        // Since each term has potentially different v_scale_k, we need:
        // ctx[d] = sum(w_k * v_k_int16 * v_scale_k) / weight_sum
        //
        // We can compute this by iterating again...
        float correct_sum = 0.0f;
        for (int k = 0; k < kv_len; ++k)
        {
            float w = static_cast<float>(int16_weights[k]);
            const Q16_1Block &v_block = V_blocks[k * blocks_per_head + block_idx];
            float v_fp32 = static_cast<float>(v_block.qs[d % BLOCK_SIZE]) * v_block.d;
            correct_sum += w * v_fp32;
        }
        ctx_formula3[d] = correct_sum / weight_sum;
    }

    double cos1 = cosine_similarity(fp32_context.data(), ctx_formula1.data(), head_dim);
    double cos2 = cosine_similarity(fp32_context.data(), ctx_formula2.data(), head_dim);
    double cos3 = cosine_similarity(fp32_context.data(), ctx_formula3.data(), head_dim);

    LOG_INFO("\nConversion formula comparison:");
    LOG_INFO("  Formula 1 (avg v_scale):      cos=" << std::fixed << std::setprecision(6) << cos1);
    LOG_INFO("  Formula 2 (per-block avg):    cos=" << std::fixed << std::setprecision(6) << cos2);
    LOG_INFO("  Formula 3 (weighted correct): cos=" << std::fixed << std::setprecision(6) << cos3);

    LOG_INFO("\n  Sample values:");
    for (int d = 0; d < std::min(4, head_dim); ++d)
    {
        LOG_INFO("    [" << d << "] FP32=" << fp32_context[d]
                         << ", F1=" << ctx_formula1[d]
                         << ", F2=" << ctx_formula2[d]
                         << ", F3=" << ctx_formula3[d]);
    }

    EXPECT_GT(cos3, 0.999) << "Formula 3 (correct) should have >0.999 cosine similarity";
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
