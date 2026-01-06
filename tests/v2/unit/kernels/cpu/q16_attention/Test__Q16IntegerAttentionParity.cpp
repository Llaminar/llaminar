/**
 * @file Test__Q16IntegerAttentionParity.cpp
 * @brief Parity tests comparing Q16_1 integer attention vs FP32 reference
 *
 * Validates Q16IntegerAttention produces results within acceptable tolerance
 * of the FP32 reference implementation (CPUAttentionKernelT<FP32>).
 *
 * Metrics:
 * - Cosine similarity: Target > 0.995
 * - Max absolute difference
 * - RMSE
 *
 * @see docs/v2/PROJECT_Q16_INTEGER_ATTENTION_V2.md Phase 8
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "kernels/cpu/attention/q16_1/ref/Q16IntegerAttentionRef.h"
#include "kernels/cpu/attention/CPUAttentionKernelT.h"
#include "tensors/BlockStructures.h"
#include "tensors/Tensors.h"

using namespace llaminar2::kernels::q16_1;
using namespace llaminar2;

// =============================================================================
// Test Fixture
// =============================================================================

class Test__Q16IntegerAttentionParity : public ::testing::Test
{
protected:
    std::mt19937 rng_{42};

    /**
     * @brief Compute FP32 reference attention.
     *
     * Standard attention: softmax(Q @ K^T / sqrt(d)) @ V
     * For fair comparison, uses the same causal masking as Q16 prefill.
     *
     * @param Q FP32 query [seq_len_q, num_heads, head_dim]
     * @param K FP32 key [kv_len, num_kv_heads, head_dim]
     * @param V FP32 value [kv_len, num_kv_heads, head_dim]
     * @param output FP32 output [seq_len_q, num_heads, head_dim]
     * @param seq_len_q Query sequence length
     * @param kv_len KV length
     * @param num_heads Number of query heads
     * @param num_kv_heads Number of KV heads
     * @param head_dim Head dimension
     * @param causal Use causal masking (for prefill)
     */
    void fp32ReferenceAttention(
        const float *Q,
        const float *K,
        const float *V,
        float *output,
        int seq_len_q,
        int kv_len,
        int num_heads,
        int num_kv_heads,
        int head_dim,
        bool causal)
    {
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        for (int h = 0; h < num_heads; ++h)
        {
            int kv_h = (num_kv_heads == num_heads) ? h : (h / (num_heads / num_kv_heads));

            for (int q_pos = 0; q_pos < seq_len_q; ++q_pos)
            {
                // Compute Q @ K^T
                std::vector<float> scores(kv_len);
                float max_score = -std::numeric_limits<float>::infinity();

                for (int k_pos = 0; k_pos < kv_len; ++k_pos)
                {
                    // Causal mask
                    if (causal && k_pos > q_pos)
                    {
                        scores[k_pos] = -std::numeric_limits<float>::infinity();
                        continue;
                    }

                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; ++d)
                    {
                        float q_val = Q[(h * seq_len_q + q_pos) * head_dim + d];
                        float k_val = K[(k_pos * num_kv_heads + kv_h) * head_dim + d];
                        dot += q_val * k_val;
                    }
                    scores[k_pos] = dot * scale;
                    max_score = std::max(max_score, scores[k_pos]);
                }

                // Softmax
                float sum = 0.0f;
                for (int k_pos = 0; k_pos < kv_len; ++k_pos)
                {
                    if (scores[k_pos] > -1e30f)
                    {
                        scores[k_pos] = std::exp(scores[k_pos] - max_score);
                        sum += scores[k_pos];
                    }
                    else
                    {
                        scores[k_pos] = 0.0f;
                    }
                }
                for (int k_pos = 0; k_pos < kv_len; ++k_pos)
                {
                    scores[k_pos] /= sum;
                }

                // Weighted sum of V
                // Output layout: [seq_len_q, num_heads, head_dim] to match pipeline
                for (int d = 0; d < head_dim; ++d)
                {
                    float val = 0.0f;
                    for (int k_pos = 0; k_pos < kv_len; ++k_pos)
                    {
                        float v_val = V[(k_pos * num_kv_heads + kv_h) * head_dim + d];
                        val += scores[k_pos] * v_val;
                    }
                    output[(q_pos * num_heads + h) * head_dim + d] = val;
                }
            }
        }
    }

    /**
     * @brief Transpose K/V from [kv_len, num_kv_heads, head_dim] to [num_kv_heads, kv_len, head_dim].
     *
     * The FP32 reference uses [kv_len, num_kv_heads, head_dim] but Q16 expects [num_kv_heads, kv_len, head_dim].
     */
    std::vector<float> transposeKV(
        const std::vector<float> &input,
        int kv_len,
        int num_kv_heads,
        int head_dim)
    {
        std::vector<float> output(input.size());
        for (int k = 0; k < kv_len; ++k)
        {
            for (int kv_h = 0; kv_h < num_kv_heads; ++kv_h)
            {
                for (int d = 0; d < head_dim; ++d)
                {
                    // Input: [kv_len, num_kv_heads, head_dim]
                    int src_idx = (k * num_kv_heads + kv_h) * head_dim + d;
                    // Output: [num_kv_heads, kv_len, head_dim]
                    int dst_idx = (kv_h * kv_len + k) * head_dim + d;
                    output[dst_idx] = input[src_idx];
                }
            }
        }
        return output;
    }

    /**
     * @brief Get MAX_SAFE_INT16 for a given head_dim to prevent INT32 overflow.
     *
     * Formula: MAX_SAFE_INT16 = floor(sqrt(INT32_MAX / (head_dim / 16)))
     * This ensures that head_dim INT16*INT16 products can sum without overflow.
     */
    static int16_t get_max_safe_int16(int head_dim)
    {
        // From: head_dim * MAX_SAFE^2 < INT32_MAX (2.1e9)
        // MAX_SAFE = sqrt(INT32_MAX / head_dim) = sqrt(2.1e9 / head_dim)
        double max_safe = std::sqrt(2147483647.0 / static_cast<double>(head_dim));
        return static_cast<int16_t>(std::floor(max_safe));
    }

    /**
     * @brief Quantize FP32 tensor to Q16_1 blocks with fixed kv_cache_scale.
     *
     * Uses the same formula as the real pipeline:
     * - d = kv_cache_scale / 32767.0f
     * - int16_val = round(fp32_val * 32767.0f / kv_cache_scale)
     * - Clamp to ±MAX_SAFE_INT16 to prevent INT32 overflow in dot products
     * - Representable range: [-kv_cache_scale, +kv_cache_scale]
     */
    template <typename BlockType>
    std::vector<BlockType> quantizeToQ16(
        const std::vector<float> &fp32_data,
        int rows,
        int cols,
        float kv_cache_scale,
        int head_dim = 64)
    {
        constexpr int BLOCK_SIZE = BlockType::BLOCK_SIZE;
        int blocks_per_row = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int total_blocks = rows * blocks_per_row;

        // Get safe INT16 limit for this head_dim
        int16_t max_safe = get_max_safe_int16(head_dim);

        std::vector<BlockType> blocks(total_blocks);
        // Fixed scale formula matching pipeline
        float d = kv_cache_scale / 32767.0f;
        std::cout << "DEBUG: quantizeToQ16 scale=" << kv_cache_scale << " d=" << d << std::endl;
        float quant_factor = 32767.0f / kv_cache_scale;

        for (int r = 0; r < rows; ++r)
        {
            for (int b = 0; b < blocks_per_row; ++b)
            {
                BlockType &block = blocks[r * blocks_per_row + b];
                block.d = d;
                int32_t sum = 0;

                int start_col = b * BLOCK_SIZE;
                for (int i = 0; i < BLOCK_SIZE; ++i)
                {
                    int col = start_col + i;
                    float val = (col < cols) ? fp32_data[r * cols + col] : 0.0f;
                    int32_t q = static_cast<int32_t>(std::round(val * quant_factor));
                    // Clamp to safe range to prevent INT32 overflow in dot products
                    q = std::clamp(q, static_cast<int32_t>(-max_safe), static_cast<int32_t>(max_safe));
                    block.qs[i] = static_cast<int16_t>(q);
                    sum += block.qs[i];
                }
                block.sum_qs = sum;
            }
        }
        return blocks;
    }

    /**
     * @brief Dequantize Q16_1 blocks back to FP32.
     */
    template <typename BlockType>
    std::vector<float> dequantizeFromQ16(
        const std::vector<BlockType> &blocks,
        int rows,
        int cols)
    {
        constexpr int BLOCK_SIZE = BlockType::BLOCK_SIZE;
        int blocks_per_row = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;

        std::vector<float> fp32_data(rows * cols);

        for (int r = 0; r < rows; ++r)
        {
            for (int b = 0; b < blocks_per_row; ++b)
            {
                const BlockType &block = blocks[r * blocks_per_row + b];
                int start_col = b * BLOCK_SIZE;

                for (int i = 0; i < BLOCK_SIZE; ++i)
                {
                    int col = start_col + i;
                    if (col < cols)
                    {
                        fp32_data[r * cols + col] = block.qs[i] * block.d;
                    }
                }
            }
        }
        return fp32_data;
    }

    /**
     * @brief Generate random FP32 data with normal distribution.
     */
    std::vector<float> generateNormalData(int count, float mean = 0.0f, float stddev = 1.0f)
    {
        std::normal_distribution<float> dist(mean, stddev);
        std::vector<float> data(count);
        for (int i = 0; i < count; ++i)
        {
            data[i] = dist(rng_);
        }
        return data;
    }

    /**
     * @brief Compute cosine similarity between two vectors.
     */
    float cosineSimilarity(const float *a, const float *b, int n)
    {
        float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
        for (int i = 0; i < n; ++i)
        {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }
        if (norm_a < 1e-10f || norm_b < 1e-10f)
            return 0.0f;
        return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
    }

    /**
     * @brief Compute maximum absolute difference.
     */
    float maxAbsDiff(const float *a, const float *b, int n)
    {
        float max_diff = 0.0f;
        for (int i = 0; i < n; ++i)
        {
            max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
        }
        return max_diff;
    }

    /**
     * @brief Compute RMSE between two vectors.
     */
    float rmse(const float *a, const float *b, int n)
    {
        float sum_sq = 0.0f;
        for (int i = 0; i < n; ++i)
        {
            float diff = a[i] - b[i];
            sum_sq += diff * diff;
        }
        return std::sqrt(sum_sq / n);
    }

    /**
     * @brief Print parity metrics.
     */
    void printMetrics(const std::string &label, float cos_sim, float max_diff, float rmse_val)
    {
        std::cout << "[PARITY] " << label << ":\n"
                  << "  Cosine similarity: " << std::fixed << std::setprecision(6) << cos_sim << "\n"
                  << "  Max absolute diff: " << std::scientific << std::setprecision(4) << max_diff << "\n"
                  << "  RMSE:             " << std::scientific << std::setprecision(4) << rmse_val << "\n";
    }
};

// =============================================================================
// Decode Parity Tests (seq_len_q = 1)
// =============================================================================

// =============================================================================
// Pipeline Scale Constant (matches kv_cache_scale in real pipeline)
// =============================================================================
// IMPORTANT: This MUST match the real pipeline's kv_cache_scale (GraphSchema.h:427)
// kv_cache_scale = 256.0 means the representable range is [-256.0, +256.0]
// Block scale d = kv_cache_scale / 32767.0f ≈ 0.00781
// Quantization: int16_val = round(fp32_val * 32767.0f / kv_cache_scale)
// Dequantization: fp32_val = int16_val * d
//
// HISTORY:
// - Originally scale=64, but Q projection values reach 130+ causing 42% clipping
// - Changed to scale=256 to handle Q max_abs ~130 without clipping
// Using kv_cache_scale=256 ensures VNNI safety (INT16*INT16 dot products)
constexpr float KV_CACHE_SCALE = 256.0f;

TEST_F(Test__Q16IntegerAttentionParity, Decode_Block64_MHA_Parity)
{
    // Multi-head attention (MHA) configuration
    constexpr int HEAD_DIM = 64;
    constexpr int KV_LEN = 32;
    constexpr int NUM_HEADS = 4;
    constexpr int NUM_KV_HEADS = 4;
    constexpr int SEQ_LEN_Q = 1;
    // Block scale d = kv_cache_scale / 32767 (this is what the attention kernel uses)
    constexpr float BLOCK_SCALE = KV_CACHE_SCALE / 32767.0f;

    const int total_q = SEQ_LEN_Q * NUM_HEADS * HEAD_DIM;
    const int total_kv = KV_LEN * NUM_KV_HEADS * HEAD_DIM;
    const int total_out = SEQ_LEN_Q * NUM_HEADS * HEAD_DIM;

    // Generate random FP32 data with realistic distributions
    // Real pipeline activations have:
    // - Typical values: stddev ~2-5 (mean 0)
    // - Occasional peaks up to ~130 (but clamped by kv_cache_scale=64)
    // Using stddev=4.0 gives realistic spread: ~95% of values in [-12, 12]
    auto Q_fp32 = generateNormalData(total_q, 0.0f, 4.0f);
    auto K_fp32 = generateNormalData(total_kv, 0.0f, 4.0f);
    auto V_fp32 = generateNormalData(total_kv, 0.0f, 4.0f);

    // Compute FP32 reference
    std::vector<float> ref_output(total_out);
    fp32ReferenceAttention(
        Q_fp32.data(), K_fp32.data(), V_fp32.data(),
        ref_output.data(),
        SEQ_LEN_Q, KV_LEN, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM,
        false // not causal for decode
    );

    // Transpose K/V from [kv_len, num_kv_heads, head_dim] to [num_kv_heads, kv_len, head_dim]
    // to match Q16 expected layout
    auto K_transposed = transposeKV(K_fp32, KV_LEN, NUM_KV_HEADS, HEAD_DIM);
    auto V_transposed = transposeKV(V_fp32, KV_LEN, NUM_KV_HEADS, HEAD_DIM);

    // Quantize to Q16_1 using the pipeline's kv_cache_scale
    auto Q_q16 = quantizeToQ16<Q16_1Block_64>(Q_fp32, NUM_HEADS, HEAD_DIM, KV_CACHE_SCALE, HEAD_DIM);
    auto K_q16 = quantizeToQ16<Q16_1Block_64>(K_transposed, NUM_KV_HEADS * KV_LEN, HEAD_DIM, KV_CACHE_SCALE, HEAD_DIM);
    auto V_q16 = quantizeToQ16<Q16_1Block_64>(V_transposed, NUM_KV_HEADS * KV_LEN, HEAD_DIM, KV_CACHE_SCALE, HEAD_DIM);

    // Head scales = block scale d = kv_cache_scale / 32767
    std::vector<float> q_scales(NUM_HEADS, BLOCK_SCALE);
    std::vector<float> kv_scales(NUM_KV_HEADS, BLOCK_SCALE);

    // Snapshot buffers for Q16 output
    std::vector<float> q16_context(total_out);
    std::vector<float> q16_weights(NUM_HEADS * KV_LEN); // For debug

    // Setup Q16 params
    Q16IntegerAttentionParams params;
    params.Q = Q_q16.data();
    params.K = K_q16.data();
    params.V = V_q16.data();
    params.q_head_scales = q_scales.data();
    params.kv_head_scales = kv_scales.data();
    params.seq_len_q = SEQ_LEN_Q;
    params.kv_len = KV_LEN;
    params.num_heads = NUM_HEADS;
    params.num_kv_heads = NUM_KV_HEADS;
    params.head_dim = HEAD_DIM;
    params.d_model = NUM_HEADS * HEAD_DIM;
    params.block_size = Q16BlockSize::BLOCK_64;
    params.snapshot_context = q16_context.data();
    params.snapshot_weights = q16_weights.data(); // Capture weights

    // Execute Q16 attention
    bool success = q16_integer_attention_decode(params);
    ASSERT_TRUE(success);
    // Debug: check for NaN/Inf in output
    int nan_count = 0, inf_count = 0;
    for (size_t i = 0; i < q16_context.size(); ++i)
    {
        if (std::isnan(q16_context[i]))
            ++nan_count;
        if (std::isinf(q16_context[i]))
            ++inf_count;
    }
    if (nan_count || inf_count)
    {
        std::cout << "DEBUG: Q16 context has " << nan_count << " NaN, " << inf_count << " Inf\n";
        std::cout << "DEBUG: first 8 context values: ";
        for (int i = 0; i < 8; ++i)
            std::cout << q16_context[i] << " ";
        std::cout << "\n";
    }
    // Compute parity metrics
    float cos_sim = cosineSimilarity(ref_output.data(), q16_context.data(), total_out);
    float max_diff = maxAbsDiff(ref_output.data(), q16_context.data(), total_out);
    float rmse_val = rmse(ref_output.data(), q16_context.data(), total_out);

    printMetrics("Decode MHA Block64", cos_sim, max_diff, rmse_val);

    // Debug: print first 8 values from head 0
    std::cout << "  REF first 8: ";
    for (int i = 0; i < 8; ++i)
        std::cout << ref_output[i] << " ";
    std::cout << "\n  Q16 first 8: ";
    for (int i = 0; i < 8; ++i)
        std::cout << q16_context[i] << " ";
    std::cout << "\n";

    // Debug: compute FP32 reference weights for head 0
    {
        float scale = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));
        std::vector<float> ref_scores(KV_LEN);
        float max_s = -1e30f;
        for (int k = 0; k < KV_LEN; ++k)
        {
            float dot = 0;
            for (int d = 0; d < HEAD_DIM; ++d)
            {
                dot += Q_fp32[d] * K_fp32[k * HEAD_DIM + d]; // MHA: kv_h = h = 0
            }
            ref_scores[k] = dot * scale;
            max_s = std::max(max_s, ref_scores[k]);
        }
        float sum = 0;
        std::vector<float> ref_weights(KV_LEN);
        for (int k = 0; k < KV_LEN; ++k)
        {
            ref_weights[k] = std::exp(ref_scores[k] - max_s);
            sum += ref_weights[k];
        }
        for (int k = 0; k < KV_LEN; ++k)
            ref_weights[k] /= sum;

        std::cout << "  FP32 weights[0:8]: ";
        for (int i = 0; i < 8; ++i)
            std::cout << ref_weights[i] << " ";
        std::cout << "\n  Q16  weights[0:8]: ";
        for (int i = 0; i < 8; ++i)
            std::cout << q16_weights[i] << " ";
        std::cout << "\n";

        // Weight sum
        float ref_sum = 0, q16_sum = 0;
        for (int k = 0; k < KV_LEN; ++k)
        {
            ref_sum += ref_weights[k];
            q16_sum += q16_weights[k];
        }
        std::cout << "  Weight sums: FP32=" << ref_sum << " Q16=" << q16_sum << "\n";

        // Debug: show actual quantized values
        std::cout << "  Q16 quantized Q[0:4]: ";
        for (int i = 0; i < 4; ++i)
            std::cout << Q_q16[0].qs[i] << " ";
        std::cout << "\n  Q16 quantized K[0][0:4]: ";
        for (int i = 0; i < 4; ++i)
            std::cout << K_q16[0].qs[i] << " ";
        std::cout << "\n  BLOCK_SCALE = " << BLOCK_SCALE << ", qk_scale = " << (BLOCK_SCALE * BLOCK_SCALE / std::sqrt(static_cast<float>(HEAD_DIM))) << "\n";
        std::cout << "  max_safe_int16 = " << get_max_safe_int16(HEAD_DIM) << "\n";

        // Manually compute a few dot products to verify
        int32_t dot0 = 0, dot1 = 0;
        for (int d = 0; d < HEAD_DIM; ++d)
        {
            dot0 += static_cast<int32_t>(Q_q16[0].qs[d]) * static_cast<int32_t>(K_q16[0].qs[d]);
            dot1 += static_cast<int32_t>(Q_q16[0].qs[d]) * static_cast<int32_t>(K_q16[1].qs[d]);
        }
        std::cout << "  Manual dot products: dot0=" << dot0 << ", dot1=" << dot1 << "\n";
        std::cout << "  FP32 scores: " << (dot0 * BLOCK_SCALE * BLOCK_SCALE / std::sqrt(static_cast<float>(HEAD_DIM)))
                  << ", " << (dot1 * BLOCK_SCALE * BLOCK_SCALE / std::sqrt(static_cast<float>(HEAD_DIM))) << "\n";
    }

    // Assert parity thresholds
    // Q16 quantization introduces error, so we use relaxed thresholds
    EXPECT_GT(cos_sim, 0.95f) << "Cosine similarity should be > 0.95";
    EXPECT_LT(max_diff, 1.0f) << "Max difference should be < 1.0";
}

TEST_F(Test__Q16IntegerAttentionParity, Decode_Block64_GQA_Parity)
{
    // Grouped-query attention (GQA) configuration
    constexpr int HEAD_DIM = 64;
    constexpr int KV_LEN = 32;
    constexpr int NUM_HEADS = 8;
    constexpr int NUM_KV_HEADS = 2; // 4:1 GQA ratio
    constexpr int SEQ_LEN_Q = 1;
    constexpr float BLOCK_SCALE = KV_CACHE_SCALE / 32767.0f;

    const int total_q = SEQ_LEN_Q * NUM_HEADS * HEAD_DIM;
    const int total_kv = KV_LEN * NUM_KV_HEADS * HEAD_DIM;
    const int total_out = SEQ_LEN_Q * NUM_HEADS * HEAD_DIM;

    auto Q_fp32 = generateNormalData(total_q, 0.0f, 4.0f);
    auto K_fp32 = generateNormalData(total_kv, 0.0f, 4.0f);
    auto V_fp32 = generateNormalData(total_kv, 0.0f, 4.0f);

    std::vector<float> ref_output(total_out);
    fp32ReferenceAttention(
        Q_fp32.data(), K_fp32.data(), V_fp32.data(),
        ref_output.data(),
        SEQ_LEN_Q, KV_LEN, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM,
        false);

    // Transpose K/V to Q16 expected layout [num_kv_heads, kv_len, head_dim]
    auto K_transposed = transposeKV(K_fp32, KV_LEN, NUM_KV_HEADS, HEAD_DIM);
    auto V_transposed = transposeKV(V_fp32, KV_LEN, NUM_KV_HEADS, HEAD_DIM);

    auto Q_q16 = quantizeToQ16<Q16_1Block_64>(Q_fp32, NUM_HEADS, HEAD_DIM, KV_CACHE_SCALE, HEAD_DIM);
    auto K_q16 = quantizeToQ16<Q16_1Block_64>(K_transposed, NUM_KV_HEADS * KV_LEN, HEAD_DIM, KV_CACHE_SCALE, HEAD_DIM);
    auto V_q16 = quantizeToQ16<Q16_1Block_64>(V_transposed, NUM_KV_HEADS * KV_LEN, HEAD_DIM, KV_CACHE_SCALE, HEAD_DIM);

    std::vector<float> q_scales(NUM_HEADS, BLOCK_SCALE);
    std::vector<float> kv_scales(NUM_KV_HEADS, BLOCK_SCALE);
    std::vector<float> q16_context(total_out);

    Q16IntegerAttentionParams params;
    params.Q = Q_q16.data();
    params.K = K_q16.data();
    params.V = V_q16.data();
    params.q_head_scales = q_scales.data();
    params.kv_head_scales = kv_scales.data();
    params.seq_len_q = SEQ_LEN_Q;
    params.kv_len = KV_LEN;
    params.num_heads = NUM_HEADS;
    params.num_kv_heads = NUM_KV_HEADS;
    params.head_dim = HEAD_DIM;
    params.d_model = NUM_HEADS * HEAD_DIM;
    params.block_size = Q16BlockSize::BLOCK_64;
    params.snapshot_context = q16_context.data();

    bool success = q16_integer_attention_decode(params);
    ASSERT_TRUE(success);

    float cos_sim = cosineSimilarity(ref_output.data(), q16_context.data(), total_out);
    float max_diff = maxAbsDiff(ref_output.data(), q16_context.data(), total_out);
    float rmse_val = rmse(ref_output.data(), q16_context.data(), total_out);

    printMetrics("Decode GQA Block64", cos_sim, max_diff, rmse_val);

    EXPECT_GT(cos_sim, 0.95f) << "Cosine similarity should be > 0.95";
    EXPECT_LT(max_diff, 1.0f) << "Max difference should be < 1.0";
}

TEST_F(Test__Q16IntegerAttentionParity, Decode_Block64_GQA_Qwen2_05B_Parity)
{
    // Qwen2-0.5B exact configuration: 14 query heads, 2 KV heads (7:1 GQA ratio)
    // This matches the HybridQ16 pipeline test configuration
    constexpr int HEAD_DIM = 64;
    constexpr int KV_LEN = 9; // Same as HybridQ16 test prefill length
    constexpr int NUM_HEADS = 14;
    constexpr int NUM_KV_HEADS = 2; // 7:1 GQA ratio
    constexpr int SEQ_LEN_Q = 9;    // Prefill mode (seq_len == kv_len)
    constexpr float BLOCK_SCALE = KV_CACHE_SCALE / 32767.0f;

    const int total_q = SEQ_LEN_Q * NUM_HEADS * HEAD_DIM;
    const int total_kv = KV_LEN * NUM_KV_HEADS * HEAD_DIM;
    const int total_out = SEQ_LEN_Q * NUM_HEADS * HEAD_DIM;

    auto Q_fp32 = generateNormalData(total_q, 0.0f, 4.0f);
    auto K_fp32 = generateNormalData(total_kv, 0.0f, 4.0f);
    auto V_fp32 = generateNormalData(total_kv, 0.0f, 4.0f);

    std::vector<float> ref_output(total_out);
    fp32ReferenceAttention(
        Q_fp32.data(), K_fp32.data(), V_fp32.data(),
        ref_output.data(),
        SEQ_LEN_Q, KV_LEN, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM,
        true // causal for prefill
    );

    // Transpose K/V to Q16 expected layout [num_kv_heads, kv_len, head_dim]
    auto K_transposed = transposeKV(K_fp32, KV_LEN, NUM_KV_HEADS, HEAD_DIM);
    auto V_transposed = transposeKV(V_fp32, KV_LEN, NUM_KV_HEADS, HEAD_DIM);

    auto Q_q16 = quantizeToQ16<Q16_1Block_64>(Q_fp32, SEQ_LEN_Q * NUM_HEADS, HEAD_DIM, KV_CACHE_SCALE, HEAD_DIM);
    auto K_q16 = quantizeToQ16<Q16_1Block_64>(K_transposed, NUM_KV_HEADS * KV_LEN, HEAD_DIM, KV_CACHE_SCALE, HEAD_DIM);
    auto V_q16 = quantizeToQ16<Q16_1Block_64>(V_transposed, NUM_KV_HEADS * KV_LEN, HEAD_DIM, KV_CACHE_SCALE, HEAD_DIM);

    std::vector<float> q_scales(SEQ_LEN_Q * NUM_HEADS, BLOCK_SCALE);
    std::vector<float> kv_scales(NUM_KV_HEADS, BLOCK_SCALE);
    std::vector<float> q16_context(total_out);

    Q16IntegerAttentionParams params;
    params.Q = Q_q16.data();
    params.K = K_q16.data();
    params.V = V_q16.data();
    params.q_head_scales = q_scales.data();
    params.kv_head_scales = kv_scales.data();
    params.seq_len_q = SEQ_LEN_Q;
    params.kv_len = KV_LEN;
    params.num_heads = NUM_HEADS;
    params.num_kv_heads = NUM_KV_HEADS;
    params.head_dim = HEAD_DIM;
    params.d_model = NUM_HEADS * HEAD_DIM;
    params.block_size = Q16BlockSize::BLOCK_64;
    // Note: prefill path has hardcoded causal=true
    params.snapshot_context = q16_context.data();

    // Prefill mode
    bool success = q16_integer_attention_prefill(params);
    ASSERT_TRUE(success);

    float cos_sim = cosineSimilarity(ref_output.data(), q16_context.data(), total_out);
    float max_diff = maxAbsDiff(ref_output.data(), q16_context.data(), total_out);
    float rmse_val = rmse(ref_output.data(), q16_context.data(), total_out);

    printMetrics("Prefill GQA Qwen2-0.5B (14:2)", cos_sim, max_diff, rmse_val);

    // Print first 8 values per head for debugging
    std::cout << "  REF head0 first 8: ";
    for (int i = 0; i < 8; ++i)
        std::cout << ref_output[i] << " ";
    std::cout << "\n  Q16 head0 first 8: ";
    for (int i = 0; i < 8; ++i)
        std::cout << q16_context[i] << " ";
    std::cout << "\n";

    // Check per-head cosine similarity to identify problematic heads
    std::cout << "  Per-head cosine similarity:\n";
    for (int h = 0; h < NUM_HEADS; ++h)
    {
        int kv_h = h / (NUM_HEADS / NUM_KV_HEADS); // GQA mapping
        int head_offset = h * HEAD_DIM;
        float head_cos = 0.0f;
        for (int q = 0; q < SEQ_LEN_Q; ++q)
        {
            int q_offset = q * NUM_HEADS * HEAD_DIM + head_offset;
            float dot = 0, norm_ref = 0, norm_q16 = 0;
            for (int d = 0; d < HEAD_DIM; ++d)
            {
                dot += ref_output[q_offset + d] * q16_context[q_offset + d];
                norm_ref += ref_output[q_offset + d] * ref_output[q_offset + d];
                norm_q16 += q16_context[q_offset + d] * q16_context[q_offset + d];
            }
            if (norm_ref > 1e-12 && norm_q16 > 1e-12)
                head_cos += dot / (std::sqrt(norm_ref) * std::sqrt(norm_q16));
        }
        head_cos /= SEQ_LEN_Q;
        std::cout << "    Head " << h << " (kv_h=" << kv_h << "): " << head_cos << "\n";
    }

    EXPECT_GT(cos_sim, 0.95f) << "Cosine similarity should be > 0.95";
    EXPECT_LT(max_diff, 1.0f) << "Max difference should be < 1.0";
}

TEST_F(Test__Q16IntegerAttentionParity, Prefill_Block64_GQA_Qwen2_05B_HeadMajorStride)
{
    // Same Qwen2-0.5B config but with HEAD_MAJOR sparse cache layout
    // This mimics the actual pipeline where kv_head_stride = max_seq_len (4096)
    constexpr int HEAD_DIM = 64;
    constexpr int KV_LEN = 9;         // Same as HybridQ16 test
    constexpr int MAX_SEQ_LEN = 4096; // Sparse cache allocation
    constexpr int NUM_HEADS = 14;
    constexpr int NUM_KV_HEADS = 2;
    constexpr int SEQ_LEN_Q = 9;
    constexpr float BLOCK_SCALE = KV_CACHE_SCALE / 32767.0f;

    const int total_q = SEQ_LEN_Q * NUM_HEADS * HEAD_DIM;
    const int total_kv_dense = KV_LEN * NUM_KV_HEADS * HEAD_DIM; // Dense layout for reference
    const int total_out = SEQ_LEN_Q * NUM_HEADS * HEAD_DIM;

    auto Q_fp32 = generateNormalData(total_q, 0.0f, 4.0f);
    // For K/V, generate in dense layout first for reference
    auto K_fp32_dense = generateNormalData(total_kv_dense, 0.0f, 4.0f);
    auto V_fp32_dense = generateNormalData(total_kv_dense, 0.0f, 4.0f);

    // Compute FP32 reference with dense layout
    std::vector<float> ref_output(total_out);
    fp32ReferenceAttention(
        Q_fp32.data(), K_fp32_dense.data(), V_fp32_dense.data(),
        ref_output.data(),
        SEQ_LEN_Q, KV_LEN, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM,
        true // causal for prefill
    );

    // Transpose K/V to head-major layout [num_kv_heads, kv_len, head_dim]
    auto K_transposed = transposeKV(K_fp32_dense, KV_LEN, NUM_KV_HEADS, HEAD_DIM);
    auto V_transposed = transposeKV(V_fp32_dense, KV_LEN, NUM_KV_HEADS, HEAD_DIM);

    // Quantize to Q16_1 for Q (dense layout)
    auto Q_q16 = quantizeToQ16<Q16_1Block_64>(Q_fp32, SEQ_LEN_Q * NUM_HEADS, HEAD_DIM, KV_CACHE_SCALE, HEAD_DIM);

    // For K/V: allocate sparse HEAD_MAJOR layout [num_kv_heads * max_seq_len, head_dim]
    // Each head's data starts at offset head_idx * MAX_SEQ_LEN
    const int blocks_per_row = 1; // head_dim=64, block_size=64
    const int sparse_kv_blocks = NUM_KV_HEADS * MAX_SEQ_LEN * blocks_per_row;
    std::vector<Q16_1Block_64> K_sparse(sparse_kv_blocks); // Zero-initialized
    std::vector<Q16_1Block_64> V_sparse(sparse_kv_blocks); // Zero-initialized

    // Quantize dense K/V
    auto K_q16_dense = quantizeToQ16<Q16_1Block_64>(K_transposed, NUM_KV_HEADS * KV_LEN, HEAD_DIM, KV_CACHE_SCALE, HEAD_DIM);
    auto V_q16_dense = quantizeToQ16<Q16_1Block_64>(V_transposed, NUM_KV_HEADS * KV_LEN, HEAD_DIM, KV_CACHE_SCALE, HEAD_DIM);

    // Copy dense data into sparse layout
    // Dense layout: [h * KV_LEN + pos] -> Sparse layout: [h * MAX_SEQ_LEN + pos]
    for (int h = 0; h < NUM_KV_HEADS; ++h)
    {
        for (int pos = 0; pos < KV_LEN; ++pos)
        {
            int dense_idx = h * KV_LEN + pos;
            int sparse_idx = h * MAX_SEQ_LEN + pos;
            K_sparse[sparse_idx] = K_q16_dense[dense_idx];
            V_sparse[sparse_idx] = V_q16_dense[dense_idx];
        }
    }

    std::vector<float> q_scales(SEQ_LEN_Q * NUM_HEADS, BLOCK_SCALE);
    std::vector<float> kv_scales(NUM_KV_HEADS, BLOCK_SCALE);
    std::vector<float> q16_context(total_out);

    Q16IntegerAttentionParams params;
    params.Q = Q_q16.data();
    params.K = K_sparse.data(); // Sparse layout
    params.V = V_sparse.data(); // Sparse layout
    params.q_head_scales = q_scales.data();
    params.kv_head_scales = kv_scales.data();
    params.seq_len_q = SEQ_LEN_Q;
    params.kv_len = KV_LEN;
    params.num_heads = NUM_HEADS;
    params.num_kv_heads = NUM_KV_HEADS;
    params.head_dim = HEAD_DIM;
    params.d_model = NUM_HEADS * HEAD_DIM;
    params.block_size = Q16BlockSize::BLOCK_64;
    params.kv_head_stride = MAX_SEQ_LEN; // KEY: Set stride for sparse layout
    params.snapshot_context = q16_context.data();

    bool success = q16_integer_attention_prefill(params);
    ASSERT_TRUE(success);

    float cos_sim = cosineSimilarity(ref_output.data(), q16_context.data(), total_out);
    float max_diff = maxAbsDiff(ref_output.data(), q16_context.data(), total_out);
    float rmse_val = rmse(ref_output.data(), q16_context.data(), total_out);

    printMetrics("Prefill GQA Qwen2-0.5B HEAD_MAJOR (stride=4096)", cos_sim, max_diff, rmse_val);

    // Print first 8 values for debugging
    std::cout << "  REF head0 first 8: ";
    for (int i = 0; i < 8; ++i)
        std::cout << ref_output[i] << " ";
    std::cout << "\n  Q16 head0 first 8: ";
    for (int i = 0; i < 8; ++i)
        std::cout << q16_context[i] << " ";
    std::cout << "\n";

    // Per-head analysis
    std::cout << "  Per-head cosine similarity:\n";
    for (int h = 0; h < NUM_HEADS; ++h)
    {
        int kv_h = h / (NUM_HEADS / NUM_KV_HEADS);
        float head_cos = 0.0f;
        for (int q = 0; q < SEQ_LEN_Q; ++q)
        {
            int q_offset = q * NUM_HEADS * HEAD_DIM + h * HEAD_DIM;
            float dot = 0, norm_ref = 0, norm_q16 = 0;
            for (int d = 0; d < HEAD_DIM; ++d)
            {
                dot += ref_output[q_offset + d] * q16_context[q_offset + d];
                norm_ref += ref_output[q_offset + d] * ref_output[q_offset + d];
                norm_q16 += q16_context[q_offset + d] * q16_context[q_offset + d];
            }
            if (norm_ref > 1e-12 && norm_q16 > 1e-12)
                head_cos += dot / (std::sqrt(norm_ref) * std::sqrt(norm_q16));
        }
        head_cos /= SEQ_LEN_Q;
        std::cout << "    Head " << h << " (kv_h=" << kv_h << "): " << head_cos << "\n";
    }

    EXPECT_GT(cos_sim, 0.95f) << "Cosine similarity should be > 0.95";
    EXPECT_LT(max_diff, 1.0f) << "Max difference should be < 1.0";
}

TEST_F(Test__Q16IntegerAttentionParity, Decode_Block128_Parity)
{
    constexpr int HEAD_DIM = 128;
    constexpr int KV_LEN = 32;
    constexpr int NUM_HEADS = 4;
    constexpr int NUM_KV_HEADS = 4;
    constexpr int SEQ_LEN_Q = 1;
    constexpr float BLOCK_SCALE = KV_CACHE_SCALE / 32767.0f;

    const int total_q = SEQ_LEN_Q * NUM_HEADS * HEAD_DIM;
    const int total_kv = KV_LEN * NUM_KV_HEADS * HEAD_DIM;
    const int total_out = SEQ_LEN_Q * NUM_HEADS * HEAD_DIM;

    auto Q_fp32 = generateNormalData(total_q, 0.0f, 4.0f);
    auto K_fp32 = generateNormalData(total_kv, 0.0f, 4.0f);
    auto V_fp32 = generateNormalData(total_kv, 0.0f, 4.0f);

    std::vector<float> ref_output(total_out);
    fp32ReferenceAttention(
        Q_fp32.data(), K_fp32.data(), V_fp32.data(),
        ref_output.data(),
        SEQ_LEN_Q, KV_LEN, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM,
        false);

    // Transpose K/V to Q16 expected layout
    auto K_transposed = transposeKV(K_fp32, KV_LEN, NUM_KV_HEADS, HEAD_DIM);
    auto V_transposed = transposeKV(V_fp32, KV_LEN, NUM_KV_HEADS, HEAD_DIM);

    auto Q_q16 = quantizeToQ16<Q16_1Block_128>(Q_fp32, NUM_HEADS, HEAD_DIM, KV_CACHE_SCALE, HEAD_DIM);
    auto K_q16 = quantizeToQ16<Q16_1Block_128>(K_transposed, NUM_KV_HEADS * KV_LEN, HEAD_DIM, KV_CACHE_SCALE, HEAD_DIM);
    auto V_q16 = quantizeToQ16<Q16_1Block_128>(V_transposed, NUM_KV_HEADS * KV_LEN, HEAD_DIM, KV_CACHE_SCALE, HEAD_DIM);

    std::vector<float> q_scales(NUM_HEADS, BLOCK_SCALE);
    std::vector<float> kv_scales(NUM_KV_HEADS, BLOCK_SCALE);
    std::vector<float> q16_context(total_out);

    Q16IntegerAttentionParams params;
    params.Q = Q_q16.data();
    params.K = K_q16.data();
    params.V = V_q16.data();
    params.q_head_scales = q_scales.data();
    params.kv_head_scales = kv_scales.data();
    params.seq_len_q = SEQ_LEN_Q;
    params.kv_len = KV_LEN;
    params.num_heads = NUM_HEADS;
    params.num_kv_heads = NUM_KV_HEADS;
    params.head_dim = HEAD_DIM;
    params.d_model = NUM_HEADS * HEAD_DIM;
    params.block_size = Q16BlockSize::BLOCK_128;
    params.snapshot_context = q16_context.data();

    bool success = q16_integer_attention_decode(params);
    ASSERT_TRUE(success);

    float cos_sim = cosineSimilarity(ref_output.data(), q16_context.data(), total_out);
    float max_diff = maxAbsDiff(ref_output.data(), q16_context.data(), total_out);
    float rmse_val = rmse(ref_output.data(), q16_context.data(), total_out);

    printMetrics("Decode Block128", cos_sim, max_diff, rmse_val);

    EXPECT_GT(cos_sim, 0.95f) << "Cosine similarity should be > 0.95";
}

// =============================================================================
// Prefill Parity Tests (seq_len_q > 1)
// =============================================================================

TEST_F(Test__Q16IntegerAttentionParity, Prefill_Block64_SmallSequence_Parity)
{
    constexpr int HEAD_DIM = 64;
    constexpr int SEQ_LEN_Q = 8;
    constexpr int KV_LEN = 8;
    constexpr int NUM_HEADS = 2;
    constexpr int NUM_KV_HEADS = 2;
    constexpr float BLOCK_SCALE = KV_CACHE_SCALE / 32767.0f;

    const int total_q = SEQ_LEN_Q * NUM_HEADS * HEAD_DIM;
    const int total_kv = KV_LEN * NUM_KV_HEADS * HEAD_DIM;
    const int total_out = SEQ_LEN_Q * NUM_HEADS * HEAD_DIM;

    auto Q_fp32 = generateNormalData(total_q, 0.0f, 4.0f);
    auto K_fp32 = generateNormalData(total_kv, 0.0f, 4.0f);
    auto V_fp32 = generateNormalData(total_kv, 0.0f, 4.0f);

    std::vector<float> ref_output(total_out);
    fp32ReferenceAttention(
        Q_fp32.data(), K_fp32.data(), V_fp32.data(),
        ref_output.data(),
        SEQ_LEN_Q, KV_LEN, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM,
        true // causal for prefill
    );

    // Transpose K/V to Q16 expected layout
    auto K_transposed = transposeKV(K_fp32, KV_LEN, NUM_KV_HEADS, HEAD_DIM);
    auto V_transposed = transposeKV(V_fp32, KV_LEN, NUM_KV_HEADS, HEAD_DIM);

    auto Q_q16 = quantizeToQ16<Q16_1Block_64>(Q_fp32, SEQ_LEN_Q * NUM_HEADS, HEAD_DIM, KV_CACHE_SCALE, HEAD_DIM);
    auto K_q16 = quantizeToQ16<Q16_1Block_64>(K_transposed, NUM_KV_HEADS * KV_LEN, HEAD_DIM, KV_CACHE_SCALE, HEAD_DIM);
    auto V_q16 = quantizeToQ16<Q16_1Block_64>(V_transposed, NUM_KV_HEADS * KV_LEN, HEAD_DIM, KV_CACHE_SCALE, HEAD_DIM);

    std::vector<float> q_scales(NUM_HEADS, BLOCK_SCALE);
    std::vector<float> kv_scales(NUM_KV_HEADS, BLOCK_SCALE);
    std::vector<float> q16_context(total_out);

    Q16IntegerAttentionParams params;
    params.Q = Q_q16.data();
    params.K = K_q16.data();
    params.V = V_q16.data();
    params.q_head_scales = q_scales.data();
    params.kv_head_scales = kv_scales.data();
    params.seq_len_q = SEQ_LEN_Q;
    params.kv_len = KV_LEN;
    params.num_heads = NUM_HEADS;
    params.num_kv_heads = NUM_KV_HEADS;
    params.head_dim = HEAD_DIM;
    params.d_model = NUM_HEADS * HEAD_DIM;
    params.block_size = Q16BlockSize::BLOCK_64;
    params.snapshot_context = q16_context.data();

    bool success = q16_integer_attention_prefill(params);
    ASSERT_TRUE(success);

    float cos_sim = cosineSimilarity(ref_output.data(), q16_context.data(), total_out);
    float max_diff = maxAbsDiff(ref_output.data(), q16_context.data(), total_out);
    float rmse_val = rmse(ref_output.data(), q16_context.data(), total_out);

    printMetrics("Prefill Small Block64", cos_sim, max_diff, rmse_val);

    EXPECT_GT(cos_sim, 0.90f) << "Cosine similarity should be > 0.90";
}

TEST_F(Test__Q16IntegerAttentionParity, Prefill_Block64_LargerSequence_Parity)
{
    constexpr int HEAD_DIM = 64;
    constexpr int SEQ_LEN_Q = 32;
    constexpr int KV_LEN = 64;
    constexpr int NUM_HEADS = 4;
    constexpr int NUM_KV_HEADS = 2;
    constexpr float BLOCK_SCALE = KV_CACHE_SCALE / 32767.0f;

    const int total_q = SEQ_LEN_Q * NUM_HEADS * HEAD_DIM;
    const int total_kv = KV_LEN * NUM_KV_HEADS * HEAD_DIM;
    const int total_out = SEQ_LEN_Q * NUM_HEADS * HEAD_DIM;

    auto Q_fp32 = generateNormalData(total_q, 0.0f, 4.0f);
    auto K_fp32 = generateNormalData(total_kv, 0.0f, 4.0f);
    auto V_fp32 = generateNormalData(total_kv, 0.0f, 4.0f);

    std::vector<float> ref_output(total_out);
    fp32ReferenceAttention(
        Q_fp32.data(), K_fp32.data(), V_fp32.data(),
        ref_output.data(),
        SEQ_LEN_Q, KV_LEN, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM,
        true);

    // Transpose K/V to Q16 expected layout
    auto K_transposed = transposeKV(K_fp32, KV_LEN, NUM_KV_HEADS, HEAD_DIM);
    auto V_transposed = transposeKV(V_fp32, KV_LEN, NUM_KV_HEADS, HEAD_DIM);

    auto Q_q16 = quantizeToQ16<Q16_1Block_64>(Q_fp32, SEQ_LEN_Q * NUM_HEADS, HEAD_DIM, KV_CACHE_SCALE, HEAD_DIM);
    auto K_q16 = quantizeToQ16<Q16_1Block_64>(K_transposed, NUM_KV_HEADS * KV_LEN, HEAD_DIM, KV_CACHE_SCALE, HEAD_DIM);
    auto V_q16 = quantizeToQ16<Q16_1Block_64>(V_transposed, NUM_KV_HEADS * KV_LEN, HEAD_DIM, KV_CACHE_SCALE, HEAD_DIM);

    std::vector<float> q_scales(NUM_HEADS, BLOCK_SCALE);
    std::vector<float> kv_scales(NUM_KV_HEADS, BLOCK_SCALE);
    std::vector<float> q16_context(total_out);

    Q16IntegerAttentionParams params;
    params.Q = Q_q16.data();
    params.K = K_q16.data();
    params.V = V_q16.data();
    params.q_head_scales = q_scales.data();
    params.kv_head_scales = kv_scales.data();
    params.seq_len_q = SEQ_LEN_Q;
    params.kv_len = KV_LEN;
    params.num_heads = NUM_HEADS;
    params.num_kv_heads = NUM_KV_HEADS;
    params.head_dim = HEAD_DIM;
    params.d_model = NUM_HEADS * HEAD_DIM;
    params.block_size = Q16BlockSize::BLOCK_64;
    params.snapshot_context = q16_context.data();

    bool success = q16_integer_attention_prefill(params);
    ASSERT_TRUE(success);

    float cos_sim = cosineSimilarity(ref_output.data(), q16_context.data(), total_out);
    float max_diff = maxAbsDiff(ref_output.data(), q16_context.data(), total_out);
    float rmse_val = rmse(ref_output.data(), q16_context.data(), total_out);

    printMetrics("Prefill Larger Block64", cos_sim, max_diff, rmse_val);

    EXPECT_GT(cos_sim, 0.85f) << "Cosine similarity should be > 0.85 for larger sequences";
}

// =============================================================================
// Softmax Weight Parity
// =============================================================================

TEST_F(Test__Q16IntegerAttentionParity, SoftmaxWeights_SumToOne)
{
    constexpr int HEAD_DIM = 64;
    constexpr int KV_LEN = 32;
    constexpr int NUM_HEADS = 4;
    constexpr int NUM_KV_HEADS = 4;
    constexpr int SEQ_LEN_Q = 1;
    constexpr float BLOCK_SCALE = KV_CACHE_SCALE / 32767.0f;

    const int total_q = SEQ_LEN_Q * NUM_HEADS * HEAD_DIM;
    const int total_kv = KV_LEN * NUM_KV_HEADS * HEAD_DIM;

    auto Q_fp32 = generateNormalData(total_q, 0.0f, 4.0f);
    auto K_fp32 = generateNormalData(total_kv, 0.0f, 4.0f);
    auto V_fp32 = generateNormalData(total_kv, 0.0f, 4.0f);

    // Transpose K/V to Q16 expected layout
    auto K_transposed = transposeKV(K_fp32, KV_LEN, NUM_KV_HEADS, HEAD_DIM);
    auto V_transposed = transposeKV(V_fp32, KV_LEN, NUM_KV_HEADS, HEAD_DIM);

    auto Q_q16 = quantizeToQ16<Q16_1Block_64>(Q_fp32, NUM_HEADS, HEAD_DIM, KV_CACHE_SCALE, HEAD_DIM);
    auto K_q16 = quantizeToQ16<Q16_1Block_64>(K_transposed, NUM_KV_HEADS * KV_LEN, HEAD_DIM, KV_CACHE_SCALE, HEAD_DIM);
    auto V_q16 = quantizeToQ16<Q16_1Block_64>(V_transposed, NUM_KV_HEADS * KV_LEN, HEAD_DIM, KV_CACHE_SCALE, HEAD_DIM);

    std::vector<float> q_scales(NUM_HEADS, BLOCK_SCALE);
    std::vector<float> kv_scales(NUM_KV_HEADS, BLOCK_SCALE);
    std::vector<float> q16_weights(NUM_HEADS * KV_LEN);

    Q16IntegerAttentionParams params;
    params.Q = Q_q16.data();
    params.K = K_q16.data();
    params.V = V_q16.data();
    params.q_head_scales = q_scales.data();
    params.kv_head_scales = kv_scales.data();
    params.seq_len_q = SEQ_LEN_Q;
    params.kv_len = KV_LEN;
    params.num_heads = NUM_HEADS;
    params.num_kv_heads = NUM_KV_HEADS;
    params.head_dim = HEAD_DIM;
    params.d_model = NUM_HEADS * HEAD_DIM;
    params.block_size = Q16BlockSize::BLOCK_64;
    params.snapshot_weights = q16_weights.data();

    bool success = q16_integer_attention_decode(params);
    ASSERT_TRUE(success);

    // Verify softmax weights sum to 1.0 for each head
    for (int h = 0; h < NUM_HEADS; ++h)
    {
        float sum = 0.0f;
        for (int k = 0; k < KV_LEN; ++k)
        {
            float w = q16_weights[h * KV_LEN + k];
            EXPECT_GE(w, 0.0f) << "Weight should be non-negative at h=" << h << " k=" << k;
            sum += w;
        }
        EXPECT_NEAR(sum, 1.0f, 0.02f) << "Weights should sum to 1.0 for head " << h;
    }
}

// =============================================================================
// Long Sequence Tests (stress testing for accumulator overflow)
// =============================================================================

/**
 * @class LongSequenceDecodeTest
 * @brief Long sequence decode tests - INVESTIGATES NUMERICAL LIMITS.
 *
 * Tests the Q16 integer attention decode path with long KV sequences to
 * verify INT32 accumulators don't overflow and accuracy is maintained.
 *
 * KNOWN LIMITATION (KV_CACHE_SCALE = 8.0):
 * With the pipeline's kv_cache_scale = 8.0, the resulting qk_scale is:
 *   qk_scale = (8/32767)² / sqrt(64) ≈ 7.45e-9
 *
 * This causes a numerical precision issue in the LUT-based online softmax:
 * - beta = qk_scale * log2(e) ≈ 1.07e-8
 * - M = beta * 2^24 ≈ 180
 * - For INT32 score delta ~= 5M, t_fixed = (delta * M) >> 13 ≈ 110K
 * - ip = t_fixed >> 11 ≈ 54 (much larger than 31 threshold)
 * - Result: All non-max positions get weight = 0, producing uniform weights
 *
 * This test suite documents these limitations rather than expecting parity.
 * The Q16 attention produces numerically stable output (no NaN/Inf) and
 * weights sum to 1.0, but softmax weight distribution differs from FP32.
 *
 * @see docs/v2/PROJECT_Q16_INTEGER_ATTENTION_V2.md for improvement roadmap
 */
class LongSequenceDecodeTest : public Test__Q16IntegerAttentionParity
{
protected:
    void runDecodeTest(int kv_len, const std::string &label)
    {
        constexpr int HEAD_DIM = 64;
        constexpr int NUM_HEADS = 4;
        constexpr int NUM_KV_HEADS = 4;
        constexpr int SEQ_LEN_Q = 1;
        constexpr float BLOCK_SCALE = KV_CACHE_SCALE / 32767.0f;

        const int total_q = SEQ_LEN_Q * NUM_HEADS * HEAD_DIM;
        const int total_kv = kv_len * NUM_KV_HEADS * HEAD_DIM;
        const int total_out = SEQ_LEN_Q * NUM_HEADS * HEAD_DIM;

        auto Q_fp32 = generateNormalData(total_q, 0.0f, 4.0f);
        auto K_fp32 = generateNormalData(total_kv, 0.0f, 4.0f);
        auto V_fp32 = generateNormalData(total_kv, 0.0f, 4.0f);

        std::vector<float> ref_output(total_out);
        fp32ReferenceAttention(
            Q_fp32.data(), K_fp32.data(), V_fp32.data(),
            ref_output.data(),
            SEQ_LEN_Q, kv_len, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM,
            false);

        auto K_transposed = transposeKV(K_fp32, kv_len, NUM_KV_HEADS, HEAD_DIM);
        auto V_transposed = transposeKV(V_fp32, kv_len, NUM_KV_HEADS, HEAD_DIM);

        auto Q_q16 = quantizeToQ16<Q16_1Block_64>(Q_fp32, NUM_HEADS, HEAD_DIM, KV_CACHE_SCALE, HEAD_DIM);
        auto K_q16 = quantizeToQ16<Q16_1Block_64>(K_transposed, NUM_KV_HEADS * kv_len, HEAD_DIM, KV_CACHE_SCALE, HEAD_DIM);
        auto V_q16 = quantizeToQ16<Q16_1Block_64>(V_transposed, NUM_KV_HEADS * kv_len, HEAD_DIM, KV_CACHE_SCALE, HEAD_DIM);

        std::vector<float> q_scales(NUM_HEADS, BLOCK_SCALE);
        std::vector<float> kv_scales(NUM_KV_HEADS, BLOCK_SCALE);
        std::vector<float> q16_context(total_out);
        std::vector<float> q16_weights(NUM_HEADS * kv_len);

        Q16IntegerAttentionParams params;
        params.Q = Q_q16.data();
        params.K = K_q16.data();
        params.V = V_q16.data();
        params.q_head_scales = q_scales.data();
        params.kv_head_scales = kv_scales.data();
        params.seq_len_q = SEQ_LEN_Q;
        params.kv_len = kv_len;
        params.num_heads = NUM_HEADS;
        params.num_kv_heads = NUM_KV_HEADS;
        params.head_dim = HEAD_DIM;
        params.d_model = NUM_HEADS * HEAD_DIM;
        params.block_size = Q16BlockSize::BLOCK_64;
        params.snapshot_context = q16_context.data();
        params.snapshot_weights = q16_weights.data();

        bool success = q16_integer_attention_decode(params);
        ASSERT_TRUE(success) << "Decode failed for kv_len=" << kv_len;

        // Check for NaN/Inf
        int nan_count = 0, inf_count = 0;
        for (size_t i = 0; i < q16_context.size(); ++i)
        {
            if (std::isnan(q16_context[i]))
                ++nan_count;
            if (std::isinf(q16_context[i]))
                ++inf_count;
        }
        EXPECT_EQ(nan_count, 0) << "Found " << nan_count << " NaN values for kv_len=" << kv_len;
        EXPECT_EQ(inf_count, 0) << "Found " << inf_count << " Inf values for kv_len=" << kv_len;

        float cos_sim = cosineSimilarity(ref_output.data(), q16_context.data(), total_out);
        float max_diff = maxAbsDiff(ref_output.data(), q16_context.data(), total_out);
        float rmse_val = rmse(ref_output.data(), q16_context.data(), total_out);

        printMetrics(label, cos_sim, max_diff, rmse_val);

        // Verify weights sum to 1.0
        for (int h = 0; h < NUM_HEADS; ++h)
        {
            float sum = 0.0f;
            for (int k = 0; k < kv_len; ++k)
            {
                sum += q16_weights[h * kv_len + k];
            }
            EXPECT_NEAR(sum, 1.0f, 0.05f) << "Weights should sum to 1.0 for head " << h << " kv_len=" << kv_len;
        }

        // Use relaxed threshold for long sequences (quantization error accumulates)
        float min_cos_sim = (kv_len > 4096) ? 0.85f : 0.90f;
        EXPECT_GT(cos_sim, min_cos_sim) << "Cosine similarity should be > " << min_cos_sim << " for kv_len=" << kv_len;
    }
};

TEST_F(LongSequenceDecodeTest, Decode_KVLen_512)
{
    runDecodeTest(512, "Decode KV=512");
}

TEST_F(LongSequenceDecodeTest, Decode_KVLen_1024)
{
    runDecodeTest(1024, "Decode KV=1024");
}

TEST_F(LongSequenceDecodeTest, Decode_KVLen_2048)
{
    runDecodeTest(2048, "Decode KV=2048");
}

TEST_F(LongSequenceDecodeTest, Decode_KVLen_4096)
{
    runDecodeTest(4096, "Decode KV=4096");
}

TEST_F(LongSequenceDecodeTest, Decode_KVLen_8192)
{
    runDecodeTest(8192, "Decode KV=8192");
}

TEST_F(LongSequenceDecodeTest, Decode_KVLen_16384)
{
    runDecodeTest(16384, "Decode KV=16384");
}

// =============================================================================
// Long Sequence Prefill Tests
// =============================================================================

/**
 * @brief Parameterized helper for long sequence prefill tests.
 *
 * Tests the Q16 integer attention prefill path with longer sequences.
 * Uses causal masking.
 */
class LongSequencePrefillTest : public Test__Q16IntegerAttentionParity
{
protected:
    void runPrefillTest(int seq_len, const std::string &label)
    {
        constexpr int HEAD_DIM = 64;
        constexpr int NUM_HEADS = 4;
        constexpr int NUM_KV_HEADS = 2; // GQA
        constexpr float BLOCK_SCALE = KV_CACHE_SCALE / 32767.0f;
        const int kv_len = seq_len; // For prefill, kv_len = seq_len

        const int total_q = seq_len * NUM_HEADS * HEAD_DIM;
        const int total_kv = kv_len * NUM_KV_HEADS * HEAD_DIM;
        const int total_out = seq_len * NUM_HEADS * HEAD_DIM;

        auto Q_fp32 = generateNormalData(total_q, 0.0f, 4.0f);
        auto K_fp32 = generateNormalData(total_kv, 0.0f, 4.0f);
        auto V_fp32 = generateNormalData(total_kv, 0.0f, 4.0f);

        std::vector<float> ref_output(total_out);
        fp32ReferenceAttention(
            Q_fp32.data(), K_fp32.data(), V_fp32.data(),
            ref_output.data(),
            seq_len, kv_len, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM,
            true); // causal

        auto K_transposed = transposeKV(K_fp32, kv_len, NUM_KV_HEADS, HEAD_DIM);
        auto V_transposed = transposeKV(V_fp32, kv_len, NUM_KV_HEADS, HEAD_DIM);

        auto Q_q16 = quantizeToQ16<Q16_1Block_64>(Q_fp32, seq_len * NUM_HEADS, HEAD_DIM, KV_CACHE_SCALE, HEAD_DIM);
        auto K_q16 = quantizeToQ16<Q16_1Block_64>(K_transposed, NUM_KV_HEADS * kv_len, HEAD_DIM, KV_CACHE_SCALE, HEAD_DIM);
        auto V_q16 = quantizeToQ16<Q16_1Block_64>(V_transposed, NUM_KV_HEADS * kv_len, HEAD_DIM, KV_CACHE_SCALE, HEAD_DIM);

        std::vector<float> q_scales(NUM_HEADS, BLOCK_SCALE);
        std::vector<float> kv_scales(NUM_KV_HEADS, BLOCK_SCALE);
        std::vector<float> q16_context(total_out);

        Q16IntegerAttentionParams params;
        params.Q = Q_q16.data();
        params.K = K_q16.data();
        params.V = V_q16.data();
        params.q_head_scales = q_scales.data();
        params.kv_head_scales = kv_scales.data();
        params.seq_len_q = seq_len;
        params.kv_len = kv_len;
        params.num_heads = NUM_HEADS;
        params.num_kv_heads = NUM_KV_HEADS;
        params.head_dim = HEAD_DIM;
        params.d_model = NUM_HEADS * HEAD_DIM;
        params.block_size = Q16BlockSize::BLOCK_64;
        params.snapshot_context = q16_context.data();

        bool success = q16_integer_attention_prefill(params);
        ASSERT_TRUE(success) << "Prefill failed for seq_len=" << seq_len;

        // Check for NaN/Inf
        int nan_count = 0, inf_count = 0;
        for (size_t i = 0; i < q16_context.size(); ++i)
        {
            if (std::isnan(q16_context[i]))
                ++nan_count;
            if (std::isinf(q16_context[i]))
                ++inf_count;
        }
        EXPECT_EQ(nan_count, 0) << "Found " << nan_count << " NaN values for seq_len=" << seq_len;
        EXPECT_EQ(inf_count, 0) << "Found " << inf_count << " Inf values for seq_len=" << seq_len;

        float cos_sim = cosineSimilarity(ref_output.data(), q16_context.data(), total_out);
        float max_diff = maxAbsDiff(ref_output.data(), q16_context.data(), total_out);
        float rmse_val = rmse(ref_output.data(), q16_context.data(), total_out);

        printMetrics(label, cos_sim, max_diff, rmse_val);

        // Prefill has more error accumulation, use relaxed thresholds
        float min_cos_sim = (seq_len > 2048) ? 0.75f : 0.80f;
        EXPECT_GT(cos_sim, min_cos_sim) << "Cosine similarity should be > " << min_cos_sim << " for seq_len=" << seq_len;
    }
};

TEST_F(LongSequencePrefillTest, Prefill_SeqLen_128)
{
    runPrefillTest(128, "Prefill Seq=128");
}

TEST_F(LongSequencePrefillTest, Prefill_SeqLen_256)
{
    runPrefillTest(256, "Prefill Seq=256");
}

TEST_F(LongSequencePrefillTest, Prefill_SeqLen_512)
{
    runPrefillTest(512, "Prefill Seq=512");
}

TEST_F(LongSequencePrefillTest, Prefill_SeqLen_1024)
{
    runPrefillTest(1024, "Prefill Seq=1024");
}

TEST_F(LongSequencePrefillTest, Prefill_SeqLen_2048)
{
    runPrefillTest(2048, "Prefill Seq=2048");
}

TEST_F(LongSequencePrefillTest, Prefill_SeqLen_4096)
{
    runPrefillTest(4096, "Prefill Seq=4096");
}

// =============================================================================
// GQA Tests - Grouped Query Attention (n_heads != n_kv_heads)
// Tests Qwen2-0.5B configuration: 14 heads, 2 KV heads (7:1 ratio)
// =============================================================================

/**
 * @brief Test fixture for GQA (Grouped Query Attention) parity tests.
 *
 * These tests verify Q16 attention correctness with n_heads != n_kv_heads,
 * which is the configuration used by Qwen2-0.5B (14 query heads, 2 KV heads).
 */
class GQAAttentionParityTest : public Test__Q16IntegerAttentionParity
{
protected:
    // Qwen2-0.5B GQA configuration
    static constexpr int NUM_HEADS = 14;
    static constexpr int NUM_KV_HEADS = 2;
    static constexpr int HEAD_DIM = 64;
    static constexpr float KV_CACHE_SCALE = 256.0f; // Must match pipeline (was 8, then 64)
    static constexpr float BLOCK_SCALE = KV_CACHE_SCALE / 32767.0f;

    /**
     * @brief Compute FP32 reference using CPUAttentionKernelT<FP32>
     *
     * This uses the actual Llaminar FP32 attention kernel as reference,
     * not a simplified hand-rolled implementation.
     *
     * LAYOUT CONVERSION:
     * - Inline ref & Q16 use Q in [num_heads, seq_len, head_dim] (heads-major)
     * - CPUAttentionKernelT uses Q in [seq_len, num_heads, head_dim] (seq-major)
     * - K/V layouts match: [kv_len, num_kv_heads, head_dim]
     * - Inline ref outputs to [seq_len, num_heads, head_dim]
     * - CPUAttentionKernelT outputs to [seq_len, num_heads, head_dim]
     *
     * So we need to:
     * 1. Transpose Q from [num_heads, seq_len, head_dim] to [seq_len, num_heads, head_dim]
     * 2. K/V can be used directly
     * 3. Output is already in [seq_len, num_heads, head_dim] - no transpose needed
     */
    void computeRealFP32Reference(
        const std::vector<float> &Q_fp32,
        const std::vector<float> &K_fp32,
        const std::vector<float> &V_fp32,
        std::vector<float> &output,
        int seq_len_q,
        int kv_len,
        bool causal)
    {
        // Transpose Q from [num_heads, seq_len_q, head_dim] to [seq_len_q, num_heads, head_dim]
        std::vector<float> Q_transposed(seq_len_q * NUM_HEADS * HEAD_DIM);
        for (int q = 0; q < seq_len_q; ++q)
        {
            for (int h = 0; h < NUM_HEADS; ++h)
            {
                for (int d = 0; d < HEAD_DIM; ++d)
                {
                    // Source: [h][q][d] = (h * seq_len_q + q) * head_dim + d
                    // Dest: [q][h][d] = (q * num_heads + h) * head_dim + d
                    Q_transposed[(q * NUM_HEADS + h) * HEAD_DIM + d] =
                        Q_fp32[(h * seq_len_q + q) * HEAD_DIM + d];
                }
            }
        }

        // Allocate output
        output.resize(seq_len_q * NUM_HEADS * HEAD_DIM);

        // Create the real FP32 attention kernel
        CPUAttentionKernelT<ActivationPrecision::FP32> kernel;

        // Use compute_decode() when seq_len != kv_len (decode case), otherwise compute()
        bool success;
        if (seq_len_q != kv_len)
        {
            // Decode: seq_len_q=1, kv_len=32 (or whatever)
            success = kernel.compute_decode(
                Q_transposed.data(), K_fp32.data(), V_fp32.data(), output.data(),
                seq_len_q, // seq_len
                kv_len,    // kv_len
                NUM_HEADS,
                NUM_KV_HEADS,
                HEAD_DIM,
                causal,
                -1,      // window_size
                nullptr, // workspace_scores
                nullptr, // workspace_buffer
                nullptr, // workspace_context
                nullptr, // workspace_mask
                false,   // use_bf16
                nullptr, // mpi_ctx
                -1);     // device_idx
        }
        else
        {
            // Prefill: seq_len_q == kv_len
            success = kernel.compute(
                Q_transposed.data(), K_fp32.data(), V_fp32.data(), output.data(),
                seq_len_q, // seq_len
                NUM_HEADS,
                NUM_KV_HEADS,
                HEAD_DIM,
                causal,
                -1,      // window_size
                nullptr, // workspace_scores
                nullptr, // workspace_buffer
                nullptr, // workspace_context
                nullptr, // workspace_mask
                false,   // use_bf16
                nullptr, // mpi_ctx
                -1);     // device_idx
        }

        ASSERT_TRUE(success) << "CPUAttentionKernelT<FP32> compute failed";

        // Output is [seq_len_q, num_heads, head_dim] - matches inline ref output layout
        // No transpose needed!
    }
};

/**
 * @brief GQA Decode test using inline FP32 reference (7:1 ratio)
 */
TEST_F(GQAAttentionParityTest, Decode_GQA_7to1_InlineReference)
{
    // Decode: seq_len_q=1, kv_len=32
    constexpr int SEQ_LEN_Q = 1;
    constexpr int KV_LEN = 32;

    const int total_q = SEQ_LEN_Q * NUM_HEADS * HEAD_DIM;
    const int total_kv = KV_LEN * NUM_KV_HEADS * HEAD_DIM;
    const int total_out = SEQ_LEN_Q * NUM_HEADS * HEAD_DIM;

    // Generate random data in [-kv_cache_scale, +kv_cache_scale]
    auto Q_fp32 = generateNormalData(total_q, 0.0f, 4.0f);
    auto K_fp32 = generateNormalData(total_kv, 0.0f, 4.0f);
    auto V_fp32 = generateNormalData(total_kv, 0.0f, 4.0f);

    // FP32 reference (inline implementation)
    std::vector<float> ref_output(total_out);
    fp32ReferenceAttention(
        Q_fp32.data(), K_fp32.data(), V_fp32.data(),
        ref_output.data(),
        SEQ_LEN_Q, KV_LEN, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM,
        false // not causal for decode
    );

    // Transpose K/V to head-major for Q16 kernel
    std::vector<float> K_transposed = this->transposeKV(K_fp32, KV_LEN, NUM_KV_HEADS, HEAD_DIM);
    std::vector<float> V_transposed = this->transposeKV(V_fp32, KV_LEN, NUM_KV_HEADS, HEAD_DIM);

    // Quantize to Q16_1
    auto Q_q16 = quantizeToQ16<Q16_1Block_64>(Q_fp32, NUM_HEADS, HEAD_DIM, KV_CACHE_SCALE, HEAD_DIM);
    auto K_q16 = quantizeToQ16<Q16_1Block_64>(K_transposed, NUM_KV_HEADS * KV_LEN, HEAD_DIM, KV_CACHE_SCALE, HEAD_DIM);
    auto V_q16 = quantizeToQ16<Q16_1Block_64>(V_transposed, NUM_KV_HEADS * KV_LEN, HEAD_DIM, KV_CACHE_SCALE, HEAD_DIM);

    std::vector<float> q_scales(NUM_HEADS, BLOCK_SCALE);
    std::vector<float> kv_scales(NUM_KV_HEADS, BLOCK_SCALE);
    std::vector<float> q16_context(total_out);

    Q16IntegerAttentionParams params;
    params.Q = Q_q16.data();
    params.K = K_q16.data();
    params.V = V_q16.data();
    params.q_head_scales = q_scales.data();
    params.kv_head_scales = kv_scales.data();
    params.seq_len_q = SEQ_LEN_Q;
    params.kv_len = KV_LEN;
    params.num_heads = NUM_HEADS;
    params.num_kv_heads = NUM_KV_HEADS;
    params.head_dim = HEAD_DIM;
    params.d_model = NUM_HEADS * HEAD_DIM;
    params.block_size = Q16BlockSize::BLOCK_64;
    params.snapshot_context = q16_context.data();

    bool success = q16_integer_attention_decode(params);
    ASSERT_TRUE(success) << "GQA Decode failed";

    float cos_sim = cosineSimilarity(ref_output.data(), q16_context.data(), total_out);
    float max_diff = maxAbsDiff(ref_output.data(), q16_context.data(), total_out);
    float rmse_val = rmse(ref_output.data(), q16_context.data(), total_out);

    std::cout << "[PARITY] GQA Decode (14:2, inline ref):" << std::endl;
    std::cout << "  Cosine similarity: " << std::fixed << std::setprecision(6) << cos_sim << std::endl;
    std::cout << "  Max absolute diff: " << std::scientific << std::setprecision(4) << max_diff << std::endl;
    std::cout << "  RMSE:             " << rmse_val << std::endl;

    EXPECT_GT(cos_sim, 0.99f) << "GQA decode should have >0.99 cosine similarity with inline ref";
}

/**
 * @brief GQA Prefill test using inline FP32 reference (7:1 ratio)
 */
TEST_F(GQAAttentionParityTest, Prefill_GQA_7to1_InlineReference)
{
    // Prefill: seq_len=9 (Qwen2 prompt), causal
    constexpr int SEQ_LEN = 9;
    constexpr int KV_LEN = SEQ_LEN; // Prefill: kv_len == seq_len

    const int total_q = SEQ_LEN * NUM_HEADS * HEAD_DIM;
    const int total_kv = KV_LEN * NUM_KV_HEADS * HEAD_DIM;
    const int total_out = SEQ_LEN * NUM_HEADS * HEAD_DIM;

    auto Q_fp32 = generateNormalData(total_q, 0.0f, 4.0f);
    auto K_fp32 = generateNormalData(total_kv, 0.0f, 4.0f);
    auto V_fp32 = generateNormalData(total_kv, 0.0f, 4.0f);

    // FP32 reference (inline, causal)
    std::vector<float> ref_output(total_out);
    fp32ReferenceAttention(
        Q_fp32.data(), K_fp32.data(), V_fp32.data(),
        ref_output.data(),
        SEQ_LEN, KV_LEN, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM,
        true // causal for prefill
    );

    // Transpose K/V to head-major
    std::vector<float> K_transposed = this->transposeKV(K_fp32, KV_LEN, NUM_KV_HEADS, HEAD_DIM);
    std::vector<float> V_transposed = this->transposeKV(V_fp32, KV_LEN, NUM_KV_HEADS, HEAD_DIM);

    // Quantize
    auto Q_q16 = quantizeToQ16<Q16_1Block_64>(Q_fp32, SEQ_LEN * NUM_HEADS, HEAD_DIM, KV_CACHE_SCALE, HEAD_DIM);
    auto K_q16 = quantizeToQ16<Q16_1Block_64>(K_transposed, NUM_KV_HEADS * KV_LEN, HEAD_DIM, KV_CACHE_SCALE, HEAD_DIM);
    auto V_q16 = quantizeToQ16<Q16_1Block_64>(V_transposed, NUM_KV_HEADS * KV_LEN, HEAD_DIM, KV_CACHE_SCALE, HEAD_DIM);

    std::vector<float> q_scales(NUM_HEADS, BLOCK_SCALE);
    std::vector<float> kv_scales(NUM_KV_HEADS, BLOCK_SCALE);
    std::vector<float> q16_context(total_out);

    Q16IntegerAttentionParams params;
    params.Q = Q_q16.data();
    params.K = K_q16.data();
    params.V = V_q16.data();
    params.q_head_scales = q_scales.data();
    params.kv_head_scales = kv_scales.data();
    params.seq_len_q = SEQ_LEN;
    params.kv_len = KV_LEN;
    params.num_heads = NUM_HEADS;
    params.num_kv_heads = NUM_KV_HEADS;
    params.head_dim = HEAD_DIM;
    params.d_model = NUM_HEADS * HEAD_DIM;
    params.block_size = Q16BlockSize::BLOCK_64;
    // Note: prefill handles causal mask internally
    params.snapshot_context = q16_context.data();

    bool success = q16_integer_attention_prefill(params);
    ASSERT_TRUE(success) << "GQA Prefill failed";

    float cos_sim = cosineSimilarity(ref_output.data(), q16_context.data(), total_out);
    float max_diff = maxAbsDiff(ref_output.data(), q16_context.data(), total_out);
    float rmse_val = rmse(ref_output.data(), q16_context.data(), total_out);

    std::cout << "[PARITY] GQA Prefill (14:2, inline ref, causal):" << std::endl;
    std::cout << "  Cosine similarity: " << std::fixed << std::setprecision(6) << cos_sim << std::endl;
    std::cout << "  Max absolute diff: " << std::scientific << std::setprecision(4) << max_diff << std::endl;
    std::cout << "  RMSE:             " << rmse_val << std::endl;

    // Prefill with GQA has more numerical challenges
    EXPECT_GT(cos_sim, 0.95f) << "GQA prefill should have >0.95 cosine similarity with inline ref";
}

/**
 * @brief GQA Decode test using real CPUAttentionKernelT<FP32> as reference
 */
TEST_F(GQAAttentionParityTest, Decode_GQA_7to1_RealFP32Reference)
{
    constexpr int SEQ_LEN_Q = 1;
    constexpr int KV_LEN = 32;

    const int total_q = SEQ_LEN_Q * NUM_HEADS * HEAD_DIM;
    const int total_kv = KV_LEN * NUM_KV_HEADS * HEAD_DIM;
    const int total_out = SEQ_LEN_Q * NUM_HEADS * HEAD_DIM;

    // Generate random data
    auto Q_fp32 = generateNormalData(total_q, 0.0f, 4.0f);
    auto K_fp32 = generateNormalData(total_kv, 0.0f, 4.0f);
    auto V_fp32 = generateNormalData(total_kv, 0.0f, 4.0f);

    // Real FP32 reference using CPUAttentionKernelT<FP32>
    std::vector<float> ref_output;
    computeRealFP32Reference(Q_fp32, K_fp32, V_fp32, ref_output, SEQ_LEN_Q, KV_LEN, false);

    // Transpose K/V to head-major for Q16
    std::vector<float> K_transposed = this->transposeKV(K_fp32, KV_LEN, NUM_KV_HEADS, HEAD_DIM);
    std::vector<float> V_transposed = this->transposeKV(V_fp32, KV_LEN, NUM_KV_HEADS, HEAD_DIM);

    // Quantize
    auto Q_q16 = quantizeToQ16<Q16_1Block_64>(Q_fp32, NUM_HEADS, HEAD_DIM, KV_CACHE_SCALE, HEAD_DIM);
    auto K_q16 = quantizeToQ16<Q16_1Block_64>(K_transposed, NUM_KV_HEADS * KV_LEN, HEAD_DIM, KV_CACHE_SCALE, HEAD_DIM);
    auto V_q16 = quantizeToQ16<Q16_1Block_64>(V_transposed, NUM_KV_HEADS * KV_LEN, HEAD_DIM, KV_CACHE_SCALE, HEAD_DIM);

    std::vector<float> q_scales(NUM_HEADS, BLOCK_SCALE);
    std::vector<float> kv_scales(NUM_KV_HEADS, BLOCK_SCALE);
    std::vector<float> q16_context(total_out);

    Q16IntegerAttentionParams params;
    params.Q = Q_q16.data();
    params.K = K_q16.data();
    params.V = V_q16.data();
    params.q_head_scales = q_scales.data();
    params.kv_head_scales = kv_scales.data();
    params.seq_len_q = SEQ_LEN_Q;
    params.kv_len = KV_LEN;
    params.num_heads = NUM_HEADS;
    params.num_kv_heads = NUM_KV_HEADS;
    params.head_dim = HEAD_DIM;
    params.d_model = NUM_HEADS * HEAD_DIM;
    params.block_size = Q16BlockSize::BLOCK_64;
    params.snapshot_context = q16_context.data();

    bool success = q16_integer_attention_decode(params);
    ASSERT_TRUE(success) << "GQA Decode failed";

    float cos_sim = cosineSimilarity(ref_output.data(), q16_context.data(), total_out);
    float max_diff = maxAbsDiff(ref_output.data(), q16_context.data(), total_out);
    float rmse_val = rmse(ref_output.data(), q16_context.data(), total_out);

    std::cout << "[PARITY] GQA Decode (14:2, CPUAttentionKernelT<FP32>):" << std::endl;
    std::cout << "  Cosine similarity: " << std::fixed << std::setprecision(6) << cos_sim << std::endl;
    std::cout << "  Max absolute diff: " << std::scientific << std::setprecision(4) << max_diff << std::endl;
    std::cout << "  RMSE:             " << rmse_val << std::endl;

    EXPECT_GT(cos_sim, 0.99f) << "GQA decode should have >0.99 cosine similarity with real FP32 ref";
}

/**
 * @brief GQA Prefill test using real CPUAttentionKernelT<FP32> as reference
 */
TEST_F(GQAAttentionParityTest, Prefill_GQA_7to1_RealFP32Reference)
{
    constexpr int SEQ_LEN = 9;
    constexpr int KV_LEN = SEQ_LEN;

    const int total_q = SEQ_LEN * NUM_HEADS * HEAD_DIM;
    const int total_kv = KV_LEN * NUM_KV_HEADS * HEAD_DIM;
    const int total_out = SEQ_LEN * NUM_HEADS * HEAD_DIM;

    auto Q_fp32 = generateNormalData(total_q, 0.0f, 4.0f);
    auto K_fp32 = generateNormalData(total_kv, 0.0f, 4.0f);
    auto V_fp32 = generateNormalData(total_kv, 0.0f, 4.0f);

    // Real FP32 reference using CPUAttentionKernelT<FP32>
    std::vector<float> ref_output;
    computeRealFP32Reference(Q_fp32, K_fp32, V_fp32, ref_output, SEQ_LEN, KV_LEN, true);

    // Transpose K/V to head-major
    std::vector<float> K_transposed = this->transposeKV(K_fp32, KV_LEN, NUM_KV_HEADS, HEAD_DIM);
    std::vector<float> V_transposed = this->transposeKV(V_fp32, KV_LEN, NUM_KV_HEADS, HEAD_DIM);

    // Quantize
    auto Q_q16 = quantizeToQ16<Q16_1Block_64>(Q_fp32, SEQ_LEN * NUM_HEADS, HEAD_DIM, KV_CACHE_SCALE, HEAD_DIM);
    auto K_q16 = quantizeToQ16<Q16_1Block_64>(K_transposed, NUM_KV_HEADS * KV_LEN, HEAD_DIM, KV_CACHE_SCALE, HEAD_DIM);
    auto V_q16 = quantizeToQ16<Q16_1Block_64>(V_transposed, NUM_KV_HEADS * KV_LEN, HEAD_DIM, KV_CACHE_SCALE, HEAD_DIM);

    std::vector<float> q_scales(NUM_HEADS, BLOCK_SCALE);
    std::vector<float> kv_scales(NUM_KV_HEADS, BLOCK_SCALE);
    std::vector<float> q16_context(total_out);

    Q16IntegerAttentionParams params;
    params.Q = Q_q16.data();
    params.K = K_q16.data();
    params.V = V_q16.data();
    params.q_head_scales = q_scales.data();
    params.kv_head_scales = kv_scales.data();
    params.seq_len_q = SEQ_LEN;
    params.kv_len = KV_LEN;
    params.num_heads = NUM_HEADS;
    params.num_kv_heads = NUM_KV_HEADS;
    params.head_dim = HEAD_DIM;
    params.d_model = NUM_HEADS * HEAD_DIM;
    params.block_size = Q16BlockSize::BLOCK_64;
    // Note: prefill handles causal mask internally
    params.snapshot_context = q16_context.data();

    bool success = q16_integer_attention_prefill(params);
    ASSERT_TRUE(success) << "GQA Prefill failed";

    float cos_sim = cosineSimilarity(ref_output.data(), q16_context.data(), total_out);
    float max_diff = maxAbsDiff(ref_output.data(), q16_context.data(), total_out);
    float rmse_val = rmse(ref_output.data(), q16_context.data(), total_out);

    std::cout << "[PARITY] GQA Prefill (14:2, CPUAttentionKernelT<FP32>, causal):" << std::endl;
    std::cout << "  Cosine similarity: " << std::fixed << std::setprecision(6) << cos_sim << std::endl;
    std::cout << "  Max absolute diff: " << std::scientific << std::setprecision(4) << max_diff << std::endl;
    std::cout << "  RMSE:             " << rmse_val << std::endl;

    EXPECT_GT(cos_sim, 0.95f) << "GQA prefill should have >0.95 cosine similarity with real FP32 ref";
}

// =============================================================================
// Per-Position K Scale Parity Tests (Phase 8 - HybridQ16 K Precision Fix)
// =============================================================================

/**
 * @brief Test fixture for per-position K scale parity tests.
 *
 * These tests verify that Q16 integer attention with per-position K scales
 * produces correct results compared to an FP32 reference implementation
 * that applies the same per-position scaling.
 */
class PerPositionKScaleParityTest : public Test__Q16IntegerAttentionParity
{
protected:
    static constexpr int HEAD_DIM = 64;
    static constexpr int NUM_HEADS = 14;   // Qwen2-0.5B config
    static constexpr int NUM_KV_HEADS = 2; // GQA 7:1

    /**
     * @brief Compute FP32 reference attention WITH per-position K scales.
     */
    void fp32ReferenceWithPerPositionKScales(
        const float *Q_fp32, // [num_heads, head_dim] - dequantized values
        const float *K_fp32, // [num_kv_heads, kv_len, head_dim] - dequantized values
        const float *V_fp32, // [num_kv_heads, kv_len, head_dim] - dequantized values
        float *output,       // [num_heads, head_dim]
        int kv_len,
        float q_scale,
        const float *k_position_scales, // [kv_len * num_kv_heads]
        float v_scale)
    {
        const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));

        for (int h = 0; h < NUM_HEADS; ++h)
        {
            int kv_h = h / (NUM_HEADS / NUM_KV_HEADS);
            const float *Q_head = Q_fp32 + h * HEAD_DIM;

            std::vector<float> scores(kv_len);
            float max_score = -std::numeric_limits<float>::infinity();

            for (int k_pos = 0; k_pos < kv_len; ++k_pos)
            {
                const float *K_row = K_fp32 + (kv_h * kv_len + k_pos) * HEAD_DIM;
                float dot = 0.0f;
                for (int d = 0; d < HEAD_DIM; ++d)
                {
                    dot += Q_head[d] * K_row[d];
                }
                scores[k_pos] = dot * inv_sqrt_d;
                max_score = std::max(max_score, scores[k_pos]);
            }

            float sum_exp = 0.0f;
            for (int k_pos = 0; k_pos < kv_len; ++k_pos)
            {
                scores[k_pos] = std::exp(scores[k_pos] - max_score);
                sum_exp += scores[k_pos];
            }
            for (int k_pos = 0; k_pos < kv_len; ++k_pos)
            {
                scores[k_pos] /= sum_exp;
            }

            float *out_head = output + h * HEAD_DIM;
            for (int d = 0; d < HEAD_DIM; ++d)
            {
                float val = 0.0f;
                for (int k_pos = 0; k_pos < kv_len; ++k_pos)
                {
                    const float *V_row = V_fp32 + (kv_h * kv_len + k_pos) * HEAD_DIM;
                    val += scores[k_pos] * V_row[d];
                }
                out_head[d] = val;
            }
        }
    }

    /**
     * @brief Quantize with per-position scales (simulating RoPE output).
     */
    template <typename BlockType>
    std::vector<BlockType> quantizeWithPerPositionScales(
        const std::vector<float> &fp32_data,
        int num_kv_heads,
        int kv_len,
        int head_dim,
        const std::vector<float> &position_scales)
    {
        constexpr int BLOCK_SIZE = BlockType::BLOCK_SIZE;
        int blocks_per_row = (head_dim + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int total_rows = num_kv_heads * kv_len;
        int total_blocks = total_rows * blocks_per_row;

        int16_t max_safe = get_max_safe_int16(head_dim);
        std::vector<BlockType> blocks(total_blocks);

        for (int kv_h = 0; kv_h < num_kv_heads; ++kv_h)
        {
            for (int k_pos = 0; k_pos < kv_len; ++k_pos)
            {
                int row = kv_h * kv_len + k_pos;
                float scale = position_scales[k_pos * num_kv_heads + kv_h];
                float quant_factor = 1.0f / scale;

                for (int b = 0; b < blocks_per_row; ++b)
                {
                    BlockType &block = blocks[row * blocks_per_row + b];
                    block.d = scale;
                    int32_t sum = 0;

                    int start_col = b * BLOCK_SIZE;
                    for (int i = 0; i < BLOCK_SIZE; ++i)
                    {
                        int col = start_col + i;
                        float val = (col < head_dim) ? fp32_data[row * head_dim + col] : 0.0f;
                        int32_t q = static_cast<int32_t>(std::round(val * quant_factor));
                        q = std::clamp(q, static_cast<int32_t>(-max_safe), static_cast<int32_t>(max_safe));
                        block.qs[i] = static_cast<int16_t>(q);
                        sum += block.qs[i];
                    }
                    block.sum_qs = sum;
                }
            }
        }
        return blocks;
    }
};

/**
 * @test Per-position K scales with uniform scales (baseline)
 */
TEST_F(PerPositionKScaleParityTest, Decode_UniformKScales_MatchesStandard)
{
    constexpr int KV_LEN = 9;
    constexpr float BLOCK_SCALE = KV_CACHE_SCALE / 32767.0f;

    const int total_q = NUM_HEADS * HEAD_DIM;
    const int total_kv = NUM_KV_HEADS * KV_LEN * HEAD_DIM;
    const int total_out = NUM_HEADS * HEAD_DIM;

    auto Q_fp32 = generateNormalData(total_q, 0.0f, 4.0f);
    auto K_fp32 = generateNormalData(total_kv, 0.0f, 4.0f);
    auto V_fp32 = generateNormalData(total_kv, 0.0f, 4.0f);

    std::vector<float> k_position_scales(KV_LEN * NUM_KV_HEADS, BLOCK_SCALE);

    auto Q_q16 = quantizeToQ16<Q16_1Block_64>(Q_fp32, NUM_HEADS, HEAD_DIM, KV_CACHE_SCALE, HEAD_DIM);
    auto V_q16 = quantizeToQ16<Q16_1Block_64>(V_fp32, NUM_KV_HEADS * KV_LEN, HEAD_DIM, KV_CACHE_SCALE, HEAD_DIM);
    auto K_q16 = quantizeWithPerPositionScales<Q16_1Block_64>(K_fp32, NUM_KV_HEADS, KV_LEN, HEAD_DIM, k_position_scales);

    auto Q_dequant = dequantizeFromQ16(Q_q16, NUM_HEADS, HEAD_DIM);
    auto K_dequant = dequantizeFromQ16(K_q16, NUM_KV_HEADS * KV_LEN, HEAD_DIM);
    auto V_dequant = dequantizeFromQ16(V_q16, NUM_KV_HEADS * KV_LEN, HEAD_DIM);

    std::vector<float> ref_output(total_out);
    fp32ReferenceWithPerPositionKScales(
        Q_dequant.data(), K_dequant.data(), V_dequant.data(),
        ref_output.data(), KV_LEN,
        BLOCK_SCALE, k_position_scales.data(), BLOCK_SCALE);

    std::vector<float> q_scales(NUM_HEADS, BLOCK_SCALE);
    std::vector<float> kv_scales(NUM_KV_HEADS, BLOCK_SCALE);
    std::vector<float> q16_context(total_out);

    Q16IntegerAttentionParams params;
    params.Q = Q_q16.data();
    params.K = K_q16.data();
    params.V = V_q16.data();
    params.q_head_scales = q_scales.data();
    params.kv_head_scales = kv_scales.data();
    params.k_position_scales = k_position_scales.data();
    params.seq_len_q = 1;
    params.kv_len = KV_LEN;
    params.num_heads = NUM_HEADS;
    params.num_kv_heads = NUM_KV_HEADS;
    params.head_dim = HEAD_DIM;
    params.d_model = NUM_HEADS * HEAD_DIM;
    params.block_size = Q16BlockSize::BLOCK_64;
    params.snapshot_context = q16_context.data();

    bool success = q16_integer_attention_decode(params);
    ASSERT_TRUE(success) << "Decode with uniform per-position K scales failed";

    float cos_sim = cosineSimilarity(ref_output.data(), q16_context.data(), total_out);
    float max_diff = maxAbsDiff(ref_output.data(), q16_context.data(), total_out);
    float rmse_val = rmse(ref_output.data(), q16_context.data(), total_out);

    std::cout << "\n[PARITY] Per-Position K Scales (UNIFORM) - Decode:" << std::endl;
    std::cout << "  Cosine similarity: " << std::fixed << std::setprecision(6) << cos_sim << std::endl;
    std::cout << "  Max absolute diff: " << std::scientific << std::setprecision(4) << max_diff << std::endl;
    std::cout << "  RMSE:             " << rmse_val << std::endl;

    EXPECT_GT(cos_sim, 0.99f) << "Uniform per-position K scales should match standard attention";
}

/**
 * @test Per-position K scales with varying scales (realistic HybridQ16 scenario)
 */
TEST_F(PerPositionKScaleParityTest, Decode_VaryingKScales_MatchesFP32Reference)
{
    constexpr int KV_LEN = 9;
    constexpr float Q_SCALE = 64.0f / 32767.0f;

    const int total_q = NUM_HEADS * HEAD_DIM;
    const int total_kv = NUM_KV_HEADS * KV_LEN * HEAD_DIM;
    const int total_out = NUM_HEADS * HEAD_DIM;

    auto Q_fp32 = generateNormalData(total_q, 0.0f, 0.3f);
    auto K_fp32 = generateNormalData(total_kv, 0.0f, 0.3f);
    auto V_fp32 = generateNormalData(total_kv, 0.0f, 0.3f);

    // Varying K scales - realistic range (0.001 to 0.004)
    std::vector<float> k_position_scales(KV_LEN * NUM_KV_HEADS);
    for (int pos = 0; pos < KV_LEN; ++pos)
    {
        for (int kv_h = 0; kv_h < NUM_KV_HEADS; ++kv_h)
        {
            float base_scale = Q_SCALE * (0.5f + 0.5f * static_cast<float>(pos) / (KV_LEN - 1));
            k_position_scales[pos * NUM_KV_HEADS + kv_h] = base_scale;
        }
    }

    auto Q_q16 = quantizeToQ16<Q16_1Block_64>(Q_fp32, NUM_HEADS, HEAD_DIM, 64.0f, HEAD_DIM);
    auto K_q16 = quantizeWithPerPositionScales<Q16_1Block_64>(K_fp32, NUM_KV_HEADS, KV_LEN, HEAD_DIM, k_position_scales);
    auto V_q16 = quantizeToQ16<Q16_1Block_64>(V_fp32, NUM_KV_HEADS * KV_LEN, HEAD_DIM, 64.0f, HEAD_DIM);

    auto Q_dequant = dequantizeFromQ16(Q_q16, NUM_HEADS, HEAD_DIM);
    auto K_dequant = dequantizeFromQ16(K_q16, NUM_KV_HEADS * KV_LEN, HEAD_DIM);
    auto V_dequant = dequantizeFromQ16(V_q16, NUM_KV_HEADS * KV_LEN, HEAD_DIM);

    std::vector<float> ref_output(total_out);
    fp32ReferenceWithPerPositionKScales(
        Q_dequant.data(), K_dequant.data(), V_dequant.data(),
        ref_output.data(), KV_LEN,
        Q_SCALE, k_position_scales.data(), Q_SCALE);

    std::vector<float> q_scales(NUM_HEADS, Q_SCALE);
    std::vector<float> kv_scales(NUM_KV_HEADS, Q_SCALE);
    std::vector<float> q16_context(total_out);

    Q16IntegerAttentionParams params;
    params.Q = Q_q16.data();
    params.K = K_q16.data();
    params.V = V_q16.data();
    params.q_head_scales = q_scales.data();
    params.kv_head_scales = kv_scales.data();
    params.k_position_scales = k_position_scales.data();
    params.seq_len_q = 1;
    params.kv_len = KV_LEN;
    params.num_heads = NUM_HEADS;
    params.num_kv_heads = NUM_KV_HEADS;
    params.head_dim = HEAD_DIM;
    params.d_model = NUM_HEADS * HEAD_DIM;
    params.block_size = Q16BlockSize::BLOCK_64;
    params.snapshot_context = q16_context.data();

    bool success = q16_integer_attention_decode(params);
    ASSERT_TRUE(success) << "Decode with varying per-position K scales failed";

    float cos_sim = cosineSimilarity(ref_output.data(), q16_context.data(), total_out);
    float max_diff = maxAbsDiff(ref_output.data(), q16_context.data(), total_out);
    float rmse_val = rmse(ref_output.data(), q16_context.data(), total_out);

    std::cout << "\n[PARITY] Per-Position K Scales (VARYING) - Decode:" << std::endl;
    std::cout << "  K scale range: " << k_position_scales.front() << " to " << k_position_scales.back() << std::endl;
    std::cout << "  Cosine similarity: " << std::fixed << std::setprecision(6) << cos_sim << std::endl;
    std::cout << "  Max absolute diff: " << std::scientific << std::setprecision(4) << max_diff << std::endl;
    std::cout << "  RMSE:             " << rmse_val << std::endl;

    EXPECT_GT(cos_sim, 0.99f) << "Varying per-position K scales should match FP32 reference";
}

/**
 * @test Per-position K scales with realistic pipeline scale variation (540x range)
 *
 * This test uses the actual scale range observed in the HybridQ16 pipeline:
 * - Layer 4 has K scales as low as 2.41e-05
 * - Layer 0/8 have K scales up to 0.013
 * - This is a 540× variation, much larger than the 100× test
 */
TEST_F(PerPositionKScaleParityTest, Decode_RealisticPipelineScaleRange_MatchesFP32Reference)
{
    constexpr int KV_LEN = 9;
    constexpr float Q_SCALE = 64.0f / 32767.0f; // ~0.00195

    const int total_q = NUM_HEADS * HEAD_DIM;
    const int total_kv = NUM_KV_HEADS * KV_LEN * HEAD_DIM;
    const int total_out = NUM_HEADS * HEAD_DIM;

    auto Q_fp32 = generateNormalData(total_q, 0.0f, 0.3f);
    auto K_fp32 = generateNormalData(total_kv, 0.0f, 0.3f);
    auto V_fp32 = generateNormalData(total_kv, 0.0f, 0.3f);

    // Realistic K scale variation: 2.4e-05 to 0.013 (540x range)
    // These are the actual scales observed in the HybridQ16 pipeline
    std::vector<float> k_position_scales(KV_LEN * NUM_KV_HEADS);
    // Use scales that mimic the real pipeline's layer-by-layer variation
    const float real_scales[] = {0.00796, 0.000764, 0.00239, 0.000179, 2.41e-05, 2.47e-05, 0.000417, 3.08e-05, 0.0130};
    for (int pos = 0; pos < KV_LEN; ++pos)
    {
        for (int kv_h = 0; kv_h < NUM_KV_HEADS; ++kv_h)
        {
            k_position_scales[pos * NUM_KV_HEADS + kv_h] = real_scales[pos];
        }
    }

    auto Q_q16 = quantizeToQ16<Q16_1Block_64>(Q_fp32, NUM_HEADS, HEAD_DIM, 64.0f, HEAD_DIM);
    auto K_q16 = quantizeWithPerPositionScales<Q16_1Block_64>(K_fp32, NUM_KV_HEADS, KV_LEN, HEAD_DIM, k_position_scales);
    auto V_q16 = quantizeToQ16<Q16_1Block_64>(V_fp32, NUM_KV_HEADS * KV_LEN, HEAD_DIM, 64.0f, HEAD_DIM);

    auto Q_dequant = dequantizeFromQ16(Q_q16, NUM_HEADS, HEAD_DIM);
    auto K_dequant = dequantizeFromQ16(K_q16, NUM_KV_HEADS * KV_LEN, HEAD_DIM);
    auto V_dequant = dequantizeFromQ16(V_q16, NUM_KV_HEADS * KV_LEN, HEAD_DIM);

    std::vector<float> ref_output(total_out);
    fp32ReferenceWithPerPositionKScales(
        Q_dequant.data(), K_dequant.data(), V_dequant.data(),
        ref_output.data(), KV_LEN,
        Q_SCALE, k_position_scales.data(), Q_SCALE);

    std::vector<float> q_scales(NUM_HEADS, Q_SCALE);
    std::vector<float> kv_scales(NUM_KV_HEADS, Q_SCALE);
    std::vector<float> q16_context(total_out);

    Q16IntegerAttentionParams params;
    params.Q = Q_q16.data();
    params.K = K_q16.data();
    params.V = V_q16.data();
    params.q_head_scales = q_scales.data();
    params.kv_head_scales = kv_scales.data();
    params.k_position_scales = k_position_scales.data();
    params.seq_len_q = 1;
    params.kv_len = KV_LEN;
    params.num_heads = NUM_HEADS;
    params.num_kv_heads = NUM_KV_HEADS;
    params.head_dim = HEAD_DIM;
    params.d_model = NUM_HEADS * HEAD_DIM;
    params.block_size = Q16BlockSize::BLOCK_64;
    params.snapshot_context = q16_context.data();

    bool success = q16_integer_attention_decode(params);
    ASSERT_TRUE(success) << "Decode with realistic pipeline K scale variation failed";

    float cos_sim = cosineSimilarity(ref_output.data(), q16_context.data(), total_out);
    float max_diff = maxAbsDiff(ref_output.data(), q16_context.data(), total_out);
    float rmse_val = rmse(ref_output.data(), q16_context.data(), total_out);

    std::cout << "\n[PARITY] Per-Position K Scales (REALISTIC PIPELINE 540x) - Decode:" << std::endl;
    std::cout << "  K scale range: " << *std::min_element(k_position_scales.begin(), k_position_scales.end())
              << " to " << *std::max_element(k_position_scales.begin(), k_position_scales.end())
              << " (" << (*std::max_element(k_position_scales.begin(), k_position_scales.end()) / *std::min_element(k_position_scales.begin(), k_position_scales.end())) << "x range)" << std::endl;
    std::cout << "  Cosine similarity: " << std::fixed << std::setprecision(6) << cos_sim << std::endl;
    std::cout << "  Max absolute diff: " << std::scientific << std::setprecision(4) << max_diff << std::endl;
    std::cout << "  RMSE:             " << rmse_val << std::endl;

    EXPECT_GT(cos_sim, 0.95f) << "Realistic pipeline K scale variation should match FP32 reasonably";
}

/**
 * @test Per-position K scales with large scale variation (stress test)
 */
TEST_F(PerPositionKScaleParityTest, Decode_LargeScaleVariation_MatchesFP32Reference)
{
    constexpr int KV_LEN = 9;
    constexpr float Q_SCALE = 64.0f / 32767.0f;

    const int total_q = NUM_HEADS * HEAD_DIM;
    const int total_kv = NUM_KV_HEADS * KV_LEN * HEAD_DIM;
    const int total_out = NUM_HEADS * HEAD_DIM;

    auto Q_fp32 = generateNormalData(total_q, 0.0f, 0.3f);
    auto K_fp32 = generateNormalData(total_kv, 0.0f, 0.3f);
    auto V_fp32 = generateNormalData(total_kv, 0.0f, 0.3f);

    // Large K scale variation: 0.0001 to 0.01 (100x range)
    std::vector<float> k_position_scales(KV_LEN * NUM_KV_HEADS);
    for (int pos = 0; pos < KV_LEN; ++pos)
    {
        for (int kv_h = 0; kv_h < NUM_KV_HEADS; ++kv_h)
        {
            float t = static_cast<float>(pos) / (KV_LEN - 1);
            float scale = 0.0001f * std::pow(100.0f, t);
            k_position_scales[pos * NUM_KV_HEADS + kv_h] = scale;
        }
    }

    auto Q_q16 = quantizeToQ16<Q16_1Block_64>(Q_fp32, NUM_HEADS, HEAD_DIM, 64.0f, HEAD_DIM);
    auto K_q16 = quantizeWithPerPositionScales<Q16_1Block_64>(K_fp32, NUM_KV_HEADS, KV_LEN, HEAD_DIM, k_position_scales);
    auto V_q16 = quantizeToQ16<Q16_1Block_64>(V_fp32, NUM_KV_HEADS * KV_LEN, HEAD_DIM, 64.0f, HEAD_DIM);

    auto Q_dequant = dequantizeFromQ16(Q_q16, NUM_HEADS, HEAD_DIM);
    auto K_dequant = dequantizeFromQ16(K_q16, NUM_KV_HEADS * KV_LEN, HEAD_DIM);
    auto V_dequant = dequantizeFromQ16(V_q16, NUM_KV_HEADS * KV_LEN, HEAD_DIM);

    std::vector<float> ref_output(total_out);
    fp32ReferenceWithPerPositionKScales(
        Q_dequant.data(), K_dequant.data(), V_dequant.data(),
        ref_output.data(), KV_LEN,
        Q_SCALE, k_position_scales.data(), Q_SCALE);

    std::vector<float> q_scales(NUM_HEADS, Q_SCALE);
    std::vector<float> kv_scales(NUM_KV_HEADS, Q_SCALE);
    std::vector<float> q16_context(total_out);

    Q16IntegerAttentionParams params;
    params.Q = Q_q16.data();
    params.K = K_q16.data();
    params.V = V_q16.data();
    params.q_head_scales = q_scales.data();
    params.kv_head_scales = kv_scales.data();
    params.k_position_scales = k_position_scales.data();
    params.seq_len_q = 1;
    params.kv_len = KV_LEN;
    params.num_heads = NUM_HEADS;
    params.num_kv_heads = NUM_KV_HEADS;
    params.head_dim = HEAD_DIM;
    params.d_model = NUM_HEADS * HEAD_DIM;
    params.block_size = Q16BlockSize::BLOCK_64;
    params.snapshot_context = q16_context.data();

    bool success = q16_integer_attention_decode(params);
    ASSERT_TRUE(success) << "Decode with large K scale variation failed";

    float cos_sim = cosineSimilarity(ref_output.data(), q16_context.data(), total_out);
    float max_diff = maxAbsDiff(ref_output.data(), q16_context.data(), total_out);
    float rmse_val = rmse(ref_output.data(), q16_context.data(), total_out);

    std::cout << "\n[PARITY] Per-Position K Scales (LARGE VARIATION 100x) - Decode:" << std::endl;
    std::cout << "  K scale range: " << k_position_scales.front() << " to " << k_position_scales.back() << std::endl;
    std::cout << "  Cosine similarity: " << std::fixed << std::setprecision(6) << cos_sim << std::endl;
    std::cout << "  Max absolute diff: " << std::scientific << std::setprecision(4) << max_diff << std::endl;
    std::cout << "  RMSE:             " << rmse_val << std::endl;

    EXPECT_GT(cos_sim, 0.95f) << "Large K scale variation should still match FP32 reasonably";
}

/**
 * @test Debug test to trace the full computation
 */
TEST_F(PerPositionKScaleParityTest, Debug_TraceComputation)
{
    constexpr int KV_LEN = 3;
    constexpr float Q_SCALE = 64.0f / 32767.0f; // ~0.00195

    // Small test case for debugging
    const int total_q = NUM_HEADS * HEAD_DIM;
    const int total_kv = NUM_KV_HEADS * KV_LEN * HEAD_DIM;
    const int total_out = NUM_HEADS * HEAD_DIM;

    // Generate deterministic data
    std::mt19937 gen(12345);
    std::normal_distribution<float> dist(0.0f, 0.3f);

    std::vector<float> Q_fp32(total_q);
    std::vector<float> K_fp32(total_kv);
    std::vector<float> V_fp32(total_kv);

    for (auto &v : Q_fp32)
        v = dist(gen);
    for (auto &v : K_fp32)
        v = dist(gen);
    for (auto &v : V_fp32)
        v = dist(gen);

    // Uniform K scales (should match standard path)
    std::vector<float> k_position_scales(KV_LEN * NUM_KV_HEADS, Q_SCALE);

    // Quantize
    auto Q_q16 = quantizeToQ16<Q16_1Block_64>(Q_fp32, NUM_HEADS, HEAD_DIM, 64.0f, HEAD_DIM);
    auto K_q16 = quantizeWithPerPositionScales<Q16_1Block_64>(K_fp32, NUM_KV_HEADS, KV_LEN, HEAD_DIM, k_position_scales);
    auto V_q16 = quantizeToQ16<Q16_1Block_64>(V_fp32, NUM_KV_HEADS * KV_LEN, HEAD_DIM, 64.0f, HEAD_DIM);

    // Dequantize for reference
    auto Q_dequant = dequantizeFromQ16(Q_q16, NUM_HEADS, HEAD_DIM);
    auto K_dequant = dequantizeFromQ16(K_q16, NUM_KV_HEADS * KV_LEN, HEAD_DIM);
    auto V_dequant = dequantizeFromQ16(V_q16, NUM_KV_HEADS * KV_LEN, HEAD_DIM);

    // Debug: Print first head's data
    std::cout << "\n=== DEBUG: Trace Computation ===" << std::endl;
    std::cout << "Q_SCALE = " << Q_SCALE << std::endl;
    std::cout << "Q_dequant[0][0:4]: ";
    for (int i = 0; i < 4; ++i)
        std::cout << Q_dequant[i] << " ";
    std::cout << std::endl;

    std::cout << "K_dequant[pos=0][0:4]: ";
    for (int i = 0; i < 4; ++i)
        std::cout << K_dequant[i] << " ";
    std::cout << std::endl;

    // Manual FP32 attention for head 0
    const float inv_sqrt_d = 1.0f / std::sqrt(64.0f);
    std::cout << "inv_sqrt_d = " << inv_sqrt_d << std::endl;

    std::vector<float> scores(KV_LEN);
    for (int k = 0; k < KV_LEN; ++k)
    {
        float dot = 0.0f;
        for (int d = 0; d < HEAD_DIM; ++d)
        {
            dot += Q_dequant[d] * K_dequant[k * HEAD_DIM + d];
        }
        scores[k] = dot * inv_sqrt_d;
    }
    std::cout << "FP32 scores (head 0): ";
    for (float s : scores)
        std::cout << s << " ";
    std::cout << std::endl;

    // Softmax
    float max_score = *std::max_element(scores.begin(), scores.end());
    std::vector<float> weights(KV_LEN);
    float sum_exp = 0.0f;
    for (int k = 0; k < KV_LEN; ++k)
    {
        weights[k] = std::exp(scores[k] - max_score);
        sum_exp += weights[k];
    }
    for (float &w : weights)
        w /= sum_exp;
    std::cout << "FP32 weights: ";
    for (float w : weights)
        std::cout << w << " ";
    std::cout << std::endl;

    // FP32 reference output
    std::vector<float> ref_output(total_out);
    fp32ReferenceWithPerPositionKScales(
        Q_dequant.data(), K_dequant.data(), V_dequant.data(),
        ref_output.data(), KV_LEN,
        Q_SCALE, k_position_scales.data(), Q_SCALE);

    std::cout << "FP32 ref output[0][0:4]: ";
    for (int i = 0; i < 4; ++i)
        std::cout << ref_output[i] << " ";
    std::cout << std::endl;

    // Q16 integer attention
    std::vector<float> q_scales(NUM_HEADS, Q_SCALE);
    std::vector<float> kv_scales(NUM_KV_HEADS, Q_SCALE);
    std::vector<float> q16_context(total_out);

    Q16IntegerAttentionParams params;
    params.Q = Q_q16.data();
    params.K = K_q16.data();
    params.V = V_q16.data();
    params.q_head_scales = q_scales.data();
    params.kv_head_scales = kv_scales.data();
    params.k_position_scales = k_position_scales.data();
    params.seq_len_q = 1;
    params.kv_len = KV_LEN;
    params.num_heads = NUM_HEADS;
    params.num_kv_heads = NUM_KV_HEADS;
    params.head_dim = HEAD_DIM;
    params.d_model = NUM_HEADS * HEAD_DIM;
    params.block_size = Q16BlockSize::BLOCK_64;
    params.snapshot_context = q16_context.data();

    bool success = q16_integer_attention_decode(params);
    ASSERT_TRUE(success);

    std::cout << "Q16 output[0][0:4]: ";
    for (int i = 0; i < 4; ++i)
        std::cout << q16_context[i] << " ";
    std::cout << std::endl;

    float cos_sim = cosineSimilarity(ref_output.data(), q16_context.data(), total_out);
    std::cout << "Cosine similarity: " << cos_sim << std::endl;

    // Also run WITHOUT k_position_scales to compare
    params.k_position_scales = nullptr;
    std::vector<float> q16_context_standard(total_out);
    params.snapshot_context = q16_context_standard.data();

    success = q16_integer_attention_decode(params);
    ASSERT_TRUE(success);

    std::cout << "Q16 standard output[0][0:4]: ";
    for (int i = 0; i < 4; ++i)
        std::cout << q16_context_standard[i] << " ";
    std::cout << std::endl;

    float cos_sim_standard = cosineSimilarity(ref_output.data(), q16_context_standard.data(), total_out);
    std::cout << "Standard path cosine similarity: " << cos_sim_standard << std::endl;

    // Compare per-position vs standard outputs
    float cos_sim_paths = cosineSimilarity(q16_context.data(), q16_context_standard.data(), total_out);
    std::cout << "Per-position vs Standard cosine: " << cos_sim_paths << std::endl;

    EXPECT_GT(cos_sim_standard, 0.99f) << "Standard path should work";
}
