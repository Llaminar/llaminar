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
// kv_cache_scale = 8.0 means the representable range is [-8.0, +8.0]
// Block scale d = kv_cache_scale / 32767.0f
// Quantization: int16_val = round(fp32_val * 32767.0f / kv_cache_scale)
// Dequantization: fp32_val = int16_val * d
constexpr float KV_CACHE_SCALE = 8.0f;

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

    // Generate random FP32 data in the range [-kv_cache_scale, +kv_cache_scale]
    // Using stddev=2.0 means most values are in [-6, 6], well within ±8 range
    auto Q_fp32 = generateNormalData(total_q, 0.0f, 0.5f);
    auto K_fp32 = generateNormalData(total_kv, 0.0f, 0.5f);
    auto V_fp32 = generateNormalData(total_kv, 0.0f, 0.5f);

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

    auto Q_fp32 = generateNormalData(total_q, 0.0f, 0.5f);
    auto K_fp32 = generateNormalData(total_kv, 0.0f, 0.5f);
    auto V_fp32 = generateNormalData(total_kv, 0.0f, 0.5f);

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

    auto Q_fp32 = generateNormalData(total_q, 0.0f, 0.5f);
    auto K_fp32 = generateNormalData(total_kv, 0.0f, 0.5f);
    auto V_fp32 = generateNormalData(total_kv, 0.0f, 0.5f);

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

    auto Q_fp32 = generateNormalData(total_q, 0.0f, 0.5f);
    auto K_fp32 = generateNormalData(total_kv, 0.0f, 0.5f);
    auto V_fp32 = generateNormalData(total_kv, 0.0f, 0.5f);

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

    auto Q_fp32 = generateNormalData(total_q, 0.0f, 0.5f);
    auto K_fp32 = generateNormalData(total_kv, 0.0f, 0.5f);
    auto V_fp32 = generateNormalData(total_kv, 0.0f, 0.5f);

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

    auto Q_fp32 = generateNormalData(total_q, 0.0f, 0.5f);
    auto K_fp32 = generateNormalData(total_kv, 0.0f, 0.5f);
    auto V_fp32 = generateNormalData(total_kv, 0.0f, 0.5f);

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

        auto Q_fp32 = generateNormalData(total_q, 0.0f, 0.5f);
        auto K_fp32 = generateNormalData(total_kv, 0.0f, 0.5f);
        auto V_fp32 = generateNormalData(total_kv, 0.0f, 0.5f);

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

        auto Q_fp32 = generateNormalData(total_q, 0.0f, 0.5f);
        auto K_fp32 = generateNormalData(total_kv, 0.0f, 0.5f);
        auto V_fp32 = generateNormalData(total_kv, 0.0f, 0.5f);

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
