/**
 * @file attention_primitives.cpp
 * @brief Refactored attention primitives with clean scalar + OpenMP implementation
 * @author David Sanftenberg
 *
 * This refactored version prioritizes:
 * - Maintainability and readability
 * - Correct GQA support (separate q_heads/k_heads)
 * - Proper freq_base handling matching HuggingFace transformers
 * - Clean separation of concerns
 * - OpenMP parallelization without complexity
 */

#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>
#include <cstdio>
#include <cstring>
#include <array>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "utils/debug_env.h"
#include "kernels/common/attention_primitives.h"
#include "kernels/common/softmax_core.h"
#include "logger.h"

namespace llaminar::attn
{
    // ============================================================================
    // HELPER FUNCTIONS - RoPE Components
    // ============================================================================

    /**
     * @brief Compute inverse frequencies for RoPE
     *
     * Following HuggingFace transformers implementation:
     * inv_freq[i] = 1.0 / (freq_base ^ (2*i / head_dim))
     *
     * @param head_dim Dimension of each attention head
     * @param freq_base Base frequency (typically 10000.0 or model-specific)
     * @return Vector of inverse frequencies, length = head_dim/2
     */
    static std::vector<float> compute_inv_freq(int head_dim, float freq_base)
    {
        const int half_dim = head_dim / 2;
        std::vector<float> inv_freq(half_dim);

        for (int i = 0; i < half_dim; ++i)
        {
            // inv_freq[i] = 1.0 / (freq_base ^ (2*i / head_dim))
            inv_freq[i] = 1.0f / std::pow(freq_base, (2.0f * i) / head_dim);
        }

        return inv_freq;
    }

    /**
     * @brief Apply RoPE rotation to a single head
     *
     * Uses the "rotate_half" pattern from HuggingFace:
     * - Split head_dim into two halves [x0, x1, ..., x_{d/2-1}] and [x_{d/2}, ..., x_{d-1}]
     * - For each pair (x[i], x[i + head_dim/2]):
     *   - new_x[i] = x[i] * cos(angle[i]) - x[i + head_dim/2] * sin(angle[i])
     *   - new_x[i + head_dim/2] = x[i] * sin(angle[i]) + x[i + head_dim/2] * cos(angle[i])
     *
     * @param head_ptr Pointer to head data (length head_dim)
     * @param position Token position in sequence
     * @param inv_freq Inverse frequencies (length head_dim/2)
     * @param head_dim Dimension of the head
     */
    static void apply_rope_to_head(float *head_ptr, int position,
                                   const std::vector<float> &inv_freq,
                                   int head_dim)
    {
        const int half_dim = head_dim / 2;

        for (int i = 0; i < half_dim; ++i)
        {
            // Compute angle for this frequency and position
            const float angle = position * inv_freq[i];
            const float cos_angle = std::cos(angle);
            const float sin_angle = std::sin(angle);

            // Indices for the two elements to rotate
            const int idx_first = i;
            const int idx_second = i + half_dim;

            // Read original values
            const float x_first = head_ptr[idx_first];
            const float x_second = head_ptr[idx_second];

            // Apply rotation (rotate_half pattern)
            head_ptr[idx_first] = x_first * cos_angle - x_second * sin_angle;
            head_ptr[idx_second] = x_first * sin_angle + x_second * cos_angle;
        }
    }

    /**
     * @brief Apply RoPE to Q or K tensor
     *
     * Layout: [seq_len, num_heads, head_dim] in row-major order
     *
     * @param tensor Pointer to Q or K tensor (modified in-place)
     * @param seq_len Sequence length
     * @param num_heads Number of heads in this tensor
     * @param head_dim Dimension per head
     * @param n_past Number of tokens already processed (for position calculation)
     * @param inv_freq Precomputed inverse frequencies
     */
    static void apply_rope_to_tensor(float *tensor, int seq_len, int num_heads,
                                     int head_dim, int n_past,
                                     const std::vector<float> &inv_freq)
    {
        const auto &env = llaminar::debugEnv().attention;

// Parallelize over (token, head) pairs
// Each iteration is independent, making this perfectly parallelizable
#pragma omp parallel for collapse(2) if (!env.prim_force_scalar)
        for (int t = 0; t < seq_len; ++t)
        {
            for (int h = 0; h < num_heads; ++h)
            {
                // Calculate absolute position
                const int position = n_past + t;

                // Pointer to this specific head: tensor[t, h, :]
                float *head_ptr = tensor + (size_t)t * num_heads * head_dim + (size_t)h * head_dim;

                // Apply rotation
                apply_rope_to_head(head_ptr, position, inv_freq, head_dim);
            }
        }
    }

    // ============================================================================
    // PUBLIC API - RoPE
    // ============================================================================

    /**
     * @brief Apply Rotary Position Embedding (RoPE) to Q and K tensors
     *
     * This implementation:
     * - Supports both MHA (q_heads == k_heads) and GQA (q_heads != k_heads)
     * - Uses the rotate_half pattern from HuggingFace transformers
     * - Respects model-specific freq_base parameter
     * - Parallelizes with OpenMP over (token, head) pairs
     *
     * Layout: [seq_len, num_heads, head_dim] in row-major order
     *
     * @param q Query tensor (modified in-place)
     * @param k Key tensor (modified in-place)
     * @param seq_len Sequence length
     * @param head_dim Dimension per head
     * @param q_heads Number of query heads
     * @param k_heads Number of key heads (may differ for GQA)
     * @param n_past Number of tokens already processed
     * @param freq_base Base frequency for RoPE (model-specific, e.g., 10000.0 or 1000000.0)
     */
    void apply_rope(float *q, float *k, int seq_len, int head_dim,
                    int q_heads, int k_heads, int n_past, float freq_base)
    {
        const auto &env = llaminar::debugEnv().attention;

        // Validation
        if (head_dim % 2 != 0)
        {
            LOG_ERROR("[RoPE] head_dim must be even, got " << head_dim);
            return;
        }

        if (seq_len <= 0 || q_heads <= 0 || k_heads <= 0)
        {
            LOG_WARN("[RoPE] Invalid dimensions: seq_len=" << seq_len
                                                           << " q_heads=" << q_heads << " k_heads=" << k_heads);
            return;
        }

        // Diagnostic trace (limited to first few calls)
        static int call_count = 0;
        if (call_count < 3)
        {
            LOG_INFO("[RoPE] call=" << call_count
                                    << " seq_len=" << seq_len
                                    << " head_dim=" << head_dim
                                    << " q_heads=" << q_heads
                                    << " k_heads=" << k_heads
                                    << " n_past=" << n_past
                                    << " freq_base=" << freq_base);
            call_count++;
        }

        // Compute inverse frequencies (same for Q and K)
        const std::vector<float> inv_freq = compute_inv_freq(head_dim, freq_base);

        // Apply RoPE to Q tensor
        apply_rope_to_tensor(q, seq_len, q_heads, head_dim, n_past, inv_freq);

        // Apply RoPE to K tensor (may have different number of heads for GQA)
        apply_rope_to_tensor(k, seq_len, k_heads, head_dim, n_past, inv_freq);

        // Optional diagnostics
        if (env.internal_diff && llaminar::debugEnv().pipeline.layer_token_diff && seq_len > 0)
        {
            // Sample last token, first head
            const size_t q_offset = (size_t)(seq_len - 1) * q_heads * head_dim;
            const size_t k_offset = (size_t)(seq_len - 1) * k_heads * head_dim;

            LOG_DEBUG("[RoPE] After rotation, last token first head Q[0:4]="
                      << q[q_offset] << "," << q[q_offset + 1] << ","
                      << q[q_offset + 2] << "," << q[q_offset + 3]);
            LOG_DEBUG("[RoPE] After rotation, last token first head K[0:4]="
                      << k[k_offset] << "," << k[k_offset + 1] << ","
                      << k[k_offset + 2] << "," << k[k_offset + 3]);
        }
    }

    // ============================================================================
    // HELPER FUNCTIONS - Attention Scores
    // ============================================================================

    /**
     * @brief Get pointer to a specific row in the scores matrix
     *
     * Scores layout: [heads, seq_len, seq_len]
     */
    static inline float *head_row(float *scores, int h, int i, int seq_len)
    {
        return scores + (size_t)h * seq_len * seq_len + (size_t)i * seq_len;
    }

    static inline const float *head_row(const float *scores, int h, int i, int seq_len)
    {
        return scores + (size_t)h * seq_len * seq_len + (size_t)i * seq_len;
    }

    // ============================================================================
    // PUBLIC API - QK Scores
    // ============================================================================

    /**
     * @brief Compute Q @ K^T scores and optionally apply softmax
     *
     * Computes attention scores = (Q @ K^T) * scale, where scale = 1/sqrt(head_dim)
     * Optionally applies causal masking and softmax.
     *
     * @param q Query tensor [seq_len, heads, head_dim]
     * @param k Key tensor [seq_len, heads, head_dim]
     * @param scores Output scores [heads, seq_len, seq_len]
     * @param seq_len Sequence length
     * @param head_dim Dimension per head
     * @param heads Number of heads
     * @param causal Whether to apply causal masking
     * @param apply_softmax Whether to apply softmax after computing scores
     */
    void compute_qk_scores(const float *q, const float *k, float *scores,
                           int seq_len, int head_dim, int heads,
                           bool causal, bool apply_softmax)
    {
        const auto &env = llaminar::debugEnv().attention;
        const float scale = 1.0f / std::sqrt((float)head_dim);

// Parallelize over (head, query_position) pairs
#pragma omp parallel for collapse(2) if (!env.prim_force_scalar)
        for (int h = 0; h < heads; ++h)
        {
            for (int i = 0; i < seq_len; ++i)
            {
                float *score_row = head_row(scores, h, i, seq_len);
                const float *qi = q + (size_t)i * heads * head_dim + (size_t)h * head_dim;

                // Compute dot product with each key position
                for (int j = 0; j < seq_len; ++j)
                {
                    // Apply causal mask: can't attend to future tokens
                    if (causal && j > i)
                    {
                        score_row[j] = -std::numeric_limits<float>::infinity();
                        continue;
                    }

                    // Compute Q[i] @ K[j]^T
                    const float *kj = k + (size_t)j * heads * head_dim + (size_t)h * head_dim;
                    float dot = 0.0f;

                    for (int d = 0; d < head_dim; ++d)
                    {
                        dot += qi[d] * kj[d];
                    }

                    score_row[j] = dot * scale;
                }
            }
        }

        // Apply softmax if requested
        if (apply_softmax)
        {
            for (int h = 0; h < heads; ++h)
            {
                llaminar::kernels::SoftmaxRowArgs args;
                args.scores = scores + (size_t)h * seq_len * seq_len;
                args.rows = seq_len;
                args.cols = seq_len;
                args.causal = causal;
                args.scale = 1.0f; // Already applied scale above

                llaminar::kernels::softmax_row_major(args);
            }
        }
    }

    // ============================================================================
    // PUBLIC API - Apply Attention Scores to Values
    // ============================================================================

    /**
     * @brief Apply attention scores to values: out = scores @ V
     *
     * @param scores Attention scores [heads, seq_len, seq_len]
     * @param v Value tensor [seq_len, heads, head_dim]
     * @param out Output tensor [seq_len, heads, head_dim]
     * @param seq_len Sequence length
     * @param head_dim Dimension per head
     * @param heads Number of heads
     */
    void apply_scores_to_v(const float *scores, const float *v, float *out,
                           int seq_len, int head_dim, int heads)
    {
        const auto &env = llaminar::debugEnv().attention;

        // Initialize output to zero
        std::fill(out, out + (size_t)seq_len * heads * head_dim, 0.0f);

// Parallelize over (head, output_position) pairs
#pragma omp parallel for collapse(2) if (!env.prim_force_scalar)
        for (int h = 0; h < heads; ++h)
        {
            for (int i = 0; i < seq_len; ++i)
            {
                float *out_ptr = out + (size_t)i * heads * head_dim + (size_t)h * head_dim;
                const float *score_row = head_row(scores, h, i, seq_len);

                // Weighted sum: out[i] = sum_j scores[i,j] * V[j]
                for (int j = 0; j < seq_len; ++j)
                {
                    const float weight = score_row[j];
                    const float *vj = v + (size_t)j * heads * head_dim + (size_t)h * head_dim;

                    for (int d = 0; d < head_dim; ++d)
                    {
                        out_ptr[d] += weight * vj[d];
                    }
                }
            }
        }
    }

    // ============================================================================
    // PUBLIC API - Fused Attention
    // ============================================================================

    /**
     * @brief Fused attention computation (for large sequences)
     *
     * For very large sequences, computing full scores matrix may be memory-prohibitive.
     * This implements online softmax with accumulation.
     */
    static void fused_attention_recompute(const float *q, const float *k, const float *v,
                                          float *out, int seq_len, int head_dim, int heads,
                                          bool causal)
    {
        const float scale = 1.0f / std::sqrt((float)head_dim);

        // Initialize output
        std::fill(out, out + (size_t)seq_len * heads * head_dim, 0.0f);

        // Per-row max and sum for online softmax
        std::vector<float> row_max((size_t)heads * seq_len, -std::numeric_limits<float>::infinity());
        std::vector<double> row_sum((size_t)heads * seq_len, 0.0);

// First pass: compute row max
#pragma omp parallel for collapse(2)
        for (int h = 0; h < heads; ++h)
        {
            for (int i = 0; i < seq_len; ++i)
            {
                const float *qi = q + (size_t)i * heads * head_dim + (size_t)h * head_dim;
                float max_score = -std::numeric_limits<float>::infinity();

                for (int j = 0; j < seq_len; ++j)
                {
                    if (causal && j > i)
                        continue;

                    const float *kj = k + (size_t)j * heads * head_dim + (size_t)h * head_dim;
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; ++d)
                    {
                        dot += qi[d] * kj[d];
                    }
                    max_score = std::max(max_score, dot * scale);
                }

                row_max[h * seq_len + i] = max_score;
            }
        }

// Second pass: compute exp and accumulate
#pragma omp parallel for collapse(2)
        for (int h = 0; h < heads; ++h)
        {
            for (int i = 0; i < seq_len; ++i)
            {
                const float *qi = q + (size_t)i * heads * head_dim + (size_t)h * head_dim;
                float *out_ptr = out + (size_t)i * heads * head_dim + (size_t)h * head_dim;
                const float max_score = row_max[h * seq_len + i];
                double sum = 0.0;

                for (int j = 0; j < seq_len; ++j)
                {
                    if (causal && j > i)
                        continue;

                    // Recompute score
                    const float *kj = k + (size_t)j * heads * head_dim + (size_t)h * head_dim;
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; ++d)
                    {
                        dot += qi[d] * kj[d];
                    }

                    const float score = std::exp((dot * scale) - max_score);
                    sum += score;

                    // Accumulate: out += score * V[j]
                    const float *vj = v + (size_t)j * heads * head_dim + (size_t)h * head_dim;
                    for (int d = 0; d < head_dim; ++d)
                    {
                        out_ptr[d] += score * vj[d];
                    }
                }

                row_sum[h * seq_len + i] = sum;
            }
        }

// Normalize by sum
#pragma omp parallel for collapse(2)
        for (int h = 0; h < heads; ++h)
        {
            for (int i = 0; i < seq_len; ++i)
            {
                float *out_ptr = out + (size_t)i * heads * head_dim + (size_t)h * head_dim;
                const double sum = row_sum[h * seq_len + i];

                if (sum > 0.0)
                {
                    const float inv_sum = 1.0f / (float)sum;
                    for (int d = 0; d < head_dim; ++d)
                    {
                        out_ptr[d] *= inv_sum;
                    }
                }
            }
        }
    }

    /**
     * @brief Complete fused attention with adaptive path selection
     *
     * Automatically selects between:
     * - Standard path (materialize scores): better for small sequences
     * - Recompute path (online softmax): better for large sequences
     */
    void fused_attention(const float *q, const float *k, const float *v, float *out,
                         int seq_len, int head_dim, int heads, bool causal)
    {
        const auto &env = llaminar::debugEnv().attention;

        // Determine whether to use fused recompute path
        const bool use_fused = env.prim_fused_recompute_threshold > 0 &&
                               seq_len >= env.prim_fused_recompute_threshold;

        if (use_fused && !env.prim_disable_fused)
        {
            fused_attention_recompute(q, k, v, out, seq_len, head_dim, heads, causal);
        }
        else
        {
            // Standard path: materialize scores
            std::vector<float> scores((size_t)heads * seq_len * seq_len);
            compute_qk_scores(q, k, scores.data(), seq_len, head_dim, heads, causal, true);
            apply_scores_to_v(scores.data(), v, out, seq_len, head_dim, heads);
        }
    }

    // ============================================================================
    // PUBLIC API - GQA/MHA Expansion
    // ============================================================================

    void expand_kv_for_gqa(const float *k_compact, const float *v_compact,
                           float *k_expanded, float *v_expanded,
                           int seq_len, int head_dim, int n_heads, int n_kv_heads,
                           int head_offset, int total_q_heads)
    {
        // Each KV head serves a group of consecutive Q heads
        // For Qwen: 14 Q heads, 2 KV heads → group_size = 14/2 = 7
        //   Q heads [0-6] → KV head 0
        //   Q heads [7-13] → KV head 1
        // Formula: kv_head = global_h / group_size
        //
        // In distributed setting:
        //   global_h = local_h + head_offset
        //   total_q_heads = global count of Q heads
        //   group_size = total_q_heads / n_kv_heads

        // Infer total_q_heads if not provided
        if (total_q_heads < 0)
        {
            // Single-rank case: n_heads is the total
            total_q_heads = n_heads;
        }

        const int group_size = total_q_heads / n_kv_heads;

#pragma omp parallel for schedule(static)
        for (int t = 0; t < seq_len; ++t)
        {
            for (int h = 0; h < n_heads; ++h)
            {
                const int global_h = h + head_offset;
                const int kv_head = global_h / group_size;

                const float *k_src = k_compact + (size_t)t * n_kv_heads * head_dim + (size_t)kv_head * head_dim;
                const float *v_src = v_compact + (size_t)t * n_kv_heads * head_dim + (size_t)kv_head * head_dim;

                float *k_dst = k_expanded + (size_t)t * n_heads * head_dim + (size_t)h * head_dim;
                float *v_dst = v_expanded + (size_t)t * n_heads * head_dim + (size_t)h * head_dim;

                std::memcpy(k_dst, k_src, head_dim * sizeof(float));
                std::memcpy(v_dst, v_src, head_dim * sizeof(float));
            }
        }
    }

    void expand_kv_for_mha(const float *k_compact, const float *v_compact,
                           float *k_expanded, float *v_expanded,
                           int seq_len, int kv_head_dim, int total_head_dim)
    {
// Simple copy when dimensions match (MHA case)
#pragma omp parallel for schedule(static)
        for (int t = 0; t < seq_len; ++t)
        {
            const float *k_src = k_compact + (size_t)t * kv_head_dim;
            const float *v_src = v_compact + (size_t)t * kv_head_dim;

            float *k_dst = k_expanded + (size_t)t * total_head_dim;
            float *v_dst = v_expanded + (size_t)t * total_head_dim;

            std::memcpy(k_dst, k_src, std::min(kv_head_dim, total_head_dim) * sizeof(float));
            std::memcpy(v_dst, v_src, std::min(kv_head_dim, total_head_dim) * sizeof(float));

            // Zero-pad if total_head_dim > kv_head_dim
            if (total_head_dim > kv_head_dim)
            {
                std::fill(k_dst + kv_head_dim, k_dst + total_head_dim, 0.0f);
                std::fill(v_dst + kv_head_dim, v_dst + total_head_dim, 0.0f);
            }
        }
    }

    // ============================================================================
    // PUBLIC API - Validation Utilities
    // ============================================================================

    RowSoftmaxStats validate_softmax_rows(const float *scores, int seq_len, int heads)
    {
        RowSoftmaxStats stats{0.0f, 0.0f, 0.0f};

        if (seq_len <= 0 || heads <= 0)
            return stats;

        float max_deviation = 0.0f;
        float max_negative = 0.0f;
        float max_prob = 0.0f;

#pragma omp parallel for reduction(max : max_deviation, max_negative, max_prob)
        for (int h = 0; h < heads; ++h)
        {
            for (int i = 0; i < seq_len; ++i)
            {
                const float *row = head_row(scores, h, i, seq_len);

                float row_sum = 0.0f;
                float row_min = 0.0f;
                float row_max = 0.0f;

                for (int j = 0; j < seq_len; ++j)
                {
                    const float val = row[j];
                    row_sum += val;
                    row_min = std::min(row_min, val);
                    row_max = std::max(row_max, val);
                }

                const float deviation = std::abs(row_sum - 1.0f);
                max_deviation = std::max(max_deviation, deviation);
                max_negative = std::min(max_negative, row_min);
                max_prob = std::max(max_prob, row_max);
            }
        }

        stats.max_row_deviation = max_deviation;
        stats.max_negative = max_negative;
        stats.max_prob = max_prob;

        return stats;
    }

} // namespace llaminar::attn
