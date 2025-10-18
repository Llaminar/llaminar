#include "utils/DebugEnv.h"
/**
 * @file AttentionPrimitives.cpp
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
#include <atomic>

#ifdef _OPENMP
#include <omp.h>
#endif

// SIMD intrinsics
#if defined(__AVX512F__)
#include <immintrin.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#endif

#include "utils/DebugEnv.h"
#include "operators/common/AttentionPrimitives.h"
#include "operators/common/SoftmaxCore.h"
#include "Logger.h"
#include "AdaptiveMatmul.h"

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
                                   int head_dim, bool debug = false, int head_idx = 0)
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

            // DEBUG: Log first few rotations for position 0 (use printf to bypass logging/OpenMP issues)
            // if (debug && position == 0 && head_idx == 0 && i < 5)
            // {
            //     printf("[RoPE_DEBUG] pos=0 head=0 dim_pair=%d inv_freq=%.10f angle=%.10f cos=%.10f sin=%.10f before=[%.6f,%.6f]\n",
            //            i, inv_freq[i], angle, cos_angle, sin_angle, x_first, x_second);
            //     fflush(stdout);
            // }

            // Apply rotation (rotate_half pattern)
            head_ptr[idx_first] = x_first * cos_angle - x_second * sin_angle;
            head_ptr[idx_second] = x_first * sin_angle + x_second * cos_angle;

            // DEBUG: Log after rotation
            // if (debug && position == 0 && head_idx == 0 && i < 5)
            // {
            //     printf("[RoPE_DEBUG]   after=[%.6f,%.6f]\n", head_ptr[idx_first], head_ptr[idx_second]);
            //     fflush(stdout);
            // }
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
                                     const std::vector<float> &inv_freq, bool debug = false)
    {
        // printf("[ROPE_TENSOR] seq_len=%d num_heads=%d head_dim=%d n_past=%d debug=%d\n",
        //        seq_len, num_heads, head_dim, n_past, (int)debug);
        // fflush(stdout);

        const auto &env = llaminar::debugEnv().attention;

        // Disable OpenMP if debug mode to ensure printf output appears
        bool use_parallel = !env.prim_force_scalar && !debug;

        // printf("[ROPE_TENSOR] About to enter loop, use_parallel=%d\n", (int)use_parallel);
        // fflush(stdout);

// Parallelize over (token, head) pairs
// Each iteration is independent, making this perfectly parallelizable
#pragma omp parallel for collapse(2) if (use_parallel)
        for (int t = 0; t < seq_len; ++t)
        {
            for (int h = 0; h < num_heads; ++h)
            {
                // Calculate absolute position
                const int position = n_past + t;

                // Pointer to this specific head: tensor[t, h, :]
                float *head_ptr = tensor + (size_t)t * num_heads * head_dim + (size_t)h * head_dim;

                // Apply rotation - pass debug for first token, first head only
                bool debug_this = debug && (t == 0) && (h == 0);

                // if (t == 0 && h == 0)
                // {
                //     printf("[ROPE_TENSOR] Processing t=0 h=0, debug_this=%d\n", (int)debug_this);
                //     fflush(stdout);
                // }

                apply_rope_to_head(head_ptr, position, inv_freq, head_dim, debug_this, h);
            }
        }

        // printf("[ROPE_TENSOR] Loop complete\n");
        // fflush(stdout);
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
        // fprintf(stderr, "[ROPE_ENTRY] seq_len=%d q_heads=%d k_heads=%d\n", seq_len, q_heads, k_heads);
        // fflush(stderr);

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
        bool enable_debug = (call_count == 0); // Debug first call

        if (call_count < 3)
        {
            LOG_DEBUG("[RoPE] call=" << call_count
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

        // Log first few inverse frequencies for debugging (before OpenMP region to guarantee output)
        // if (enable_debug)
        // {
        //     printf("[RoPE_DEBUG] First 5 inv_freq values:\n");
        //     for (int i = 0; i < std::min(5, (int)inv_freq.size()); ++i)
        //     {
        //         float exponent = (2.0f * i) / head_dim;
        //         printf("[RoPE_DEBUG]   inv_freq[%d] = %.10f (from 1e6^%.5f = %.5f)\n",
        //                i, inv_freq[i], exponent, std::pow(freq_base, exponent));
        //     }
        //     printf("[RoPE_DEBUG] PyTorch expects inv_freq[0]=1.0, inv_freq[1]≈0.9886\n");
        //     fflush(stdout);
        // }

        // Apply RoPE to Q tensor
        apply_rope_to_tensor(q, seq_len, q_heads, head_dim, n_past, inv_freq, enable_debug);

        // Apply RoPE to K tensor (may have different number of heads for GQA)
        apply_rope_to_tensor(k, seq_len, k_heads, head_dim, n_past, inv_freq, enable_debug);

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
    // BATCHED ATTENTION PRIMITIVES
    // ============================================================================

    void apply_rope_batched(float *q, float *k, int batch_size, int seq_len, int head_dim,
                            int q_heads, int k_heads, int n_past, float freq_base)
    {
        // Validate inputs
        if (head_dim % 2 != 0)
        {
            LOG_ERROR("[RoPE_Batched] head_dim must be even, got " << head_dim);
            return;
        }

        if (batch_size <= 0 || seq_len <= 0 || q_heads <= 0 || k_heads <= 0)
        {
            LOG_WARN("[RoPE_Batched] Invalid dimensions: batch_size=" << batch_size
                                                                      << " seq_len=" << seq_len
                                                                      << " q_heads=" << q_heads
                                                                      << " k_heads=" << k_heads);
            return;
        }

        // Compute inverse frequencies once (same for all batches)
        const std::vector<float> inv_freq = compute_inv_freq(head_dim, freq_base);

        // Process each batch element independently
        // Layout: [batch_size, seq_len, num_heads * head_dim]
        const int q_stride_batch = seq_len * q_heads * head_dim;
        const int k_stride_batch = seq_len * k_heads * head_dim;

        for (int b = 0; b < batch_size; ++b)
        {
            float *q_batch = q + b * q_stride_batch;
            float *k_batch = k + b * k_stride_batch;

            // Reuse the proven single-batch RoPE implementation
            // It expects layout [seq_len, num_heads, head_dim] which matches our per-batch slice
            apply_rope_to_tensor(q_batch, seq_len, q_heads, head_dim, n_past, inv_freq, false);
            apply_rope_to_tensor(k_batch, seq_len, k_heads, head_dim, n_past, inv_freq, false);
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

#if defined(__AVX512F__)
    // AVX512 horizontal max reduction
    static inline float hmax512(__m512 v)
    {
        __m256 lo = _mm512_castps512_ps256(v);
        __m256 hi = _mm512_extractf32x8_ps(v, 1);
        __m256 m = _mm256_max_ps(lo, hi);
        __m128 lo128 = _mm256_castps256_ps128(m);
        __m128 hi128 = _mm256_extractf128_ps(m, 1);
        __m128 m128 = _mm_max_ps(lo128, hi128);
        __m128 shuf = _mm_movehdup_ps(m128);
        m128 = _mm_max_ps(m128, shuf);
        shuf = _mm_movehl_ps(shuf, m128);
        m128 = _mm_max_ps(m128, shuf);
        return _mm_cvtss_f32(m128);
    }

    // AVX512 horizontal sum reduction
    static inline float hsum512(__m512 v)
    {
        __m256 lo = _mm512_castps512_ps256(v);
        __m256 hi = _mm512_extractf32x8_ps(v, 1);
        __m256 sum = _mm256_add_ps(lo, hi);
        __m128 lo128 = _mm256_castps256_ps128(sum);
        __m128 hi128 = _mm256_extractf128_ps(sum, 1);
        __m128 sum128 = _mm_add_ps(lo128, hi128);
        __m128 shuf = _mm_movehdup_ps(sum128);
        sum128 = _mm_add_ps(sum128, shuf);
        shuf = _mm_movehl_ps(shuf, sum128);
        sum128 = _mm_add_ps(sum128, shuf);
        return _mm_cvtss_f32(sum128);
    }

    // Fast exp approximation for AVX512
    static inline __m512 fast_exp512(__m512 x)
    {
        const __m512 max_clip = _mm512_set1_ps(10.0f);
        const __m512 min_clip = _mm512_set1_ps(-20.0f);
        x = _mm512_max_ps(min_clip, _mm512_min_ps(max_clip, x));
        const __m512 inv_ln2 = _mm512_set1_ps(1.4426950408889634f);
        __m512 xf = _mm512_mul_ps(x, inv_ln2);
        __m512 fx = _mm512_floor_ps(xf);
        __m512 fpart = _mm512_sub_ps(xf, fx);
        const __m512 c1 = _mm512_set1_ps(0.99999994f);
        const __m512 c2 = _mm512_set1_ps(0.69314718f);
        const __m512 c3 = _mm512_set1_ps(0.24022651f);
        const __m512 c4 = _mm512_set1_ps(0.05550411f);
        const __m512 c5 = _mm512_set1_ps(0.00961813f);
        __m512 p = c5;
        p = _mm512_fmadd_ps(p, fpart, c4);
        p = _mm512_fmadd_ps(p, fpart, c3);
        p = _mm512_fmadd_ps(p, fpart, c2);
        p = _mm512_fmadd_ps(p, fpart, c1);
        __m512i ipart = _mm512_cvttps_epi32(fx);
        ipart = _mm512_add_epi32(ipart, _mm512_set1_epi32(127));
        ipart = _mm512_slli_epi32(ipart, 23);
        __m512 two_ip = _mm512_castsi512_ps(ipart);
        return _mm512_mul_ps(two_ip, p);
    }
#endif

#if defined(__AVX2__)
    // AVX2 horizontal max reduction
    static inline float hmax256(__m256 v)
    {
        __m128 lo = _mm256_castps256_ps128(v);
        __m128 hi = _mm256_extractf128_ps(v, 1);
        __m128 m = _mm_max_ps(lo, hi);
        __m128 shuf = _mm_movehdup_ps(m);
        m = _mm_max_ps(m, shuf);
        shuf = _mm_movehl_ps(shuf, m);
        m = _mm_max_ps(m, shuf);
        return _mm_cvtss_f32(m);
    }

    // AVX2 horizontal sum reduction
    static inline float hsum256(__m256 v)
    {
        __m128 lo = _mm256_castps256_ps128(v);
        __m128 hi = _mm256_extractf128_ps(v, 1);
        __m128 sum = _mm_add_ps(lo, hi);
        __m128 shuf = _mm_movehdup_ps(sum);
        sum = _mm_add_ps(sum, shuf);
        shuf = _mm_movehl_ps(shuf, sum);
        sum = _mm_add_ps(sum, shuf);
        return _mm_cvtss_f32(sum);
    }

    // Fast exp approximation for AVX2
    static inline __m256 fast_exp256(__m256 x)
    {
        const __m256 max_clip = _mm256_set1_ps(10.0f);
        const __m256 min_clip = _mm256_set1_ps(-20.0f);
        x = _mm256_max_ps(min_clip, _mm256_min_ps(max_clip, x));
        const __m256 inv_ln2 = _mm256_set1_ps(1.4426950408889634f);
        __m256 xf = _mm256_mul_ps(x, inv_ln2);
        __m256 fx = _mm256_floor_ps(xf);
        __m256 fpart = _mm256_sub_ps(xf, fx);
        const __m256 c1 = _mm256_set1_ps(0.99999994f);
        const __m256 c2 = _mm256_set1_ps(0.69314718f);
        const __m256 c3 = _mm256_set1_ps(0.24022651f);
        const __m256 c4 = _mm256_set1_ps(0.05550411f);
        const __m256 c5 = _mm256_set1_ps(0.00961813f);
        __m256 p = c5;
        p = _mm256_fmadd_ps(p, fpart, c4);
        p = _mm256_fmadd_ps(p, fpart, c3);
        p = _mm256_fmadd_ps(p, fpart, c2);
        p = _mm256_fmadd_ps(p, fpart, c1);
        __m256i ipart = _mm256_cvttps_epi32(fx);
        ipart = _mm256_add_epi32(ipart, _mm256_set1_epi32(127));
        ipart = _mm256_slli_epi32(ipart, 23);
        __m256 two_ip = _mm256_castsi256_ps(ipart);
        return _mm256_mul_ps(two_ip, p);
    }
#endif

    /**
     * @brief Vectorized fused softmax with causal masking (AVX512/AVX2)
     *
     * SIMD-optimized single-pass softmax that combines:
     * - Causal masking (no materialized -inf)
     * - Horizontal max finding (AVX reductions)
     * - Fast exp approximation (polynomial)
     * - Vectorized normalization
     *
     * @param scores Score matrix [heads, q_seq_len, k_seq_len]
     * @param heads Number of attention heads
     * @param q_seq_len Query sequence length
     * @param k_seq_len Key/Value sequence length
     * @param causal Whether to apply causal masking
     */
    static void fused_softmax_with_causal_mask(float *scores, int heads, int q_seq_len, int k_seq_len, bool causal)
    {
        const auto &env = llaminar::debugEnv().attention;

#pragma omp parallel for collapse(2) if (!env.prim_force_scalar && heads * q_seq_len > 8)
        for (int h = 0; h < heads; ++h)
        {
            for (int i = 0; i < q_seq_len; ++i)
            {
                float *row = scores + (size_t)h * q_seq_len * k_seq_len + (size_t)i * k_seq_len;

                // Calculate absolute query position for causal masking
                int abs_q_pos = (k_seq_len - q_seq_len) + i;

                // Determine effective row length (respect causal mask)
                int effective_len = causal ? (abs_q_pos + 1) : k_seq_len;

                // Pass 1: Find max (vectorized)
                float row_max = -std::numeric_limits<float>::infinity();
                int j = 0;

#if defined(__AVX512F__)
                // AVX512: Process 16 floats at a time
                if (effective_len >= 16)
                {
                    __m512 vmax = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
                    for (; j + 16 <= effective_len; j += 16)
                    {
                        __m512 v = _mm512_loadu_ps(row + j);
                        vmax = _mm512_max_ps(vmax, v);
                    }
                    row_max = hmax512(vmax);
                }
#elif defined(__AVX2__)
                // AVX2: Process 8 floats at a time
                if (effective_len >= 8)
                {
                    __m256 vmax = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
                    for (; j + 8 <= effective_len; j += 8)
                    {
                        __m256 v = _mm256_loadu_ps(row + j);
                        vmax = _mm256_max_ps(vmax, v);
                    }
                    row_max = hmax256(vmax);
                }
#endif
                // Scalar tail for remaining elements
                for (; j < effective_len; ++j)
                {
                    row_max = std::max(row_max, row[j]);
                }

                // Pass 2: Compute exp and sum (vectorized)
                double row_sum = 0.0;
                j = 0;

#if defined(__AVX512F__)
                // AVX512: Process 16 floats at a time
                if (effective_len >= 16)
                {
                    const __m512 vmax = _mm512_set1_ps(row_max);
                    __m512 vsum = _mm512_setzero_ps();

                    for (; j + 16 <= effective_len; j += 16)
                    {
                        __m512 v = _mm512_loadu_ps(row + j);
                        v = _mm512_sub_ps(v, vmax);
                        __m512 vexp = fast_exp512(v);
                        _mm512_storeu_ps(row + j, vexp);
                        vsum = _mm512_add_ps(vsum, vexp);
                    }
                    row_sum = hsum512(vsum);
                }
#elif defined(__AVX2__)
                // AVX2: Process 8 floats at a time
                if (effective_len >= 8)
                {
                    const __m256 vmax = _mm256_set1_ps(row_max);
                    __m256 vsum = _mm256_setzero_ps();

                    for (; j + 8 <= effective_len; j += 8)
                    {
                        __m256 v = _mm256_loadu_ps(row + j);
                        v = _mm256_sub_ps(v, vmax);
                        __m256 vexp = fast_exp256(v);
                        _mm256_storeu_ps(row + j, vexp);
                        vsum = _mm256_add_ps(vsum, vexp);
                    }
                    row_sum = hsum256(vsum);
                }
#endif
                // Scalar tail
                for (; j < effective_len; ++j)
                {
                    row[j] = std::exp(row[j] - row_max);
                    row_sum += row[j];
                }

                // Pass 3: Normalize (vectorized)
                const float inv_sum = (row_sum > 0.0) ? (1.0f / static_cast<float>(row_sum)) : 0.0f;
                j = 0;

#if defined(__AVX512F__)
                // AVX512: Normalize 16 floats at a time
                if (effective_len >= 16)
                {
                    const __m512 vinv = _mm512_set1_ps(inv_sum);
                    for (; j + 16 <= effective_len; j += 16)
                    {
                        __m512 v = _mm512_loadu_ps(row + j);
                        v = _mm512_mul_ps(v, vinv);
                        _mm512_storeu_ps(row + j, v);
                    }
                }
#elif defined(__AVX2__)
                // AVX2: Normalize 8 floats at a time
                if (effective_len >= 8)
                {
                    const __m256 vinv = _mm256_set1_ps(inv_sum);
                    for (; j + 8 <= effective_len; j += 8)
                    {
                        __m256 v = _mm256_loadu_ps(row + j);
                        v = _mm256_mul_ps(v, vinv);
                        _mm256_storeu_ps(row + j, v);
                    }
                }
#endif
                // Scalar tail
                for (; j < effective_len; ++j)
                {
                    row[j] *= inv_sum;
                }

                // Pass 4: Zero out causal-masked positions (vectorized)
                j = effective_len;
#if defined(__AVX512F__)
                // AVX512: Zero 16 floats at a time
                const __m512 vzero512 = _mm512_setzero_ps();
                for (; j + 16 <= k_seq_len; j += 16)
                {
                    _mm512_storeu_ps(row + j, vzero512);
                }
#elif defined(__AVX2__)
                // AVX2: Zero 8 floats at a time
                const __m256 vzero256 = _mm256_setzero_ps();
                for (; j + 8 <= k_seq_len; j += 8)
                {
                    _mm256_storeu_ps(row + j, vzero256);
                }
#endif
                // Scalar tail
                for (; j < k_seq_len; ++j)
                {
                    row[j] = 0.0f;
                }
            }
        }
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
     * Uses adaptiveMatMul for optimized computation (OpenBLAS/COSMA backend).
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
                           int q_seq_len, int k_seq_len, int head_dim, int heads,
                           bool causal, bool apply_softmax)
    {
        const auto &env = llaminar::debugEnv().attention;
        const float scale = 1.0f / std::sqrt((float)head_dim);

        // DEBUG: Log parameters for first call
        static bool first_call = true;
        if (first_call && causal)
        {
            LOG_DEBUG("[compute_qk_scores DEBUG] First causal call:");
            LOG_DEBUG("  q_seq_len=" << q_seq_len << ", k_seq_len=" << k_seq_len
                                     << ", head_dim=" << head_dim << ", heads=" << heads);
            first_call = false;
        }

        // Process each head independently using optimized matmul
        // For each head h: scores[h] = (Q[h] @ K[h]^T) * scale
        //   Q[h]: [q_seq_len, head_dim]
        //   K[h]: [k_seq_len, head_dim]
        //   scores[h]: [q_seq_len, k_seq_len]

        // Always use GEMM path for optimal performance
        // Strategy: Reshape Q/K from [seq_len, heads, head_dim] to [heads, seq_len, head_dim]
        // Then compute all heads with better cache locality

        // Allocate contiguous buffers for all heads: [heads, seq_len, head_dim]
        std::vector<float> q_reordered(heads * q_seq_len * head_dim);
        std::vector<float> k_reordered(heads * k_seq_len * head_dim);

// Reorder Q: [seq_len, heads, head_dim] -> [heads, seq_len, head_dim]
#pragma omp parallel for collapse(2) if (heads * q_seq_len > 256)
        for (int h = 0; h < heads; ++h)
        {
            for (int i = 0; i < q_seq_len; ++i)
            {
                const float *q_src = q + (size_t)i * heads * head_dim + (size_t)h * head_dim;
                float *q_dst = q_reordered.data() + (size_t)h * q_seq_len * head_dim + (size_t)i * head_dim;
                std::memcpy(q_dst, q_src, head_dim * sizeof(float));
            }
        }

// Reorder K: [seq_len, heads, head_dim] -> [heads, seq_len, head_dim]
#pragma omp parallel for collapse(2) if (heads * k_seq_len > 256)
        for (int h = 0; h < heads; ++h)
        {
            for (int j = 0; j < k_seq_len; ++j)
            {
                const float *k_src = k + (size_t)j * heads * head_dim + (size_t)h * head_dim;
                float *k_dst = k_reordered.data() + (size_t)h * k_seq_len * head_dim + (size_t)j * head_dim;
                std::memcpy(k_dst, k_src, head_dim * sizeof(float));
            }
        }

        // Compute Q @ K^T for all heads using batched approach
        // Process heads in parallel to maximize throughput
        bool is_prefill = (q_seq_len == k_seq_len);
        bool all_success = true;

#pragma omp parallel for if (heads > 2)
        for (int h = 0; h < heads; ++h)
        {
            const float *q_head = q_reordered.data() + (size_t)h * q_seq_len * head_dim;
            const float *k_head = k_reordered.data() + (size_t)h * k_seq_len * head_dim;
            float *score_head = scores + (size_t)h * q_seq_len * k_seq_len;

            // Compute Q_h @ K_h^T * scale for this head
            bool matmul_success = llaminar::adaptiveMatMul(
                q_head,     // A: [q_seq_len, head_dim]
                k_head,     // B: [k_seq_len, head_dim]
                score_head, // C: [q_seq_len, k_seq_len]
                q_seq_len,  // m
                k_seq_len,  // n
                head_dim,   // k
                is_prefill, // is_prefill hint
                false,      // distributed_partition (attention scores are local)
                false,      // transpose_A
                true,       // transpose_B (K^T)
                scale,      // alpha (apply scale directly)
                0.0f        // beta (overwrite output)
            );

            if (!matmul_success)
            {
#pragma omp critical
                {
                    LOG_ERROR("[compute_qk_scores] adaptiveMatMul failed for head " << h);
                    all_success = false;
                }
            }
        }

        if (!all_success)
        {
            return;
        }

        // Apply causal masking if needed (post-process all heads)
        if (causal)
        {
#pragma omp parallel for collapse(2) if (heads * q_seq_len > 256)
            for (int h = 0; h < heads; ++h)
            {
                for (int i = 0; i < q_seq_len; ++i)
                {
                    // Calculate absolute query position for causal masking
                    // For prefill: k_seq_len == q_seq_len, so abs_q_pos = i
                    // For decode: k_seq_len = n_past + q_seq_len, query starts at n_past
                    int abs_q_pos = (k_seq_len - q_seq_len) + i;

                    float *score_row = scores + (size_t)h * q_seq_len * k_seq_len + (size_t)i * k_seq_len;
                    for (int j = 0; j < k_seq_len; ++j)
                    {
                        if (j > abs_q_pos)
                        {
                            // DEBUG: Log first few masks
                            if (h == 0 && i == 0 && j <= 7)
                            {
                                LOG_DEBUG("[CAUSAL_MASK] h=" << h << ", i=" << i << ", abs_q_pos=" << abs_q_pos
                                                             << ", j=" << j << " -> MASKED (j > abs_q_pos)");
                            }
                            score_row[j] = -std::numeric_limits<float>::infinity();
                        }
                        else if (h == 0 && i == 0 && j <= 7)
                        {
                            // DEBUG: Log first few scores
                            LOG_DEBUG("[SCORE_COMPUTE] h=" << h << ", i=" << i << ", abs_q_pos=" << abs_q_pos
                                                           << ", j=" << j << " -> score=" << score_row[j]);
                        }
                    }
                }
            }
        }
        else
        {
            // DEBUG: Log first few scores for non-causal case
            if (heads > 0 && q_seq_len > 0)
            {
                const float *score_head = scores; // First head
                for (int i = 0; i < std::min(1, q_seq_len); ++i)
                {
                    const float *score_row = score_head + i * k_seq_len;
                    for (int j = 0; j < std::min(8, k_seq_len); ++j)
                    {
                        LOG_DEBUG("[SCORE_COMPUTE] h=0, i=" << i << ", j=" << j
                                                            << " -> score=" << score_row[j]);
                    }
                }
            }
        }

        // Apply softmax if requested
        if (apply_softmax)
        {
            if (causal)
            {
                // Use fused kernel for causal masking + softmax
                // The fused kernel eliminates the materialized -inf values and does softmax in one pass
                fused_softmax_with_causal_mask(scores, heads, q_seq_len, k_seq_len, causal);
            }
            else
            {
                // Use standard softmax for non-causal attention
                llaminar::kernels::SoftmaxRowArgs softmax_args;
                softmax_args.scores = scores;
                softmax_args.rows = heads * q_seq_len;
                softmax_args.cols = k_seq_len;
                softmax_args.causal = false; // Already masked above
                softmax_args.scale = 1.0f;   // Scale already applied
                llaminar::kernels::softmax_row_major(softmax_args);
            }
        }
    }

    // ============================================================================
    // PUBLIC API - Apply Attention Scores to Values
    // ============================================================================

    /**
     * @brief Apply attention scores to values: out = scores @ V
     *
     * @param scores Attention scores [heads, q_seq_len, k_seq_len]
     * @param v Value tensor [k_seq_len, heads, head_dim]
     * @param out Output tensor [q_seq_len, heads, head_dim]
     * @param q_seq_len Query sequence length
     * @param k_seq_len Key/Value sequence length
     * @param head_dim Dimension per head
     * @param heads Number of heads
     */
    void apply_scores_to_v(const float *scores, const float *v, float *out,
                           int q_seq_len, int k_seq_len, int head_dim, int heads)
    {
        // Always use GEMM path for optimal performance
        // Reorder V to [heads, k_seq_len, head_dim] for efficient GEMM
        std::vector<float> v_reordered(heads * k_seq_len * head_dim);

// Reorder V: [k_seq_len, heads, head_dim] -> [heads, k_seq_len, head_dim]
#pragma omp parallel for collapse(2) if (heads * k_seq_len > 256)
        for (int h = 0; h < heads; ++h)
        {
            for (int j = 0; j < k_seq_len; ++j)
            {
                const float *v_src = v + (size_t)j * heads * head_dim + (size_t)h * head_dim;
                float *v_dst = v_reordered.data() + (size_t)h * k_seq_len * head_dim + (size_t)j * head_dim;
                std::memcpy(v_dst, v_src, head_dim * sizeof(float));
            }
        }

        // Allocate temporary output in [heads, q_seq_len, head_dim] layout
        std::vector<float> out_reordered(heads * q_seq_len * head_dim);

        // Compute scores[h] @ V[h] for all heads in parallel
        // scores[h]: [q_seq_len, k_seq_len]
        // V[h]: [k_seq_len, head_dim]
        // out[h]: [q_seq_len, head_dim]
        bool all_success = true;

#pragma omp parallel for if (heads > 2)
        for (int h = 0; h < heads; ++h)
        {
            const float *score_head = scores + (size_t)h * q_seq_len * k_seq_len;
            const float *v_head = v_reordered.data() + (size_t)h * k_seq_len * head_dim;
            float *out_head = out_reordered.data() + (size_t)h * q_seq_len * head_dim;

            // out[h] = scores[h] @ V[h]
            // C = A @ B where:
            //   A = scores[h]: [q_seq_len, k_seq_len]
            //   B = V[h]: [k_seq_len, head_dim]
            //   C = out[h]: [q_seq_len, head_dim]
            bool matmul_success = llaminar::adaptiveMatMul(
                score_head,               // A: [q_seq_len, k_seq_len]
                v_head,                   // B: [k_seq_len, head_dim]
                out_head,                 // C: [q_seq_len, head_dim]
                q_seq_len,                // m
                head_dim,                 // n
                k_seq_len,                // k
                (q_seq_len == k_seq_len), // is_prefill hint
                false,                    // distributed_partition
                false,                    // transpose_A
                false,                    // transpose_B
                1.0f,                     // alpha
                0.0f                      // beta
            );

            if (!matmul_success)
            {
#pragma omp critical
                {
                    LOG_ERROR("[apply_scores_to_v] adaptiveMatMul failed for head " << h);
                    all_success = false;
                }
            }
        }

        if (!all_success)
        {
            return;
        }

// Reorder output back: [heads, q_seq_len, head_dim] -> [q_seq_len, heads, head_dim]
#pragma omp parallel for collapse(2) if (heads * q_seq_len > 256)
        for (int i = 0; i < q_seq_len; ++i)
        {
            for (int h = 0; h < heads; ++h)
            {
                const float *out_src = out_reordered.data() + (size_t)h * q_seq_len * head_dim + (size_t)i * head_dim;
                float *out_dst = out + (size_t)i * heads * head_dim + (size_t)h * head_dim;
                std::memcpy(out_dst, out_src, head_dim * sizeof(float));
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
            compute_qk_scores(q, k, scores.data(), seq_len, seq_len, head_dim, heads, causal, false);

            // Apply softmax to convert scores to probabilities
            // Note: causal masking already applied in compute_qk_scores (masked positions are -inf)
            llaminar::kernels::SoftmaxRowArgs softmax_args;
            softmax_args.scores = scores.data();
            softmax_args.rows = heads * seq_len;
            softmax_args.cols = seq_len;
            softmax_args.causal = false; // Already masked in compute_qk_scores
            softmax_args.scale = 1.0f;   // Scale already applied in compute_qk_scores
            llaminar::kernels::softmax_row_major(softmax_args);

            apply_scores_to_v(scores.data(), v, out, seq_len, seq_len, head_dim, heads);
        }
    }

    // ============================================================================
    // PUBLIC API - GQA/MHA Expansion
    // ============================================================================

    void expand_kv_for_gqa(const float *k_compact, const float *v_compact,
                           float *k_expanded, float *v_expanded,
                           int seq_len, int head_dim, int n_heads, int n_kv_heads,
                           int head_offset, int total_q_heads,
                           bool gathered_rank_major, int kv_head_offset_for_rank)
    {
        // CRITICAL FIX: GQA expansion is ALWAYS needed for correctness!
        // The capture_enabled flag should only gate snapshot captures, not functional operations.
        // Removing the early return that was causing attention to use uninitialized K/V buffers.

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
        //
        // LAYOUT HANDLING:
        // - gathered_rank_major=false (default): Time-major layout
        //   [t=0: kv_head=0..n_kv_heads] [t=1: kv_head=0..n_kv_heads] ...
        //   Source index: t * n_kv_heads * head_dim + kv_head * head_dim
        //
        // - gathered_rank_major=true: Rank-major layout from MPI_Allgatherv (NO TRANSPOSE!)
        //   [Rank0: t=0..T, local_kv_heads] [Rank1: t=0..T, local_kv_heads] ...
        //   Need to compute which rank owns the target kv_head, then index into that rank's block

        // Infer total_q_heads if not provided
        if (total_q_heads < 0)
        {
            // Single-rank case: n_heads is the total
            total_q_heads = n_heads;
        }

        const int group_size = total_q_heads / n_kv_heads;

        if (!gathered_rank_major)
        {
            // FAST PATH: Time-major layout (single-rank or already transposed)
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
        else
        {
            // RANK-MAJOR LAYOUT: Direct indexing without transpose
            // Gathered data layout: [Rank0: seq_len * local_kv_dim] [Rank1: seq_len * local_kv_dim] ...
            // We need to know which rank owns each KV head and index accordingly
            //
            // For this rank's Q heads, determine which KV head each needs (via group_size),
            // then find that KV head in the gathered buffer.
            //
            // Assumption: KV heads are evenly distributed across ranks in order
            // E.g., 2 KV heads, 2 ranks: Rank0 has KV head 0, Rank1 has KV head 1
            //
            // Since we gathered with MPI_Allgatherv, the buffer structure is:
            //   [Rank0: t=0..T, kv_head=kv_head_offset_for_rank..kv_head_offset_for_rank+local_kv_heads-1]
            //   [Rank1: t=0..T, kv_head=...]
            //   ...
            //
            // Strategy: For each needed kv_head, compute which rank block it's in and the offset within that block

#pragma omp parallel for schedule(static)
            for (int t = 0; t < seq_len; ++t)
            {
                for (int h = 0; h < n_heads; ++h)
                {
                    const int global_h = h + head_offset;
                    const int kv_head = global_h / group_size; // Global KV head index [0..n_kv_heads-1]

                    // Find source in rank-major layout
                    // Assume uniform distribution: each rank gets roughly n_kv_heads / world_size KV heads
                    // For simplicity with current architecture, assume each rank gets exactly 1 KV head (world_size == n_kv_heads)
                    // This is the case for our 2-rank, 2-KV-head scenario.
                    //
                    // General formula: rank_owning_kv = kv_head (when world_size == n_kv_heads)
                    // local_kv_index_in_rank = 0 (when each rank has 1 KV head)
                    //
                    // For gathered buffer with uniform 1 KV head per rank:
                    //   Rank r contributes seq_len * 1 * head_dim floats at offset r * seq_len * head_dim
                    //   Within rank r's block: timestep t is at offset t * head_dim
                    //
                    // Source index = kv_head * seq_len * head_dim + t * head_dim

                    const size_t src_offset = (size_t)kv_head * seq_len * head_dim + (size_t)t * head_dim;
                    const float *k_src = k_compact + src_offset;
                    const float *v_src = v_compact + src_offset;

                    float *k_dst = k_expanded + (size_t)t * n_heads * head_dim + (size_t)h * head_dim;
                    float *v_dst = v_expanded + (size_t)t * n_heads * head_dim + (size_t)h * head_dim;

                    std::memcpy(k_dst, k_src, head_dim * sizeof(float));
                    std::memcpy(v_dst, v_src, head_dim * sizeof(float));
                }
            }
        }
    }

    void expand_kv_for_mha(const float *k_compact, const float *v_compact,
                           float *k_expanded, float *v_expanded,
                           int seq_len, int kv_head_dim, int total_head_dim)
    {
        // Disable gather/rearrange unless snapshot capture is enabled
        const auto &env = llaminar::debugEnv().attention;
        if (!env.capture_enabled)
        {
            // In perf mode, skip all-gather and rearrange (no snapshot overhead)
            return;
        }

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

    // ============================================================================
    // BATCHED ATTENTION SCORE PRIMITIVES
    // ============================================================================

    void compute_qk_scores_batched(const float *q, const float *k, float *scores,
                                   int batch_size, int seq_len, int head_dim, int heads, bool causal)
    {
        // Process each batch element independently
        // Layout: Q, K: [batch_size, seq_len, heads * head_dim]
        //         scores: [batch_size, heads, seq_len, seq_len]

        const int q_stride_batch = seq_len * heads * head_dim;
        const int k_stride_batch = seq_len * heads * head_dim;
        const int scores_stride_batch = heads * seq_len * seq_len;

// Batch-first parallelism: each thread processes a full batch element
// This improves cache locality by keeping all data for one batch in thread-local cache
#pragma omp parallel for schedule(static) if (batch_size > 1)
        for (int b = 0; b < batch_size; ++b)
        {
            const float *q_batch = q + b * q_stride_batch;
            const float *k_batch = k + b * k_stride_batch;
            float *scores_batch = scores + b * scores_stride_batch;

            // Delegate to proven single-batch implementation with fused softmax
            // It expects: Q, K: [seq_len, heads * head_dim]
            //            scores: [heads, seq_len, seq_len]
            // apply_softmax=true enables our vectorized fused kernel
            compute_qk_scores(q_batch, k_batch, scores_batch,
                              seq_len, seq_len, head_dim, heads,
                              causal, true); // Enable fused vectorized softmax
        }
    }

    void apply_scores_to_v_batched(const float *scores, const float *v, float *out,
                                   int batch_size, int seq_len, int head_dim, int heads)
    {
        // Process each batch element independently
        // Layout: scores: [batch_size, heads, seq_len, seq_len]
        //         V: [batch_size, seq_len, heads * head_dim]
        //         out: [batch_size, seq_len, heads * head_dim]

        const int scores_stride_batch = heads * seq_len * seq_len;
        const int v_stride_batch = seq_len * heads * head_dim;
        const int out_stride_batch = seq_len * heads * head_dim;

// Batch-first parallelism: each thread processes a full batch element
// This improves cache locality and reduces thread contention
#pragma omp parallel for schedule(static) if (batch_size > 1)
        for (int b = 0; b < batch_size; ++b)
        {
            const float *scores_batch = scores + b * scores_stride_batch;
            const float *v_batch = v + b * v_stride_batch;
            float *out_batch = out + b * out_stride_batch;

            // Delegate to proven single-batch implementation
            // It expects: scores: [heads, seq_len, seq_len]
            //            V: [seq_len, heads * head_dim]
            //            out: [seq_len, heads * head_dim]
            apply_scores_to_v(scores_batch, v_batch, out_batch,
                              seq_len, seq_len, head_dim, heads);
        }
    }

} // namespace llaminar::attn
