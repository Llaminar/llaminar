/**
 * @file RoPEPrimitives.cpp
 * @brief Vectorized RoPE implementation (ported from V1)
 * @author David Sanftenberg
 */

#include "RoPEPrimitives.h"
#include "../../../tensors/SIMDHelpers.h"
#include "../../../tensors/FP16Utils.h"
#include <cmath>
#include <cstring>
#include <unordered_map>
#include <mutex>
#include <array>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "../../../utils/OpenMPUtils.h"

#if defined(__AVX512F__)
#include <immintrin.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#endif

namespace llaminar2::primitives
{

    // ============================================================================
    // Inverse Frequency Cache
    // ============================================================================

    namespace
    {
        static std::unordered_map<uint64_t, std::vector<float>> g_inv_freq_cache;
        static std::mutex g_inv_freq_mutex;

        uint64_t make_cache_key(int head_dim, float freq_base)
        {
            uint32_t freq_bits;
            std::memcpy(&freq_bits, &freq_base, sizeof(float));
            return ((uint64_t)head_dim << 32) | freq_bits;
        }
    } // anonymous namespace

    const std::vector<float> &get_inv_freq_cached(int head_dim, float freq_base)
    {
        uint64_t key = make_cache_key(head_dim, freq_base);

        {
            std::lock_guard<std::mutex> lock(g_inv_freq_mutex);
            auto it = g_inv_freq_cache.find(key);
            if (it != g_inv_freq_cache.end())
            {
                return it->second;
            }
        }

        // Compute inverse frequencies
        const int half_dim = head_dim / 2;
        std::vector<float> inv_freq(half_dim);
        const float log_base = std::log(freq_base);

        for (int i = 0; i < half_dim; ++i)
        {
            float exponent = (2.0f * i) / head_dim;
            inv_freq[i] = std::exp(-log_base * exponent);
        }

        std::lock_guard<std::mutex> lock(g_inv_freq_mutex);
        auto res = g_inv_freq_cache.emplace(key, std::move(inv_freq));
        return res.first->second;
    }

    // ============================================================================
    // Vectorized RoPE Application - Separated Implementations
    // ============================================================================

#if defined(__AVX2__)
    /**
     * @brief Apply RoPE rotation using cached sin/cos (AVX2)
     */
    void apply_rope_to_head_cached_avx2(
        float *head_ptr,
        const float *cos_cache,
        const float *sin_cache,
        int head_dim)
    {
        const int half_dim = head_dim / 2;
        int i = 0;

        for (; i + 8 <= half_dim; i += 8)
        {
            __m256 x_first = _mm256_loadu_ps(head_ptr + i);
            __m256 x_second = _mm256_loadu_ps(head_ptr + i + half_dim);
            __m256 cos_vec = _mm256_loadu_ps(cos_cache + i);
            __m256 sin_vec = _mm256_loadu_ps(sin_cache + i);

            __m256 new_first = _mm256_sub_ps(
                _mm256_mul_ps(x_first, cos_vec),
                _mm256_mul_ps(x_second, sin_vec));
            __m256 new_second = _mm256_add_ps(
                _mm256_mul_ps(x_first, sin_vec),
                _mm256_mul_ps(x_second, cos_vec));

            _mm256_storeu_ps(head_ptr + i, new_first);
            _mm256_storeu_ps(head_ptr + i + half_dim, new_second);
        }

        // Tail
        for (; i < half_dim; ++i)
        {
            float x_first = head_ptr[i];
            float x_second = head_ptr[i + half_dim];
            float cos_val = cos_cache[i];
            float sin_val = sin_cache[i];

            head_ptr[i] = x_first * cos_val - x_second * sin_val;
            head_ptr[i + half_dim] = x_first * sin_val + x_second * cos_val;
        }
    }
#endif

#if defined(__AVX512F__)
    /**
     * @brief Apply RoPE rotation using cached sin/cos (AVX512)
     */
    void apply_rope_to_head_cached_avx512(
        float *head_ptr,
        const float *cos_cache,
        const float *sin_cache,
        int head_dim)
    {
        const int half_dim = head_dim / 2;
        int i = 0;

        for (; i + 16 <= half_dim; i += 16)
        {
            __m512 x_first = _mm512_loadu_ps(head_ptr + i);
            __m512 x_second = _mm512_loadu_ps(head_ptr + i + half_dim);
            __m512 cos_vec = _mm512_loadu_ps(cos_cache + i);
            __m512 sin_vec = _mm512_loadu_ps(sin_cache + i);

            __m512 new_first = _mm512_sub_ps(
                _mm512_mul_ps(x_first, cos_vec),
                _mm512_mul_ps(x_second, sin_vec));
            __m512 new_second = _mm512_add_ps(
                _mm512_mul_ps(x_first, sin_vec),
                _mm512_mul_ps(x_second, cos_vec));

            _mm512_storeu_ps(head_ptr + i, new_first);
            _mm512_storeu_ps(head_ptr + i + half_dim, new_second);
        }

        // Tail
        for (; i < half_dim; ++i)
        {
            float x_first = head_ptr[i];
            float x_second = head_ptr[i + half_dim];
            float cos_val = cos_cache[i];
            float sin_val = sin_cache[i];

            head_ptr[i] = x_first * cos_val - x_second * sin_val;
            head_ptr[i + half_dim] = x_first * sin_val + x_second * cos_val;
        }
    }
#endif

    /**
     * @brief Apply RoPE rotation to a single head (scalar implementation)
     *
     * This is the reference implementation used for:
     * - Scalar-only builds
     * - Tail processing after vectorized loops
     * - Testing/validation
     */
    void apply_rope_to_head_scalar(
        float *head_ptr,
        int position,
        const std::vector<float> &inv_freq,
        int head_dim,
        int start_idx)
    {
        const int half_dim = head_dim / 2;

        for (int i = start_idx; i < half_dim; ++i)
        {
            float angle = position * inv_freq[i];
            float cos_val = std::cos(angle);
            float sin_val = std::sin(angle);

            float x_first = head_ptr[i];
            float x_second = head_ptr[i + half_dim];

            head_ptr[i] = x_first * cos_val - x_second * sin_val;
            head_ptr[i + half_dim] = x_first * sin_val + x_second * cos_val;
        }
    }

#if defined(__AVX2__)
    /**
     * @brief Apply RoPE rotation to a single head (AVX2 implementation)
     *
     * Processes 8 float pairs at a time using AVX2 intrinsics
     *
     * @return Number of pairs processed (always multiple of 8)
     */
    int apply_rope_to_head_avx2(
        float *head_ptr,
        int position,
        const std::vector<float> &inv_freq,
        int head_dim)
    {
        const int half_dim = head_dim / 2;
        int i = 0;

        // Process 8 pairs at a time
        for (; i + 8 <= half_dim; i += 8)
        {
            // Compute angles
            alignas(32) float angles[8];
            for (int lane = 0; lane < 8; ++lane)
            {
                angles[lane] = position * inv_freq[i + lane];
            }

            // Compute sin/cos
            alignas(32) float cos_vals[8];
            alignas(32) float sin_vals[8];
            for (int lane = 0; lane < 8; ++lane)
            {
                cos_vals[lane] = std::cos(angles[lane]);
                sin_vals[lane] = std::sin(angles[lane]);
            }

            // Load and rotate
            __m256 x_first = _mm256_loadu_ps(head_ptr + i);
            __m256 x_second = _mm256_loadu_ps(head_ptr + i + half_dim);
            __m256 cos_vec = _mm256_loadu_ps(cos_vals);
            __m256 sin_vec = _mm256_loadu_ps(sin_vals);

            __m256 new_first = _mm256_sub_ps(
                _mm256_mul_ps(x_first, cos_vec),
                _mm256_mul_ps(x_second, sin_vec));
            __m256 new_second = _mm256_add_ps(
                _mm256_mul_ps(x_first, sin_vec),
                _mm256_mul_ps(x_second, cos_vec));

            _mm256_storeu_ps(head_ptr + i, new_first);
            _mm256_storeu_ps(head_ptr + i + half_dim, new_second);
        }

        return i;
    }
#endif

#if defined(__AVX512F__)
    /**
     * @brief Apply RoPE rotation to a single head (AVX512 implementation)
     *
     * Processes 16 float pairs at a time using AVX512 intrinsics
     *
     * @return Number of pairs processed (always multiple of 16)
     */
    int apply_rope_to_head_avx512(
        float *head_ptr,
        int position,
        const std::vector<float> &inv_freq,
        int head_dim)
    {
        const int half_dim = head_dim / 2;
        int i = 0;

        // Process 16 pairs at a time
        for (; i + 16 <= half_dim; i += 16)
        {
            // Compute angles for 16 pairs
            alignas(64) float angles[16];
            for (int lane = 0; lane < 16; ++lane)
            {
                angles[lane] = position * inv_freq[i + lane];
            }

            // Compute sin/cos
            alignas(64) float cos_vals[16];
            alignas(64) float sin_vals[16];
            for (int lane = 0; lane < 16; ++lane)
            {
                cos_vals[lane] = std::cos(angles[lane]);
                sin_vals[lane] = std::sin(angles[lane]);
            }

            // Load values
            __m512 x_first = _mm512_loadu_ps(head_ptr + i);
            __m512 x_second = _mm512_loadu_ps(head_ptr + i + half_dim);
            __m512 cos_vec = _mm512_loadu_ps(cos_vals);
            __m512 sin_vec = _mm512_loadu_ps(sin_vals);

            // Rotate: new_first = x_first * cos - x_second * sin
            //         new_second = x_first * sin + x_second * cos
            __m512 new_first = _mm512_sub_ps(
                _mm512_mul_ps(x_first, cos_vec),
                _mm512_mul_ps(x_second, sin_vec));
            __m512 new_second = _mm512_add_ps(
                _mm512_mul_ps(x_first, sin_vec),
                _mm512_mul_ps(x_second, cos_vec));

            _mm512_storeu_ps(head_ptr + i, new_first);
            _mm512_storeu_ps(head_ptr + i + half_dim, new_second);
        }

        return i;
    }
#endif

    /**
     * @brief Apply RoPE rotation to a single head (dispatches to best available implementation)
     */
    static void apply_rope_to_head_vectorized(
        float *head_ptr,
        int position,
        const std::vector<float> &inv_freq,
        int head_dim)
    {
        int processed = 0;

#if defined(__AVX512F__)
        processed = apply_rope_to_head_avx512(head_ptr, position, inv_freq, head_dim);
#elif defined(__AVX2__)
        processed = apply_rope_to_head_avx2(head_ptr, position, inv_freq, head_dim);
#endif

        // Scalar tail
        apply_rope_to_head_scalar(head_ptr, position, inv_freq, head_dim, processed);
    }

    /**
     * @brief Apply RoPE to tensor with angle recurrence (prefill optimization)
     *
     * Uses pre-computed sin/cos tables generated via recurrence to avoid
     * expensive trigonometric calls in the hot loop.
     */
    static void apply_rope_to_tensor_recurrence(
        float *tensor,
        int seq_len, int num_heads, int head_dim,
        int n_past,
        const std::vector<float> &inv_freq)
    {
        const int half_dim = head_dim / 2;

        // 1. Pre-compute sin/cos tables for the whole sequence using recurrence
        // This avoids computing sin/cos for every head and every token repeatedly
        std::vector<float> cos_table(seq_len * half_dim);
        std::vector<float> sin_table(seq_len * half_dim);

        // Compute deltas (rotation per position step)
        std::vector<float> cos_delta(half_dim);
        std::vector<float> sin_delta(half_dim);
        for (int i = 0; i < half_dim; ++i)
        {
            cos_delta[i] = std::cos(inv_freq[i]);
            sin_delta[i] = std::sin(inv_freq[i]);
        }

        // Initialize first position (n_past)
        for (int i = 0; i < half_dim; ++i)
        {
            float ang = n_past * inv_freq[i];
            cos_table[i] = std::cos(ang);
            sin_table[i] = std::sin(ang);
        }

        // Recurrence for t > 0
        // This is serial but extremely fast (vectorizable by compiler)
        for (int t = 1; t < seq_len; ++t)
        {
            int prev_offset = (t - 1) * half_dim;
            int curr_offset = t * half_dim;

            for (int i = 0; i < half_dim; ++i)
            {
                float c = cos_table[prev_offset + i];
                float s = sin_table[prev_offset + i];
                float cd = cos_delta[i];
                float sd = sin_delta[i];

                cos_table[curr_offset + i] = c * cd - s * sd;
                sin_table[curr_offset + i] = s * cd + c * sd;
            }
        }

        // 2. Apply rotation in parallel using cached tables
        auto do_rope_work = [&]()
        {
#pragma omp for collapse(2) schedule(static)
            for (int t = 0; t < seq_len; ++t)
            {
                for (int h = 0; h < num_heads; ++h)
                {
                    float *head_ptr = tensor + (t * num_heads + h) * head_dim;
                    const float *cos_ptr = cos_table.data() + t * half_dim;
                    const float *sin_ptr = sin_table.data() + t * half_dim;

#if defined(__AVX512F__)
                    apply_rope_to_head_cached_avx512(head_ptr, cos_ptr, sin_ptr, head_dim);
#elif defined(__AVX2__)
                    apply_rope_to_head_cached_avx2(head_ptr, cos_ptr, sin_ptr, head_dim);
#else
                    // Fallback scalar implementation
                    for (int i = 0; i < half_dim; ++i)
                    {
                        float x_first = head_ptr[i];
                        float x_second = head_ptr[i + half_dim];
                        float cos_val = cos_ptr[i];
                        float sin_val = sin_ptr[i];

                        head_ptr[i] = x_first * cos_val - x_second * sin_val;
                        head_ptr[i + half_dim] = x_first * sin_val + x_second * cos_val;
                    }
#endif
                }
            }
        };
        OMP_WORKSHARE_REGION(do_rope_work);
    }

    void update_rope_cache(
        int head_dim, float freq_base, int target_pos,
        RoPEPersistentState &state)
    {
        // Reset state if parameters changed
        if (state.cached_head_dim != head_dim ||
            state.cached_freq_base != freq_base)
        {
            state.cached_head_dim = head_dim;
            state.cached_freq_base = freq_base;
            state.reset();
        }

        const auto &inv_freq = get_inv_freq_cached(head_dim, freq_base);
        const int half_dim = head_dim / 2;

        // Initialize or reset if position jumped
        if (state.last_pos == -1 || target_pos < state.last_pos)
        {
            state.cos_curr.resize(half_dim);
            state.sin_curr.resize(half_dim);
            state.cos_delta.resize(half_dim);
            state.sin_delta.resize(half_dim);

            for (int i = 0; i < half_dim; ++i)
            {
                const float delta = inv_freq[i];
                state.cos_delta[i] = std::cos(delta);
                state.sin_delta[i] = std::sin(delta);

                const float ang = target_pos * inv_freq[i];
                state.cos_curr[i] = std::cos(ang);
                state.sin_curr[i] = std::sin(ang);
            }
            state.last_pos = target_pos;
        }
        // Advance via complex recurrence
        else if (target_pos > state.last_pos)
        {
            int steps = target_pos - state.last_pos;
            for (int step = 0; step < steps; ++step)
            {
                for (int i = 0; i < half_dim; ++i)
                {
                    float c = state.cos_curr[i];
                    float s = state.sin_curr[i];
                    float cd = state.cos_delta[i];
                    float sd = state.sin_delta[i];

                    state.cos_curr[i] = c * cd - s * sd;
                    state.sin_curr[i] = s * cd + c * sd;
                }
            }
            state.last_pos = target_pos;
        }
    }

    /**
     * @brief Apply RoPE with persistent state (single-token decode optimization)
     */
    static void apply_rope_from_cache(
        float *q, float *k,
        int q_heads, int k_heads, int head_dim,
        const RoPEPersistentState &state)
    {
        const int half_dim = head_dim / 2;

        // Apply rotation to all heads using cached sin/cos
        auto apply_to_heads = [&](float *tensor, int num_heads)
        {
            for (int h = 0; h < num_heads; ++h)
            {
                float *head_ptr = tensor + h * head_dim;
#if defined(__AVX512F__)
                apply_rope_to_head_cached_avx512(head_ptr, state.cos_curr.data(), state.sin_curr.data(), head_dim);
#elif defined(__AVX2__)
                apply_rope_to_head_cached_avx2(head_ptr, state.cos_curr.data(), state.sin_curr.data(), head_dim);
#else
                for (int i = 0; i < half_dim; ++i)
                {
                    float x_first = head_ptr[i];
                    float x_second = head_ptr[i + half_dim];
                    float cos_val = state.cos_curr[i];
                    float sin_val = state.sin_curr[i];

                    head_ptr[i] = x_first * cos_val - x_second * sin_val;
                    head_ptr[i + half_dim] = x_first * sin_val + x_second * cos_val;
                }
#endif
            }
        };

        apply_to_heads(q, q_heads);
        if (k)
        {
            apply_to_heads(k, k_heads);
        }
    } // ============================================================================
    // Public API
    // ============================================================================

    void apply_rope_vectorized(
        float *q, float *k,
        int seq_len, int head_dim,
        int q_heads, int k_heads,
        int n_past, float freq_base,
        RoPEPersistentState *persistent_state)
    {
        if (head_dim % 2 != 0)
        {
            return; // head_dim must be even
        }

        const auto &inv_freq = get_inv_freq_cached(head_dim, freq_base);

        // Single-token decode with persistent state
        if (seq_len == 1 && persistent_state)
        {
            update_rope_cache(head_dim, freq_base, n_past, *persistent_state);

            apply_rope_from_cache(
                q, k, q_heads, k_heads, head_dim,
                *persistent_state);
        }
        else
        {
            // Prefill or no persistent state: use angle recurrence
            apply_rope_to_tensor_recurrence(q, seq_len, q_heads, head_dim, n_past, inv_freq);
            apply_rope_to_tensor_recurrence(k, seq_len, k_heads, head_dim, n_past, inv_freq);
        }
    }

    // ============================================================================
    // BF16 Native Precision Implementations
    // ============================================================================

#if defined(__AVX512F__)
    void apply_rope_to_head_cached_bf16_avx512(
        uint16_t *head_ptr,
        const float *cos_cache,
        const float *sin_cache,
        int head_dim)
    {
        const int half_dim = head_dim / 2;
        int i = 0;

        for (; i + 16 <= half_dim; i += 16)
        {
            // Load BF16
            __m256i bf16_first = _mm256_loadu_si256((__m256i *)(head_ptr + i));
            __m256i bf16_second = _mm256_loadu_si256((__m256i *)(head_ptr + i + half_dim));

            // Convert to FP32 (shift left 16)
            __m512i u32_first = _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_first), 16);
            __m512i u32_second = _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_second), 16);

            __m512 x_first = _mm512_castsi512_ps(u32_first);
            __m512 x_second = _mm512_castsi512_ps(u32_second);

            __m512 cos_vec = _mm512_loadu_ps(cos_cache + i);
            __m512 sin_vec = _mm512_loadu_ps(sin_cache + i);

            __m512 new_first = _mm512_sub_ps(
                _mm512_mul_ps(x_first, cos_vec),
                _mm512_mul_ps(x_second, sin_vec));
            __m512 new_second = _mm512_add_ps(
                _mm512_mul_ps(x_first, sin_vec),
                _mm512_mul_ps(x_second, cos_vec));

            // Convert back to BF16 (shift right 16)
            __m512i res_first = _mm512_srli_epi32(_mm512_castps_si512(new_first), 16);
            __m512i res_second = _mm512_srli_epi32(_mm512_castps_si512(new_second), 16);

            // Pack 32-bit integers to 16-bit integers
            __m256i out_first = _mm512_cvtepi32_epi16(res_first);
            __m256i out_second = _mm512_cvtepi32_epi16(res_second);

            _mm256_storeu_si256((__m256i *)(head_ptr + i), out_first);
            _mm256_storeu_si256((__m256i *)(head_ptr + i + half_dim), out_second);
        }

        // Tail
        for (; i < half_dim; ++i)
        {
            float x_first = simd::bf16_to_fp32(head_ptr[i]);
            float x_second = simd::bf16_to_fp32(head_ptr[i + half_dim]);
            float c = cos_cache[i];
            float s = sin_cache[i];
            float n1 = x_first * c - x_second * s;
            float n2 = x_first * s + x_second * c;
            head_ptr[i] = simd::fp32_to_bf16(n1);
            head_ptr[i + half_dim] = simd::fp32_to_bf16(n2);
        }
    }
#endif

#if defined(__AVX2__)
    void apply_rope_to_head_cached_bf16_avx2(
        uint16_t *head_ptr,
        const float *cos_cache,
        const float *sin_cache,
        int head_dim)
    {
        const int half_dim = head_dim / 2;
        int i = 0;

        for (; i + 8 <= half_dim; i += 8)
        {
            // Load 8 BF16 values (128 bits)
            __m128i bf16_first = _mm_loadu_si128((__m128i *)(head_ptr + i));
            __m128i bf16_second = _mm_loadu_si128((__m128i *)(head_ptr + i + half_dim));

            // Convert to FP32 (expand to 32-bit, shift left 16)
            __m256i u32_first = _mm256_slli_epi32(_mm256_cvtepu16_epi32(bf16_first), 16);
            __m256i u32_second = _mm256_slli_epi32(_mm256_cvtepu16_epi32(bf16_second), 16);

            __m256 x_first = _mm256_castsi256_ps(u32_first);
            __m256 x_second = _mm256_castsi256_ps(u32_second);

            __m256 cos_vec = _mm256_loadu_ps(cos_cache + i);
            __m256 sin_vec = _mm256_loadu_ps(sin_cache + i);

            __m256 new_first = _mm256_sub_ps(
                _mm256_mul_ps(x_first, cos_vec),
                _mm256_mul_ps(x_second, sin_vec));
            __m256 new_second = _mm256_add_ps(
                _mm256_mul_ps(x_first, sin_vec),
                _mm256_mul_ps(x_second, cos_vec));

            // Convert back (shift right 16)
            __m256i res_first = _mm256_srli_epi32(_mm256_castps_si256(new_first), 16);
            __m256i res_second = _mm256_srli_epi32(_mm256_castps_si256(new_second), 16);

            // Pack using packus and permute
            __m256i packed_first = _mm256_packus_epi32(res_first, _mm256_setzero_si256());
            __m256i permuted_first = _mm256_permute4x64_epi64(packed_first, _MM_SHUFFLE(3, 1, 2, 0));
            __m128i out_first = _mm256_castsi256_si128(permuted_first);

            __m256i packed_second = _mm256_packus_epi32(res_second, _mm256_setzero_si256());
            __m256i permuted_second = _mm256_permute4x64_epi64(packed_second, _MM_SHUFFLE(3, 1, 2, 0));
            __m128i out_second = _mm256_castsi256_si128(permuted_second);

            _mm_storeu_si128((__m128i *)(head_ptr + i), out_first);
            _mm_storeu_si128((__m128i *)(head_ptr + i + half_dim), out_second);
        }

        // Tail
        for (; i < half_dim; ++i)
        {
            float x_first = simd::bf16_to_fp32(head_ptr[i]);
            float x_second = simd::bf16_to_fp32(head_ptr[i + half_dim]);
            float c = cos_cache[i];
            float s = sin_cache[i];
            float n1 = x_first * c - x_second * s;
            float n2 = x_first * s + x_second * c;
            head_ptr[i] = simd::fp32_to_bf16(n1);
            head_ptr[i + half_dim] = simd::fp32_to_bf16(n2);
        }
    }
#endif

    void apply_rope_to_head_bf16_scalar(
        uint16_t *head_ptr,
        int position,
        const std::vector<float> &inv_freq,
        int head_dim,
        int start_idx)
    {
        const int half_dim = head_dim / 2;

        for (int i = start_idx; i < half_dim; ++i)
        {
            float angle = position * inv_freq[i];
            float cos_val = std::cos(angle);
            float sin_val = std::sin(angle);

            // Convert BF16 → FP32
            float x_first = simd::bf16_to_fp32(head_ptr[i]);
            float x_second = simd::bf16_to_fp32(head_ptr[i + half_dim]);

            // Rotate
            float new_first = x_first * cos_val - x_second * sin_val;
            float new_second = x_first * sin_val + x_second * cos_val;

            // Convert FP32 → BF16
            head_ptr[i] = simd::fp32_to_bf16(new_first);
            head_ptr[i + half_dim] = simd::fp32_to_bf16(new_second);
        }
    }

#if defined(__AVX2__)
    int apply_rope_to_head_bf16_avx2(
        uint16_t *head_ptr,
        int position,
        const std::vector<float> &inv_freq,
        int head_dim)
    {
        const int half_dim = head_dim / 2;
        int i = 0;

        // Process 8 pairs at a time
        for (; i + 8 <= half_dim; i += 8)
        {
            // Compute angles
            alignas(32) float angles[8];
            for (int lane = 0; lane < 8; ++lane)
            {
                angles[lane] = position * inv_freq[i + lane];
            }

            // Compute sin/cos
            alignas(32) float cos_vals[8];
            alignas(32) float sin_vals[8];
            for (int lane = 0; lane < 8; ++lane)
            {
                cos_vals[lane] = std::cos(angles[lane]);
                sin_vals[lane] = std::sin(angles[lane]);
            }

            // Load 8 BF16 values (128 bits)
            __m128i bf16_first = _mm_loadu_si128((__m128i *)(head_ptr + i));
            __m128i bf16_second = _mm_loadu_si128((__m128i *)(head_ptr + i + half_dim));

            // Convert to FP32 (expand to 32-bit, shift left 16)
            __m256i u32_first = _mm256_slli_epi32(_mm256_cvtepu16_epi32(bf16_first), 16);
            __m256i u32_second = _mm256_slli_epi32(_mm256_cvtepu16_epi32(bf16_second), 16);

            __m256 x_first_vec = _mm256_castsi256_ps(u32_first);
            __m256 x_second_vec = _mm256_castsi256_ps(u32_second);

            // Load sin/cos
            __m256 cos_vec = _mm256_loadu_ps(cos_vals);
            __m256 sin_vec = _mm256_loadu_ps(sin_vals);

            // Rotate
            __m256 new_first = _mm256_sub_ps(
                _mm256_mul_ps(x_first_vec, cos_vec),
                _mm256_mul_ps(x_second_vec, sin_vec));
            __m256 new_second = _mm256_add_ps(
                _mm256_mul_ps(x_first_vec, sin_vec),
                _mm256_mul_ps(x_second_vec, cos_vec));

            // Convert back (shift right 16)
            __m256i res_first = _mm256_srli_epi32(_mm256_castps_si256(new_first), 16);
            __m256i res_second = _mm256_srli_epi32(_mm256_castps_si256(new_second), 16);

            // Pack using packus and permute
            __m256i packed_first = _mm256_packus_epi32(res_first, _mm256_setzero_si256());
            __m256i permuted_first = _mm256_permute4x64_epi64(packed_first, _MM_SHUFFLE(3, 1, 2, 0));
            __m128i out_first = _mm256_castsi256_si128(permuted_first);

            __m256i packed_second = _mm256_packus_epi32(res_second, _mm256_setzero_si256());
            __m256i permuted_second = _mm256_permute4x64_epi64(packed_second, _MM_SHUFFLE(3, 1, 2, 0));
            __m128i out_second = _mm256_castsi256_si128(permuted_second);

            // Store back
            _mm_storeu_si128((__m128i *)(head_ptr + i), out_first);
            _mm_storeu_si128((__m128i *)(head_ptr + i + half_dim), out_second);
        }

        return i;
    }
#endif

#if defined(__AVX512F__)
    int apply_rope_to_head_bf16_avx512(
        uint16_t *head_ptr,
        int position,
        const std::vector<float> &inv_freq,
        int head_dim)
    {
        const int half_dim = head_dim / 2;
        int i = 0;

        // Process 16 pairs at a time
        for (; i + 16 <= half_dim; i += 16)
        {
            // Compute angles for 16 pairs
            alignas(64) float angles[16];
            for (int lane = 0; lane < 16; ++lane)
            {
                angles[lane] = position * inv_freq[i + lane];
            }

            // Compute sin/cos
            alignas(64) float cos_vals[16];
            alignas(64) float sin_vals[16];
            for (int lane = 0; lane < 16; ++lane)
            {
                cos_vals[lane] = std::cos(angles[lane]);
                sin_vals[lane] = std::sin(angles[lane]);
            }

            // Load BF16 values (16 values = 256 bits)
            __m256i bf16_first = _mm256_loadu_si256((__m256i *)(head_ptr + i));
            __m256i bf16_second = _mm256_loadu_si256((__m256i *)(head_ptr + i + half_dim));

            // Convert BF16 → FP32 (vectorized)
            // Shift left by 16 to zero-extend mantissa
            __m512i u32_first = _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_first), 16);
            __m512i u32_second = _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_second), 16);

            __m512 x_first = _mm512_castsi512_ps(u32_first);
            __m512 x_second = _mm512_castsi512_ps(u32_second);

            // Load sin/cos
            __m512 cos_vec = _mm512_loadu_ps(cos_vals);
            __m512 sin_vec = _mm512_loadu_ps(sin_vals);

            // Rotate
            __m512 new_first = _mm512_sub_ps(
                _mm512_mul_ps(x_first, cos_vec),
                _mm512_mul_ps(x_second, sin_vec));
            __m512 new_second = _mm512_add_ps(
                _mm512_mul_ps(x_first, sin_vec),
                _mm512_mul_ps(x_second, cos_vec));

            // Convert FP32 → BF16 (vectorized truncation)
            // Shift right by 16
            __m512i res_first = _mm512_srli_epi32(_mm512_castps_si512(new_first), 16);
            __m512i res_second = _mm512_srli_epi32(_mm512_castps_si512(new_second), 16);

            // Pack 32-bit integers to 16-bit integers
            __m256i out_first = _mm512_cvtepi32_epi16(res_first);
            __m256i out_second = _mm512_cvtepi32_epi16(res_second);

            // Store back
            _mm256_storeu_si256((__m256i *)(head_ptr + i), out_first);
            _mm256_storeu_si256((__m256i *)(head_ptr + i + half_dim), out_second);
        }

        return i;
    }
#endif

    static void apply_rope_to_head_bf16_vectorized(
        uint16_t *head_ptr,
        int position,
        const std::vector<float> &inv_freq,
        int head_dim)
    {
        int processed = 0;

#if defined(__AVX512F__)
        processed = apply_rope_to_head_bf16_avx512(head_ptr, position, inv_freq, head_dim);
#elif defined(__AVX2__)
        processed = apply_rope_to_head_bf16_avx2(head_ptr, position, inv_freq, head_dim);
#endif

        // Scalar tail
        apply_rope_to_head_bf16_scalar(head_ptr, position, inv_freq, head_dim, processed);
    }

    static void apply_rope_to_tensor_bf16_recurrence(
        uint16_t *tensor,
        int seq_len, int num_heads, int head_dim,
        int n_past,
        const std::vector<float> &inv_freq)
    {
        const int half_dim = head_dim / 2;

        // 1. Pre-compute sin/cos tables for the whole sequence using recurrence
        std::vector<float> cos_table(seq_len * half_dim);
        std::vector<float> sin_table(seq_len * half_dim);

        // Compute deltas
        std::vector<float> cos_delta(half_dim);
        std::vector<float> sin_delta(half_dim);
        for (int i = 0; i < half_dim; ++i)
        {
            cos_delta[i] = std::cos(inv_freq[i]);
            sin_delta[i] = std::sin(inv_freq[i]);
        }

        // Initialize first position
        for (int i = 0; i < half_dim; ++i)
        {
            float ang = n_past * inv_freq[i];
            cos_table[i] = std::cos(ang);
            sin_table[i] = std::sin(ang);
        }

        // Recurrence
        for (int t = 1; t < seq_len; ++t)
        {
            int prev_offset = (t - 1) * half_dim;
            int curr_offset = t * half_dim;

            for (int i = 0; i < half_dim; ++i)
            {
                float c = cos_table[prev_offset + i];
                float s = sin_table[prev_offset + i];
                float cd = cos_delta[i];
                float sd = sin_delta[i];

                cos_table[curr_offset + i] = c * cd - s * sd;
                sin_table[curr_offset + i] = s * cd + c * sd;
            }
        }

        // 2. Apply rotation in parallel using cached tables
        auto do_rope_bf16_work = [&]()
        {
#pragma omp for collapse(2) schedule(static)
            for (int t = 0; t < seq_len; ++t)
            {
                for (int h = 0; h < num_heads; ++h)
                {
                    uint16_t *head_ptr = tensor + (t * num_heads + h) * head_dim;
                    const float *cos_ptr = cos_table.data() + t * half_dim;
                    const float *sin_ptr = sin_table.data() + t * half_dim;

#if defined(__AVX512F__)
                    apply_rope_to_head_cached_bf16_avx512(head_ptr, cos_ptr, sin_ptr, head_dim);
#elif defined(__AVX2__)
                    apply_rope_to_head_cached_bf16_avx2(head_ptr, cos_ptr, sin_ptr, head_dim);
#else
                    // Fallback scalar implementation
                    for (int i = 0; i < half_dim; ++i)
                    {
                        float x_first = simd::bf16_to_fp32(head_ptr[i]);
                        float x_second = simd::bf16_to_fp32(head_ptr[i + half_dim]);
                        float cos_val = cos_ptr[i];
                        float sin_val = sin_ptr[i];

                        float n1 = x_first * cos_val - x_second * sin_val;
                        float n2 = x_first * sin_val + x_second * cos_val;

                        head_ptr[i] = simd::fp32_to_bf16(n1);
                        head_ptr[i + half_dim] = simd::fp32_to_bf16(n2);
                    }
#endif
                }
            }
        };
        OMP_WORKSHARE_REGION(do_rope_bf16_work);
    }

    static void apply_rope_bf16_from_cache(
        uint16_t *q, uint16_t *k,
        int q_heads, int k_heads, int head_dim,
        const RoPEPersistentState &state)
    {
        const int half_dim = head_dim / 2;

        auto apply_to_heads = [&](uint16_t *tensor, int num_heads)
        {
            for (int h = 0; h < num_heads; ++h)
            {
                uint16_t *head_ptr = tensor + h * head_dim;
#if defined(__AVX512F__)
                apply_rope_to_head_cached_bf16_avx512(head_ptr, state.cos_curr.data(), state.sin_curr.data(), head_dim);
#elif defined(__AVX2__)
                apply_rope_to_head_cached_bf16_avx2(head_ptr, state.cos_curr.data(), state.sin_curr.data(), head_dim);
#else
                for (int i = 0; i < half_dim; ++i)
                {
                    float x_first = simd::bf16_to_fp32(head_ptr[i]);
                    float x_second = simd::bf16_to_fp32(head_ptr[i + half_dim]);
                    float cos_val = state.cos_curr[i];
                    float sin_val = state.sin_curr[i];

                    float n1 = x_first * cos_val - x_second * sin_val;
                    float n2 = x_first * sin_val + x_second * cos_val;

                    head_ptr[i] = simd::fp32_to_bf16(n1);
                    head_ptr[i + half_dim] = simd::fp32_to_bf16(n2);
                }
#endif
            }
        };

        apply_to_heads(q, q_heads);
        if (k)
            apply_to_heads(k, k_heads);
    }

    static void apply_rope_to_tensor_bf16_direct(
        uint16_t *tensor,
        int seq_len, int num_heads, int head_dim,
        int n_past,
        const std::vector<float> &inv_freq)
    {
        // Avoid OpenMP overhead for single-token decode
        if (seq_len == 1)
        {
            for (int h = 0; h < num_heads; ++h)
            {
                uint16_t *head_ptr = tensor + h * head_dim;
                apply_rope_to_head_bf16_vectorized(head_ptr, n_past, inv_freq, head_dim);
            }
        }
        else
        {
            auto do_rope_direct_work = [&]()
            {
#pragma omp for collapse(2) schedule(static)
                for (int t = 0; t < seq_len; ++t)
                {
                    for (int h = 0; h < num_heads; ++h)
                    {
                        uint16_t *head_ptr = tensor + (t * num_heads + h) * head_dim;
                        apply_rope_to_head_bf16_vectorized(head_ptr, n_past + t, inv_freq, head_dim);
                    }
                }
            };
            OMP_WORKSHARE_REGION(do_rope_direct_work);
        }
    }

    void apply_rope_bf16(
        uint16_t *q_bf16, uint16_t *k_bf16,
        int seq_len, int head_dim,
        int q_heads, int k_heads,
        int n_past, float freq_base,
        RoPEPersistentState *persistent_state)
    {
        if (head_dim % 2 != 0)
        {
            return; // head_dim must be even
        }

        const auto &inv_freq = get_inv_freq_cached(head_dim, freq_base);

        // Single-token decode with persistent state optimization
        if (seq_len == 1 && persistent_state)
        {
            update_rope_cache(head_dim, freq_base, n_past, *persistent_state);
            apply_rope_bf16_from_cache(q_bf16, k_bf16, q_heads, k_heads, head_dim, *persistent_state);
            return;
        }

        // Use direct computation for short sequences to avoid recurrence overhead
        if (seq_len < 32)
        {
            apply_rope_to_tensor_bf16_direct(q_bf16, seq_len, q_heads, head_dim, n_past, inv_freq);
            apply_rope_to_tensor_bf16_direct(k_bf16, seq_len, k_heads, head_dim, n_past, inv_freq);
        }
        else
        {
            // Use recurrence optimization for longer sequences
            apply_rope_to_tensor_bf16_recurrence(q_bf16, seq_len, q_heads, head_dim, n_past, inv_freq);
            apply_rope_to_tensor_bf16_recurrence(k_bf16, seq_len, k_heads, head_dim, n_past, inv_freq);
        }
    }

    // ============================================================================
    // FP16 Native Precision Implementations
    // ============================================================================

#if defined(__AVX512F__)
    void apply_rope_to_head_cached_fp16_avx512(
        uint16_t *head_ptr,
        const float *cos_cache,
        const float *sin_cache,
        int head_dim)
    {
        const int half_dim = head_dim / 2;
        int i = 0;

        for (; i + 16 <= half_dim; i += 16)
        {
            // Load FP16 (16 values = 256 bits)
            __m256i fp16_first = _mm256_loadu_si256((__m256i *)(head_ptr + i));
            __m256i fp16_second = _mm256_loadu_si256((__m256i *)(head_ptr + i + half_dim));

            // Convert to FP32
            __m512 x_first = _mm512_cvtph_ps(fp16_first);
            __m512 x_second = _mm512_cvtph_ps(fp16_second);

            __m512 cos_vec = _mm512_loadu_ps(cos_cache + i);
            __m512 sin_vec = _mm512_loadu_ps(sin_cache + i);

            __m512 new_first = _mm512_sub_ps(
                _mm512_mul_ps(x_first, cos_vec),
                _mm512_mul_ps(x_second, sin_vec));
            __m512 new_second = _mm512_add_ps(
                _mm512_mul_ps(x_first, sin_vec),
                _mm512_mul_ps(x_second, cos_vec));

            // Convert back to FP16
            __m256i out_first = _mm512_cvtps_ph(new_first, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m256i out_second = _mm512_cvtps_ph(new_second, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

            _mm256_storeu_si256((__m256i *)(head_ptr + i), out_first);
            _mm256_storeu_si256((__m256i *)(head_ptr + i + half_dim), out_second);
        }

        // Tail
        for (; i < half_dim; ++i)
        {
            float x_first = simd::fp16_to_fp32(head_ptr[i]);
            float x_second = simd::fp16_to_fp32(head_ptr[i + half_dim]);
            float c = cos_cache[i];
            float s = sin_cache[i];
            float n1 = x_first * c - x_second * s;
            float n2 = x_first * s + x_second * c;
            head_ptr[i] = simd::fp32_to_fp16(n1);
            head_ptr[i + half_dim] = simd::fp32_to_fp16(n2);
        }
    }
#endif

#if defined(__AVX2__)
    void apply_rope_to_head_cached_fp16_avx2(
        uint16_t *head_ptr,
        const float *cos_cache,
        const float *sin_cache,
        int head_dim)
    {
        const int half_dim = head_dim / 2;
        int i = 0;

        for (; i + 8 <= half_dim; i += 8)
        {
            // Load FP16 (8 values = 128 bits)
            __m128i fp16_first = _mm_loadu_si128((__m128i *)(head_ptr + i));
            __m128i fp16_second = _mm_loadu_si128((__m128i *)(head_ptr + i + half_dim));

            // Convert to FP32
            __m256 x_first = _mm256_cvtph_ps(fp16_first);
            __m256 x_second = _mm256_cvtph_ps(fp16_second);

            __m256 cos_vec = _mm256_loadu_ps(cos_cache + i);
            __m256 sin_vec = _mm256_loadu_ps(sin_cache + i);

            __m256 new_first = _mm256_sub_ps(
                _mm256_mul_ps(x_first, cos_vec),
                _mm256_mul_ps(x_second, sin_vec));
            __m256 new_second = _mm256_add_ps(
                _mm256_mul_ps(x_first, sin_vec),
                _mm256_mul_ps(x_second, cos_vec));

            // Convert back to FP16
            __m128i out_first = _mm256_cvtps_ph(new_first, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m128i out_second = _mm256_cvtps_ph(new_second, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

            _mm_storeu_si128((__m128i *)(head_ptr + i), out_first);
            _mm_storeu_si128((__m128i *)(head_ptr + i + half_dim), out_second);
        }

        // Tail
        for (; i < half_dim; ++i)
        {
            float x_first = simd::fp16_to_fp32(head_ptr[i]);
            float x_second = simd::fp16_to_fp32(head_ptr[i + half_dim]);
            float c = cos_cache[i];
            float s = sin_cache[i];
            float n1 = x_first * c - x_second * s;
            float n2 = x_first * s + x_second * c;
            head_ptr[i] = simd::fp32_to_fp16(n1);
            head_ptr[i + half_dim] = simd::fp32_to_fp16(n2);
        }
    }
#endif

    void apply_rope_to_head_fp16_scalar(
        uint16_t *head_ptr,
        int position,
        const std::vector<float> &inv_freq,
        int head_dim,
        int start_idx)
    {
        const int half_dim = head_dim / 2;

        for (int i = start_idx; i < half_dim; ++i)
        {
            float angle = position * inv_freq[i];
            float cos_val = std::cos(angle);
            float sin_val = std::sin(angle);

            // Convert FP16 → FP32
            float x_first = fp16_to_fp32(head_ptr[i]);
            float x_second = fp16_to_fp32(head_ptr[i + half_dim]);

            // Rotate
            float new_first = x_first * cos_val - x_second * sin_val;
            float new_second = x_first * sin_val + x_second * cos_val;

            // Convert FP32 → FP16
            head_ptr[i] = fp32_to_fp16(new_first);
            head_ptr[i + half_dim] = fp32_to_fp16(new_second);
        }
    }

#if defined(__AVX2__)
    int apply_rope_to_head_fp16_avx2(
        uint16_t *head_ptr,
        int position,
        const std::vector<float> &inv_freq,
        int head_dim)
    {
        const int half_dim = head_dim / 2;
        int i = 0;

        // Process 8 pairs at a time
        for (; i + 8 <= half_dim; i += 8)
        {
            // Compute angles
            alignas(32) float angles[8];
            for (int lane = 0; lane < 8; ++lane)
            {
                angles[lane] = position * inv_freq[i + lane];
            }

            // Compute sin/cos
            alignas(32) float cos_vals[8];
            alignas(32) float sin_vals[8];
            for (int lane = 0; lane < 8; ++lane)
            {
                cos_vals[lane] = std::cos(angles[lane]);
                sin_vals[lane] = std::sin(angles[lane]);
            }

            // Load FP16 values
            alignas(32) uint16_t first_fp16[8];
            alignas(32) uint16_t second_fp16[8];
            std::memcpy(first_fp16, head_ptr + i, 8 * sizeof(uint16_t));
            std::memcpy(second_fp16, head_ptr + i + half_dim, 8 * sizeof(uint16_t));

            // Convert FP16 → FP32
            alignas(32) float x_first[8];
            alignas(32) float x_second[8];
            for (int lane = 0; lane < 8; ++lane)
            {
                x_first[lane] = fp16_to_fp32(first_fp16[lane]);
                x_second[lane] = fp16_to_fp32(second_fp16[lane]);
            }

            // Load and rotate
            __m256 x_first_vec = _mm256_loadu_ps(x_first);
            __m256 x_second_vec = _mm256_loadu_ps(x_second);
            __m256 cos_vec = _mm256_loadu_ps(cos_vals);
            __m256 sin_vec = _mm256_loadu_ps(sin_vals);

            __m256 new_first = _mm256_sub_ps(
                _mm256_mul_ps(x_first_vec, cos_vec),
                _mm256_mul_ps(x_second_vec, sin_vec));
            __m256 new_second = _mm256_add_ps(
                _mm256_mul_ps(x_first_vec, sin_vec),
                _mm256_mul_ps(x_second_vec, cos_vec));

            // Store back to FP32 buffers
            _mm256_storeu_ps(x_first, new_first);
            _mm256_storeu_ps(x_second, new_second);

            // Convert FP32 → FP16
            for (int lane = 0; lane < 8; ++lane)
            {
                first_fp16[lane] = fp32_to_fp16(x_first[lane]);
                second_fp16[lane] = fp32_to_fp16(x_second[lane]);
            }

            // Write back
            std::memcpy(head_ptr + i, first_fp16, 8 * sizeof(uint16_t));
            std::memcpy(head_ptr + i + half_dim, second_fp16, 8 * sizeof(uint16_t));
        }

        return i;
    }
#endif

#if defined(__AVX512F__)
    int apply_rope_to_head_fp16_avx512(
        uint16_t *head_ptr,
        int position,
        const std::vector<float> &inv_freq,
        int head_dim)
    {
        const int half_dim = head_dim / 2;
        int i = 0;

        // Process 16 pairs at a time
        for (; i + 16 <= half_dim; i += 16)
        {
            // Compute angles for 16 pairs
            alignas(64) float angles[16];
            for (int lane = 0; lane < 16; ++lane)
            {
                angles[lane] = position * inv_freq[i + lane];
            }

            // Compute sin/cos
            alignas(64) float cos_vals[16];
            alignas(64) float sin_vals[16];
            for (int lane = 0; lane < 16; ++lane)
            {
                cos_vals[lane] = std::cos(angles[lane]);
                sin_vals[lane] = std::sin(angles[lane]);
            }

            // Load FP16 values
            alignas(64) uint16_t first_fp16[16];
            alignas(64) uint16_t second_fp16[16];
            std::memcpy(first_fp16, head_ptr + i, 16 * sizeof(uint16_t));
            std::memcpy(second_fp16, head_ptr + i + half_dim, 16 * sizeof(uint16_t));

            // Convert FP16 → FP32
            alignas(64) float x_first[16];
            alignas(64) float x_second[16];
            for (int lane = 0; lane < 16; ++lane)
            {
                x_first[lane] = fp16_to_fp32(first_fp16[lane]);
                x_second[lane] = fp16_to_fp32(second_fp16[lane]);
            }

            // Load values
            __m512 x_first_vec = _mm512_loadu_ps(x_first);
            __m512 x_second_vec = _mm512_loadu_ps(x_second);
            __m512 cos_vec = _mm512_loadu_ps(cos_vals);
            __m512 sin_vec = _mm512_loadu_ps(sin_vals);

            // Rotate
            __m512 new_first = _mm512_sub_ps(
                _mm512_mul_ps(x_first_vec, cos_vec),
                _mm512_mul_ps(x_second_vec, sin_vec));
            __m512 new_second = _mm512_add_ps(
                _mm512_mul_ps(x_first_vec, sin_vec),
                _mm512_mul_ps(x_second_vec, cos_vec));

            // Store back to FP32 buffers
            _mm512_storeu_ps(x_first, new_first);
            _mm512_storeu_ps(x_second, new_second);

            // Convert FP32 → FP16
            for (int lane = 0; lane < 16; ++lane)
            {
                first_fp16[lane] = fp32_to_fp16(x_first[lane]);
                second_fp16[lane] = fp32_to_fp16(x_second[lane]);
            }

            // Write back
            std::memcpy(head_ptr + i, first_fp16, 16 * sizeof(uint16_t));
            std::memcpy(head_ptr + i + half_dim, second_fp16, 16 * sizeof(uint16_t));
        }

        return i;
    }
#endif

    static void apply_rope_to_head_fp16_vectorized(
        uint16_t *head_ptr,
        int position,
        const std::vector<float> &inv_freq,
        int head_dim)
    {
        int processed = 0;

#if defined(__AVX512F__)
        processed = apply_rope_to_head_fp16_avx512(head_ptr, position, inv_freq, head_dim);
#elif defined(__AVX2__)
        processed = apply_rope_to_head_fp16_avx2(head_ptr, position, inv_freq, head_dim);
#endif

        // Scalar tail
        apply_rope_to_head_fp16_scalar(head_ptr, position, inv_freq, head_dim, processed);
    }

    static void apply_rope_to_tensor_fp16_recurrence(
        uint16_t *tensor,
        int seq_len, int num_heads, int head_dim,
        int n_past,
        const std::vector<float> &inv_freq)
    {
        const int half_dim = head_dim / 2;

        // 1. Pre-compute sin/cos tables for the whole sequence using recurrence
        std::vector<float> cos_table(seq_len * half_dim);
        std::vector<float> sin_table(seq_len * half_dim);

        // Compute deltas
        std::vector<float> cos_delta(half_dim);
        std::vector<float> sin_delta(half_dim);
        for (int i = 0; i < half_dim; ++i)
        {
            cos_delta[i] = std::cos(inv_freq[i]);
            sin_delta[i] = std::sin(inv_freq[i]);
        }

        // Initialize first position
        for (int i = 0; i < half_dim; ++i)
        {
            float ang = n_past * inv_freq[i];
            cos_table[i] = std::cos(ang);
            sin_table[i] = std::sin(ang);
        }

        // Recurrence
        for (int t = 1; t < seq_len; ++t)
        {
            int prev_offset = (t - 1) * half_dim;
            int curr_offset = t * half_dim;

            for (int i = 0; i < half_dim; ++i)
            {
                float c = cos_table[prev_offset + i];
                float s = sin_table[prev_offset + i];
                float cd = cos_delta[i];
                float sd = sin_delta[i];

                cos_table[curr_offset + i] = c * cd - s * sd;
                sin_table[curr_offset + i] = s * cd + c * sd;
            }
        }

        // 2. Apply rotation in parallel using cached tables
        auto do_rope_fp16_work = [&]()
        {
#pragma omp for collapse(2) schedule(static)
            for (int t = 0; t < seq_len; ++t)
            {
                for (int h = 0; h < num_heads; ++h)
                {
                    uint16_t *head_ptr = tensor + (t * num_heads + h) * head_dim;
                    const float *cos_ptr = cos_table.data() + t * half_dim;
                    const float *sin_ptr = sin_table.data() + t * half_dim;

#if defined(__AVX512F__)
                    apply_rope_to_head_cached_fp16_avx512(head_ptr, cos_ptr, sin_ptr, head_dim);
#elif defined(__AVX2__)
                    apply_rope_to_head_cached_fp16_avx2(head_ptr, cos_ptr, sin_ptr, head_dim);
#else
                    // Fallback scalar implementation
                    for (int i = 0; i < half_dim; ++i)
                    {
                        float x_first = simd::fp16_to_fp32(head_ptr[i]);
                        float x_second = simd::fp16_to_fp32(head_ptr[i + half_dim]);
                        float cos_val = cos_ptr[i];
                        float sin_val = sin_ptr[i];

                        float n1 = x_first * cos_val - x_second * sin_val;
                        float n2 = x_first * sin_val + x_second * cos_val;

                        head_ptr[i] = simd::fp32_to_fp16(n1);
                        head_ptr[i + half_dim] = simd::fp32_to_fp16(n2);
                    }
#endif
                }
            }
        };
        OMP_WORKSHARE_REGION(do_rope_fp16_work);
    }

    static void apply_rope_fp16_from_cache(
        uint16_t *q, uint16_t *k,
        int q_heads, int k_heads, int head_dim,
        const RoPEPersistentState &state)
    {
        const int half_dim = head_dim / 2;

        auto apply_to_heads = [&](uint16_t *tensor, int num_heads)
        {
            for (int h = 0; h < num_heads; ++h)
            {
                uint16_t *head_ptr = tensor + h * head_dim;
#if defined(__AVX512F__)
                apply_rope_to_head_cached_fp16_avx512(head_ptr, state.cos_curr.data(), state.sin_curr.data(), head_dim);
#elif defined(__AVX2__)
                apply_rope_to_head_cached_fp16_avx2(head_ptr, state.cos_curr.data(), state.sin_curr.data(), head_dim);
#else
                for (int i = 0; i < half_dim; ++i)
                {
                    float x_first = simd::fp16_to_fp32(head_ptr[i]);
                    float x_second = simd::fp16_to_fp32(head_ptr[i + half_dim]);
                    float cos_val = state.cos_curr[i];
                    float sin_val = state.sin_curr[i];

                    float n1 = x_first * cos_val - x_second * sin_val;
                    float n2 = x_first * sin_val + x_second * cos_val;

                    head_ptr[i] = simd::fp32_to_fp16(n1);
                    head_ptr[i + half_dim] = simd::fp32_to_fp16(n2);
                }
#endif
            }
        };

        apply_to_heads(q, q_heads);
        if (k)
            apply_to_heads(k, k_heads);
    }

    static void apply_rope_to_tensor_fp16_direct(
        uint16_t *tensor,
        int seq_len, int num_heads, int head_dim,
        int n_past,
        const std::vector<float> &inv_freq)
    {
        // Avoid OpenMP overhead for single-token decode
        if (seq_len == 1)
        {
            for (int h = 0; h < num_heads; ++h)
            {
                uint16_t *head_ptr = tensor + h * head_dim;
                apply_rope_to_head_fp16_vectorized(head_ptr, n_past, inv_freq, head_dim);
            }
        }
        else
        {
            auto do_rope_fp16_direct_work = [&]()
            {
#pragma omp for collapse(2) schedule(static)
                for (int t = 0; t < seq_len; ++t)
                {
                    for (int h = 0; h < num_heads; ++h)
                    {
                        uint16_t *head_ptr = tensor + (t * num_heads + h) * head_dim;
                        apply_rope_to_head_fp16_vectorized(head_ptr, n_past + t, inv_freq, head_dim);
                    }
                }
            };
            OMP_WORKSHARE_REGION(do_rope_fp16_direct_work);
        }
    }

    void apply_rope_fp16(
        uint16_t *q_fp16, uint16_t *k_fp16,
        int seq_len, int head_dim,
        int q_heads, int k_heads,
        int n_past, float freq_base,
        RoPEPersistentState *persistent_state)
    {
        if (head_dim % 2 != 0)
        {
            return; // head_dim must be even
        }

        const auto &inv_freq = get_inv_freq_cached(head_dim, freq_base);

        // Single-token decode with persistent state optimization
        if (seq_len == 1 && persistent_state)
        {
            update_rope_cache(head_dim, freq_base, n_past, *persistent_state);
            apply_rope_fp16_from_cache(q_fp16, k_fp16, q_heads, k_heads, head_dim, *persistent_state);
            return;
        }

        // Use direct computation for short sequences to avoid recurrence overhead
        if (seq_len < 32)
        {
            apply_rope_to_tensor_fp16_direct(q_fp16, seq_len, q_heads, head_dim, n_past, inv_freq);
            apply_rope_to_tensor_fp16_direct(k_fp16, seq_len, k_heads, head_dim, n_past, inv_freq);
        }
        else
        {
            // Use recurrence optimization for longer sequences
            apply_rope_to_tensor_fp16_recurrence(q_fp16, seq_len, q_heads, head_dim, n_past, inv_freq);
            apply_rope_to_tensor_fp16_recurrence(k_fp16, seq_len, k_heads, head_dim, n_past, inv_freq);
        }
    }

    // ============================================================================
    // INT32 Stub (Not Supported)
    // ============================================================================

    bool apply_rope_int32(
        int32_t *q_int32, int32_t *k_int32,
        int seq_len, int head_dim,
        int q_heads, int k_heads,
        int n_past, float freq_base)
    {
        // RoPE is an activation operation and cannot be applied to quantized INT32 accumulators
        return false;
    }

    // ============================================================================
    // Q8_1 Pure-Integer RoPE Implementation
    // ============================================================================

    void compute_rope_sincos_q15(
        int position,
        const std::vector<float> &inv_freq,
        int head_dim,
        RoPESinCosQ15 &out)
    {
        const int half_dim = head_dim / 2;
        out.resize(half_dim);

        for (int i = 0; i < half_dim; ++i)
        {
            float angle = position * inv_freq[i];
            float cos_val, sin_val;
#if defined(__GNUC__)
            sincosf(angle, &sin_val, &cos_val);
#else
            cos_val = std::cos(angle);
            sin_val = std::sin(angle);
#endif
            // Convert to Q15: multiply by 32767 and clamp
            // Q15 range: [-32768, 32767] represents [-1.0, ~1.0)
            out.cos_q15[i] = static_cast<int16_t>(std::max(-32767.0f, std::min(32767.0f, cos_val * 32767.0f)));
            out.sin_q15[i] = static_cast<int16_t>(std::max(-32767.0f, std::min(32767.0f, sin_val * 32767.0f)));
        }
    }

    /**
     * @brief Helper: FP16 to FP32 conversion for Q8_1 scale
     */
    static inline float fp16_to_fp32_rope(uint16_t h)
    {
        // IEEE 754 half-precision to single-precision conversion
        uint32_t sign = (h >> 15) & 0x1;
        uint32_t exp = (h >> 10) & 0x1F;
        uint32_t mant = h & 0x3FF;

        if (exp == 0)
        {
            if (mant == 0)
            {
                // Zero
                uint32_t result = sign << 31;
                float f;
                std::memcpy(&f, &result, sizeof(float));
                return f;
            }
            else
            {
                // Subnormal
                while ((mant & 0x400) == 0)
                {
                    mant <<= 1;
                    exp--;
                }
                exp++;
                mant &= 0x3FF;
            }
        }
        else if (exp == 31)
        {
            // Inf or NaN
            uint32_t result = (sign << 31) | 0x7F800000 | (mant << 13);
            float f;
            std::memcpy(&f, &result, sizeof(float));
            return f;
        }

        exp = exp + (127 - 15);
        uint32_t result = (sign << 31) | (exp << 23) | (mant << 13);
        float f;
        std::memcpy(&f, &result, sizeof(float));
        return f;
    }

    /**
     * @brief Helper: FP32 to FP16 conversion for Q8_1 scale
     */
    // static inline uint16_t fp32_to_fp16_rope(float f) - REMOVED, use llaminar2::fp32_to_fp16

    void apply_rope_q8_1_integer_head(
        Q8_1Block *head_blocks,
        int blocks_per_head,
        const int16_t *cos_q15,
        const int16_t *sin_q15)
    {
        const int half_blocks = blocks_per_head / 2;

        // NOTE: We use FP32 for rotation to correctly handle scale differences between blocks.
        // The previous "pure integer" implementation assumed scaleA == scaleB, which is incorrect
        // and caused massive divergence (20x) in dot products.
        //
        // While this involves int8->fp32->int8 conversion, it is necessary for correctness.
        // Modern CPUs have fast int<->float conversion instructions.

        for (int b = 0; b < half_blocks; ++b)
        {
            Q8_1Block &blockA = head_blocks[b];
            Q8_1Block &blockB = head_blocks[b + half_blocks];

            float scaleA = fp16_to_fp32_rope(blockA.d);
            float scaleB = fp16_to_fp32_rope(blockB.d);

            const int16_t *cos_ptr = cos_q15 + b * 32;
            const int16_t *sin_ptr = sin_q15 + b * 32;

            float rotA[32];
            float rotB[32];

            // Compute rotated values in FP32
            // We unroll the loop slightly for better pipelining
            for (int i = 0; i < 32; ++i)
            {
                // Dequantize
                float x = static_cast<float>(blockA.qs[i]) * scaleA;
                float y = static_cast<float>(blockB.qs[i]) * scaleB;

                // Convert Q15 sin/cos to float
                float c = static_cast<float>(cos_ptr[i]) * (1.0f / 32767.0f);
                float s = static_cast<float>(sin_ptr[i]) * (1.0f / 32767.0f);

                // Rotate
                rotA[i] = x * c - y * s;
                rotB[i] = x * s + y * c;
            }

            // Requantize blockA
            {
                float max_val = 0.0f;
                for (int i = 0; i < 32; ++i)
                {
                    max_val = std::max(max_val, std::abs(rotA[i]));
                }

                float new_scale = max_val / 127.0f;
                if (new_scale < 1e-20f)
                    new_scale = 1e-20f; // Avoid div by zero
                float inv_scale = 1.0f / new_scale;

                int32_t sum_qs = 0;
                for (int i = 0; i < 32; ++i)
                {
                    int32_t q = static_cast<int32_t>(std::round(rotA[i] * inv_scale));
                    q = std::max(-127, std::min(127, q));
                    blockA.qs[i] = static_cast<int8_t>(q);
                    sum_qs += q;
                }
                blockA.d = llaminar2::fp32_to_fp16(new_scale);
                blockA.sum_qs = static_cast<int16_t>(sum_qs);
            }

            // Requantize blockB
            {
                float max_val = 0.0f;
                for (int i = 0; i < 32; ++i)
                {
                    max_val = std::max(max_val, std::abs(rotB[i]));
                }

                float new_scale = max_val / 127.0f;
                if (new_scale < 1e-20f)
                    new_scale = 1e-20f;
                float inv_scale = 1.0f / new_scale;

                int32_t sum_qs = 0;
                for (int i = 0; i < 32; ++i)
                {
                    int32_t q = static_cast<int32_t>(std::round(rotB[i] * inv_scale));
                    q = std::max(-127, std::min(127, q));
                    blockB.qs[i] = static_cast<int8_t>(q);
                    sum_qs += q;
                }
                blockB.d = llaminar2::fp32_to_fp16(new_scale);
                blockB.sum_qs = static_cast<int16_t>(sum_qs);
            }
        }
    }

    void apply_rope_q8_1_integer(
        Q8_1Block *Q,
        Q8_1Block *K,
        const int *position_ids,
        int seq_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        float rope_theta,
        RoPEPersistentState *persistent_state)
    {
        if (head_dim % 32 != 0)
        {
            return; // head_dim must be divisible by Q8_1 block size
        }

        const int blocks_per_head = head_dim / 32;
        const int q_stride_blocks = n_heads * blocks_per_head;
        const int k_stride_blocks = n_kv_heads * blocks_per_head;
        const int half_dim = head_dim / 2;

        // Get inverse frequencies
        const auto &inv_freq = get_inv_freq_cached(head_dim, rope_theta);

        // Precompute sin/cos tables (Q15)
        // Use thread_local to avoid allocation overhead for small batches (decode)
        thread_local std::vector<int16_t> tls_cos_table;
        thread_local std::vector<int16_t> tls_sin_table;

        size_t required_size = seq_len * half_dim;
        if (tls_cos_table.size() < required_size)
        {
            tls_cos_table.resize(required_size);
            tls_sin_table.resize(required_size);
        }

        int16_t *cos_table = tls_cos_table.data();
        int16_t *sin_table = tls_sin_table.data();

        // Optimization for single-token decode with persistent state
        if (seq_len == 1 && persistent_state)
        {
            int pos = position_ids ? position_ids[0] : 0;
            if (pos >= 0)
            {
                // Update persistent state (uses recurrence)
                update_rope_cache(head_dim, rope_theta, pos, *persistent_state);

                // Convert cached floats to Q15 integers
                const float *cos_curr = persistent_state->cos_curr.data();
                const float *sin_curr = persistent_state->sin_curr.data();

                for (int i = 0; i < half_dim; ++i)
                {
                    cos_table[i] = (int16_t)(cos_curr[i] * 32767.0f);
                    sin_table[i] = (int16_t)(sin_curr[i] * 32767.0f);
                }
            }
        }
        else
        {
            // Check for contiguous positions
            bool contiguous = true;
            int start_pos = 0;
            if (position_ids)
            {
                start_pos = position_ids[0];
                if (seq_len > 1)
                {
                    for (int i = 1; i < seq_len; ++i)
                    {
                        if (position_ids[i] != start_pos + i)
                        {
                            contiguous = false;
                            break;
                        }
                    }
                }
            }
            else
            {
                start_pos = 0;
            }

            // Generate tables
            if (contiguous && seq_len > 1)
            {
                // Use recurrence for sequential positions
                std::vector<float> cos_delta(half_dim);
                std::vector<float> sin_delta(half_dim);
                for (int i = 0; i < half_dim; ++i)
                {
                    cos_delta[i] = std::cos(inv_freq[i]);
                    sin_delta[i] = std::sin(inv_freq[i]);
                }

                // Init first row
                std::vector<float> curr_cos(half_dim);
                std::vector<float> curr_sin(half_dim);
                for (int i = 0; i < half_dim; ++i)
                {
                    float ang = start_pos * inv_freq[i];
                    curr_cos[i] = std::cos(ang);
                    curr_sin[i] = std::sin(ang);

                    cos_table[i] = (int16_t)(curr_cos[i] * 32767.0f);
                    sin_table[i] = (int16_t)(curr_sin[i] * 32767.0f);
                }

                // Recurrence
                for (int t = 1; t < seq_len; ++t)
                {
                    int offset = t * half_dim;
                    for (int i = 0; i < half_dim; ++i)
                    {
                        float c = curr_cos[i];
                        float s = curr_sin[i];
                        float cd = cos_delta[i];
                        float sd = sin_delta[i];

                        float nc = c * cd - s * sd;
                        float ns = s * cd + c * sd;

                        curr_cos[i] = nc;
                        curr_sin[i] = ns;

                        cos_table[offset + i] = (int16_t)(nc * 32767.0f);
                        sin_table[offset + i] = (int16_t)(ns * 32767.0f);
                    }
                }
            }
            else
            {
                // Parallel compute for non-contiguous or single token (without persistent state)
                auto do_cos_sin_compute = [&]()
                {
#pragma omp for
                    for (int t = 0; t < seq_len; ++t)
                    {
                        int pos = position_ids ? position_ids[t] : t;
                        int offset = t * half_dim;
                        for (int i = 0; i < half_dim; ++i)
                        {
                            float ang = pos * inv_freq[i];
                            float c = std::cos(ang);
                            float s = std::sin(ang);

                            cos_table[offset + i] = (int16_t)(c * 32767.0f);
                            sin_table[offset + i] = (int16_t)(s * 32767.0f);
                        }
                    }
                };
                if (seq_len > 1)
                {
                    OMP_WORKSHARE_REGION(do_cos_sin_compute);
                }
                else
                {
                    do_cos_sin_compute();
                }
            }
        }

        // Apply RoPE using precomputed tables
        // Collapse heads and tokens for better load balancing
        if (seq_len == 1)
        {
            // Serial execution for single token (decode optimization)
            int offset = 0; // t=0
            const int16_t *c_ptr = cos_table + offset;
            const int16_t *s_ptr = sin_table + offset;

            // Apply to Q
            for (int h = 0; h < n_heads; ++h)
            {
                Q8_1Block *head_ptr = Q + h * blocks_per_head;
                apply_rope_q8_1_integer_head(head_ptr, blocks_per_head, c_ptr, s_ptr);
            }

            // Apply to K
            if (K)
            {
                for (int h = 0; h < n_kv_heads; ++h)
                {
                    Q8_1Block *head_ptr = K + h * blocks_per_head;
                    apply_rope_q8_1_integer_head(head_ptr, blocks_per_head, c_ptr, s_ptr);
                }
            }
        }
        else
        {
            auto do_apply_q_work = [&]()
            {
#pragma omp for collapse(2)
                for (int t = 0; t < seq_len; ++t)
                {
                    for (int h = 0; h < n_heads; ++h)
                    {
                        Q8_1Block *head_ptr = Q + t * q_stride_blocks + h * blocks_per_head;
                        const int16_t *c_ptr = cos_table + t * half_dim;
                        const int16_t *s_ptr = sin_table + t * half_dim;
                        apply_rope_q8_1_integer_head(head_ptr, blocks_per_head, c_ptr, s_ptr);
                    }
                }
            };
            OMP_WORKSHARE_REGION(do_apply_q_work);

            if (K)
            {
                auto do_apply_k_work = [&]()
                {
#pragma omp for collapse(2)
                    for (int t = 0; t < seq_len; ++t)
                    {
                        for (int h = 0; h < n_kv_heads; ++h)
                        {
                            Q8_1Block *head_ptr = K + t * k_stride_blocks + h * blocks_per_head;
                            const int16_t *c_ptr = cos_table + t * half_dim;
                            const int16_t *s_ptr = sin_table + t * half_dim;
                            apply_rope_q8_1_integer_head(head_ptr, blocks_per_head, c_ptr, s_ptr);
                        }
                    }
                };
                OMP_WORKSHARE_REGION(do_apply_k_work);
            }
        }
    }

} // namespace llaminar2::primitives
