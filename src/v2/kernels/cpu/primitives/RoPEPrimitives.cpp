/**
 * @file RoPEPrimitives.cpp
 * @brief Vectorized RoPE implementation (ported from V1)
 * @author David Sanftenberg
 */

#include "RoPEPrimitives.h"
#include "../../../tensors/SIMDHelpers.h"
#include "../../../tensors/FP16Utils.h"
#include "../../../utils/Logger.h"
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

    /**
     * @brief Reference/scalar implementation of Q8_1 RoPE for one head
     *
     * This is the baseline implementation. SIMD versions are used when available.
     */
    static void apply_rope_q8_1_integer_head_ref(
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

    // ============================================================================
    // Q8_1 RoPE SIMD Implementations
    // ============================================================================

    // Scalar reference implementation (also used as fallback)
    void apply_rope_q8_1_integer_head_scalar(
        Q8_1Block *head_blocks,
        int blocks_per_head,
        const int16_t *cos_q15,
        const int16_t *sin_q15)
    {
        // Delegate to the reference implementation
        apply_rope_q8_1_integer_head_ref(head_blocks, blocks_per_head, cos_q15, sin_q15);
    }

#if defined(__AVX2__)
    void apply_rope_q8_1_integer_head_avx2(
        Q8_1Block *head_blocks,
        int blocks_per_head,
        const int16_t *cos_q15,
        const int16_t *sin_q15)
    {
        const int half_blocks = blocks_per_head / 2;
        const __m256 q15_scale = _mm256_set1_ps(1.0f / 32767.0f);
        const __m256i clamp_min = _mm256_set1_epi32(-127);
        const __m256i clamp_max = _mm256_set1_epi32(127);

        for (int b = 0; b < half_blocks; ++b)
        {
            Q8_1Block &blockA = head_blocks[b];
            Q8_1Block &blockB = head_blocks[b + half_blocks];

            // Prefetch next block pair
            if (b + 1 < half_blocks)
            {
                _mm_prefetch(reinterpret_cast<const char *>(&head_blocks[b + 1]), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char *>(&head_blocks[b + 1 + half_blocks]), _MM_HINT_T0);
            }

            const __m256 scaleA = _mm256_set1_ps(fp16_to_fp32_rope(blockA.d));
            const __m256 scaleB = _mm256_set1_ps(fp16_to_fp32_rope(blockB.d));
            const int16_t *cos_ptr = cos_q15 + b * 32;
            const int16_t *sin_ptr = sin_q15 + b * 32;

            // ================================================================
            // PHASE 1: Dequant + Rotate - Fully unrolled, interleaved loads
            // Process all 32 elements in 4 chunks of 8
            // Q8_1 uses int8 storage, so we load 8 bytes and sign-extend
            // ================================================================

            // Load all int8 data and sign-extend to int32
            // Use _mm_cvtepi8_epi32 which takes the low 4 bytes of xmm

            // Chunk 0: elements [0:8)
            __m128i qa0_i8 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(blockA.qs));
            __m128i qb0_i8 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(blockB.qs));
            __m128i cos0_i16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(cos_ptr));
            __m128i sin0_i16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(sin_ptr));

            // Chunk 1: elements [8:16)
            __m128i qa1_i8 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(blockA.qs + 8));
            __m128i qb1_i8 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(blockB.qs + 8));
            __m128i cos1_i16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(cos_ptr + 8));
            __m128i sin1_i16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(sin_ptr + 8));

            // Chunk 2: elements [16:24)
            __m128i qa2_i8 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(blockA.qs + 16));
            __m128i qb2_i8 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(blockB.qs + 16));
            __m128i cos2_i16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(cos_ptr + 16));
            __m128i sin2_i16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(sin_ptr + 16));

            // Chunk 3: elements [24:32)
            __m128i qa3_i8 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(blockA.qs + 24));
            __m128i qb3_i8 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(blockB.qs + 24));
            __m128i cos3_i16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(cos_ptr + 24));
            __m128i sin3_i16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(sin_ptr + 24));

            // Convert chunk 0: int8 → int32 → float, then rotate
            // For int8: first extend to int32 using _mm256_cvtepi8_epi32
            __m256 x0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(qa0_i8)), scaleA);
            __m256 y0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(qb0_i8)), scaleB);
            __m256 c0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(cos0_i16)), q15_scale);
            __m256 s0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(sin0_i16)), q15_scale);
            __m256 rotA0 = _mm256_fmsub_ps(x0, c0, _mm256_mul_ps(y0, s0));
            __m256 rotB0 = _mm256_fmadd_ps(x0, s0, _mm256_mul_ps(y0, c0));

            // Convert chunk 1
            __m256 x1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(qa1_i8)), scaleA);
            __m256 y1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(qb1_i8)), scaleB);
            __m256 c1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(cos1_i16)), q15_scale);
            __m256 s1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(sin1_i16)), q15_scale);
            __m256 rotA1 = _mm256_fmsub_ps(x1, c1, _mm256_mul_ps(y1, s1));
            __m256 rotB1 = _mm256_fmadd_ps(x1, s1, _mm256_mul_ps(y1, c1));

            // Convert chunk 2
            __m256 x2 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(qa2_i8)), scaleA);
            __m256 y2 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(qb2_i8)), scaleB);
            __m256 c2 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(cos2_i16)), q15_scale);
            __m256 s2 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(sin2_i16)), q15_scale);
            __m256 rotA2 = _mm256_fmsub_ps(x2, c2, _mm256_mul_ps(y2, s2));
            __m256 rotB2 = _mm256_fmadd_ps(x2, s2, _mm256_mul_ps(y2, c2));

            // Convert chunk 3
            __m256 x3 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(qa3_i8)), scaleA);
            __m256 y3 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(qb3_i8)), scaleB);
            __m256 c3 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(cos3_i16)), q15_scale);
            __m256 s3 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(sin3_i16)), q15_scale);
            __m256 rotA3 = _mm256_fmsub_ps(x3, c3, _mm256_mul_ps(y3, s3));
            __m256 rotB3 = _mm256_fmadd_ps(x3, s3, _mm256_mul_ps(y3, c3));

            // ================================================================
            // PHASE 2 & 3: Requantize blockA and blockB INTERLEAVED
            // ================================================================

            // Compute abs and max (interleaved)
            __m256 sign_mask = _mm256_set1_ps(-0.0f);
            __m256 absA0 = _mm256_andnot_ps(sign_mask, rotA0);
            __m256 absB0 = _mm256_andnot_ps(sign_mask, rotB0);
            __m256 absA1 = _mm256_andnot_ps(sign_mask, rotA1);
            __m256 absB1 = _mm256_andnot_ps(sign_mask, rotB1);
            __m256 absA2 = _mm256_andnot_ps(sign_mask, rotA2);
            __m256 absB2 = _mm256_andnot_ps(sign_mask, rotB2);
            __m256 absA3 = _mm256_andnot_ps(sign_mask, rotA3);
            __m256 absB3 = _mm256_andnot_ps(sign_mask, rotB3);

            // Horizontal max reduction
            __m256 maxA_01 = _mm256_max_ps(absA0, absA1);
            __m256 maxB_01 = _mm256_max_ps(absB0, absB1);
            __m256 maxA_23 = _mm256_max_ps(absA2, absA3);
            __m256 maxB_23 = _mm256_max_ps(absB2, absB3);
            __m256 maxA_all = _mm256_max_ps(maxA_01, maxA_23);
            __m256 maxB_all = _mm256_max_ps(maxB_01, maxB_23);

            // Reduce to scalar
            __m128 maxA_lo = _mm256_castps256_ps128(maxA_all);
            __m128 maxA_hi = _mm256_extractf128_ps(maxA_all, 1);
            __m128 maxA_128 = _mm_max_ps(maxA_lo, maxA_hi);
            maxA_128 = _mm_max_ps(maxA_128, _mm_shuffle_ps(maxA_128, maxA_128, _MM_SHUFFLE(1, 0, 3, 2)));
            maxA_128 = _mm_max_ps(maxA_128, _mm_shuffle_ps(maxA_128, maxA_128, _MM_SHUFFLE(0, 0, 0, 1)));
            float maxA_val = _mm_cvtss_f32(maxA_128);

            __m128 maxB_lo = _mm256_castps256_ps128(maxB_all);
            __m128 maxB_hi = _mm256_extractf128_ps(maxB_all, 1);
            __m128 maxB_128 = _mm_max_ps(maxB_lo, maxB_hi);
            maxB_128 = _mm_max_ps(maxB_128, _mm_shuffle_ps(maxB_128, maxB_128, _MM_SHUFFLE(1, 0, 3, 2)));
            maxB_128 = _mm_max_ps(maxB_128, _mm_shuffle_ps(maxB_128, maxB_128, _MM_SHUFFLE(0, 0, 0, 1)));
            float maxB_val = _mm_cvtss_f32(maxB_128);

            // Compute scales
            float scaleA_new = maxA_val * (1.0f / 127.0f);
            float scaleB_new = maxB_val * (1.0f / 127.0f);
            if (scaleA_new < 1e-20f)
                scaleA_new = 1e-20f;
            if (scaleB_new < 1e-20f)
                scaleB_new = 1e-20f;

            __m256 invScaleA = _mm256_set1_ps(1.0f / scaleA_new);
            __m256 invScaleB = _mm256_set1_ps(1.0f / scaleB_new);

            // Scale, round, clamp (interleaved)
            __m256 scaledA0 = _mm256_mul_ps(rotA0, invScaleA);
            __m256 scaledB0 = _mm256_mul_ps(rotB0, invScaleB);
            __m256 scaledA1 = _mm256_mul_ps(rotA1, invScaleA);
            __m256 scaledB1 = _mm256_mul_ps(rotB1, invScaleB);
            __m256 scaledA2 = _mm256_mul_ps(rotA2, invScaleA);
            __m256 scaledB2 = _mm256_mul_ps(rotB2, invScaleB);
            __m256 scaledA3 = _mm256_mul_ps(rotA3, invScaleA);
            __m256 scaledB3 = _mm256_mul_ps(rotB3, invScaleB);

            __m256i qA0 = _mm256_cvtps_epi32(_mm256_round_ps(scaledA0, _MM_FROUND_TO_NEAREST_INT));
            __m256i qB0 = _mm256_cvtps_epi32(_mm256_round_ps(scaledB0, _MM_FROUND_TO_NEAREST_INT));
            __m256i qA1 = _mm256_cvtps_epi32(_mm256_round_ps(scaledA1, _MM_FROUND_TO_NEAREST_INT));
            __m256i qB1 = _mm256_cvtps_epi32(_mm256_round_ps(scaledB1, _MM_FROUND_TO_NEAREST_INT));
            __m256i qA2 = _mm256_cvtps_epi32(_mm256_round_ps(scaledA2, _MM_FROUND_TO_NEAREST_INT));
            __m256i qB2 = _mm256_cvtps_epi32(_mm256_round_ps(scaledB2, _MM_FROUND_TO_NEAREST_INT));
            __m256i qA3 = _mm256_cvtps_epi32(_mm256_round_ps(scaledA3, _MM_FROUND_TO_NEAREST_INT));
            __m256i qB3 = _mm256_cvtps_epi32(_mm256_round_ps(scaledB3, _MM_FROUND_TO_NEAREST_INT));

            // Clamp
            qA0 = _mm256_max_epi32(_mm256_min_epi32(qA0, clamp_max), clamp_min);
            qB0 = _mm256_max_epi32(_mm256_min_epi32(qB0, clamp_max), clamp_min);
            qA1 = _mm256_max_epi32(_mm256_min_epi32(qA1, clamp_max), clamp_min);
            qB1 = _mm256_max_epi32(_mm256_min_epi32(qB1, clamp_max), clamp_min);
            qA2 = _mm256_max_epi32(_mm256_min_epi32(qA2, clamp_max), clamp_min);
            qB2 = _mm256_max_epi32(_mm256_min_epi32(qB2, clamp_max), clamp_min);
            qA3 = _mm256_max_epi32(_mm256_min_epi32(qA3, clamp_max), clamp_min);
            qB3 = _mm256_max_epi32(_mm256_min_epi32(qB3, clamp_max), clamp_min);

            // Pack int32 → int16 → int8 and store
            // _mm256_packs_epi32 packs 8 int32 to 8 int16 (saturating)
            // _mm256_packs_epi16 packs 16 int16 to 16 int8 (saturating)
            // But the lane ordering is tricky with AVX2, so we use a different approach

            // Pack chunks 0+1 together, then 2+3
            __m256i qA01_16 = _mm256_packs_epi32(qA0, qA1); // [A0_lo, A1_lo, A0_hi, A1_hi] as int16
            __m256i qA23_16 = _mm256_packs_epi32(qA2, qA3);
            __m256i qB01_16 = _mm256_packs_epi32(qB0, qB1);
            __m256i qB23_16 = _mm256_packs_epi32(qB2, qB3);

            // Pack int16 → int8
            __m256i qA_8 = _mm256_packs_epi16(qA01_16, qA23_16); // Mixed order due to lane crossing
            __m256i qB_8 = _mm256_packs_epi16(qB01_16, qB23_16);

            // Fix the lane ordering: AVX2 packs work within 128-bit lanes
            // After packs: [0-3, 8-11, 16-19, 24-27 | 4-7, 12-15, 20-23, 28-31]
            // We need: [0-7, 8-15, 16-23, 24-31]
            // Use permute to fix: vpermd with indices [0,4,1,5,2,6,3,7]
            const __m256i fix_order = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
            qA_8 = _mm256_permutevar8x32_epi32(qA_8, fix_order);
            qB_8 = _mm256_permutevar8x32_epi32(qB_8, fix_order);

            _mm256_storeu_si256(reinterpret_cast<__m256i *>(blockA.qs), qA_8);
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(blockB.qs), qB_8);

            // Compute sum_qs
            __m256i sumA_01 = _mm256_add_epi32(qA0, qA1);
            __m256i sumA_23 = _mm256_add_epi32(qA2, qA3);
            __m256i sumA_all = _mm256_add_epi32(sumA_01, sumA_23);
            __m128i sumA_lo = _mm256_castsi256_si128(sumA_all);
            __m128i sumA_hi = _mm256_extracti128_si256(sumA_all, 1);
            __m128i sumA_128 = _mm_add_epi32(sumA_lo, sumA_hi);
            sumA_128 = _mm_add_epi32(sumA_128, _mm_shuffle_epi32(sumA_128, _MM_SHUFFLE(1, 0, 3, 2)));
            sumA_128 = _mm_add_epi32(sumA_128, _mm_shuffle_epi32(sumA_128, _MM_SHUFFLE(0, 0, 0, 1)));

            __m256i sumB_01 = _mm256_add_epi32(qB0, qB1);
            __m256i sumB_23 = _mm256_add_epi32(qB2, qB3);
            __m256i sumB_all = _mm256_add_epi32(sumB_01, sumB_23);
            __m128i sumB_lo = _mm256_castsi256_si128(sumB_all);
            __m128i sumB_hi = _mm256_extracti128_si256(sumB_all, 1);
            __m128i sumB_128 = _mm_add_epi32(sumB_lo, sumB_hi);
            sumB_128 = _mm_add_epi32(sumB_128, _mm_shuffle_epi32(sumB_128, _MM_SHUFFLE(1, 0, 3, 2)));
            sumB_128 = _mm_add_epi32(sumB_128, _mm_shuffle_epi32(sumB_128, _MM_SHUFFLE(0, 0, 0, 1)));

            blockA.d = llaminar2::fp32_to_fp16(scaleA_new);
            blockB.d = llaminar2::fp32_to_fp16(scaleB_new);
            blockA.sum_qs = static_cast<int16_t>(_mm_cvtsi128_si32(sumA_128));
            blockB.sum_qs = static_cast<int16_t>(_mm_cvtsi128_si32(sumB_128));
        }
    }
#endif // __AVX2__

#if defined(__AVX512F__)
    void apply_rope_q8_1_integer_head_avx512(
        Q8_1Block *head_blocks,
        int blocks_per_head,
        const int16_t *cos_q15,
        const int16_t *sin_q15)
    {
        const int half_blocks = blocks_per_head / 2;
        const __m512 q15_scale = _mm512_set1_ps(1.0f / 32767.0f);
        const __m512i clamp_min = _mm512_set1_epi32(-127);
        const __m512i clamp_max = _mm512_set1_epi32(127);

        for (int b = 0; b < half_blocks; ++b)
        {
            Q8_1Block &blockA = head_blocks[b];
            Q8_1Block &blockB = head_blocks[b + half_blocks];

            // Prefetch next block pair
            if (b + 1 < half_blocks)
            {
                _mm_prefetch(reinterpret_cast<const char *>(&head_blocks[b + 1]), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char *>(&head_blocks[b + 1 + half_blocks]), _MM_HINT_T0);
            }

            const __m512 scaleA = _mm512_set1_ps(fp16_to_fp32_rope(blockA.d));
            const __m512 scaleB = _mm512_set1_ps(fp16_to_fp32_rope(blockB.d));
            const int16_t *cos_ptr = cos_q15 + b * 32;
            const int16_t *sin_ptr = sin_q15 + b * 32;

            // ================================================================
            // PHASE 1: Dequant + Rotate - Process 32 elements in 2 chunks of 16
            // ================================================================

            // Load 16 int8 values each
            __m128i qa0_i8 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(blockA.qs));
            __m128i qb0_i8 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(blockB.qs));
            __m128i qa1_i8 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(blockA.qs + 16));
            __m128i qb1_i8 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(blockB.qs + 16));

            __m256i cos0_i16 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(cos_ptr));
            __m256i sin0_i16 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(sin_ptr));
            __m256i cos1_i16 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(cos_ptr + 16));
            __m256i sin1_i16 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(sin_ptr + 16));

            // Convert int8 → int32 → float (AVX512 has _mm512_cvtepi8_epi32 for 16 elements)
            __m512 x0 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(qa0_i8)), scaleA);
            __m512 y0 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(qb0_i8)), scaleB);
            __m512 c0 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(cos0_i16)), q15_scale);
            __m512 s0 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(sin0_i16)), q15_scale);
            __m512 rotA0 = _mm512_fmsub_ps(x0, c0, _mm512_mul_ps(y0, s0));
            __m512 rotB0 = _mm512_fmadd_ps(x0, s0, _mm512_mul_ps(y0, c0));

            __m512 x1 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(qa1_i8)), scaleA);
            __m512 y1 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(qb1_i8)), scaleB);
            __m512 c1 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(cos1_i16)), q15_scale);
            __m512 s1 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(sin1_i16)), q15_scale);
            __m512 rotA1 = _mm512_fmsub_ps(x1, c1, _mm512_mul_ps(y1, s1));
            __m512 rotB1 = _mm512_fmadd_ps(x1, s1, _mm512_mul_ps(y1, c1));

            // ================================================================
            // PHASE 2 & 3: Requantize blockA and blockB INTERLEAVED
            // ================================================================

            // Compute abs and max (interleaved)
            __m512 absA0 = _mm512_abs_ps(rotA0);
            __m512 absB0 = _mm512_abs_ps(rotB0);
            __m512 absA1 = _mm512_abs_ps(rotA1);
            __m512 absB1 = _mm512_abs_ps(rotB1);

            __m512 maxA_all = _mm512_max_ps(absA0, absA1);
            __m512 maxB_all = _mm512_max_ps(absB0, absB1);

            float maxA_val = _mm512_reduce_max_ps(maxA_all);
            float maxB_val = _mm512_reduce_max_ps(maxB_all);

            // Compute scales
            float scaleA_new = maxA_val * (1.0f / 127.0f);
            float scaleB_new = maxB_val * (1.0f / 127.0f);
            if (scaleA_new < 1e-20f)
                scaleA_new = 1e-20f;
            if (scaleB_new < 1e-20f)
                scaleB_new = 1e-20f;

            __m512 invScaleA = _mm512_set1_ps(1.0f / scaleA_new);
            __m512 invScaleB = _mm512_set1_ps(1.0f / scaleB_new);

            // Scale, round, clamp (interleaved)
            __m512 scaledA0 = _mm512_mul_ps(rotA0, invScaleA);
            __m512 scaledB0 = _mm512_mul_ps(rotB0, invScaleB);
            __m512 scaledA1 = _mm512_mul_ps(rotA1, invScaleA);
            __m512 scaledB1 = _mm512_mul_ps(rotB1, invScaleB);

            __m512i qA0 = _mm512_cvtps_epi32(_mm512_roundscale_ps(scaledA0, _MM_FROUND_TO_NEAREST_INT));
            __m512i qB0 = _mm512_cvtps_epi32(_mm512_roundscale_ps(scaledB0, _MM_FROUND_TO_NEAREST_INT));
            __m512i qA1 = _mm512_cvtps_epi32(_mm512_roundscale_ps(scaledA1, _MM_FROUND_TO_NEAREST_INT));
            __m512i qB1 = _mm512_cvtps_epi32(_mm512_roundscale_ps(scaledB1, _MM_FROUND_TO_NEAREST_INT));

            qA0 = _mm512_max_epi32(_mm512_min_epi32(qA0, clamp_max), clamp_min);
            qB0 = _mm512_max_epi32(_mm512_min_epi32(qB0, clamp_max), clamp_min);
            qA1 = _mm512_max_epi32(_mm512_min_epi32(qA1, clamp_max), clamp_min);
            qB1 = _mm512_max_epi32(_mm512_min_epi32(qB1, clamp_max), clamp_min);

            // Pack int32 → int8 using AVX512 _mm512_cvtepi32_epi8
            __m128i qA0_8 = _mm512_cvtepi32_epi8(qA0);
            __m128i qB0_8 = _mm512_cvtepi32_epi8(qB0);
            __m128i qA1_8 = _mm512_cvtepi32_epi8(qA1);
            __m128i qB1_8 = _mm512_cvtepi32_epi8(qB1);

            _mm_storeu_si128(reinterpret_cast<__m128i *>(blockA.qs), qA0_8);
            _mm_storeu_si128(reinterpret_cast<__m128i *>(blockB.qs), qB0_8);
            _mm_storeu_si128(reinterpret_cast<__m128i *>(blockA.qs + 16), qA1_8);
            _mm_storeu_si128(reinterpret_cast<__m128i *>(blockB.qs + 16), qB1_8);

            // Compute sum_qs
            int32_t sumA = _mm512_reduce_add_epi32(_mm512_add_epi32(qA0, qA1));
            int32_t sumB = _mm512_reduce_add_epi32(_mm512_add_epi32(qB0, qB1));

            blockA.d = llaminar2::fp32_to_fp16(scaleA_new);
            blockB.d = llaminar2::fp32_to_fp16(scaleB_new);
            blockA.sum_qs = static_cast<int16_t>(sumA);
            blockB.sum_qs = static_cast<int16_t>(sumB);
        }
    }
#endif // __AVX512F__

    /**
     * @brief Apply Q8_1 RoPE to one head with automatic SIMD dispatch
     *
     * Dispatches to the best available SIMD implementation:
     * - AVX512 if available (fastest, processes 16 elements at a time)
     * - AVX2 if available (8 elements at a time)
     * - Scalar fallback (baseline)
     */
    void apply_rope_q8_1_integer_head(
        Q8_1Block *head_blocks,
        int blocks_per_head,
        const int16_t *cos_q15,
        const int16_t *sin_q15)
    {
        // Dispatch to best available SIMD implementation
#if defined(__AVX512F__)
        apply_rope_q8_1_integer_head_avx512(head_blocks, blocks_per_head, cos_q15, sin_q15);
#elif defined(__AVX2__)
        apply_rope_q8_1_integer_head_avx2(head_blocks, blocks_per_head, cos_q15, sin_q15);
#else
        apply_rope_q8_1_integer_head_scalar(head_blocks, blocks_per_head, cos_q15, sin_q15);
#endif
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

    // =========================================================================
    // Q8_1 → FP32 RoPE (Hybrid Mode - No Requantization)
    // =========================================================================

    void apply_rope_q8_1_to_fp32(
        const Q8_1Block *Q_in,
        const Q8_1Block *K_in,
        float *Q_out,
        float *K_out,
        const int *position_ids,
        int seq_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        float rope_theta)
    {
        if (!Q_in || !Q_out)
        {
            LOG_ERROR("apply_rope_q8_1_to_fp32: Q_in and Q_out must not be null");
            return;
        }

        if (head_dim % 32 != 0)
        {
            LOG_ERROR("apply_rope_q8_1_to_fp32: head_dim (" << head_dim << ") must be divisible by 32");
            return;
        }

        const int blocks_per_head = head_dim / 32;
        const int half_dim = head_dim / 2;
        const int q_stride_blocks = n_heads * blocks_per_head;
        const int k_stride_blocks = n_kv_heads * blocks_per_head;
        const int q_stride_floats = n_heads * head_dim;
        const int k_stride_floats = n_kv_heads * head_dim;

        // Pre-compute FP32 sin/cos table for all positions
        // Shape: [seq_len, half_dim]
        std::vector<float> cos_table(seq_len * half_dim);
        std::vector<float> sin_table(seq_len * half_dim);

        // Compute frequency bases: inv_freq[i] = 1.0 / (theta^(2i/head_dim))
        std::vector<float> inv_freq(half_dim);
        for (int i = 0; i < half_dim; ++i)
        {
            float exponent = static_cast<float>(2 * i) / static_cast<float>(head_dim);
            inv_freq[i] = 1.0f / std::pow(rope_theta, exponent);
        }

        // Compute sin/cos for each position
        for (int t = 0; t < seq_len; ++t)
        {
            int pos = position_ids ? position_ids[t] : t;
            if (pos < 0)
                pos = 0; // Handle padding
            float *c_ptr = cos_table.data() + t * half_dim;
            float *s_ptr = sin_table.data() + t * half_dim;
            for (int i = 0; i < half_dim; ++i)
            {
                float angle = static_cast<float>(pos) * inv_freq[i];
                c_ptr[i] = std::cos(angle);
                s_ptr[i] = std::sin(angle);
            }
        }

        // Process Q tensor: dequant + rotate (no requant)
        auto do_q_work = [&]()
        {
#pragma omp for collapse(2)
            for (int t = 0; t < seq_len; ++t)
            {
                for (int h = 0; h < n_heads; ++h)
                {
                    const Q8_1Block *head_in = Q_in + t * q_stride_blocks + h * blocks_per_head;
                    float *head_out = Q_out + t * q_stride_floats + h * head_dim;
                    const float *c_ptr = cos_table.data() + t * half_dim;
                    const float *s_ptr = sin_table.data() + t * half_dim;

                    // Dequantize all blocks for this head
                    alignas(64) float dequant[256]; // Max head_dim = 256
                    for (int b = 0; b < blocks_per_head; ++b)
                    {
                        const Q8_1Block &block = head_in[b];
                        float scale = fp16_to_fp32_rope(block.d);
                        for (int i = 0; i < 32; ++i)
                        {
                            dequant[b * 32 + i] = static_cast<float>(block.qs[i]) * scale;
                        }
                    }

                    // Apply rotation and write to FP32 output
                    for (int i = 0; i < half_dim; ++i)
                    {
                        float x = dequant[i];
                        float y = dequant[i + half_dim];
                        float c = c_ptr[i];
                        float s = s_ptr[i];
                        head_out[i] = x * c - y * s;
                        head_out[i + half_dim] = x * s + y * c;
                    }
                }
            }
        };
        OMP_WORKSHARE_REGION(do_q_work);

        // Process K tensor if provided
        if (K_in && K_out)
        {
            auto do_k_work = [&]()
            {
#pragma omp for collapse(2)
                for (int t = 0; t < seq_len; ++t)
                {
                    for (int h = 0; h < n_kv_heads; ++h)
                    {
                        const Q8_1Block *head_in = K_in + t * k_stride_blocks + h * blocks_per_head;
                        float *head_out = K_out + t * k_stride_floats + h * head_dim;
                        const float *c_ptr = cos_table.data() + t * half_dim;
                        const float *s_ptr = sin_table.data() + t * half_dim;

                        // Dequantize all blocks for this head
                        alignas(64) float dequant[256];
                        for (int b = 0; b < blocks_per_head; ++b)
                        {
                            const Q8_1Block &block = head_in[b];
                            float scale = fp16_to_fp32_rope(block.d);
                            for (int i = 0; i < 32; ++i)
                            {
                                dequant[b * 32 + i] = static_cast<float>(block.qs[i]) * scale;
                            }
                        }

                        // Apply rotation and write to FP32 output
                        for (int i = 0; i < half_dim; ++i)
                        {
                            float x = dequant[i];
                            float y = dequant[i + half_dim];
                            float c = c_ptr[i];
                            float s = s_ptr[i];
                            head_out[i] = x * c - y * s;
                            head_out[i + half_dim] = x * s + y * c;
                        }
                    }
                }
            };
            OMP_WORKSHARE_REGION(do_k_work);
        }
    }

    // ============================================================================
    // Q16_1 In-Place RoPE Implementation - Vectorized
    // ============================================================================
    //
    // Algorithm: Dequant Q16_1 → Rotate in FP32 → Requant to Q16_1
    // Processes block pairs (first half vs second half of head) for RoPE rotation.
    //
    // Optimization techniques:
    // - Dual load port exploitation: Interleaved loads from A/B blocks
    // - Loop unrolling: Process 2 block pairs per outer iteration
    // - Register blocking: Keep rotated values in registers, minimize spills
    // - Prefetching: Hint at upcoming block pairs
    // - FMA fusion: Use _mm256_fmadd_ps / _mm512_fmadd_ps where beneficial
    //
    // Expected speedups: AVX512 ~2x AVX2, AVX2 ~8x scalar
    // ============================================================================

    // Q15 to FP32 conversion constant
    namespace
    {
        constexpr float Q15_TO_FP32 = 1.0f / 32767.0f;
    }

    void apply_rope_q16_1_integer_head_scalar(
        Q16_1Block *head_blocks,
        int blocks_per_head,
        const int16_t *cos_q15,
        const int16_t *sin_q15)
    {
        const int half_blocks = blocks_per_head / 2;

        for (int b = 0; b < half_blocks; ++b)
        {
            Q16_1Block &blockA = head_blocks[b];
            Q16_1Block &blockB = head_blocks[b + half_blocks];

            float scaleA = blockA.d;
            float scaleB = blockB.d;

            const int16_t *cos_ptr = cos_q15 + b * 32;
            const int16_t *sin_ptr = sin_q15 + b * 32;

            float rotA[32];
            float rotB[32];

            // Dequant + Rotate
            for (int i = 0; i < 32; ++i)
            {
                float x = static_cast<float>(blockA.qs[i]) * scaleA;
                float y = static_cast<float>(blockB.qs[i]) * scaleB;
                float c = static_cast<float>(cos_ptr[i]) * Q15_TO_FP32;
                float s = static_cast<float>(sin_ptr[i]) * Q15_TO_FP32;
                rotA[i] = x * c - y * s;
                rotB[i] = x * s + y * c;
            }

            // Requantize blockA
            {
                float max_val = 0.0f;
                for (int i = 0; i < 32; ++i)
                    max_val = std::max(max_val, std::abs(rotA[i]));

                float new_scale = max_val / 32767.0f;
                if (new_scale < 1e-20f)
                    new_scale = 1e-20f;
                float inv_scale = 1.0f / new_scale;

                int64_t sum_qs = 0;
                for (int i = 0; i < 32; ++i)
                {
                    int32_t q = static_cast<int32_t>(std::round(rotA[i] * inv_scale));
                    q = std::max(-32767, std::min(32767, q));
                    blockA.qs[i] = static_cast<int16_t>(q);
                    sum_qs += q;
                }
                blockA.d = new_scale;
                blockA.sum_qs = static_cast<int32_t>(sum_qs);
            }

            // Requantize blockB
            {
                float max_val = 0.0f;
                for (int i = 0; i < 32; ++i)
                    max_val = std::max(max_val, std::abs(rotB[i]));

                float new_scale = max_val / 32767.0f;
                if (new_scale < 1e-20f)
                    new_scale = 1e-20f;
                float inv_scale = 1.0f / new_scale;

                int64_t sum_qs = 0;
                for (int i = 0; i < 32; ++i)
                {
                    int32_t q = static_cast<int32_t>(std::round(rotB[i] * inv_scale));
                    q = std::max(-32767, std::min(32767, q));
                    blockB.qs[i] = static_cast<int16_t>(q);
                    sum_qs += q;
                }
                blockB.d = new_scale;
                blockB.sum_qs = static_cast<int32_t>(sum_qs);
            }
        }
    }

#if defined(__AVX2__)
    void apply_rope_q16_1_integer_head_avx2(
        Q16_1Block *head_blocks,
        int blocks_per_head,
        const int16_t *cos_q15,
        const int16_t *sin_q15)
    {
        const int half_blocks = blocks_per_head / 2;
        const __m256 q15_scale = _mm256_set1_ps(Q15_TO_FP32);
        const __m256 sign_mask = _mm256_set1_ps(-0.0f);
        const __m256i clamp_min = _mm256_set1_epi32(-32767);
        const __m256i clamp_max = _mm256_set1_epi32(32767);

        // Process block pairs - fully unrolled inner loop for 32-element blocks
        for (int b = 0; b < half_blocks; ++b)
        {
            Q16_1Block &blockA = head_blocks[b];
            Q16_1Block &blockB = head_blocks[b + half_blocks];

            // Prefetch next block pair (if exists)
            if (b + 1 < half_blocks)
            {
                _mm_prefetch(reinterpret_cast<const char *>(&head_blocks[b + 1]), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char *>(&head_blocks[b + 1 + half_blocks]), _MM_HINT_T0);
            }

            const __m256 scaleA = _mm256_set1_ps(blockA.d);
            const __m256 scaleB = _mm256_set1_ps(blockB.d);
            const int16_t *cos_ptr = cos_q15 + b * 32;
            const int16_t *sin_ptr = sin_q15 + b * 32;

            // ================================================================
            // PHASE 1: Dequant + Rotate - Fully unrolled, interleaved loads
            // Process all 32 elements in 4 chunks of 8
            // ================================================================

            // Chunk 0: elements [0:8)
            // Interleaved loads exploit dual load ports
            __m128i qa0_i16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(blockA.qs));
            __m128i qb0_i16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(blockB.qs));
            __m128i cos0_i16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(cos_ptr));
            __m128i sin0_i16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(sin_ptr));

            // Chunk 1: elements [8:16) - load while chunk 0 converts
            __m128i qa1_i16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(blockA.qs + 8));
            __m128i qb1_i16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(blockB.qs + 8));
            __m128i cos1_i16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(cos_ptr + 8));
            __m128i sin1_i16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(sin_ptr + 8));

            // Chunk 2: elements [16:24)
            __m128i qa2_i16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(blockA.qs + 16));
            __m128i qb2_i16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(blockB.qs + 16));
            __m128i cos2_i16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(cos_ptr + 16));
            __m128i sin2_i16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(sin_ptr + 16));

            // Chunk 3: elements [24:32)
            __m128i qa3_i16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(blockA.qs + 24));
            __m128i qb3_i16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(blockB.qs + 24));
            __m128i cos3_i16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(cos_ptr + 24));
            __m128i sin3_i16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(sin_ptr + 24));

            // Convert chunk 0: int16 → int32 → float, then rotate
            __m256 x0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(qa0_i16)), scaleA);
            __m256 y0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(qb0_i16)), scaleB);
            __m256 c0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(cos0_i16)), q15_scale);
            __m256 s0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(sin0_i16)), q15_scale);
            // rotA = x*c - y*s, rotB = x*s + y*c (use FMA)
            __m256 rotA0 = _mm256_fmsub_ps(x0, c0, _mm256_mul_ps(y0, s0));
            __m256 rotB0 = _mm256_fmadd_ps(x0, s0, _mm256_mul_ps(y0, c0));

            // Convert chunk 1
            __m256 x1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(qa1_i16)), scaleA);
            __m256 y1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(qb1_i16)), scaleB);
            __m256 c1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(cos1_i16)), q15_scale);
            __m256 s1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(sin1_i16)), q15_scale);
            __m256 rotA1 = _mm256_fmsub_ps(x1, c1, _mm256_mul_ps(y1, s1));
            __m256 rotB1 = _mm256_fmadd_ps(x1, s1, _mm256_mul_ps(y1, c1));

            // Convert chunk 2
            __m256 x2 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(qa2_i16)), scaleA);
            __m256 y2 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(qb2_i16)), scaleB);
            __m256 c2 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(cos2_i16)), q15_scale);
            __m256 s2 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(sin2_i16)), q15_scale);
            __m256 rotA2 = _mm256_fmsub_ps(x2, c2, _mm256_mul_ps(y2, s2));
            __m256 rotB2 = _mm256_fmadd_ps(x2, s2, _mm256_mul_ps(y2, c2));

            // Convert chunk 3
            __m256 x3 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(qa3_i16)), scaleA);
            __m256 y3 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(qb3_i16)), scaleB);
            __m256 c3 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(cos3_i16)), q15_scale);
            __m256 s3 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(sin3_i16)), q15_scale);
            __m256 rotA3 = _mm256_fmsub_ps(x3, c3, _mm256_mul_ps(y3, s3));
            __m256 rotB3 = _mm256_fmadd_ps(x3, s3, _mm256_mul_ps(y3, c3));

            // ================================================================
            // PHASE 2: Requantize blockA - find max, scale, pack
            // ================================================================

            // Find max_abs across all 4 chunks (parallel reduction)
            __m256 absA0 = _mm256_andnot_ps(sign_mask, rotA0);
            __m256 absA1 = _mm256_andnot_ps(sign_mask, rotA1);
            __m256 absA2 = _mm256_andnot_ps(sign_mask, rotA2);
            __m256 absA3 = _mm256_andnot_ps(sign_mask, rotA3);

            __m256 maxA_01 = _mm256_max_ps(absA0, absA1);
            __m256 maxA_23 = _mm256_max_ps(absA2, absA3);
            __m256 maxA_all = _mm256_max_ps(maxA_01, maxA_23);

            // Horizontal max reduction
            __m128 maxA_lo = _mm256_castps256_ps128(maxA_all);
            __m128 maxA_hi = _mm256_extractf128_ps(maxA_all, 1);
            __m128 maxA_128 = _mm_max_ps(maxA_lo, maxA_hi);
            maxA_128 = _mm_max_ps(maxA_128, _mm_shuffle_ps(maxA_128, maxA_128, _MM_SHUFFLE(1, 0, 3, 2)));
            maxA_128 = _mm_max_ps(maxA_128, _mm_shuffle_ps(maxA_128, maxA_128, _MM_SHUFFLE(0, 0, 0, 1)));
            float maxA_val = _mm_cvtss_f32(maxA_128);

            float scaleA_new = maxA_val / 32767.0f;
            if (scaleA_new < 1e-20f)
                scaleA_new = 1e-20f;
            __m256 invScaleA = _mm256_set1_ps(1.0f / scaleA_new);

            // Scale, round, clamp, pack all 4 chunks
            __m256 scaledA0 = _mm256_mul_ps(rotA0, invScaleA);
            __m256 scaledA1 = _mm256_mul_ps(rotA1, invScaleA);
            __m256 scaledA2 = _mm256_mul_ps(rotA2, invScaleA);
            __m256 scaledA3 = _mm256_mul_ps(rotA3, invScaleA);

            __m256i qA0 = _mm256_cvtps_epi32(_mm256_round_ps(scaledA0, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            __m256i qA1 = _mm256_cvtps_epi32(_mm256_round_ps(scaledA1, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            __m256i qA2 = _mm256_cvtps_epi32(_mm256_round_ps(scaledA2, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            __m256i qA3 = _mm256_cvtps_epi32(_mm256_round_ps(scaledA3, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));

            // Clamp
            qA0 = _mm256_max_epi32(_mm256_min_epi32(qA0, clamp_max), clamp_min);
            qA1 = _mm256_max_epi32(_mm256_min_epi32(qA1, clamp_max), clamp_min);
            qA2 = _mm256_max_epi32(_mm256_min_epi32(qA2, clamp_max), clamp_min);
            qA3 = _mm256_max_epi32(_mm256_min_epi32(qA3, clamp_max), clamp_min);

            // Pack int32 → int16 and store
            __m128i qA0_16 = _mm_packs_epi32(_mm256_castsi256_si128(qA0), _mm256_extracti128_si256(qA0, 1));
            __m128i qA1_16 = _mm_packs_epi32(_mm256_castsi256_si128(qA1), _mm256_extracti128_si256(qA1, 1));
            __m128i qA2_16 = _mm_packs_epi32(_mm256_castsi256_si128(qA2), _mm256_extracti128_si256(qA2, 1));
            __m128i qA3_16 = _mm_packs_epi32(_mm256_castsi256_si128(qA3), _mm256_extracti128_si256(qA3, 1));

            _mm_storeu_si128(reinterpret_cast<__m128i *>(blockA.qs), qA0_16);
            _mm_storeu_si128(reinterpret_cast<__m128i *>(blockA.qs + 8), qA1_16);
            _mm_storeu_si128(reinterpret_cast<__m128i *>(blockA.qs + 16), qA2_16);
            _mm_storeu_si128(reinterpret_cast<__m128i *>(blockA.qs + 24), qA3_16);

            // Compute sum_qs (parallel reduction)
            __m256i sumA_01 = _mm256_add_epi32(qA0, qA1);
            __m256i sumA_23 = _mm256_add_epi32(qA2, qA3);
            __m256i sumA_all = _mm256_add_epi32(sumA_01, sumA_23);
            __m128i sumA_lo = _mm256_castsi256_si128(sumA_all);
            __m128i sumA_hi = _mm256_extracti128_si256(sumA_all, 1);
            __m128i sumA_128 = _mm_add_epi32(sumA_lo, sumA_hi);
            sumA_128 = _mm_add_epi32(sumA_128, _mm_shuffle_epi32(sumA_128, _MM_SHUFFLE(1, 0, 3, 2)));
            sumA_128 = _mm_add_epi32(sumA_128, _mm_shuffle_epi32(sumA_128, _MM_SHUFFLE(0, 0, 0, 1)));

            blockA.d = scaleA_new;
            blockA.sum_qs = _mm_cvtsi128_si32(sumA_128);

            // ================================================================
            // PHASE 3: Requantize blockB - same pattern
            // ================================================================

            __m256 absB0 = _mm256_andnot_ps(sign_mask, rotB0);
            __m256 absB1 = _mm256_andnot_ps(sign_mask, rotB1);
            __m256 absB2 = _mm256_andnot_ps(sign_mask, rotB2);
            __m256 absB3 = _mm256_andnot_ps(sign_mask, rotB3);

            __m256 maxB_01 = _mm256_max_ps(absB0, absB1);
            __m256 maxB_23 = _mm256_max_ps(absB2, absB3);
            __m256 maxB_all = _mm256_max_ps(maxB_01, maxB_23);

            __m128 maxB_lo = _mm256_castps256_ps128(maxB_all);
            __m128 maxB_hi = _mm256_extractf128_ps(maxB_all, 1);
            __m128 maxB_128 = _mm_max_ps(maxB_lo, maxB_hi);
            maxB_128 = _mm_max_ps(maxB_128, _mm_shuffle_ps(maxB_128, maxB_128, _MM_SHUFFLE(1, 0, 3, 2)));
            maxB_128 = _mm_max_ps(maxB_128, _mm_shuffle_ps(maxB_128, maxB_128, _MM_SHUFFLE(0, 0, 0, 1)));
            float maxB_val = _mm_cvtss_f32(maxB_128);

            float scaleB_new = maxB_val / 32767.0f;
            if (scaleB_new < 1e-20f)
                scaleB_new = 1e-20f;
            __m256 invScaleB = _mm256_set1_ps(1.0f / scaleB_new);

            __m256 scaledB0 = _mm256_mul_ps(rotB0, invScaleB);
            __m256 scaledB1 = _mm256_mul_ps(rotB1, invScaleB);
            __m256 scaledB2 = _mm256_mul_ps(rotB2, invScaleB);
            __m256 scaledB3 = _mm256_mul_ps(rotB3, invScaleB);

            __m256i qB0 = _mm256_cvtps_epi32(_mm256_round_ps(scaledB0, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            __m256i qB1 = _mm256_cvtps_epi32(_mm256_round_ps(scaledB1, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            __m256i qB2 = _mm256_cvtps_epi32(_mm256_round_ps(scaledB2, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            __m256i qB3 = _mm256_cvtps_epi32(_mm256_round_ps(scaledB3, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));

            qB0 = _mm256_max_epi32(_mm256_min_epi32(qB0, clamp_max), clamp_min);
            qB1 = _mm256_max_epi32(_mm256_min_epi32(qB1, clamp_max), clamp_min);
            qB2 = _mm256_max_epi32(_mm256_min_epi32(qB2, clamp_max), clamp_min);
            qB3 = _mm256_max_epi32(_mm256_min_epi32(qB3, clamp_max), clamp_min);

            __m128i qB0_16 = _mm_packs_epi32(_mm256_castsi256_si128(qB0), _mm256_extracti128_si256(qB0, 1));
            __m128i qB1_16 = _mm_packs_epi32(_mm256_castsi256_si128(qB1), _mm256_extracti128_si256(qB1, 1));
            __m128i qB2_16 = _mm_packs_epi32(_mm256_castsi256_si128(qB2), _mm256_extracti128_si256(qB2, 1));
            __m128i qB3_16 = _mm_packs_epi32(_mm256_castsi256_si128(qB3), _mm256_extracti128_si256(qB3, 1));

            _mm_storeu_si128(reinterpret_cast<__m128i *>(blockB.qs), qB0_16);
            _mm_storeu_si128(reinterpret_cast<__m128i *>(blockB.qs + 8), qB1_16);
            _mm_storeu_si128(reinterpret_cast<__m128i *>(blockB.qs + 16), qB2_16);
            _mm_storeu_si128(reinterpret_cast<__m128i *>(blockB.qs + 24), qB3_16);

            __m256i sumB_01 = _mm256_add_epi32(qB0, qB1);
            __m256i sumB_23 = _mm256_add_epi32(qB2, qB3);
            __m256i sumB_all = _mm256_add_epi32(sumB_01, sumB_23);
            __m128i sumB_lo = _mm256_castsi256_si128(sumB_all);
            __m128i sumB_hi = _mm256_extracti128_si256(sumB_all, 1);
            __m128i sumB_128 = _mm_add_epi32(sumB_lo, sumB_hi);
            sumB_128 = _mm_add_epi32(sumB_128, _mm_shuffle_epi32(sumB_128, _MM_SHUFFLE(1, 0, 3, 2)));
            sumB_128 = _mm_add_epi32(sumB_128, _mm_shuffle_epi32(sumB_128, _MM_SHUFFLE(0, 0, 0, 1)));

            blockB.d = scaleB_new;
            blockB.sum_qs = _mm_cvtsi128_si32(sumB_128);
        }
    }
#endif // __AVX2__

#if defined(__AVX512F__)
    void apply_rope_q16_1_integer_head_avx512(
        Q16_1Block *head_blocks,
        int blocks_per_head,
        const int16_t *cos_q15,
        const int16_t *sin_q15)
    {
        const int half_blocks = blocks_per_head / 2;
        const __m512 q15_scale = _mm512_set1_ps(Q15_TO_FP32);

        const __m512i clamp_min = _mm512_set1_epi32(-32767);
        const __m512i clamp_max = _mm512_set1_epi32(32767);

        for (int b = 0; b < half_blocks; ++b)
        {
            Q16_1Block &blockA = head_blocks[b];
            Q16_1Block &blockB = head_blocks[b + half_blocks];

            // Prefetch next block pair
            if (b + 1 < half_blocks)
            {
                _mm_prefetch(reinterpret_cast<const char *>(&head_blocks[b + 1]), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char *>(&head_blocks[b + 1 + half_blocks]), _MM_HINT_T0);
            }

            const __m512 scaleA = _mm512_set1_ps(blockA.d);
            const __m512 scaleB = _mm512_set1_ps(blockB.d);
            const int16_t *cos_ptr = cos_q15 + b * 32;
            const int16_t *sin_ptr = sin_q15 + b * 32;

            // ================================================================
            // PHASE 1: Dequant + Rotate - Fully unrolled, interleaved loads
            // Process all 32 elements in 2 chunks of 16
            // ================================================================

            // Load all data upfront (exploit dual load ports)
            __m256i qa0_i16 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(blockA.qs));
            __m256i qb0_i16 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(blockB.qs));
            __m256i qa1_i16 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(blockA.qs + 16));
            __m256i qb1_i16 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(blockB.qs + 16));

            __m256i cos0_i16 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(cos_ptr));
            __m256i sin0_i16 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(sin_ptr));
            __m256i cos1_i16 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(cos_ptr + 16));
            __m256i sin1_i16 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(sin_ptr + 16));

            // Convert and rotate chunk 0 (elements 0-15)
            __m512 x0 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(qa0_i16)), scaleA);
            __m512 y0 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(qb0_i16)), scaleB);
            __m512 c0 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(cos0_i16)), q15_scale);
            __m512 s0 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(sin0_i16)), q15_scale);
            // rotA = x*c - y*s, rotB = x*s + y*c (use FMA)
            __m512 rotA0 = _mm512_fmsub_ps(x0, c0, _mm512_mul_ps(y0, s0));
            __m512 rotB0 = _mm512_fmadd_ps(x0, s0, _mm512_mul_ps(y0, c0));

            // Convert and rotate chunk 1 (elements 16-31)
            __m512 x1 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(qa1_i16)), scaleA);
            __m512 y1 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(qb1_i16)), scaleB);
            __m512 c1 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(cos1_i16)), q15_scale);
            __m512 s1 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(sin1_i16)), q15_scale);
            __m512 rotA1 = _mm512_fmsub_ps(x1, c1, _mm512_mul_ps(y1, s1));
            __m512 rotB1 = _mm512_fmadd_ps(x1, s1, _mm512_mul_ps(y1, c1));

            // ================================================================
            // PHASE 2 & 3: Requantize blockA and blockB INTERLEAVED
            // Process both blocks together to hide latency and maximize ILP
            // ================================================================

            // 1. Compute Abs and Max (Interleaved)
            __m512 absA0 = _mm512_abs_ps(rotA0);
            __m512 absB0 = _mm512_abs_ps(rotB0);
            __m512 absA1 = _mm512_abs_ps(rotA1);
            __m512 absB1 = _mm512_abs_ps(rotB1);

            __m512 maxA_all = _mm512_max_ps(absA0, absA1);
            __m512 maxB_all = _mm512_max_ps(absB0, absB1);

            // Horizontal reductions (expensive, but independent)
            float maxA_val = _mm512_reduce_max_ps(maxA_all);
            float maxB_val = _mm512_reduce_max_ps(maxB_all);

            // 2. Compute Scales (Scalar ops, high latency)
            // Use multiplication by reciprocal for speed
            float scaleA_new = maxA_val * (1.0f / 32767.0f);
            float scaleB_new = maxB_val * (1.0f / 32767.0f);

            if (scaleA_new < 1e-20f)
                scaleA_new = 1e-20f;
            if (scaleB_new < 1e-20f)
                scaleB_new = 1e-20f;

            // 3. Compute Inverse Scales (Scalar division)
            // Compiler can interleave these independent divisions
            __m512 invScaleA = _mm512_set1_ps(1.0f / scaleA_new);
            __m512 invScaleB = _mm512_set1_ps(1.0f / scaleB_new);

            // 4. Scale, Round, Clamp (Interleaved)
            __m512 scaledA0 = _mm512_mul_ps(rotA0, invScaleA);
            __m512 scaledB0 = _mm512_mul_ps(rotB0, invScaleB);
            __m512 scaledA1 = _mm512_mul_ps(rotA1, invScaleA);
            __m512 scaledB1 = _mm512_mul_ps(rotB1, invScaleB);

            __m512i qA0 = _mm512_cvtps_epi32(_mm512_roundscale_ps(scaledA0, _MM_FROUND_TO_NEAREST_INT));
            __m512i qB0 = _mm512_cvtps_epi32(_mm512_roundscale_ps(scaledB0, _MM_FROUND_TO_NEAREST_INT));
            __m512i qA1 = _mm512_cvtps_epi32(_mm512_roundscale_ps(scaledA1, _MM_FROUND_TO_NEAREST_INT));
            __m512i qB1 = _mm512_cvtps_epi32(_mm512_roundscale_ps(scaledB1, _MM_FROUND_TO_NEAREST_INT));

            qA0 = _mm512_max_epi32(_mm512_min_epi32(qA0, clamp_max), clamp_min);
            qB0 = _mm512_max_epi32(_mm512_min_epi32(qB0, clamp_max), clamp_min);
            qA1 = _mm512_max_epi32(_mm512_min_epi32(qA1, clamp_max), clamp_min);
            qB1 = _mm512_max_epi32(_mm512_min_epi32(qB1, clamp_max), clamp_min);

            // 5. Pack and Store (Interleaved)
            __m256i qA0_16 = _mm512_cvtepi32_epi16(qA0);
            __m256i qB0_16 = _mm512_cvtepi32_epi16(qB0);
            __m256i qA1_16 = _mm512_cvtepi32_epi16(qA1);
            __m256i qB1_16 = _mm512_cvtepi32_epi16(qB1);

            _mm256_storeu_si256(reinterpret_cast<__m256i *>(blockA.qs), qA0_16);
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(blockB.qs), qB0_16);
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(blockA.qs + 16), qA1_16);
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(blockB.qs + 16), qB1_16);

            // 6. Compute Sums (Optimized Reduction)
            // Add vectors first, then reduce (saves 2 reductions)
            int32_t sumA = _mm512_reduce_add_epi32(_mm512_add_epi32(qA0, qA1));
            int32_t sumB = _mm512_reduce_add_epi32(_mm512_add_epi32(qB0, qB1));

            blockA.d = scaleA_new;
            blockB.d = scaleB_new;
            blockA.sum_qs = sumA;
            blockB.sum_qs = sumB;
        }
    }
#endif // __AVX512F__

    void apply_rope_q16_1_integer_head(
        Q16_1Block *head_blocks,
        int blocks_per_head,
        const int16_t *cos_q15,
        const int16_t *sin_q15)
    {
        // Dispatch to best available SIMD implementation
#if defined(__AVX512F__)
        apply_rope_q16_1_integer_head_avx512(head_blocks, blocks_per_head, cos_q15, sin_q15);
#elif defined(__AVX2__)
        apply_rope_q16_1_integer_head_avx2(head_blocks, blocks_per_head, cos_q15, sin_q15);
#else
        apply_rope_q16_1_integer_head_scalar(head_blocks, blocks_per_head, cos_q15, sin_q15);
#endif
    }

    void apply_rope_q16_1_integer(
        Q16_1Block *Q,
        Q16_1Block *K,
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
            return; // head_dim must be divisible by Q16_1 block size
        }

        const int blocks_per_head = head_dim / 32;
        const int q_stride_blocks = n_heads * blocks_per_head;
        const int k_stride_blocks = n_kv_heads * blocks_per_head;
        const int half_dim = head_dim / 2;

        // Get inverse frequencies (shared with Q8_1 RoPE)
        const auto &inv_freq = get_inv_freq_cached(head_dim, rope_theta);

        // Precompute sin/cos tables for all positions (Q15 format)
        // We reuse the same Q15 format as Q8_1 since sin/cos are bounded to [-1, 1]
        std::vector<int16_t> cos_table(seq_len * half_dim);
        std::vector<int16_t> sin_table(seq_len * half_dim);

        for (int t = 0; t < seq_len; ++t)
        {
            const int pos = position_ids[t];
            if (pos < 0)
                continue; // Skip padding

            for (int i = 0; i < half_dim; ++i)
            {
                float angle = static_cast<float>(pos) * inv_freq[i];
                float c = std::cos(angle);
                float s = std::sin(angle);
                // Quantize to Q15
                cos_table[t * half_dim + i] = static_cast<int16_t>(std::round(c * 32767.0f));
                sin_table[t * half_dim + i] = static_cast<int16_t>(std::round(s * 32767.0f));
            }
        }

        // Process Q tensor
        auto do_q_work = [&]()
        {
#pragma omp for collapse(2)
            for (int t = 0; t < seq_len; ++t)
            {
                for (int h = 0; h < n_heads; ++h)
                {
                    const int pos = position_ids[t];
                    if (pos < 0)
                        continue;

                    Q16_1Block *head_ptr = Q + t * q_stride_blocks + h * blocks_per_head;
                    const int16_t *c_ptr = cos_table.data() + t * half_dim;
                    const int16_t *s_ptr = sin_table.data() + t * half_dim;
                    apply_rope_q16_1_integer_head(head_ptr, blocks_per_head, c_ptr, s_ptr);
                }
            }
        };
        OMP_WORKSHARE_REGION(do_q_work);

        // Process K tensor if provided
        if (K)
        {
            auto do_k_work = [&]()
            {
#pragma omp for collapse(2)
                for (int t = 0; t < seq_len; ++t)
                {
                    for (int h = 0; h < n_kv_heads; ++h)
                    {
                        const int pos = position_ids[t];
                        if (pos < 0)
                            continue;

                        Q16_1Block *head_ptr = K + t * k_stride_blocks + h * blocks_per_head;
                        const int16_t *c_ptr = cos_table.data() + t * half_dim;
                        const int16_t *s_ptr = sin_table.data() + t * half_dim;
                        apply_rope_q16_1_integer_head(head_ptr, blocks_per_head, c_ptr, s_ptr);
                    }
                }
            };
            OMP_WORKSHARE_REGION(do_k_work);
        }

        (void)persistent_state; // TODO: Add persistent state optimization for decode
    }

    // =========================================================================
    // Q8_1 → Q16_1 RoPE (HybridQ16 Mode) - Integer-Only Vectorized
    // =========================================================================
    //
    // Algorithm: Common-scale integer rotation (inspired by Int8RequantRef)
    //
    // 1. Compute common scale across entire head (max|dequant_val| for all blocks)
    // 2. Compute per-block scale ratios: ratio_b = block.d / common_scale
    // 3. Integer rotation with Q15 sin/cos:
    //    - Scale int8 to int16 range: scaled = qs[i] * ratio_q8 (Q8 fixed-point)
    //    - Rotate: (a * cos_q15 - b * sin_q15) >> 15
    // 4. Output: uniform Q16_1 scale across all blocks in head
    //
    // Benefits:
    // - Inner loop is pure integer (int16 × int16 → int32 → int16)
    // - Uniform output scale simplifies downstream INT8 attention requant
    // - Vectorized with AVX512/AVX2
    // =========================================================================

    namespace
    {
        // Helper: Compute max|dequantized value| across a Q8_1 head (for common scale)
        inline float compute_q8_1_head_max_abs(const Q8_1Block *blocks, int blocks_per_head)
        {
            float max_abs = 0.0f;

#if defined(__AVX512F__)
            __m512 vmax = _mm512_setzero_ps();
            for (int b = 0; b < blocks_per_head; ++b)
            {
                const float scale = fp16_to_fp32_rope(blocks[b].d);
                const __m512 vscale = _mm512_set1_ps(scale);

                // Load 32 int8 values
                const __m256i bytes = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(blocks[b].qs));

                // Convert low 16 int8 → int16 → int32 → float
                const __m128i lo_bytes = _mm256_castsi256_si128(bytes);
                const __m256i lo16_i16 = _mm256_cvtepi8_epi16(lo_bytes);
                const __m512i lo16_i32 = _mm512_cvtepi16_epi32(lo16_i16);
                const __m512 lo16_f32 = _mm512_cvtepi32_ps(lo16_i32);

                // Convert high 16 int8 → int16 → int32 → float
                const __m128i hi_bytes = _mm256_extracti128_si256(bytes, 1);
                const __m256i hi16_i16 = _mm256_cvtepi8_epi16(hi_bytes);
                const __m512i hi16_i32 = _mm512_cvtepi16_epi32(hi16_i16);
                const __m512 hi16_f32 = _mm512_cvtepi32_ps(hi16_i32);

                // Scale and find max abs
                const __m512 scaled_lo = _mm512_mul_ps(lo16_f32, vscale);
                const __m512 scaled_hi = _mm512_mul_ps(hi16_f32, vscale);
                const __m512 abs_lo = _mm512_abs_ps(scaled_lo);
                const __m512 abs_hi = _mm512_abs_ps(scaled_hi);
                vmax = _mm512_max_ps(vmax, _mm512_max_ps(abs_lo, abs_hi));
            }
            max_abs = _mm512_reduce_max_ps(vmax);

#elif defined(__AVX2__)
            __m256 vmax = _mm256_setzero_ps();
            const __m256 sign_mask = _mm256_set1_ps(-0.0f);

            for (int b = 0; b < blocks_per_head; ++b)
            {
                const float scale = fp16_to_fp32_rope(blocks[b].d);
                const __m256 vscale = _mm256_set1_ps(scale);

                // Process 32 int8 in 4 chunks of 8
                for (int chunk = 0; chunk < 4; ++chunk)
                {
                    const __m128i bytes8 = _mm_loadl_epi64(
                        reinterpret_cast<const __m128i *>(blocks[b].qs + chunk * 8));
                    const __m256i i32 = _mm256_cvtepi8_epi32(bytes8);
                    const __m256 f32 = _mm256_cvtepi32_ps(i32);
                    const __m256 scaled = _mm256_mul_ps(f32, vscale);
                    const __m256 abs_val = _mm256_andnot_ps(sign_mask, scaled);
                    vmax = _mm256_max_ps(vmax, abs_val);
                }
            }
            // Horizontal max reduction
            __m128 hi = _mm256_extractf128_ps(vmax, 1);
            __m128 lo = _mm256_castps256_ps128(vmax);
            __m128 m = _mm_max_ps(lo, hi);
            m = _mm_max_ps(m, _mm_shuffle_ps(m, m, _MM_SHUFFLE(2, 3, 0, 1)));
            m = _mm_max_ps(m, _mm_shuffle_ps(m, m, _MM_SHUFFLE(1, 0, 3, 2)));
            max_abs = _mm_cvtss_f32(m);
#else
            // Scalar fallback
            for (int b = 0; b < blocks_per_head; ++b)
            {
                const float scale = fp16_to_fp32_rope(blocks[b].d);
                for (int i = 0; i < 32; ++i)
                {
                    float val = static_cast<float>(blocks[b].qs[i]) * scale;
                    max_abs = std::max(max_abs, std::abs(val));
                }
            }
#endif
            return max_abs;
        }

        // Helper: Rotate one Q8_1 block pair to Q16_1 using integer arithmetic
        // BlockA is at position b, BlockB is at position b+half_blocks
        // They form rotation pairs: (A[i], B[i]) for i in 0..31
        inline void rotate_q8_1_block_pair_to_q16_1_avx512(
            const Q8_1Block &blockA,
            const Q8_1Block &blockB,
            Q16_1Block &outA,
            Q16_1Block &outB,
            const int16_t *cos_q15, // 32 values for this block
            const int16_t *sin_q15, // 32 values for this block
            float common_scale)
        {
#if defined(__AVX512F__)
            // Compute scale ratios as Q8 fixed-point (ratio * 256)
            const float scaleA = fp16_to_fp32_rope(blockA.d);
            const float scaleB = fp16_to_fp32_rope(blockB.d);
            const float ratioA = scaleA / common_scale;
            const float ratioB = scaleB / common_scale;

            // Q8 fixed-point ratios (8 fractional bits) - max value ~256
            const int16_t ratioA_q8 = static_cast<int16_t>(std::round(ratioA * 256.0f));
            const int16_t ratioB_q8 = static_cast<int16_t>(std::round(ratioB * 256.0f));

            // Load 32 int8 values from each block
            const __m256i bytesA = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(blockA.qs));
            const __m256i bytesB = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(blockB.qs));

            // Convert int8 → int16 (sign-extend)
            // Low 16 bytes
            const __m256i a_lo_i16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(bytesA));
            const __m256i a_hi_i16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytesA, 1));
            const __m256i b_lo_i16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(bytesB));
            const __m256i b_hi_i16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytesB, 1));

            // Scale by ratio: int16 * int16 → int32, then >> 8 to get Q8 result
            const __m256i vRatioA = _mm256_set1_epi16(ratioA_q8);
            const __m256i vRatioB = _mm256_set1_epi16(ratioB_q8);

            // For int16 * int16 → int16 with shift, use _mm256_mulhi_epi16 (gives high 16 bits)
            // But we want (a * ratio) >> 8, so let's compute properly:
            // Use 32-bit intermediates for precision

            // Process in 16-element chunks using int32 arithmetic
            // Low 16 elements of A
            __m512i a_lo_i32 = _mm512_cvtepi16_epi32(a_lo_i16);
            __m512i b_lo_i32 = _mm512_cvtepi16_epi32(b_lo_i16);
            __m512i a_hi_i32 = _mm512_cvtepi16_epi32(a_hi_i16);
            __m512i b_hi_i32 = _mm512_cvtepi16_epi32(b_hi_i16);

            // Scale by ratio (in int32): (qs * ratio_q8) >> 8 gives scaled value in ~int8 range
            // But we want output in int16 range for Q16_1, so keep more precision
            // Actually: (int8 * 256) gives int16 range, then rotation preserves range
            const __m512i vRatioA_i32 = _mm512_set1_epi32(ratioA_q8);
            const __m512i vRatioB_i32 = _mm512_set1_epi32(ratioB_q8);

            // scaled_a = (a * ratioA_q8) - keeps full precision for rotation
            __m512i scaled_a_lo = _mm512_mullo_epi32(a_lo_i32, vRatioA_i32);
            __m512i scaled_a_hi = _mm512_mullo_epi32(a_hi_i32, vRatioA_i32);
            __m512i scaled_b_lo = _mm512_mullo_epi32(b_lo_i32, vRatioB_i32);
            __m512i scaled_b_hi = _mm512_mullo_epi32(b_hi_i32, vRatioB_i32);

            // Load Q15 sin/cos (16-bit values)
            const __m256i cos_lo = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(cos_q15));
            const __m256i cos_hi = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(cos_q15 + 16));
            const __m256i sin_lo = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(sin_q15));
            const __m256i sin_hi = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(sin_q15 + 16));

            // Expand to int32 for multiply
            const __m512i cos_lo_i32 = _mm512_cvtepi16_epi32(cos_lo);
            const __m512i cos_hi_i32 = _mm512_cvtepi16_epi32(cos_hi);
            const __m512i sin_lo_i32 = _mm512_cvtepi16_epi32(sin_lo);
            const __m512i sin_hi_i32 = _mm512_cvtepi16_epi32(sin_hi);

            // Rotation: out_a = a * cos - b * sin
            //           out_b = a * sin + b * cos
            // We need: (scaled_a * cos_q15 - scaled_b * sin_q15) >> (8 + 15)
            // = (scaled_a * cos_q15 - scaled_b * sin_q15) >> 23
            // But scaled_a is (int8 * ratio_q8), range ~[-32768, 32767]
            // scaled_a * cos_q15 range: [-32768 * 32767, 32767 * 32767] = ~[-1B, 1B]
            // After subtraction: ~[-2B, 2B], needs >> 23 to fit int16

            // out_a_lo = (scaled_a_lo * cos - scaled_b_lo * sin) >> 15
            __m512i ac_lo = _mm512_mullo_epi32(scaled_a_lo, cos_lo_i32);
            __m512i bs_lo = _mm512_mullo_epi32(scaled_b_lo, sin_lo_i32);
            // Symmetric rounding before >> 15
            __m512i tmp_a_lo = _mm512_sub_epi32(ac_lo, bs_lo);
            __m512i sign_a_lo = _mm512_srai_epi32(tmp_a_lo, 31);
            __m512i bias_a_lo = _mm512_sub_epi32(
                _mm512_set1_epi32(1 << 14),
                _mm512_and_si512(sign_a_lo, _mm512_set1_epi32(1 << 15)));
            __m512i out_a_lo_i32 = _mm512_srai_epi32(_mm512_add_epi32(tmp_a_lo, bias_a_lo), 15);

            __m512i ac_hi = _mm512_mullo_epi32(scaled_a_hi, cos_hi_i32);
            __m512i bs_hi = _mm512_mullo_epi32(scaled_b_hi, sin_hi_i32);
            __m512i tmp_a_hi = _mm512_sub_epi32(ac_hi, bs_hi);
            __m512i sign_a_hi = _mm512_srai_epi32(tmp_a_hi, 31);
            __m512i bias_a_hi = _mm512_sub_epi32(
                _mm512_set1_epi32(1 << 14),
                _mm512_and_si512(sign_a_hi, _mm512_set1_epi32(1 << 15)));
            __m512i out_a_hi_i32 = _mm512_srai_epi32(_mm512_add_epi32(tmp_a_hi, bias_a_hi), 15);

            // out_b_lo = (scaled_a_lo * sin + scaled_b_lo * cos) >> 15
            __m512i as_lo = _mm512_mullo_epi32(scaled_a_lo, sin_lo_i32);
            __m512i bc_lo = _mm512_mullo_epi32(scaled_b_lo, cos_lo_i32);
            __m512i tmp_b_lo = _mm512_add_epi32(as_lo, bc_lo);
            __m512i sign_b_lo = _mm512_srai_epi32(tmp_b_lo, 31);
            __m512i bias_b_lo = _mm512_sub_epi32(
                _mm512_set1_epi32(1 << 14),
                _mm512_and_si512(sign_b_lo, _mm512_set1_epi32(1 << 15)));
            __m512i out_b_lo_i32 = _mm512_srai_epi32(_mm512_add_epi32(tmp_b_lo, bias_b_lo), 15);

            __m512i as_hi = _mm512_mullo_epi32(scaled_a_hi, sin_hi_i32);
            __m512i bc_hi = _mm512_mullo_epi32(scaled_b_hi, cos_hi_i32);
            __m512i tmp_b_hi = _mm512_add_epi32(as_hi, bc_hi);
            __m512i sign_b_hi = _mm512_srai_epi32(tmp_b_hi, 31);
            __m512i bias_b_hi = _mm512_sub_epi32(
                _mm512_set1_epi32(1 << 14),
                _mm512_and_si512(sign_b_hi, _mm512_set1_epi32(1 << 15)));
            __m512i out_b_hi_i32 = _mm512_srai_epi32(_mm512_add_epi32(tmp_b_hi, bias_b_hi), 15);

            // Pack to int16 in a stable, sequential order.
            // Using 256-bit pack avoids tricky 512-bit lane interleaving.
            const __m256i a_lo_256 = _mm512_castsi512_si256(out_a_lo_i32);
            const __m256i a_hi_256 = _mm512_extracti64x4_epi64(out_a_lo_i32, 1);
            const __m256i a0_i16 = _mm256_packs_epi32(a_lo_256, a_hi_256); // 16×int16

            const __m256i a2_lo_256 = _mm512_castsi512_si256(out_a_hi_i32);
            const __m256i a2_hi_256 = _mm512_extracti64x4_epi64(out_a_hi_i32, 1);
            const __m256i a1_i16 = _mm256_packs_epi32(a2_lo_256, a2_hi_256); // 16×int16

            const __m256i b_lo_256 = _mm512_castsi512_si256(out_b_lo_i32);
            const __m256i b_hi_256 = _mm512_extracti64x4_epi64(out_b_lo_i32, 1);
            const __m256i b0_i16 = _mm256_packs_epi32(b_lo_256, b_hi_256);

            const __m256i b2_lo_256 = _mm512_castsi512_si256(out_b_hi_i32);
            const __m256i b2_hi_256 = _mm512_extracti64x4_epi64(out_b_hi_i32, 1);
            const __m256i b1_i16 = _mm256_packs_epi32(b2_lo_256, b2_hi_256);

            _mm256_storeu_si256(reinterpret_cast<__m256i *>(outA.qs), a0_i16);
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(outA.qs + 16), a1_i16);
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(outB.qs), b0_i16);
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(outB.qs + 16), b1_i16);

            // Sum can be taken from the int32 results directly (before saturation to int16).
            const int32_t sum_a = _mm512_reduce_add_epi32(out_a_lo_i32) + _mm512_reduce_add_epi32(out_a_hi_i32);
            const int32_t sum_b = _mm512_reduce_add_epi32(out_b_lo_i32) + _mm512_reduce_add_epi32(out_b_hi_i32);

            // Set output block metadata
            // common_scale is per-head max / 127, output needs scale for int16 range
            // Output value range after rotation is roughly [-127*256, 127*256] >> 23 ≈ [-3.9, 3.9]
            // Wait, that's wrong. Let me recalculate:
            // scaled_a = int8 * ratio_q8 ≈ [-127 * 256, 127 * 256] = [-32512, 32512] (fits int16)
            // scaled_a * cos_q15 ≈ [-32512 * 32767] ≈ [-1.06B] (fits int32)
            // After >> 23: [-1.06B / 8M] ≈ [-127] range
            // Hmm, we're losing precision. Need different shift strategy.

            // Actually the output is ~[-256, 256] which is int16 range but wastes bits.
            // For Q16_1 we want to use full [-32767, 32767] range.
            // Solution: shift by (8 + 15 - 8) = 15 to keep more precision
            // This gives output range [-32512, 32512] which fits Q16_1

            // Keep the implicit *256 factor from ratio_q8 in the int16 output.
            // Compensate in the output scale.
            outA.d = common_scale / 256.0f;
            outA.sum_qs = sum_a;
            outB.d = common_scale / 256.0f;
            outB.sum_qs = sum_b;

#else
            // Fallback defined below
            (void)blockA;
            (void)blockB;
            (void)outA;
            (void)outB;
            (void)cos_q15;
            (void)sin_q15;
            (void)common_scale;
#endif
        }

        // AVX2 version
        inline void rotate_q8_1_block_pair_to_q16_1_avx2(
            const Q8_1Block &blockA,
            const Q8_1Block &blockB,
            Q16_1Block &outA,
            Q16_1Block &outB,
            const int16_t *cos_q15,
            const int16_t *sin_q15,
            float common_scale)
        {
#if defined(__AVX2__)
            const float scaleA = fp16_to_fp32_rope(blockA.d);
            const float scaleB = fp16_to_fp32_rope(blockB.d);
            const float ratioA = scaleA / common_scale;
            const float ratioB = scaleB / common_scale;
            const int16_t ratioA_q8 = static_cast<int16_t>(std::round(ratioA * 256.0f));
            const int16_t ratioB_q8 = static_cast<int16_t>(std::round(ratioB * 256.0f));

            int32_t sum_a = 0, sum_b = 0;

            // Process 8 elements at a time (AVX2 has 256-bit = 8×int32)
            for (int chunk = 0; chunk < 4; ++chunk)
            {
                const int offset = chunk * 8;

                // Load 8 int8 values
                const __m128i a_bytes = _mm_loadl_epi64(
                    reinterpret_cast<const __m128i *>(blockA.qs + offset));
                const __m128i b_bytes = _mm_loadl_epi64(
                    reinterpret_cast<const __m128i *>(blockB.qs + offset));

                // Convert to int32
                __m256i a_i32 = _mm256_cvtepi8_epi32(a_bytes);
                __m256i b_i32 = _mm256_cvtepi8_epi32(b_bytes);

                // Scale by ratio
                const __m256i vRatioA = _mm256_set1_epi32(ratioA_q8);
                const __m256i vRatioB = _mm256_set1_epi32(ratioB_q8);
                __m256i scaled_a = _mm256_mullo_epi32(a_i32, vRatioA);
                __m256i scaled_b = _mm256_mullo_epi32(b_i32, vRatioB);

                // Load sin/cos (int16 → int32)
                const __m128i cos_i16 = _mm_loadu_si128(
                    reinterpret_cast<const __m128i *>(cos_q15 + offset));
                const __m128i sin_i16 = _mm_loadu_si128(
                    reinterpret_cast<const __m128i *>(sin_q15 + offset));
                __m256i cos_i32 = _mm256_cvtepi16_epi32(cos_i16);
                __m256i sin_i32 = _mm256_cvtepi16_epi32(sin_i16);

                // Rotate
                __m256i ac = _mm256_mullo_epi32(scaled_a, cos_i32);
                __m256i bs = _mm256_mullo_epi32(scaled_b, sin_i32);
                __m256i as = _mm256_mullo_epi32(scaled_a, sin_i32);
                __m256i bc = _mm256_mullo_epi32(scaled_b, cos_i32);

                // Symmetric rounding before >> 15
                __m256i tmp_a = _mm256_sub_epi32(ac, bs);
                __m256i sign_a = _mm256_srai_epi32(tmp_a, 31);
                __m256i bias_a = _mm256_sub_epi32(
                    _mm256_set1_epi32(1 << 14),
                    _mm256_and_si256(sign_a, _mm256_set1_epi32(1 << 15)));
                __m256i out_a_i32 = _mm256_srai_epi32(_mm256_add_epi32(tmp_a, bias_a), 15);

                __m256i tmp_b = _mm256_add_epi32(as, bc);
                __m256i sign_b = _mm256_srai_epi32(tmp_b, 31);
                __m256i bias_b = _mm256_sub_epi32(
                    _mm256_set1_epi32(1 << 14),
                    _mm256_and_si256(sign_b, _mm256_set1_epi32(1 << 15)));
                __m256i out_b_i32 = _mm256_srai_epi32(_mm256_add_epi32(tmp_b, bias_b), 15);

                // Pack 8×int32 -> 8×int16 in correct order.
                const __m128i out_a_lo_128 = _mm256_castsi256_si128(out_a_i32);
                const __m128i out_a_hi_128 = _mm256_extracti128_si256(out_a_i32, 1);
                const __m128i out_a_i16 = _mm_packs_epi32(out_a_lo_128, out_a_hi_128);

                const __m128i out_b_lo_128 = _mm256_castsi256_si128(out_b_i32);
                const __m128i out_b_hi_128 = _mm256_extracti128_si256(out_b_i32, 1);
                const __m128i out_b_i16 = _mm_packs_epi32(out_b_lo_128, out_b_hi_128);

                _mm_storeu_si128(reinterpret_cast<__m128i *>(outA.qs + offset), out_a_i16);
                _mm_storeu_si128(reinterpret_cast<__m128i *>(outB.qs + offset), out_b_i16);

                // Accumulate sum
                // Sum the int32 values before packing
                __m128i lo_a = _mm256_castsi256_si128(out_a_i32);
                __m128i hi_a = _mm256_extracti128_si256(out_a_i32, 1);
                __m128i sum_a_128 = _mm_add_epi32(lo_a, hi_a);
                sum_a_128 = _mm_add_epi32(sum_a_128, _mm_shuffle_epi32(sum_a_128, _MM_SHUFFLE(2, 3, 0, 1)));
                sum_a_128 = _mm_add_epi32(sum_a_128, _mm_shuffle_epi32(sum_a_128, _MM_SHUFFLE(1, 0, 3, 2)));
                sum_a += _mm_cvtsi128_si32(sum_a_128);

                __m128i lo_b = _mm256_castsi256_si128(out_b_i32);
                __m128i hi_b = _mm256_extracti128_si256(out_b_i32, 1);
                __m128i sum_b_128 = _mm_add_epi32(lo_b, hi_b);
                sum_b_128 = _mm_add_epi32(sum_b_128, _mm_shuffle_epi32(sum_b_128, _MM_SHUFFLE(2, 3, 0, 1)));
                sum_b_128 = _mm_add_epi32(sum_b_128, _mm_shuffle_epi32(sum_b_128, _MM_SHUFFLE(1, 0, 3, 2)));
                sum_b += _mm_cvtsi128_si32(sum_b_128);
            }

            outA.d = common_scale / 256.0f;
            outA.sum_qs = sum_a;
            outB.d = common_scale / 256.0f;
            outB.sum_qs = sum_b;
#else
            (void)blockA;
            (void)blockB;
            (void)outA;
            (void)outB;
            (void)cos_q15;
            (void)sin_q15;
            (void)common_scale;
#endif
        }

        // Scalar fallback
        inline void rotate_q8_1_block_pair_to_q16_1_scalar(
            const Q8_1Block &blockA,
            const Q8_1Block &blockB,
            Q16_1Block &outA,
            Q16_1Block &outB,
            const int16_t *cos_q15,
            const int16_t *sin_q15,
            float common_scale)
        {
            const float scaleA = fp16_to_fp32_rope(blockA.d);
            const float scaleB = fp16_to_fp32_rope(blockB.d);
            const float ratioA = scaleA / common_scale;
            const float ratioB = scaleB / common_scale;
            const int32_t ratioA_q8 = static_cast<int32_t>(std::round(ratioA * 256.0f));
            const int32_t ratioB_q8 = static_cast<int32_t>(std::round(ratioB * 256.0f));

            int32_t sum_a = 0, sum_b = 0;

            for (int i = 0; i < 32; ++i)
            {
                // Scale int8 to common scale (result in ~int16 range)
                int32_t scaled_a = static_cast<int32_t>(blockA.qs[i]) * ratioA_q8;
                int32_t scaled_b = static_cast<int32_t>(blockB.qs[i]) * ratioB_q8;

                // Get Q15 sin/cos
                int32_t cos_val = cos_q15[i];
                int32_t sin_val = sin_q15[i];

                // Rotate in integer domain.
                // scaled_* has an implicit *256 factor from ratio_q8.
                // cos/sin are Q15. Compute with int64 intermediates to avoid overflow:
                // scaled_* can be ~4e6, multiplying by ~3e4 exceeds int32 range.
                int64_t tmp_a = (static_cast<int64_t>(scaled_a) * cos_val - static_cast<int64_t>(scaled_b) * sin_val);
                int64_t tmp_b = (static_cast<int64_t>(scaled_a) * sin_val + static_cast<int64_t>(scaled_b) * cos_val);

                // Symmetric rounding before >> 15
                tmp_a += (tmp_a >= 0) ? (1LL << 14) : -(1LL << 14);
                tmp_b += (tmp_b >= 0) ? (1LL << 14) : -(1LL << 14);
                int32_t out_a = static_cast<int32_t>(tmp_a >> 15);
                int32_t out_b = static_cast<int32_t>(tmp_b >> 15);

                // Clamp to int16 range
                out_a = std::max(-32767, std::min(32767, out_a));
                out_b = std::max(-32767, std::min(32767, out_b));

                outA.qs[i] = static_cast<int16_t>(out_a);
                outB.qs[i] = static_cast<int16_t>(out_b);

                sum_a += out_a;
                sum_b += out_b;
            }

            outA.d = common_scale / 256.0f;
            outA.sum_qs = sum_a;
            outB.d = common_scale / 256.0f;
            outB.sum_qs = sum_b;
        }

        // Dispatch to best available implementation
        inline void rotate_q8_1_block_pair_to_q16_1(
            const Q8_1Block &blockA,
            const Q8_1Block &blockB,
            Q16_1Block &outA,
            Q16_1Block &outB,
            const int16_t *cos_q15,
            const int16_t *sin_q15,
            float common_scale)
        {
            // Correctness-first: use the scalar implementation with int64 intermediates.
            // The AVX2/AVX512 variants use int32 products and can overflow for typical ranges.
            rotate_q8_1_block_pair_to_q16_1_scalar(blockA, blockB, outA, outB, cos_q15, sin_q15, common_scale);
        }

        // Process one head: compute common scale, then rotate all block pairs
        inline float apply_rope_q8_1_to_q16_1_head(
            const Q8_1Block *head_in,
            Q16_1Block *head_out,
            int blocks_per_head,
            const int16_t *cos_q15, // [half_dim] = [blocks_per_head/2 * 32]
            const int16_t *sin_q15)
        {
            const int half_blocks = blocks_per_head / 2;

            // Step 1: Compute common scale (max|dequant| across all blocks)
            float max_abs = compute_q8_1_head_max_abs(head_in, blocks_per_head);
            if (max_abs < 1e-20f)
                max_abs = 1e-20f;

            // common_scale: the scale such that max_abs / common_scale ≈ 127
            // This normalizes values to int8-ish range before rotation
            float common_scale = max_abs / 127.0f;

            // Step 2: Rotate each block pair
            // Block b pairs with block b + half_blocks
            // sin/cos for block b are at offset b * 32
            for (int b = 0; b < half_blocks; ++b)
            {
                rotate_q8_1_block_pair_to_q16_1(
                    head_in[b],
                    head_in[b + half_blocks],
                    head_out[b],
                    head_out[b + half_blocks],
                    cos_q15 + b * 32,
                    sin_q15 + b * 32,
                    common_scale);
            }

            return common_scale;
        }

    } // anonymous namespace

    void apply_rope_q8_1_to_q16_1(
        const Q8_1Block *Q_in,
        const Q8_1Block *K_in,
        Q16_1Block *Q_out,
        Q16_1Block *K_out,
        const int *position_ids,
        int seq_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        float rope_theta)
    {
        if (!Q_in || !Q_out)
        {
            LOG_ERROR("apply_rope_q8_1_to_q16_1: Q_in and Q_out must not be null");
            return;
        }

        if (head_dim % 32 != 0)
        {
            LOG_ERROR("apply_rope_q8_1_to_q16_1: head_dim (" << head_dim << ") must be divisible by 32");
            return;
        }

        const int blocks_per_head = head_dim / 32;
        const int half_dim = head_dim / 2;
        const int q_stride_blocks = n_heads * blocks_per_head;
        const int k_stride_blocks = n_kv_heads * blocks_per_head;

        // Get cached inverse frequencies
        const auto &inv_freq = get_inv_freq_cached(head_dim, rope_theta);

        // Pre-compute Q15 sin/cos tables for all positions
        // Shape: [seq_len, half_dim]
        // Q15 format: value * 32767 stored as int16_t
        std::vector<int16_t> cos_q15(seq_len * half_dim);
        std::vector<int16_t> sin_q15(seq_len * half_dim);

        for (int t = 0; t < seq_len; ++t)
        {
            int pos = position_ids ? position_ids[t] : t;
            if (pos < 0)
                pos = 0;

            int16_t *c_ptr = cos_q15.data() + t * half_dim;
            int16_t *s_ptr = sin_q15.data() + t * half_dim;

            for (int i = 0; i < half_dim; ++i)
            {
                float angle = static_cast<float>(pos) * inv_freq[i];
                // Use rounded Q15 sin/cos to avoid truncation bias.
                int32_t c = static_cast<int32_t>(std::lround(std::cos(angle) * 32767.0f));
                int32_t s = static_cast<int32_t>(std::lround(std::sin(angle) * 32767.0f));
                if (c > 32767)
                    c = 32767;
                if (c < -32767)
                    c = -32767;
                if (s > 32767)
                    s = 32767;
                if (s < -32767)
                    s = -32767;
                c_ptr[i] = static_cast<int16_t>(c);
                s_ptr[i] = static_cast<int16_t>(s);
            }
        }

        // Process Q tensor with OMP parallelization
        auto do_q_work = [&]()
        {
#pragma omp for collapse(2)
            for (int t = 0; t < seq_len; ++t)
            {
                for (int h = 0; h < n_heads; ++h)
                {
                    const Q8_1Block *head_in = Q_in + t * q_stride_blocks + h * blocks_per_head;
                    Q16_1Block *head_out = Q_out + t * q_stride_blocks + h * blocks_per_head;
                    const int16_t *c_ptr = cos_q15.data() + t * half_dim;
                    const int16_t *s_ptr = sin_q15.data() + t * half_dim;

                    (void)apply_rope_q8_1_to_q16_1_head(head_in, head_out, blocks_per_head, c_ptr, s_ptr);
                }
            }
        };
        OMP_WORKSHARE_REGION(do_q_work);

        // Process K tensor if provided
        if (K_in && K_out)
        {
            auto do_k_work = [&]()
            {
#pragma omp for collapse(2)
                for (int t = 0; t < seq_len; ++t)
                {
                    for (int h = 0; h < n_kv_heads; ++h)
                    {
                        const Q8_1Block *head_in = K_in + t * k_stride_blocks + h * blocks_per_head;
                        Q16_1Block *head_out = K_out + t * k_stride_blocks + h * blocks_per_head;
                        const int16_t *c_ptr = cos_q15.data() + t * half_dim;
                        const int16_t *s_ptr = sin_q15.data() + t * half_dim;

                        (void)apply_rope_q8_1_to_q16_1_head(head_in, head_out, blocks_per_head, c_ptr, s_ptr);
                    }
                }
            };
            OMP_WORKSHARE_REGION(do_k_work);
        }
    }

} // namespace llaminar2::primitives
