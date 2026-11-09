/**
 * @file RoPEInline.h
 * @brief Bounded native FP32 rotary fragments for attention and TurboQuant.
 *
 * Architecture-native NEON/AVX-512/AVX2 leaves retain original physical halves
 * and the declared multiply/add program. The K wrapper shares absolute-position
 * coefficients across heads in fixed 128-pair fragments, not a head-sized stack
 * array with an implicit 256-value limit. No allocation, tensor mirror or
 * coherence publication belongs to these borrowed synchronous operations.
 */
#pragma once
#include "RoPEPrimitives.h"
#include "utils/OpenMPUtils.h"
#include "utils/CPUFeatures.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#if LLAMINAR_COMPILED_WITH_NEON
#include "NEONRoPEPrimitives.h"
#elif defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#endif

namespace llaminar2
{
    namespace rope_inline_detail
    {
        /** @brief Reject an output/source alias without forming overflowing end pointers.
         * @param a First bank; @param a_bytes Its complete byte extent.
         * @param b Second bank; @param b_bytes Its complete byte extent.
         * @throws std::invalid_argument On overlap of nonempty banks.
         */
        inline void require_disjoint(const void *a, std::size_t a_bytes, const void *b, std::size_t b_bytes)
        {
            const auto x = reinterpret_cast<std::uintptr_t>(a), y = reinterpret_cast<std::uintptr_t>(b);
            if (a_bytes && b_bytes && (x <= y ? y - x < a_bytes : x - y < b_bytes))
                throw std::invalid_argument("Inline RoPE writable storage aliases retained source coefficients");
        }

        /** @brief Validate complete original fragment banks before any mutation.
         * @param first First writable half; @param second Matching original second half.
         * @param cosine Original cosine words; @param sine Matching sine words.
         * @param count Nonnegative pair count, including a bounded final fragment.
         * @throws std::invalid_argument On malformed storage or writable/coefficient overlap.
         */
        inline void require_pairs(float *first, float *second, const float *cosine, const float *sine, int count)
        {
            if (count < 0 || (count && (!first || !second || !cosine || !sine)))
                throw std::invalid_argument("Inline RoPE requires complete original pair/coefficient banks");
            const auto bytes = static_cast<std::size_t>(count) * sizeof(float);
            require_disjoint(first, bytes, second, bytes);
            require_disjoint(first, bytes, cosine, bytes); require_disjoint(first, bytes, sine, bytes);
            require_disjoint(second, bytes, cosine, bytes); require_disjoint(second, bytes, sine, bytes);
        }

        /** @brief Explicit scalar diagnostic fragment, never the ARM native implementation.
         * @param first Original first half; @param second Original second half.
         * @param cosine Original cosine words; @param sine Original sine words.
         * @param count Complete original pair count.
         */
        inline void scalar_pairs(float *first, float *second, const float *cosine, const float *sine, int count)
        {
            require_pairs(first, second, cosine, sine, count);
            for (int i = 0; i < count; ++i)
            {
                const float x = first[i], y = second[i];
                first[i] = x * cosine[i] - y * sine[i];
                second[i] = x * sine[i] + y * cosine[i];
            }
        }

#if defined(__AVX2__)
        /** @brief Rotate native eight-lane fragments at their original physical midpoint.
         * @param first Original first half; @param second Original second half.
         * @param cosine Original cosine words; @param sine Original sine words.
         * @param count Complete admitted pair count.
         */
        inline void avx2_pairs(float *first, float *second, const float *cosine, const float *sine, int count)
        {
            require_pairs(first, second, cosine, sine, count);
            int i = 0;
            for (; i <= count - 8; i += 8)
            {
                const auto x = _mm256_loadu_ps(first + i), y = _mm256_loadu_ps(second + i);
                const auto c = _mm256_loadu_ps(cosine + i), s = _mm256_loadu_ps(sine + i);
                _mm256_storeu_ps(first + i, _mm256_sub_ps(_mm256_mul_ps(x, c), _mm256_mul_ps(y, s)));
                _mm256_storeu_ps(second + i, _mm256_add_ps(_mm256_mul_ps(x, s), _mm256_mul_ps(y, c)));
            }
            for (; i < count; ++i)
            {
                const float x = first[i], y = second[i];
                first[i] = x * cosine[i] - y * sine[i];
                second[i] = x * sine[i] + y * cosine[i];
            }
        }
#endif
#if defined(__AVX512F__)
        /** @brief Rotate native sixteen-lane fragments without changing head or coefficient coordinates.
         * @param first Original first half; @param second Original second half.
         * @param cosine Original cosine words; @param sine Original sine words.
         * @param count Complete admitted pair count.
         */
        inline void avx512_pairs(float *first, float *second, const float *cosine, const float *sine, int count)
        {
            require_pairs(first, second, cosine, sine, count);
            int i = 0;
            for (; i <= count - 16; i += 16)
            {
                const auto x = _mm512_loadu_ps(first + i), y = _mm512_loadu_ps(second + i);
                const auto c = _mm512_loadu_ps(cosine + i), s = _mm512_loadu_ps(sine + i);
                _mm512_storeu_ps(first + i, _mm512_sub_ps(_mm512_mul_ps(x, c), _mm512_mul_ps(y, s)));
                _mm512_storeu_ps(second + i, _mm512_add_ps(_mm512_mul_ps(x, s), _mm512_mul_ps(y, c)));
            }
            for (; i < count; ++i)
            {
                const float x = first[i], y = second[i];
                first[i] = x * cosine[i] - y * sine[i];
                second[i] = x * sine[i] + y * cosine[i];
            }
        }
#endif
        /** @brief Execute the native fragment matching this compilation envelope.
         * @param first Original first half; @param second Original second half.
         * @param cosine Original cosine words; @param sine Original sine words.
         * @param count Complete admitted pair count.
         */
        inline void native_pairs(float *first, float *second, const float *cosine, const float *sine, int count)
        {
#if LLAMINAR_COMPILED_WITH_NEON
            primitives::neon_rope::rotate_pairs<primitives::neon_rope::Storage::FP32>(first, second, cosine, sine, count);
#elif defined(__AVX512F__)
            avx512_pairs(first, second, cosine, sine, count);
#elif defined(__AVX2__)
            avx2_pairs(first, second, cosine, sine, count);
#else
            scalar_pairs(first, second, cosine, sine, count);
#endif
        }

        /** @brief Check the original complete head geometry before midpoint arithmetic.
         * @param head Original writable head.
         * @param dim Nonnegative even complete rotary width.
         * @throws std::invalid_argument On malformed geometry/storage.
         */
        inline void require_head(float *head, int dim)
        {
            if (dim < 0 || dim % 2 || (dim && !head))
                throw std::invalid_argument("Inline RoPE requires a complete even head");
        }
    } // namespace rope_inline_detail

    /** @brief Explicit scalar diagnostic over one original head.
     * @param head_ptr Original in-place head; @param cos_cache Original cosine words.
     * @param sin_cache Original sine words; @param head_dim Nonnegative even width.
     */
    inline void apply_rope_to_head_inline_scalar(float *head_ptr, const float *cos_cache, const float *sin_cache, int head_dim)
    {
        rope_inline_detail::require_head(head_ptr, head_dim);
        if (head_dim) rope_inline_detail::scalar_pairs(head_ptr, head_ptr + head_dim / 2, cos_cache, sin_cache, head_dim / 2);
    }
#if defined(__AVX2__)
    /** @brief Explicit AVX2 original-head entrypoint.
     * @param head_ptr Original head; @param cos_cache Original cosine words.
     * @param sin_cache Matching sine words; @param head_dim Complete even width.
     */
    inline void apply_rope_to_head_inline_avx2(float *head_ptr, const float *cos_cache, const float *sin_cache, int head_dim)
    {
        rope_inline_detail::require_head(head_ptr, head_dim);
        if (head_dim) rope_inline_detail::avx2_pairs(head_ptr, head_ptr + head_dim / 2, cos_cache, sin_cache, head_dim / 2);
    }
#endif
#if defined(__AVX512F__)
    /** @brief Explicit AVX512 original-head entrypoint.
     * @param head_ptr Original head; @param cos_cache Original cosine words.
     * @param sin_cache Matching sine words; @param head_dim Complete even width.
     */
    inline void apply_rope_to_head_inline_avx512(float *head_ptr, const float *cos_cache, const float *sin_cache, int head_dim)
    {
        rope_inline_detail::require_head(head_ptr, head_dim);
        if (head_dim) rope_inline_detail::avx512_pairs(head_ptr, head_ptr + head_dim / 2, cos_cache, sin_cache, head_dim / 2);
    }
#endif
    /** @brief Rotate one original FP32 head using the architecture-native pair program.
     * @param head_ptr Original writable head; @param cos_cache Borrowed cosine coefficients.
     * @param sin_cache Matching sine words; @param head_dim Nonnegative even width.
     * @throws std::invalid_argument Before mutation on invalid geometry or coefficient aliases.
     */
    inline void apply_rope_to_head_inline(float *head_ptr, const float *cos_cache, const float *sin_cache, int head_dim)
    {
        rope_inline_detail::require_head(head_ptr, head_dim);
        if (head_dim) rope_inline_detail::native_pairs(head_ptr, head_ptr + head_dim / 2, cos_cache, sin_cache, head_dim / 2);
    }

    /** @brief Rotate original FP32 K rows with bounded coefficients shared across heads.
     * @param k_fp32 Original position-major writable K storage.
     * @param seq_len Nonnegative physical row count.
     * @param head_dim Positive even complete physical head width.
     * @param n_kv_heads Nonnegative head count per original row.
     * @param rope_theta Positive finite inverse-frequency base.
     * @param position_start Absolute first row position, with a representable final position.
     * @param rope_dim Zero for the full head, or an even positive prefix no larger than head_dim.
     * @throws std::invalid_argument Before mutation on invalid geometry, missing storage or source overlap.
     */
    inline void apply_rope_to_k_fp32(float *k_fp32, int seq_len, int head_dim, int n_kv_heads,
        float rope_theta, int position_start, int rope_dim = 0)
    {
        const int effective = rope_dim == 0 ? head_dim : rope_dim;
        if (seq_len < 0 || n_kv_heads < 0 || head_dim <= 0 || head_dim % 2 || effective <= 0 ||
            effective % 2 || effective > head_dim || !std::isfinite(rope_theta) || rope_theta <= 0 ||
            (seq_len && position_start > std::numeric_limits<int>::max() - (seq_len - 1)))
            throw std::invalid_argument("Inline K RoPE received invalid head/row/position geometry");
        const auto row_words = static_cast<std::size_t>(n_kv_heads) * head_dim;
        if (row_words && static_cast<std::size_t>(seq_len) > std::numeric_limits<std::size_t>::max() / sizeof(float) / row_words)
            throw std::invalid_argument("Inline K RoPE physical bank extent overflows");
        if (!seq_len || !n_kv_heads)
        {
            (void)llaminar::v2::beginWorkshare(false);
            return;
        }
        if (!k_fp32) throw std::invalid_argument("Inline K RoPE requires live original storage");
        const int half = effective / 2;
        const auto &inv_freq = primitives::get_inv_freq_cached(effective, rope_theta);
        rope_inline_detail::require_disjoint(k_fp32, static_cast<std::size_t>(seq_len) * row_words * sizeof(float),
            inv_freq.data(), inv_freq.size() * sizeof(float));
        auto work = [&]()
        {
            constexpr int pairs_per_fragment = 128;
            float cosine[pairs_per_fragment], sine[pairs_per_fragment];
#pragma omp for schedule(static)
            for (int row = 0; row < seq_len; ++row)
            {
                for (int first = 0; first < half;)
                {
                    const int count = std::min(pairs_per_fragment, half - first);
                    for (int i = 0; i < count; ++i)
                    {
                        const float angle = static_cast<float>(position_start + row) * inv_freq[static_cast<std::size_t>(first + i)];
                        cosine[i] = std::cos(angle); sine[i] = std::sin(angle);
                    }
                    for (int head = 0; head < n_kv_heads; ++head)
                    {
                        float *original = k_fp32 + static_cast<std::size_t>(row) * row_words + static_cast<std::size_t>(head) * head_dim;
                        rope_inline_detail::native_pairs(original + first, original + half + first, cosine, sine, count);
                    }
                    first += count;
                }
            }
        };
        // False standalone hints may not split an existing team into private
        // one-thread regions that would each rotate every original K row.
        OMP_WORKSHARE_REGION_IF(work, omp_in_parallel() || seq_len >= 4);
    }
} // namespace llaminar2
