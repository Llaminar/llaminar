/**
 * @file NEONRoPEPrimitives.h
 * @brief Native AArch64 rotary pairs over original floating storage coordinates.
 *
 * Each pair retains separately rounded multiply/subtract/add operations. Both
 * original halves are consumed before their stores, and bounded final fragments
 * execute the identical four-lane program rather than a different scalar/FMA
 * tail. Coefficients are borrowed from the absolute-position authority; these
 * leaves allocate nothing, retain no owner and perform no coherence publication.
 */
#pragma once
#if !defined(__aarch64__) || !defined(__ARM_NEON)
#error "Native rotary pairs require AArch64 ASIMD"
#endif
#include "tensors/NEONFP16.h"
#include <arm_neon.h>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <type_traits>

#if defined(__GNUC__) && !defined(__clang__)
#define LLAMINAR_NEON_ROPE_EXACT __attribute__((optimize("fp-contract=off", "no-associative-math")))
#else
#define LLAMINAR_NEON_ROPE_EXACT
#endif

namespace llaminar2::primitives::neon_rope
{
    /** @brief Actual retained element encoding, not an activation precision inference. */
    enum class Storage { FP32, Half, BFloat16 };
    /** @brief Native storage word selected by the explicit encoding. */
    template <Storage Format>
        requires (Format == Storage::FP32 || Format == Storage::Half || Format == Storage::BFloat16)
    using Word = std::conditional_t<Format == Storage::FP32, float, std::uint16_t>;

    /** @brief Decode exactly four original words into independent native lanes.
     * @tparam Format Original retained element encoding.
     * @param source Four readable original words, with no alignment requirement.
     * @return Source-order binary32 words, retaining portable half/BF16 payload semantics.
     */
    template <Storage Format>
    [[gnu::always_inline]] inline float32x4_t load4(const Word<Format> *source) noexcept
    {
        if constexpr (Format == Storage::FP32) return vld1q_f32(source);
        else if constexpr (Format == Storage::Half) return neon_fp16::widen(vld1_u16(source));
        else return vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(source), 16));
    }

    /** @brief Store four completed lanes through the original storage bit program.
     * @tparam Format Explicit destination encoding, identical to the source encoding.
     * @param destination Four writable original words.
     * @param values Four separately rounded FP32 rotation results.
     */
    template <Storage Format>
    [[gnu::always_inline]] inline void store4(Word<Format> *destination, float32x4_t values) noexcept
    {
        if constexpr (Format == Storage::FP32) vst1q_f32(destination, values);
        else if constexpr (Format == Storage::Half) vst1_u16(destination, neon_fp16::narrow(values));
        else
        {
            const auto bits = vreinterpretq_u32_f32(values);
            const auto rounding = vaddq_u32(vdupq_n_u32(0x7fffu), vandq_u32(vshrq_n_u32(bits, 16), vdupq_n_u32(1)));
            vst1_u16(destination, vmovn_u32(vshrq_n_u32(vaddq_u32(bits, rounding), 16)));
        }
    }

    /** @brief Rotate original halves with no FMA contraction or lane regrouping.
     * @tparam Format Explicit retained encoding.
     * @param first First writable half, disjoint from the second and coefficient banks.
     * @param second Matching writable half in original physical head coordinates.
     * @param cosine Original readable coefficient words for count pairs.
     * @param sine Matching original sine words.
     * @param count Nonnegative pair count; bounded final fragments are supported.
     * @throws std::invalid_argument Before access on invalid count, missing banks or writable/coefficient overlap.
     */
    template <Storage Format>
    LLAMINAR_NEON_ROPE_EXACT inline void rotate_pairs(
        Word<Format> *first, Word<Format> *second,
        const float *cosine, const float *sine, int count)
    {
#if defined(__clang__)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
#endif
        if (count < 0 || (count && (!first || !second || !cosine || !sine)))
            throw std::invalid_argument("Native RoPE requires complete pair and coefficient storage");
        const auto words = static_cast<std::size_t>(count) * sizeof(Word<Format>);
        const auto coefficients = static_cast<std::size_t>(count) * sizeof(float);
        const auto reject_overlap = [](const void *a, std::size_t a_bytes, const void *b, std::size_t b_bytes) {
            const auto x = reinterpret_cast<std::uintptr_t>(a), y = reinterpret_cast<std::uintptr_t>(b);
            if (a_bytes && b_bytes && (x <= y ? y - x < a_bytes : x - y < b_bytes))
                throw std::invalid_argument("Native RoPE writable halves and coefficient banks must be disjoint");
        };
        reject_overlap(first, words, second, words);
        reject_overlap(first, words, cosine, coefficients);
        reject_overlap(first, words, sine, coefficients);
        reject_overlap(second, words, cosine, coefficients);
        reject_overlap(second, words, sine, coefficients);
        int index = 0;
        for (; index <= count - 4; index += 4)
        {
            const auto x = load4<Format>(first + index), y = load4<Format>(second + index);
            const auto c = vld1q_f32(cosine + index), s = vld1q_f32(sine + index);
            const auto a = vsubq_f32(vmulq_f32(x, c), vmulq_f32(y, s));
            const auto b = vaddq_f32(vmulq_f32(x, s), vmulq_f32(y, c));
            store4<Format>(first + index, a);
            store4<Format>(second + index, b);
        }
        if (index != count)
        {
            Word<Format> x[4] = {}, y[4] = {}, a[4], b[4];
            float c[4] = {}, s[4] = {};
            const auto live = static_cast<std::size_t>(count - index);
            std::memcpy(x, first + index, live * sizeof(Word<Format>));
            std::memcpy(y, second + index, live * sizeof(Word<Format>));
            std::memcpy(c, cosine + index, live * sizeof(float));
            std::memcpy(s, sine + index, live * sizeof(float));
            const auto xv = load4<Format>(x), yv = load4<Format>(y);
            const auto cv = vld1q_f32(c), sv = vld1q_f32(s);
            store4<Format>(a, vsubq_f32(vmulq_f32(xv, cv), vmulq_f32(yv, sv)));
            store4<Format>(b, vaddq_f32(vmulq_f32(xv, sv), vmulq_f32(yv, cv)));
            std::memcpy(first + index, a, live * sizeof(Word<Format>));
            std::memcpy(second + index, b, live * sizeof(Word<Format>));
        }
    }
    /** @brief Rotate one original floating head with a checked physical midpoint.
     * @tparam Format Actual retained storage encoding.
     * @param head Original in-place head, containing head_dim words.
     * @param cosine Borrowed coefficients for head_dim/2 pairs.
     * @param sine Matching original sine coefficients.
     * @param head_dim Nonnegative even physical rotary width.
     * @throws std::invalid_argument Before access on malformed head or coefficient storage.
     */
    template <Storage Format>
    inline void rotate_head(Word<Format> *head, const float *cosine, const float *sine, int head_dim)
    {
        if (head_dim < 0 || head_dim % 2 != 0 || (head_dim && !head))
            throw std::invalid_argument("Native RoPE requires a nonnegative even head and live storage");
        if (!head_dim) return;
        rotate_pairs<Format>(head, head + head_dim / 2, cosine, sine, head_dim / 2);
    }
} // namespace llaminar2::primitives::neon_rope
#undef LLAMINAR_NEON_ROPE_EXACT
