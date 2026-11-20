/**
 * @file CPUAttentionKeyQ8.h
 * @brief Allocation-free native NEON/AVX2/AVX-512 kernels for the shared AQ8 key format.
 *
 * The cache owns the immutable FP32 anchor. These kernels encode its residual
 * directly from production projection rows, and reconstruct one head directly
 * from compressed blocks. They do not allocate, materialize a cache shadow,
 * change a basis, or publish ring state. Rational interval comparisons preserve
 * the scalar/CUDA/ROCm encoder's exact code decisions without sqrt/cbrt calls.
 */
#pragma once

#include "tensors/BlockStructures.h"
#include "utils/CPUFeatures.h"

#include <algorithm>
#include <bit>
#include <cstring>
#if LLAMINAR_COMPILED_WITH_NEON
#include <arm_neon.h>
#elif defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif
#include <limits>
#include <span>
#include <stdexcept>

namespace llaminar2::cpu::attention_key_q8
{
/** @brief Explicit ISA selection for conformance and isolated profiling. */
enum class ISA { Automatic, AVX2, AVX512, NEON };

/** @brief Resolve one native leaf without substituting another requested implementation.
 * @param requested Automatic policy or one explicit compiled/native instruction family.
 * @return Authenticated implementation selected before arithmetic.
 * @throws std::invalid_argument For scalar, unknown, foreign or unavailable routes.
 * @throws std::runtime_error For an incompatible compiler instruction envelope.
 */
inline ISA resolveISA(ISA requested)
{
    // Authenticate the entire compiler envelope even for an explicitly narrower
    // leaf: a DotProd binary may contain those instructions outside this codec.
    const auto active = activeISALevel();
    if (requested == ISA::Automatic)
    {
        switch (active)
        {
        case ISALevel::AVX512: requested = ISA::AVX512; break;
        case ISALevel::AVX2: requested = ISA::AVX2; break;
        case ISALevel::NEON:
        case ISALevel::NEONDotProd: requested = ISA::NEON; break;
        default: throw std::invalid_argument("CPU AQ8 has no scalar production execution mode");
        }
    }
#if defined(__AVX512F__)
    if ((requested == ISA::Automatic || requested == ISA::AVX512) && cpu_supports_avx512())
        return ISA::AVX512;
#endif
#if defined(__AVX2__)
    if ((requested == ISA::Automatic || requested == ISA::AVX2) && cpu_supports_avx2())
        return ISA::AVX2;
#endif
#if LLAMINAR_COMPILED_WITH_NEON
    if (requested == ISA::NEON && cpu_supports_neon()) return ISA::NEON;
#endif
    throw std::invalid_argument("CPU AQ8 requires an available compiled native NEON, AVX2 or AVX-512 ISA");
}

namespace detail
{
#if defined(__GNUC__) && !defined(__clang__)
#define LLAMINAR_AQ8_ARITHMETIC __attribute__((optimize("fp-contract=off", "no-associative-math")))
#else
#define LLAMINAR_AQ8_ARITHMETIC
#endif

/** @brief Compare bounded borrowed byte envelopes without overflowing their end addresses.
 * @param a First live root; @param an Its complete physical byte count.
 * @param b Second live root; @param bn Its complete physical byte count.
 * @return Whether the nonempty ranges intersect; no ownership or capacity ledger is created.
 */
inline bool overlaps(const void *a, std::size_t an, const void *b, std::size_t bn) noexcept
{
    if (!an || !bn) return false;
    const auto x = reinterpret_cast<std::uintptr_t>(a), y = reinterpret_cast<std::uintptr_t>(b);
    return x <= y ? y - x < an : x - y < bn;
}

#if LLAMINAR_COMPILED_WITH_NEON
/** @brief Encode native four-lane residuals using the exact shared rational decisions.
 * @tparam D Complete physical head width, divisible by eight.
 * @param source Original finite source head; @param anchor Immutable request basis.
 * @param output Sole disjoint destination block, untouched until all residuals validate.
 * @throws std::domain_error If an anchored residual is nonfinite, before any publication.
 * @details The finite/maximum pass precedes all stores. Seven independent lane
 * decisions use the oracle's separately rounded products, including subnormal
 * and overflowing rational comparisons. Two native four-lane chains overlap
 * their dependent searches; one exact eight-byte store preserves neighbours.
 * No roots, decoded head or codebook mirrors are needed.
 */
template <int D>
LLAMINAR_AQ8_ARITHMETIC inline void quantizeNEON(
    const float *source, const float *anchor, AttentionKeyQ8Block<D> &output)
{
#if defined(__clang__)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
#endif
    static_assert(D % 8 == 0 && std::endian::native == std::endian::little);
    auto maximum = vdupq_n_f32(0.0f);
    auto valid = vdupq_n_u32(~std::uint32_t{0});
    const auto largest = vdupq_n_f32(std::numeric_limits<float>::max());
    for (int coordinate = 0; coordinate < D; coordinate += 4)
    {
        const auto residual = vsubq_f32(vld1q_f32(source + coordinate), vld1q_f32(anchor + coordinate));
        const auto magnitude = vabsq_f32(residual);
        valid = vandq_u32(valid, vcleq_f32(magnitude, largest));
        maximum = vmaxq_f32(maximum, magnitude);
    }
    if (vminvq_u32(valid) != ~std::uint32_t{0})
        throw std::domain_error("CPU AQ8 requires finite anchored key residuals");
    const float max_abs = vmaxvq_f32(maximum);
    output.quadratic_scale = max_abs * AttentionKeyQ8Block<D>::INVERSE_MAX_CODE_SQUARED;
    if (max_abs == 0.0f)
    {
        std::fill(std::begin(output.codes), std::end(output.codes), std::int8_t{0});
        return;
    }
    const auto vmax = vdupq_n_f32(max_abs);
    // Consume both original fragments before publishing either. Merely asking
    // for an outer-loop unroll leaves two serial chains separated by a store
    // that may alias the next loads in compiler analysis. Explicit independent
    // bounds let the native scheduler overlap the seven comparisons safely.
    for (int coordinate = 0; coordinate < D; coordinate += 8)
    {
        const auto residual_lo = vsubq_f32(vld1q_f32(source + coordinate), vld1q_f32(anchor + coordinate));
        const auto residual_hi = vsubq_f32(vld1q_f32(source + coordinate + 4), vld1q_f32(anchor + coordinate + 4));
        const auto magnitude_lo = vmulq_n_f32(vabsq_f32(residual_lo), 254.0f * 254.0f);
        const auto magnitude_hi = vmulq_n_f32(vabsq_f32(residual_hi), 254.0f * 254.0f);
        auto lower_lo = vdupq_n_u32(0), lower_hi = lower_lo;
        for (int bit = 6; bit >= 0; --bit)
        {
            const auto flag = vdupq_n_u32(1U << bit);
            const auto candidate_lo = vorrq_u32(lower_lo, flag), candidate_hi = vorrq_u32(lower_hi, flag);
            const auto odd_lo = vsubq_u32(vshlq_n_u32(candidate_lo, 1), vdupq_n_u32(1));
            const auto odd_hi = vsubq_u32(vshlq_n_u32(candidate_hi, 1), vdupq_n_u32(1));
            const auto boundary_lo = vmulq_f32(vmax, vcvtq_f32_u32(vmulq_u32(odd_lo, odd_lo)));
            const auto boundary_hi = vmulq_f32(vmax, vcvtq_f32_u32(vmulq_u32(odd_hi, odd_hi)));
            lower_lo = vbslq_u32(vcltq_f32(magnitude_lo, boundary_lo), lower_lo, candidate_lo);
            lower_hi = vbslq_u32(vcltq_f32(magnitude_hi, boundary_hi), lower_hi, candidate_hi);
        }
        const auto negative_lo = vreinterpretq_s32_u32(vcltq_f32(residual_lo, vdupq_n_f32(0)));
        const auto negative_hi = vreinterpretq_s32_u32(vcltq_f32(residual_hi, vdupq_n_f32(0)));
        const auto codes_lo = vsubq_s32(veorq_s32(vreinterpretq_s32_u32(lower_lo), negative_lo), negative_lo);
        const auto codes_hi = vsubq_s32(veorq_s32(vreinterpretq_s32_u32(lower_hi), negative_hi), negative_hi);
        vst1_s8(output.codes + coordinate, vmovn_s16(vcombine_s16(vmovn_s32(codes_lo), vmovn_s32(codes_hi))));
    }
}

/** @brief Decode native lanes in the exact multiply/multiply/add physical program.
 * @tparam D Complete physical head width.
 * @param input Validated cache-owned block; @param anchor Immutable request basis.
 * @param output FP32 destination disjoint from both retained cache authorities.
 * @details Input publication/import owns scale and reserved-code validation;
 * this leaf neither rescans cache metadata nor creates a decoded-cache mirror.
 */
template <int D>
LLAMINAR_AQ8_ARITHMETIC inline void dequantizeNEON(
    const AttentionKeyQ8Block<D> &input, const float *anchor, float *output)
{
#if defined(__clang__)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
#endif
    static_assert(D % 8 == 0 && std::endian::native == std::endian::little);
    const float scale = input.quadratic_scale;
    for (int coordinate = 0; coordinate < D; coordinate += 8)
    {
        // A complete eight-byte group widens directly with SSHLL/SSHLL2. The
        // four-byte form otherwise encourages a table-expansion constant that
        // the caller must preserve across ISA admission. Retain the immutable
        // scale once per head, not as repeated alias-sensitive metadata loads.
        const auto shorts = vmovl_s8(vld1_s8(input.codes + coordinate));
        const auto lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(shorts)));
        const auto hi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(shorts)));
        const auto residual_lo = vmulq_f32(vmulq_n_f32(lo, scale), vabsq_f32(lo));
        const auto residual_hi = vmulq_f32(vmulq_n_f32(hi, scale), vabsq_f32(hi));
        vst1q_f32(output + coordinate, vaddq_f32(residual_lo, vld1q_f32(anchor + coordinate)));
        vst1q_f32(output + coordinate + 4, vaddq_f32(residual_hi, vld1q_f32(anchor + coordinate + 4)));
    }
}
#endif

#if defined(__AVX2__)
/**
 * @brief Encode a complete head using eight independent rational searches.
 * @param source Original finite FP32 key coordinates.
 * @param anchor Immutable finite FP32 basis for this cache head.
 * @param output Destination block; no bytes are published for invalid inputs.
 */
template <int D>
LLAMINAR_AQ8_ARITHMETIC inline void quantizeAVX2(
    const float *source, const float *anchor, AttentionKeyQ8Block<D> &output)
{
#if defined(__clang__)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
#endif
    const __m256 sign = _mm256_set1_ps(-0.0f);
    __m256 maximum = _mm256_setzero_ps();
    __m256 valid = _mm256_castsi256_ps(_mm256_set1_epi32(-1));
    const __m256 largest = _mm256_set1_ps(std::numeric_limits<float>::max());
    for (int coordinate = 0; coordinate < D; coordinate += 8)
    {
        const __m256 residual = _mm256_sub_ps(_mm256_loadu_ps(source + coordinate),
                                             _mm256_loadu_ps(anchor + coordinate));
        const __m256 magnitude = _mm256_andnot_ps(sign, residual);
        valid = _mm256_and_ps(valid, _mm256_cmp_ps(magnitude, largest, _CMP_LE_OQ));
        maximum = _mm256_max_ps(maximum, magnitude);
    }
    if (_mm256_movemask_ps(valid) != 255)
        throw std::domain_error("CPU AQ8 requires finite anchored key residuals");
    __m128 reduced = _mm_max_ps(_mm256_castps256_ps128(maximum), _mm256_extractf128_ps(maximum, 1));
    reduced = _mm_max_ps(reduced, _mm_movehl_ps(reduced, reduced));
    reduced = _mm_max_ss(reduced, _mm_shuffle_ps(reduced, reduced, 0x55));
    const float max_abs = _mm_cvtss_f32(reduced);
    output.quadratic_scale = max_abs * AttentionKeyQ8Block<D>::INVERSE_MAX_CODE_SQUARED;
    if (max_abs == 0.0f)
    {
        std::fill(std::begin(output.codes), std::end(output.codes), int8_t{0});
        return;
    }
    const __m256 vmax = _mm256_set1_ps(max_abs);
    const __m256 denominator = _mm256_set1_ps(254.0f * 254.0f);
    const __m256i one = _mm256_set1_epi32(1);
    for (int coordinate = 0; coordinate < D; coordinate += 8)
    {
        const __m256 residual = _mm256_sub_ps(_mm256_loadu_ps(source + coordinate),
                                             _mm256_loadu_ps(anchor + coordinate));
        const __m256 magnitude = _mm256_mul_ps(_mm256_andnot_ps(sign, residual), denominator);
        __m256i lower = _mm256_setzero_si256();
        // Set one code bit at a time, from most to least significant. Candidate
        // c begins at the same rational boundary (2*c-1)^2 used by the scalar
        // lower_bound. This needs only one live bound and one blend per step;
        // no division, approximate root, lane ballot, or retry can change ties.
        // AVX2 has only sixteen vector registers. Unrolling all seven steps
        // hoists seven broadcast constants and spills two of them into every
        // head's stack frame on GCC. Partial unrolling keeps the whole search
        // register-resident; this is an ISA constraint, not an arithmetic change.
#if defined(__clang__)
#pragma clang loop unroll_count(2)
#elif defined(__GNUC__)
#pragma GCC unroll 2
#endif
        for (int bit = 6; bit >= 0; --bit)
        {
            const __m256i candidate = _mm256_or_si256(lower, _mm256_set1_epi32(1 << bit));
            const __m256i odd = _mm256_sub_epi32(_mm256_slli_epi32(candidate, 1), one);
            const __m256 boundary = _mm256_mul_ps(vmax, _mm256_cvtepi32_ps(_mm256_mullo_epi32(odd, odd)));
            const __m256i less = _mm256_castps_si256(_mm256_cmp_ps(magnitude, boundary, _CMP_LT_OQ));
            lower = _mm256_blendv_epi8(candidate, lower, less);
        }
        const __m256i negative = _mm256_castps_si256(_mm256_cmp_ps(residual, _mm256_setzero_ps(), _CMP_LT_OQ));
        const __m256i codes = _mm256_sub_epi32(_mm256_xor_si256(lower, negative), negative);
        // Packs operate within 128-bit lanes. Collapse the halves explicitly
        // before the second pack so coordinates remain in their original order.
        const __m128i short_codes = _mm_packs_epi32(_mm256_castsi256_si128(codes),
                                                  _mm256_extracti128_si256(codes, 1));
        _mm_storel_epi64(reinterpret_cast<__m128i *>(output.codes + coordinate),
                         _mm_packs_epi16(short_codes, short_codes));
    }
}

/** @brief Decode one head with the oracle's exact multiply/multiply/add order. */
template <int D>
LLAMINAR_AQ8_ARITHMETIC inline void dequantizeAVX2(
    const AttentionKeyQ8Block<D> &input, const float *anchor, float *output)
{
#if defined(__clang__)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
#endif
    const __m256 scale = _mm256_set1_ps(input.quadratic_scale);
    const __m256 sign = _mm256_set1_ps(-0.0f);
    for (int coordinate = 0; coordinate < D; coordinate += 8)
    {
        const __m128i raw = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(input.codes + coordinate));
        const __m256 code = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(raw));
        const __m256 residual = _mm256_mul_ps(_mm256_mul_ps(scale, code), _mm256_andnot_ps(sign, code));
        _mm256_storeu_ps(output + coordinate, _mm256_add_ps(residual, _mm256_loadu_ps(anchor + coordinate)));
    }
}
#endif

#if defined(__AVX512F__)
/** @brief Sixteen-lane counterpart of the exact AVX2 encoder. */
template <int D>
LLAMINAR_AQ8_ARITHMETIC inline void quantizeAVX512(
    const float *source, const float *anchor, AttentionKeyQ8Block<D> &output)
{
#if defined(__clang__)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
#endif
    __m512 maximum = _mm512_setzero_ps();
    __mmask16 valid = 0xffff;
    const __m512 largest = _mm512_set1_ps(std::numeric_limits<float>::max());
    for (int coordinate = 0; coordinate < D; coordinate += 16)
    {
        const __m512 residual = _mm512_sub_ps(_mm512_loadu_ps(source + coordinate), _mm512_loadu_ps(anchor + coordinate));
        const __m512 magnitude = _mm512_abs_ps(residual);
        valid &= _mm512_cmp_ps_mask(magnitude, largest, _CMP_LE_OQ);
        maximum = _mm512_max_ps(maximum, magnitude);
    }
    if (valid != 0xffff)
        throw std::domain_error("CPU AQ8 requires finite anchored key residuals");
    const float max_abs = _mm512_reduce_max_ps(maximum);
    output.quadratic_scale = max_abs * AttentionKeyQ8Block<D>::INVERSE_MAX_CODE_SQUARED;
    if (max_abs == 0.0f)
    {
        std::fill(std::begin(output.codes), std::end(output.codes), int8_t{0});
        return;
    }
    const __m512 vmax = _mm512_set1_ps(max_abs);
    const __m512 denominator = _mm512_set1_ps(254.0f * 254.0f);
    const __m512i one = _mm512_set1_epi32(1);
    for (int coordinate = 0; coordinate < D; coordinate += 16)
    {
        const __m512 residual = _mm512_sub_ps(_mm512_loadu_ps(source + coordinate), _mm512_loadu_ps(anchor + coordinate));
        const __m512 magnitude = _mm512_mul_ps(_mm512_abs_ps(residual), denominator);
        __m512i lower = _mm512_setzero_si512();
        // The AVX2 and AVX-512 searches use identical rational boundaries;
        // only the SIMD mask representation differs.
        for (int bit = 6; bit >= 0; --bit)
        {
            const __m512i candidate = _mm512_or_si512(lower, _mm512_set1_epi32(1 << bit));
            const __m512i odd = _mm512_sub_epi32(_mm512_slli_epi32(candidate, 1), one);
            const __m512 boundary = _mm512_mul_ps(vmax, _mm512_cvtepi32_ps(_mm512_mullo_epi32(odd, odd)));
            const __mmask16 less = _mm512_cmp_ps_mask(magnitude, boundary, _CMP_LT_OQ);
            lower = _mm512_mask_mov_epi32(candidate, less, lower);
        }
        const __mmask16 negative = _mm512_cmp_ps_mask(residual, _mm512_setzero_ps(), _CMP_LT_OQ);
        const __m512i codes = _mm512_mask_sub_epi32(lower, negative, _mm512_setzero_si512(), lower);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(output.codes + coordinate), _mm512_cvtsepi32_epi8(codes));
    }
}

/** @brief Decode sixteen coordinates without an FP32 cache shadow or FMA. */
template <int D>
LLAMINAR_AQ8_ARITHMETIC inline void dequantizeAVX512(
    const AttentionKeyQ8Block<D> &input, const float *anchor, float *output)
{
#if defined(__clang__)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
#endif
    const __m512 scale = _mm512_set1_ps(input.quadratic_scale);
    for (int coordinate = 0; coordinate < D; coordinate += 16)
    {
        const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i *>(input.codes + coordinate));
        const __m512 code = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(raw));
        const __m512 residual = _mm512_mul_ps(_mm512_mul_ps(scale, code), _mm512_abs_ps(code));
        _mm512_storeu_ps(output + coordinate, _mm512_add_ps(residual, _mm512_loadu_ps(anchor + coordinate)));
    }
}
#endif
#undef LLAMINAR_AQ8_ARITHMETIC
} // namespace detail

/**
 * @brief Encode one anchored native key head through an authenticated CPU ISA.
 * @param source Exactly D finite source coordinates.
 * @param anchor Exactly D immutable finite anchor coordinates.
 * @param output Sole destination physical block.
 * @param isa Explicit or runtime-selected implementation.
 * @throws std::invalid_argument For null/aliased banks or unavailable ISA intent.
 * @throws std::domain_error For nonfinite residuals, before any block byte is authored.
 */
template <int D>
inline void quantize(std::span<const float, D> source, std::span<const float, D> anchor,
                     AttentionKeyQ8Block<D> &output, ISA isa = ISA::Automatic)
{
    static_assert(D == 64 || D == 128 || D == 256);
    if (!source.data() || !anchor.data())
        throw std::invalid_argument("CPU AQ8 requires non-null source and anchor");
    if (detail::overlaps(&output, sizeof(output), source.data(), D * sizeof(float)) ||
        detail::overlaps(&output, sizeof(output), anchor.data(), D * sizeof(float)))
        throw std::invalid_argument("CPU AQ8 output aliases an original source or immutable anchor");
    const ISA resolved = resolveISA(isa);
#if LLAMINAR_COMPILED_WITH_NEON
    if (resolved == ISA::NEON)
        return detail::quantizeNEON<D>(source.data(), anchor.data(), output);
#endif
#if defined(__AVX512F__)
    if (resolved == ISA::AVX512)
        return detail::quantizeAVX512<D>(source.data(), anchor.data(), output);
#endif
#if defined(__AVX2__)
    if (resolved == ISA::AVX2)
        return detail::quantizeAVX2<D>(source.data(), anchor.data(), output);
#endif
    throw std::logic_error("CPU AQ8 ISA resolution and compiled kernels disagree");
}

/**
 * @brief Reconstruct one head, preserving the scalar physical decoding order.
 * @param input Validated cache-owned AQ8 block (including nonnegative scale).
 * @param anchor Its immutable request basis.
 * @param output Caller-owned D-coordinate span, disjoint from the compressed block and immutable anchor; may be reusable row scratch.
 * @param isa Explicit or runtime-selected implementation.
 *
 * @throws std::invalid_argument For null/aliased banks or unavailable ISA intent.
 * @details Physical payload validation belongs to publication/import, not every hot
 * cache read. No full-buffer NaN scan or reserved-code scan is added here.
 */
template <int D>
inline void dequantize(const AttentionKeyQ8Block<D> &input, std::span<const float, D> anchor,
                       std::span<float, D> output, ISA isa = ISA::Automatic)
{
    static_assert(D == 64 || D == 128 || D == 256);
    if (!anchor.data() || !output.data())
        throw std::invalid_argument("CPU AQ8 requires non-null anchor and output");
    if (detail::overlaps(output.data(), D * sizeof(float), &input, sizeof(input)) ||
        detail::overlaps(output.data(), D * sizeof(float), anchor.data(), D * sizeof(float)))
        throw std::invalid_argument("CPU AQ8 decoded output aliases compressed storage or its immutable anchor");
    const ISA resolved = resolveISA(isa);
#if LLAMINAR_COMPILED_WITH_NEON
    if (resolved == ISA::NEON)
        return detail::dequantizeNEON<D>(input, anchor.data(), output.data());
#endif
#if defined(__AVX512F__)
    if (resolved == ISA::AVX512)
        return detail::dequantizeAVX512<D>(input, anchor.data(), output.data());
#endif
#if defined(__AVX2__)
    if (resolved == ISA::AVX2)
        return detail::dequantizeAVX2<D>(input, anchor.data(), output.data());
#endif
    throw std::logic_error("CPU AQ8 ISA resolution and compiled kernels disagree");
}
} // namespace llaminar2::cpu::attention_key_q8
