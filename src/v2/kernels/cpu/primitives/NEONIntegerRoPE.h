/**
 * @file NEONIntegerRoPE.h
 * @brief Native widened-integer and fixed-scale RoPE over original KV coordinates.
 *
 * Q8-ratio and Q16-ratio programs retain their distinct metadata, bias/shift and
 * clamp rules. All element products are native widened integer operations; no
 * int32 wraparound is used to approximate the specified arithmetic. Conversion
 * outputs are disjoint; exact Q16 reuse defers every scale write until original
 * payload/scale reads finish. No full-head mirror, allocation or coherence
 * publication is introduced by these synchronous borrowed leaves.
 */
#pragma once
#include "NEONQuantizedRoPE.h"
#include <limits>

#if defined(__GNUC__) && !defined(__clang__)
#define LLAMINAR_NEON_INTEGER_ROPE_EXACT __attribute__((optimize("fp-contract=off", "no-associative-math")))
#else
#define LLAMINAR_NEON_INTEGER_ROPE_EXACT
#endif

namespace llaminar2::primitives::neon_rope
{
    /** @brief Physical Q16 encodings accepted by conversion and uniform-scale rotation. */
    template <typename Block>
    concept Q16Block = std::is_same_v<Block, Q16_1Block> || std::is_same_v<Block, Q16_1Block_64> ||
        std::is_same_v<Block, Q16_1Block_128>;
    /** @brief Existing Q15 bias programs, not interchangeable rounding descriptions. */
    enum class Q15Bias { PositiveHalf, SignedHalf };
    /** @brief Explicit allowed input/output ownership relation. */
    enum class Reuse { Disjoint, ExactOrDisjoint };
    /** @brief Four original first/second-half integer results. */
    struct IntegerPair { int32x4_t first, second; };
    /** @brief Original physical head geometry, without allocating or extending an owner. */
    struct IntegerHead { int input_blocks, output_blocks, half; std::size_t input_bytes, output_bytes; };

    /** @brief Reject nonempty overlapping byte envelopes without overflowing end pointers.
     * @param a First bank; @param a_bytes Complete first extent.
     * @param b Second bank; @param b_bytes Complete second extent.
     */
    inline void require_integer_disjoint(const void *a, std::size_t a_bytes, const void *b, std::size_t b_bytes)
    {
        const auto x = reinterpret_cast<std::uintptr_t>(a), y = reinterpret_cast<std::uintptr_t>(b);
        if (a_bytes && b_bytes && (x <= y ? y - x < a_bytes : x - y < b_bytes))
            throw std::invalid_argument("Native integer RoPE storage aliases a retained input/coefficient authority");
    }

    /** @brief Validate complete original blocks and output/coefficient ownership before writes.
     * @tparam In Original physical input block; @tparam Out Actual Q16 output block.
     * @param input Original readable input bank; @param output Writable output bank.
     * @param dim Positive representable complete logical head width.
     * @param cosine Original Q15 cosine plane; @param sine Matching sine plane.
     * @param reuse Explicit input/output ownership rule.
     * @return Checked original block counts, midpoint and byte extents.
     */
    template <QuantizedRoPEBlock In, Q16Block Out>
    inline IntegerHead require_integer_head(const In *input, Out *output, int dim,
        const std::int16_t *cosine, const std::int16_t *sine, Reuse reuse)
    {
        constexpr int in_width = static_cast<int>(In::BLOCK_SIZE), out_width = static_cast<int>(Out::BLOCK_SIZE);
        if (reuse != Reuse::Disjoint && reuse != Reuse::ExactOrDisjoint)
            throw std::invalid_argument("Native integer RoPE requires an explicit storage-reuse policy");
        if (!input || !output || !cosine || !sine || dim <= 0 || dim % in_width || dim % out_width)
            throw std::invalid_argument("Native integer RoPE requires complete original head/block geometry");
        const IntegerHead geometry{dim / in_width, dim / out_width, dim / 2,
            std::size_t(dim / in_width) * sizeof(In), std::size_t(dim / out_width) * sizeof(Out)};
        if (reuse != Reuse::ExactOrDisjoint || static_cast<const void *>(input) != static_cast<void *>(output))
            require_integer_disjoint(input, geometry.input_bytes, output, geometry.output_bytes);
        else if constexpr (!std::is_same_v<In, Out>)
            throw std::invalid_argument("Native integer RoPE exact reuse requires identical physical encodings");
        require_integer_disjoint(output, geometry.output_bytes, cosine, std::size_t(geometry.half) * sizeof(std::int16_t));
        require_integer_disjoint(output, geometry.output_bytes, sine, std::size_t(geometry.half) * sizeof(std::int16_t));
        return geometry;
    }

    /** @brief Widen exactly four signed bytes without an overread.
     * @param source Four readable original payload bytes.
     * @return Four exact original integer words.
     */
    [[gnu::always_inline]] inline int32x4_t load_integer4(const std::int8_t *source) noexcept
    {
        std::int32_t word;
        std::memcpy(&word, source, sizeof(word));
        return vmovl_s16(vget_low_s16(vmovl_s8(vreinterpret_s8_s32(vdup_n_s32(word)))));
    }
    /** @brief Widen exactly four signed half words.
     * @param source Four readable original payload words.
     * @return Four exact original integers, including -32768.
     */
    [[gnu::always_inline]] inline int32x4_t load_integer4(const std::int16_t *source) noexcept
    { return vmovl_s16(vld1_s16(source)); }

    /** @brief Apply the original scalar metadata ties-away ratio program with checked conversion.
     * @param source_scale Finite original scale.
     * @param destination_scale Positive finite target scale.
     * @param factor Exact 256 or 65536 fixed-point multiplier.
     * @return Representable signed fixed-point ratio.
     */
    LLAMINAR_NEON_INTEGER_ROPE_EXACT inline std::int32_t ratio(float source_scale, float destination_scale, float factor)
    {
#if defined(__clang__)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
#endif
        if (!std::isfinite(source_scale) || !std::isfinite(destination_scale) || destination_scale <= 0)
            throw std::invalid_argument("Native integer RoPE requires finite source and positive destination scales");
        const float scaled = (source_scale / destination_scale) * factor;
        if (!std::isfinite(scaled) || scaled < -2147483648.0f || scaled >= 2147483648.0f)
            throw std::invalid_argument("Native integer RoPE fixed-point scale ratio is not representable");
        return static_cast<std::int32_t>(std::round(scaled));
    }

    /** @brief Derive a Q8 ratio without inventing values for a numerically empty block.
     * @param block Original complete input block.
     * @param common_scale Positive admitted common scale.
     * @return Original ties-away Q8 ratio; an all-zero payload has exact scaled value zero.
     */
    inline std::int32_t q8_ratio(const Q8_1Block &block, float common_scale)
    {
        const float original_scale = QuantizedMath<Q8_1Block>::scale(block);
        const auto nonzero = vorrq_s8(vld1q_s8(block.qs), vld1q_s8(block.qs + 16));
        if (vmaxvq_u8(vreinterpretq_u8_s8(nonzero)) == 0) return 0;
        return ratio(original_scale, common_scale, 256.0f);
    }

    /** @brief Rescale four Q16 payload words through exact widened products and arithmetic shift.
     * @param values Original signed payload words.
     * @param fixed_ratio Original signed Q16 metadata ratio.
     * @return Four exact `(value*ratio)>>16` words without int32 product wraparound.
     */
    [[gnu::always_inline]] inline int32x4_t rescale_q16(int32x4_t values, std::int32_t fixed_ratio) noexcept
    {
        return vcombine_s32(vshrn_n_s64(vmull_n_s32(vget_low_s32(values), fixed_ratio), 16),
                            vshrn_n_s64(vmull_n_s32(vget_high_s32(values), fixed_ratio), 16));
    }

    /** @brief Apply one exact Q15 bias/shift to two widened values before final range clamping.
     * @tparam Bias Explicit original positive or signed bias program.
     * @param value Two exact native widened integer results.
     * @return Saturating int32 narrowing; every caller then applies its much smaller Q16 bound.
     */
    template <Q15Bias Bias>
    [[gnu::always_inline]] inline int32x2_t finish_q15(int64x2_t value) noexcept
    {
        static_assert(Bias == Q15Bias::PositiveHalf || Bias == Q15Bias::SignedHalf);
        auto bias = vdupq_n_s64(1LL << 14);
        if constexpr (Bias == Q15Bias::SignedHalf)
            bias = vbslq_s64(vcltq_s64(value, vdupq_n_s64(0)), vdupq_n_s64(-(1LL << 14)), bias);
        return vqmovn_s64(vshrq_n_s64(vaddq_s64(value, bias), 15));
    }

    /** @brief Rotate four original integer pairs using native 64-bit intermediates.
     * @tparam Bias Exact legacy signed-bias or ordinary positive-bias Q15 program.
     * @tparam Limit Fixed-cache 16383 or full Q16 32767 saturation bound.
     * @param x First-half original scaled words; @param y Second-half original scaled words.
     * @param cosine Four original cosine words; @param sine Matching original sine words.
     * @return Two saturated original-order four-lane results.
     */
    template <Q15Bias Bias, int Limit>
    [[gnu::always_inline]] inline IntegerPair integer_rotate4(int32x4_t x, int32x4_t y,
        const std::int16_t *cosine, const std::int16_t *sine) noexcept
    {
        static_assert(Limit == 16383 || Limit == 32767);
        static_assert(Bias == Q15Bias::PositiveHalf || Bias == Q15Bias::SignedHalf);
        const auto c = load_integer4(cosine), s = load_integer4(sine);
        const auto finish = [](int64x2_t value) { return finish_q15<Bias>(value); };
        const auto xl = vget_low_s32(x), xh = vget_high_s32(x), yl = vget_low_s32(y), yh = vget_high_s32(y);
        const auto cl = vget_low_s32(c), ch = vget_high_s32(c), sl = vget_low_s32(s), sh = vget_high_s32(s);
        auto a = vcombine_s32(finish(vsubq_s64(vmull_s32(xl, cl), vmull_s32(yl, sl))),
                             finish(vsubq_s64(vmull_s32(xh, ch), vmull_s32(yh, sh))));
        auto b = vcombine_s32(finish(vaddq_s64(vmull_s32(xl, sl), vmull_s32(yl, cl))),
                             finish(vaddq_s64(vmull_s32(xh, sh), vmull_s32(yh, ch))));
        const auto clamp = [](int32x4_t value) { return vmaxq_s32(vdupq_n_s32(-Limit), vminq_s32(vdupq_n_s32(Limit), value)); };
        return {clamp(a), clamp(b)};
    }

    /** @brief Rotate exact byte/Q15/ratio products without an overflowing scaled-int32 intermediate.
     * @tparam Bias Original normalized or legacy bias program.
     * @param a Four original first-half payload bytes; @param b Matching second-half bytes.
     * @param ra First signed Q8 ratio; @param rb Matching second signed ratio.
     * @param cosine Four original Q15 cosine words; @param sine Matching sine words.
     * @return Four saturated result pairs, including every representable signed ratio endpoint.
     * @details Byte times Q15 fits int32. Multiplication by its signed ratio then
     * widens to int64 before either sum. Even ratio=-2^24 with byte=-128 cannot
     * wrap; reassociation is exact because no integer rounding occurs before Q15.
     */
    template <Q15Bias Bias>
    [[gnu::always_inline]] inline IntegerPair q8_rotate4(const std::int8_t *a, const std::int8_t *b,
        std::int32_t ra, std::int32_t rb, const std::int16_t *cosine, const std::int16_t *sine) noexcept
    {
        const auto x = load_integer4(a), y = load_integer4(b), c = load_integer4(cosine), s = load_integer4(sine);
        const auto xc = vmulq_s32(x, c), ys = vmulq_s32(y, s), xs = vmulq_s32(x, s), yc = vmulq_s32(y, c);
        auto first = vcombine_s32(
            finish_q15<Bias>(vsubq_s64(vmull_n_s32(vget_low_s32(xc), ra), vmull_n_s32(vget_low_s32(ys), rb))),
            finish_q15<Bias>(vsubq_s64(vmull_n_s32(vget_high_s32(xc), ra), vmull_n_s32(vget_high_s32(ys), rb))));
        auto second = vcombine_s32(
            finish_q15<Bias>(vaddq_s64(vmull_n_s32(vget_low_s32(xs), ra), vmull_n_s32(vget_low_s32(yc), rb))),
            finish_q15<Bias>(vaddq_s64(vmull_n_s32(vget_high_s32(xs), ra), vmull_n_s32(vget_high_s32(yc), rb))));
        const auto clamp = [](int32x4_t v) { return vmaxq_s32(vdupq_n_s32(-32767), vminq_s32(vdupq_n_s32(32767), v)); };
        return {clamp(first), clamp(second)};
    }

    /** @brief Store four original logical Q16 words without crossing a physical block.
     * @tparam Out Actual original output encoding.
     * @param output Original writable bank; @param index Original logical word coordinate.
     * @param values Four already saturated integer results.
     */
    template <Q16Block Out>
    [[gnu::always_inline]] inline void store_integer4(Out *output, int index, int32x4_t values) noexcept
    {
        constexpr int width = static_cast<int>(Out::BLOCK_SIZE);
        vst1_s16(output[index / width].qs + index % width, vqmovn_s32(values));
    }

    /** @brief Complete every output block's metadata after all original input scale reads.
     * @tparam Out Actual retained output block.
     * @param output Sole writable output bank; @param blocks Complete physical block count.
     * @param scale Published common finite scale.
     */
    template <Q16Block Out> inline void finish_integer_head(Out *output, int blocks, float scale)
    {
        constexpr int width = static_cast<int>(Out::BLOCK_SIZE);
        for (int block = 0; block < blocks; ++block)
        {
            auto sum = vdupq_n_s32(0);
            for (int i = 0; i < width; i += 4) sum = vaddq_s32(sum, load_integer4(output[block].qs + i));
            output[block].d = scale;
            output[block].sum_qs = vaddvq_s32(sum);
        }
    }

    /** @brief Distinct complete Q8 conversion policies, not interchangeable bias/safety choices. */
    enum class Q8Conversion { Normalized, LegacySigned };
    /** @brief Explicit legacy/common and stored/output scale results from one calculation. */
    struct Q8Scales { float common, output; };

    /** @brief Convert an original Q8 head through its explicit ratio/bias program.
     * @tparam Program Normalized safety/positive-bias or legacy signed-bias semantics.
     * @tparam Out Actual Q16 output block.
     * @param input Original disjoint readable Q8 head; @param output Writable Q16 head.
     * @param dim Complete original logical width.
     * @param cosine Original Q15 cosine plane; @param sine Matching original sine plane.
     * @return Common output scale after complete metadata publication.
     */
    template <Q8Conversion Program, Q16Block Out>
    LLAMINAR_NEON_INTEGER_ROPE_EXACT inline Q8Scales q8_conversion(const Q8_1Block *input, Out *output, int dim,
        const std::int16_t *cosine, const std::int16_t *sine)
    {
#if defined(__clang__)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
#endif
        static_assert(Program == Q8Conversion::Normalized || Program == Q8Conversion::LegacySigned);
        const auto geometry = require_integer_head(input, output, dim, cosine, sine, Reuse::Disjoint);
        auto maximum = vdupq_n_f32(0.0f);
        for (int block = 0; block < geometry.input_blocks; ++block)
        {
            const float scale = QuantizedMath<Q8_1Block>::scale(input[block]);
            for (int i = 0; i < 32; i += 4)
                maximum = vmaxq_f32(maximum, vabsq_f32(vmulq_n_f32(simd::neon_typed::load4(input[block].qs + i), scale)));
        }
        const float maximum_value = std::max(vmaxvq_f32(maximum), 1e-20f);
        const float common_scale = [&] {
            if constexpr (Program == Q8Conversion::Normalized) return (maximum_value * 1.415f) / 127.0f;
            else return maximum_value / 127.0f;
        }();
        if (!std::isfinite(common_scale) || common_scale <= 0)
            throw std::invalid_argument("Native Q8/Q16 common scale is not finite and positive");
        for (int first = 0; first < geometry.half;)
        {
            const int second = first + geometry.half;
            const auto &a = input[first / 32], &b = input[second / 32];
            const auto ra = q8_ratio(a, common_scale), rb = q8_ratio(b, common_scale);
            const int count = std::min({32 - first % 32, 32 - second % 32, geometry.half - first});
            for (int i = 0; i < count; i += 4)
            {
                const auto rotated = q8_rotate4<Program == Q8Conversion::Normalized ? Q15Bias::PositiveHalf : Q15Bias::SignedHalf>(
                    a.qs + first % 32 + i, b.qs + second % 32 + i, ra, rb, cosine + first + i, sine + first + i);
                store_integer4(output, first + i, rotated.first); store_integer4(output, second + i, rotated.second);
            }
            first += count;
        }
        const float output_scale = common_scale / 256.0f;
        finish_integer_head(output, geometry.output_blocks, output_scale);
        return {common_scale, output_scale};
    }

    /** @brief Select the canonical normalized Q8-to-Q16 program.
     * @tparam Out Actual retained Q16 encoding.
     * @param input Original disjoint Q8 head; @param output Complete writable Q16 head.
     * @param dim Logical width; @param cosine Original cosine words; @param sine Original sine words.
     * @return Exact stored output scale.
     */
    template <Q16Block Out> inline float q8_to_q16(const Q8_1Block *input, Out *output, int dim,
        const std::int16_t *cosine, const std::int16_t *sine)
    { return q8_conversion<Q8Conversion::Normalized>(input, output, dim, cosine, sine).output; }

    /** @brief Select the legacy signed-bias program, retaining its historical COMMON-scale return value.
     * @param input Original disjoint Q8 head; @param output Complete writable Q16_1 head.
     * @param blocks Complete physical Q8 block count, including single/odd geometries.
     * @param cosine Original cosine words; @param sine Original sine words.
     * @return Original common scale, while physical output d is common/256.
     */
    inline float q8_legacy_head(const Q8_1Block *input, Q16_1Block *output, int blocks,
        const std::int16_t *cosine, const std::int16_t *sine)
    {
        if (blocks <= 0 || blocks > std::numeric_limits<int>::max() / 32)
            throw std::invalid_argument("Native legacy Q8/Q16 head has unrepresentable geometry");
        return q8_conversion<Q8Conversion::LegacySigned>(input, output, blocks * 32, cosine, sine).common;
    }

    /** @brief Convert Q8 through the original fixed-cache FP32 intermediate/ties-away program.
     * @tparam Out Actual Q16 output block.
     * @param input Original disjoint Q8 head; @param output Writable fixed-scale Q16 bank.
     * @param dim Complete logical head width.
     * @param cosine Original Q15 cosine words; @param sine Matching sine words.
     * @param cache_scale Positive finite fixed-cache amplitude.
     */
    template <Q16Block Out>
    LLAMINAR_NEON_INTEGER_ROPE_EXACT inline void q8_to_q16_fixed(const Q8_1Block *input, Out *output, int dim,
        const std::int16_t *cosine, const std::int16_t *sine, float cache_scale)
    {
#if defined(__clang__)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
#endif
        const auto geometry = require_integer_head(input, output, dim, cosine, sine, Reuse::Disjoint);
        const float fixed = cache_scale / 32767.0f, inverse = 32767.0f / cache_scale;
        if (!std::isfinite(fixed) || fixed <= 0 || !std::isfinite(inverse))
            throw std::invalid_argument("Native fixed Q8/Q16 cache scale is not representable");
        for (int first = 0; first < geometry.half; first += 4)
        {
            const int second = first + geometry.half;
            const auto &a = input[first / 32], &b = input[second / 32];
            const auto x = vmulq_n_f32(simd::neon_typed::load4(a.qs + first % 32), QuantizedMath<Q8_1Block>::scale(a));
            const auto y = vmulq_n_f32(simd::neon_typed::load4(b.qs + second % 32), QuantizedMath<Q8_1Block>::scale(b));
            const auto c = vmulq_n_f32(vcvtq_f32_s32(load_integer4(cosine + first)), 1.0f / 32767.0f);
            const auto s = vmulq_n_f32(vcvtq_f32_s32(load_integer4(sine + first)), 1.0f / 32767.0f);
            const auto quantize = [inverse](float32x4_t value) {
                return vcvtaq_s32_f32(vmaxq_f32(vdupq_n_f32(-16383.0f), vminq_f32(vdupq_n_f32(16383.0f), vmulq_n_f32(value, inverse))));
            };
            store_integer4(output, first, quantize(vsubq_f32(vmulq_f32(x, c), vmulq_f32(y, s))));
            store_integer4(output, second, quantize(vaddq_f32(vmulq_f32(x, s), vmulq_f32(y, c))));
        }
        finish_integer_head(output, geometry.output_blocks, fixed);
    }

    /** @brief Execute exact Q16-ratio/Q15 rotation while original source scales remain intact.
     * @tparam Block Actual identical input/output Q16 encoding.
     * @tparam Limit Fixed-cache 16383 or dynamic 32767 saturation limit.
     * @param input Original readable head; @param output Exact-reused or disjoint writable head.
     * @param dim Complete logical width.
     * @param cosine Original Q15 cosine words; @param sine Matching sine words.
     * @param destination_scale Positive finite common output scale.
     */
    template <Q16Block Block, int Limit>
    inline void q16_to_q16(const Block *input, Block *output, int dim,
        const std::int16_t *cosine, const std::int16_t *sine, float destination_scale)
    {
        const auto geometry = require_integer_head(input, output, dim, cosine, sine, Reuse::ExactOrDisjoint);
        if (!std::isfinite(destination_scale) || destination_scale <= 0)
            throw std::invalid_argument("Native Q16 destination scale must be finite and positive");
        for (int first = 0; first < geometry.half;)
        {
            const int second = first + geometry.half;
            const auto &a = input[first / Block::BLOCK_SIZE], &b = input[second / Block::BLOCK_SIZE];
            const auto ra = ratio(QuantizedMath<Block>::scale(a), destination_scale, 65536.0f);
            const auto rb = ratio(QuantizedMath<Block>::scale(b), destination_scale, 65536.0f);
            const int count = std::min({int(Block::BLOCK_SIZE) - first % int(Block::BLOCK_SIZE),
                int(Block::BLOCK_SIZE) - second % int(Block::BLOCK_SIZE), geometry.half - first});
            for (int i = 0; i < count; i += 4)
            {
                const auto x = rescale_q16(load_integer4(a.qs + first % Block::BLOCK_SIZE + i), ra);
                const auto y = rescale_q16(load_integer4(b.qs + second % Block::BLOCK_SIZE + i), rb);
                const auto rotated = integer_rotate4<Q15Bias::PositiveHalf, Limit>(x, y, cosine + first + i, sine + first + i);
                store_integer4(output, first + i, rotated.first); store_integer4(output, second + i, rotated.second);
            }
            first += count;
        }
        // No scale is written before all original payload/scale reads finish,
        // including middle blocks split across the physical head midpoint.
        finish_integer_head(output, geometry.output_blocks, destination_scale);
    }

    /** @brief Select the native fixed-cache Q16 program without a scalar substitution.
     * @tparam Block Original physical encoding.
     * @param input Original readable head; @param output Exact-reused or disjoint writable head.
     * @param dim Complete original logical width.
     * @param cosine Original cosine words; @param sine Matching original sine words.
     * @param cache_scale Positive finite fixed-cache amplitude.
     */
    template <Q16Block Block> inline void q16_fixed(const Block *input, Block *output, int dim,
        const std::int16_t *cosine, const std::int16_t *sine, float cache_scale)
    { q16_to_q16<Block, 16383>(input, output, dim, cosine, sine, cache_scale / 32767.0f); }

    /** @brief Publish dynamic uniform-scale Q16 only after every original scale read completes.
     * @tparam Block Actual physical Q16 encoding.
     * @param input Original readable head; @param output Exact-reused or disjoint writable head.
     * @param dim Complete original logical width.
     * @param cosine Original Q15 cosine words; @param sine Matching sine words.
     * @param output_scale Separate caller-owned terminal scale word, not inside either bank/coefficient plane.
     */
    template <Q16Block Block> inline void q16_dynamic(const Block *input, Block *output, int dim,
        const std::int16_t *cosine, const std::int16_t *sine, float *output_scale)
    {
        const auto geometry = require_integer_head(input, output, dim, cosine, sine, Reuse::ExactOrDisjoint);
        if (!output_scale) throw std::invalid_argument("Native dynamic Q16 requires its output scale word");
        require_integer_disjoint(output_scale, sizeof(float), input, geometry.input_bytes);
        require_integer_disjoint(output_scale, sizeof(float), output, geometry.output_bytes);
        require_integer_disjoint(output_scale, sizeof(float), cosine, std::size_t(geometry.half) * sizeof(std::int16_t));
        require_integer_disjoint(output_scale, sizeof(float), sine, std::size_t(geometry.half) * sizeof(std::int16_t));
        float maximum = 0;
        for (int block = 0; block < geometry.input_blocks; ++block)
            maximum = std::max(maximum, std::abs(QuantizedMath<Block>::scale(input[block])));
        const float unified = std::max(maximum, 1e-20f);
        q16_to_q16<Block, 32767>(input, output, dim, cosine, sine, unified);
        *output_scale = unified;
    }
    /** @brief Rotate one legacy Q8 block pair through its exact signed-bias widened-integer program.
     * @param a Original first input block; @param b Original matching second input block.
     * @param out_a First disjoint Q16 output block; @param out_b Matching disjoint output block.
     * @param cosine Original 32-word Q15 cosine fragment; @param sine Matching sine fragment.
     * @param common_scale Positive finite original common scale, not an inferred output d.
     */
    inline void q8_pair_to_q16(const Q8_1Block &a, const Q8_1Block &b, Q16_1Block &out_a, Q16_1Block &out_b,
        const std::int16_t *cosine, const std::int16_t *sine, float common_scale)
    {
        if (!cosine || !sine || !std::isfinite(common_scale) || common_scale <= 0)
            throw std::invalid_argument("Native legacy Q8/Q16 pair requires complete finite scale/coefficient geometry");
        require_integer_disjoint(&out_a, sizeof(out_a), &out_b, sizeof(out_b));
        for (const auto *output : {&out_a, &out_b})
        {
            require_integer_disjoint(output, sizeof(*output), &a, sizeof(a));
            require_integer_disjoint(output, sizeof(*output), &b, sizeof(b));
            require_integer_disjoint(output, sizeof(*output), cosine, 32 * sizeof(std::int16_t));
            require_integer_disjoint(output, sizeof(*output), sine, 32 * sizeof(std::int16_t));
        }
        const auto ra = q8_ratio(a, common_scale), rb = q8_ratio(b, common_scale);
        for (int i = 0; i < 32; i += 4)
        {
            const auto result = q8_rotate4<Q15Bias::SignedHalf>(a.qs + i, b.qs + i, ra, rb, cosine + i, sine + i);
            vst1_s16(out_a.qs + i, vqmovn_s32(result.first)); vst1_s16(out_b.qs + i, vqmovn_s32(result.second));
        }
        finish_integer_head(&out_a, 1, common_scale / 256.0f);
        finish_integer_head(&out_b, 1, common_scale / 256.0f);
    }

    /** @brief Decode and rotate an original Q8 head directly into its FP32 output coordinates.
     * @param input Complete original disjoint Q8 head.
     * @param output Complete original writable FP32 head.
     * @param dim Positive Q8-block-aligned logical width, with no fixed head-capacity limit.
     * @param cosine Borrowed FP32 cosine plane; @param sine Matching sine plane.
     */
    LLAMINAR_NEON_INTEGER_ROPE_EXACT inline void q8_to_fp32_head(const Q8_1Block *input, float *output, int dim,
        const float *cosine, const float *sine)
    {
#if defined(__clang__)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
#endif
        if (!input || !output || !cosine || !sine || dim <= 0 || dim % 32)
            throw std::invalid_argument("Native hybrid Q8/FP32 RoPE requires complete original geometry");
        const int half = dim / 2;
        const auto output_bytes = std::size_t(dim) * sizeof(float);
        require_integer_disjoint(input, std::size_t(dim / 32) * sizeof(Q8_1Block), output, output_bytes);
        require_integer_disjoint(output, output_bytes, cosine, std::size_t(half) * sizeof(float));
        require_integer_disjoint(output, output_bytes, sine, std::size_t(half) * sizeof(float));
        for (int first = 0; first < half; first += 4)
        {
            const int second = first + half;
            const auto &a = input[first / 32], &b = input[second / 32];
            const auto x = vmulq_n_f32(simd::neon_typed::load4(a.qs + first % 32), QuantizedMath<Q8_1Block>::scale(a));
            const auto y = vmulq_n_f32(simd::neon_typed::load4(b.qs + second % 32), QuantizedMath<Q8_1Block>::scale(b));
            const auto c = vld1q_f32(cosine + first), s = vld1q_f32(sine + first);
            vst1q_f32(output + first, vsubq_f32(vmulq_f32(x, c), vmulq_f32(y, s)));
            vst1q_f32(output + second, vaddq_f32(vmulq_f32(x, s), vmulq_f32(y, c)));
        }
    }
} // namespace llaminar2::primitives::neon_rope
#undef LLAMINAR_NEON_INTEGER_ROPE_EXACT
