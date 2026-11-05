/**
 * @file NEONQuantizedRoPE.h
 * @brief Native AArch64 Q15 rotation and per-block quantization for retained KV heads.
 *
 * The cycle scheduler consumes original halves before replacing their payload
 * or scale. This provider preserves the production quantized RoPE FMA order,
 * ties-even publication, original Q8 half/full Q16 scales and exact signed sums.
 * Private fragments are bounded by the physical block, never the full head or
 * cache; no allocation, repacking, scalar whole-head path or state publication
 * is performed here.
 */
#pragma once
#if !defined(__aarch64__) || !defined(__ARM_NEON)
#error "Native quantized rotary math requires AArch64 ASIMD"
#endif
#include "CPUQuantizedRoPECycle.h"
#include "tensors/NEONTypedTensorOps.h"
#include <arm_neon.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <type_traits>

#if defined(__GNUC__) && !defined(__clang__)
#define LLAMINAR_NEON_QUANT_ROPE_EXACT __attribute__((optimize("fp-contract=off", "no-associative-math")))
#else
#define LLAMINAR_NEON_QUANT_ROPE_EXACT
#endif

namespace llaminar2::primitives::neon_rope
{
    /** @brief Architecture-native arithmetic for each retained quantized block family.
     * @tparam Block Original Q8 or Q16 block, never a decoded/cache mirror.
     */
    template <QuantizedRoPEBlock Block>
    struct QuantizedMath final
    {
        /** @brief Read the original scale through the encoding's portable metadata ABI.
         * @param block Original immutable block, before its last input consumption.
         * @return Finite original half/full scale, including a legitimate signed zero.
         * @throws std::invalid_argument On nonfinite retained scale metadata.
         */
        [[gnu::always_inline]] static inline float scale(const Block &block)
        {
            const float value = [&] {
                if constexpr (std::is_same_v<Block, Q8_1Block>) return fp16_to_fp32(block.d);
                else return block.d;
            }();
            if (!std::isfinite(value)) throw std::invalid_argument("Native quantized RoPE encountered a nonfinite scale");
            return value;
        }

        /** @brief Rotate complete four-lane fragments of two original half-blocks.
         * @param a Original first-half source block.
         * @param a_offset Original first-half word offset inside a.
         * @param b Original matching second-half source block.
         * @param b_offset Original second-half word offset inside b.
         * @param cosine Original Q15 cosine fragment.
         * @param sine Matching sine fragment.
         * @param out_a Private original-order first-half results.
         * @param out_b Matching second-half results.
         * @param count Complete native fragment width, a multiple of four.
         */
        LLAMINAR_NEON_QUANT_ROPE_EXACT static inline void rotateHalf(
            const Block &a, int a_offset, const Block &b, int b_offset,
            const std::int16_t *cosine, const std::int16_t *sine,
            float *out_a, float *out_b, int count)
        {
#if defined(__clang__)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
#endif
            const float a_scale = scale(a), b_scale = scale(b);
            for (int index = 0; index < count; index += 4)
            {
                const auto x = vmulq_n_f32(simd::neon_typed::load4(a.qs + a_offset + index), a_scale);
                const auto y = vmulq_n_f32(simd::neon_typed::load4(b.qs + b_offset + index), b_scale);
                const auto c = vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vld1_s16(cosine + index))), 1.0f / 32767.0f);
                const auto s = vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vld1_s16(sine + index))), 1.0f / 32767.0f);
                // Match the native quantized program: y*s and y*c round first;
                // the original x products are then the explicit fused edges.
                vst1q_f32(out_a + index, vfmaq_f32(vnegq_f32(vmulq_f32(y, s)), x, c));
                vst1q_f32(out_b + index, vfmaq_f32(vmulq_f32(y, c), x, s));
            }
        }

        /** @brief Publish a fully consumed physical block with its original native scale/RNE program.
         * @param values Complete bounded original-order rotated block.
         * @param output Sole physical block whose two original halves are already consumed.
         * @throws std::invalid_argument On nonfinite or unrepresentable resulting scale.
         */
        LLAMINAR_NEON_QUANT_ROPE_EXACT static inline void requantize(const float *values, Block &output)
        {
#if defined(__clang__)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
#endif
            constexpr int width = static_cast<int>(Block::BLOCK_SIZE);
            constexpr int limit = std::is_same_v<Block, Q8_1Block> ? 127 : 32767;
            auto maximum = vdupq_n_f32(0.0f);
            for (int index = 0; index < width; index += 4)
                maximum = vmaxq_f32(maximum, vabsq_f32(vld1q_f32(values + index)));
            const float max_value = vmaxvq_f32(maximum);
            if (!std::isfinite(max_value)) throw std::invalid_argument("Native quantized RoPE result is not finite");
            const float output_scale = std::max(max_value / static_cast<float>(limit), 1e-20f);
            const float inverse = 1.0f / output_scale;
            const auto half_scale = [&] {
                if constexpr (std::is_same_v<Block, Q8_1Block>) return fp32_to_fp16(output_scale);
                else return std::uint16_t{0};
            }();
            if constexpr (std::is_same_v<Block, Q8_1Block>)
                if ((half_scale & 0x7c00u) == 0x7c00u)
                    throw std::invalid_argument("Native Q8 RoPE output scale exceeds finite half storage");
            auto sum = vdupq_n_s32(0);
            for (int index = 0; index < width; index += 4)
            {
                const auto scaled = vmulq_n_f32(vld1q_f32(values + index), inverse);
                const auto bounded = vmaxq_f32(vdupq_n_f32(-static_cast<float>(limit)),
                    vminq_f32(vdupq_n_f32(static_cast<float>(limit)), scaled));
                const auto q = vcvtnq_s32_f32(bounded);
                sum = vaddq_s32(sum, q);
                if constexpr (std::is_same_v<Block, Q8_1Block>)
                {
                    const auto bytes = vqmovn_s16(vcombine_s16(vqmovn_s32(q), vdup_n_s16(0)));
                    const auto word = vget_lane_s32(vreinterpret_s32_s8(bytes), 0);
                    std::memcpy(output.qs + index, &word, sizeof(word));
                }
                else vst1_s16(output.qs + index, vqmovn_s32(q));
            }
            if constexpr (std::is_same_v<Block, Q8_1Block>) output.d = half_scale;
            else output.d = output_scale;
            output.sum_qs = static_cast<decltype(output.sum_qs)>(vaddvq_s32(sum));
        }
    };

    /** @brief Select the native bounded-cycle program for an original retained head.
     * @tparam Block Actual quantized storage block.
     * @param head Original in-place physical block bank.
     * @param blocks Complete physical block count, including single and odd counts.
     * @param cosine Borrowed original Q15 coefficients.
     * @param sine Matching original sine coefficients.
     */
    template <QuantizedRoPEBlock Block>
    inline void rotate_quantized(Block *head, int blocks, const std::int16_t *cosine, const std::int16_t *sine)
    { rotate_quantized_head_in_place<Block, QuantizedMath<Block>>(head, blocks, cosine, sine); }
} // namespace llaminar2::primitives::neon_rope
#undef LLAMINAR_NEON_QUANT_ROPE_EXACT
