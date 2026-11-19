/**
 * @file NEONInt8BlockRequantize.h
 * @brief NEON row max/requantize for 32-code INT8 blocks (Q8_0, Q8_1).
 *
 * Both programs are byte-identical to the portable scalar leaves:
 *  - max: |code| is computed in 8-bit with wraparound and read unsigned, so
 *    -128 contributes 128 exactly as the scalar `int` absolute value does;
 *    the block value is the separately rounded `scale * float(max)`.
 *  - requantize: `rescale = scale * inverse` once per block, then per code
 *    `value = float(code) * rescale` (multiply, never fused), `value +/- 0.5f`
 *    as a rounded FP32 add, truncation toward zero, and clamp to [-127, 127].
 * That is the scalar program's exact sequence of FP32 roundings, not a
 * round-to-nearest shortcut, so NEON and Scalar agree on every input.
 */
#pragma once
#include "FP16Utils.h"
#include <arm_neon.h>
#include <cstddef>
#include <cstdint>

namespace llaminar2::neon_int8_block
{
    /** @return Largest |dequantized code| over the row, matching the scalar leaf. */
    template<class Block>
    inline float findMaxAbs(const Block *blocks, std::size_t count)
    {
        static_assert(sizeof(Block{}.qs) == 32, "NEON INT8 block program requires 32 codes");
        float maximum = 0.0f;
        for (std::size_t b = 0; b < count; ++b)
        {
            const int8x16_t low = vld1q_s8(blocks[b].qs);
            const int8x16_t high = vld1q_s8(blocks[b].qs + 16);
            // vabsq_s8 wraps -128 to 0x80; reading it unsigned yields 128.
            const uint8x16_t magnitude = vmaxq_u8(vreinterpretq_u8_s8(vabsq_s8(low)),
                                                  vreinterpretq_u8_s8(vabsq_s8(high)));
            const float value = fp16_to_fp32(blocks[b].d) * static_cast<float>(vmaxvq_u8(magnitude));
            if (value > maximum)
                maximum = value;
        }
        return maximum;
    }

    /** @brief Requantize eight codes with the scalar rounding sequence. */
    inline int32x4_t roundHalfAwayAsScalar(float32x4_t value)
    {
        const float32x4_t half = vbslq_f32(vcgeq_f32(value, vdupq_n_f32(0.0f)),
                                           vdupq_n_f32(0.5f), vdupq_n_f32(-0.5f));
        const int32x4_t truncated = vcvtq_s32_f32(vaddq_f32(value, half));
        return vmaxq_s32(vminq_s32(truncated, vdupq_n_s32(127)), vdupq_n_s32(-127));
    }

    /** @brief Write 32 requantized codes per block, byte-identical to the scalar leaf. */
    template<class Block>
    inline void requantize(const Block *blocks, std::size_t count, float inverse, std::int8_t *output)
    {
        static_assert(sizeof(Block{}.qs) == 32, "NEON INT8 block program requires 32 codes");
        for (std::size_t b = 0; b < count; ++b)
        {
            const float32x4_t rescale = vdupq_n_f32(fp16_to_fp32(blocks[b].d) * inverse);
            std::int8_t *destination = output + b * 32;
            for (int half = 0; half < 2; ++half)
            {
                const int8x16_t codes = vld1q_s8(blocks[b].qs + half * 16);
                const int16x8_t wide_low = vmovl_s8(vget_low_s8(codes));
                const int16x8_t wide_high = vmovl_s8(vget_high_s8(codes));
                const int32x4_t lanes[4] = {vmovl_s16(vget_low_s16(wide_low)), vmovl_s16(vget_high_s16(wide_low)),
                                            vmovl_s16(vget_low_s16(wide_high)), vmovl_s16(vget_high_s16(wide_high))};
                int32x4_t rounded[4];
                for (int q = 0; q < 4; ++q)
                    rounded[q] = roundHalfAwayAsScalar(vmulq_f32(vcvtq_f32_s32(lanes[q]), rescale));
                const int16x8_t narrow_low = vcombine_s16(vmovn_s32(rounded[0]), vmovn_s32(rounded[1]));
                const int16x8_t narrow_high = vcombine_s16(vmovn_s32(rounded[2]), vmovn_s32(rounded[3]));
                vst1q_s8(destination + half * 16, vcombine_s8(vmovn_s16(narrow_low), vmovn_s16(narrow_high)));
            }
        }
    }
}
