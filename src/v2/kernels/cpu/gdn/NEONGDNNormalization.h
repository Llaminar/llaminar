/**
 * @file NEONGDNNormalization.h
 * @brief NEON squared-norm and scaling programs for GDN Q/K L2 normalization.
 *
 * The squared norm reproduces the AVX2 eight-lane program: two four-lane FMA
 * chains hold lanes 0-3 and 4-7 of each eight-element group (one AVX2 ymm),
 * folded as ((a0+a4)+(a1+a5)) + ((a2+a6)+(a3+a7)); remaining squares are a
 * separately rounded multiply then add in increasing column order. Scaling is
 * one separately rounded multiply per element. Header-only so the arithmetic
 * can be tested without the engine core.
 */
#pragma once
#include "kernels/common/DeviceFP32NumericalContract.h"
#include <arm_neon.h>

namespace llaminar2::neon_gdn
{
    /** @return Sum of squares of @p n floats with the AVX2 eight-lane program. */
    inline float norm_sq_avx2_program(const float *data, int n)
    {
        float32x4_t low = vdupq_n_f32(0.0f), high = vdupq_n_f32(0.0f);
        int i = 0;
        for (; i + 8 <= n; i += 8)
        {
            const float32x4_t a = vld1q_f32(data + i), b = vld1q_f32(data + i + 4);
            low = vfmaq_f32(low, a, a);
            high = vfmaq_f32(high, b, b);
        }
        const float32x4_t paired = vaddq_f32(low, high);
        float result = device_fp32_contract::add(
            device_fp32_contract::add(vgetq_lane_f32(paired, 0), vgetq_lane_f32(paired, 1)),
            device_fp32_contract::add(vgetq_lane_f32(paired, 2), vgetq_lane_f32(paired, 3)));
        for (; i < n; ++i)
            result = device_fp32_contract::add(result, device_fp32_contract::multiply(data[i], data[i]));
        return result;
    }

    /** @brief dst[i] = src[i] * factor, separately rounded; src may equal dst. */
    inline void scale_rows(const float *src, float *dst, int n, float factor)
    {
        const float32x4_t f = vdupq_n_f32(factor);
        int i = 0;
        for (; i + 4 <= n; i += 4)
            vst1q_f32(dst + i, vmulq_f32(vld1q_f32(src + i), f));
        for (; i < n; ++i)
            dst[i] = device_fp32_contract::multiply(src[i], factor);
    }
}
