/**
 * @file CPUNativeVNNIFP16.h
 * @brief Exact hardware FP16 scale conversion for CPU NativeVNNI kernels.
 *
 * NativeVNNI consumes one Q8_1 activation scale for every 32-element K block.
 * Calling the portable scalar FP16 decoder in that inner loop expands a value
 * that is exactly representable in FP32 through a branch-heavy normalization
 * sequence. Llaminar's x86 build envelopes require F16C; AArch64 ASIMD
 * provides binary16-to-binary32 conversion. Finite scales use the native
 * conversion instruction on either architecture.
 * Non-finite encodings retain the portable conversion so signaling/quiet NaN
 * payload bits remain identical to the repository-wide scalar contract.
 */

#pragma once

#include <cstdint>
#if defined(__F16C__)
#include <immintrin.h>
#elif defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#include "tensors/FP16Utils.h"

namespace llaminar2::cpu::native_vnni
{
    /**
     * @brief Convert one packed activation scale to FP32 without approximation.
     *
     * Every finite IEEE-754 binary16 number is exactly representable as FP32,
     * so F16C/ASIMD cannot introduce a rounding difference. NaN and infinity use
     * the scalar bit-expansion routine because hardware is permitted to quiet
     * signaling NaNs.  Keeping that rare diagnostic case on the portable path
     * makes this helper byte-identical over all 65,536 input encodings.
     *
     * @param value Raw IEEE-754 binary16 bits from a Q8_1 block.
     * @return The exact FP32 expansion of `value`.
     */
    inline float nativeVNNIFP16ScaleToFP32(uint16_t value)
    {
#if defined(__F16C__)
        if ((value & 0x7c00u) != 0x7c00u)
            return _cvtsh_ss(value);
#elif defined(__aarch64__) && defined(__ARM_NEON)
        if ((value & 0x7c00u) != 0x7c00u)
        {
            const float16x4_t half = vreinterpret_f16_u16(vdup_n_u16(value));
            return vgetq_lane_f32(vcvt_f32_f16(half), 0);
        }
#endif
        return llaminar2::fp16_to_fp32(value);
    }
} // namespace llaminar2::cpu::native_vnni
