/**
 * @file CPUNativeVNNIFP16.h
 * @brief Exact hardware FP16 scale conversion for CPU NativeVNNI kernels.
 *
 * NativeVNNI consumes one Q8_1 activation scale for every 32-element K block.
 * Calling the portable scalar FP16 decoder in that inner loop expands a value
 * that is exactly representable in FP32 through a branch-heavy normalization
 * sequence.  Llaminar's AVX2 and AVX-512 CPU build envelopes both require
 * F16C, so finite scale values can use the native conversion instruction.
 * Non-finite encodings retain the portable conversion so signaling/quiet NaN
 * payload bits remain identical to the repository-wide scalar contract.
 */

#pragma once

#include <cstdint>
#include <immintrin.h>

#include "tensors/FP16Utils.h"

namespace llaminar2::cpu::native_vnni
{
    /**
     * @brief Convert one packed activation scale to FP32 without approximation.
     *
     * Every finite IEEE-754 binary16 number is exactly representable as FP32,
     * so F16C cannot introduce a rounding difference.  NaN and infinity use
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
#endif
        return llaminar2::fp16_to_fp32(value);
    }
} // namespace llaminar2::cpu::native_vnni
