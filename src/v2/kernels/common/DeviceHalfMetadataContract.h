/**
 * @file DeviceHalfMetadataContract.h
 * @brief Canonical binary16 publication for cross-backend prepared metadata.
 *
 * Movable NativeVNNI experts persist derived scales, offsets, and correction
 * terms as binary16 bit patterns. IEEE-754 permits both positive and negative
 * zero, but those encodings are arithmetically equivalent and CUDA and HIP may
 * choose different signs after an otherwise identical derived expression.
 * Expert movement compares and transfers these metadata blobs byte-for-byte,
 * so publication must choose one representation independently of residency.
 */

#pragma once

#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)

namespace llaminar2
{
    /**
     * @brief Convert a computed float metadata value to canonical binary16 bits.
     *
     * This helper canonicalizes both binary16 zero encodings to positive zero.
     * Every nonzero value, including infinities and NaNs, retains the exact
     * round-to-nearest encoding produced by the device conversion intrinsic.
     * Raw source metadata is intentionally not passed through this function:
     * only values derived while preparing a movable expert use this publication
     * contract.
     *
     * @param value Computed scale, offset, or correction value to persist.
     * @return Canonical binary16 bit representation of @p value.
     */
    __device__ __forceinline__ uint16_t canonicalPreparedHalfBits(
        float value) noexcept
    {
        const uint16_t bits = __half_as_ushort(__float2half_rn(value));
        return (bits & 0x7fffu) == 0u ? 0u : bits;
    }
}

#endif
