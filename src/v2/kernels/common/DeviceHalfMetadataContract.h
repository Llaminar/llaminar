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
#define LLAMINAR_HALF_CONTRACT_INLINE __host__ __device__ __forceinline__
#else
#define LLAMINAR_HALF_CONTRACT_INLINE inline
#endif

namespace llaminar2
{
    /**
     * @brief Convert a computed float metadata value to canonical binary16 bits.
     *
     * This helper canonicalizes both binary16 zero encodings to positive zero.
     * Every nonzero value, including infinities and NaNs, retains the exact
     * round-to-nearest encoding produced by the device conversion intrinsic.
     * Raw source metadata is intentionally not passed through this function:
     * only values derived while preparing a GPU-aligned expert use this publication
     * contract.
     *
     * @param value Computed scale, offset, or correction value to persist.
     * @return Canonical binary16 bit representation of @p value.
     */
    LLAMINAR_HALF_CONTRACT_INLINE uint16_t canonicalPreparedHalfBits(
        float value) noexcept
    {
        union Word
        {
            float value;
            std::uint32_t bits;
        } source{};
        source.value = value;

        const std::uint16_t sign = static_cast<std::uint16_t>(
            (source.bits >> 16u) & 0x8000u);
        const std::uint32_t source_exponent =
            (source.bits >> 23u) & 0xffu;
        std::uint32_t mantissa = source.bits & 0x007fffffu;

        if (source_exponent == 0xffu)
        {
            if (mantissa == 0u)
                return static_cast<std::uint16_t>(sign | 0x7c00u);
            // Retain the high NaN payload and force a quiet nonzero payload.
            return static_cast<std::uint16_t>(
                sign | 0x7c00u | (mantissa >> 13u) | 0x0200u);
        }

        int half_exponent =
            static_cast<int>(source_exponent) - 127 + 15;
        if (half_exponent >= 31)
            return static_cast<std::uint16_t>(sign | 0x7c00u);

        if (half_exponent <= 0)
        {
            if (half_exponent < -10)
                return 0u;

            // Install the implicit binary32 leading one, then round the
            // shifted subnormal significand to nearest with ties to even.
            mantissa |= 0x00800000u;
            const int shift = 14 - half_exponent;
            std::uint32_t rounded = mantissa >> shift;
            const std::uint32_t remainder_mask =
                (std::uint32_t{1} << shift) - 1u;
            const std::uint32_t remainder = mantissa & remainder_mask;
            const std::uint32_t halfway =
                std::uint32_t{1} << (shift - 1);
            if (remainder > halfway ||
                (remainder == halfway && (rounded & 1u) != 0u))
            {
                ++rounded;
            }
            const std::uint16_t bits = static_cast<std::uint16_t>(
                sign | static_cast<std::uint16_t>(rounded));
            return (bits & 0x7fffu) == 0u ? 0u : bits;
        }

        std::uint32_t rounded_mantissa = mantissa >> 13u;
        const std::uint32_t remainder = mantissa & 0x1fffu;
        if (remainder > 0x1000u ||
            (remainder == 0x1000u &&
             (rounded_mantissa & 1u) != 0u))
        {
            ++rounded_mantissa;
            if (rounded_mantissa == 0x0400u)
            {
                rounded_mantissa = 0u;
                ++half_exponent;
                if (half_exponent >= 31)
                    return static_cast<std::uint16_t>(sign | 0x7c00u);
            }
        }
        return static_cast<std::uint16_t>(
            sign |
            (static_cast<std::uint16_t>(half_exponent) << 10u) |
            static_cast<std::uint16_t>(rounded_mantissa));
    }

    /**
     * @brief Round one derived value through the canonical binary16 metadata ABI.
     *
     * CPU NativeVNNI embeds Q8 activation scales in `Q8_1Block::d`, while GPU
     * kernels carry the same scale in a binary32 sidecar for coalesced loads.
     * Returning the widened canonical binary16 word makes those two physical
     * representations describe exactly the same arithmetic value.
     *
     * @param value Finite derived metadata value to round.
     * @return Canonical binary16 value widened exactly to binary32.
     */
    LLAMINAR_HALF_CONTRACT_INLINE float canonicalPreparedHalfValue(
        float value) noexcept
    {
        const std::uint16_t half_bits = canonicalPreparedHalfBits(value);
        const std::uint32_t sign =
            static_cast<std::uint32_t>(half_bits & 0x8000u) << 16u;
        int exponent = static_cast<int>((half_bits >> 10u) & 0x1fu);
        std::uint32_t mantissa = half_bits & 0x03ffu;
        std::uint32_t widened_bits = 0u;
        if (exponent == 0)
        {
            if (mantissa == 0u)
            {
                widened_bits = sign;
            }
            else
            {
                exponent = 1;
                while ((mantissa & 0x0400u) == 0u)
                {
                    mantissa <<= 1u;
                    --exponent;
                }
                mantissa &= 0x03ffu;
                widened_bits = sign |
                    (static_cast<std::uint32_t>(exponent + 112) << 23u) |
                    (mantissa << 13u);
            }
        }
        else if (exponent == 31)
        {
            widened_bits = sign | 0x7f800000u | (mantissa << 13u);
        }
        else
        {
            widened_bits = sign |
                (static_cast<std::uint32_t>(exponent + 112) << 23u) |
                (mantissa << 13u);
        }

        union Word
        {
            std::uint32_t bits;
            float value;
        } result{};
        result.bits = widened_bits;
        return result.value;
    }
}

#undef LLAMINAR_HALF_CONTRACT_INLINE
