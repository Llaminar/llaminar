/**
 * @file NativeVnniQ6Scale.h
 * @brief Lossless four-byte Q6_K scale metadata shared by CPU and GPU packing.
 *
 * Each logical block contains two sixteen-value groups. Its primary metadata
 * word retains the source superblock's FP16 base verbatim; its secondary word
 * contains the two original signed eight-bit coefficients. The secondary word
 * is opaque integer metadata and must never be interpreted as an FP16 value.
 * Widening the base and multiplying each coefficient in FP32 recovers the
 * original factors exactly, without narrowing derived scales or adding storage.
 * Consumers retain their existing two integer dots and ordered FP32 products.
 */
#pragma once

#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_Q6_SCALE_HD __host__ __device__
#else
#define LLAMINAR_Q6_SCALE_HD
#endif

namespace llaminar2::native_vnni_q6_scale
{
    /**
     * @brief Pack both source coefficients without arithmetic conversion.
     * @param low_coefficient Signed scale for values zero through fifteen.
     * @param high_coefficient Signed scale for values sixteen through thirty-one.
     * @return Opaque secondary metadata word, low coefficient in bits zero to seven.
     */
    LLAMINAR_Q6_SCALE_HD constexpr uint16_t pack(
        int8_t low_coefficient, int8_t high_coefficient) noexcept
    {
        return static_cast<uint16_t>(static_cast<uint8_t>(low_coefficient)) |
               (static_cast<uint16_t>(static_cast<uint8_t>(high_coefficient)) << 8);
    }

    /**
     * @brief Sign-extend the first coefficient without an out-of-range signed cast.
     * @param coefficients Opaque secondary metadata word.
     * @return Original signed coefficient for the first sixteen values.
     */
    LLAMINAR_Q6_SCALE_HD constexpr int low(uint16_t coefficients) noexcept
    {
        return static_cast<int>(coefficients & 0xffu) -
               ((coefficients & 0x80u) != 0 ? 256 : 0);
    }

    /**
     * @brief Sign-extend the second coefficient without interpreting FP16 bits.
     * @param coefficients Opaque secondary metadata word.
     * @return Original signed coefficient for the last sixteen values.
     */
    LLAMINAR_Q6_SCALE_HD constexpr int high(uint16_t coefficients) noexcept
    {
        return static_cast<int>(coefficients >> 8) -
               ((coefficients & 0x8000u) != 0 ? 256 : 0);
    }

    static_assert(low(pack(-128, 127)) == -128);
    static_assert(high(pack(-128, 127)) == 127);
    static_assert(low(pack(127, -128)) == 127);
    static_assert(high(pack(127, -128)) == -128);
}

#undef LLAMINAR_Q6_SCALE_HD
