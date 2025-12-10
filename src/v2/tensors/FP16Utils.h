#pragma once

#include <cstdint>
#include <cstring>

/**
 * @file FP16Utils.h
 * @brief IEEE 754 FP16 conversion utilities for v2 architecture
 * @author David Sanftenberg
 *
 * Provides fast FP16 ↔ FP32 conversion without external dependencies.
 * Uses bit manipulation for portable conversion.
 */

namespace llaminar2
{

    /**
     * @brief Convert FP16 (uint16_t) to FP32
     * @param h FP16 value as uint16_t (IEEE 754 binary16)
     * @return FP32 value
     *
     * Handles: normals, denormals, infinities, NaNs
     */
    inline float fp16_to_fp32(uint16_t h)
    {
        uint32_t sign = (h & 0x8000u) << 16;
        int32_t exp = (h & 0x7C00u) >> 10;
        uint32_t mantissa = (h & 0x03FFu);

        uint32_t f;
        if (exp == 0)
        {
            // Zero or denormal
            if (mantissa == 0)
            {
                f = sign; // Signed zero
            }
            else
            {
                // Denormal: normalize it
                exp = 1;
                while ((mantissa & 0x400) == 0)
                {
                    mantissa <<= 1;
                    exp--;
                }
                mantissa &= 0x3FF;
                f = sign | ((static_cast<uint32_t>(exp + 112)) << 23) | (mantissa << 13);
            }
        }
        else if (exp == 0x1F)
        {
            // Infinity or NaN
            f = sign | 0x7F800000u | (mantissa << 13);
        }
        else
        {
            // Normal number
            f = sign | ((static_cast<uint32_t>(exp + 112)) << 23) | (mantissa << 13);
        }

        float result;
        std::memcpy(&result, &f, sizeof(float));
        return result;
    }

    /**
     * @brief Convert FP32 to FP16 (uint16_t)
     * @param f FP32 value
     * @return FP16 value as uint16_t (IEEE 754 binary16)
     *
     * Uses round-to-nearest-even. Clamps overflow to ±infinity.
     */
    inline uint16_t fp32_to_fp16(float f)
    {
        uint32_t bits;
        std::memcpy(&bits, &f, sizeof(float));

        uint32_t sign = (bits & 0x80000000u) >> 16;
        int32_t exp = ((bits & 0x7F800000u) >> 23) - 127 + 15;
        uint32_t mantissa = (bits & 0x007FFFFFu);

        if (exp <= 0)
        {
            // Zero or denormal
            if (exp < -10)
            {
                return static_cast<uint16_t>(sign); // Underflow to zero
            }
            // Denormal: shift mantissa right and add implicit 1
            mantissa = (mantissa | 0x00800000u) >> (1 - exp);
            return static_cast<uint16_t>(sign | (mantissa >> 13));
        }
        else if (exp >= 0x1F)
        {
            // Overflow to infinity or preserve NaN
            if (exp == 0xFF - 127 + 15 && mantissa != 0)
            {
                // NaN
                return static_cast<uint16_t>(sign | 0x7C00u | (mantissa >> 13));
            }
            return static_cast<uint16_t>(sign | 0x7C00u); // Infinity
        }
        else
        {
            // Normal number: round to nearest even
            uint32_t rounding = (mantissa >> 13) & 1;
            mantissa += 0x0FFFu + rounding;
            if (mantissa >= 0x00800000u)
            {
                // Rounding caused overflow, increment exponent
                exp++;
                mantissa = 0;
            }
            return static_cast<uint16_t>(sign | (exp << 10) | (mantissa >> 13));
        }
    }

} // namespace llaminar2
