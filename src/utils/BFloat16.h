// BFloat16.h - minimal bfloat16 helper (stores upper 16 bits of IEEE754 float)
#pragma once
#include <cstdint>
#include <cstring>

namespace llaminar
{

    struct bfloat16
    {
        uint16_t data;
        bfloat16() : data(0) {}
        explicit bfloat16(float f) { *this = from_float(f); }
        static bfloat16 from_float(float f)
        {
            bfloat16 bf;
            uint32_t u;
            std::memcpy(&u, &f, 4);
            // Round to nearest even: add 0x7FFF + lsb of top to lower 16 bits before shift
            uint32_t lsb = (u >> 16) & 1u; // LSB of truncated part for tie-to-even
            uint32_t rounding_bias = 0x7FFFu + lsb;
            u += rounding_bias;
            bf.data = (uint16_t)(u >> 16);
            return bf;
        }
        operator float() const
        {
            uint32_t u = (uint32_t)data << 16;
            float f;
            std::memcpy(&f, &u, 4);
            return f;
        }
    };

} // namespace llaminar
