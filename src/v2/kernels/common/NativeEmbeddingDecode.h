/**
 * @file NativeEmbeddingDecode.h
 * @brief Direct source-block embedding decode without weight requantization.
 *
 * Each invocation reads one requested source value. Codes, grids, scales and
 * minima keep their GGUF representation; derived scales remain binary32 and
 * never pass through a second quantizer or binary16 metadata publication.
 * Format selection is a launch-time specialization shared by CUDA and HIP.
 */
#pragma once
#include "kernels/common/DeviceFP32NumericalContract.h"
#include "kernels/common/NativeEmbeddingIQTables.h"
#include "tensors/BlockStructures.h"
#include <cstdint>
#include <cstddef>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_NATIVE_EMBED_INLINE __device__ __forceinline__
#else
#define LLAMINAR_NATIVE_EMBED_INLINE inline
#endif

namespace llaminar2::native_embedding
{
    /** @brief Map a source format to its exact packed native block. */
    template<TensorType> struct SourceBlock;
#define LLAMINAR_EMBED_BLOCK(NAME) \
    template<> struct SourceBlock<TensorType::NAME> { using type = NAME##Block; };
    LLAMINAR_EMBED_BLOCK(Q4_0)
    LLAMINAR_EMBED_BLOCK(Q4_1)
    LLAMINAR_EMBED_BLOCK(Q5_0)
    LLAMINAR_EMBED_BLOCK(Q5_1)
    LLAMINAR_EMBED_BLOCK(Q8_0)
    LLAMINAR_EMBED_BLOCK(Q8_1)
    LLAMINAR_EMBED_BLOCK(Q2_K)
    LLAMINAR_EMBED_BLOCK(Q3_K)
    LLAMINAR_EMBED_BLOCK(Q4_K)
    LLAMINAR_EMBED_BLOCK(Q5_K)
    LLAMINAR_EMBED_BLOCK(Q6_K)
    LLAMINAR_EMBED_BLOCK(Q8_K)
    LLAMINAR_EMBED_BLOCK(IQ1_S)
    LLAMINAR_EMBED_BLOCK(IQ1_M)
    LLAMINAR_EMBED_BLOCK(IQ2_XXS)
    LLAMINAR_EMBED_BLOCK(IQ2_XS)
    LLAMINAR_EMBED_BLOCK(IQ2_S)
    LLAMINAR_EMBED_BLOCK(IQ3_XXS)
    LLAMINAR_EMBED_BLOCK(IQ3_S)
    LLAMINAR_EMBED_BLOCK(IQ4_NL)
    LLAMINAR_EMBED_BLOCK(IQ4_XS)
#undef LLAMINAR_EMBED_BLOCK

    /**
     * @brief Read unaligned little-endian source metadata without wider loads.
     * @param pointer Two source bytes, possibly inside a packed IQ block.
     * @return Exact unsigned metadata word.
     */
    LLAMINAR_NATIVE_EMBED_INLINE std::uint16_t word16(const void *pointer)
    {
        const auto *p = static_cast<const std::uint8_t *>(pointer);
        return std::uint16_t(p[0]) | (std::uint16_t(p[1]) << 8);
    }

    /**
     * @brief Read an unaligned little-endian four-byte source field.
     * @param pointer Four source bytes.
     * @return Exact unsigned metadata word.
     */
    LLAMINAR_NATIVE_EMBED_INLINE std::uint32_t word32(const void *pointer)
    {
        const auto *p = static_cast<const std::uint8_t *>(pointer);
        return std::uint32_t(p[0]) | (std::uint32_t(p[1]) << 8) |
               (std::uint32_t(p[2]) << 16) | (std::uint32_t(p[3]) << 24);
    }

    /**
     * @brief Widen an original binary16 word, preserving subnormals and signs.
     * @param value Original source scale or minimum bits.
     * @return Exact binary32 value; no canonicalization or second rounding.
     */
    LLAMINAR_NATIVE_EMBED_INLINE float half(std::uint16_t value)
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        return __half2float(*reinterpret_cast<const __half *>(&value));
#else
        const std::uint32_t sign = std::uint32_t(value & 0x8000u) << 16;
        int exponent = (value >> 10) & 31;
        std::uint32_t mantissa = value & 1023;
        if (exponent == 0)
        {
            if (mantissa == 0)
                return device_fp32_contract::floatFromBits(sign);
            exponent = 1;
            while ((mantissa & 1024u) == 0)
            {
                mantissa <<= 1;
                --exponent;
            }
            mantissa &= 1023u;
        }
        else if (exponent == 31)
            return device_fp32_contract::floatFromBits(
                sign | 0x7f800000u | (mantissa << 13));
        return device_fp32_contract::floatFromBits(
            sign | (std::uint32_t(exponent + 112) << 23) | (mantissa << 13));
#endif
    }

    /**
     * @brief Expand the seven-bit even-parity IQ sign code without a table.
     * @param index Seven original sign bits.
     * @return Eight-bit sign mask, identical to ksigns_iq2xs.
     */
    LLAMINAR_NATIVE_EMBED_INLINE unsigned signs(unsigned index)
    {
        unsigned parity = index;
        parity ^= parity >> 4;
        parity ^= parity >> 2;
        parity ^= parity >> 1;
        return index | ((parity & 1u) << 7);
    }

    /**
     * @brief Select the fixed IQ4 nonlinear integer value without new storage.
     * @param index Original four-bit code.
     * @return Canonical signed integer grid value.
     */
    LLAMINAR_NATIVE_EMBED_INLINE float iq4(unsigned index)
    {
        // Packed constants keep the table in instructions/registers on both
        // backends rather than adding another device codebook allocation.
        const std::uint64_t lo = 0xf6eadDCFbfad9881ULL;
        const std::uint64_t hi = 0x7159453526190d01ULL;
        const unsigned byte = unsigned(((index < 8 ? lo : hi) >>
                                        (8 * (index & 7))) & 255u);
        return float(int(byte) - (byte >= 128 ? 256 : 0));
    }

    /**
     * @brief Decode one native source element in a contiguous vocabulary row.
     * @tparam Format Original quantized tensor type, fixed per kernel launch.
     * @param row First native block of the selected source row.
     * @param column Logical element within that row.
     * @param tables Setup-bound immutable IQ grids; unused formats erase them.
     * @return Source FP32 dequantized value with explicit arithmetic edges.
     */
    template<TensorType Format>
    LLAMINAR_NATIVE_EMBED_INLINE float decode(
        const void *row, std::size_t column, const NativeEmbeddingIQTables &tables)
    {
        using Block = typename SourceBlock<Format>::type;
        using device_fp32_contract::add;
        using device_fp32_contract::multiply;
        const auto &b = static_cast<const Block *>(row)[column / Block::BLOCK_SIZE];
        const unsigned i = unsigned(column % Block::BLOCK_SIZE);
        if constexpr (Format == TensorType::Q4_0)
        {
            const int code = (b.qs[i & 15] >> (4 * (i >> 4))) & 15;
            return multiply(half(b.d), float(code - 8));
        }
        else if constexpr (Format == TensorType::Q4_1)
        {
            const int code = (b.qs[i & 15] >> (4 * (i >> 4))) & 15;
            return add(multiply(half(b.d), float(code)), half(b.m));
        }
        else if constexpr (Format == TensorType::Q5_0 || Format == TensorType::Q5_1)
        {
            int code = ((b.qs[i & 15] >> (4 * (i >> 4))) & 15) |
                       (((b.qh[i >> 3] >> (i & 7)) & 1) << 4);
            if constexpr (Format == TensorType::Q5_0)
                return multiply(half(b.d), float(code - 16));
            else
                return add(multiply(half(b.d), float(code)), half(b.m));
        }
        else if constexpr (Format == TensorType::Q8_0 || Format == TensorType::Q8_1)
            return multiply(half(b.d), float(b.qs[i]));
        else if constexpr (Format == TensorType::Q8_K)
            // Original FP32 source scales share the canonical non-FTZ multiply;
            // no format-local CUDA arithmetic implementation is required.
            return multiply(b.d, float(b.qs[i]));
        else if constexpr (Format == TensorType::Q2_K)
        {
            const unsigned metadata = b.scales[i / 16];
            const int code = (b.qs[(i / 128) * 32 + (i & 31)] >>
                              (2 * ((i & 127) / 32))) & 3;
            return add(multiply(multiply(half(b.d), float(metadata & 15)),
                                float(code)),
                       -multiply(half(b.dmin), float(metadata >> 4)));
        }
        else if constexpr (Format == TensorType::Q3_K)
        {
            const unsigned group = i / 16;
            const unsigned low = (b.scales[group & 7] >> (4 * (group / 8))) & 15;
            const unsigned high = (b.scales[8 + (group & 3)] >>
                                   (2 * (group / 4))) & 3;
            const int scale = int(low | (high << 4)) - 32;
            const int code = int((b.qs[(i / 128) * 32 + (i & 31)] >>
                                   (2 * ((i & 127) / 32))) & 3) -
                             ((b.hmask[i & 31] & (1u << (i / 32))) ? 0 : 4);
            return multiply(multiply(half(b.d), float(scale)), float(code));
        }
        else if constexpr (Format == TensorType::Q4_K || Format == TensorType::Q5_K)
        {
            const unsigned group = i / 32;
            const unsigned scale = group < 4 ? b.scales[group] & 63 :
                (b.scales[group + 4] & 15) | ((b.scales[group - 4] >> 6) << 4);
            const unsigned minimum = group < 4 ? b.scales[group + 4] & 63 :
                (b.scales[group + 4] >> 4) | ((b.scales[group] >> 6) << 4);
            int code = (b.qs[(i / 64) * 32 + (i & 31)] >>
                        (4 * ((i / 32) & 1))) & 15;
            if constexpr (Format == TensorType::Q5_K)
                code |= ((b.qh[i & 31] >> group) & 1) << 4;
            return add(multiply(multiply(half(b.d), float(scale)), float(code)),
                       -multiply(half(b.dmin), float(minimum)));
        }
        else if constexpr (Format == TensorType::Q6_K)
        {
            const unsigned section = (i & 127) / 32;
            const unsigned lo_index = (i / 128) * 64 + (section & 1) * 32 + (i & 31);
            const unsigned hi_index = (i / 128) * 32 + (i & 31);
            const int code = int(((b.ql[lo_index] >> (4 * (section / 2))) & 15) |
                                (((b.qh[hi_index] >> (2 * section)) & 3) << 4)) - 32;
            return multiply(multiply(half(b.d), float(b.scales[i / 16])), float(code));
        }
        else if constexpr (Format == TensorType::IQ4_NL)
            return multiply(half(b.d), iq4((b.qs[i & 15] >> (4 * (i / 16))) & 15));
        else if constexpr (Format == TensorType::IQ4_XS)
        {
            const unsigned group = i / 32;
            const int scale = int(((b.scales_l[group / 2] >> (4 * (group & 1))) & 15) |
                                   (((word16(&b.scales_h) >> (2 * group)) & 3) << 4)) - 32;
            const unsigned code = (b.qs[group * 16 + (i & 15)] >>
                                   (4 * ((i & 31) / 16))) & 15;
            return multiply(multiply(half(b.d), float(scale)), iq4(code));
        }
        else if constexpr (Format == TensorType::IQ2_XXS)
        {
            const auto *bytes = reinterpret_cast<const std::uint8_t *>(b.qs) + (i / 32) * 8;
            const unsigned local_group = (i & 31) / 8;
            const unsigned metadata = word32(bytes + 4);
            const unsigned index = bytes[local_group];
            const unsigned magnitude = unsigned((tables.iq2xxs[index] >> (8 * (i & 7))) & 255);
            const unsigned mask = signs((metadata >> (7 * local_group)) & 127);
            const float scale = multiply(multiply(half(b.d), .5f + float(metadata >> 28)), .25f);
            const float value = multiply(scale, float(magnitude));
            return mask & (1u << (i & 7)) ? -value : value;
        }
        else if constexpr (Format == TensorType::IQ2_XS || Format == TensorType::IQ2_S)
        {
            const unsigned group = i / 32;
            const unsigned local_group = (i & 31) / 8;
            unsigned index, mask;
            std::uint64_t grid;
            if constexpr (Format == TensorType::IQ2_XS)
            {
                const unsigned packed = word16(reinterpret_cast<const std::uint8_t *>(b.qs) + 2 * (i / 8));
                index = packed & 511;
                mask = signs(packed >> 9);
                grid = tables.iq2xs[index];
            }
            else
            {
                index = b.qs[i / 8] | (((b.qh[group] >> (2 * local_group)) & 3) << 8);
                mask = b.qs[32 + i / 8];
                grid = tables.iq2s[index];
            }
            const unsigned magnitude = unsigned((grid >> (8 * (i & 7))) & 255);
            const unsigned subscale = (b.scales[group] >> (4 * (local_group / 2))) & 15;
            const float scale = multiply(multiply(half(b.d), .5f + float(subscale)), .25f);
            const float value = multiply(scale, float(magnitude));
            return mask & (1u << (i & 7)) ? -value : value;
        }
        else if constexpr (Format == TensorType::IQ3_XXS || Format == TensorType::IQ3_S)
        {
            const unsigned group = i / 32;
            unsigned mask, index;
            float scale;
            std::uint32_t grid;
            if constexpr (Format == TensorType::IQ3_XXS)
            {
                const unsigned metadata = word32(b.qs + 64 + group * 4);
                index = b.qs[i / 4];
                mask = signs((metadata >> (7 * ((i & 31) / 8))) & 127);
                scale = multiply(multiply(half(b.d), .5f + float(metadata >> 28)), .5f);
                grid = tables.iq3xxs[index];
            }
            else
            {
                index = b.qs[i / 4] | (((b.qh[group] >> ((i & 31) / 4)) & 1) << 8);
                mask = b.signs[i / 8];
                const unsigned subscale = (b.scales[group / 2] >> (4 * (group & 1))) & 15;
                scale = multiply(half(b.d), float(1 + 2 * subscale));
                grid = tables.iq3s[index];
            }
            const unsigned magnitude = (grid >> (8 * (i & 3))) & 255;
            const float value = multiply(scale, float(magnitude));
            return mask & (1u << (i & 7)) ? -value : value;
        }
        else if constexpr (Format == TensorType::IQ1_S || Format == TensorType::IQ1_M)
        {
            unsigned index;
            float scale, delta;
            if constexpr (Format == TensorType::IQ1_S)
            {
                const unsigned metadata = word16(reinterpret_cast<const std::uint8_t *>(b.qh) + 2 * (i / 32));
                index = b.qs[i / 8] | (((metadata >> (3 * ((i & 31) / 8))) & 7) << 8);
                scale = multiply(half(b.d), float(2 * ((metadata >> 12) & 7) + 1));
                delta = metadata & 0x8000u ? -.125f : .125f;
            }
            else
            {
                const unsigned s0 = word16(b.scales);
                const unsigned s1 = word16(b.scales + 2);
                const unsigned s2 = word16(b.scales + 4);
                const unsigned s3 = word16(b.scales + 6);
                const auto bits = std::uint16_t((s0 >> 12) | ((s1 >> 8) & 0xf0) |
                                                ((s2 >> 4) & 0xf00) | (s3 & 0xf000));
                const unsigned metadata = word16(b.scales + 2 * (i / 64));
                const unsigned subscale = (metadata >> (3 * ((i & 63) / 16))) & 7;
                const unsigned high = b.qh[i / 16] >> (4 * ((i / 8) & 1));
                index = b.qs[i / 8] | ((high & 7) << 8);
                scale = multiply(half(bits), float(2 * subscale + 1));
                delta = high & 8 ? -.125f : .125f;
            }
            const unsigned byte = unsigned((tables.iq1s[index] >> (8 * (i & 7))) & 255);
            const int grid = int(byte) - (byte >= 128 ? 256 : 0);
            return multiply(scale, add(float(grid), delta));
        }
    }
}
#undef LLAMINAR_NATIVE_EMBED_INLINE
