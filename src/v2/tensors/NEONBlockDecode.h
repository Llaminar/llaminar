/**
 * @file NEONBlockDecode.h
 * @brief Native AArch64 signed-byte decode of the simple 32-element quantized blocks.
 *
 * Source-layout nibble/high-bit decoding is shared by CPU tensor conversion
 * and NativeQuantised preparation/diagnostics. It preserves original integer values;
 * scale/minimum application belongs to the separate numerical contract. Each
 * call reads one complete typed block and writes exactly 32 signed bytes.
 */
#pragma once

#include "BlockStructures.h"
#include "IQQuantTables.h"

#if !defined(__aarch64__) || !defined(__ARM_NEON)
#error "NEON block decode requires native AArch64 ASIMD"
#endif
#include <arm_neon.h>
#include <cstdint>

namespace llaminar2::neon_blocks
{
    /**
     * @brief Decode low/high nibble halves with an exact integer centering bias.
     * @tparam Bias Zero for unsigned Q4_1 or eight for signed Q4_0.
     * @param bytes Sixteen source nibble bytes.
     * @param destination Exactly 32 writable signed bytes, in source element order.
     */
    template <int Bias>
    inline void unpackNibbles(const std::uint8_t *bytes, std::int8_t *destination) noexcept
    {
        static_assert(Bias == 0 || Bias == 8);
        const uint8x16_t packed = vld1q_u8(bytes);
        const int8x16_t low = vreinterpretq_s8_u8(vandq_u8(packed, vdupq_n_u8(15)));
        const int8x16_t high = vreinterpretq_s8_u8(vshrq_n_u8(packed, 4));
        vst1q_s8(destination, vsubq_s8(low, vdupq_n_s8(Bias)));
        vst1q_s8(destination + 16, vsubq_s8(high, vdupq_n_s8(Bias)));
    }

    /**
     * @brief Decode the two high-bit planes of a five-bit source block.
     * @tparam Bias Sixteen for Q5_0, zero for Q5_1.
     * @param nibbles Sixteen low/high nibble bytes.
     * @param high_bits Four little-endian high-bit bytes, eight source values each.
     * @param destination Exactly 32 writable signed source-order bytes.
     */
    template <int Bias>
    inline void unpackFiveBit(const std::uint8_t *nibbles,
        const std::uint8_t *high_bits, std::int8_t *destination) noexcept
    {
        static_assert(Bias == 0 || Bias == 16);
        const uint8x16_t packed = vld1q_u8(nibbles);
        const uint8x16_t positions{1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
        const uint8x16_t first_high = vcombine_u8(vdup_n_u8(high_bits[0]), vdup_n_u8(high_bits[1]));
        const uint8x16_t second_high = vcombine_u8(vdup_n_u8(high_bits[2]), vdup_n_u8(high_bits[3]));
        const uint8x16_t first_bit = vandq_u8(vtstq_u8(first_high, positions), vdupq_n_u8(16));
        const uint8x16_t second_bit = vandq_u8(vtstq_u8(second_high, positions), vdupq_n_u8(16));
        const int8x16_t first = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(packed, vdupq_n_u8(15)), first_bit));
        const int8x16_t second = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(packed, 4), second_bit));
        vst1q_s8(destination, vsubq_s8(first, vdupq_n_s8(Bias)));
        vst1q_s8(destination + 16, vsubq_s8(second, vdupq_n_s8(Bias)));
    }

    /** @brief Expand Q4_0 integers without applying its half scale.
     * @param block Complete typed source block.
     * @param destination Exactly 32 signed output bytes.
     */
    inline void unpack(const Q4_0Block &block, std::int8_t *destination) noexcept
    { unpackNibbles<8>(block.qs, destination); }

    /** @brief Expand Q4_1 integers without applying its scale or additive minimum.
     * @param block Complete typed source block.
     * @param destination Exactly 32 signed output bytes.
     */
    inline void unpack(const Q4_1Block &block, std::int8_t *destination) noexcept
    { unpackNibbles<0>(block.qs, destination); }

    /** @brief Expand IQ4_NL through the canonical signed non-linear codebook.
     * @param block Complete typed source block.
     * @param destination Exactly 32 signed output bytes.
     */
    inline void unpack(const IQ4_NLBlock &block, std::int8_t *destination) noexcept
    {
        const uint8x16_t packed = vld1q_u8(block.qs);
        const int8x16_t table = vld1q_s8(kvalues_iq4nl_i8);
        vst1q_s8(destination, vqtbl1q_s8(table, vandq_u8(packed, vdupq_n_u8(15))));
        vst1q_s8(destination + 16, vqtbl1q_s8(table, vshrq_n_u8(packed, 4)));
    }

    /** @brief Expand Q5_0 high bits/nibbles to the exact signed integer domain.
     * @param block Complete typed source block.
     * @param destination Exactly 32 signed output bytes.
     */
    inline void unpack(const Q5_0Block &block, std::int8_t *destination) noexcept
    { unpackFiveBit<16>(block.qs, block.qh, destination); }

    /** @brief Expand Q5_1 high bits/nibbles without its separate minimum correction.
     * @param block Complete typed source block.
     * @param destination Exactly 32 signed output bytes.
     */
    inline void unpack(const Q5_1Block &block, std::int8_t *destination) noexcept
    { unpackFiveBit<0>(block.qs, block.qh, destination); }
} // namespace llaminar2::neon_blocks
