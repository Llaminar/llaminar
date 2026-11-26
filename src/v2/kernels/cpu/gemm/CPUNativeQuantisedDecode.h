/**
 * @file CPUNativeQuantisedDecode.h
 * @brief Inline decode functions for native quantized formats to INT8.
 *
 * Shared between weight preparation and diagnostic source-order reconstruction.
 * Simple source blocks use native NEON on AArch64 and the existing x86 tensor
 * helpers on x86. Hot interleaved kernels consume their own native vector
 * decode, without constructing a source-order or expanded matrix.
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <stdexcept>

#include "tensors/BlockStructures.h"
#include "CPUNativeQuantisedEncoding.h"
#if defined(__aarch64__) && defined(__ARM_NEON)
#include "tensors/NEONBlockDecode.h"
#else
#include "tensors/SIMDHelpers.h"
#endif

namespace llaminar2::cpu::native_quantised
{

    // =========================================================================
    // Q4_0 nibble decode: 16 bytes → 32 INT8 values (range [-8, +7])
    // =========================================================================

    /**
     * @brief Expand Q4_0 nibbles to centered signed source integers.
     * @param payload Complete 16-byte prepared payload, excluding scales/minima.
     * @param dst Exactly 32 writable signed source-order bytes.
     */
    inline void decode_q4_0_block(const uint8_t *__restrict payload, int8_t *__restrict dst)
    {
        Q4_0Block block;
        block.d = 0;
        std::memcpy(block.qs, payload, 16);
#if defined(__aarch64__) && defined(__ARM_NEON)
        neon_blocks::unpack(block, dst);
#else
        simd::unpack_q4_0_to_int8(block, dst);
#endif
    }

    // =========================================================================
    // IQ4_NL nibble decode: 16 bytes → 32 INT8 via non-linear LUT
    // =========================================================================

    /**
     * @brief Expand IQ4_NL nibbles through the immutable signed codebook.
     * @param payload Complete 16-byte prepared payload, excluding scales/minima.
     * @param dst Exactly 32 writable signed source-order bytes.
     */
    inline void decode_iq4_nl_block(const uint8_t *__restrict payload, int8_t *__restrict dst)
    {
        IQ4_NLBlock block;
        block.d = 0;
        std::memcpy(block.qs, payload, 16);
#if defined(__aarch64__) && defined(__ARM_NEON)
        neon_blocks::unpack(block, dst);
#else
        simd::unpack_iq4_nl_to_int8(block, dst);
#endif
    }

    // =========================================================================
    // Q4_1 nibble decode: 16 bytes → 32 INT8 values (range [0, 15])
    // =========================================================================

    /**
     * @brief Expand Q4_1 nibbles without applying its separate minimum.
     * @param payload Complete 16-byte prepared payload, excluding scales/minima.
     * @param dst Exactly 32 writable signed source-order bytes.
     */
    inline void decode_q4_1_block(const uint8_t *__restrict payload, int8_t *__restrict dst)
    {
        Q4_1Block block;
        block.d = 0;
        block.m = 0;
        std::memcpy(block.qs, payload, 16);
#if defined(__aarch64__) && defined(__ARM_NEON)
        neon_blocks::unpack(block, dst);
#else
        simd::unpack_q4_1_to_int8(block, dst);
#endif
    }

    // =========================================================================
    // IQ4_XS nibble decode: 16 bytes → 32 INT8 via IQ4_NL non-linear LUT
    // (IQ4_XS uses same kvalues_iq4nl codebook as IQ4_NL)
    // =========================================================================

    /**
     * @brief Expand an IQ4_XS prepared nibble block without modifying source scales.
     * @param payload Complete 16-byte prepared payload, excluding scales/minima.
     * @param dst Exactly 32 writable signed source-order bytes.
     */
    inline void decode_iq4_xs_block(const uint8_t *__restrict payload, int8_t *__restrict dst)
    {
        // IQ4_XS sub-blocks use the same nibble→int8 mapping as IQ4_NL
        IQ4_NLBlock block;
        block.d = 0;
        std::memcpy(block.qs, payload, 16);
#if defined(__aarch64__) && defined(__ARM_NEON)
        neon_blocks::unpack(block, dst);
#else
        simd::unpack_iq4_nl_to_int8(block, dst);
#endif
    }

    // =========================================================================
    // Q5_0 decode: 20 bytes (qs[16] + qh[4]) → 32 INT8 (range [-16, +15])
    // =========================================================================

    /**
     * @brief Expand Q5_0 payload/high bits to centered signed source integers.
     * @param payload Complete 20-byte prepared payload, excluding scales/minima.
     * @param dst Exactly 32 writable signed source-order bytes.
     */
    inline void decode_q5_0_block(const uint8_t *__restrict payload, int8_t *__restrict dst)
    {
        Q5_0Block block;
        block.d = 0;
        std::memcpy(block.qs, payload, 16);
        std::memcpy(block.qh, payload + 16, 4);
#if defined(__aarch64__) && defined(__ARM_NEON)
        neon_blocks::unpack(block, dst);
#else
        simd::unpack_q5_0_to_int8(block, dst);
#endif
    }

    // =========================================================================
    // Q5_1 decode: 20 bytes (qs[16] + qh[4]) → 32 INT8 (range [0, 31])
    // =========================================================================

    /**
     * @brief Expand Q5_1 payload/high bits without its separate minimum.
     * @param payload Complete 20-byte prepared payload, excluding scales/minima.
     * @param dst Exactly 32 writable signed source-order bytes.
     */
    inline void decode_q5_1_block(const uint8_t *__restrict payload, int8_t *__restrict dst)
    {
        Q5_1Block block;
        block.d = 0;
        block.m = 0;
        std::memcpy(block.qs, payload, 16);
        std::memcpy(block.qh, payload + 16, 4);
#if defined(__aarch64__) && defined(__ARM_NEON)
        neon_blocks::unpack(block, dst);
#else
        simd::unpack_q5_1_to_int8(block, dst);
#endif
    }

    // =========================================================================
    // Helper: is this codebook's prepared payload decodable to signed bytes?
    // Source superblocks are not prepared payloads. packVnniBlock extracts
    // their scale metadata first; a prepared IQuant payload then contains all
    // grid indices and signs needed for integer-only decode.
    // =========================================================================

    /** @brief Whether integer-only decode of a prepared payload is implemented. */
    inline bool is_payload_decodable(uint8_t codebook_id)
    {
        switch (codebook_id)
        {
        case 0: // Q4_0
        case 4: // IQ4_NL / IQ4_XS (both use kvalues_iq4nl LUT)
        case 5: // Q4_1
        case 6: // Q5_0
        case 7: // Q5_1
        case 9: // Q3_K
        case 10: // Q2_K
        case 11: // IQ3_S
        case 12: // IQ3_XXS
        case 13: // IQ2_S
        case 14: // IQ2_XS
        case 15: // IQ2_XXS
        case 16: // IQ1_S (delta remains separate minimum metadata)
        case 17: // IQ1_M
            return true;
        default:
            return false;
        }
    }

    // =========================================================================
    // Generic dispatcher by codebook_id. Decode never folds scales/minima
    // into the integer values: doing so adds another quantization step.
    // =========================================================================

    /**
     * @brief Expand one prepared native payload without modifying its integers.
     * @param codebook_id Payload decoder identity, not a source GGUF block tag.
     * @param payload Complete prepared payload from packVnniBlock().
     * @param dst Destination for exactly 32 signed bytes in source element order.
     * @throws std::invalid_argument If no exact integer decoder is installed.
     *
     * Scale/minimum application is deliberately absent. Grid values, signs,
     * and centering are exact; IQ1_S's fractional delta remains separate so
     * CPU and GPU use the same FP32 correction tree after the integer dot.
     */
    inline void decode_native_block(uint8_t codebook_id,
                                    const uint8_t *__restrict payload,
                                    int8_t *__restrict dst)
    {
        switch (codebook_id)
        {
        case 0: // Q4_0
            decode_q4_0_block(payload, dst);
            break;
        case 4: // IQ4_NL
            decode_iq4_nl_block(payload, dst);
            break;
        case 5: // Q4_1
            decode_q4_1_block(payload, dst);
            break;
        case 6: // Q5_0
            decode_q5_0_block(payload, dst);
            break;
        case 7: // Q5_1
            decode_q5_1_block(payload, dst);
            break;
        case 9: // Q3_K: two sixteen-value halves, each with four interleaved planes.
        case 10: // Q2_K: the same low planes without the Q3 high bits.
        {
            uint32_t high = 0;
            if (codebook_id == 9)
                std::memcpy(&high, payload + 8, sizeof(high));
            for (int element = 0; element < 32; ++element)
            {
                const int half = element / 16;
                const int group = (element % 16) / 4;
                int value = (payload[half * 4 + element % 4] >> (group * 2)) & 3;
                if (codebook_id == 9)
                    value += ((high >> element) & 1) * 4 - 4;
                dst[element] = static_cast<int8_t>(value);
            }
            break;
        }
        case 13: // IQ2_S: four signed grids, two independent half-block scales.
        case 14: // IQ2_XS: narrower grid index with the same arithmetic geometry.
            for (int group = 0; group < 4; ++group)
            {
                const int bits = codebook_id == 13 ? 2 : 1;
                const int index = payload[group] |
                    (((payload[4] >> (group * bits)) & ((1 << bits) - 1)) << 8);
                const uint64_t grid = codebook_id == 13 ? iq2s_grid[index] : iq2xs_grid[index];
                for (int lane = 0; lane < 8; ++lane)
                {
                    const int value = (grid >> (8 * lane)) & 255;
                    dst[group * 8 + lane] = static_cast<int8_t>(
                        (payload[5 + group] & (1 << lane)) ? -value : value);
                }
            }
            break;
        case 17: // IQ1_M: preserve signed integer grids; delta bits stay separate.
            for (int group = 0; group < 4; ++group)
            {
                const int index = payload[group] |
                    (((payload[4 + group / 2] >> ((group % 2) * 4)) & 7) << 8);
                const uint64_t grid = iq1s_grid[index];
                std::memcpy(dst + group * 8, &grid, sizeof(grid));
            }
            break;
        case 11: // IQ3_S: eight four-element grid entries and four sign bytes.
        case 12: // IQ3_XXS: same integer domain, smaller grid indices.
            for (int group = 0; group < 8; ++group)
            {
                const bool iq3s = codebook_id == 11;
                const int index = payload[group] |
                    (iq3s ? ((payload[8] >> group) & 1) << 8 : 0);
                const uint32_t grid = iq3s ? iq3s_grid[index] : iq3xxs_grid[index];
                const uint8_t signs = payload[(iq3s ? 9 : 8) + group / 2];
                for (int lane = 0; lane < 4; ++lane)
                {
                    const int value = (grid >> (8 * lane)) & 255;
                    dst[group * 4 + lane] = static_cast<int8_t>(
                        (signs & (1 << ((group % 2) * 4 + lane))) ? -value : value);
                }
            }
            break;
        case 15: // IQ2_XXS: four eight-element grids, one sign byte each.
            for (int group = 0; group < 4; ++group)
            {
                const uint64_t grid = iq2xxs_grid[payload[group]];
                for (int lane = 0; lane < 8; ++lane)
                {
                    const int value = (grid >> (8 * lane)) & 255;
                    dst[group * 8 + lane] = static_cast<int8_t>(
                        (payload[4 + group] & (1 << lane)) ? -value : value);
                }
            }
            break;
        case 16: // IQ1_S: signed {-1,0,1} grids; do not absorb the delta.
        {
            const uint16_t high = static_cast<uint16_t>(payload[4]) |
                (static_cast<uint16_t>(payload[5]) << 8);
            for (int group = 0; group < 4; ++group)
            {
                const int index = payload[group] | (((high >> (3 * group)) & 7) << 8);
                const uint64_t grid = iq1s_grid[index];
                std::memcpy(dst + group * 8, &grid, sizeof(grid));
            }
            break;
        }
        default:
            throw std::invalid_argument("Unsupported native payload integer decode");
        }
    }

    // =========================================================================
    // Canonical 16-entry signed LUTs and lossless 4-bit nibble encoders for
    // repacked sub-8-bit IQ formats (IQ3_S, IQ3_XXS, IQ2_XXS, IQ1_S).
    // =========================================================================

    /// IQ3_S exact 16-value odd signed domain: 2*nibble - 15 in [-15, +15].
    alignas(16) inline constexpr int8_t kIQ3SDecodeLUT[16] = {
        -15, -13, -11, -9, -7, -5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 15};

    /// IQ3_XXS exact 16-value signed grid domain.
    alignas(16) inline constexpr int8_t kIQ3XXSDecodeLUT[16] = {
        -62, -52, -44, -36, -28, -20, -12, -4, 4, 12, 20, 28, 36, 44, 52, 62};

    /// IQ2_XXS exact 6-value signed grid domain (index 0 = 0 so zero padding decodes cleanly).
    alignas(16) inline constexpr int8_t kIQ2XXSDecodeLUT[16] = {
        0, -43, -25, -8, 8, 25, 43, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    /// IQ1_S exact 3-value signed grid domain (index 0 = 0 so zero padding decodes cleanly).
    alignas(16) inline constexpr int8_t kIQ1SDecodeLUT[16] = {
        0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    /**
     * @brief Encode one signed decoded weight into its exact 4-bit nibble LUT index.
     * @param codebook_id One of 11 (IQ3_S), 12 (IQ3_XXS), 15 (IQ2_XXS), or 16 (IQ1_S).
     * @param value Exact signed decoded integer from the source grid.
     * @return Four-bit nibble in [0, 15] such that LUT[nibble] == value.
     * @throws std::invalid_argument If value is outside the codebook's exact domain.
     */
    inline uint8_t encode_nibble_for_codebook(uint8_t codebook_id, int8_t value)
    {
        switch (codebook_id)
        {
        case 11: // IQ3_S: odd integers in [-15, +15]
            if (value < -15 || value > 15 || (value & 1) == 0)
                throw std::invalid_argument("IQ3_S value outside canonical odd grid domain");
            return static_cast<uint8_t>((static_cast<int>(value) + 15) >> 1);
        case 12: // IQ3_XXS: exact 16-entry signed domain
            switch (value)
            {
            case -62: return 0;  case -52: return 1;  case -44: return 2;  case -36: return 3;
            case -28: return 4;  case -20: return 5;  case -12: return 6;  case -4:  return 7;
            case 4:   return 8;  case 12:  return 9;  case 20:  return 10; case 28:  return 11;
            case 36:  return 12; case 44:  return 13; case 52:  return 14; case 62:  return 15;
            default:
                throw std::invalid_argument("IQ3_XXS value outside canonical grid domain");
            }
        case 15: // IQ2_XXS: exact 6-value signed domain
            switch (value)
            {
            case -43: return 1; case -25: return 2; case -8: return 3;
            case 8:   return 4; case 25:  return 5; case 43: return 6;
            default:
                throw std::invalid_argument("IQ2_XXS value outside canonical grid domain");
            }
        case 16: // IQ1_S: exact {-1, 0, +1} domain
            if (value == 0) return 0;
            if (value == 1) return 1;
            if (value == -1) return 2;
            throw std::invalid_argument("IQ1_S value outside canonical {-1,0,1} domain");
        default:
            throw std::invalid_argument("Unsupported repacked nibble codebook");
        }
    }

    /**
     * @brief Pack 32 signed decoded integers into canonical 16-byte low/high nibble order.
     *
     * Byte i in [0, 15] stores element i in its low nibble and element i + 16 in
     * its high nibble, matching Q4_0/IQ4_NL/IQ4_XS/Q4_1 VNNI nibble layout.
     */
    inline void encode_nibble_lut_block(uint8_t codebook_id,
                                        const int8_t *__restrict decoded,
                                        uint8_t *__restrict nibble_payload_16)
    {
        for (int i = 0; i < 16; ++i)
        {
            const uint8_t lo = encode_nibble_for_codebook(codebook_id, decoded[i]);
            const uint8_t hi = encode_nibble_for_codebook(codebook_id, decoded[i + 16]);
            nibble_payload_16[i] = static_cast<uint8_t>(lo | (hi << 4));
        }
    }

    /**
     * @brief Decode one 16-byte prepared nibble-LUT block to 32 signed INT8 values.
     *
     * For Q4_0/IQ4_NL/IQ4_XS/Q4_1 this delegates to decode_native_block(); for
     * repacked IQ3_S/IQ3_XXS/IQ2_XXS/IQ1_S it expands the 16 packed nibble bytes
     * through their canonical 16-entry signed LUTs.
     */
    inline void decode_nibble_lut_block(uint8_t codebook_id,
                                        const uint8_t *__restrict nibble_payload_16,
                                        int8_t *__restrict dst)
    {
        const int8_t *lut = nullptr;
        switch (codebook_id)
        {
        case 0:
        case 4:
        case 5:
            decode_native_block(codebook_id, nibble_payload_16, dst);
            return;
        case 11: lut = kIQ3SDecodeLUT; break;
        case 12: lut = kIQ3XXSDecodeLUT; break;
        case 15: lut = kIQ2XXSDecodeLUT; break;
        case 16: lut = kIQ1SDecodeLUT; break;
        default:
            throw std::invalid_argument("Unsupported prepared nibble-LUT decode codebook");
        }
        for (int i = 0; i < 16; ++i)
        {
            const uint8_t packed = nibble_payload_16[i];
            dst[i] = lut[packed & 0x0Fu];
            dst[i + 16] = lut[packed >> 4];
        }
    }

} // namespace llaminar2::cpu::native_quantised
