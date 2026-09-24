/**
 * @file CPUNativeVNNIDecode.h
 * @brief Inline decode functions for native quantized formats to INT8.
 *
 * Shared between the weight packer (pack-time VNNI transpose) and the
 * kernel (runtime scalar reference path).
 */

#pragma once

#include <cstdint>
#include <cstring>

#include "tensors/BlockStructures.h"
#include "tensors/SIMDHelpers.h"

namespace llaminar2::cpu::native_vnni
{

    // =========================================================================
    // Q4_0 nibble decode: 16 bytes → 32 INT8 values (range [-8, +7])
    // =========================================================================

    inline void decode_q4_0_block(const uint8_t *__restrict payload, int8_t *__restrict dst)
    {
        Q4_0Block block;
        block.d = 0;
        std::memcpy(block.qs, payload, 16);
        simd::unpack_q4_0_to_int8(block, dst);
    }

    // =========================================================================
    // IQ4_NL nibble decode: 16 bytes → 32 INT8 via non-linear LUT
    // =========================================================================

    inline void decode_iq4_nl_block(const uint8_t *__restrict payload, int8_t *__restrict dst)
    {
        IQ4_NLBlock block;
        block.d = 0;
        std::memcpy(block.qs, payload, 16);
        simd::unpack_iq4_nl_to_int8(block, dst);
    }

    // =========================================================================
    // Q4_1 nibble decode: 16 bytes → 32 INT8 values (range [0, 15])
    // =========================================================================

    inline void decode_q4_1_block(const uint8_t *__restrict payload, int8_t *__restrict dst)
    {
        Q4_1Block block;
        block.d = 0;
        block.m = 0;
        std::memcpy(block.qs, payload, 16);
        simd::unpack_q4_1_to_int8(block, dst);
    }

    // =========================================================================
    // IQ4_XS nibble decode: 16 bytes → 32 INT8 via IQ4_NL non-linear LUT
    // (IQ4_XS uses same kvalues_iq4nl codebook as IQ4_NL)
    // =========================================================================

    inline void decode_iq4_xs_block(const uint8_t *__restrict payload, int8_t *__restrict dst)
    {
        // IQ4_XS sub-blocks use the same nibble→int8 mapping as IQ4_NL
        IQ4_NLBlock block;
        block.d = 0;
        std::memcpy(block.qs, payload, 16);
        simd::unpack_iq4_nl_to_int8(block, dst);
    }

    // =========================================================================
    // Q5_0 decode: 20 bytes (qs[16] + qh[4]) → 32 INT8 (range [-16, +15])
    // =========================================================================

    inline void decode_q5_0_block(const uint8_t *__restrict payload, int8_t *__restrict dst)
    {
        Q5_0Block block;
        block.d = 0;
        std::memcpy(block.qs, payload, 16);
        std::memcpy(block.qh, payload + 16, 4);
        simd::unpack_q5_0_to_int8(block, dst);
    }

    // =========================================================================
    // Q5_1 decode: 20 bytes (qs[16] + qh[4]) → 32 INT8 (range [0, 31])
    // =========================================================================

    inline void decode_q5_1_block(const uint8_t *__restrict payload, int8_t *__restrict dst)
    {
        Q5_1Block block;
        block.d = 0;
        block.m = 0;
        std::memcpy(block.qs, payload, 16);
        std::memcpy(block.qh, payload + 16, 4);
        simd::unpack_q5_1_to_int8(block, dst);
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
    // Helper: does this codebook use 4-bit vpshufb LUT decode in the GEMV?
    //
    // Only 4-bit formats where each nibble maps to exactly one INT8 value
    // via a 16-entry LUT can use the vpshufb fast path. All other formats
    // use pre-decoded INT8 in the GEMV inner loop.
    // =========================================================================

    inline bool is_nibble_lut_format(uint8_t codebook_id)
    {
        switch (codebook_id)
        {
        case 0: // Q4_0:  nibble - 8 → [-8, +7]
        case 4: // IQ4_NL / IQ4_XS: kvalues_iq4nl LUT
        case 5: // Q4_1:  nibble → [0, 15]
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

} // namespace llaminar2::cpu::native_vnni
