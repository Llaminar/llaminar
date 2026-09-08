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
    // Helper: is this codebook decodable from raw payload bytes?
    //
    // Per-block formats (Q4_0, IQ4_NL, Q4_1, Q5_0, Q5_1, IQ4_XS) store their
    // payload as self-contained raw bytes that can be decoded without external
    // context. Superblock formats (Q6_K, Q3_K, Q2_K, IQ2/3*) require the full
    // superblock header for correct decode, so their payload is NOT decodable
    // in isolation — use unpack_superblock_to_int8() at pack time instead.
    // =========================================================================

    inline bool is_payload_decodable(uint8_t codebook_id)
    {
        switch (codebook_id)
        {
        case 0: // Q4_0
        case 4: // IQ4_NL / IQ4_XS (both use kvalues_iq4nl LUT)
        case 5: // Q4_1
        case 6: // Q5_0
        case 7: // Q5_1
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
    // Generic dispatcher by codebook_id
    //
    // Only works for payload-decodable formats (per-block formats).
    // For superblock formats, the packer uses unpack_superblock_to_int8()
    // at pack time and stores pre-decoded INT8 values directly.
    // =========================================================================

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
        default:
            // Superblock formats (Q6_K, Q3_K, Q2_K, IQ2/3/1*) are not
            // payload-decodable. The packer pre-decodes them via
            // unpack_superblock_to_int8() at pack time.
            std::memset(dst, 0, 32);
            break;
        }
    }

} // namespace llaminar2::cpu::native_vnni
