/**
 * @file NativeQuantisedSuperblockMetadata.h
 * @brief Exact superblock-coded GPU metadata codec for GGUF superblock formats.
 *
 * GGUF superblock formats store one FP16 `d` (and `dmin`) per 256 values and a
 * few bits of scale code per 32-value block. The legacy GPU representation
 * replicates or pre-multiplies that information into per-32 FP16 planes
 * (primary, secondary and, for Q2_K, a packed effective-minimum word), which
 * makes the device footprint up to 1.6x the file.
 *
 * The superblock-coded representation stores instead:
 *  - a code plane: per 32-value block, exactly the source scale-code bits
 *    (superblockCodeBits() bits, bit-packed);
 *  - a superblock plane: per source superblock, the FP16 `d` and, where the
 *    format has minima, the FP16 `dmin` (superblockFieldBytes() bytes).
 *
 * decodeSuperblockMetadata() reconstructs the legacy staged words bit for bit:
 * it evaluates the repack kernels' float expressions in the same order and
 * publishes them through canonicalPreparedHalfBits(). Every consumer of the
 * staged words therefore keeps its instructions, reduction order and output
 * bytes. Test__NativeQuantisedSuperblockMetadata proves the equality
 * exhaustively (every code value x every d/dmin bit pattern) against the
 * literal repack expressions.
 *
 * Physical codebooks 5 and 7 are shared by Q4_1/Q4_K and Q5_1/Q5_K (identical
 * legacy planes), so the K-quant members need distinct physical codebooks
 * before they can adopt this representation; they are not superblock-coded yet.
 *
 * The codec is host/device: the same functions run in repack, loaders,
 * migration kernels and device-free tests.
 */
#pragma once

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#elif defined(__CUDACC__)
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#endif

#include "kernels/common/DeviceHalfMetadataContract.h"
#include "tensors/NativeQuantisedIQ2SScale.h"
#include "tensors/NativeQuantisedIQ3SScale.h"
#include "tensors/NativeQuantisedIQ4XSScale.h"
#include "tensors/NativeQuantisedQ6Scale.h"

#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_SUPERBLOCK_META_HD __host__ __device__ __forceinline__
#else
#define LLAMINAR_SUPERBLOCK_META_HD inline
#endif

namespace llaminar2::native_quantised_superblock
{
    /** @brief Values per source superblock (every GGUF K/IQ superblock format). */
    inline constexpr std::uint32_t kSuperblockValues = 256;
    /** @brief 32-value blocks per source superblock. */
    inline constexpr std::uint32_t kBlocksPerSuperblock = kSuperblockValues / 32;

    /**
     * @brief Legacy per-block staged metadata words, as the per-32 planes held them.
     *
     * `primary` is the primary-plane FP16 word, `secondary` the secondary
     * (mins/coefficient) word and `emin` the Q2_K packed effective-minimum
     * word. Unused fields are zero, exactly as absent planes read today.
     */
    struct StagedBlockMetadata
    {
        std::uint16_t primary = 0;
        std::uint16_t secondary = 0;
        std::uint32_t emin = 0;
    };

    /**
     * @brief Exact source scale-code bits per 32-value block of a codebook.
     *
     * These are the GGUF superblock's per-block code bits; the code plane
     * stores them bit-packed, so coded metadata matches the file density:
     * Q4_K/Q5_K/Q3_K 12 (two 6-bit fields), Q6_K/Q2_K 16 (two bytes),
     * IQ2_S/IQ2_XS 8, IQ4_XS/IQ1_M 6, IQ3_S/IQ3_XXS/IQ2_XXS/IQ1_S 4.
     * @return 0 for codebooks that are not superblock-coded.
     */
    LLAMINAR_SUPERBLOCK_META_HD constexpr std::uint32_t superblockCodeBits(std::uint8_t codebook) noexcept
    {
        switch (codebook)
        {
        case 5:  // Q4_K: 6-bit scale, 6-bit min
        case 7:  // Q5_K: 6-bit scale, 6-bit min
        case 9:  // Q3_K: two 6-bit scales
            return 12;
        case 8:  // Q6_K: two int8 coefficients
        case 10: // Q2_K: two scale/min bytes
            return 16;
        case 13: // IQ2_S: two 4-bit nibbles
        case 14: // IQ2_XS: two 4-bit nibbles
            return 8;
        case 17: // IQ1_M: two 3-bit scales
        case 24: // IQ4_XS source-scale physical codebook: 6-bit ls
            return 6;
        case 11: // IQ3_S
        case 12: // IQ3_XXS
        case 15: // IQ2_XXS
        case 16: // IQ1_S: 3-bit selector + delta sign
            return 4;
        default:
            return 0;
        }
    }

    /** @brief Whole-byte code width, used only where codes are addressed bytewise (tests). */
    LLAMINAR_SUPERBLOCK_META_HD constexpr std::uint32_t superblockCodeBytes(std::uint8_t codebook) noexcept
    {
        return (superblockCodeBits(codebook) + 7u) / 8u;
    }

    /**
     * @brief Whether a coded matrix keeps its per-block code inside the payload.
     *
     * IQ3_XXS (12), IQ2_XXS (15) and IQ2_XS (14) store each sign selector as an
     * 8-bit mask whose top bit is the even-parity of its seven sign bits, so
     * the four parity bits of a block are recomputable and carry the 4-bit
     * scale code instead (IQ2_XS also uses the unused high nibble of its qh
     * byte for the rest of its 8-bit code). IQ1_S (16) already stores its
     * scale selector and delta sign in qh bits 12..15. These matrices have no
     * code plane; the decoder restores sign parity (idempotent for per-block
     * payloads, whose parity bits are already correct).
     */
    LLAMINAR_SUPERBLOCK_META_HD constexpr bool superblockCodeInPayload(std::uint8_t codebook) noexcept
    {
        return codebook == 12 || codebook == 14 || codebook == 15 || codebook == 16;
    }

    /** @return Payload bytes per block of a code-in-payload codebook (0 otherwise). */
    LLAMINAR_SUPERBLOCK_META_HD constexpr std::uint32_t superblockCodePayloadBytes(std::uint8_t codebook) noexcept
    {
        return codebook == 12 ? 12u : codebook == 14 ? 9u : codebook == 15 ? 8u : codebook == 16 ? 6u : 0u;
    }

    /** @return Bits per block of the separate code plane (0 when the code lives in the payload). */
    LLAMINAR_SUPERBLOCK_META_HD constexpr std::uint32_t superblockCodePlaneBits(std::uint8_t codebook) noexcept
    {
        return superblockCodeInPayload(codebook) ? 0u : superblockCodeBits(codebook);
    }

    /**
     * @brief Padding after a bit-packed code plane.
     *
     * Readers fetch the up-to-three bytes covering a code (bit offset < 8 plus
     * at most 16 code bits); padding keeps the last code's fetch in bounds.
     */
    inline constexpr std::uint32_t kSuperblockCodePlanePadBytes = 4;

    /** @brief Restore the even-parity top bit of a seven-bit IQ sign selector (ksigns_iq2xs). */
    LLAMINAR_SUPERBLOCK_META_HD constexpr std::uint8_t restoreSignParity(std::uint8_t mask) noexcept
    {
        std::uint32_t parity = mask & 0x7Fu;
        parity ^= parity >> 4;
        parity ^= parity >> 2;
        parity ^= parity >> 1;
        return static_cast<std::uint8_t>((mask & 0x7Fu) | ((parity & 1u) << 7));
    }

    /**
     * @brief Read the code of a code-in-payload block.
     * @param payload The block's payload bytes (coded layout).
     */
    template <std::uint8_t Codebook>
    LLAMINAR_SUPERBLOCK_META_HD std::uint32_t codeFromPayload(const std::uint8_t *payload) noexcept
    {
        static_assert(superblockCodeInPayload(Codebook));
        if constexpr (Codebook == 16)
            return static_cast<std::uint32_t>(payload[5] >> 4);
        else
        {
            constexpr int signs = Codebook == 12 ? 8 : Codebook == 15 ? 4 : 5;
            std::uint32_t code = 0;
            for (int i = 0; i < 4; ++i)
                code |= static_cast<std::uint32_t>(payload[signs + i] >> 7) << i;
            if constexpr (Codebook == 14)
                code |= static_cast<std::uint32_t>(payload[4] >> 4) << 4;
            return code;
        }
    }

    /**
     * @brief Embed a block code into its (per-block-layout) payload bytes in place.
     * @param payload Payload exactly as the per-block repack emits it.
     * @param code Source code (nibble for 12/15, byte for 14; ignored for 16).
     */
    template <std::uint8_t Codebook>
    LLAMINAR_SUPERBLOCK_META_HD void embedCodeInPayload(std::uint8_t *payload, std::uint32_t code) noexcept
    {
        static_assert(superblockCodeInPayload(Codebook));
        if constexpr (Codebook != 16)
        {
            constexpr int signs = Codebook == 12 ? 8 : Codebook == 15 ? 4 : 5;
            for (int i = 0; i < 4; ++i)
                payload[signs + i] = static_cast<std::uint8_t>((payload[signs + i] & 0x7Fu) | (((code >> i) & 1u) << 7));
            if constexpr (Codebook == 14)
                payload[4] = static_cast<std::uint8_t>((payload[4] & 0x0Fu) | ((code >> 4) << 4));
        }
    }

    /** @return Whether @p codebook uses the superblock-coded representation. */
    LLAMINAR_SUPERBLOCK_META_HD constexpr bool isSuperblockCoded(std::uint8_t codebook) noexcept
    {
        return superblockCodeBytes(codebook) != 0;
    }

    /** @return Whether the format stores `dmin` beside `d` in its superblock word. */
    LLAMINAR_SUPERBLOCK_META_HD constexpr bool superblockHasDmin(std::uint8_t codebook) noexcept
    {
        return codebook == 5 || codebook == 7 || codebook == 10;
    }

    /** @return Superblock-plane bytes per source superblock (0 when not coded). */
    LLAMINAR_SUPERBLOCK_META_HD constexpr std::uint32_t superblockFieldBytes(std::uint8_t codebook) noexcept
    {
        return isSuperblockCoded(codebook) ? (superblockHasDmin(codebook) ? 4u : 2u) : 0u;
    }

    /** @brief Superblock word: `d` bits in the low half, `dmin` bits in the high half. */
    LLAMINAR_SUPERBLOCK_META_HD constexpr std::uint32_t packSuperblockWord(std::uint16_t d_bits,
                                                                        std::uint16_t dmin_bits = 0) noexcept
    {
        return static_cast<std::uint32_t>(d_bits) | (static_cast<std::uint32_t>(dmin_bits) << 16);
    }

    /**
     * @brief Exact IEEE binary16 -> binary32 widening (subnormals, inf, NaN).
     *
     * Equal to CUDA/HIP `__half2float` for every bit pattern; implemented with
     * integer operations so host and device produce identical floats.
     */
    LLAMINAR_SUPERBLOCK_META_HD float halfBitsToFloat(std::uint16_t h) noexcept
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        // Hardware widening is exact for every binary16 pattern and keeps
        // decode register pressure at the legacy kernels' level.
        return __half2float(__ushort_as_half(h));
#endif
        const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
        std::uint32_t exponent = (h >> 10) & 0x1Fu;
        std::uint32_t mantissa = h & 0x3FFu;
        std::uint32_t bits;
        if (exponent == 0x1Fu)
            bits = sign | 0x7F800000u | (mantissa << 13);
        else if (exponent != 0)
            bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
        else if (mantissa == 0)
            bits = sign;
        else
        {
            // Normalize the binary16 subnormal into a binary32 normal.
            std::uint32_t e = 113u;
            while ((mantissa & 0x400u) == 0)
            {
                mantissa <<= 1;
                --e;
            }
            bits = sign | (e << 23) | ((mantissa & 0x3FFu) << 13);
        }
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        return __uint_as_float(bits);
#else
        float out;
        std::memcpy(&out, &bits, sizeof(out));
        return out;
#endif
    }

    /**
     * @brief canonicalPreparedHalfBits() with the hardware conversion on device.
     *
     * `__float2half_rn` is IEEE round-to-nearest-even like the software
     * contract; the contract's signed-zero canonicalization is applied here,
     * and NaN keeps the contract's exact payload rule (cold path only).
     */
    LLAMINAR_SUPERBLOCK_META_HD std::uint16_t preparedHalfBits(float value) noexcept
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        if (value != value)
            return canonicalPreparedHalfBits(value);
        const std::uint16_t bits = __half_as_ushort(__float2half_rn(value));
        return bits == 0x8000u ? std::uint16_t{0} : bits;
#else
        return canonicalPreparedHalfBits(value);
#endif
    }

    /**
     * @brief Reconstruct the legacy staged words of one block.
     * @tparam Codebook Physical execution codebook (superblock-coded).
     * @param code Code-plane value of the block (zero-extended).
     * @param superblock Superblock word of the block's source superblock.
     * @return Staged words bit-identical to the legacy per-32 planes.
     *
     * Each expression is the corresponding repack kernel's, evaluated in the
     * same order (e.g. `(d * (0.5 + s)) * 0.25`), so binary32 rounding matches.
     */
    template <std::uint8_t Codebook>
    LLAMINAR_SUPERBLOCK_META_HD StagedBlockMetadata decodeSuperblockMetadata(std::uint32_t code,
                                                                              std::uint32_t superblock) noexcept
    {
        static_assert(isSuperblockCoded(Codebook), "decodeSuperblockMetadata requires a superblock-coded codebook");
        const std::uint16_t d_bits = static_cast<std::uint16_t>(superblock & 0xFFFFu);
        StagedBlockMetadata out;
        if constexpr (Codebook == 13 || Codebook == 11 || Codebook == 24 || Codebook == 8)
        {
            // Source-scale formats: the primary word is the original `d` and
            // the secondary word is the original coefficient, zero-extended.
            out.primary = d_bits;
            out.secondary = static_cast<std::uint16_t>(code);
        }
        else
        {
            const float d = halfBitsToFloat(d_bits);
            if constexpr (Codebook == 15)
            {
                out.primary = preparedHalfBits(d * (0.5f + static_cast<float>(code & 15u)) * 0.25f);
            }
            else if constexpr (Codebook == 14)
            {
                out.primary = preparedHalfBits(d * (0.5f + static_cast<float>(code & 15u)) * 0.25f);
                out.secondary = preparedHalfBits(d * (0.5f + static_cast<float>((code >> 4) & 15u)) * 0.25f);
            }
            else if constexpr (Codebook == 12)
            {
                out.primary = preparedHalfBits(d * (0.5f + static_cast<float>(code & 15u)) * 0.5f);
            }
            else if constexpr (Codebook == 16)
            {
                // code bits 0..2: scale selector; bit 3: delta sign (qh bit 15).
                const float dl = d * (2.0f * static_cast<float>(code & 7u) + 1.0f);
                const float delta = (code & 8u) ? -0.125f : 0.125f;
                out.primary = preparedHalfBits(dl);
                out.secondary = preparedHalfBits(dl * delta);
            }
            else if constexpr (Codebook == 17)
            {
                // code bits 0..2: low 3-bit scale; bits 3..5: high 3-bit scale.
                out.primary = preparedHalfBits(d * (2.0f * static_cast<float>(code & 7u) + 1.0f));
                out.secondary = preparedHalfBits(d * (2.0f * static_cast<float>((code >> 3) & 7u) + 1.0f));
            }
            else if constexpr (Codebook == 9)
            {
                // code: two unpacked 6-bit scales (low, high); stored biased by 32.
                out.primary = preparedHalfBits(d * static_cast<float>(static_cast<int>(code & 63u) - 32));
                out.secondary =
                    preparedHalfBits(d * static_cast<float>(static_cast<int>((code >> 6) & 63u) - 32));
            }
            else if constexpr (Codebook == 5 || Codebook == 7)
            {
                // Q4_K/Q5_K: code = 6-bit scale (low) | 6-bit min << 6.
                const float dmin = halfBitsToFloat(static_cast<std::uint16_t>(superblock >> 16));
                out.primary = preparedHalfBits(d * static_cast<float>(code & 63u));
                out.secondary = preparedHalfBits(-dmin * static_cast<float>((code >> 6) & 63u));
            }
            else if constexpr (Codebook == 10)
            {
                // code: original scale bytes of the two 16-value halves (low, high).
                const float dmin = halfBitsToFloat(static_cast<std::uint16_t>(superblock >> 16));
                const std::uint32_t lo = code & 0xFFu, hi = (code >> 8) & 0xFFu;
                out.primary = preparedHalfBits(d * static_cast<float>(lo & 0xFu));
                out.secondary = preparedHalfBits(d * static_cast<float>(hi & 0xFu));
                const std::uint16_t emin_lo = preparedHalfBits(-dmin * static_cast<float>(lo >> 4));
                const std::uint16_t emin_hi = preparedHalfBits(-dmin * static_cast<float>(hi >> 4));
                out.emin = static_cast<std::uint32_t>(emin_lo) | (static_cast<std::uint32_t>(emin_hi) << 16);
            }
        }
        return out;
    }

    /** @brief Runtime-codebook form of decodeSuperblockMetadata() (host tooling, tests). */
    LLAMINAR_SUPERBLOCK_META_HD StagedBlockMetadata decodeSuperblockMetadata(std::uint8_t codebook, std::uint32_t code,
                                                                              std::uint32_t superblock) noexcept
    {
        switch (codebook)
        {
        case 5: return decodeSuperblockMetadata<5>(code, superblock);
        case 7: return decodeSuperblockMetadata<7>(code, superblock);
        case 8: return decodeSuperblockMetadata<8>(code, superblock);
        case 9: return decodeSuperblockMetadata<9>(code, superblock);
        case 10: return decodeSuperblockMetadata<10>(code, superblock);
        case 11: return decodeSuperblockMetadata<11>(code, superblock);
        case 12: return decodeSuperblockMetadata<12>(code, superblock);
        case 13: return decodeSuperblockMetadata<13>(code, superblock);
        case 14: return decodeSuperblockMetadata<14>(code, superblock);
        case 15: return decodeSuperblockMetadata<15>(code, superblock);
        case 16: return decodeSuperblockMetadata<16>(code, superblock);
        case 17: return decodeSuperblockMetadata<17>(code, superblock);
        case 24: return decodeSuperblockMetadata<24>(code, superblock);
        default: return {};
        }
    }

    /**
     * @brief Typed address of one block's metadata.
     *
     * `linear` indexes the payload and code plane (unchanged legacy index);
     * `superblock` indexes the superblock plane.
     */
    struct BlockAddress
    {
        std::size_t linear = 0;
        std::size_t superblock = 0;
    };

    /**
     * @brief Source-superblock phase of a prepared matrix's first K block.
     *
     * A prepared matrix may start inside a source superblock (bounded column
     * slices). Its local block `b` belongs to local superblock
     * `(b + phase) / 8`.
     */
    LLAMINAR_SUPERBLOCK_META_HD constexpr std::uint32_t superblockPhase(std::uint32_t first_source_block) noexcept
    {
        return first_source_block % kBlocksPerSuperblock;
    }

    /** @return Local superblocks per row for @p blocks_per_row blocks starting at @p phase. */
    LLAMINAR_SUPERBLOCK_META_HD constexpr std::uint32_t superblocksPerRow(std::uint32_t blocks_per_row,
                                                                       std::uint32_t phase) noexcept
    {
        return (blocks_per_row + phase + kBlocksPerSuperblock - 1) / kBlocksPerSuperblock;
    }

    /**
     * @brief Address of block (@p block, @p row) in a block-major matrix (`linear = block * rows + row`).
     * @param rows Matrix rows N (the block-major row period).
     */
    LLAMINAR_SUPERBLOCK_META_HD constexpr BlockAddress blockMajorAddress(std::uint32_t block, std::uint32_t row,
                                                                      std::uint32_t rows, std::uint32_t phase) noexcept
    {
        return {static_cast<std::size_t>(block) * rows + row,
                static_cast<std::size_t>((block + phase) / kBlocksPerSuperblock) * rows + row};
    }

    /**
     * @brief Address of block (@p block, @p row) in a row-major matrix (`linear = row * blocks_per_row + block`).
     */
    LLAMINAR_SUPERBLOCK_META_HD constexpr BlockAddress rowMajorAddress(std::uint32_t block, std::uint32_t row,
                                                                    std::uint32_t blocks_per_row,
                                                                    std::uint32_t phase) noexcept
    {
        return {static_cast<std::size_t>(row) * blocks_per_row + block,
                static_cast<std::size_t>(row) * superblocksPerRow(blocks_per_row, phase) +
                    (block + phase) / kBlocksPerSuperblock};
    }

    /**
     * @brief Plane extents of one superblock-coded matrix.
     * @param rows Matrix rows N.
     * @param blocks_per_row Prepared 32-value blocks per row.
     * @param phase superblockPhase() of the first prepared block.
     */
    struct PlaneBytes
    {
        std::size_t code = 0;
        std::size_t superblock = 0;
    };

    LLAMINAR_SUPERBLOCK_META_HD constexpr PlaneBytes planeBytes(std::uint8_t codebook, std::size_t rows,
                                                             std::uint32_t blocks_per_row, std::uint32_t phase) noexcept
    {
        const std::size_t code_bits = rows * blocks_per_row * superblockCodePlaneBits(codebook);
        return {code_bits ? (code_bits + 7u) / 8u + kSuperblockCodePlanePadBytes : 0u,
                rows * superblocksPerRow(blocks_per_row, phase) * superblockFieldBytes(codebook)};
    }

    /**
     * @brief Read @p bits (<= 16) starting at bit @p bit of a byte plane.
     *
     * Fetches only the bytes that hold the field (at most three), so the
     * plane needs kSuperblockCodePlanePadBytes of padding and no alignment.
     */
    LLAMINAR_SUPERBLOCK_META_HD std::uint32_t loadPackedBits(const std::uint8_t *plane, std::size_t bit,
                                                             std::uint32_t bits) noexcept
    {
        const std::uint8_t *p = plane + (bit >> 3);
        const std::uint32_t shift = static_cast<std::uint32_t>(bit & 7u);
        std::uint32_t word = p[0];
        if (shift + bits > 8u) word |= static_cast<std::uint32_t>(p[1]) << 8;
        if (shift + bits > 16u) word |= static_cast<std::uint32_t>(p[2]) << 16;
        return (word >> shift) & ((1u << bits) - 1u);
    }

    /**
     * @brief Load one block's code.
     * @tparam Codebook Physical execution codebook.
     * @param code_plane Bit-packed code plane (or the payload for code-in-payload formats).
     * @param linear Block index in the matrix (payload index).
     * @param bit_shift Bit offset of the matrix's first code inside @p code_plane (0..7).
     */
    template <std::uint8_t Codebook>
    LLAMINAR_SUPERBLOCK_META_HD std::uint32_t loadCode(const void *code_plane, std::size_t linear,
                                                       std::uint32_t bit_shift = 0) noexcept
    {
        // Code-in-payload matrices bind the payload base as their code plane.
        if constexpr (superblockCodeInPayload(Codebook))
        {
            (void)bit_shift;
            return codeFromPayload<Codebook>(static_cast<const std::uint8_t *>(code_plane) +
                                             linear * superblockCodePayloadBytes(Codebook));
        }
        else
        {
            constexpr std::uint32_t bits = superblockCodeBits(Codebook);
            return loadPackedBits(static_cast<const std::uint8_t *>(code_plane),
                                  bit_shift + linear * bits, bits);
        }
    }

    /** @brief Runtime-codebook form of loadCode() (host tooling, tests). */
    LLAMINAR_SUPERBLOCK_META_HD std::uint32_t loadCode(std::uint8_t codebook, const void *code_plane,
                                                       std::size_t linear, std::uint32_t bit_shift = 0) noexcept
    {
        switch (codebook)
        {
        case 12: return loadCode<12>(code_plane, linear);
        case 14: return loadCode<14>(code_plane, linear);
        case 15: return loadCode<15>(code_plane, linear);
        case 16: return loadCode<16>(code_plane, linear);
        default:
        {
            const std::uint32_t bits = superblockCodeBits(codebook);
            return bits ? loadPackedBits(static_cast<const std::uint8_t *>(code_plane), bit_shift + linear * bits, bits)
                        : 0u;
        }
        }
    }

#if defined(__CUDACC__) || defined(__HIPCC__)
    /**
     * @brief Store @p bits of @p value at bit @p bit of a 4-byte-aligned plane.
     *
     * Concurrent producers own disjoint bit ranges: each clears and sets only
     * its own bits with atomicAnd/atomicOr on the (at most two) covering words,
     * so any interleaving publishes every field exactly and no prior zeroing is
     * required.
     */
    __device__ __forceinline__ void storePackedBits(void *plane, std::size_t bit, std::uint32_t bits,
                                                    std::uint32_t value)
    {
        auto *words = static_cast<unsigned int *>(plane);
        const std::size_t word = bit >> 5;
        const std::uint32_t shift = static_cast<std::uint32_t>(bit & 31u);
        const std::uint64_t mask = ((std::uint64_t{1} << bits) - 1u) << shift;
        const std::uint64_t field = (static_cast<std::uint64_t>(value) << shift) & mask;
        atomicAnd(&words[word], ~static_cast<unsigned int>(mask));
        atomicOr(&words[word], static_cast<unsigned int>(field));
        if (mask >> 32)
        {
            atomicAnd(&words[word + 1], ~static_cast<unsigned int>(mask >> 32));
            atomicOr(&words[word + 1], static_cast<unsigned int>(field >> 32));
        }
    }
#endif

    /** @brief Load one superblock word (d, or d|dmin<<16). */
    template <std::uint8_t Codebook>
    LLAMINAR_SUPERBLOCK_META_HD std::uint32_t loadSuperblock(const void *superblock_plane, std::size_t index) noexcept
    {
        if constexpr (superblockHasDmin(Codebook))
            return static_cast<const std::uint32_t *>(superblock_plane)[index];
        else
            return static_cast<const std::uint16_t *>(superblock_plane)[index];
    }

    /**
     * @brief Load and reconstruct the staged words of one block.
     * @param code_plane Code plane base (the matrix `scales` slot).
     * @param superblock_plane Superblock plane base (the matrix `mins` slot).
     */
    template <std::uint8_t Codebook>
    LLAMINAR_SUPERBLOCK_META_HD StagedBlockMetadata loadBlockMetadata(const void *code_plane,
                                                                       const void *superblock_plane,
                                                                       BlockAddress address,
                                                                       std::uint32_t code_bit_shift = 0) noexcept
    {
        return decodeSuperblockMetadata<Codebook>(loadCode<Codebook>(code_plane, address.linear, code_bit_shift),
                                                  loadSuperblock<Codebook>(superblock_plane, address.superblock));
    }

    /**
     * @brief Undo embedCodeInPayload(): restore a coded block's per-block payload bytes.
     *
     * Sign selectors regain their even-parity top bit and IQ2_XS's spare qh
     * nibble is cleared, reproducing exactly what the per-block repack emits.
     * IQ1_S and separate-plane formats are unchanged.
     */
    LLAMINAR_SUPERBLOCK_META_HD void restoreCodedPayload(std::uint8_t codebook, std::uint8_t *payload) noexcept
    {
        if (codebook != 12 && codebook != 14 && codebook != 15)
            return;
        const int signs = codebook == 12 ? 8 : codebook == 15 ? 4 : 5;
        for (int i = 0; i < 4; ++i)
            payload[signs + i] = restoreSignParity(payload[signs + i]);
        if (codebook == 14)
            payload[4] = static_cast<std::uint8_t>(payload[4] & 0x0Fu);
    }

    /**
     * @brief Decode a whole block-major coded matrix into legacy per-block words.
     *
     * Host tooling for oracles, diagnostics and tests that inspect prepared
     * bytes: reconstructs the exact staged primary/secondary/emin words every
     * block's reader sees (block index blk * rows + row) and restores the
     * per-block payload of code-in-payload formats in place.
     * @param codebook Codable physical codebook.
     * @param rows Matrix rows (block-major stride).
     * @param blocks_per_row Prepared K blocks.
     * @param phase Superblock phase of the first prepared block.
     * @param bit_shift Code-plane bit offset of the first code.
     * @param payload Payload bytes (payload_bytes per block), restored in place.
     * @param code_plane Bit-packed code plane (ignored for code-in-payload formats).
     * @param superblock_plane Superblock words (2 or 4 bytes each).
     * @param primary,secondary,emin Outputs of rows * blocks_per_row words (may be null).
     */
    inline void decodeCodedMatrix(std::uint8_t codebook, std::size_t rows, std::uint32_t blocks_per_row,
                                  std::uint32_t phase, std::uint32_t bit_shift, std::uint32_t payload_bytes,
                                  std::uint8_t *payload, const std::uint8_t *code_plane,
                                  const void *superblock_plane, std::uint16_t *primary, std::uint16_t *secondary,
                                  std::uint32_t *emin) noexcept
    {
        const bool in_payload = superblockCodeInPayload(codebook);
        const bool wide = superblockFieldBytes(codebook) == 4;
        for (std::uint32_t blk = 0; blk < blocks_per_row; ++blk)
            for (std::size_t row = 0; row < rows; ++row)
            {
                const auto address = blockMajorAddress(blk, static_cast<std::uint32_t>(row),
                                                       static_cast<std::uint32_t>(rows), phase);
                std::uint8_t *block = payload + address.linear * payload_bytes;
                const std::uint32_t code = in_payload ? loadCode(codebook, block, 0)
                                                      : loadCode(codebook, code_plane, address.linear, bit_shift);
                const std::uint32_t word = wide ? static_cast<const std::uint32_t *>(superblock_plane)[address.superblock]
                                                : static_cast<const std::uint16_t *>(superblock_plane)[address.superblock];
                const auto words = decodeSuperblockMetadata(codebook, code, word);
                if (primary) primary[address.linear] = words.primary;
                if (secondary) secondary[address.linear] = words.secondary;
                if (emin) emin[address.linear] = words.emin;
                if (in_payload)
                    restoreCodedPayload(codebook, block);
            }
    }
} // namespace llaminar2::native_quantised_superblock

#undef LLAMINAR_SUPERBLOCK_META_HD
