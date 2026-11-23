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
 *    (superblockCodeBytes() bytes);
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
#elif defined(__CUDACC__)
#include <cuda_runtime.h>
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
     * @brief Per-32 code-plane width of a superblock-coded execution codebook.
     * @param codebook Physical execution codebook.
     * @return 1 or 2 bytes for superblock-coded codebooks, 0 otherwise.
     */
    LLAMINAR_SUPERBLOCK_META_HD constexpr std::uint32_t superblockCodeBytes(std::uint8_t codebook) noexcept
    {
        switch (codebook)
        {
        case 24: // IQ4_XS source-scale physical codebook (6-bit ls)
        case 11: // IQ3_S (4-bit nibble)
        case 12: // IQ3_XXS (4-bit nibble)
        case 13: // IQ2_S (two 4-bit nibbles)
        case 14: // IQ2_XS (two 4-bit nibbles)
        case 15: // IQ2_XXS (4-bit nibble)
        case 16: // IQ1_S (3-bit scale + delta sign)
        case 17: // IQ1_M (two 3-bit scales)
            return 1;
        case 8:  // Q6_K (two int8 coefficients)
        case 9:  // Q3_K (two 6-bit scales)
        case 10: // Q2_K (two scale/min bytes)
            return 2;
        default:
            return 0;
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
        return codebook == 10;
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
                out.primary = canonicalPreparedHalfBits(d * (0.5f + static_cast<float>(code & 15u)) * 0.25f);
            }
            else if constexpr (Codebook == 14)
            {
                out.primary = canonicalPreparedHalfBits(d * (0.5f + static_cast<float>(code & 15u)) * 0.25f);
                out.secondary = canonicalPreparedHalfBits(d * (0.5f + static_cast<float>((code >> 4) & 15u)) * 0.25f);
            }
            else if constexpr (Codebook == 12)
            {
                out.primary = canonicalPreparedHalfBits(d * (0.5f + static_cast<float>(code & 15u)) * 0.5f);
            }
            else if constexpr (Codebook == 16)
            {
                // code bits 0..2: scale selector; bit 3: delta sign (qh bit 15).
                const float dl = d * (2.0f * static_cast<float>(code & 7u) + 1.0f);
                const float delta = (code & 8u) ? -0.125f : 0.125f;
                out.primary = canonicalPreparedHalfBits(dl);
                out.secondary = canonicalPreparedHalfBits(dl * delta);
            }
            else if constexpr (Codebook == 17)
            {
                // code bits 0..2: low 3-bit scale; bits 3..5: high 3-bit scale.
                out.primary = canonicalPreparedHalfBits(d * (2.0f * static_cast<float>(code & 7u) + 1.0f));
                out.secondary = canonicalPreparedHalfBits(d * (2.0f * static_cast<float>((code >> 3) & 7u) + 1.0f));
            }
            else if constexpr (Codebook == 9)
            {
                // code: two unpacked 6-bit scales (low, high); stored biased by 32.
                out.primary = canonicalPreparedHalfBits(d * static_cast<float>(static_cast<int>(code & 63u) - 32));
                out.secondary =
                    canonicalPreparedHalfBits(d * static_cast<float>(static_cast<int>((code >> 6) & 63u) - 32));
            }
            else if constexpr (Codebook == 10)
            {
                // code: original scale bytes of the two 16-value halves (low, high).
                const float dmin = halfBitsToFloat(static_cast<std::uint16_t>(superblock >> 16));
                const std::uint32_t lo = code & 0xFFu, hi = (code >> 8) & 0xFFu;
                out.primary = canonicalPreparedHalfBits(d * static_cast<float>(lo & 0xFu));
                out.secondary = canonicalPreparedHalfBits(d * static_cast<float>(hi & 0xFu));
                const std::uint16_t emin_lo = canonicalPreparedHalfBits(-dmin * static_cast<float>(lo >> 4));
                const std::uint16_t emin_hi = canonicalPreparedHalfBits(-dmin * static_cast<float>(hi >> 4));
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
        return {rows * blocks_per_row * superblockCodeBytes(codebook),
                rows * superblocksPerRow(blocks_per_row, phase) * superblockFieldBytes(codebook)};
    }

    /**
     * @brief Load one block's code from the code plane.
     * @tparam Codebook Physical execution codebook.
     */
    template <std::uint8_t Codebook>
    LLAMINAR_SUPERBLOCK_META_HD std::uint32_t loadCode(const void *code_plane, std::size_t linear) noexcept
    {
        if constexpr (superblockCodeBytes(Codebook) == 1)
            return static_cast<const std::uint8_t *>(code_plane)[linear];
        else
            return static_cast<const std::uint16_t *>(code_plane)[linear];
    }

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
                                                                       BlockAddress address) noexcept
    {
        return decodeSuperblockMetadata<Codebook>(loadCode<Codebook>(code_plane, address.linear),
                                                  loadSuperblock<Codebook>(superblock_plane, address.superblock));
    }
} // namespace llaminar2::native_quantised_superblock

#undef LLAMINAR_SUPERBLOCK_META_HD
