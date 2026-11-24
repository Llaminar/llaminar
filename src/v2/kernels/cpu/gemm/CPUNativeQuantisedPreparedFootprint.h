/**
 * @file CPUNativeQuantisedPreparedFootprint.h
 * @brief Exact cache footprint contract for prepared CPU NativeQuantised matrices.
 *
 * NativeQuantised kernels consume a prepared representation, not the source GGUF
 * payload.  Cache policy therefore must be expressed in bytes of the prepared
 * interleaved stream and its live activation/output panels. A typed footprint
 * distinguishes each prepared unit from the original source block geometry.
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include "tensors/NativeQuantisedFormatInfo.h"
#include "tensors/NativeQuantisedSuperblockMetadata.h"

namespace llaminar2::cpu::native_quantised
{

    /**
     * @brief Physical encoding consumed by CPU NativeQuantised decode kernels.
     *
     * Source GGUF codebooks and prepared CPU encodings are deliberately
     * separate concepts. Several source formats share one prepared encoding;
     * the three Q8 formats retain one signed byte per logical weight.
     * Cache policy belongs to this execution encoding because it determines
     * the bytes and decode work seen by the hot kernel.
     */
    /**
     * @brief Interpretation of the primary scale field in CPU units.
     *
     * MXFP4 is the only prepared format with exponent-byte scales.
     */
    enum class CPUNativeQuantisedScaleEncoding : std::uint8_t
    {
        FP16 = 0,          ///< IEEE binary16 scale.
        E8M0 = 1,          ///< Native MXFP4 exponent byte; 0 is the prepared zero marker.
    };

    enum class CPUNativeQuantisedEncoding : std::uint8_t
    {
        NibbleLUT = 0,          ///< Four-bit payload decoded in the hot loop.
        SignedInt8SingleScale = 1,       ///< One prepared signed byte per logical weight.
        Q6KNativeDualScale = 2, ///< Six-bit payload, original half base, two int8 coefficients.
        CompactMultiScale = 3, ///< Native payload, two scales, optional two minima.
        NibbleLUTSourceScale = 4, ///< IQ4_XS nibble payload, original half base, raw coefficient.
        // Value 5 is reserved for the rejected historical IQ3_S layout.
        Q4KSuperblockCoded = 6, ///< Unsigned nibble payload, 12-bit code and a superblock d/dmin word.
        Q5NativeSingleScale = 7, ///< Q5_0/Q5_1 low nibbles and fifth bits, with FP16 d and optional m.
    };

    /// Reserved packed-weight ID; never a valid prepared CPU encoding.
    inline constexpr std::uint8_t kRetiredIQ3SExpandedSourceScaleEncoding = 5;
    /// The one rejection message for that ID at every receive and publication boundary.
    inline constexpr const char *kRetiredIQ3SExpandedSourceScaleDiagnostic =
        "CPU prepared encoding 5 (IQ3SExpandedSourceScale) is retired; IQ3_S requires CompactMultiScale";

    /** @return Whether a sender- or owner-supplied prepared encoding names the retired ID 5. */
    [[nodiscard]] constexpr bool isRetiredCPUPreparedEncoding(CPUNativeQuantisedEncoding encoding) noexcept
    {
        return static_cast<std::uint8_t>(encoding) == kRetiredIQ3SExpandedSourceScaleEncoding;
    }

    /** @return Whether the original source uses the ordinary Q5 block, not Q5_K. */
    [[nodiscard]] constexpr bool isQ5NativeSingleScaleSource(std::uint8_t codebook, bool superblock) noexcept
    {
        return !superblock && (codebook == 6 || codebook == 7);
    }

    /** @brief Q5's 16 nibble bytes plus four paired fifth-bit bytes per padded column. */
    struct Q5NativePayload
    {
        static constexpr int groups = 4;
        static constexpr int bytes = 64 * 20;

        /** @return Offset of four original nibble bytes for one column and K group. */
        static constexpr int nibbleOffset(int group, int column) noexcept { return group * 256 + column * 4; }
        /** @return Offset of the fifth bits of groups @p group and @p group + 4 in one byte. */
        static constexpr int fifthOffset(int group, int column) noexcept { return 1024 + column * 4 + group; }
    };

    /**
     * @brief Scale-word encoding of a prepared CPU unit, derived from its physical identity.
     * @param codebook_id Source codebook retained by the prepared matrix.
     * @param encoding Physical prepared encoding.
     * @return E8M0 for native MXFP4; FP16 for every other prepared format.
     *
     * Derived rather than stored, so no construction or import path can leave
     * it stale; archives and wire records seal it through their version.
     */
    [[nodiscard]] constexpr CPUNativeQuantisedScaleEncoding preparedScaleEncoding(
        std::uint8_t codebook_id, CPUNativeQuantisedEncoding encoding) noexcept
    {
        return codebook_id == 25 && encoding == CPUNativeQuantisedEncoding::NibbleLUT
                   ? CPUNativeQuantisedScaleEncoding::E8M0
                   : CPUNativeQuantisedScaleEncoding::FP16;
    }

    /**
     * @brief Return whether an encoding value names a supported execution path.
     *
     * Keeping this validation beside the enum lets footprint construction fail
     * closed if an invalid serialized or test-injected value reaches policy
     * selection.
     */
    [[nodiscard]] constexpr bool isKnownPreparedEncoding(
        CPUNativeQuantisedEncoding encoding) noexcept
    {
        switch (encoding)
        {
        case CPUNativeQuantisedEncoding::Q4KSuperblockCoded:
        case CPUNativeQuantisedEncoding::NibbleLUT:
        case CPUNativeQuantisedEncoding::NibbleLUTSourceScale:
        case CPUNativeQuantisedEncoding::SignedInt8SingleScale:
        case CPUNativeQuantisedEncoding::Q6KNativeDualScale:
        case CPUNativeQuantisedEncoding::CompactMultiScale:
        case CPUNativeQuantisedEncoding::Q5NativeSingleScale:
            return true;
        }
        return false;
    }

    /**
     * @brief Source payload bytes per 32-value block of a compact multi-scale codebook.
     *
     * Compact units are superblock-coded: they keep exactly these payload bytes
     * (with the code embedded for IQ3_XXS/IQ2_XS/IQ2_XXS/IQ1_S, the GPU's
     * coded payload) and the code planes below, so no byte beyond the source
     * file's information is stored.
     * @return 0 for codebooks outside the compact family.
     */
    [[nodiscard]] constexpr std::uint32_t preparedCompactPayloadBytes(int codebook_id) noexcept
    {
        switch (codebook_id)
        {
        case 7: return 20;   // Q5_K: 16 low-nibble bytes + 4 B/column in group-major bitplanes.
        case 9: case 12: return 12;  // Q3_K, IQ3_XXS
        case 13: case 14: return 9;  // IQ2_S, IQ2_XS
        case 10: case 15: return 8;  // Q2_K, IQ2_XXS
        case 16: case 17: return 6;  // IQ1_S, IQ1_M
        case 11: return 13;  // IQ3_S: eight indices, one high-bit byte and four sign bytes.
        default: return 0;
        }
    }

    /**
     * @brief Column-interleaved payload geometry of a compact block.
     *
     * Whole 32-bit payload words are stored as `[word][64 columns]` (256 bytes
     * per word, as before); the remaining 1 or 2 bytes form one narrower
     * plane `[64 columns][tail]` that loads zero-extended into the same lane.
     * Q5_K has four whole low-nibble words; its remaining 256 bytes are eight
     * 32-byte high-bit planes, not a fifth word or a column-tail plane.
     */
    [[nodiscard]] constexpr std::uint32_t preparedCompactPayloadWords(int codebook_id) noexcept
    {
        return codebook_id == 7 ? 4u : preparedCompactPayloadBytes(codebook_id) / 4u;
    }
    /** @return Bytes per column of the narrower payload tail plane (0, 1 or 2). */
    [[nodiscard]] constexpr std::uint32_t preparedCompactPayloadTailBytes(int codebook_id) noexcept
    {
        return preparedCompactPayloadBytes(codebook_id) % 4u;
    }
    /** @return Bits per column of a compact block's separate code (0 when the code is in the payload). */
    [[nodiscard]] constexpr std::uint32_t preparedCompactCodeBits(int codebook_id) noexcept
    {
        return native_quantised_superblock::superblockCodePlaneBits(static_cast<std::uint8_t>(codebook_id));
    }

#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_CPU_BLOCK_GEOMETRY_HD __host__ __device__
#else
#define LLAMINAR_CPU_BLOCK_GEOMETRY_HD
#endif
    /**
     * @brief Physical superblock-coded metadata codebook of a prepared encoding.
     *
     * Most encodings use their source codebook directly as the codec's physical
     * codebook (code plane width and `d`/`dmin` field bytes). IQ4_XS source-scale
     * units are the exception: the source codebook is 4 (IQ4_NL payload), but the
     * superblock codec's six-bit `ls` and single `d` fields use the IQ4_XS
     * source-scale physical codebook 24 (kNativeQuantisedIQ4XSSourceScaleCodebook).
     * @param encoding Physical prepared encoding.
     * @param codebook_id Source codebook retained by the prepared matrix.
     * @return Codebook passed to NativeQuantisedSuperblockMetadata for this unit.
     */
    [[nodiscard]] constexpr std::uint8_t superblockMetadataCodebook(
        CPUNativeQuantisedEncoding encoding, int codebook_id) noexcept
    {
        return encoding == CPUNativeQuantisedEncoding::NibbleLUTSourceScale
                   ? kNativeQuantisedIQ4XSSourceScaleCodebook
                   : static_cast<std::uint8_t>(codebook_id);
    }

    /** @brief Offset of the first code plane in a superblock-coded CPU unit.
     * @param data_stride Bytes of payload per 64-column unit.
     * @return Byte offset relative to the admitted block; no column sums precede it.
     */
    [[nodiscard]] constexpr std::uint32_t preparedCodedPlaneOffset(
        CPUNativeQuantisedEncoding, std::uint32_t data_stride) noexcept
    {
        return data_stride;
    }

    /**
     * @brief Whether an encoding stores superblock-coded metadata.
     *
     * Coded units carry per-block codes and one superblock word per
     * (column, source superblock) instead of per-block FP16 words. Their
     * superblock header follows the last block of each source superblock;
     * see CPUNativeQuantisedBlockGeometry.
     */
    [[nodiscard]] LLAMINAR_CPU_BLOCK_GEOMETRY_HD constexpr bool preparedSuperblockCoded(CPUNativeQuantisedEncoding encoding) noexcept
    {
        return encoding == CPUNativeQuantisedEncoding::CompactMultiScale ||
               encoding == CPUNativeQuantisedEncoding::Q6KNativeDualScale ||
               encoding == CPUNativeQuantisedEncoding::NibbleLUTSourceScale ||
               encoding == CPUNativeQuantisedEncoding::Q4KSuperblockCoded;
    }

    /**
     * @brief Header bytes after each 64-column superblock unit (0 for per-block encodings).
     * @return 64 columns x the superblock word (`d`, or `d | dmin << 16` for Q2_K and Q5_K).
     */
    [[nodiscard]] constexpr std::uint32_t preparedSuperblockHeaderBytes(
        CPUNativeQuantisedEncoding encoding, int codebook_id) noexcept
    {
        return preparedSuperblockCoded(encoding)
            ? 64u * native_quantised_superblock::superblockFieldBytes(
                       superblockMetadataCodebook(encoding, codebook_id))
            : 0u;
    }

    /**
     * @brief Block and superblock-header addressing of one prepared (or wire) matrix.
     *
     * Per-block encodings (header_bytes == 0) keep `(chunk * blocks + block) *
     * block_stride`. Coded encodings store a chunk row's blocks contiguously
     * and place each source superblock's header (64 words for a resident
     * matrix, one word per live column on the wire) directly after the last
     * block of that superblock inside the row. `phase` is the position of the
     * row's first block inside its source superblock, so a row that starts or
     * ends mid-superblock still carries the complete words of every
     * superblock it touches (the same duplication the GPU coded planes use).
     * Resident CPU matrices are packed superblock-aligned (phase 0); wire
     * intersections may carry any phase.
     */
    struct CPUNativeQuantisedBlockGeometry
    {
        std::size_t block_stride = 0;   ///< Bytes of one block (all of its columns).
        std::size_t header_bytes = 0;   ///< Bytes of one superblock header (0 for per-block encodings).
        std::size_t blocks_per_row = 0; ///< K blocks per chunk row.
        std::size_t phase = 0;          ///< First block's position in its source superblock (0..7).

        /** @return Superblock headers per chunk row (0 for per-block encodings). */
        [[nodiscard]] LLAMINAR_CPU_BLOCK_GEOMETRY_HD constexpr std::size_t superblocksPerRow() const noexcept
        {
            return header_bytes ? (blocks_per_row + phase + 7) / 8 : 0;
        }
        /** @return Bytes of one complete chunk row. */
        [[nodiscard]] LLAMINAR_CPU_BLOCK_GEOMETRY_HD constexpr std::size_t rowBytes() const noexcept
        {
            return blocks_per_row * block_stride + superblocksPerRow() * header_bytes;
        }
        /** @return Offset of block @p block of chunk row @p chunk. */
        [[nodiscard]] LLAMINAR_CPU_BLOCK_GEOMETRY_HD constexpr std::size_t blockOffset(std::size_t chunk, std::size_t block) const noexcept
        {
            const std::size_t headers = header_bytes ? (block + phase) / 8 : 0;
            return chunk * rowBytes() + block * block_stride + headers * header_bytes;
        }
        /** @return Whether @p block is the last block of its superblock inside the row. */
        [[nodiscard]] LLAMINAR_CPU_BLOCK_GEOMETRY_HD constexpr bool closesSuperblock(std::size_t block) const noexcept
        {
            return header_bytes && (block + 1 == blocks_per_row || (block + phase) % 8 == 7);
        }
        /** @return Exclusive logical end of the source superblock serving @p block, including a ragged tail. */
        [[nodiscard]] constexpr std::size_t superblockEnd(std::size_t block) const noexcept
        {
            const std::size_t end = ((block + phase) / 8 + 1) * 8 - phase;
            return end < blocks_per_row ? end : blocks_per_row;
        }
        /** @return Offset of the header serving block @p block of chunk row @p chunk. */
        [[nodiscard]] LLAMINAR_CPU_BLOCK_GEOMETRY_HD constexpr std::size_t headerOffset(std::size_t chunk, std::size_t block) const noexcept
        {
            const std::size_t superblock = (block + phase) / 8;
            const std::size_t last_end = superblock * 8 + 8 - phase; // one past the superblock's last block
            const std::size_t end = last_end < blocks_per_row ? last_end : blocks_per_row;
            return chunk * rowBytes() + end * block_stride + superblock * header_bytes;
        }
        /** @return Bytes of @p chunks complete chunk rows. */
        [[nodiscard]] LLAMINAR_CPU_BLOCK_GEOMETRY_HD constexpr std::size_t matrixBytes(std::size_t chunks) const noexcept
        {
            return chunks * rowBytes();
        }
    };

#undef LLAMINAR_CPU_BLOCK_GEOMETRY_HD

    /**
     * @brief Return the pure weight-data bytes in one prepared CPU unit.
     *
     * A unit always spans 64 output columns and one 32-value K block. Keeping
     * this geometry beside the encoding enum gives CPU execution and
     * heterogeneous tier conversion one authoritative layout definition.
     */
    [[nodiscard]] constexpr std::uint32_t preparedDataStride(
        CPUNativeQuantisedEncoding encoding, int codebook_id) noexcept
    {
        switch (encoding)
        {
        case CPUNativeQuantisedEncoding::Q4KSuperblockCoded:
        case CPUNativeQuantisedEncoding::NibbleLUT:
        case CPUNativeQuantisedEncoding::NibbleLUTSourceScale:
            return 1024;
        case CPUNativeQuantisedEncoding::CompactMultiScale:
            return preparedCompactPayloadBytes(codebook_id) * 64u;
        case CPUNativeQuantisedEncoding::SignedInt8SingleScale:
            return 2048;
        case CPUNativeQuantisedEncoding::Q6KNativeDualScale:
            return 1536;
        case CPUNativeQuantisedEncoding::Q5NativeSingleScale:
            return Q5NativePayload::bytes;
        }
        return 0;
    }

    /** @return Whether the original source carries one signed Q8 byte per value. */
    [[nodiscard]] constexpr bool isSignedInt8SingleScaleSource(int codebook_id) noexcept
    {
        using namespace native_quantised_formats;
        return codebook_id == Q8_0.codebook_id || codebook_id == Q8_1.codebook_id || codebook_id == Q8_K.codebook_id;
    }

    /**
     * @brief Return all bytes in one prepared CPU unit, including metadata.
     * @param encoding Physical execution encoding.
     * @param is_asymmetric Whether one FP16 minimum is stored per column.
     */
    [[nodiscard]] constexpr std::uint32_t preparedInterleavedBlockStride(
        CPUNativeQuantisedEncoding encoding,
        bool is_asymmetric,
        int codebook_id) noexcept
    {
        const std::uint32_t data_stride = preparedDataStride(encoding, codebook_id);
        if (data_stride == 0)
            return 0;
        // Coded blocks hold the exact payload and the exact-width code plane
        // (64 columns x code bits) inside the unit; the superblock words follow
        // each eight-block unit (preparedSuperblockHeaderBytes).
        if (encoding == CPUNativeQuantisedEncoding::Q6KNativeDualScale ||
            encoding == CPUNativeQuantisedEncoding::NibbleLUTSourceScale)
            return preparedCodedPlaneOffset(encoding, data_stride) +
                   8u * native_quantised_superblock::superblockCodePlaneBits(superblockMetadataCodebook(encoding, codebook_id));
        if (preparedScaleEncoding(codebook_id, encoding) == CPUNativeQuantisedScaleEncoding::E8M0)
            return data_stride + 64u;
        if (encoding == CPUNativeQuantisedEncoding::Q4KSuperblockCoded)
            return data_stride + 8u * native_quantised_superblock::superblockCodePlaneBits(5);
        if (encoding == CPUNativeQuantisedEncoding::CompactMultiScale)
            return data_stride + 8u * preparedCompactCodeBits(codebook_id);
        return data_stride + 128u + (is_asymmetric ? 128u : 0u);
    }

    /**
     * @brief Addressing of a resident prepared matrix (superblock-aligned, phase 0).
     * @param blocks_per_row K blocks of every chunk row.
     */
    [[nodiscard]] constexpr CPUNativeQuantisedBlockGeometry preparedBlockGeometry(
        CPUNativeQuantisedEncoding encoding, bool is_asymmetric, int codebook_id, std::size_t blocks_per_row) noexcept
    {
        return CPUNativeQuantisedBlockGeometry{
            .block_stride = preparedInterleavedBlockStride(encoding, is_asymmetric, codebook_id),
            .header_bytes = preparedSuperblockHeaderBytes(encoding, codebook_id),
            .blocks_per_row = blocks_per_row,
            .phase = 0,
        };
    }

    /**
     * @brief Exact resident bytes touched by one logical NativeQuantised tile unit.
     *
     * A weight unit is one 64-column N chunk by one 32-element K block and
     * includes every inline datum consumed by the kernel: packed values,
     * scales and optional minima.  Activation and output units
     * describe one M row so a scheduler can account for the physical row tile
     * independently of the matrix's source codebook.
     */
    struct NativeQuantisedPreparedFootprint
    {
        CPUNativeQuantisedEncoding encoding =
            CPUNativeQuantisedEncoding::SignedInt8SingleScale;
        bool is_asymmetric = false;
        std::uint64_t weight_bytes_per_n_chunk_k_block = 0;
        std::uint64_t activation_bytes_per_row_k_block = 0;
        std::uint64_t output_bytes_per_row_n_chunk = 0;

        /** @brief Return whether all execution-footprint dimensions are valid. */
        [[nodiscard]] constexpr bool isValid() const noexcept
        {
            return isKnownPreparedEncoding(encoding) &&
                   weight_bytes_per_n_chunk_k_block > 0 &&
                   activation_bytes_per_row_k_block > 0 &&
                   output_bytes_per_row_n_chunk > 0;
        }
    };

} // namespace llaminar2::cpu::native_quantised
