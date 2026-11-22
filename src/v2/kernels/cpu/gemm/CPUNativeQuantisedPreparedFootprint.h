/**
 * @file CPUNativeQuantisedPreparedFootprint.h
 * @brief Exact cache footprint contract for prepared CPU NativeQuantised matrices.
 *
 * NativeQuantised kernels consume a prepared representation, not the source GGUF
 * payload.  Cache policy therefore must be expressed in bytes of the prepared
 * interleaved stream and its live activation/output panels. IQ3_S source-scale
 * encoding retains decoded grid bytes and adds one raw coefficient vector.  Keeping those
 * quantities in a typed value prevents a compact source block size from being
 * mistaken for the substantially different execution layout.
 */

#pragma once

#include <cstdint>

namespace llaminar2::cpu::native_quantised
{

    /**
     * @brief Physical encoding consumed by CPU NativeQuantised decode kernels.
     *
     * Source GGUF codebooks and prepared CPU encodings are deliberately
     * separate concepts. Several source formats share one prepared encoding,
     * while rotation converts every source format to expanded signed INT8.
     * Cache policy belongs to this execution encoding because it determines
     * the bytes and decode work seen by the hot kernel.
     */
    /**
     * @brief Interpretation of the 16-bit primary scale word in CPU units.
     *
     * Chosen from the prepared physical representation, never from the source
     * codebook alone: rotated (expanded INT8) MXFP4 keeps source codebook 25 but
     * carries ordinary FP16 scales.
     */
    enum class CPUNativeQuantisedScaleEncoding : std::uint8_t
    {
        FP16 = 0,           ///< IEEE binary16 scale (every format today).
        PowerOfTwoBF16 = 1, ///< Native MXFP4 E8M0 as mxfp4::powerOfTwoBF16Word.
    };

    enum class CPUNativeQuantisedEncoding : std::uint8_t
    {
        NibbleLUT = 0,          ///< Four-bit payload decoded in the hot loop.
        ExpandedInt8 = 1,       ///< One prepared signed byte per logical weight.
        Q6KNativeDualScale = 2, ///< Six-bit payload, original half base, two int8 coefficients.
        CompactMultiScale = 3, ///< Native payload, two scales, optional two minima.
        NibbleLUTSourceScale = 4, ///< IQ4_XS nibble payload, original half base, raw coefficient.
        IQ3SExpandedSourceScale = 5, ///< Original IQ3_S signed grid, base and raw nibble.
    };

    /**
     * @brief Scale-word encoding of a prepared CPU unit, derived from its physical identity.
     * @param codebook_id Source codebook retained by the prepared matrix.
     * @param encoding Physical prepared encoding.
     * @return PowerOfTwoBF16 only for MXFP4 in its native nibble encoding;
     *         rotated (expanded INT8) MXFP4 and every other format carry FP16.
     *
     * Derived rather than stored, so no construction or import path can leave
     * it stale; archives and wire records seal it through their version.
     */
    [[nodiscard]] constexpr CPUNativeQuantisedScaleEncoding preparedScaleEncoding(
        std::uint8_t codebook_id, CPUNativeQuantisedEncoding encoding) noexcept
    {
        return codebook_id == 25 && encoding == CPUNativeQuantisedEncoding::NibbleLUT
                   ? CPUNativeQuantisedScaleEncoding::PowerOfTwoBF16
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
        case CPUNativeQuantisedEncoding::NibbleLUT:
        case CPUNativeQuantisedEncoding::NibbleLUTSourceScale:
        case CPUNativeQuantisedEncoding::IQ3SExpandedSourceScale:
        case CPUNativeQuantisedEncoding::ExpandedInt8:
        case CPUNativeQuantisedEncoding::Q6KNativeDualScale:
        case CPUNativeQuantisedEncoding::CompactMultiScale:
            return true;
        }
        return false;
    }

    /**
     * @brief Return the pure weight-data bytes in one prepared CPU unit.
     *
     * A unit always spans 64 output columns and one 32-value K block. Keeping
     * this geometry beside the encoding enum gives CPU execution and
     * heterogeneous tier conversion one authoritative layout definition.
     */
    [[nodiscard]] constexpr std::uint32_t preparedCompactPayloadWords(
        int codebook_id) noexcept
    {
        // The compact interleaved unit places one 32-bit word per payload
        // byte-quad, 256 bytes apart (64 columns x 4 bytes). Only the words
        // the decode actually reads are retained: three words cover the
        // 9-12 byte payloads (Q3_K/IQ2_S/IQ2_XS/IQ3_XXS), two cover the
        // 6-8 byte payloads (Q2_K/IQ2_XXS/IQ1_S/IQ1_M).
        switch (codebook_id)
        {
        case 9: case 12: case 13: case 14: return 3;
        default: return 2;
        }
    }

    [[nodiscard]] constexpr std::uint32_t preparedDataStride(
        CPUNativeQuantisedEncoding encoding, int codebook_id = 0) noexcept
    {
        switch (encoding)
        {
        case CPUNativeQuantisedEncoding::NibbleLUT:
        case CPUNativeQuantisedEncoding::NibbleLUTSourceScale:
            return 1024;
        case CPUNativeQuantisedEncoding::CompactMultiScale:
            return preparedCompactPayloadWords(codebook_id) * 256u;
        case CPUNativeQuantisedEncoding::ExpandedInt8:
        case CPUNativeQuantisedEncoding::IQ3SExpandedSourceScale:
            return 2048;
        case CPUNativeQuantisedEncoding::Q6KNativeDualScale:
            return 1536;
        }
        return 0;
    }

    /**
     * @brief Return all bytes in one prepared CPU unit, including metadata.
     * @param encoding Physical execution encoding.
     * @param is_asymmetric Whether one FP16 minimum is stored per column.
     */
    [[nodiscard]] constexpr std::uint32_t preparedInterleavedBlockStride(
        CPUNativeQuantisedEncoding encoding,
        bool is_asymmetric,
        int codebook_id = 0) noexcept
    {
        const std::uint32_t data_stride = preparedDataStride(encoding, codebook_id);
        if (data_stride == 0)
            return 0;
        if (encoding == CPUNativeQuantisedEncoding::NibbleLUTSourceScale)
            return 1408;
        if (encoding == CPUNativeQuantisedEncoding::IQ3SExpandedSourceScale)
            return 2432;
        if (encoding == CPUNativeQuantisedEncoding::Q6KNativeDualScale)
            return 1792;
        // The compact family reserves exactly the payload words each codebook
        // decodes, plus two FP16 scale vectors and the effective-minimum
        // vector. Unused bytes are zero, not another ledger.
        if (encoding == CPUNativeQuantisedEncoding::CompactMultiScale)
            return data_stride + 512u;
        return data_stride + 256u + (is_asymmetric ? 128u : 0u);
    }

    /**
     * @brief Exact resident bytes touched by one logical NativeQuantised tile unit.
     *
     * A weight unit is one 64-column N chunk by one 32-element K block and
     * includes every inline datum consumed by the kernel: packed values,
     * compensation, scales, and optional minima.  Activation and output units
     * describe one M row so a scheduler can account for the physical row tile
     * independently of the matrix's source codebook.
     */
    struct NativeQuantisedPreparedFootprint
    {
        CPUNativeQuantisedEncoding encoding =
            CPUNativeQuantisedEncoding::ExpandedInt8;
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
