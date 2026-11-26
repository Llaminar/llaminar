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
    enum class CPUNativeQuantisedEncoding : std::uint8_t
    {
        NibbleLUT = 0,          ///< Four-bit payload decoded in the hot loop.
        ExpandedInt8 = 1,       ///< One prepared signed byte per logical weight.
        Q6KNativeDualScale = 2, ///< Six-bit payload, original half base, two int8 coefficients.
        CompactMultiScale = 3, ///< Native payload, two scales, optional two minima.
        NibbleLUTSourceScale = 4, ///< IQ4_XS / IQ3_S nibble payload, original half base, raw coefficient.
        CompactFiveBit = 6, ///< Five-bit low-nibble + high-mask planes, scales, optional minima (Q5_0/Q5_1/Q5_K).
        UnsignedNibble = 7, ///< Four-bit unsigned nibble payload, scales, optional minima, zero inline comp (Q4_0/Q4_1/Q4_K).
        UnsignedInt8 = 8, ///< Eight-bit unsigned (w_i8 ^ 0x80) payload, scales, optional minima, zero inline comp (Q8_0/Q8_1/Q8_K).
    };

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
        case CPUNativeQuantisedEncoding::ExpandedInt8:
        case CPUNativeQuantisedEncoding::Q6KNativeDualScale:
        case CPUNativeQuantisedEncoding::CompactMultiScale:
        case CPUNativeQuantisedEncoding::CompactFiveBit:
        case CPUNativeQuantisedEncoding::UnsignedNibble:
        case CPUNativeQuantisedEncoding::UnsignedInt8:
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
    [[nodiscard]] constexpr std::uint32_t preparedDataStride(
        CPUNativeQuantisedEncoding encoding) noexcept
    {
        switch (encoding)
        {
        case CPUNativeQuantisedEncoding::NibbleLUT:
        case CPUNativeQuantisedEncoding::NibbleLUTSourceScale:
        case CPUNativeQuantisedEncoding::CompactMultiScale:
        case CPUNativeQuantisedEncoding::UnsignedNibble:
            return 1024;
        case CPUNativeQuantisedEncoding::ExpandedInt8:
        case CPUNativeQuantisedEncoding::UnsignedInt8:
            return 2048;
        case CPUNativeQuantisedEncoding::Q6KNativeDualScale:
            return 1536;
        case CPUNativeQuantisedEncoding::CompactFiveBit:
            return 1280;
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
        bool is_asymmetric) noexcept
    {
        const std::uint32_t data_stride = preparedDataStride(encoding);
        if (data_stride == 0)
            return 0;
        if (encoding == CPUNativeQuantisedEncoding::NibbleLUTSourceScale)
            return 1408;
        if (encoding == CPUNativeQuantisedEncoding::Q6KNativeDualScale)
            return 1792;
        // A fixed padded native payload plus two scale vectors and one optional
        // effective-minimum vector. Unused bytes are zero, not another ledger.
        if (encoding == CPUNativeQuantisedEncoding::CompactMultiScale)
            return 1536;
        if (encoding == CPUNativeQuantisedEncoding::CompactFiveBit ||
            encoding == CPUNativeQuantisedEncoding::UnsignedNibble ||
            encoding == CPUNativeQuantisedEncoding::UnsignedInt8)
            return data_stride + 128u + (is_asymmetric ? 128u : 0u);
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
