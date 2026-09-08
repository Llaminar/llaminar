/**
 * @file CPUNativeVNNIPreparedFootprint.h
 * @brief Exact cache footprint contract for prepared CPU NativeVNNI matrices.
 *
 * NativeVNNI kernels consume a prepared representation, not the source GGUF
 * payload.  Cache policy therefore must be expressed in bytes of the prepared
 * interleaved stream and its live activation/output panels.  Keeping those
 * quantities in a typed value prevents a compact source block size from being
 * mistaken for the substantially different execution layout.
 */

#pragma once

#include <cstdint>

namespace llaminar2::cpu::native_vnni
{

    /**
     * @brief Physical encoding consumed by CPU NativeVNNI decode kernels.
     *
     * Source GGUF codebooks and prepared CPU encodings are deliberately
     * separate concepts. Several source formats share one prepared encoding,
     * while rotation converts every source format to expanded signed INT8.
     * Cache policy belongs to this execution encoding because it determines
     * the bytes and decode work seen by the hot kernel.
     */
    enum class CPUNativeVNNIEncoding : std::uint8_t
    {
        NibbleLUT = 0,          ///< Four-bit payload decoded in the hot loop.
        ExpandedInt8 = 1,       ///< One prepared signed byte per logical weight.
        Q6KNativeDualScale = 2, ///< Native six-bit payload with two FP16 scales.
    };

    /**
     * @brief Return whether an encoding value names a supported execution path.
     *
     * Keeping this validation beside the enum lets footprint construction fail
     * closed if an invalid serialized or test-injected value reaches policy
     * selection.
     */
    [[nodiscard]] constexpr bool isKnownPreparedEncoding(
        CPUNativeVNNIEncoding encoding) noexcept
    {
        switch (encoding)
        {
        case CPUNativeVNNIEncoding::NibbleLUT:
        case CPUNativeVNNIEncoding::ExpandedInt8:
        case CPUNativeVNNIEncoding::Q6KNativeDualScale:
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
        CPUNativeVNNIEncoding encoding) noexcept
    {
        switch (encoding)
        {
        case CPUNativeVNNIEncoding::NibbleLUT:
            return 1024;
        case CPUNativeVNNIEncoding::ExpandedInt8:
            return 2048;
        case CPUNativeVNNIEncoding::Q6KNativeDualScale:
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
        CPUNativeVNNIEncoding encoding,
        bool is_asymmetric) noexcept
    {
        const std::uint32_t data_stride = preparedDataStride(encoding);
        if (data_stride == 0)
            return 0;
        if (encoding == CPUNativeVNNIEncoding::Q6KNativeDualScale)
            return 1792;
        return data_stride + 256u + (is_asymmetric ? 128u : 0u);
    }

    /**
     * @brief Exact resident bytes touched by one logical NativeVNNI tile unit.
     *
     * A weight unit is one 64-column N chunk by one 32-element K block and
     * includes every inline datum consumed by the kernel: packed values,
     * compensation, scales, and optional minima.  Activation and output units
     * describe one M row so a scheduler can account for the physical row tile
     * independently of the matrix's source codebook.
     */
    struct NativeVNNIPreparedFootprint
    {
        CPUNativeVNNIEncoding encoding =
            CPUNativeVNNIEncoding::ExpandedInt8;
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

} // namespace llaminar2::cpu::native_vnni
