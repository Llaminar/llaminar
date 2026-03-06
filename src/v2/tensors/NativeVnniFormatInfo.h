#ifndef LLAMINAR2_TENSORS_NATIVEVNNIFORMATINFO_H
#define LLAMINAR2_TENSORS_NATIVEVNNIFORMATINFO_H

/**
 * @file NativeVnniFormatInfo.h
 * @brief Intrinsic format metadata for native-VNNI quantization formats
 *
 * Describes encoding properties that are invariant for each quantization format:
 * payload size, symmetry, super-block layout, and magnitude bounds. These are
 * returned by IINT8Unpackable::vnniFormatInfo() on each tensor class, replacing
 * the external dispatch table.
 *
 * Codebook IDs match the NativeVNNIFormat enum in ROCmGemvKernel_native_VNNI.hip.
 */

#pragma once

#include <cstdint>

namespace llaminar2
{
    /**
     * @brief Metadata descriptor for a native-VNNI quantization format
     *
     * Each native-VNNI format (≤6-bit) has fixed metadata that determines:
     * - How the GEMV/GEMM GPU kernel decodes the payload (codebook_id)
     * - How many bytes each block's payload occupies (payload_bytes)
     * - Whether min-value correction is needed (is_asymmetric)
     * - Whether 256→8×32 sub-block decomposition is needed (is_superblock)
     * - Conservative element magnitude for INT8 re-quantization (max_abs_factor)
     */
    struct NativeVnniFormatInfo
    {
        uint8_t codebook_id;  ///< Kernel dispatch ID (0=Q4_0, 4=IQ4_NL, 5=Q4_1, ...)
        int payload_bytes;    ///< Bytes per 32-element block payload
        bool is_asymmetric;   ///< True if format has min-value offset
        bool is_superblock;   ///< True if format has 256-element super-blocks (8×32)
        bool has_emins;       ///< True if format needs separate embedded mins (Q2_K only)
        float max_abs_factor; ///< Conservative max |element| / scale
    };

} // namespace llaminar2

#endif // LLAMINAR2_TENSORS_NATIVEVNNIFORMATINFO_H
