#ifndef LLAMINAR2_TENSORS_VNNIPACKCONTEXT_H
#define LLAMINAR2_TENSORS_VNNIPACKCONTEXT_H

/**
 * @file VnniPackContext.h
 * @brief Shared packing context for native-VNNI block interleaving
 *
 * Contains the output buffers and layout parameters needed by
 * IINT8Unpackable::packVnniBlock() to write one 32-element block
 * into the interleaved payload/scale/min arrays consumed by
 * ROCm GEMV/GEMM GPU kernels.
 *
 * The helpers (linearIdx, payloadDst, superBlocksPerRow) encode
 * the *interleaved-by-N* memory layout used for coalesced GPU access.
 */

#pragma once

#include <cstdint>
#include <cstddef>

namespace llaminar2
{

    /**
     * @brief Shared state for the block-level VNNI packing loop
     *
     * Holds all output buffers and layout parameters needed by the
     * per-format packVnniBlock() implementations.  Constructed once
     * by packNativeVNNI() and passed to every block packing call.
     */
    struct VnniPackContext
    {
        const uint8_t *raw_bytes; ///< Tensor raw data (from tensor->raw_data())
        int N;                    ///< Output features (rows)
        int K;                    ///< Input features (columns)
        int blocks_per_row;       ///< K / 32
        int payload_bytes;        ///< Bytes per 32-element block payload

        // Output arrays (pre-allocated by packNativeVNNI)
        uint8_t *payload_array; ///< [blocks_per_row × N × payload_bytes]
        uint16_t *scales_array; ///< [blocks_per_row × N]
        uint16_t *mins_array;   ///< [blocks_per_row × N] (nullptr for symmetric)
        uint32_t *emins_array;  ///< [blocks_per_row × N] (nullptr except Q2_K)
    };

    // =================================================================
    // Inline helpers for VNNI interleaved layout
    // =================================================================

    /// Compute the interleaved linear index for coalesced GPU access
    inline size_t vnniLinearIdx(const VnniPackContext &ctx, int n, int b)
    {
        return static_cast<size_t>(b) * ctx.N + static_cast<size_t>(n);
    }

    /// Get the payload destination pointer for a given linear index
    inline uint8_t *vnniPayloadDst(const VnniPackContext &ctx, size_t linear)
    {
        return ctx.payload_array + linear * ctx.payload_bytes;
    }

    /// Compute the number of 256-element super-blocks per row
    inline int vnniSuperBlocksPerRow(int K) { return (K + 255) / 256; }

} // namespace llaminar2

#endif // LLAMINAR2_TENSORS_VNNIPACKCONTEXT_H
