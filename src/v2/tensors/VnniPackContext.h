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
#include <stdexcept>

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

        /// First source K block represented in these destination arrays.
        /// Zero keeps whole-matrix packing unchanged. A nonzero origin lets
        /// streaming preparation reuse a bounded tile without rebasing source
        /// coordinates or forming pointers before the start of an allocation.
        int destination_block_origin = 0;
        /// Number of blocks in the destination window; zero means the complete
        /// remaining source range. Positive counts bound streaming scratch.
        int destination_block_count = 0;
    };

    // =================================================================
    // Inline helpers for VNNI interleaved layout
    // =================================================================

    /**
     * @brief Locate a source block in a complete or windowed destination.
     * @param ctx Owned destination geometry and source-coordinate window.
     * @param n Destination row, independent of the source tensor row.
     * @param b Absolute source K-block index.
     * @return Window-local interleaved index.
     * @throws std::out_of_range If the coordinates precede the window or
     *         exceed the declared source/row geometry.
     */
    inline size_t vnniLinearIdx(const VnniPackContext &ctx, int n, int b)
    {
        if (ctx.destination_block_origin < 0 || ctx.destination_block_count < 0 ||
            b < ctx.destination_block_origin || b >= ctx.blocks_per_row ||
            (ctx.destination_block_count != 0 &&
             b - ctx.destination_block_origin >= ctx.destination_block_count) ||
            n < 0 || n >= ctx.N)
            throw std::out_of_range("NativeVNNI packing coordinates outside destination window");
        return static_cast<size_t>(b - ctx.destination_block_origin) * ctx.N +
               static_cast<size_t>(n);
    }

    /** @brief Address a validated window-local block in the payload array. */
    inline uint8_t *vnniPayloadDst(const VnniPackContext &ctx, size_t linear)
    {
        return ctx.payload_array + linear * ctx.payload_bytes;
    }

    /** @brief Number of original 256-value source blocks, including a tail. */
    inline int vnniSuperBlocksPerRow(int K) { return (K + 255) / 256; }

} // namespace llaminar2

#endif // LLAMINAR2_TENSORS_VNNIPACKCONTEXT_H
