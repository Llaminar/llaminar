#pragma once

/**
 * @file VnniRepackKernels.h
 * @brief Host-side declarations for ROCm/HIP VNNI repack kernels
 *
 * Provides GPU kernels that repack raw GGUF quantized blocks into
 * VNNI-interleaved layout directly on device, eliminating the need
 * for CPU-side packing during weight loading.
 *
 * The VNNI layout interleaves blocks by output feature (N dimension):
 *   payload[b * N + n]  — 16 bytes of repacked nibbles
 *   scales[b * N + n]   — Q6/IQ2_S/IQ4_XS original half d, otherwise FP16 scale
 *   mins[b * N + n]     — Q6 signed pair, IQ2_S raw nibble byte, IQ4_XS encoded6;
 *                        otherwise format-specific auxiliary metadata
 *
 * Where b = block index within a row, n = output feature index.
 */

#include "loaders/gpu_pipeline/RepackFormat.h"  // RepackFormat enum (shared)

#include <cstdint>
#include <cstddef>

namespace llaminar2 {

/**
 * @brief Launch a GPU kernel to repack raw GGUF blocks into VNNI layout.
 *
 * All pointers must be device pointers. The kernel is launched asynchronously
 * on the exact non-null producer stream. Production consumers must wait on the
 * producer's published event instead of synchronizing the stream or device.
 *
 * @param format       Quantization format of the raw blocks
 * @param d_raw_blocks Raw GGUF block data on device (uploaded as-is from host)
 * @param d_payload    Output VNNI payload [blocks_per_row * N * payload_bytes]
 * @param d_scales     Output metadata [blocks_per_row * N]: original half d for
 *                     Q6/IQ2_S/IQ4_XS, otherwise ordinary binary16 scales
 * @param d_mins       Raw Q6 signed pair, IQ2_S nibble byte, IQ4_XS encoded6, or
 *                     format-specific half auxiliary; never half-convert source words;
 *                     nullptr only when the selected format has no secondary metadata
 * @param N            Number of output features (rows in weight matrix)
 * @param K            Number of input features (columns in weight matrix)
 * @param output_N     Row stride of the complete packed destination
 * @param output_row_offset First destination row represented by this source chunk
 * @param packed_group_rows Rows per independently addressable packed matrix;
 *        zero selects one ordinary output_N-row matrix
 * @param allocation_payload_bytes_per_block Physical payload capacity for one
 *        logical block; may exceed the compact live payload for recyclable
 *        ExpertOverlay slots
 * @param stream       Exact non-null HIP producer stream
 * @return true on successful kernel launch, false on error or unsupported format
 */
bool launchVnniRepack(
    RepackFormat format,
    const void* d_raw_blocks,
    uint8_t* d_payload,
    uint16_t* d_scales,
    uint16_t* d_mins,
    int N, int K,
    void* stream);

/**
 * @brief Extended overload accepting d_emins for Q2_K format.
 *
 * @param d_emins      Output VNNI effective mins (uint32_t), nullptr except for Q2_K
 * @see launchVnniRepack (7-param overload) for other parameters
 */
bool launchVnniRepack(
    RepackFormat format,
    const void* d_raw_blocks,
    uint8_t* d_payload,
    uint16_t* d_scales,
    uint16_t* d_mins,
    uint32_t* d_emins,
    int N, int K,
    int output_N,
    int output_row_offset,
    int packed_group_rows,
    int allocation_payload_bytes_per_block,
    void* stream);

/** @brief Compatibility overload using compact payload allocation geometry. */
bool launchVnniRepack(
    RepackFormat format,
    const void* d_raw_blocks,
    uint8_t* d_payload,
    uint16_t* d_scales,
    uint16_t* d_mins,
    uint32_t* d_emins,
    int N, int K,
    int output_N,
    int output_row_offset,
    int packed_group_rows,
    void* stream);

/**
 * @brief Row-chunk compatibility overload selecting one ordinary packed matrix.
 */
bool launchVnniRepack(
    RepackFormat format,
    const void* d_raw_blocks,
    uint8_t* d_payload,
    uint16_t* d_scales,
    uint16_t* d_mins,
    uint32_t* d_emins,
    int N, int K,
    int output_N,
    int output_row_offset,
    void* stream);

/**
 * @brief Full-matrix compatibility overload for callers without row chunks.
 */
bool launchVnniRepack(
    RepackFormat format,
    const void* d_raw_blocks,
    uint8_t* d_payload,
    uint16_t* d_scales,
    uint16_t* d_mins,
    uint32_t* d_emins,
    int N, int K,
    void* stream);

} // namespace llaminar2
