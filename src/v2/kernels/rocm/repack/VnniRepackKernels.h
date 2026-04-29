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
 *   scales[b * N + n]   — FP16 scale per block
 *   mins[b * N + n]     — FP16 min per block (asymmetric only)
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
 * on the given stream; call hipDeviceSynchronize() or stream sync before
 * reading results.
 *
 * @param format       Quantization format of the raw blocks
 * @param d_raw_blocks Raw GGUF block data on device (uploaded as-is from host)
 * @param d_payload    Output VNNI payload [blocks_per_row * N * payload_bytes]
 * @param d_scales     Output VNNI scales [blocks_per_row * N] (FP16 as uint16_t)
 * @param d_mins       Output VNNI mins [blocks_per_row * N] (FP16), nullptr for symmetric
 * @param N            Number of output features (rows in weight matrix)
 * @param K            Number of input features (columns in weight matrix)
 * @param stream       HIP stream to launch on (nullptr = default stream)
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
    void* stream);

} // namespace llaminar2
