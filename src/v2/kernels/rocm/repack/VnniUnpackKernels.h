#pragma once

/**
 * @file VnniUnpackKernels.h
 * @note Q8_K source d is FP32. Existing prepared arrays store FP16 scales,
 * so reverse conversion preserves only that prepared scale, not arbitrary
 * original d bits. Original source/native embedding storage remains exact.
 * @brief ROCm/HIP kernels for reverse repack: GPU separated VNNI → raw GGUF blocks.
 *
 * HIP counterpart of kernels/cuda/repack/CUDAVnniUnpackKernels.h.
 * Q8_K codes are preserved; its FP32 scale is narrowed by preparation. Its partial
 * sums are derived from those bytes.  See CUDAVnniUnpackKernels.h for the full
 * format table shared by both backends.
 */

#include "loaders/gpu_pipeline/RepackFormat.h"

#include <cstdint>

namespace llaminar2 {

/**
 * @brief Launch HIP kernel to reverse-repack GPU separated VNNI → raw GGUF blocks.
 *
 * Supports Q8_1 and prepared-FP16-scale Q8_K reconstruction. Returns
 * false for unimplemented superblock reverse formats. Q6 source-factor metadata
 * now preserves its original scales, but this API does not yet reconstruct its
 * bitplanes into complete source superblocks; no implicit reverse path is added.
 *
 * @param format       Quantization format
 * @param d_payload    GPU separated payload [blocks_per_row * N * payload_bytes]
 * @param d_scales     GPU separated scales  [blocks_per_row * N] (FP16)
 * @param d_mins       GPU separated mins    [blocks_per_row * N] (FP16), nullptr if symmetric
 * @param d_raw_blocks Output: raw GGUF blocks [N * blocks_per_row * block_size]
 * @param N            Output features (rows)
 * @param K            Input features (columns)
 * @param stream       HIP stream (nullptr = default)
 * @return true on success, false on unsupported format or launch error
 */
bool launchVnniUnpack(
    RepackFormat format,
    const uint8_t* d_payload,
    const uint16_t* d_scales,
    const uint16_t* d_mins,
    void* d_raw_blocks,
    int N, int K,
    void* stream);

} // namespace llaminar2
