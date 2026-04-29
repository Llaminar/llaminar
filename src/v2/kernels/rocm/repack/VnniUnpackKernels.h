#pragma once

/**
 * @file VnniUnpackKernels.h
 * @brief ROCm/HIP kernels for reverse repack: GPU separated VNNI → raw GGUF blocks.
 *
 * HIP counterpart of kernels/cuda/repack/CUDAVnniUnpackKernels.h.
 * Only per-block formats are reversible (see CUDAVnniUnpackKernels.h for table).
 */

#include "loaders/gpu_pipeline/RepackFormat.h"

#include <cstdint>

namespace llaminar2 {

/**
 * @brief Launch HIP kernel to reverse-repack GPU separated VNNI → raw GGUF blocks.
 *
 * Only supports per-block formats (Q4_0, IQ4_NL, Q4_1, Q5_0, Q5_1, Q8_0).
 * Returns false for unsupported (lossy) formats.
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
