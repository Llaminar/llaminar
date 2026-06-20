#pragma once

/**
 * @file CUDAVnniUnpackKernels.h
 * @brief CUDA kernels for reverse repack: GPU separated VNNI → raw GGUF blocks.
 *
 * These are the exact inverse of CUDAVnniRepackKernels.  They reconstruct
 * the original raw GGUF block layout from the separated payload/scale/min
 * arrays used by GPU GEMM kernels.
 *
 * ## Reversibility
 *
 * Only **per-block formats** (32-element blocks) are fully reversible because
 * their scales are stored as-is in FP16.  Superblock formats (Q4_K, Q5_K, etc.)
 * decompose the superblock scale structure into per-sub-block FP16 values during
 * forward repack — this is lossy and cannot be inverted.
 *
 * | Format  | Reversible | Notes                                    |
 * |---------|------------|------------------------------------------|
 * | Q4_0    | ✅         | 16 B payload + FP16 scale                |
 * | IQ4_NL  | ✅         | Same block layout as Q4_0                |
 * | Q4_1    | ✅         | 16 B payload + FP16 scale + FP16 min     |
 * | Q5_0    | ✅         | 20 B payload + FP16 scale                |
 * | Q5_1    | ✅         | 20 B payload + FP16 scale + FP16 min     |
 * | Q8_0    | ✅         | 32 B payload + FP16 scale                |
 * | Q4_K    | ❌         | Superblock scale decomposition is lossy  |
 * | Q5_K    | ❌         | Superblock scale decomposition is lossy  |
 * | Q6_K    | ❌         | Superblock scale decomposition is lossy  |
 * | Q3_K    | ❌         | Superblock scale decomposition is lossy  |
 * | Q2_K    | ❌         | Superblock scale decomposition is lossy  |
 * | IQ*     | ❌         | Scale nibble expressions are lossy        |
 */

#include "loaders/gpu_pipeline/RepackFormat.h"

#include <cstdint>

namespace llaminar2 {

/**
 * @brief Check whether a format supports lossless GPU→raw-GGUF reverse repack.
 */
inline bool isReversibleFormat(RepackFormat format)
{
    switch (format) {
    case RepackFormat::Q4_0:
    case RepackFormat::IQ4_NL:
    case RepackFormat::Q4_1:
    case RepackFormat::Q5_0:
    case RepackFormat::Q5_1:
    case RepackFormat::Q8_0:
        return true;
    default:
        return false;
    }
}

/**
 * @brief Launch CUDA kernel to reverse-repack GPU separated VNNI → raw GGUF blocks.
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
 * @param stream       CUDA stream (nullptr = default)
 * @return true on success, false on unsupported format or launch error
 */
bool launchVnniUnpackCUDA(
    RepackFormat format,
    const uint8_t* d_payload,
    const uint16_t* d_scales,
    const uint16_t* d_mins,
    void* d_raw_blocks,
    int N, int K,
    void* stream);

} // namespace llaminar2
