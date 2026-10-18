#pragma once

/**
 * @file CUDAVnniUnpackKernels.h
 * @note Q8_K source d is FP32. Existing prepared arrays store FP16 scales,
 * so reverse conversion preserves only that prepared scale, not arbitrary
 * original d bits. Original source/native embedding storage remains exact.
 * @brief CUDA kernels for reverse repack: GPU separated VNNI → raw GGUF blocks.
 *
 * These are the exact inverse of CUDAVnniRepackKernels.  They reconstruct
 * the original raw GGUF block layout from the separated payload/scale/min
 * arrays used by GPU GEMM kernels.
 *
 * ## Reversibility
 *
 * Formats are reversible when every source field is either retained directly or
 * derivable from the retained payload.  Most scale-compressed superblock formats
 * are lossy after decomposition.  Q8_K retains exact INT8 codes and reconstructible sums, but its original
 * FP32 scale has already been narrowed by the prepared FP16 metadata ABI.
 *
 * | Format  | Reversible | Notes                                    |
 * |---------|------------|------------------------------------------|
 * | Q4_0    | ✅         | 16 B payload + FP16 scale                |
 * | IQ4_NL  | ✅         | Same block layout as Q4_0                |
 * | Q4_1    | ✅         | 16 B payload + FP16 scale + FP16 min     |
 * | Q5_0    | ✅         | 20 B payload + FP16 scale                |
 * | Q5_1    | ✅         | 20 B payload + FP16 scale + FP16 min     |
 * | Q8_0    | ✅         | 32 B payload + FP16 scale                |
 * | Q8_1    | ✅         | Sum reconstructed from payload           |
 * | Q8_K    | FP16 scale | Codes/sums exact; source FP32 d is narrowed |
 * | Q4_K    | ❌         | Superblock scale decomposition is lossy  |
 * | Q5_K    | ❌         | Superblock scale decomposition is lossy  |
 * | Q6_K    | ❌         | Source-tuple reverse unpack not implemented |
 * | Q3_K    | ❌         | Superblock scale decomposition is lossy  |
 * | Q2_K    | ❌         | Superblock scale decomposition is lossy  |
 * | IQ*     | ❌         | Scale nibble expressions are lossy        |
 */

#include "loaders/gpu_pipeline/RepackFormat.h"

#include <cstdint>

namespace llaminar2 {

/**
 * @brief Check whether the existing prepared layout supports GPU→raw reconstruction.
 * @note Q8_K reconstruction retains prepared FP16 scale precision only.
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
    case RepackFormat::Q8_1:
    case RepackFormat::Q8_K:
        return true;
    default:
        return false;
    }
}

/**
 * @brief Launch CUDA kernel to reverse-repack GPU separated VNNI → raw GGUF blocks.
 *
 * Supports the reconstruction formats above, including FP16-scale Q8_K. Returns
 * false for scale-compressed superblock formats whose original metadata cannot
 * be reconstructed from the prepared representation.
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
