#pragma once

/**
 * @file CUDAVnniRepackKernels.h
 * @brief Host-side declarations for CUDA VNNI repack kernels
 *
 * CUDA counterpart of kernels/rocm/repack/VnniRepackKernels.h.
 * Provides GPU kernels that repack raw GGUF quantized blocks into
 * VNNI-interleaved layout directly on NVIDIA GPUs.
 */

#include "loaders/gpu_pipeline/RepackFormat.h"

#include <cstdint>
#include <cstddef>

namespace llaminar2 {

/// Launch CUDA kernel to repack raw GGUF blocks into VNNI layout (7-param).
bool launchVnniRepackCUDA(
    RepackFormat format,
    const void* d_raw_blocks,
    uint8_t* d_payload,
    uint16_t* d_scales,
    uint16_t* d_mins,
    int N, int K,
    void* stream);

/// Extended overload accepting d_emins for Q2_K format (8-param).
bool launchVnniRepackCUDA(
    RepackFormat format,
    const void* d_raw_blocks,
    uint8_t* d_payload,
    uint16_t* d_scales,
    uint16_t* d_mins,
    uint32_t* d_emins,
    int N, int K,
    void* stream);

} // namespace llaminar2
