/**
 * @file CUDAVnniUnpackKernels.cu
 * @brief CUDA kernels for reverse repack: GPU separated VNNI → raw GGUF blocks
 *
 * Exact inverse of CUDAVnniRepackKernels.cu for per-block formats.
 * Each kernel maps one thread to one (n, b) block position.
 *
 * Forward:  raw[n * bpr + b] → payload[b * N + n], scales[b * N + n]
 * Reverse:  payload[b * N + n], scales[b * N + n] → raw[n * bpr + b]
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstring>

#include "kernels/cuda/repack/CUDAVnniUnpackKernels.h"
#include "tensors/BlockStructures.h"

namespace llaminar2 {

// ============================================================================
// Alignment-safe memory helpers (same as forward kernels)
// ============================================================================

__device__ __forceinline__ void unpack_copy16(void* __restrict__ dst,
                                               const void* __restrict__ src)
{
    memcpy(dst, src, 16);
}

// ============================================================================
// Kernel 1: Q4_0 / IQ4_NL unpack (symmetric 4-bit, 18-byte blocks)
// ============================================================================

__global__ void cuda_unpack_vnni_to_q4_0(
    const uint8_t*  __restrict__ d_payload,
    const uint16_t* __restrict__ d_scales,
    Q4_0Block*      __restrict__ d_raw,
    int N, int blocks_per_row)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    if (n >= N || b >= blocks_per_row) return;

    const size_t linear = static_cast<size_t>(b) * N + n;
    auto& blk = d_raw[n * blocks_per_row + b];

    unpack_copy16(blk.qs, d_payload + linear * 16);
    blk.d = d_scales[linear];
}

// ============================================================================
// Kernel 2: Q4_1 unpack (asymmetric 4-bit, 20-byte blocks)
// ============================================================================

__global__ void cuda_unpack_vnni_to_q4_1(
    const uint8_t*  __restrict__ d_payload,
    const uint16_t* __restrict__ d_scales,
    const uint16_t* __restrict__ d_mins,
    Q4_1Block*      __restrict__ d_raw,
    int N, int blocks_per_row)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    if (n >= N || b >= blocks_per_row) return;

    const size_t linear = static_cast<size_t>(b) * N + n;
    auto& blk = d_raw[n * blocks_per_row + b];

    unpack_copy16(blk.qs, d_payload + linear * 16);
    blk.d = d_scales[linear];
    blk.m = d_mins[linear];
}

// ============================================================================
// Kernel 3: Q5_0 unpack (symmetric 5-bit, 22-byte blocks)
// ============================================================================

__global__ void cuda_unpack_vnni_to_q5_0(
    const uint8_t*  __restrict__ d_payload,
    const uint16_t* __restrict__ d_scales,
    Q5_0Block*      __restrict__ d_raw,
    int N, int blocks_per_row)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    if (n >= N || b >= blocks_per_row) return;

    const size_t linear = static_cast<size_t>(b) * N + n;
    auto& blk = d_raw[n * blocks_per_row + b];

    const uint8_t* src = d_payload + linear * 20;
    unpack_copy16(blk.qs, src);
    memcpy(blk.qh, src + 16, 4);
    blk.d = d_scales[linear];
}

// ============================================================================
// Kernel 4: Q5_1 unpack (asymmetric 5-bit, 24-byte blocks)
// ============================================================================

__global__ void cuda_unpack_vnni_to_q5_1(
    const uint8_t*  __restrict__ d_payload,
    const uint16_t* __restrict__ d_scales,
    const uint16_t* __restrict__ d_mins,
    Q5_1Block*      __restrict__ d_raw,
    int N, int blocks_per_row)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    if (n >= N || b >= blocks_per_row) return;

    const size_t linear = static_cast<size_t>(b) * N + n;
    auto& blk = d_raw[n * blocks_per_row + b];

    const uint8_t* src = d_payload + linear * 20;
    unpack_copy16(blk.qs, src);
    memcpy(blk.qh, src + 16, 4);
    blk.d = d_scales[linear];
    blk.m = d_mins[linear];
}

// ============================================================================
// Kernel 5: Q8_0 unpack (symmetric 8-bit, 34-byte blocks)
// ============================================================================

__global__ void cuda_unpack_vnni_to_q8_0(
    const uint8_t*  __restrict__ d_payload,
    const uint16_t* __restrict__ d_scales,
    Q8_0Block*      __restrict__ d_raw,
    int N, int blocks_per_row)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    if (n >= N || b >= blocks_per_row) return;

    const size_t linear = static_cast<size_t>(b) * N + n;
    auto& blk = d_raw[n * blocks_per_row + b];

    const uint8_t* src = d_payload + linear * 32;
    unpack_copy16(blk.qs, src);
    unpack_copy16(blk.qs + 16, src + 16);
    blk.d = d_scales[linear];
}

// ============================================================================
// Host dispatch
// ============================================================================

bool launchVnniUnpackCUDA(
    RepackFormat format,
    const uint8_t* d_payload,
    const uint16_t* d_scales,
    const uint16_t* d_mins,
    void* d_raw_blocks,
    int N, int K,
    void* stream)
{
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    const int blocks_per_row = K / 32;

    dim3 block(256, 1);
    dim3 grid((N + 255) / 256, blocks_per_row);

    switch (format) {
    case RepackFormat::Q4_0:
    case RepackFormat::IQ4_NL:
        cuda_unpack_vnni_to_q4_0<<<grid, block, 0, cuda_stream>>>(
            d_payload, d_scales,
            static_cast<Q4_0Block*>(d_raw_blocks),
            N, blocks_per_row);
        break;

    case RepackFormat::Q4_1:
        cuda_unpack_vnni_to_q4_1<<<grid, block, 0, cuda_stream>>>(
            d_payload, d_scales, d_mins,
            static_cast<Q4_1Block*>(d_raw_blocks),
            N, blocks_per_row);
        break;

    case RepackFormat::Q5_0:
        cuda_unpack_vnni_to_q5_0<<<grid, block, 0, cuda_stream>>>(
            d_payload, d_scales,
            static_cast<Q5_0Block*>(d_raw_blocks),
            N, blocks_per_row);
        break;

    case RepackFormat::Q5_1:
        cuda_unpack_vnni_to_q5_1<<<grid, block, 0, cuda_stream>>>(
            d_payload, d_scales, d_mins,
            static_cast<Q5_1Block*>(d_raw_blocks),
            N, blocks_per_row);
        break;

    case RepackFormat::Q8_0:
        cuda_unpack_vnni_to_q8_0<<<grid, block, 0, cuda_stream>>>(
            d_payload, d_scales,
            static_cast<Q8_0Block*>(d_raw_blocks),
            N, blocks_per_row);
        break;

    default:
        // Superblock and IQ formats are not reversible
        return false;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[launchVnniUnpackCUDA] Kernel launch failed: %s\n",
                cudaGetErrorString(err));
        return false;
    }
    return true;
}

} // namespace llaminar2
