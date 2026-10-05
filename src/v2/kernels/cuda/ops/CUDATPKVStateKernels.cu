/**
 * @file CUDATPKVStateKernels.cu
 * @brief Exact-copy row compaction and column transposition on an explicit CUDA stream.
 *
 * These kernels preserve FP32 bit patterns and borrow preallocated disjoint
 * banks. Swapping the transpose's outer dimensions packs model rows for native
 * reduce-scatter; one outer element is a valid copy, not a reason to skip it.
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <algorithm>
#include "kernels/common/DeviceRowRange.h"

namespace
{
    __global__ void tpkv_compact_rows_fp32_kernel(
        const float *__restrict__ src,
        float *__restrict__ dst,
        int tokens,
        int local_dim,
        int src_stride)
    {
        const long long total = static_cast<long long>(tokens) * local_dim;
        for (long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
             idx < total;
             idx += static_cast<long long>(blockDim.x) * gridDim.x)
        {
            const int row = static_cast<int>(idx / local_dim);
            const int col = static_cast<int>(idx - static_cast<long long>(row) * local_dim);
            dst[idx] = src[static_cast<long long>(row) * src_stride + col];
        }
    }

    __global__ void tpkv_deinterleave_rank_major_fp32_kernel(
        const float *__restrict__ rank_major,
        float *__restrict__ row_major,
        int tokens,
        int degree,
        int local_dim)
    {
        const int full_dim = degree * local_dim;
        const long long total = static_cast<long long>(tokens) * full_dim;
        for (long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
             idx < total;
             idx += static_cast<long long>(blockDim.x) * gridDim.x)
        {
            const int row = static_cast<int>(idx / full_dim);
            const int col = static_cast<int>(idx - static_cast<long long>(row) * full_dim);
            const int rank = col / local_dim;
            const int local_col = col - rank * local_dim;
            const long long src =
                (static_cast<long long>(rank) * tokens + row) * local_dim + local_col;
            row_major[idx] = rank_major[src];
        }
    }

    /** @brief Copy a device-owned live prefix while retaining native capacity rank strides. */
    __global__ void tpkv_deinterleave_live_rank_major_fp32_kernel(
        const float *__restrict__ rank_major, float *__restrict__ row_major,
        llaminar2::DeviceRowRange rows, int degree, int local_dim)
    {
        const int live = rows.activeRows();
        const int full_dim = degree * local_dim;
        const long long total = static_cast<long long>(live) * full_dim;
        for (long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
             idx < total; idx += static_cast<long long>(blockDim.x) * gridDim.x)
        {
            const int row = static_cast<int>(idx / full_dim);
            const int col = static_cast<int>(idx - static_cast<long long>(row) * full_dim);
            const int rank = col / local_dim;
            const int local_col = col - rank * local_dim;
            // Wire volume follows live, but the retained allgather bank's rank
            // stride remains capacity on every large-to-small and empty replay.
            const long long source =
                (static_cast<long long>(rank) * rows.capacity() + row) * local_dim + local_col;
            row_major[idx] = rank_major[source];
        }
    }

}

extern "C" bool cudaTPKV_compact_rows_fp32(
    const float *src,
    float *dst,
    int tokens,
    int local_dim,
    int src_stride,
    int device_ordinal,
    void *stream)
{
    if (!src || !dst || tokens <= 0 || local_dim <= 0 || src_stride < local_dim || !stream)
        return false;

    cudaError_t err = cudaSetDevice(device_ordinal);
    if (err != cudaSuccess)
    {
        std::fprintf(stderr, "[cudaTPKV_compact_rows_fp32] cudaSetDevice(%d): %s\n",
                     device_ordinal, cudaGetErrorString(err));
        return false;
    }

    const int threads = 256;
    const long long total = static_cast<long long>(tokens) * local_dim;
    int blocks = static_cast<int>((total + threads - 1) / threads);
    if (blocks > 65535)
        blocks = 65535;

    tpkv_compact_rows_fp32_kernel<<<blocks, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        src,
        dst,
        tokens,
        local_dim,
        src_stride);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::fprintf(stderr, "[cudaTPKV_compact_rows_fp32] launch: %s\n",
                     cudaGetErrorString(err));
        return false;
    }
    return true;
}

/** @brief Enqueue [rank,row,column] to [row,rank,column], including singleton dimensions. */
extern "C" bool cudaTPKV_deinterleave_rank_major_fp32(
    const float *rank_major,
    float *row_major,
    int tokens,
    int degree,
    int local_dim,
    int device_ordinal,
    void *stream)
{
    if (!rank_major || !row_major || tokens <= 0 || degree < 1 || local_dim <= 0 || !stream)
        return false;

    cudaError_t err = cudaSetDevice(device_ordinal);
    if (err != cudaSuccess)
    {
        std::fprintf(stderr, "[cudaTPKV_deinterleave_rank_major_fp32] cudaSetDevice(%d): %s\n",
                     device_ordinal, cudaGetErrorString(err));
        return false;
    }

    const int threads = 256;
    const long long total = static_cast<long long>(tokens) * degree * local_dim;
    int blocks = static_cast<int>((total + threads - 1) / threads);
    if (blocks > 65535)
        blocks = 65535;

    tpkv_deinterleave_rank_major_fp32_kernel<<<blocks, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        rank_major,
        row_major,
        tokens,
        degree,
        local_dim);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::fprintf(stderr, "[cudaTPKV_deinterleave_rank_major_fp32] launch: %s\n",
                     cudaGetErrorString(err));
        return false;
    }
    return true;
}

/** @brief Enqueue only the live prefix; exact stream and capacities were authenticated by the shared ABI. */
extern "C" bool cudaTPKV_deinterleave_live_rank_major_fp32(
    const float *rank_major, float *row_major, llaminar2::DeviceRowRange rows,
    int degree, int local_dim, int device_ordinal, void *stream)
{
    if (!rank_major || !row_major || !stream || degree < 1 || local_dim <= 0 ||
        rows.physicalRows() != rows.capacity()) return false;
    if (cudaSetDevice(device_ordinal) != cudaSuccess) return false;
    constexpr int threads = 256;
    const long long total = static_cast<long long>(rows.capacity()) * degree * local_dim;
    int processors = 0, active_blocks = 0;
    if (cudaDeviceGetAttribute(&processors, cudaDevAttrMultiProcessorCount, device_ordinal) != cudaSuccess ||
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks,
            tpkv_deinterleave_live_rank_major_fp32_kernel, threads, 0) != cudaSuccess ||
        processors <= 0 || active_blocks <= 0) return false;
    // Recording freezes one fully occupied wave of CTAs. Each CTA strides
    // over the live prefix; large-to-small and empty replays avoid launching
    // thousands of capacity-only CTAs without reading the live count on host.
    const int blocks = static_cast<int>(std::min(static_cast<long long>(processors) * active_blocks,
        (total + threads - 1) / threads));
    tpkv_deinterleave_live_rank_major_fp32_kernel<<<blocks, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        rank_major, row_major, rows, degree, local_dim);
    return cudaGetLastError() == cudaSuccess;
}
