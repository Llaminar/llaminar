/**
 * @file ROCmTPKVStateKernels.hip
 * @brief Exact-copy row compaction and column transposition on an explicit HIP stream.
 *
 * Borrowed banks retain the original FP32 bits. The same transpose assembles
 * gathered columns and, with exchanged outer dimensions, packs reduce-scatter
 * input. Singleton dimensions still enqueue the required disjoint copy.
 */
#include <hip/hip_runtime.h>
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

extern "C" bool rocmTPKV_compact_rows_fp32(
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

    hipError_t err = hipSetDevice(device_ordinal);
    if (err != hipSuccess)
    {
        std::fprintf(stderr, "[rocmTPKV_compact_rows_fp32] hipSetDevice(%d): %s\n",
                     device_ordinal, hipGetErrorString(err));
        return false;
    }

    const int threads = 256;
    const long long total = static_cast<long long>(tokens) * local_dim;
    int blocks = static_cast<int>((total + threads - 1) / threads);
    if (blocks > 65535)
        blocks = 65535;

    hipLaunchKernelGGL(tpkv_compact_rows_fp32_kernel,
                       dim3(blocks),
                       dim3(threads),
                       0,
                       static_cast<hipStream_t>(stream),
                       src,
                       dst,
                       tokens,
                       local_dim,
                       src_stride);

    err = hipGetLastError();
    if (err != hipSuccess)
    {
        std::fprintf(stderr, "[rocmTPKV_compact_rows_fp32] launch: %s\n",
                     hipGetErrorString(err));
        return false;
    }
    return true;
}

/** @brief Enqueue [rank,row,column] to [row,rank,column], including singleton dimensions. */
extern "C" bool rocmTPKV_deinterleave_rank_major_fp32(
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

    hipError_t err = hipSetDevice(device_ordinal);
    if (err != hipSuccess)
    {
        std::fprintf(stderr, "[rocmTPKV_deinterleave_rank_major_fp32] hipSetDevice(%d): %s\n",
                     device_ordinal, hipGetErrorString(err));
        return false;
    }

    const int threads = 256;
    const long long total = static_cast<long long>(tokens) * degree * local_dim;
    int blocks = static_cast<int>((total + threads - 1) / threads);
    if (blocks > 65535)
        blocks = 65535;

    hipLaunchKernelGGL(tpkv_deinterleave_rank_major_fp32_kernel,
                       dim3(blocks),
                       dim3(threads),
                       0,
                       static_cast<hipStream_t>(stream),
                       rank_major,
                       row_major,
                       tokens,
                       degree,
                       local_dim);

    err = hipGetLastError();
    if (err != hipSuccess)
    {
        std::fprintf(stderr, "[rocmTPKV_deinterleave_rank_major_fp32] launch: %s\n",
                     hipGetErrorString(err));
        return false;
    }
    return true;
}

/** @brief Enqueue only the live prefix; exact stream and capacities were authenticated by the shared ABI. */
extern "C" bool rocmTPKV_deinterleave_live_rank_major_fp32(
    const float *rank_major, float *row_major, llaminar2::DeviceRowRange rows,
    int degree, int local_dim, int device_ordinal, void *stream)
{
    if (!rank_major || !row_major || !stream || degree < 1 || local_dim <= 0 ||
        rows.physicalRows() != rows.capacity()) return false;
    if (hipSetDevice(device_ordinal) != hipSuccess) return false;
    constexpr int threads = 256;
    const long long total = static_cast<long long>(rows.capacity()) * degree * local_dim;
    int processors = 0;
    int resident_blocks = 0;
    if (hipDeviceGetAttribute(&processors, hipDeviceAttributeMultiprocessorCount, device_ordinal) != hipSuccess ||
        hipOccupancyMaxActiveBlocksPerMultiprocessor(&resident_blocks,
            tpkv_deinterleave_live_rank_major_fp32_kernel, threads, 0) != hipSuccess ||
        processors <= 0 || resident_blocks <= 0) return false;
    // One resident wave keeps small/empty replays economical while the
    // grid-stride body still copies every live element of a larger bank.
    // These immutable resource queries run during retained graph recording.
    const int blocks = static_cast<int>(std::min(static_cast<long long>(processors) * resident_blocks,
        (total + threads - 1) / threads));
    hipLaunchKernelGGL(tpkv_deinterleave_live_rank_major_fp32_kernel, dim3(blocks), dim3(threads),
        0, static_cast<hipStream_t>(stream), rank_major, row_major, rows, degree, local_dim);
    return hipGetLastError() == hipSuccess;
}
