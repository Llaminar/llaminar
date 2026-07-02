#include <hip/hip_runtime.h>
#include <cstdio>

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

extern "C" bool rocmTPKV_deinterleave_rank_major_fp32(
    const float *rank_major,
    float *row_major,
    int tokens,
    int degree,
    int local_dim,
    int device_ordinal,
    void *stream)
{
    if (!rank_major || !row_major || tokens <= 0 || degree <= 1 || local_dim <= 0 || !stream)
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
