#include <hip/hip_runtime.h>

#include <cstdio>

namespace
{
    constexpr int TINY_FP32_BLOCK = 256;
    constexpr int TINY_FP32_MAX_BATCH = 8;

    struct Fp32BatchedProjectionPointers
    {
        const float *A[TINY_FP32_MAX_BATCH];
        const float *B[TINY_FP32_MAX_BATCH];
        float *C[TINY_FP32_MAX_BATCH];
    };

    __global__ void fp32_stage_batched_projection_pointers_kernel(
        const Fp32BatchedProjectionPointers ptrs,
        const float **__restrict__ d_A_array,
        const float **__restrict__ d_B_array,
        float **__restrict__ d_C_array,
        int batch_count)
    {
        const int batch = static_cast<int>(threadIdx.x);
        if (batch >= batch_count)
            return;

        d_A_array[batch] = ptrs.A[batch];
        d_B_array[batch] = ptrs.B[batch];
        d_C_array[batch] = ptrs.C[batch];
    }

    __global__ __launch_bounds__(TINY_FP32_BLOCK)
        void fp32_tiny_batched_projection_kernel(
            const float *const *__restrict__ d_A_array,
            const float *const *__restrict__ d_B_array,
            float *const *__restrict__ d_C_array,
            int M,
            int N,
            int K)
    {
        const int n = static_cast<int>(blockIdx.x);
        const int m = static_cast<int>(blockIdx.y);
        const int batch = static_cast<int>(blockIdx.z);
        if (m >= M || n >= N)
            return;

        const float *A = d_A_array[batch];
        const float *B = d_B_array[batch];
        float *C = d_C_array[batch];
        if (!A || !B || !C)
            return;

        float sum = 0.0f;
        for (int k = static_cast<int>(threadIdx.x); k < K; k += TINY_FP32_BLOCK)
        {
            sum += A[static_cast<size_t>(m) * K + k] *
                   B[static_cast<size_t>(n) * K + k];
        }

        __shared__ float scratch[TINY_FP32_BLOCK];
        scratch[threadIdx.x] = sum;
        __syncthreads();

        for (int stride = TINY_FP32_BLOCK / 2; stride > 0; stride >>= 1)
        {
            if (threadIdx.x < stride)
                scratch[threadIdx.x] += scratch[threadIdx.x + stride];
            __syncthreads();
        }

        if (threadIdx.x == 0)
            C[static_cast<size_t>(m) * N + n] = scratch[0];
    }
} // namespace

extern "C" bool rocmFp32_stage_batched_projection_pointers(
    const float **d_A_array,
    const float **d_B_array,
    float **d_C_array,
    const float *const *h_A_ptrs,
    const float *const *h_B_ptrs,
    float *const *h_C_ptrs,
    int batch_count,
    int device_id,
    void *stream)
{
    if (!d_A_array || !d_B_array || !d_C_array ||
        !h_A_ptrs || !h_B_ptrs || !h_C_ptrs ||
        batch_count <= 0 || batch_count > TINY_FP32_MAX_BATCH ||
        !stream)
    {
        std::fprintf(
            stderr,
            "[rocmFp32_stage_batched_projection_pointers] invalid arguments: batch=%d stream=%p\n",
            batch_count,
            stream);
        return false;
    }

    Fp32BatchedProjectionPointers ptrs{};
    for (int i = 0; i < batch_count; ++i)
    {
        if (!h_A_ptrs[i] || !h_B_ptrs[i] || !h_C_ptrs[i])
        {
            std::fprintf(
                stderr,
                "[rocmFp32_stage_batched_projection_pointers] null pointer in batch %d\n",
                i);
            return false;
        }
        ptrs.A[i] = h_A_ptrs[i];
        ptrs.B[i] = h_B_ptrs[i];
        ptrs.C[i] = h_C_ptrs[i];
    }

    hipError_t err = hipSetDevice(device_id);
    if (err != hipSuccess)
    {
        std::fprintf(
            stderr,
            "[rocmFp32_stage_batched_projection_pointers] hipSetDevice failed: %s\n",
            hipGetErrorString(err));
        return false;
    }

    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    hipLaunchKernelGGL(
        fp32_stage_batched_projection_pointers_kernel,
        dim3(1),
        dim3(TINY_FP32_MAX_BATCH),
        0,
        hip_stream,
        ptrs,
        d_A_array,
        d_B_array,
        d_C_array,
        batch_count);

    err = hipGetLastError();
    if (err != hipSuccess)
    {
        std::fprintf(
            stderr,
            "[rocmFp32_stage_batched_projection_pointers] launch failed: %s\n",
            hipGetErrorString(err));
        return false;
    }
    return true;
}

extern "C" bool rocmFp32_tiny_batched_projection(
    const float *const *d_A_array,
    const float *const *d_B_array,
    float *const *d_C_array,
    int M,
    int N,
    int K,
    int batch_count,
    int device_id,
    void *stream)
{
    if (!d_A_array || !d_B_array || !d_C_array ||
        M <= 0 || M > 4 ||
        N <= 0 || N > 64 ||
        K <= 0 ||
        batch_count <= 0 || batch_count > 8)
    {
        std::fprintf(
            stderr,
            "[rocmFp32_tiny_batched_projection] invalid arguments: M=%d N=%d K=%d batch=%d\n",
            M,
            N,
            K,
            batch_count);
        return false;
    }

    hipError_t err = hipSetDevice(device_id);
    if (err != hipSuccess)
    {
        std::fprintf(
            stderr,
            "[rocmFp32_tiny_batched_projection] hipSetDevice failed: %s\n",
            hipGetErrorString(err));
        return false;
    }

    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    dim3 grid(static_cast<unsigned>(N), static_cast<unsigned>(M), static_cast<unsigned>(batch_count));
    dim3 block(TINY_FP32_BLOCK);
    hipLaunchKernelGGL(
        fp32_tiny_batched_projection_kernel,
        grid,
        block,
        0,
        hip_stream,
        d_A_array,
        d_B_array,
        d_C_array,
        M,
        N,
        K);

    err = hipGetLastError();
    if (err != hipSuccess)
    {
        std::fprintf(
            stderr,
            "[rocmFp32_tiny_batched_projection] launch failed: %s\n",
            hipGetErrorString(err));
        return false;
    }
    return true;
}
