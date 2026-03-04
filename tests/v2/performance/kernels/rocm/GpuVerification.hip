/**
 * @file GpuVerification.hip
 * @brief GPU-accelerated correctness verification for GEMM benchmarks.
 *
 * Compiled by HIP compiler (amdclang++) — no x86 SIMD here.
 * Supports multi-GPU: per-device hipBLAS handles.
 */

#include "GpuVerification.h"
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <algorithm>
#include <cmath>
#include <mutex>
#include <vector>

namespace llaminar2::test::gpu_verify {

// =============================================================================
// Per-device hipBLAS handle management
// =============================================================================

static constexpr int MAX_DEVICES = 8;
static hipblasHandle_t g_handles[MAX_DEVICES] = {};
static std::mutex g_handle_mutex;

bool initHipBLAS(int device_id) {
    if (device_id < 0 || device_id >= MAX_DEVICES)
        return false;
    std::lock_guard<std::mutex> lock(g_handle_mutex);
    if (g_handles[device_id])
        return true;
    (void)hipSetDevice(device_id);
    return hipblasCreate(&g_handles[device_id]) == HIPBLAS_STATUS_SUCCESS;
}

void destroyAllHipBLAS() {
    std::lock_guard<std::mutex> lock(g_handle_mutex);
    for (int i = 0; i < MAX_DEVICES; ++i) {
        if (g_handles[i]) {
            (void)hipSetDevice(i);
            hipblasDestroy(g_handles[i]);
            g_handles[i] = nullptr;
        }
    }
}

// =============================================================================
// GPU FP32 reference GEMM via hipBLAS
// =============================================================================

bool gpuReferenceFP32Gemm(const float *d_A, const float *d_B,
                          float *d_C, int M, int N, int K,
                          int device_id) {
    if (!initHipBLAS(device_id))
        return false;

    (void)hipSetDevice(device_id);

    const float alpha = 1.0f, beta = 0.0f;
    // Row-major C[M×N] = A[M×K] × B^T[N×K]
    // See header for full derivation.
    auto status = hipblasSgemm(g_handles[device_id],
                               HIPBLAS_OP_T, HIPBLAS_OP_N,
                               N, M, K,
                               &alpha,
                               d_B, K,
                               d_A, K,
                               &beta,
                               d_C, N);
    return status == HIPBLAS_STATUS_SUCCESS;
}

// =============================================================================
// GPU cosine similarity kernel
// =============================================================================

__global__ void cosinePartialReduceKernel(const float *a, const float *b,
                                          double *partial_dot,
                                          double *partial_norm_a,
                                          double *partial_norm_b,
                                          size_t count) {
    __shared__ double s_dot[256];
    __shared__ double s_na[256];
    __shared__ double s_nb[256];

    const size_t tid = threadIdx.x;
    const size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    double local_dot = 0.0, local_na = 0.0, local_nb = 0.0;
    for (size_t i = gid; i < count; i += stride) {
        double av = static_cast<double>(a[i]);
        double bv = static_cast<double>(b[i]);
        local_dot += av * bv;
        local_na += av * av;
        local_nb += bv * bv;
    }

    s_dot[tid] = local_dot;
    s_na[tid] = local_na;
    s_nb[tid] = local_nb;
    __syncthreads();

    // Tree reduction
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < static_cast<unsigned>(s)) {
            s_dot[tid] += s_dot[tid + s];
            s_na[tid] += s_na[tid + s];
            s_nb[tid] += s_nb[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_dot[blockIdx.x] = s_dot[0];
        partial_norm_a[blockIdx.x] = s_na[0];
        partial_norm_b[blockIdx.x] = s_nb[0];
    }
}

float gpuCosineSimilarity(const float *d_a, const float *d_b, size_t count,
                          int device_id) {
    (void)hipSetDevice(device_id);
    constexpr int BLOCK = 256;
    const int num_blocks = std::min(static_cast<int>((count + BLOCK - 1) / BLOCK), 1024);

    double *d_partial_dot = nullptr, *d_partial_na = nullptr, *d_partial_nb = nullptr;
    (void)hipMalloc(&d_partial_dot, num_blocks * sizeof(double));
    (void)hipMalloc(&d_partial_na, num_blocks * sizeof(double));
    (void)hipMalloc(&d_partial_nb, num_blocks * sizeof(double));

    cosinePartialReduceKernel<<<num_blocks, BLOCK>>>(
        d_a, d_b, d_partial_dot, d_partial_na, d_partial_nb, count);
    (void)hipDeviceSynchronize();

    // Sum partials on host
    std::vector<double> h_dot(num_blocks), h_na(num_blocks), h_nb(num_blocks);
    (void)hipMemcpy(h_dot.data(), d_partial_dot, num_blocks * sizeof(double), hipMemcpyDeviceToHost);
    (void)hipMemcpy(h_na.data(), d_partial_na, num_blocks * sizeof(double), hipMemcpyDeviceToHost);
    (void)hipMemcpy(h_nb.data(), d_partial_nb, num_blocks * sizeof(double), hipMemcpyDeviceToHost);

    (void)hipFree(d_partial_dot);
    (void)hipFree(d_partial_na);
    (void)hipFree(d_partial_nb);

    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < num_blocks; ++i) {
        dot += h_dot[i];
        na += h_na[i];
        nb += h_nb[i];
    }
    double denom = std::sqrt(na) * std::sqrt(nb);
    return (denom > 0.0) ? static_cast<float>(dot / denom) : 0.0f;
}

// =============================================================================
// GpuWeightsCache
// =============================================================================

GpuWeightsCache::~GpuWeightsCache() { free(); }

void GpuWeightsCache::free() {
    if (d_weights) {
        (void)hipSetDevice(device_id);
        (void)hipFree(d_weights);
        d_weights = nullptr;
    }
}

bool GpuWeightsCache::upload(const float *h_weights, size_t n, size_t k, int dev_id) {
    free();
    N = n;
    K = k;
    device_id = dev_id;
    (void)hipSetDevice(device_id);
    size_t bytes = N * K * sizeof(float);
    auto err = hipMalloc(&d_weights, bytes);
    if (err != hipSuccess) {
        fprintf(stderr, "[GpuWeightsCache] hipMalloc(%zu bytes) failed: %s\n",
                bytes, hipGetErrorString(err));
        return false;
    }
    err = hipMemcpy(d_weights, h_weights, bytes, hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        fprintf(stderr, "[GpuWeightsCache] hipMemcpy failed: %s\n", hipGetErrorString(err));
        (void)hipFree(d_weights);
        d_weights = nullptr;
        return false;
    }
    return true;
}

} // namespace llaminar2::test::gpu_verify
