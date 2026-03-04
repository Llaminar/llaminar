#pragma once
/**
 * @file GpuVerification.h
 * @brief GPU-accelerated correctness verification for GEMM benchmarks.
 *
 * Provides hipBLAS-based FP32 reference GEMM and GPU cosine similarity,
 * replacing the O(M*N*K) CPU reference path.
 *
 * All functions accept a device_id parameter for multi-GPU operation.
 * Implementation in GpuVerification.hip (compiled by HIP compiler).
 */

#include <cstddef>

namespace llaminar2::test::gpu_verify
{

    /// Initialize hipBLAS handle for a specific device (lazy, safe to call multiple times).
    bool initHipBLAS(int device_id = 0);

    /// Destroy all hipBLAS handles. Call once at program exit.
    void destroyAllHipBLAS();

    /// GPU FP32 reference GEMM: C[M×N] = A[M×K] × B^T[N×K]
    /// All pointers must be device memory on the specified device.
    bool gpuReferenceFP32Gemm(const float *d_A, const float *d_B,
                              float *d_C, int M, int N, int K,
                              int device_id = 0);

    /// GPU cosine similarity between two device vectors.
    /// Returns cosine similarity as float.
    float gpuCosineSimilarity(const float *d_a, const float *d_b, size_t count,
                              int device_id = 0);

    /// RAII wrapper for GPU-resident FP32 weights [N×K].
    struct GpuWeightsCache
    {
        float *d_weights = nullptr;
        size_t N = 0, K = 0;
        int device_id = 0;

        ~GpuWeightsCache();
        void free();

        /// Upload host FP32 weights to GPU. Returns false on failure.
        bool upload(const float *h_weights, size_t n, size_t k, int dev_id = 0);
    };

} // namespace llaminar2::test::gpu_verify
