/**
 * @file ROCmQuantisedGemmKernel_FP16.hip
 * @brief FP16 GEMM path for quantized INT8 weights on AMD ROCm GPUs
 *
 * This file implements an alternative approach to INT8 GEMM that converts
 * quantized INT8 data to FP16 on-the-fly and uses hipBLAS hgemm (FP16 GEMM).
 *
 * ============================================================================
 * MOTIVATION
 * ============================================================================
 *
 * On gfx906 (MI50/MI60), the native FP16 MFMA instructions may be faster than
 * INT8 paths through ComposableKernel due to:
 *   1. Better-tuned hipBLAS hgemm kernels
 *   2. Simpler scaling (separate epilogue vs D-tensor complexity)
 *   3. Native hardware support for FP16 matrix ops
 *
 * Trade-off:
 *   - INT8: 1 byte/element, but CK overhead + scale complexity
 *   - FP16: 2 bytes/element, but native acceleration + simpler path
 *
 * ============================================================================
 * APPROACH: "INT8-in, FP16-GEMM, FP32-out"
 * ============================================================================
 *
 * ┌─────────────────────────────────────────────────────────────────────────┐
 * │                                                                         │
 * │  INT8 A [M×K]  ────────► INT8→FP16 Kernel ──► FP16 A' [M×K]  ─┐         │
 * │                                                               │         │
 * │                                                               ├──► hgemm│
 * │  INT8 B [K×N]  ────────► INT8→FP16 Kernel ──► FP16 B' [K×N]  ─┘         │
 * │                                                               │         │
 * │                                                               ▼         │
 * │                                              FP16 C [M×N] (raw sum)     │
 * │                                                               │         │
 * │  scaleA [M]   ───┐                                            ▼         │
 * │                  ├──► FP16→FP32+Scale Kernel ──► FP32 E [M×N]          │
 * │  scaleB [N]   ───┘    (apply scales in epilogue)                        │
 * │                                                                         │
 * └─────────────────────────────────────────────────────────────────────────┘
 *
 * Pipeline steps:
 *   1. Convert A[M×K] from INT8 to FP16 (simple cast)
 *   2. Convert B[K×N] from INT8 to FP16 (simple cast)
 *   3. hipBLAS hgemm: C_fp16 = A_fp16 @ B_fp16
 *   4. Epilogue: E[m,n] = (float)C_fp16[m,n] * scaleA[m] * scaleB[n]
 *
 * This approach applies scales in the epilogue kernel during FP16→FP32
 * conversion, avoiding precision issues with pre-multiplying scales.
 *
 * ============================================================================
 * FUNCTIONS
 * ============================================================================
 *
 * Conversion Kernels:
 *   - int8ToFP16WithScale_A_kernel: Convert A[M×K] to FP16 with scaleA fused
 *   - int8ToFP16WithScale_B_kernel: Convert B[K×N] to FP16 with scaleB fused
 *   - fp16ToFP32_kernel: Convert FP16 to FP32 (no scaling needed)
 *
 * Orchestration:
 *   - rocmQuantGemm_executeFP16: Full INT8→FP16→GEMM→FP32 pipeline
 *
 * @author David Sanftenberg
 * @date January 2026
 */

// Note: We don't include ROCmQuantisedGemmKernel.h here to avoid MPI dependency
// in HIP compilation. The function declarations are in the header, but this
// file only needs HIP and hipBLAS.
#include "utils/Logger.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hipblas/hipblas.h>

#include <cmath>
#include <cstdint>
#include <memory>

namespace llaminar2
{
    namespace rocm
    {

        // =====================================================================
        // Helper macros
        // =====================================================================

#define HIP_CHECK_FP16(call)                                                        \
    do                                                                              \
    {                                                                               \
        hipError_t err = call;                                                      \
        if (err != hipSuccess)                                                      \
        {                                                                           \
            LOG_ERROR("[FP16 GEMM] HIP error: " << hipGetErrorString(err)           \
                                                << " at " << __FILE__               \
                                                << ":" << __LINE__);                \
            return false;                                                           \
        }                                                                           \
    } while (0)

#define HIPBLAS_CHECK_FP16(call)                                                    \
    do                                                                              \
    {                                                                               \
        hipblasStatus_t status = call;                                              \
        if (status != HIPBLAS_STATUS_SUCCESS)                                       \
        {                                                                           \
            LOG_ERROR("[FP16 GEMM] hipBLAS error: " << static_cast<int>(status)     \
                                                    << " at " << __FILE__           \
                                                    << ":" << __LINE__);            \
            return false;                                                           \
        }                                                                           \
    } while (0)

        // =====================================================================
        // Conversion Kernels
        // =====================================================================

        /**
         * @brief Convert INT8 matrix A[M×K] to FP16 with scale fused
         *
         * Each element: out[m,k] = (half)(A[m,k] * scaleA[m])
         *
         * By fusing scaleA into the conversion, we:
         *   1. Keep values in FP16 range (prevents overflow)
         *   2. The GEMM result will have scaleA × scaleB automatically applied
         *
         * Grid: (K/256, M) threads
         * Block: 256 threads per block
         */
        __global__ void int8ToFP16WithScale_A_kernel(
            const int8_t *__restrict__ A,        // [M × K] row-major
            const float *__restrict__ scaleA,    // [M] per-row scales
            __half *__restrict__ A_fp16,         // [M × K] row-major output
            int M,
            int K)
        {
            const int k = blockIdx.x * blockDim.x + threadIdx.x;
            const int m = blockIdx.y;

            if (m < M && k < K)
            {
                const size_t idx = static_cast<size_t>(m) * K + k;
                const float val = static_cast<float>(A[idx]) * scaleA[m];
                A_fp16[idx] = __float2half(val);
            }
        }

        /**
         * @brief Convert INT8 matrix B[K×N] to FP16 with scale fused
         *
         * Each element: out[k,n] = (half)(B[k,n] * scaleB[n])
         *
         * Grid: (N/256, K) threads
         * Block: 256 threads per block
         */
        __global__ void int8ToFP16WithScale_B_kernel(
            const int8_t *__restrict__ B,        // [K × N] row-major
            const float *__restrict__ scaleB,    // [N] per-column scales
            __half *__restrict__ B_fp16,         // [K × N] row-major output
            int K,
            int N)
        {
            const int n = blockIdx.x * blockDim.x + threadIdx.x;
            const int k = blockIdx.y;

            if (k < K && n < N)
            {
                const size_t idx = static_cast<size_t>(k) * N + n;
                const float val = static_cast<float>(B[idx]) * scaleB[n];
                B_fp16[idx] = __float2half(val);
            }
        }

        /**
         * @brief Convert FP16 GEMM result to FP32 (no scaling needed)
         *
         * Simple element-wise conversion since scales were already fused
         * into the INT8→FP16 conversion.
         *
         * Grid: (total_elements/256) threads
         * Block: 256 threads per block
         */
        __global__ void fp16ToFP32_kernel(
            const __half *__restrict__ C_fp16,
            float *__restrict__ E_fp32,
            size_t total_elements)
        {
            const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (idx < total_elements)
            {
                E_fp32[idx] = __half2float(C_fp16[idx]);
            }
        }

        // =====================================================================
        // FP16 GEMM Pipeline
        // =====================================================================

        /**
         * @brief Execute INT8→FP16→GEMM→FP32 pipeline
         *
         * This function:
         *   1. Converts A[M×K] INT8 → FP16
         *   2. Converts B[K×N] INT8 → FP16
         *   3. Calls hipBLAS hgemm: C_fp16 = A_fp16 @ B_fp16
         *   4. Converts C_fp16 → E_fp32 with scaleA[m] * scaleB[n] applied
         *
         * Memory Requirements:
         *   - A_fp16: M × K × 2 bytes
         *   - B_fp16: K × N × 2 bytes
         *   - C_fp16: M × N × 2 bytes
         *   Total: 2*M*K + 2*K*N + 2*M*N bytes
         *
         * @param d_A        INT8 activations [M × K] on device
         * @param d_B        INT8 weights [K × N] on device (already transposed)
         * @param d_E        FP32 output [M × N] on device
         * @param d_scaleA   Per-row scale factors [M] on device
         * @param d_scaleB   Per-column scale factors [N] on device
         * @param M          Number of rows in A and output
         * @param N          Number of columns in B and output
         * @param K          Inner dimension (cols of A, rows of B)
         * @param device_id  ROCm device ID
         * @param stream     HIP stream (optional, nullptr for default)
         * @return true on success
         */
        bool rocmQuantGemm_executeFP16_impl(
            const int8_t *d_A,
            const int8_t *d_B,
            float *d_E,
            const float *d_scaleA,
            const float *d_scaleB,
            int M,
            int N,
            int K,
            int device_id,
            void *stream_ptr)
        {
            hipStream_t stream = static_cast<hipStream_t>(stream_ptr);
            
            // Set device
            HIP_CHECK_FP16(hipSetDevice(device_id));

            // =========================================================
            // Step 1: Allocate temporary buffers
            //
            // We need FP16 buffers for A, B, and C (GEMM result).
            // Scales are applied in the epilogue kernel during FP16→FP32.
            // =========================================================

            const size_t size_A_fp16 = static_cast<size_t>(M) * K * sizeof(__half);
            const size_t size_B_fp16 = static_cast<size_t>(K) * N * sizeof(__half);
            const size_t size_C_fp16 = static_cast<size_t>(M) * N * sizeof(__half);

            __half *d_A_fp16 = nullptr;
            __half *d_B_fp16 = nullptr;
            __half *d_C_fp16 = nullptr;

            HIP_CHECK_FP16(hipMalloc(&d_A_fp16, size_A_fp16));
            HIP_CHECK_FP16(hipMalloc(&d_B_fp16, size_B_fp16));
            HIP_CHECK_FP16(hipMalloc(&d_C_fp16, size_C_fp16));

            // Cleanup lambda
            auto cleanup = [&]()
            {
                if (d_A_fp16)
                    hipFree(d_A_fp16);
                if (d_B_fp16)
                    hipFree(d_B_fp16);
                if (d_C_fp16)
                    hipFree(d_C_fp16);
            };

            constexpr int BLOCK_SIZE = 256;

            // =========================================================
            // Step 2: Convert A [M×K] from INT8 to FP16 with scaleA fused
            //
            // A'[m,k] = A[m,k] * scaleA[m]
            // This keeps values in FP16 range and applies scales.
            // =========================================================

            {
                dim3 block(BLOCK_SIZE, 1, 1);
                dim3 grid((K + BLOCK_SIZE - 1) / BLOCK_SIZE, M, 1);
                int8ToFP16WithScale_A_kernel<<<grid, block, 0, stream>>>(
                    d_A, d_scaleA, d_A_fp16, M, K);
            }

            // =========================================================
            // Step 3: Convert B [K×N] from INT8 to FP16 with scaleB fused
            //
            // B'[k,n] = B[k,n] * scaleB[n]
            // =========================================================

            {
                dim3 block(BLOCK_SIZE, 1, 1);
                dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, K, 1);
                int8ToFP16WithScale_B_kernel<<<grid, block, 0, stream>>>(
                    d_B, d_scaleB, d_B_fp16, K, N);
            }

            // =========================================================
            // Step 4: hipBLAS hgemm (FP16 GEMM)
            // =========================================================
            //
            // We want: C = A @ B (row-major)
            // Where A is [M×K], B is [K×N], C is [M×N]
            //
            // hipBLAS is column-major, so we compute: C^T = B^T @ A^T
            // This means: hipblasHgemm(opB=N, opA=N, N, M, K, ...)
            //
            // Leading dimensions:
            //   lda = K (A is [M×K] row-major, stride K)
            //   ldb = N (B is [K×N] row-major, stride N)
            //   ldc = N (C is [M×N] row-major, stride N)

            hipblasHandle_t handle;
            hipblasStatus_t blas_status = hipblasCreate(&handle);
            if (blas_status != HIPBLAS_STATUS_SUCCESS)
            {
                LOG_ERROR("[FP16 GEMM] Failed to create hipBLAS handle");
                cleanup();
                return false;
            }

            __half alpha = __float2half(1.0f);
            __half beta = __float2half(0.0f);

            blas_status = hipblasHgemm(
                handle,
                HIPBLAS_OP_N, HIPBLAS_OP_N, // opB, opA
                N, M, K,                     // N, M, K
                reinterpret_cast<const hipblasHalf*>(&alpha),
                reinterpret_cast<const hipblasHalf*>(d_B_fp16), N, // B, ldb
                reinterpret_cast<const hipblasHalf*>(d_A_fp16), K, // A, lda
                reinterpret_cast<const hipblasHalf*>(&beta),
                reinterpret_cast<hipblasHalf*>(d_C_fp16), N); // C, ldc

            hipblasDestroy(handle);

            if (blas_status != HIPBLAS_STATUS_SUCCESS)
            {
                LOG_ERROR("[FP16 GEMM] hipblasHgemm failed: " << static_cast<int>(blas_status));
                cleanup();
                return false;
            }

            // =========================================================
            // Step 5: Convert FP16 to FP32 (scales already applied)
            //
            // Simple conversion since scaleA and scaleB were already fused
            // into the A and B conversion kernels in steps 2-3.
            // =========================================================

            {
                const size_t total = static_cast<size_t>(M) * N;
                int grid = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
                fp16ToFP32_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
                    d_C_fp16, d_E, total);
            }

            // Wait for completion
            HIP_CHECK_FP16(hipStreamSynchronize(stream ? stream : 0));

            // Cleanup
            cleanup();

            return true;
        }

        // =====================================================================
        // Benchmark Helper: Direct call with pre-allocated buffers
        // =====================================================================

        /**
         * @brief Execute FP16 GEMM with externally managed buffers
         *
         * For benchmarking, we want to avoid allocation overhead in the timing loop.
         * This version takes pre-allocated FP16 buffers.
         *
         * @param d_A_fp16   Pre-allocated FP16 A buffer [M × K]
         * @param d_B_fp16   Pre-allocated FP16 B buffer [K × N]
         * @param d_C_fp16   Pre-allocated FP16 C buffer [M × N]
         * @param d_E        FP32 output [M × N]
         * @param M, N, K    Dimensions
         * @param device_id  ROCm device ID
         * @param stream     HIP stream
         * @return true on success
         */
        bool rocmQuantGemm_executeFP16_kernel_only(
            const __half *d_A_fp16,
            const __half *d_B_fp16,
            __half *d_C_fp16,
            float *d_E,
            int M,
            int N,
            int K,
            int device_id,
            hipStream_t stream)
        {
            HIP_CHECK_FP16(hipSetDevice(device_id));

            // hipBLAS hgemm
            hipblasHandle_t handle;
            HIPBLAS_CHECK_FP16(hipblasCreate(&handle));

            __half alpha = __float2half(1.0f);
            __half beta = __float2half(0.0f);

            hipblasStatus_t status = hipblasHgemm(
                handle,
                HIPBLAS_OP_N, HIPBLAS_OP_N,
                N, M, K,
                reinterpret_cast<const hipblasHalf*>(&alpha),
                reinterpret_cast<const hipblasHalf*>(d_B_fp16), N,
                reinterpret_cast<const hipblasHalf*>(d_A_fp16), K,
                reinterpret_cast<const hipblasHalf*>(&beta),
                reinterpret_cast<hipblasHalf*>(d_C_fp16), N);

            hipblasDestroy(handle);

            if (status != HIPBLAS_STATUS_SUCCESS)
            {
                LOG_ERROR("[FP16 GEMM kernel] hipblasHgemm failed");
                return false;
            }

            // Convert to FP32 (scales already fused into A_fp16 and B_fp16)
            constexpr int BLOCK_SIZE = 256;
            const size_t total = static_cast<size_t>(M) * N;
            int grid = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
            fp16ToFP32_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(d_C_fp16, d_E, total);

            return true;
        }

        /**
         * @brief Convert INT8 data to FP16 with scales fused (for benchmarking)
         *
         * Standalone conversion function for timing conversion overhead separately.
         * Fuses scaleA and scaleB into the conversion.
         */
        bool rocmQuantGemm_convertToFP16(
            const int8_t *d_A,
            const int8_t *d_B,
            const float *d_scaleA,
            const float *d_scaleB,
            __half *d_A_fp16,
            __half *d_B_fp16,
            int M,
            int N,
            int K,
            hipStream_t stream)
        {
            constexpr int BLOCK_SIZE = 256;

            // Convert A with scaleA
            {
                dim3 block(BLOCK_SIZE, 1, 1);
                dim3 grid((K + BLOCK_SIZE - 1) / BLOCK_SIZE, M, 1);
                int8ToFP16WithScale_A_kernel<<<grid, block, 0, stream>>>(
                    d_A, d_scaleA, d_A_fp16, M, K);
            }

            // Convert B with scaleB
            {
                dim3 block(BLOCK_SIZE, 1, 1);
                dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, K, 1);
                int8ToFP16WithScale_B_kernel<<<grid, block, 0, stream>>>(
                    d_B, d_scaleB, d_B_fp16, K, N);
            }

            return true;
        }

    } // namespace rocm
} // namespace llaminar2

// =====================================================================
// Extern "C" API Functions
// =====================================================================
// These wrap the namespace-internal implementations to provide C linkage
// for external callers (tests, benchmarks, etc.)
// =====================================================================

/**
 * @brief Execute FP16 GEMM with externally managed workspace buffers
 *
 * This is the allocation-free variant for hot-path execution. The caller
 * provides pre-allocated FP16 buffers that are reused across calls.
 *
 * Memory requirements (per call):
 *   - d_A_fp16: M × K × sizeof(__half) = M × K × 2 bytes
 *   - d_B_fp16: K × N × sizeof(__half) = K × N × 2 bytes
 *   - d_C_fp16: M × N × sizeof(__half) = M × N × 2 bytes
 *
 * @param d_A        INT8 activations [M × K] on device
 * @param d_B        INT8 weights [K × N] on device (already transposed)
 * @param d_E        FP32 output [M × N] on device
 * @param d_scaleA   Per-row scale factors [M] on device
 * @param d_scaleB   Per-column scale factors [N] on device
 * @param d_A_fp16   Pre-allocated FP16 buffer for A [M × K] (__half*)
 * @param d_B_fp16   Pre-allocated FP16 buffer for B [K × N] (__half*)
 * @param d_C_fp16   Pre-allocated FP16 buffer for C [M × N] (__half*)
 * @param M, N, K    Matrix dimensions
 * @param device_id  ROCm device ID
 * @param stream     HIP stream (nullptr for default)
 * @return true on success
 */
extern "C" bool rocmQuantGemm_executeFP16_cached(
    const int8_t *d_A,
    const int8_t *d_B,
    float *d_E,
    const float *d_scaleA,
    const float *d_scaleB,
    void *d_A_fp16,   // __half*, passed as void* for C linkage
    void *d_B_fp16,   // __half*, passed as void* for C linkage
    void *d_C_fp16,   // __half*, passed as void* for C linkage
    int M,
    int N,
    int K,
    int device_id,
    void *stream)
{
    using namespace llaminar2::rocm;
    
    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    __half* A_fp16 = static_cast<__half*>(d_A_fp16);
    __half* B_fp16 = static_cast<__half*>(d_B_fp16);
    __half* C_fp16 = static_cast<__half*>(d_C_fp16);
    
    hipError_t err = hipSetDevice(device_id);
    if (err != hipSuccess) {
        LOG_ERROR("[FP16 GEMM cached] hipSetDevice failed: " << hipGetErrorString(err));
        return false;
    }

    // Step 1: Convert INT8 → FP16 with scales fused
    if (!rocmQuantGemm_convertToFP16(d_A, d_B, d_scaleA, d_scaleB,
                                      A_fp16, B_fp16, M, N, K, hip_stream)) {
        LOG_ERROR("[FP16 GEMM cached] Failed to convert to FP16");
        return false;
    }
    
    // Step 2: Execute FP16 GEMM with pre-allocated buffers
    if (!rocmQuantGemm_executeFP16_kernel_only(A_fp16, B_fp16, C_fp16, d_E,
                                                M, N, K, device_id, hip_stream)) {
        LOG_ERROR("[FP16 GEMM cached] Failed to execute FP16 GEMM");
        return false;
    }
    
    // Sync
    err = hipStreamSynchronize(hip_stream ? hip_stream : 0);
    if (err != hipSuccess) {
        LOG_ERROR("[FP16 GEMM cached] hipStreamSynchronize failed: " << hipGetErrorString(err));
        return false;
    }
    
    return true;
}

/**
 * @brief Allocate FP16 buffer on device
 *
 * Helper for allocating __half buffers from C++ code that doesn't include
 * HIP headers.
 *
 * @param d_ptr      Output pointer (void** for C linkage)
 * @param count      Number of elements (not bytes)
 * @param device_id  ROCm device ID
 * @return true on success
 */
extern "C" bool rocmQuantGemm_allocFP16(void **d_ptr, size_t count, int device_id)
{
    hipError_t err = hipSetDevice(device_id);
    if (err != hipSuccess) {
        LOG_ERROR("[FP16 alloc] hipSetDevice failed");
        return false;
    }
    
    err = hipMalloc(d_ptr, count * sizeof(__half));
    if (err != hipSuccess) {
        LOG_ERROR("[FP16 alloc] hipMalloc failed for " << count << " FP16 elements");
        return false;
    }
    
    return true;
}

extern "C" bool rocmQuantGemm_executeFP16(
    const int8_t *d_A,
    const int8_t *d_B,
    float *d_E,
    const float *d_scaleA,
    const float *d_scaleB,
    int M,
    int N,
    int K,
    int device_id,
    void *stream)
{
    return llaminar2::rocm::rocmQuantGemm_executeFP16_impl(
        d_A, d_B, d_E, d_scaleA, d_scaleB, M, N, K, device_id, stream);
}
