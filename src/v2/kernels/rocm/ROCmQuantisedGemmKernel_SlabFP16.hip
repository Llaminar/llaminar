/**
 * @file ROCmQuantisedGemmKernel_SlabFP16.hip
 * @brief Slab-based chunked FP16 GEMM for memory-constrained ROCm GPUs
 *
 * This file implements a memory-bounded FP16 GEMM path that converts INT8 data
 * to FP16 in fixed-size "slabs" rather than converting entire matrices at once.
 *
 * ============================================================================
 * MOTIVATION
 * ============================================================================
 *
 * For large models (7B+), full FP16 conversion requires significant memory:
 *
 *   Example: 7B FFN layer (K=3584, N=18944, M=512)
 *     A_fp16: 512 × 3584 × 2 = 3.5 MB
 *     B_fp16: 3584 × 18944 × 2 = 136 MB  ← Problem!
 *     C_fp16: 512 × 18944 × 2 = 19.4 MB
 *     Total: ~159 MB
 *
 * On memory-constrained GPUs (MI50 16GB shared with weights + KV cache),
 * this can cause OOM or force suboptimal batch sizes.
 *
 * Solution: Convert and compute in slabs with fixed memory budget.
 *
 *   Example: 64MB budget → slab_m=256, slab_n=4096, slab_k=512
 *     slab_a_fp16: 256 × 512 × 2 = 256 KB
 *     slab_b_fp16: 512 × 4096 × 2 = 4 MB
 *     slab_c_fp16: 256 × 4096 × 2 = 2 MB
 *     Total: ~6.3 MB (25× reduction!)
 *
 * Trade-off: 70 slab GEMMs vs 1 large GEMM
 *
 * ============================================================================
 * ALGORITHM
 * ============================================================================
 *
 * For C[M×N] = A[M×K] × B[K×N]:
 *
 *   Initialize FP32 accumulator C_fp32 = 0
 *
 *   For k_start = 0 to K step slab_k:
 *     k_end = min(k_start + slab_k, K)
 *
 *     For n_start = 0 to N step slab_n:
 *       n_end = min(n_start + slab_n, N)
 *
 *       // Convert B slab: INT8 → FP16 with scaleB
 *       kernel: slab_b_fp16[k',n'] = B[k_start+k', n_start+n'] * scaleB[n_start+n']
 *
 *       For m_start = 0 to M step slab_m:
 *         m_end = min(m_start + slab_m, M)
 *
 *         // Convert A slab: INT8 → FP16 with scaleA
 *         kernel: slab_a_fp16[m',k'] = A[m_start+m', k_start+k'] * scaleA[m_start+m']
 *
 *         // FP16 GEMM
 *         slab_c_fp16 = hipblasHgemm(slab_a_fp16, slab_b_fp16)
 *
 *         // Accumulate to FP32 output (handles k-dimension accumulation)
 *         kernel: C_fp32[m_start+m', n_start+n'] += fp16_to_fp32(slab_c_fp16[m',n'])
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include "utils/Logger.h"
#include "kernels/SlabGemmConfig.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hipblas/hipblas.h>

#include <cstdint>
#include <cstring>

namespace llaminar2
{
    namespace rocm
    {
        namespace slab_fp16
        {

// ============================================================================
// Helper macros
// ============================================================================

#define HIP_CHECK_SLAB(call)                                                      \
    do                                                                            \
    {                                                                             \
        hipError_t err = call;                                                    \
        if (err != hipSuccess)                                                    \
        {                                                                         \
            LOG_ERROR("[Slab FP16] HIP error: " << hipGetErrorString(err)         \
                                                << " at " << __FILE__             \
                                                << ":" << __LINE__);              \
            return false;                                                         \
        }                                                                         \
    } while (0)

#define HIPBLAS_CHECK_SLAB(call)                                                  \
    do                                                                            \
    {                                                                             \
        hipblasStatus_t status = call;                                            \
        if (status != HIPBLAS_STATUS_SUCCESS)                                     \
        {                                                                         \
            LOG_ERROR("[Slab FP16] hipBLAS error: " << static_cast<int>(status)   \
                                                    << " at " << __FILE__         \
                                                    << ":" << __LINE__);          \
            return false;                                                         \
        }                                                                         \
    } while (0)

// ============================================================================
// Conversion Kernels for Slab Extraction
// ============================================================================

/**
 * @brief Convert INT8 A sub-matrix to FP16 slab with scale
 *
 * Extracts A[m_start:m_end, k_start:k_end] and converts to FP16.
 * Applies per-row scale during conversion: out[m',k'] = A[m,k] * scaleA[m]
 *
 * Grid: (ceil(actual_k/256), actual_m)
 * Block: 256 threads
 *
 * @param A_int8      Full INT8 A matrix [M × K] row-major
 * @param scaleA      Per-row scales [M]
 * @param slab_a_fp16 Output FP16 slab [actual_m × actual_k] row-major
 * @param full_M      Full matrix M dimension
 * @param full_K      Full matrix K dimension
 * @param m_start     Row offset in A
 * @param k_start     Column offset in A
 * @param actual_m    Slab M dimension (rows to extract)
 * @param actual_k    Slab K dimension (cols to extract)
 */
__global__ void extractSlabA_int8ToFP16_kernel(
    const int8_t *__restrict__ A_int8,
    const float *__restrict__ scaleA,
    __half *__restrict__ slab_a_fp16,
    int full_M,
    int full_K,
    int m_start,
    int k_start,
    int actual_m,
    int actual_k)
{
    const int local_k = blockIdx.x * blockDim.x + threadIdx.x;
    const int local_m = blockIdx.y;

    if (local_m < actual_m && local_k < actual_k)
    {
        const int global_m = m_start + local_m;
        const int global_k = k_start + local_k;

        // Bounds check (should always pass if actual_m/k calculated correctly)
        if (global_m < full_M && global_k < full_K)
        {
            // Read from full A matrix
            const size_t src_idx = static_cast<size_t>(global_m) * full_K + global_k;
            const float val = static_cast<float>(A_int8[src_idx]) * scaleA[global_m];

            // Write to slab (contiguous row-major)
            const size_t dst_idx = static_cast<size_t>(local_m) * actual_k + local_k;
            slab_a_fp16[dst_idx] = __float2half(val);
        }
    }
}

/**
 * @brief Convert INT8 B sub-matrix to FP16 slab with scale
 *
 * Extracts B[k_start:k_end, n_start:n_end] and converts to FP16.
 * Applies per-column scale during conversion: out[k',n'] = B[k,n] * scaleB[n]
 *
 * Grid: (ceil(actual_n/256), actual_k)
 * Block: 256 threads
 *
 * @param B_int8      Full INT8 B matrix [K × N] row-major (transposed weights)
 * @param scaleB      Per-column scales [N]
 * @param slab_b_fp16 Output FP16 slab [actual_k × actual_n] row-major
 * @param full_K      Full matrix K dimension
 * @param full_N      Full matrix N dimension
 * @param k_start     Row offset in B
 * @param n_start     Column offset in B
 * @param actual_k    Slab K dimension (rows to extract)
 * @param actual_n    Slab N dimension (cols to extract)
 */
__global__ void extractSlabB_int8ToFP16_kernel(
    const int8_t *__restrict__ B_int8,
    const float *__restrict__ scaleB,
    __half *__restrict__ slab_b_fp16,
    int full_K,
    int full_N,
    int k_start,
    int n_start,
    int actual_k,
    int actual_n)
{
    const int local_n = blockIdx.x * blockDim.x + threadIdx.x;
    const int local_k = blockIdx.y;

    if (local_k < actual_k && local_n < actual_n)
    {
        const int global_k = k_start + local_k;
        const int global_n = n_start + local_n;

        // Bounds check
        if (global_k < full_K && global_n < full_N)
        {
            // Read from full B matrix
            const size_t src_idx = static_cast<size_t>(global_k) * full_N + global_n;
            const float val = static_cast<float>(B_int8[src_idx]) * scaleB[global_n];

            // Write to slab (contiguous row-major)
            const size_t dst_idx = static_cast<size_t>(local_k) * actual_n + local_n;
            slab_b_fp16[dst_idx] = __float2half(val);
        }
    }
}

/**
 * @brief Accumulate FP16 slab C to FP32 output sub-matrix
 *
 * Converts FP16 slab result to FP32 and ADDS to output:
 *   C_fp32[m_start+m', n_start+n'] += fp16_to_fp32(slab_c_fp16[m',n'])
 *
 * This is used to accumulate partial sums across the K dimension.
 *
 * Grid: (ceil(actual_n/256), actual_m)
 * Block: 256 threads
 *
 * @param slab_c_fp16 Input FP16 slab [actual_m × actual_n] row-major
 * @param C_fp32      Output FP32 matrix [M × N] row-major
 * @param full_N      Full output N dimension (stride)
 * @param m_start     Row offset in C
 * @param n_start     Column offset in C
 * @param actual_m    Slab M dimension
 * @param actual_n    Slab N dimension
 */
__global__ void accumulateSlabC_fp16ToFP32_kernel(
    const __half *__restrict__ slab_c_fp16,
    float *__restrict__ C_fp32,
    int full_N,
    int m_start,
    int n_start,
    int actual_m,
    int actual_n)
{
    const int local_n = blockIdx.x * blockDim.x + threadIdx.x;
    const int local_m = blockIdx.y;

    if (local_m < actual_m && local_n < actual_n)
    {
        const int global_m = m_start + local_m;
        const int global_n = n_start + local_n;

        // Read FP16 from slab
        const size_t src_idx = static_cast<size_t>(local_m) * actual_n + local_n;
        const float val = __half2float(slab_c_fp16[src_idx]);

        // Accumulate to FP32 output
        const size_t dst_idx = static_cast<size_t>(global_m) * full_N + global_n;
        C_fp32[dst_idx] += val;
    }
}

/**
 * @brief Zero-initialize FP32 output matrix
 *
 * Must be called before slab accumulation to ensure correct results.
 *
 * Grid: (ceil(total_elements/256))
 * Block: 256 threads
 */
__global__ void zeroInitFP32_kernel(
    float *__restrict__ C_fp32,
    size_t total_elements)
{
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < total_elements)
    {
        C_fp32[idx] = 0.0f;
    }
}

        } // namespace slab_fp16
    } // namespace rocm
} // namespace llaminar2

// ============================================================================
// Extern "C" API
// ============================================================================

using namespace llaminar2::rocm::slab_fp16;

/**
 * @brief Execute slab-based INT8→FP16 GEMM with fixed workspace
 *
 * This is the main entry point for memory-bounded FP16 GEMM on ROCm.
 * Converts INT8 data to FP16 in slabs and accumulates FP32 result.
 *
 * @param d_A_int8     INT8 activations [M × K] on device
 * @param d_B_int8     INT8 weights [K × N] on device (transposed)
 * @param d_C_fp32     FP32 output [M × N] on device
 * @param d_scaleA     Per-row scales [M] on device
 * @param d_scaleB     Per-column scales [N] on device
 * @param M, N, K      Full GEMM dimensions
 * @param slab_a_fp16  Workspace for A slab [slab_m × slab_k] (__half*)
 * @param slab_b_fp16  Workspace for B slab [slab_k × slab_n] (__half*)
 * @param slab_c_fp16  Workspace for C slab [slab_m × slab_n] (__half*)
 * @param config       Slab configuration (dimensions)
 * @param device_id    ROCm device ID
 * @param stream       HIP stream (nullptr for default)
 * @return true on success
 */
extern "C" bool rocmQuantGemm_executeSlabFP16(
    const int8_t *d_A_int8,
    const int8_t *d_B_int8,
    float *d_C_fp32,
    const float *d_scaleA,
    const float *d_scaleB,
    int M, int N, int K,
    void *slab_a_fp16,
    void *slab_b_fp16,
    void *slab_c_fp16,
    const llaminar2::SlabGemmConfig *config,
    int device_id,
    void *stream_ptr)
{
    using namespace llaminar2;

    if (!config || !config->isValid())
    {
        LOG_ERROR("[Slab FP16] Invalid slab configuration");
        return false;
    }

    if (!d_A_int8 || !d_B_int8 || !d_C_fp32 || !d_scaleA || !d_scaleB)
    {
        LOG_ERROR("[Slab FP16] Null input pointer");
        return false;
    }

    if (!slab_a_fp16 || !slab_b_fp16 || !slab_c_fp16)
    {
        LOG_ERROR("[Slab FP16] Null workspace pointer");
        return false;
    }

    hipStream_t stream = static_cast<hipStream_t>(stream_ptr);
    HIP_CHECK_SLAB(hipSetDevice(device_id));

    __half *slab_a = static_cast<__half *>(slab_a_fp16);
    __half *slab_b = static_cast<__half *>(slab_b_fp16);
    __half *slab_c = static_cast<__half *>(slab_c_fp16);

    const int slab_m = config->slab_m;
    const int slab_n = config->slab_n;
    const int slab_k = config->slab_k;

    constexpr int BLOCK_SIZE = 256;

    // =========================================================================
    // Step 1: Zero-initialize FP32 output
    // =========================================================================

    {
        const size_t total = static_cast<size_t>(M) * N;
        int grid = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
        zeroInitFP32_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(d_C_fp32, total);
    }

    // =========================================================================
    // Step 2: Create hipBLAS handle
    // =========================================================================

    hipblasHandle_t handle;
    hipblasStatus_t blas_status = hipblasCreate(&handle);
    if (blas_status != HIPBLAS_STATUS_SUCCESS)
    {
        LOG_ERROR("[Slab FP16] Failed to create hipBLAS handle");
        return false;
    }

    if (stream)
    {
        hipblasSetStream(handle, stream);
    }

    __half alpha = __float2half(1.0f);
    __half beta = __float2half(0.0f);

    bool success = true;

    // =========================================================================
    // Step 3: Slab loop
    //
    // Loop order: K (outer) → N → M (inner)
    // This maximizes B slab reuse across M iterations.
    // =========================================================================

    LOG_DEBUG("[Slab FP16] Starting slab GEMM: M=" << M << ", N=" << N << ", K=" << K
              << ", slab_m=" << slab_m << ", slab_n=" << slab_n << ", slab_k=" << slab_k
              << ", iterations=" << config->estimateIterations(M, N, K));

    int total_gemm_calls = 0;

    for (int k_start = 0; k_start < K && success; k_start += slab_k)
    {
        const int actual_k = std::min(slab_k, K - k_start);

        for (int n_start = 0; n_start < N && success; n_start += slab_n)
        {
            const int actual_n = std::min(slab_n, N - n_start);

            // -----------------------------------------------------------------
            // Convert B slab: B[k_start:k_end, n_start:n_end] → slab_b_fp16
            // -----------------------------------------------------------------
            {
                dim3 block(BLOCK_SIZE, 1, 1);
                dim3 grid((actual_n + BLOCK_SIZE - 1) / BLOCK_SIZE, actual_k, 1);

                extractSlabB_int8ToFP16_kernel<<<grid, block, 0, stream>>>(
                    d_B_int8, d_scaleB, slab_b,
                    K, N,
                    k_start, n_start,
                    actual_k, actual_n);
            }

            for (int m_start = 0; m_start < M && success; m_start += slab_m)
            {
                const int actual_m = std::min(slab_m, M - m_start);

                // -------------------------------------------------------------
                // Convert A slab: A[m_start:m_end, k_start:k_end] → slab_a_fp16
                // -------------------------------------------------------------
                {
                    dim3 block(BLOCK_SIZE, 1, 1);
                    dim3 grid((actual_k + BLOCK_SIZE - 1) / BLOCK_SIZE, actual_m, 1);

                    extractSlabA_int8ToFP16_kernel<<<grid, block, 0, stream>>>(
                        d_A_int8, d_scaleA, slab_a,
                        M, K,
                        m_start, k_start,
                        actual_m, actual_k);
                }

                // -------------------------------------------------------------
                // FP16 GEMM: slab_c = slab_a × slab_b
                //
                // hipBLAS is column-major, so we compute C^T = B^T × A^T
                // This means: hipblasHgemm(opB=N, opA=N, N', M', K', ...)
                // where N'=actual_n, M'=actual_m, K'=actual_k
                //
                // Leading dimensions for row-major inputs:
                //   lda = actual_k (A slab is [actual_m × actual_k])
                //   ldb = actual_n (B slab is [actual_k × actual_n])
                //   ldc = actual_n (C slab is [actual_m × actual_n])
                // -------------------------------------------------------------
                {
                    hipblasStatus_t status = hipblasHgemm(
                        handle,
                        HIPBLAS_OP_N, HIPBLAS_OP_N,       // opB, opA
                        actual_n, actual_m, actual_k,     // N', M', K'
                        reinterpret_cast<const hipblasHalf *>(&alpha),
                        reinterpret_cast<const hipblasHalf *>(slab_b), actual_n, // B, ldb
                        reinterpret_cast<const hipblasHalf *>(slab_a), actual_k, // A, lda
                        reinterpret_cast<const hipblasHalf *>(&beta),
                        reinterpret_cast<hipblasHalf *>(slab_c), actual_n); // C, ldc

                    if (status != HIPBLAS_STATUS_SUCCESS)
                    {
                        LOG_ERROR("[Slab FP16] hipblasHgemm failed at slab ("
                                  << m_start << "," << n_start << "," << k_start << ")");
                        success = false;
                        break;
                    }
                }

                // -------------------------------------------------------------
                // Accumulate slab_c to C_fp32[m_start:m_end, n_start:n_end]
                // -------------------------------------------------------------
                {
                    dim3 block(BLOCK_SIZE, 1, 1);
                    dim3 grid((actual_n + BLOCK_SIZE - 1) / BLOCK_SIZE, actual_m, 1);

                    accumulateSlabC_fp16ToFP32_kernel<<<grid, block, 0, stream>>>(
                        slab_c, d_C_fp32,
                        N,
                        m_start, n_start,
                        actual_m, actual_n);
                }

                total_gemm_calls++;
            }
        }
    }

    hipblasDestroy(handle);

    // Wait for completion
    HIP_CHECK_SLAB(hipStreamSynchronize(stream ? stream : 0));

    LOG_DEBUG("[Slab FP16] Completed " << total_gemm_calls << " slab GEMMs");

    return success;
}

/**
 * @brief Check if slab FP16 GEMM should be used instead of full FP16
 *
 * Decision criteria:
 *   1. Full FP16 would exceed available workspace budget
 *   2. User forced slab mode via environment variable
 *   3. Matrix is large enough to benefit from chunking
 *
 * @param M, N, K          Full GEMM dimensions
 * @param workspace_budget Available workspace bytes
 * @return true if slab GEMM should be used
 */
extern "C" bool rocmQuantGemm_shouldUseSlabFP16(
    int M, int N, int K,
    size_t workspace_budget)
{
    // Check environment variable for force slab mode
    static bool checked_env = false;
    static bool force_slab = false;
    if (!checked_env)
    {
        const char *env = std::getenv("LLAMINAR_ROCM_GEMM_FORCE_SLAB");
        force_slab = (env && env[0] == '1');
        checked_env = true;
    }

    if (force_slab)
    {
        return true;
    }

    // Calculate full FP16 memory requirement
    const size_t a_fp16_bytes = static_cast<size_t>(M) * K * 2;
    const size_t b_fp16_bytes = static_cast<size_t>(K) * N * 2;
    const size_t c_fp16_bytes = static_cast<size_t>(M) * N * 2;
    const size_t total_full = a_fp16_bytes + b_fp16_bytes + c_fp16_bytes;

    // Use slab if full conversion exceeds 80% of budget
    // (leave headroom for other workspace needs)
    return total_full > (workspace_budget * 4 / 5);
}

/**
 * @brief Get optimal slab configuration for given dimensions and budget
 *
 * @param M, N, K          Full GEMM dimensions
 * @param workspace_budget Available workspace bytes (0 = auto 64MB)
 * @param out_config       Output configuration
 */
extern "C" void rocmQuantGemm_getSlabConfig(
    int M, int N, int K,
    size_t workspace_budget,
    llaminar2::SlabGemmConfig *out_config)
{
    if (!out_config)
        return;

    *out_config = llaminar2::SlabGemmConfig::fromBudget(
        workspace_budget, M, N, K, llaminar2::SlabDataType::FP16);
}
