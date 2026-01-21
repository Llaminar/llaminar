/**
 * @file CUDAQuantisedGemmKernel_CUTLASS.cu
 * @brief CUTLASS INT8 GEMM implementation for CUDAQuantisedGemmKernel
 *
 * This file contains the actual CUDA kernel implementations:
 * - CUTLASS INT8×INT8→INT32 GEMM using Tensor Cores
 * - Activation quantization (FP32→INT8)
 * - Output scaling (INT32→FP32)
 * - Memory management utilities
 *
 * **CUTLASS Configuration**:
 * - ThreadBlock: 128×128×64 (optimized for Tensor Core tiles)
 * - Warp: 64×64×64 (must align with Tensor Core)
 * - Instruction: 16×8×32 (mma.sync.m16n8k32.s8.s8.s32)
 * - Pipeline stages: 3 (overlapped loads/compute)
 *
 * **Memory Layout Requirements**:
 * - A: INT8 [M×K] RowMajor
 * - B: INT8 [K×N] ColumnMajor (Tensor Core requirement!)
 * - C: INT32 [M×N] RowMajor
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <iostream>
#include <cmath>
#include <algorithm>

// =========================================================================
// CUTLASS GEMM Type Definition
// =========================================================================

// CUTLASS INT8 GEMM using Tensor Cores (SM 8.0+ Ampere/Ada/Hopper)
// Uses dp4a instruction for int8×int8 dot product with int32 accumulation
using CutlassInt8Gemm = cutlass::gemm::device::Gemm<
    int8_t,                                 // ElementA
    cutlass::layout::RowMajor,              // LayoutA
    int8_t,                                 // ElementB
    cutlass::layout::ColumnMajor,           // LayoutB (MUST be ColumnMajor for Tensor Cores!)
    int32_t,                                // ElementOutput (accumulator)
    cutlass::layout::RowMajor,              // LayoutC
    int32_t,                                // ElementAccumulator
    cutlass::arch::OpClassTensorOp,         // OpClass (Tensor Cores)
    cutlass::arch::Sm80,                    // ArchTag (Ampere SM 8.0+)
    cutlass::gemm::GemmShape<128, 128, 64>, // ThreadblockShape
    cutlass::gemm::GemmShape<64, 64, 64>,   // WarpShape
    cutlass::gemm::GemmShape<16, 8, 32>,    // InstructionShape (mma.sync.m16n8k32)
    cutlass::epilogue::thread::LinearCombination<
        int32_t, 1, int32_t, int32_t>,                            // EpilogueOp (passthrough)
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, // Swizzle
    3                                                             // Pipeline stages
    >;

// =========================================================================
// CUDA Error Checking Macros
// =========================================================================

#include <stdexcept>
#include <sstream>

/**
 * @brief CUDA error check that throws on failure
 *
 * CUDA errors should never be silently ignored - they indicate serious problems
 * that will cascade if execution continues. Throwing ensures:
 * 1. Immediate failure with clear error message
 * 2. Stack trace in debug builds
 * 3. No corrupted state from partial execution
 */
#define CUDA_CHECK_THROW(call)                                               \
    do                                                                       \
    {                                                                        \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess)                                              \
        {                                                                    \
            std::ostringstream oss;                                          \
            oss << "[CUDAQuantGemm] CUDA error: " << cudaGetErrorString(err) \
                << " at " << __FILE__ << ":" << __LINE__;                    \
            throw std::runtime_error(oss.str());                             \
        }                                                                    \
    } while (0)

// Legacy macro - returns false (being phased out)
#define CUDA_CHECK(call)                                                           \
    do                                                                             \
    {                                                                              \
        cudaError_t err = call;                                                    \
        if (err != cudaSuccess)                                                    \
        {                                                                          \
            std::cerr << "[CUDAQuantGemm] CUDA error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";            \
            return false;                                                          \
        }                                                                          \
    } while (0)

#define CUDA_CHECK_VOID(call)                                                      \
    do                                                                             \
    {                                                                              \
        cudaError_t err = call;                                                    \
        if (err != cudaSuccess)                                                    \
        {                                                                          \
            std::cerr << "[CUDAQuantGemm] CUDA error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";            \
            return;                                                                \
        }                                                                          \
    } while (0)

// =========================================================================
// CUDA Kernels
// =========================================================================

namespace
{

    /**
     * @brief Quantize FP32 activations to INT8 (symmetric per-row quantization)
     *
     * Each block processes one row:
     * 1. Find max_abs across row (parallel reduction)
     * 2. Compute scale = max_abs / 127
     * 3. Quantize: A_int8[j] = round(A_fp32[j] / scale)
     *
     * Grid: (M, 1, 1) - one block per row
     * Block: (min(K, 256), 1, 1)
     */
    __global__ void quantize_activations_kernel(
        const float *A_fp32, // [M×K]
        int8_t *A_int8,      // [M×K] output
        float *scales_A,     // [M] output
        int M, int K)
    {
        int row = blockIdx.x;
        if (row >= M)
            return;

        const float *row_fp32 = A_fp32 + row * K;
        int8_t *row_int8 = A_int8 + row * K;

        // Phase 1: Find max_abs (parallel reduction in shared memory)
        __shared__ float shared_max[256];

        float local_max = 0.0f;
        for (int j = threadIdx.x; j < K; j += blockDim.x)
        {
            local_max = fmaxf(local_max, fabsf(row_fp32[j]));
        }
        shared_max[threadIdx.x] = local_max;
        __syncthreads();

        // Reduce to thread 0
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
        {
            if (threadIdx.x < stride)
            {
                shared_max[threadIdx.x] = fmaxf(shared_max[threadIdx.x],
                                                shared_max[threadIdx.x + stride]);
            }
            __syncthreads();
        }

        float max_abs = shared_max[0];
        float scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;

        if (threadIdx.x == 0)
        {
            scales_A[row] = scale;
        }
        __syncthreads();

        // Phase 2: Quantize row
        float inv_scale = 1.0f / scale;
        for (int j = threadIdx.x; j < K; j += blockDim.x)
        {
            float val = row_fp32[j] * inv_scale;
            int8_t quantized = (int8_t)rintf(fminf(127.0f, fmaxf(-127.0f, val)));
            row_int8[j] = quantized;
        }
    }

    /**
     * @brief Apply output scaling: C_fp32 = alpha * C_int32 * scales_A * scales_B + beta * C_existing + bias
     *
     * Grid: (ceil(N/16), ceil(M/16), 1)
     * Block: (16, 16, 1)
     */
    __global__ void apply_scaling_kernel(
        const int32_t *C_int32, // [M×N]
        float *C_fp32,          // [M×N] output
        const float *scales_A,  // [M] row scales
        const float *scales_B,  // [N] column scales
        int M, int N,
        float alpha, float beta,
        const float *C_existing, // For beta != 0
        const float *bias)       // [N] optional bias, broadcasted across rows
    {
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        int row = blockIdx.y * blockDim.y + threadIdx.y;

        if (row < M && col < N)
        {
            int idx = row * N + col;
            int32_t val_int32 = C_int32[idx];
            float scale_combined = scales_A[row] * scales_B[col];
            float result = alpha * (float)val_int32 * scale_combined;

            if (beta != 0.0f && C_existing != nullptr)
            {
                result += beta * C_existing[idx];
            }

            if (bias != nullptr)
            {
                result += bias[col];
            }

            C_fp32[idx] = result;
        }
    }

} // anonymous namespace

// =========================================================================
// Extern "C" API for .cpp adapter
// =========================================================================

extern "C"
{

    /**
     * @brief Upload converted INT8 weights to device
     */
    bool cudaQuantGemm_uploadWeights(
        const int8_t *h_weights_int8,
        const float *h_scales_B,
        int8_t **d_weights_int8,
        float **d_scales_B,
        int K, int N,
        int cuda_device_id)
    {
        CUDA_CHECK(cudaSetDevice(cuda_device_id));

        // Free existing if any
        if (*d_weights_int8)
        {
            cudaFree(*d_weights_int8);
            *d_weights_int8 = nullptr;
        }
        if (*d_scales_B)
        {
            cudaFree(*d_scales_B);
            *d_scales_B = nullptr;
        }

        // Allocate and upload weights [K x N] ColumnMajor
        size_t weights_bytes = static_cast<size_t>(K) * static_cast<size_t>(N) * sizeof(int8_t);
        CUDA_CHECK(cudaMalloc(d_weights_int8, weights_bytes));
        CUDA_CHECK(cudaMemcpy(*d_weights_int8, h_weights_int8, weights_bytes, cudaMemcpyHostToDevice));

        // Allocate and upload scales [N]
        size_t scales_bytes = static_cast<size_t>(N) * sizeof(float);
        CUDA_CHECK(cudaMalloc(d_scales_B, scales_bytes));
        CUDA_CHECK(cudaMemcpy(*d_scales_B, h_scales_B, scales_bytes, cudaMemcpyHostToDevice));

        return true;
    }

    /**
     * @brief Ensure work buffers are allocated for given M
     */
    bool cudaQuantGemm_ensureWorkBuffers(
        int8_t **d_A_int8,
        float **d_scales_A,
        int32_t **d_C_int32,
        int *work_buffer_M,
        int M, int K, int N,
        int cuda_device_id)
    {
        if (M <= *work_buffer_M)
        {
            return true; // Already have enough capacity
        }

        CUDA_CHECK(cudaSetDevice(cuda_device_id));

        // Free existing
        if (*d_A_int8)
        {
            cudaFree(*d_A_int8);
        }
        if (*d_scales_A)
        {
            cudaFree(*d_scales_A);
        }
        if (*d_C_int32)
        {
            cudaFree(*d_C_int32);
        }

        // Allocate new buffers with 2x headroom for growth
        int alloc_M = M * 2;

        CUDA_CHECK(cudaMalloc(d_A_int8, static_cast<size_t>(alloc_M) * K * sizeof(int8_t)));
        CUDA_CHECK(cudaMalloc(d_scales_A, static_cast<size_t>(alloc_M) * sizeof(float)));
        CUDA_CHECK(cudaMalloc(d_C_int32, static_cast<size_t>(alloc_M) * N * sizeof(int32_t)));

        *work_buffer_M = alloc_M;
        return true;
    }

    /**
     * @brief Execute CUTLASS INT8 GEMM
     */
    bool cudaQuantGemm_execute(
        const int8_t *d_A_int8,
        const int8_t *d_weights_int8,
        int32_t *d_C_int32,
        int M, int N, int K,
        int cuda_device_id)
    {
        // Validate pointers
        if (!d_A_int8 || !d_weights_int8 || !d_C_int32)
        {
            std::ostringstream oss;
            oss << "[CUDAQuantGemm::execute] Null pointer: "
                << "d_A_int8=" << (void *)d_A_int8
                << " d_weights_int8=" << (void *)d_weights_int8
                << " d_C_int32=" << (void *)d_C_int32;
            throw std::runtime_error(oss.str());
        }

        CUDA_CHECK_THROW(cudaSetDevice(cuda_device_id));

        CutlassInt8Gemm gemm_op;

        // CUTLASS problem shape: M × N × K
        // A: [M × K] RowMajor, stride = K
        // B: [K × N] ColumnMajor, stride = K (leading dimension is number of rows = K)
        // C: [M × N] RowMajor, stride = N
        typename CutlassInt8Gemm::Arguments args(
            {M, N, K},           // Problem size
            {d_A_int8, K},       // TensorRef A (RowMajor, stride = K)
            {d_weights_int8, K}, // TensorRef B (ColumnMajor, stride = K) [FIXED: was N]
            {d_C_int32, N},      // TensorRef C (RowMajor, stride = N)
            {d_C_int32, N},      // TensorRef D (output, RowMajor, stride = N)
            {1, 0}               // alpha=1, beta=0
        );

        // Check if CUTLASS can execute this problem
        cutlass::Status can_status = CutlassInt8Gemm::can_implement(args);
        if (can_status != cutlass::Status::kSuccess)
        {
            std::ostringstream oss;
            oss << "[CUDAQuantGemm] CUTLASS can_implement failed with status " << static_cast<int>(can_status)
                << " (M=" << M << ", N=" << N << ", K=" << K << ")";
            throw std::runtime_error(oss.str());
        }

        cutlass::Status status = gemm_op(args);

        if (status != cutlass::Status::kSuccess)
        {
            std::ostringstream oss;
            oss << "[CUDAQuantGemm] CUTLASS GEMM failed with status " << static_cast<int>(status)
                << " (M=" << M << ", N=" << N << ", K=" << K << ")";
            throw std::runtime_error(oss.str());
        }

        // Synchronize to catch any async errors from the kernel
        cudaError_t sync_err = cudaDeviceSynchronize();
        if (sync_err != cudaSuccess)
        {
            std::ostringstream oss;
            oss << "[CUDAQuantGemm] CUTLASS kernel async error: " << cudaGetErrorString(sync_err)
                << " (M=" << M << ", N=" << N << ", K=" << K << ")";
            throw std::runtime_error(oss.str());
        }

        return true;
    }

    /**
     * @brief Apply output scaling with optional bias
     */
    bool cudaQuantGemm_applyScaling(
        const int32_t *d_C_int32,
        float *d_C_fp32,
        const float *d_scales_A,
        const float *d_scales_B,
        int M, int N,
        float alpha, float beta,
        const float *d_C_existing,
        const float *d_bias,
        int cuda_device_id)
    {
        // Validate pointers before launching kernel
        if (!d_C_int32 || !d_C_fp32 || !d_scales_A || !d_scales_B)
        {
            std::ostringstream oss;
            oss << "[CUDAQuantGemm::applyScaling] Null pointer: "
                << "d_C_int32=" << (void *)d_C_int32
                << " d_C_fp32=" << (void *)d_C_fp32
                << " d_scales_A=" << (void *)d_scales_A
                << " d_scales_B=" << (void *)d_scales_B;
            throw std::runtime_error(oss.str());
        }

        CUDA_CHECK_THROW(cudaSetDevice(cuda_device_id));

        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (M + 15) / 16);

        apply_scaling_kernel<<<grid, block>>>(
            d_C_int32, d_C_fp32, d_scales_A, d_scales_B,
            M, N, alpha, beta, d_C_existing, d_bias);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            std::ostringstream oss;
            oss << "[CUDAQuantGemm] apply_scaling kernel launch failed: "
                << cudaGetErrorString(err)
                << " (M=" << M << ", N=" << N << ")";
            throw std::runtime_error(oss.str());
        }

        CUDA_CHECK_THROW(cudaDeviceSynchronize());
        return true;
    }

    /**
     * @brief Quantize FP32 activations to INT8
     */
    bool cudaQuantGemm_quantizeActivations(
        const float *d_A_fp32,
        int8_t *d_A_int8,
        float *d_scales_A,
        int M, int K,
        int cuda_device_id)
    {
        // Validate pointers
        if (!d_A_fp32 || !d_A_int8 || !d_scales_A)
        {
            std::ostringstream oss;
            oss << "[CUDAQuantGemm::quantizeActivations] Null pointer: "
                << "d_A_fp32=" << (void *)d_A_fp32
                << " d_A_int8=" << (void *)d_A_int8
                << " d_scales_A=" << (void *)d_scales_A;
            throw std::runtime_error(oss.str());
        }

        CUDA_CHECK_THROW(cudaSetDevice(cuda_device_id));

        dim3 grid(M);
        dim3 block(std::min(K, 256));

        quantize_activations_kernel<<<grid, block>>>(d_A_fp32, d_A_int8, d_scales_A, M, K);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            std::ostringstream oss;
            oss << "[CUDAQuantGemm] quantize kernel launch failed: "
                << cudaGetErrorString(err)
                << " (M=" << M << ", K=" << K << ")";
            throw std::runtime_error(oss.str());
        }

        CUDA_CHECK_THROW(cudaDeviceSynchronize());
        return true;
    }

    /**
     * @brief Free device memory
     */
    void cudaQuantGemm_freeDevice(void *d_ptr)
    {
        if (d_ptr)
        {
            cudaFree(d_ptr);
        }
    }

    /**
     * @brief Allocate float array on device
     */
    bool cudaQuantGemm_allocFloat(float **d_ptr, size_t count, int cuda_device_id)
    {
        CUDA_CHECK(cudaSetDevice(cuda_device_id));
        CUDA_CHECK(cudaMalloc(d_ptr, count * sizeof(float)));
        return true;
    }

    /**
     * @brief Copy floats from host to device
     */
    bool cudaQuantGemm_copyHostToDevice(float *d_dst, const float *h_src, size_t count, int cuda_device_id)
    {
        CUDA_CHECK(cudaSetDevice(cuda_device_id));
        CUDA_CHECK(cudaMemcpy(d_dst, h_src, count * sizeof(float), cudaMemcpyHostToDevice));
        return true;
    }

    /**
     * @brief Copy floats from device to host
     */
    bool cudaQuantGemm_copyDeviceToHost(float *h_dst, const float *d_src, size_t count, int cuda_device_id)
    {
        CUDA_CHECK(cudaSetDevice(cuda_device_id));
        CUDA_CHECK(cudaMemcpy(h_dst, d_src, count * sizeof(float), cudaMemcpyDeviceToHost));
        return true;
    }

    /**
     * @brief Copy int32 from device to host
     */
    bool cudaQuantGemm_copyInt32DeviceToHost(int32_t *h_dst, const int32_t *d_src, size_t count, int cuda_device_id)
    {
        CUDA_CHECK(cudaSetDevice(cuda_device_id));
        CUDA_CHECK(cudaMemcpy(h_dst, d_src, count * sizeof(int32_t), cudaMemcpyDeviceToHost));
        return true;
    }

    /**
     * @brief Set active CUDA device
     */
    bool cudaQuantGemm_setDevice(int cuda_device_id)
    {
        CUDA_CHECK(cudaSetDevice(cuda_device_id));
        return true;
    }

} // extern "C"
