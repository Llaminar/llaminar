/**
 * @file CudaGemmKernelPhase7_CUTLASS.cu
 * @brief Phase 7 implementation: CUTLASS int8 GEMM with pre-converted INT8 weights
 *
 * @author David Sanftenberg
 * @date 2025-11-05
 */

#include "CudaGemmKernelPhase7_CUTLASS.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <iostream>
#include <cmath>
#include <algorithm>

namespace llaminar
{
    namespace v2
    {

        // CUTLASS GEMM type definition (internal to .cu file)
        // Using Tensor Cores with DP4A instruction for int8 GEMM on Ampere (SM 8.0+)
        // NOTE: B must be ColumnMajor for tensor core int8 instructions (hardware requirement)
        using CutlassGemm = cutlass::gemm::device::Gemm<
            int8_t,                                 // ElementA
            cutlass::layout::RowMajor,              // LayoutA
            int8_t,                                 // ElementB
            cutlass::layout::ColumnMajor,           // LayoutB (MUST be ColumnMajor for tensor cores!)
            int32_t,                                // ElementOutput (accumulator)
            cutlass::layout::RowMajor,              // LayoutC
            int32_t,                                // ElementAccumulator
            cutlass::arch::OpClassTensorOp,         // OpClass (Tensor Cores!)
            cutlass::arch::Sm80,                    // ArchTag (Ampere SM 8.0+)
            cutlass::gemm::GemmShape<128, 128, 64>, // ThreadblockShape (optimized for tensor cores)
            cutlass::gemm::GemmShape<64, 64, 64>,   // WarpShape (must align with tensor core tile)
            cutlass::gemm::GemmShape<16, 8, 32>,    // InstructionShape (tensor core mma.sync.m16n8k32)
            cutlass::epilogue::thread::LinearCombination<
                int32_t, 1, int32_t, int32_t>,                            // EpilogueOp (simple passthrough)
            cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, // ThreadblockSwizzle
            3                                                             // Stages (pipeline depth - increased for tensor cores)
            >;

        // Pimpl implementation struct
        struct CudaGemmKernelPhase7_CUTLASS::Impl
        {
            // Device memory buffers
            int8_t *d_A_int8 = nullptr;
            int8_t *d_B_int8 = nullptr;
            int32_t *d_C_int32 = nullptr;
            float *d_scales_A = nullptr;
            float *d_scales_B = nullptr;

            // Current allocation sizes
            int allocated_M = 0;
            int allocated_N = 0;
            int allocated_K = 0;

            void ensure_buffers(int M, int N, int K);
            void free_buffers();
        };

        namespace
        {

// CUDA error checking macros
#define CUDA_CHECK(call)                                                                          \
    do                                                                                            \
    {                                                                                             \
        cudaError_t err = call;                                                                   \
        if (err != cudaSuccess)                                                                   \
        {                                                                                         \
            std::cerr << "[CUDA ERROR] " << #call << " failed at " << __FILE__ << ":" << __LINE__ \
                      << " with error: " << cudaGetErrorString(err) << "\n";                      \
            return false;                                                                         \
        }                                                                                         \
    } while (0)

#define CUDA_CHECK_KERNEL()                                                                        \
    do                                                                                             \
    {                                                                                              \
        cudaError_t err = cudaGetLastError();                                                      \
        if (err != cudaSuccess)                                                                    \
        {                                                                                          \
            std::cerr << "[CUDA ERROR] Kernel launch failed at " << __FILE__ << ":" << __LINE__    \
                      << " with error: " << cudaGetErrorString(err) << "\n";                       \
            return false;                                                                          \
        }                                                                                          \
        err = cudaDeviceSynchronize();                                                             \
        if (err != cudaSuccess)                                                                    \
        {                                                                                          \
            std::cerr << "[CUDA ERROR] Kernel execution failed at " << __FILE__ << ":" << __LINE__ \
                      << " with error: " << cudaGetErrorString(err) << "\n";                       \
            return false;                                                                          \
        }                                                                                          \
    } while (0)

            /**
             * @brief CUDA kernel: Quantize matrix A from fp32 to int8 (symmetric per-row)
             *
             * Each thread processes one row:
             * 1. Find max_abs across row
             * 2. Compute scale = max_abs / 127
             * 3. Quantize: a_int8[j] = round(a_fp32[j] / scale)
             *
             * Grid: (M blocks, 1, 1)
             * Block: (min(K, 256) threads, 1, 1)
             */
            __global__ void quantize_A_kernel(
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

                // Phase 1: Find max_abs (parallel reduction)
                __shared__ float shared_max[256];

                float local_max = 0.0f;
                for (int j = threadIdx.x; j < K; j += blockDim.x)
                {
                    local_max = fmaxf(local_max, fabsf(row_fp32[j]));
                }
                shared_max[threadIdx.x] = local_max;
                __syncthreads();

                // Reduce shared_max to thread 0
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
                    // Round to nearest int8
                    int8_t quantized = (int8_t)rintf(fminf(127.0f, fmaxf(-127.0f, val)));
                    row_int8[j] = quantized;
                }
            }

            /**
             * @brief CUDA kernel: Apply scaling to GEMM output
             *
             * C_fp32[i,j] = C_int32[i,j] × scales_A[i] × scales_B[j]
             *
             * Grid: (ceil(N/16), ceil(M/16), 1)
             * Block: (16, 16, 1)
             */
            __global__ void apply_scaling_kernel(
                const int32_t *C_int32, // [M×N]
                float *C_fp32,          // [M×N] output
                const float *scales_A,  // [M] row scales
                const float *scales_B,  // [N] column scales
                int M, int N)
            {
                int row = blockIdx.y * blockDim.y + threadIdx.y;
                int col = blockIdx.x * blockDim.x + threadIdx.x;

                if (row < M && col < N)
                {
                    int32_t val_int32 = C_int32[row * N + col];
                    float scale_combined = scales_A[row] * scales_B[col];
                    C_fp32[row * N + col] = (float)val_int32 * scale_combined;
                }
            }

        } // anonymous namespace

        // Constructor/destructor
        CudaGemmKernelPhase7_CUTLASS::CudaGemmKernelPhase7_CUTLASS()
            : impl_(new Impl()) {}

        CudaGemmKernelPhase7_CUTLASS::~CudaGemmKernelPhase7_CUTLASS()
        {
            if (impl_)
            {
                impl_->free_buffers();
                delete impl_;
            }
        }

        void CudaGemmKernelPhase7_CUTLASS::Impl::ensure_buffers(int M, int N, int K)
        {
            bool need_realloc = (M > allocated_M || N > allocated_N || K > allocated_K);

            if (!need_realloc)
                return;

            // Free old buffers
            free_buffers();

            // Allocate new buffers with padding
            cudaError_t err;
            err = cudaMalloc(&d_A_int8, M * K * sizeof(int8_t));
            if (err != cudaSuccess)
            {
                std::cerr << "[Phase7] cudaMalloc d_A_int8 failed: " << cudaGetErrorString(err) << "\n";
                return;
            }
            err = cudaMalloc(&d_B_int8, K * N * sizeof(int8_t));
            if (err != cudaSuccess)
            {
                std::cerr << "[Phase7] cudaMalloc d_B_int8 failed: " << cudaGetErrorString(err) << "\n";
                return;
            }
            err = cudaMalloc(&d_C_int32, M * N * sizeof(int32_t));
            if (err != cudaSuccess)
            {
                std::cerr << "[Phase7] cudaMalloc d_C_int32 failed: " << cudaGetErrorString(err) << "\n";
                return;
            }
            err = cudaMalloc(&d_scales_A, M * sizeof(float));
            if (err != cudaSuccess)
            {
                std::cerr << "[Phase7] cudaMalloc d_scales_A failed: " << cudaGetErrorString(err) << "\n";
                return;
            }
            err = cudaMalloc(&d_scales_B, N * sizeof(float));
            if (err != cudaSuccess)
            {
                std::cerr << "[Phase7] cudaMalloc d_scales_B failed: " << cudaGetErrorString(err) << "\n";
                return;
            }

            allocated_M = M;
            allocated_N = N;
            allocated_K = K;

            std::cout << "[Phase7] Allocated buffers: M=" << M << ", N=" << N << ", K=" << K << "\n";
        }

        void CudaGemmKernelPhase7_CUTLASS::Impl::free_buffers()
        {
            if (d_A_int8)
            {
                cudaFree(d_A_int8);
                d_A_int8 = nullptr;
            }
            if (d_B_int8)
            {
                cudaFree(d_B_int8);
                d_B_int8 = nullptr;
            }
            if (d_C_int32)
            {
                cudaFree(d_C_int32);
                d_C_int32 = nullptr;
            }
            if (d_scales_A)
            {
                cudaFree(d_scales_A);
                d_scales_A = nullptr;
            }
            if (d_scales_B)
            {
                cudaFree(d_scales_B);
                d_scales_B = nullptr;
            }

            allocated_M = allocated_N = allocated_K = 0;
        }

        bool CudaGemmKernelPhase7_CUTLASS::execute(
            const float *A_fp32,
            const int8_t *B_int8,
            const float *scales_B,
            float *C,
            int M, int N, int K)
        {
            // Step 0: Ensure device buffers are allocated
            impl_->ensure_buffers(M, N, K);

            // Step 1: Allocate and upload A to device
            float *d_A_fp32;
            CUDA_CHECK(cudaMalloc(&d_A_fp32, M * K * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(d_A_fp32, A_fp32, M * K * sizeof(float), cudaMemcpyHostToDevice));

            // Step 2: Quantize A (fp32 → int8)
            dim3 grid_A(M, 1, 1);
            dim3 block_A(std::min(K, 256), 1, 1);

            quantize_A_kernel<<<grid_A, block_A>>>(d_A_fp32, impl_->d_A_int8, impl_->d_scales_A, M, K);
            CUDA_CHECK_KERNEL();

            CUDA_CHECK(cudaFree(d_A_fp32)); // No longer needed

            // Step 3: Upload pre-converted INT8 weights and scales
            CUDA_CHECK(cudaMemcpy(impl_->d_B_int8, B_int8, K * N * sizeof(int8_t), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(impl_->d_scales_B, scales_B, N * sizeof(float), cudaMemcpyHostToDevice));

            // Step 4: Run CUTLASS GEMM (int8×int8 → int32)
            CutlassGemm gemm_op;

            typename CutlassGemm::Arguments args(
                {M, N, K},             // Problem size
                {impl_->d_A_int8, K},  // TensorRef A
                {impl_->d_B_int8, N},  // TensorRef B
                {impl_->d_C_int32, N}, // TensorRef C (unused, beta=0)
                {impl_->d_C_int32, N}, // TensorRef D (output)
                {1, 0}                 // alpha=1, beta=0
            );

            cutlass::Status status = gemm_op(args);

            if (status != cutlass::Status::kSuccess)
            {
                std::cerr << "[Phase7] CUTLASS GEMM failed: " << int(status) << "\n";
                return false;
            }

            // Step 5: Apply scaling and download result
            float *d_C_fp32;
            CUDA_CHECK(cudaMalloc(&d_C_fp32, M * N * sizeof(float)));

            dim3 grid_C((N + 15) / 16, (M + 15) / 16, 1);
            dim3 block_C(16, 16, 1);
            apply_scaling_kernel<<<grid_C, block_C>>>(impl_->d_C_int32, d_C_fp32,
                                                      impl_->d_scales_A, impl_->d_scales_B, M, N);
            CUDA_CHECK_KERNEL();

            CUDA_CHECK(cudaMemcpy(C, d_C_fp32, M * N * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaFree(d_C_fp32));

            return true;
        }

    } // namespace v2
} // namespace llaminar
