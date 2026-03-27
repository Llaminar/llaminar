/**
 * @file CUDAcuBLASQuantGemm.cu
 * @brief cuBLAS FP16 GEMM path for Q4_0 quantized weights
 *
 * Dequantizes Q4_0 native VNNI weights to FP16 on-the-fly,
 * converts FP32 activations to FP16, then uses cuBLAS FP16 tensor core
 * GEMM with FP32 accumulation.
 *
 * Enabled via LLAMINAR_CUBLAS_GEMM=1 environment variable.
 * Uses only ~136MB work buffer (largest weight matrix in FP16),
 * NOT the full INT8 expanded weights that would require ~13GB.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#include <cstdint>
#include <cstdio>

#include "kernels/cuda/gemm/CUDADeviceWorkspace.h"

// cuBLAS workspace — per-device, owned by KernelFactory via CUDACuBLASContext.
struct CUDACuBLASContext_
{
    half *d_W_fp16 = nullptr;
    half *d_A_fp16 = nullptr;
    size_t W_cap = 0;
    size_t A_cap = 0;
    cublasHandle_t handle = nullptr;
    int device_id = -1;

    void ensure(size_t W_elems, size_t A_elems)
    {
        if (W_elems > W_cap)
        {
            if (d_W_fp16)
                cudaFree(d_W_fp16);
            d_W_fp16 = nullptr;
            W_cap = 0;
            cudaSetDevice(device_id);
            if (cudaMalloc(&d_W_fp16, W_elems * sizeof(half)) == cudaSuccess)
                W_cap = W_elems;
        }
        if (A_elems > A_cap)
        {
            if (d_A_fp16)
                cudaFree(d_A_fp16);
            d_A_fp16 = nullptr;
            A_cap = 0;
            cudaSetDevice(device_id);
            if (cudaMalloc(&d_A_fp16, A_elems * sizeof(half)) == cudaSuccess)
                A_cap = A_elems;
        }
    }
};

namespace
{

    // Dequant Q4_0 payload → FP16 in column-major [K × N] layout.
    // Each Q4_0 block: 16 bytes (32 nibbles), 1 FP16 scale.
    // Blocks indexed [kb * N + n]: kb = K/32 block, n = column.
    // Q4_0 nibble layout: qs[j]&0xF → element j (0-15), qs[j]>>4 → element j+16 (16-31)
    __global__ void dequant_q40_to_fp16_kernel(
        const uint8_t *__restrict__ payload,
        const uint16_t *__restrict__ scales,
        half *__restrict__ output,
        int K, int N)
    {
        const int blocks_per_col = K / 32;
        const int total_blocks = blocks_per_col * N;

        for (int bid = blockIdx.x * blockDim.x + threadIdx.x;
             bid < total_blocks;
             bid += gridDim.x * blockDim.x)
        {
            const int kb = bid / N;
            const int n = bid % N;

            const float scale = __half2float(
                *reinterpret_cast<const __half *>(&scales[bid]));
            const uint8_t *blk = payload + static_cast<size_t>(bid) * 16;
            half *out = output + static_cast<size_t>(n) * K + kb * 32;

#pragma unroll
            for (int j = 0; j < 16; ++j)
            {
                const uint8_t byte = blk[j];
                const int lo = (byte & 0x0F) - 8;        // element j
                const int hi = ((byte >> 4) & 0x0F) - 8; // element j+16
                out[j] = __float2half(lo * scale);
                out[j + 16] = __float2half(hi * scale);
            }
        }
    }

    __global__ void convert_fp32_to_fp16_flat(
        const float *__restrict__ input,
        half *__restrict__ output,
        int count)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < count)
            output[idx] = __float2half(input[idx]);
    }

} // anonymous namespace

extern "C"
{
    bool cudaCuBLAS_fp16_gemm_q40(
        const uint8_t *d_payload,
        const uint16_t *d_scales_B,
        const float *d_A_fp32,
        float *d_C_fp32,
        int M, int N, int K,
        float alpha, float beta,
        const float *d_C_existing,
        int cuda_device_id,
        void *stream,
        CUDACuBLASContext *cublas_ctx)
    {
        if (!cublas_ctx)
            return false;
        cudaSetDevice(cuda_device_id);
        cudaStream_t s = static_cast<cudaStream_t>(stream);

        const size_t W_elems = static_cast<size_t>(K) * N;
        const size_t A_elems = static_cast<size_t>(M) * K;

        cublas_ctx->ensure(W_elems, A_elems);
        cublasSetStream(cublas_ctx->handle, s);

        // Step 1: Dequant Q4_0 → FP16 col-major [K,N]
        {
            const int total_blocks = (K / 32) * N;
            constexpr int BLK = 256;
            const int grid = (total_blocks + BLK - 1) / BLK;
            dequant_q40_to_fp16_kernel<<<grid, BLK, 0, s>>>(
                d_payload, d_scales_B, cublas_ctx->d_W_fp16, K, N);
        }

        // Step 2: FP32 → FP16 activations
        {
            constexpr int BLK = 256;
            const int grid = (static_cast<int>(A_elems) + BLK - 1) / BLK;
            convert_fp32_to_fp16_flat<<<grid, BLK, 0, s>>>(
                d_A_fp32, cublas_ctx->d_A_fp16, static_cast<int>(A_elems));
        }

        // Step 3: cuBLAS FP16 GEMM → FP32 output
        // C_rm[M,N] = A_rm[M,K] * W_cm[K,N]
        // cuBLAS: C_cm[N,M] = W_cm^T[N,K] * A_cm[K,M]
        const float beta_val = (d_C_existing && beta != 0.0f) ? beta : 0.0f;

        if (beta_val != 0.0f && d_C_existing != d_C_fp32)
        {
            cudaMemcpyAsync(d_C_fp32, d_C_existing,
                            static_cast<size_t>(M) * N * sizeof(float),
                            cudaMemcpyDeviceToDevice, s);
        }

        cublasStatus_t st = cublasGemmEx(
            cublas_ctx->handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            cublas_ctx->d_W_fp16, CUDA_R_16F, K,
            cublas_ctx->d_A_fp16, CUDA_R_16F, K,
            &beta_val,
            d_C_fp32, CUDA_R_32F, N,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP);

        if (st != CUBLAS_STATUS_SUCCESS)
        {
            fprintf(stderr, "[cuBLAS_fp16_q40] err %d (M=%d N=%d K=%d)\n",
                    static_cast<int>(st), M, N, K);
            return false;
        }

        return true;
    }

    CUDACuBLASContext *cudaCuBLASContext_create(int device_id)
    {
        auto *ctx = new (std::nothrow) CUDACuBLASContext_();
        if (!ctx)
            return nullptr;
        ctx->device_id = device_id;
        cudaSetDevice(device_id);
        cublasCreate(&ctx->handle);
        cublasSetMathMode(ctx->handle, CUBLAS_TENSOR_OP_MATH);
        return ctx;
    }

    void cudaCuBLASContext_destroy(CUDACuBLASContext *ctx)
    {
        if (!ctx)
            return;
        cudaSetDevice(ctx->device_id);
        if (ctx->d_W_fp16)
            cudaFree(ctx->d_W_fp16);
        if (ctx->d_A_fp16)
            cudaFree(ctx->d_A_fp16);
        if (ctx->handle)
            cublasDestroy(ctx->handle);
        delete ctx;
    }
}
