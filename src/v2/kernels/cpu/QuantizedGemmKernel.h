/**
 * @file QuantizedGemmKernel.h
 * @brief Generic quantized GEMM kernel using ITensorGemmTileDataProvider
 *
 * This kernel performs on-the-fly dequantization during matrix multiplication,
 * avoiding the need to materialize the full FP32 weight matrix in memory.
 * This is critical for memory-bandwidth bound operations like single-token decoding (GEMV).
 *
 * @author GitHub Copilot
 */

#pragma once

#include "../../tensors/TensorKernels.h"
#include "../../tensors/Tensors.h"
#include "../../utils/Logger.h"
#include "gemm_v4/OneDNNGemmKernel.h" // For fallback
#include <vector>
#include <cmath>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace llaminar2
{
    class QuantizedGemmKernel : public ITensorGemm
    {
    public:
        explicit QuantizedGemmKernel(const TensorBase *weight_tensor)
            : weight_tensor_(weight_tensor)
        {
            provider_ = dynamic_cast<const ITensorGemmTileDataProvider *>(weight_tensor);
            if (!provider_)
            {
                throw std::runtime_error("QuantizedGemmKernel requires a tensor implementing ITensorGemmTileDataProvider");
            }
        }

        ~QuantizedGemmKernel() override = default;

        bool supports_device(int device_idx) const override
        {
            return device_idx == -1; // CPU only
        }

        bool multiply_activations(
            const float *A, const float *B, float *C,
            int m, int n, int k,
            bool transpose_B = true,
            float alpha = 1.0f, float beta = 0.0f,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override
        {
            // If B is provided (and not ignored), we are doing activation-activation GEMM.
            // But this kernel is bound to a weight tensor.
            // If transpose_B is true, we assume B corresponds to our weight tensor (if B was passed as nullptr or ignored).
            // However, the interface says "B Right activation matrix".
            // In Qwen2Pipeline, it calls: q_gemm->multiply_activations(..., layer.wq->data(), ...)
            // So B IS the weight data (FP32).

            // BUT, we want to ignore B (which is the FP32 pointer) and use our internal provider_.

            if (!transpose_B)
            {
                LOG_ERROR("[QuantizedGemmKernel] Only transposed B (weights) supported");
                return false;
            }

            // Check dimensions
            if (static_cast<size_t>(n) != provider_->decoder_rows() || static_cast<size_t>(k) != provider_->decoder_cols())
            {
                // Note: provider_ is the weight tensor.
                // If transpose_B is true, then B is [n, k].
                // So weight tensor should be [n, k].
                // provider_->decoder_rows() is shape[0] (n), provider_->decoder_cols() is shape[1] (k).
                if (static_cast<size_t>(n) != provider_->decoder_rows() || static_cast<size_t>(k) != provider_->decoder_cols())
                {
                    LOG_ERROR("[QuantizedGemmKernel] Dimension mismatch: expected ["
                              << provider_->decoder_rows() << "x" << provider_->decoder_cols()
                              << "], got [" << n << "x" << k << "]");
                    return false;
                }
            }

            // Optimization for M=1 (GEMV)
            if (m == 1)
            {
                return gemv_quantized(A, C, n, k, alpha, beta);
            }

            // Fallback for M > 1: Use OneDNN (requires dequantization)
            // We can use the OneDNNGemmKernel for this.

            const float *B_ptr = B;
            if (!B_ptr)
            {
                // If B is not provided, we must get it from the weight tensor.
                // This triggers full dequantization if not already cached.
                // This is acceptable for M > 1 (compute bound).
                // We cast away constness of data() because ITensorGemm interface uses const float*
                // but TensorBase::data() is const float*.
                B_ptr = weight_tensor_->data();
            }

            if (B_ptr)
            {
                if (!A)
                    LOG_ERROR("[QuantizedGemmKernel] A is null");
                if (!B_ptr)
                    LOG_ERROR("[QuantizedGemmKernel] B_ptr is null");
                if (!C)
                    LOG_ERROR("[QuantizedGemmKernel] C is null");

                llaminar2::gemm_v4::OneDNNGemmKernel onednn_kernel(weight_tensor_);
                return onednn_kernel.multiply_activations(A, B_ptr, C, m, n, k, transpose_B, alpha, beta, mpi_ctx, device_idx);
            }
            else
            {
                LOG_ERROR("[QuantizedGemmKernel] Failed to obtain weight data for fallback");
                return false;
            }
        }

        // Implement other virtual methods with default/error
        bool multiply_activations_strided(
            const float *A, const float *B, float *C,
            int m, int n, int k,
            int lda, int ldb, int ldc,
            bool transpose_B = true,
            float alpha = 1.0f, float beta = 0.0f,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override
        {
            const float *B_ptr = B;
            if (!B_ptr)
            {
                B_ptr = weight_tensor_->data();
            }

            if (B_ptr)
            {
                llaminar2::gemm_v4::OneDNNGemmKernel onednn_kernel(weight_tensor_);
                return onednn_kernel.multiply_activations_strided(A, B_ptr, C, m, n, k, lda, ldb, ldc, transpose_B, alpha, beta, mpi_ctx, device_idx);
            }
            return false;
        }

    private:
        const TensorBase *weight_tensor_;
        const ITensorGemmTileDataProvider *provider_;

        bool gemv_quantized(const float *A, float *C, int n, int k, float alpha, float beta)
        {
// C = alpha * A * B^T + beta * C
// A is [1, k]
// B is [n, k] (weights)
// C is [1, n]

// Parallelize over rows of B (output elements of C)
#pragma omp parallel for schedule(static)
            for (int i = 0; i < n; ++i)
            {
                float dot = 0.0f;
                size_t block_size = provider_->block_size();
                size_t num_blocks = (k + block_size - 1) / block_size;

                // Temporary buffer for one block
                // We use a small stack buffer. Max block size is usually 256 (Q6_K).
                // Q4_0 is 32.
                alignas(64) float block_cache[256];

                for (size_t b = 0; b < num_blocks; ++b)
                {
                    // Decode block i, b
                    provider_->decode_block_at(i, b, block_cache);

                    // Compute dot product for this block
                    size_t start_k = b * block_size;
                    size_t current_block_len = std::min(block_size, static_cast<size_t>(k - start_k));

                    // Vectorized dot product
                    // We can use AVX if available, or simple loop (compiler auto-vectorization)
                    for (size_t j = 0; j < current_block_len; ++j)
                    {
                        dot += A[start_k + j] * block_cache[j];
                    }
                }

                if (beta == 0.0f)
                {
                    C[i] = alpha * dot;
                }
                else
                {
                    C[i] = alpha * dot + beta * C[i];
                }
            }
            return true;
        }
    };
}
