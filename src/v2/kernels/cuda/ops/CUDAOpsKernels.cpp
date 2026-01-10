/**
 * @file CUDAOpsKernels.cpp
 * @brief Implementation of CUDA ops kernel ITensorXxx methods
 *
 * This file contains the C++ method implementations for CUDARMSNormKernelT,
 * CUDASwiGLUKernelT, and CUDARoPEKernelT. It calls the extern "C" wrapper
 * functions defined in CUDAOpsKernels.cu.
 *
 * Separating .cu and .cpp allows nvcc to compile only the CUDA code without
 * encountering issues with MPI or other complex C++ headers.
 */

#include "CUDARMSNormKernelT.h"
#include "CUDASwiGLUKernelT.h"
#include "CUDARoPEKernelT.h"
#include "CUDAEmbeddingKernelT.h"
#include "../../../tensors/Tensors.h"
#include "../../../backends/DeviceId.h"

#include <cuda_runtime.h>
#include <cstdio>

// =========================================================================
// Extern "C" declarations for CUDA kernel wrappers
// =========================================================================

extern "C"
{
    // RMSNorm
    bool cudaOps_rmsnorm_fp32(
        const float *input, const float *gamma, float *output,
        int rows, int cols, float epsilon, int device_idx);
    bool cudaOps_rmsnorm_bf16(
        const uint16_t *input, const float *gamma, uint16_t *output,
        int rows, int cols, float epsilon, int device_idx);
    bool cudaOps_rmsnorm_fp16(
        const uint16_t *input, const float *gamma, uint16_t *output,
        int rows, int cols, float epsilon, int device_idx);

    // SwiGLU
    bool cudaOps_swiglu_fp32(
        const float *gate, const float *up, float *output,
        int size, int device_idx);
    bool cudaOps_swiglu_bf16(
        const uint16_t *gate, const uint16_t *up, uint16_t *output,
        int size, int device_idx);
    bool cudaOps_swiglu_fp16(
        const uint16_t *gate, const uint16_t *up, uint16_t *output,
        int size, int device_idx);

    // RoPE
    bool cudaOps_rope_fp32(
        float *Q, float *K, const int *position_ids,
        int seq_len, int n_heads, int n_kv_heads, int head_dim,
        float rope_theta, int device_idx);
    bool cudaOps_rope_bf16(
        uint16_t *Q, uint16_t *K, const int *position_ids,
        int seq_len, int n_heads, int n_kv_heads, int head_dim,
        float rope_theta, int device_idx);
    bool cudaOps_rope_fp16(
        uint16_t *Q, uint16_t *K, const int *position_ids,
        int seq_len, int n_heads, int n_kv_heads, int head_dim,
        float rope_theta, int device_idx);

    // Embedding lookup
    cudaError_t launch_embedding_lookup(
        const float *embed_data,
        const int *token_ids,
        float *output,
        int num_tokens,
        int d_model,
        cudaStream_t stream);
}

namespace llaminar2
{
    namespace cuda
    {

        // =========================================================================
        // CUDARMSNormKernelT<FP32> Implementation
        // =========================================================================

        bool CUDARMSNormKernelT<ActivationPrecision::FP32>::apply(
            const float *input, const float *weight, float *output,
            int rows, int cols,
            float epsilon,
            bool use_bf16,
            const MPIContext *mpi_ctx,
            int device_idx)
        {
            (void)use_bf16;
            (void)mpi_ctx;
            int dev = (device_idx >= 0) ? device_idx : device_idx_;
            return apply_typed(input, weight, output, rows, cols, epsilon, dev);
        }

        bool CUDARMSNormKernelT<ActivationPrecision::FP32>::apply_tensor(
            const TensorBase *input,
            const TensorBase *weight,
            TensorBase *output,
            int rows, int cols,
            float epsilon,
            const MPIContext *mpi_ctx,
            int device_idx)
        {
            (void)mpi_ctx;
            if (!input || !weight || !output)
                return false;
            if (input->native_type() != TensorType::FP32 || output->native_type() != TensorType::FP32)
                return false;

            int dev = (device_idx >= 0) ? device_idx : device_idx_;

            // Cast to FP32Tensor for GPU operations
            auto *input_fp32 = const_cast<FP32Tensor *>(dynamic_cast<const FP32Tensor *>(input));
            auto *weight_fp32 = const_cast<FP32Tensor *>(dynamic_cast<const FP32Tensor *>(weight));
            auto *output_fp32 = dynamic_cast<FP32Tensor *>(output);

            if (!input_fp32 || !weight_fp32 || !output_fp32)
                return false;

            // Coherence handled automatically by GraphExecutor

            // Get device pointers
            const float *d_input = static_cast<const float *>(input_fp32->gpu_data_ptr());
            const float *d_weight = static_cast<const float *>(weight_fp32->gpu_data_ptr());
            float *d_output = static_cast<float *>(output_fp32->gpu_data_ptr());

            bool ok = cudaOps_rmsnorm_fp32(d_input, d_weight, d_output, rows, cols, epsilon, dev);
            if (ok)
            {
                cudaDeviceSynchronize();
            }
            return ok;
        }

        bool CUDARMSNormKernelT<ActivationPrecision::FP32>::apply_typed(
            const float *input,
            const float *gamma,
            float *output,
            int rows, int cols,
            float epsilon,
            int device_idx)
        {
            int dev = (device_idx >= 0) ? device_idx : device_idx_;
            bool ok = cudaOps_rmsnorm_fp32(input, gamma, output, rows, cols, epsilon, dev);
            if (ok)
                cudaDeviceSynchronize();
            return ok;
        }

        // =========================================================================
        // CUDARMSNormKernelT<BF16> Implementation
        // =========================================================================

        bool CUDARMSNormKernelT<ActivationPrecision::BF16>::apply_bf16(
            const uint16_t *input, const float *weight, uint16_t *output,
            int rows, int cols, float epsilon, int device_idx)
        {
            int dev = (device_idx >= 0) ? device_idx : device_idx_;
            return apply_typed(input, weight, output, rows, cols, epsilon, dev);
        }

        bool CUDARMSNormKernelT<ActivationPrecision::BF16>::apply_tensor(
            const TensorBase *input,
            const TensorBase *weight,
            TensorBase *output,
            int rows, int cols,
            float epsilon,
            const MPIContext *mpi_ctx,
            int device_idx)
        {
            (void)mpi_ctx;
            if (!input || !weight || !output)
                return false;
            if (input->native_type() != TensorType::BF16 || output->native_type() != TensorType::BF16)
                return false;

            auto *in_bf16 = const_cast<BF16Tensor *>(dynamic_cast<const BF16Tensor *>(input));
            auto *weight_fp32 = const_cast<FP32Tensor *>(dynamic_cast<const FP32Tensor *>(weight));
            auto *out_bf16 = dynamic_cast<BF16Tensor *>(output);
            if (!in_bf16 || !out_bf16)
                return false;

            int dev = (device_idx >= 0) ? device_idx : device_idx_;

            // Coherence handled automatically by GraphExecutor

            // Get device pointers
            const uint16_t *d_input = static_cast<const uint16_t *>(in_bf16->gpu_data_ptr());
            const float *d_weight = weight_fp32 ? static_cast<const float *>(weight_fp32->gpu_data_ptr()) : weight->data();
            uint16_t *d_output = static_cast<uint16_t *>(out_bf16->gpu_data_ptr());

            bool ok = cudaOps_rmsnorm_bf16(d_input, d_weight, d_output, rows, cols, epsilon, dev);
            if (ok)
            {
                cudaDeviceSynchronize();
            }
            return ok;
        }

        bool CUDARMSNormKernelT<ActivationPrecision::BF16>::apply_typed(
            const uint16_t *input,
            const float *gamma,
            uint16_t *output,
            int rows, int cols,
            float epsilon,
            int device_idx)
        {
            int dev = (device_idx >= 0) ? device_idx : device_idx_;
            bool ok = cudaOps_rmsnorm_bf16(input, gamma, output, rows, cols, epsilon, dev);
            if (ok)
                cudaDeviceSynchronize();
            return ok;
        }

        // =========================================================================
        // CUDARMSNormKernelT<FP16> Implementation
        // =========================================================================

        bool CUDARMSNormKernelT<ActivationPrecision::FP16>::apply_fp16(
            const uint16_t *input, const float *weight, uint16_t *output,
            int rows, int cols, float epsilon, int device_idx)
        {
            int dev = (device_idx >= 0) ? device_idx : device_idx_;
            return apply_typed(input, weight, output, rows, cols, epsilon, dev);
        }

        bool CUDARMSNormKernelT<ActivationPrecision::FP16>::apply_tensor(
            const TensorBase *input,
            const TensorBase *weight,
            TensorBase *output,
            int rows, int cols,
            float epsilon,
            const MPIContext *mpi_ctx,
            int device_idx)
        {
            (void)mpi_ctx;
            if (!input || !weight || !output)
                return false;
            if (input->native_type() != TensorType::FP16 || output->native_type() != TensorType::FP16)
                return false;

            auto *in_fp16 = const_cast<FP16Tensor *>(dynamic_cast<const FP16Tensor *>(input));
            auto *weight_fp32 = const_cast<FP32Tensor *>(dynamic_cast<const FP32Tensor *>(weight));
            auto *out_fp16 = dynamic_cast<FP16Tensor *>(output);
            if (!in_fp16 || !out_fp16)
                return false;

            int dev = (device_idx >= 0) ? device_idx : device_idx_;

            // Coherence handled automatically by GraphExecutor

            // Get device pointers
            const uint16_t *d_input = static_cast<const uint16_t *>(in_fp16->gpu_data_ptr());
            const float *d_weight = weight_fp32 ? static_cast<const float *>(weight_fp32->gpu_data_ptr()) : weight->data();
            uint16_t *d_output = static_cast<uint16_t *>(out_fp16->gpu_data_ptr());

            bool ok = cudaOps_rmsnorm_fp16(d_input, d_weight, d_output, rows, cols, epsilon, dev);
            if (ok)
            {
                cudaDeviceSynchronize();
            }
            return ok;
        }

        bool CUDARMSNormKernelT<ActivationPrecision::FP16>::apply_typed(
            const uint16_t *input,
            const float *gamma,
            uint16_t *output,
            int rows, int cols,
            float epsilon,
            int device_idx)
        {
            int dev = (device_idx >= 0) ? device_idx : device_idx_;
            bool ok = cudaOps_rmsnorm_fp16(input, gamma, output, rows, cols, epsilon, dev);
            if (ok)
                cudaDeviceSynchronize();
            return ok;
        }

        // =========================================================================
        // CUDASwiGLUKernelT<FP32> Implementation
        // =========================================================================

        bool CUDASwiGLUKernelT<ActivationPrecision::FP32>::apply(
            const float *gate, const float *up, float *output,
            int rows, int cols,
            bool add_residual,
            const MPIContext *mpi_ctx,
            int device_idx)
        {
            (void)add_residual; // TODO: implement residual addition
            (void)mpi_ctx;
            int size = rows * cols;
            return apply_typed(gate, up, output, size);
        }

        bool CUDASwiGLUKernelT<ActivationPrecision::FP32>::apply_tensor(
            const TensorBase *gate,
            const TensorBase *up,
            TensorBase *output,
            int rows, int cols,
            bool add_residual,
            const MPIContext *mpi_ctx,
            int device_idx)
        {
            (void)add_residual;
            (void)mpi_ctx;
            if (!gate || !up || !output)
                return false;
            if (gate->native_type() != TensorType::FP32 ||
                up->native_type() != TensorType::FP32 ||
                output->native_type() != TensorType::FP32)
                return false;

            int dev = (device_idx >= 0) ? device_idx : device_idx_;

            // Cast to FP32Tensor for GPU operations
            auto *gate_fp32 = const_cast<FP32Tensor *>(dynamic_cast<const FP32Tensor *>(gate));
            auto *up_fp32 = const_cast<FP32Tensor *>(dynamic_cast<const FP32Tensor *>(up));
            auto *output_fp32 = dynamic_cast<FP32Tensor *>(output);

            if (!gate_fp32 || !up_fp32 || !output_fp32)
                return false;

            // Coherence handled automatically by GraphExecutor

            // Get device pointers
            const float *d_gate = static_cast<const float *>(gate_fp32->gpu_data_ptr());
            const float *d_up = static_cast<const float *>(up_fp32->gpu_data_ptr());
            float *d_output = static_cast<float *>(output_fp32->gpu_data_ptr());

            int size = rows * cols;
            bool ok = cudaOps_swiglu_fp32(d_gate, d_up, d_output, size, dev);
            if (ok)
            {
                cudaDeviceSynchronize();
            }
            return ok;
        }

        bool CUDASwiGLUKernelT<ActivationPrecision::FP32>::apply_typed(
            const float *gate,
            const float *up,
            float *output,
            int size,
            int device_idx)
        {
            int dev = (device_idx >= 0) ? device_idx : device_idx_;
            bool ok = cudaOps_swiglu_fp32(gate, up, output, size, dev);
            if (ok)
                cudaDeviceSynchronize();
            return ok;
        }

        // =========================================================================
        // CUDASwiGLUKernelT<BF16> Implementation
        // =========================================================================

        bool CUDASwiGLUKernelT<ActivationPrecision::BF16>::apply_bf16(
            const uint16_t *gate, const uint16_t *up, uint16_t *output,
            int rows, int cols,
            bool add_residual,
            const MPIContext *mpi_ctx,
            int device_idx)
        {
            (void)add_residual;
            (void)mpi_ctx;
            (void)device_idx;
            int size = rows * cols;
            return apply_typed(gate, up, output, size);
        }

        bool CUDASwiGLUKernelT<ActivationPrecision::BF16>::apply_tensor(
            const TensorBase *gate,
            const TensorBase *up,
            TensorBase *output,
            int rows, int cols,
            bool add_residual,
            const MPIContext *mpi_ctx,
            int device_idx)
        {
            (void)add_residual;
            (void)mpi_ctx;
            if (!gate || !up || !output)
                return false;
            if (gate->native_type() != TensorType::BF16 ||
                up->native_type() != TensorType::BF16 ||
                output->native_type() != TensorType::BF16)
                return false;

            auto *gate_bf16 = const_cast<BF16Tensor *>(dynamic_cast<const BF16Tensor *>(gate));
            auto *up_bf16 = const_cast<BF16Tensor *>(dynamic_cast<const BF16Tensor *>(up));
            auto *out_bf16 = dynamic_cast<BF16Tensor *>(output);
            if (!gate_bf16 || !up_bf16 || !out_bf16)
                return false;

            int dev = (device_idx >= 0) ? device_idx : device_idx_;

            // Coherence handled automatically by GraphExecutor

            // Get device pointers
            const uint16_t *d_gate = static_cast<const uint16_t *>(gate_bf16->gpu_data_ptr());
            const uint16_t *d_up = static_cast<const uint16_t *>(up_bf16->gpu_data_ptr());
            uint16_t *d_output = static_cast<uint16_t *>(out_bf16->gpu_data_ptr());

            int size = rows * cols;
            bool ok = cudaOps_swiglu_bf16(d_gate, d_up, d_output, size, dev);
            if (ok)
            {
                cudaDeviceSynchronize();
            }
            return ok;
        }

        bool CUDASwiGLUKernelT<ActivationPrecision::BF16>::apply_typed(
            const uint16_t *gate,
            const uint16_t *up,
            uint16_t *output,
            int size,
            int device_idx)
        {
            int dev = (device_idx >= 0) ? device_idx : device_idx_;
            bool ok = cudaOps_swiglu_bf16(gate, up, output, size, dev);
            if (ok)
                cudaDeviceSynchronize();
            return ok;
        }

        // =========================================================================
        // CUDASwiGLUKernelT<FP16> Implementation
        // =========================================================================

        bool CUDASwiGLUKernelT<ActivationPrecision::FP16>::apply_fp16(
            const uint16_t *gate, const uint16_t *up, uint16_t *output,
            int rows, int cols,
            bool add_residual,
            const MPIContext *mpi_ctx,
            int device_idx)
        {
            (void)add_residual;
            (void)mpi_ctx;
            (void)device_idx;
            int size = rows * cols;
            return apply_typed(gate, up, output, size);
        }

        bool CUDASwiGLUKernelT<ActivationPrecision::FP16>::apply_tensor(
            const TensorBase *gate,
            const TensorBase *up,
            TensorBase *output,
            int rows, int cols,
            bool add_residual,
            const MPIContext *mpi_ctx,
            int device_idx)
        {
            (void)add_residual;
            (void)mpi_ctx;
            if (!gate || !up || !output)
                return false;
            if (gate->native_type() != TensorType::FP16 ||
                up->native_type() != TensorType::FP16 ||
                output->native_type() != TensorType::FP16)
                return false;

            auto *gate_fp16 = const_cast<FP16Tensor *>(dynamic_cast<const FP16Tensor *>(gate));
            auto *up_fp16 = const_cast<FP16Tensor *>(dynamic_cast<const FP16Tensor *>(up));
            auto *out_fp16 = dynamic_cast<FP16Tensor *>(output);
            if (!gate_fp16 || !up_fp16 || !out_fp16)
                return false;

            int dev = (device_idx >= 0) ? device_idx : device_idx_;

            // Coherence handled automatically by GraphExecutor

            // Get device pointers
            const uint16_t *d_gate = static_cast<const uint16_t *>(gate_fp16->gpu_data_ptr());
            const uint16_t *d_up = static_cast<const uint16_t *>(up_fp16->gpu_data_ptr());
            uint16_t *d_output = static_cast<uint16_t *>(out_fp16->gpu_data_ptr());

            int size = rows * cols;
            bool ok = cudaOps_swiglu_fp16(d_gate, d_up, d_output, size, dev);
            if (ok)
            {
                cudaDeviceSynchronize();
            }
            return ok;
        }

        bool CUDASwiGLUKernelT<ActivationPrecision::FP16>::apply_typed(
            const uint16_t *gate,
            const uint16_t *up,
            uint16_t *output,
            int size,
            int device_idx)
        {
            int dev = (device_idx >= 0) ? device_idx : device_idx_;
            bool ok = cudaOps_swiglu_fp16(gate, up, output, size, dev);
            if (ok)
                cudaDeviceSynchronize();
            return ok;
        }

        // =========================================================================
        // CUDARoPEKernelT<FP32> Implementation
        // =========================================================================

        bool CUDARoPEKernelT<ActivationPrecision::FP32>::apply(
            float *data, float *output,
            const int *pos_ids,
            int batch_size, int seq_len, int head_dim, int num_heads,
            float theta_base, bool interleaved,
            const MPIContext *mpi_ctx,
            int device_idx)
        {
            (void)output;      // In-place operation
            (void)batch_size;  // Already folded into seq_len
            (void)interleaved; // TODO: support interleaved layout
            (void)mpi_ctx;
            int dev = (device_idx >= 0) ? device_idx : device_idx_;
            return apply_typed(data, nullptr, pos_ids, seq_len, num_heads, num_heads, head_dim, theta_base, dev);
        }

        bool CUDARoPEKernelT<ActivationPrecision::FP32>::apply_typed(
            float *Q,
            float *K,
            const int *position_ids,
            int seq_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            float rope_theta,
            int device_idx)
        {
            int dev = (device_idx >= 0) ? device_idx : device_idx_;
            bool ok = cudaOps_rope_fp32(Q, K, position_ids, seq_len, n_heads, n_kv_heads,
                                        head_dim, rope_theta, dev);
            if (ok)
                cudaDeviceSynchronize();
            return ok;
        }

        // =========================================================================
        // CUDARoPEKernelT<BF16> Implementation
        // =========================================================================

        bool CUDARoPEKernelT<ActivationPrecision::BF16>::apply_bf16(
            uint16_t *data, uint16_t *output,
            const int *pos_ids,
            int batch_size, int seq_len, int head_dim, int num_heads,
            float theta_base, int device_idx)
        {
            (void)output;
            (void)batch_size;
            int dev = (device_idx >= 0) ? device_idx : device_idx_;
            return apply_typed(data, nullptr, pos_ids, seq_len, num_heads, num_heads, head_dim, theta_base, dev);
        }

        bool CUDARoPEKernelT<ActivationPrecision::BF16>::apply_typed(
            uint16_t *Q,
            uint16_t *K,
            const int *position_ids,
            int seq_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            float rope_theta,
            int device_idx)
        {
            int dev = (device_idx >= 0) ? device_idx : device_idx_;
            bool ok = cudaOps_rope_bf16(Q, K, position_ids, seq_len, n_heads, n_kv_heads,
                                        head_dim, rope_theta, dev);
            if (ok)
                cudaDeviceSynchronize();
            return ok;
        }

        // =========================================================================
        // CUDARoPEKernelT<FP16> Implementation
        // =========================================================================

        bool CUDARoPEKernelT<ActivationPrecision::FP16>::apply_fp16(
            uint16_t *data, uint16_t *output,
            const int *pos_ids,
            int batch_size, int seq_len, int head_dim, int num_heads,
            float theta_base, int device_idx)
        {
            (void)output;
            (void)batch_size;
            int dev = (device_idx >= 0) ? device_idx : device_idx_;
            return apply_typed(data, nullptr, pos_ids, seq_len, num_heads, num_heads, head_dim, theta_base, dev);
        }

        bool CUDARoPEKernelT<ActivationPrecision::FP16>::apply_typed(
            uint16_t *Q,
            uint16_t *K,
            const int *position_ids,
            int seq_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            float rope_theta,
            int device_idx)
        {
            int dev = (device_idx >= 0) ? device_idx : device_idx_;
            bool ok = cudaOps_rope_fp16(Q, K, position_ids, seq_len, n_heads, n_kv_heads,
                                        head_dim, rope_theta, dev);
            if (ok)
                cudaDeviceSynchronize();
            return ok;
        }

    } // namespace cuda

    // =========================================================================
    // CUDAEmbeddingKernelT Implementation (in llaminar2 namespace, not cuda)
    // =========================================================================

    bool CUDAEmbeddingKernelT::apply(
        const float *embed_data,
        const int *token_ids,
        int num_tokens,
        int d_model,
        float *output,
        const MPIContext *mpi_ctx,
        int device_idx)
    {
        (void)mpi_ctx;
        int dev = (device_idx >= 0) ? device_idx : device_idx_;
        (void)dev; // Device selection handled by caller setting active device

        cudaError_t err = launch_embedding_lookup(embed_data, token_ids, output,
                                                  num_tokens, d_model, nullptr);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "[CUDAEmbeddingKernelT] Kernel launch failed: %s\n",
                    cudaGetErrorString(err));
            return false;
        }
        cudaDeviceSynchronize();
        return true;
    }

    bool CUDAEmbeddingKernelT::apply_bf16(
        const float *embed_data,
        const int *token_ids,
        int num_tokens,
        int d_model,
        uint16_t *output,
        const MPIContext *mpi_ctx,
        int device_idx)
    {
        (void)embed_data;
        (void)token_ids;
        (void)num_tokens;
        (void)d_model;
        (void)output;
        (void)mpi_ctx;
        (void)device_idx;
        fprintf(stderr, "[CUDAEmbeddingKernelT] BF16 output not yet implemented\n");
        return false;
    }

    bool CUDAEmbeddingKernelT::apply_fp16(
        const float *embed_data,
        const int *token_ids,
        int num_tokens,
        int d_model,
        uint16_t *output,
        const MPIContext *mpi_ctx,
        int device_idx)
    {
        (void)embed_data;
        (void)token_ids;
        (void)num_tokens;
        (void)d_model;
        (void)output;
        (void)mpi_ctx;
        (void)device_idx;
        fprintf(stderr, "[CUDAEmbeddingKernelT] FP16 output not yet implemented\n");
        return false;
    }

    bool CUDAEmbeddingKernelT::apply_q8_1(
        const float *embed_data,
        const int *token_ids,
        int num_tokens,
        int d_model,
        void *output,
        const MPIContext *mpi_ctx,
        int device_idx)
    {
        (void)embed_data;
        (void)token_ids;
        (void)num_tokens;
        (void)d_model;
        (void)output;
        (void)mpi_ctx;
        (void)device_idx;
        fprintf(stderr, "[CUDAEmbeddingKernelT] Q8_1 output not yet implemented\n");
        return false;
    }

    bool CUDAEmbeddingKernelT::apply_tensor(
        const TensorBase *embed_table,
        const int *token_ids,
        int num_tokens,
        int d_model,
        TensorBase *output,
        const MPIContext *mpi_ctx,
        int device_idx)
    {
        // Output must be FP32 (common for all embedding operations)
        if (output->native_type() != TensorType::FP32)
        {
            fprintf(stderr, "[CUDAEmbeddingKernelT] Output must be FP32 tensor\n");
            return false;
        }

        auto *output_fp32 = dynamic_cast<FP32Tensor *>(output);
        if (!output_fp32)
        {
            fprintf(stderr, "[CUDAEmbeddingKernelT] Output tensor cast to FP32 failed\n");
            return false;
        }

        // Get dequantized FP32 data from any tensor type via data()
        // This handles Q4_0, Q8_0, FP32, etc. automatically
        const float *embed_data = embed_table->data();
        if (!embed_data)
        {
            fprintf(stderr, "[CUDAEmbeddingKernelT] Failed to get embedding data\n");
            return false;
        }

        // Determine target CUDA device
        int dev = (device_idx >= 0) ? device_idx : device_idx_;
        DeviceId target_device = DeviceId::cuda(dev);

        // =====================================================================
        // Step 1: Allocate and copy token_ids to GPU
        // =====================================================================
        int *d_token_ids = nullptr;
        size_t token_bytes = static_cast<size_t>(num_tokens) * sizeof(int);
        cudaError_t err = cudaMalloc(&d_token_ids, token_bytes);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "[CUDAEmbeddingKernelT] Failed to allocate GPU memory for token_ids: %s\n",
                    cudaGetErrorString(err));
            return false;
        }
        err = cudaMemcpy(d_token_ids, token_ids, token_bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "[CUDAEmbeddingKernelT] Failed to copy token_ids to GPU: %s\n",
                    cudaGetErrorString(err));
            cudaFree(d_token_ids);
            return false;
        }

        // =====================================================================
        // Step 2: Get GPU pointer for output (coherence handled by GraphExecutor)
        // =====================================================================
        float *d_output = static_cast<float *>(output_fp32->gpu_data_ptr());
        if (!d_output)
        {
            fprintf(stderr, "[CUDAEmbeddingKernelT] Output GPU pointer is null\n");
            cudaFree(d_token_ids);
            return false;
        }

        // =====================================================================
        // Step 3: Get or upload embedding table to GPU
        // =====================================================================
        float *d_embed = nullptr;
        bool need_free_embed = false;

        // Check if embed_table is FP32 and on GPU
        auto *embed_fp32 = dynamic_cast<const FP32Tensor *>(embed_table);
        if (embed_fp32 && embed_fp32->isOnGPU())
        {
            // Fast path: FP32 tensor already on GPU
            d_embed = const_cast<float *>(static_cast<const float *>(embed_fp32->gpu_data_ptr()));
        }
        else
        {
            // Slow path: Need to copy dequantized embedding data to GPU
            size_t vocab_size = embed_table->rows();
            size_t embed_dim = static_cast<size_t>(d_model);
            size_t embed_bytes = vocab_size * embed_dim * sizeof(float);

            err = cudaMalloc(&d_embed, embed_bytes);
            if (err != cudaSuccess)
            {
                fprintf(stderr, "[CUDAEmbeddingKernelT] Failed to allocate GPU memory for embeddings: %s\n",
                        cudaGetErrorString(err));
                cudaFree(d_token_ids);
                return false;
            }

            err = cudaMemcpy(d_embed, embed_data, embed_bytes, cudaMemcpyHostToDevice);
            if (err != cudaSuccess)
            {
                fprintf(stderr, "[CUDAEmbeddingKernelT] Failed to copy embeddings to GPU: %s\n",
                        cudaGetErrorString(err));
                cudaFree(d_embed);
                cudaFree(d_token_ids);
                return false;
            }
            need_free_embed = true;
        }

        // =====================================================================
        // Step 4: Execute kernel with all GPU pointers
        // =====================================================================
        bool result = apply(d_embed, d_token_ids, num_tokens, d_model, d_output, mpi_ctx, device_idx);

        // =====================================================================
        // Step 5: Cleanup temporary GPU buffers
        // =====================================================================
        if (need_free_embed)
        {
            cudaFree(d_embed);
        }
        cudaFree(d_token_ids);

        return result;
    }

} // namespace llaminar2
