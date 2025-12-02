/**
 * @file GemmOp.h
 * @brief Self-validating GEMM (matrix multiplication) operation
 *
 * Encapsulates the full weight GEMM workflow:
 * 1. Validate activation/weight/output tensors
 * 2. Create GEMM kernel from weight tensor
 * 3. Execute kernel with error handling
 * 4. Capture snapshot (if enabled)
 *
 * Supports both:
 * - Weight GEMM: C = A @ W^T (activations × weight tensor)
 * - Activation GEMM: C = A @ B^T (activations × activations)
 *
 * Usage:
 * @code
 * GemmOp gemm;
 *
 * // Weight projection (creates kernel from weight tensor)
 * TRY_OP(gemm(hidden, layer.wq.get(), Q_buf, seq_len, n_heads * head_dim, d_model,
 *             "layer0_Q_PROJ", mpi, device));
 *
 * // Activation matmul (for attention: Q @ K^T)
 * TRY_OP(gemm.activations(Q, K, scores, m, n, k, true, scale, 0.0f,
 *                         "layer0_ATTN_SCORES", mpi, device));
 * @endcode
 *
 * @author David Sanftenberg
 */

#pragma once

#include "Op.h"

namespace llaminar2
{

    /**
     * @brief Self-validating GEMM operation for weight projections
     *
     * Replaces the verbose pattern:
     * @code
     * VALIDATE_KERNEL(gemm, layer.wq->createGemm(), "Q GEMM kernel");
     * VALIDATE_OP(gemm->multiply_activations(hidden->data(), nullptr, Q->mutable_data(),
     *             seq_len, n_heads * head_dim, d_model, true, 1.0f, 0.0f, mpi, device), "Q projection");
     * CAPTURE_SNAPSHOT_VIEW("layer0_Q_PROJ", Q, seq_len, n_heads * head_dim);
     * @endcode
     *
     * With a single call:
     * @code
     * TRY_OP(gemm(hidden, layer.wq.get(), Q_buf, seq_len, q_dim, k_dim, "Q_PROJ", mpi, device));
     * @endcode
     */
    class GemmOp : public OpBase
    {
    public:
        const char *name() const override { return "GemmOp"; }

        /**
         * @brief Execute weight projection: C = A @ W^T
         *
         * Creates GEMM kernel from weight tensor and executes.
         * Weight tensor determines the kernel type (quantized, FP32, etc.).
         *
         * @param A Input activations tensor [m, k]
         * @param W Weight tensor [n, k] (transposed storage)
         * @param C Output tensor [m, n]
         * @param m Number of rows (sequence length)
         * @param n Number of output features
         * @param k Number of input features
         * @param snapshot_key Snapshot identifier (nullptr to skip)
         * @param mpi_ctx MPI context (nullptr for single-node)
         * @param device_idx Device index (-1 for CPU)
         * @param alpha Scale factor for A@W (default: 1.0f)
         * @param beta Scale factor for existing C (default: 0.0f)
         *
         * @return true on success, false on validation or execution failure
         */
        bool operator()(
            const TensorBase *A,
            TensorBase *W,
            TensorBase *C,
            int m,
            int n,
            int k,
            const char *snapshot_key = nullptr,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1,
            float alpha = 1.0f,
            float beta = 0.0f)
        {
            // 1. Validate inputs
            if (!validateTensor(A, "activation (A)"))
                return false;
            if (!validateTensor(W, "weight (W)"))
                return false;
            if (!validateTensor(C, "output (C)"))
                return false;
            if (!validateDimensions(m, k, "input A"))
                return false;
            if (!validateDimensions(m, n, "output C"))
                return false;

            // 2. Get cached kernel from weight tensor (avoids expensive repacking every call)
            auto *gemm_kernel = W->getOrCreateGemm();
            if (!gemm_kernel)
            {
                logError("failed to get GEMM kernel from weight tensor");
                return false;
            }

            // 3. Execute: C = A @ W^T (transpose_B = true for weights)
            if (!gemm_kernel->multiply_activations(
                    A->data(),
                    nullptr, // B is packed in kernel (weight tensor)
                    C->mutable_data(),
                    m, n, k,
                    true, // transpose_B
                    alpha, beta,
                    mpi_ctx,
                    device_idx))
            {
                logError("GEMM kernel execution failed");
                return false;
            }

            // Note: Snapshot capture is handled by the calling pipeline
            (void)snapshot_key;

            return true;
        }

        /**
         * @brief Execute activation-activation matmul: C = A @ B^T (or A @ B)
         *
         * For attention: Q @ K^T or scores @ V.
         * Does not use a weight tensor - operates on activation buffers directly.
         *
         * @param A Left activation tensor [m, k]
         * @param B Right activation tensor [n, k] if transpose_B, [k, n] otherwise
         * @param C Output tensor [m, n]
         * @param m Number of rows in A
         * @param n Number of rows in B (if transpose_B) or columns in B
         * @param k Inner dimension
         * @param transpose_B Whether to transpose B (true for Q@K^T)
         * @param alpha Scale factor (e.g., 1/sqrt(d_k) for attention)
         * @param beta Scale factor for existing C
         * @param snapshot_key Snapshot identifier (nullptr to skip)
         * @param mpi_ctx MPI context (nullptr for single-node)
         * @param device_idx Device index (-1 for CPU)
         *
         * @return true on success, false on failure
         */
        bool activations(
            TensorBase *A,
            TensorBase *B,
            TensorBase *C,
            int m,
            int n,
            int k,
            bool transpose_B = true,
            float alpha = 1.0f,
            float beta = 0.0f,
            const char *snapshot_key = nullptr,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1)
        {
            // 1. Validate inputs
            if (!validateTensor(A, "activation A"))
                return false;
            if (!validateTensor(B, "activation B"))
                return false;
            if (!validateTensor(C, "output C"))
                return false;
            if (!validateDimensions(m, k, "input A"))
                return false;

            // 2. Create kernel from A (any activation tensor can create GEMM kernel)
            auto gemm_kernel = A->createGemm();
            if (!gemm_kernel)
            {
                logError("failed to create GEMM kernel for activation matmul");
                return false;
            }

            // 3. Execute: C = A @ B^T (or A @ B)
            if (!gemm_kernel->multiply_activations(
                    A->data(),
                    B->data(),
                    C->mutable_data(),
                    m, n, k,
                    transpose_B,
                    alpha, beta,
                    mpi_ctx,
                    device_idx))
            {
                logError("activation GEMM execution failed");
                return false;
            }

            // Note: Snapshot capture is handled by the calling pipeline
            (void)snapshot_key;

            return true;
        }

        /**
         * @brief Execute weight projection with raw float pointers
         *
         * Alternative for when activation data is not in a TensorBase.
         */
        bool operator()(
            const float *A_data,
            TensorBase *W,
            float *C_data,
            int m,
            int n,
            int k,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1,
            float alpha = 1.0f,
            float beta = 0.0f)
        {
            // 1. Validate inputs
            if (!validatePointer(A_data, "activation data (A)"))
                return false;
            if (!validateTensor(W, "weight (W)"))
                return false;
            if (!validatePointer(C_data, "output data (C)"))
                return false;

            // 2. Get cached kernel from weight tensor
            auto *gemm_kernel = W->getOrCreateGemm();
            if (!gemm_kernel)
            {
                logError("failed to get GEMM kernel from weight tensor");
                return false;
            }

            // 3. Execute
            if (!gemm_kernel->multiply_activations(
                    A_data,
                    nullptr,
                    C_data,
                    m, n, k,
                    true,
                    alpha, beta,
                    mpi_ctx,
                    device_idx))
            {
                logError("GEMM kernel execution failed");
                return false;
            }

            return true;
        }
    };

} // namespace llaminar2
