/**
 * @file INT8GemmKernel.h
 * @brief CPU INT8 GEMM kernel using AVX512-VNNI
 *
 * @author David Sanftenberg
 */

#pragma once

#include "../../tensors/TensorKernels.h"
#include "../../tensors/Tensors.h"

namespace llaminar2
{

    /**
     * @brief CPU INT8 GEMM kernel (AVX512-VNNI acceleration)
     *
     * Uses block-quantized INT8 format with per-block FP32 scales:
     * - Weight blocks: int8_t[block_size] with float scale
     * - Activation blocks: int8_t[block_size] with float scale
     * - Output: FP32 = sum(int8_A * int8_B * scale_A * scale_B)
     *
     * Hardware acceleration:
     * - AVX512-VNNI: VPDPBUSD instruction (4× INT8 throughput vs FP32)
     * - Fallback: AVX512F/AVX2 scalar INT8 accumulation
     */
    class INT8GemmKernel : public ITensorGemm
    {
    public:
        /**
         * @brief Construct GEMM kernel for INT8 weight tensor
         *
         * @param weight_tensor INT8 weight tensor (B matrix)
         */
        explicit INT8GemmKernel(const INT8Tensor *weight_tensor)
            : weight_tensor_(weight_tensor)
        {
        }

        ~INT8GemmKernel() override = default;

        bool supports_device(int device_idx) const override
        {
            return device_idx == -1; // CPU only
        }

        bool multiply(
            const float *A, float *C,
            int m, int n, int k,
            bool transpose_B = true,
            float alpha = 1.0f, float beta = 0.0f,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override;

        bool multiply_activations(
            const float *A, const float *B, float *C,
            int m, int n, int k,
            bool transpose_B = true,
            float alpha = 1.0f, float beta = 0.0f,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override;

        bool multiply_activations_strided(
            const float *A, const float *B, float *C,
            int m, int n, int k,
            int lda, int ldb, int ldc,
            bool transpose_B = true,
            float alpha = 1.0f, float beta = 0.0f,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override;

        /**
         * @brief INT8 GEMM with INT32 output (no dequantization)
         *
         * Returns raw INT32 accumulator from OneDNN s8s8s32 matmul.
         * This is the foundation for full INT8 pipelines where activations
         * remain in INT8/INT32 between layers.
         *
         * Output: C_int32[i,j] = Σ(A_int8[i,k] × B_int8[k,j])
         *
         * To convert to FP32: C_fp32[i,j] = C_int32[i,j] × scale_A[i] × scale_B[j]
         * To requantize to INT8: See INT32→INT8 requantization kernel
         *
         * @param A Input FP32 activation matrix [m, k] (will be quantized per-row)
         * @param C_int32 Output INT32 accumulator matrix [m, n]
         * @param m Number of rows
         * @param n Number of columns in output
         * @param k Number of columns in A / rows in B
         * @param transpose_B Whether B is stored transposed
         * @param[out] A_row_scales_out Optional: output per-row scales for A [m]
         * @return true if successful
         */
        bool multiply_int32(
            const float *A, int32_t *C_int32,
            int m, int n, int k,
            bool transpose_B = true,
            float *A_row_scales_out = nullptr);

    private:
        const INT8Tensor *weight_tensor_;

        /**
         * @brief Convert FP32 activations to per-row quantized INT8
         *
         * OneDNN-compatible quantization with per-row scales (mask = 1<<0).
         * Each row of A gets independent scale for better accuracy.
         *
         * @param A_fp32 Input FP32 activation matrix [m, k]
         * @param m Number of rows
         * @param k Number of columns
         * @param[out] A_int8 Output INT8 data
         * @param[out] row_scales Output per-row scales [m]
         */
        void quantize_activations_per_row(
            const float *A_fp32,
            int m, int k,
            int8_t *A_int8,
            float *row_scales) const;

        /**
         * @brief INT8×INT8 GEMM using OneDNN s8s8s32 matmul with flexible output
         *
         * Unified implementation that can output either:
         * - FP32: Dequantized output (current production path)
         * - INT32: Raw accumulator (for full INT8 pipelines)
         *
         * @param A_int8 INT8 activation matrix [m, k]
         * @param A_row_scales Per-row scales for A [m]
         * @param B_int8 INT8 weight matrix [k, n] or [n, k] if transposed
         * @param B_col_scales Per-column scales for B [n]
         * @param C_fp32 FP32 output matrix [m, n] (if not nullptr)
         * @param C_int32 INT32 output matrix [m, n] (if not nullptr)
         * @param m Number of rows in A and C
         * @param n Number of columns in B and C
         * @param k Number of columns in A and rows in B
         * @param transpose_B Whether B is transposed
         * @param alpha Scaling factor for A*B (FP32 output only)
         * @param beta Scaling factor for existing C (FP32 output only)
         */
        void gemm_int8_perchannel_flexible(
            const int8_t *A_int8,
            const float *A_row_scales,
            const int8_t *B_int8,
            const float *B_col_scales,
            float *C_fp32,
            int32_t *C_int32,
            int m, int n, int k,
            bool transpose_B,
            float alpha, float beta) const;

        /**
         * @brief INT8×INT8 GEMM using OneDNN s8s8s32 matmul (FP32 output)
         *
         * Legacy wrapper for backward compatibility.
         * Calls gemm_int8_perchannel_flexible with C_int32=nullptr.
         */
        void gemm_int8_perchannel(
            const int8_t *A_int8,
            const float *A_row_scales,
            const int8_t *B_int8,
            const float *B_col_scales,
            float *C,
            int m, int n, int k,
            bool transpose_B,
            float alpha, float beta) const;
    };

} // namespace llaminar2
