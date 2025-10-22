/**
 * @file ITensorGemm.h
 * @brief Abstract interface for tensor-specific GEMM operations.
 *
 * This interface enables tensors to provide optimized GEMM implementations
 * tailored to their storage format (FP32, BF16, quantized, etc.).
 *
 * Pattern:
 *  - Each tensor type (FP32Tensor, IQ4_NLTensor, IQ4_XSTensor, Q8_0Tensor, etc.)
 *    can optionally implement ITensorGemm to provide an optimized kernel.
 *  - For quantized tensors, this enables fusing dequantization with matrix
 *    multiplication, avoiding full materialization of dequantized weights.
 *  - For FP32/BF16 tensors, this provides a consistent interface for BLAS calls.
 *  - adaptiveMatMul checks if the weight tensor supports createGemm(), and if so,
 *    uses the optimized path.
 *  - Fallback: If createGemm() returns nullptr, fall back to generic BLAS.
 *
 * Benefits:
 *  - Centralized abstraction: ~50 lines of interface, ~250 lines per implementation.
 *  - No code duplication: Type detection happens once in adaptiveMatMul.
 *  - Extensible: New tensor formats just implement this interface.
 *  - Opt-in: Tensors without optimized kernels return nullptr, use generic fallback.
 *
 * Memory Model:
 *  - Implementations may use tiled/streaming operations to control memory footprint.
 *  - The interface does not mandate specific tiling strategies.
 *
 * @author David Sanftenberg
 */

#pragma once

#include <memory>
#include <cstddef>

namespace llaminar
{

    /**
     * @brief Abstract interface for tensor-specific GEMM operations.
     *
     * Implementations provide a multiply() method that computes:
     *   C = alpha * A @ B^T + beta * C
     * where B is the weight tensor (possibly quantized), and A/C are activations.
     *
     * For quantized tensors, the implementation may fuse dequantization with GEMM
     * to avoid full weight materialization. For FP32/BF16 tensors, this provides
     * a consistent interface for BLAS calls.
     */
    class ITensorGemm
    {
    public:
        virtual ~ITensorGemm() = default;

        /**
         * @brief Perform tensor-specific GEMM: C = alpha * A @ B^T + beta * C
         *
         * @param A Input activations [m, k] in row-major order (FP32)
         * @param C Output matrix [m, n] in row-major order (FP32)
         * @param m Number of rows in A and C
         * @param n Number of columns in C (rows in B before transpose)
         * @param k Number of columns in A (columns in B)
         * @param transpose_B If true, treat B as [n, k] and transpose; if false, B is [k, n]
         * @param alpha Scaling factor for A @ B^T
         * @param beta Scaling factor for existing C (0.0 = overwrite C)
         * @param row_offset Starting row index in B (for distributed computation, default=0)
         * @param row_count Number of rows to use from B (default=n, use all rows)
         *
         * @return true if operation succeeded, false otherwise
         *
         * @note B (weight tensor) is implicitly provided by the implementation.
         *       For quantized tensors, B is in compressed format and handled internally.
         *       For FP32/BF16 tensors, B is in native format.
         *       row_offset/row_count allow MPI ranks to compute only their portion of output features.
         */
        virtual bool multiply(const float *A, float *C,
                              int m, int n, int k,
                              bool transpose_B = true,
                              float alpha = 1.0f,
                              float beta = 0.0f,
                              int row_offset = 0,
                              int row_count = -1) = 0;

        /**
         * @brief Check if this implementation supports the given operation size.
         *
         * Allows implementations to reject very small or very large operations
         * and fall back to generic BLAS.
         *
         * @param m Number of rows in A
         * @param n Number of columns in C
         * @param k Number of columns in A
         *
         * @return true if this implementation can handle (m, n, k), false to fall back
         */
        virtual bool supports(int m, int n, int k) const
        {
            // Default: support all sizes
            return true;
        }

        /**
         * @brief Get a descriptive name for this GEMM implementation.
         *
         * Used for logging and diagnostics.
         *
         * @return Name string (e.g., "FP32_BLAS_Gemm", "IQ4_NL_FusedGemm")
         */
        virtual const char *name() const = 0;

        /**
         * @brief Check if this implementation supports BF16 activations.
         *
         * @return true if multiply_bf16() is implemented, false otherwise
         */
        virtual bool supports_bf16() const { return false; }

        /**
         * @brief Perform tensor-specific GEMM with BF16 activations: C = alpha * A @ B^T + beta * C
         *
         * @param A_bf16 Input activations [m, k] in row-major order (BF16, stored as uint16_t)
         * @param C Output matrix [m, n] in row-major order (FP32)
         * @param m Number of rows in A and C
         * @param n Number of columns in C (rows in B before transpose)
         * @param k Number of columns in A (columns in B)
         * @param transpose_B If true, treat B as [n, k] and transpose; if false, B is [k, n]
         * @param alpha Scaling factor for A @ B^T
         * @param beta Scaling factor for existing C (0.0 = overwrite C)
         * @param row_offset Starting row index in B (for distributed computation, default=0)
         * @param row_count Number of rows to use from B (default=n, use all rows)
         *
         * @return true if operation succeeded, false otherwise
         *
         * @note Default implementation returns false (not supported).
         *       Override in derived classes that implement BF16 support.
         */
        virtual bool multiply_bf16(const uint16_t *A_bf16, float *C,
                                   int m, int n, int k,
                                   bool transpose_B = true,
                                   float alpha = 1.0f,
                                   float beta = 0.0f,
                                   int row_offset = 0,
                                   int row_count = -1)
        {
            return false; // Not supported by default
        }
    };

} // namespace llaminar
