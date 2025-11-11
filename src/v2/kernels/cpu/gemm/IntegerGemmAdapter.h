/**
 * @file IntegerGemmAdapter.h
 * @brief ITensorGemm adapter for Q8_0 integer GEMM path
 *
 * This adapter wraps IntegerGemmKernelTemplate to provide the ITensorGemm interface,
 * enabling integration with the existing tensor abstraction layer.
 *
 * Key difference from FP32 GEMM adapter:
 * - Accepts Q8_0Tensor inputs (not FP32Tensor)
 * - Outputs Q8_0Tensor (not FP32Tensor)
 * - Uses Q8_0BlockProvider (cached or zero-copy weight access)
 *
 * @author David Sanftenberg
 * @date November 2025
 */

#pragma once

#include "IntegerGemmKernelTemplate.h"
#include "GemmWeightCache.h"
#include "../../../tensors/TensorKernels.h"
#include "../../../tensors/Tensors.h"
#include <memory>
#include <stdexcept>

namespace llaminar2
{
    namespace kernels
    {
        namespace gemm
        {

            /**
             * @brief ITensorGemm implementation for Q8_0 integer GEMM
             *
             * This adapter provides a tensor-level interface to the integer GEMM kernel.
             * It handles:
             * - Tensor type checking (ensure inputs are Q8_0 or compatible)
             * - Weight accessor creation (decode weights to Q8_0)
             * - Dimension validation
             * - Output allocation (if needed)
             *
             * Usage:
             * ```cpp
             * Q8_0Tensor A({M, K});  // Pre-quantized activations
             * IQ4_NLTensor B({K, N}); // Quantized weights
             * Q8_0Tensor C({M, N});   // Output (pre-allocated or auto-allocated)
             *
             * auto gemm = createIntegerGemm(&A, &B);
             * gemm->multiply(A.data(), C.data(), M, N, K, weight_accessor);
             * ```
             */
            template <typename ISA, int MR, int NR, int UNROLL_K = 4, int PREFETCH_DIST = 2>
            class IntegerGemmAdapter : public ITensorGemm
            {
            public:
                using Kernel_t = IntegerGemmKernel<ISA, MR, NR, UNROLL_K, PREFETCH_DIST>;

                /**
                 * @brief Construct adapter with optional pre-allocated tensors
                 *
                 * @param A Optional A tensor (for metadata)
                 * @param B Weight tensor (required for accessor creation)
                 */
                IntegerGemmAdapter(const Q8_0Tensor *A, const TensorBase *B)
                    : A_(A), B_(B)
                {
                    if (!B)
                    {
                        throw std::invalid_argument("IntegerGemmAdapter: B tensor is required");
                    }

                    // Create weight provider (cached or zero-copy)
                    weight_provider_ = createWeightProvider(B);
                    if (!weight_provider_)
                    {
                        throw std::invalid_argument("IntegerGemmAdapter: Unsupported weight format");
                    }
                }

                /**
                 * @brief Execute Q8_0 integer GEMM
                 *
                 * @param A Q8_0 activation blocks (input)
                 * @param C Q8_0 result blocks (output)
                 * @param m Number of rows
                 * @param n Number of columns
                 * @param k Inner dimension
                 * @param decoder Unused (kept for interface compatibility)
                 * @param transpose_B Unused (weights assumed transposed)
                 * @param alpha Unused (Q8_0 path doesn't support scaling yet)
                 * @param beta Unused (Q8_0 path doesn't support accumulation yet)
                 * @return true on success
                 */
                bool multiply(
                    const float *A_fp32,
                    float *C_fp32,
                    int m, int n, int k,
                    const ITensorGemmTileDataProvider *decoder,
                    bool transpose_B = false,
                    float alpha = 1.0f,
                    float beta = 0.0f) override
                {
                    // This interface expects FP32 pointers, but we need Q8_0
                    // This is a design mismatch - we should use a different interface
                    // For now, throw an error
                    throw std::runtime_error(
                        "IntegerGemmAdapter::multiply with FP32 pointers not supported. "
                        "Use multiplyQ8_0() instead or convert tensors.");
                }

                /**
                 * @brief Execute Q8_0 integer GEMM (native interface)
                 *
                 * This is the native interface for integer GEMM. It accepts Q8_0 blocks
                 * directly without requiring FP32 conversion.
                 *
                 * @param A Q8_0 activation blocks [m × k_blocks]
                 * @param C Q8_0 result blocks [m × n_blocks]
                 * @param m Number of rows
                 * @param n Number of columns
                 * @param k Number of inner dimension elements
                 * @return true on success
                 */
                bool multiplyQ8_0(
                    const Q8_0Block *A,
                    Q8_0Block *C,
                    int m, int n, int k)
                {
                    return Kernel_t::multiply(A, *weight_provider_, C, m, n, k);
                }

                /**
                 * @brief Get configuration string for debugging
                 */
                std::string getConfigString() const
                {
                    std::ostringstream oss;
                    oss << "IntegerGemm_MR" << MR << "_NR" << NR
                        << "_Unroll" << UNROLL_K << "_Prefetch" << PREFETCH_DIST;
                    return oss.str();
                }

            private:
                const Q8_0Tensor *A_;
                const TensorBase *B_;
                std::unique_ptr<Q8_0BlockProvider> weight_provider_;
            };

            /**
             * @brief Factory function for integer GEMM adapter
             *
             * @param A Q8_0 activation tensor
             * @param B Quantized weight tensor (IQ4_NL, Q6_K, Q8_0, etc.)
             * @return Unique pointer to ITensorGemm adapter
             */
            template <typename ISA = simd::AVX512VNNITag>
            std::unique_ptr<ITensorGemm> createIntegerGemm(
                const Q8_0Tensor *A,
                const TensorBase *B)
            {
                // Default configuration: MR=8, NR=8, UNROLL_K=4, PREFETCH_DIST=2
                return std::make_unique<IntegerGemmAdapter<ISA, 8, 8, 4, 2>>(A, B);
            }

            /**
             * @brief Extended factory with template parameters
             */
            template <typename ISA, int MR, int NR, int UNROLL_K = 4, int PREFETCH_DIST = 2>
            std::unique_ptr<ITensorGemm> createIntegerGemmWithConfig(
                const Q8_0Tensor *A,
                const TensorBase *B)
            {
                return std::make_unique<IntegerGemmAdapter<ISA, MR, NR, UNROLL_K, PREFETCH_DIST>>(A, B);
            }

        } // namespace gemm
    } // namespace kernels
} // namespace llaminar2
