/**
 * @file INT8PackedGemm.h
 * @brief INT8×INT8→INT32 GEMM adapter with AVX512 VNNI backend
 *
 * This file provides a high-level interface for INT8 matrix multiplication
 * using AVX512 VNNI instructions, with automatic packing and dequantization.
 *
 * @author David Sanftenberg
 */

#pragma once

#include "../../../../tensors/TensorKernels.h"
#include <memory>
#include <vector>

// Forward declaration for IQuantizedGemmVariant
namespace llaminar
{
    namespace v2
    {
        namespace kernels
        {
            class IQuantizedGemmVariant;
        }
    }
}

namespace llaminar2
{
    namespace kernels
    {
        namespace gemm
        {

            /**
             * @brief Factory function to create INT8 GEMM kernel
             *
             * Creates an optimized INT8×INT8→INT32 GEMM kernel using AVX512 VNNI.
             * The kernel handles:
             * - Automatic packing of A and B matrices
             * - INT8 computation with VNNI instructions
             * - Dequantization to float output
             * - Cache blocking for large matrices
             *
             * @param A Tensor containing INT8 matrix A (if available for metadata)
             * @param B Tensor containing INT8 matrix B (if available for metadata)
             * @return Unique pointer to ITensorGemm kernel, or nullptr if INT8 unsupported
             */
            std::unique_ptr<ITensorGemm> createINT8PackedGemm(
                const TensorBase *A = nullptr,
                const TensorBase *B = nullptr);

            /**
             * @brief Check if INT8 GEMM is supported on this CPU
             *
             * Requires AVX512F and AVX512VNNI instruction sets.
             *
             * @return true if INT8 GEMM is available, false otherwise
             */
            bool isINT8GemmSupported();

            /**
             * @brief Register all INT8 VNNI micro-kernel variants for auto-tuning
             *
             * Creates IQuantizedGemmVariant adapters for all INT8 micro-kernel configurations.
             * These variants can be used by the auto-tuner to find optimal tile/unroll/prefetch
             * parameters for INT8×INT8→INT32 GEMM operations.
             *
             * The variants use:
             * - ISA: AVX512VNNI (INT8 dpbusd instruction)
             * - MR (tile rows): {1, 2, 4, 8, 16, 32}
             * - NR (tile cols): {1, 2, 4, 6, 8, 16, 32}
             * - UNROLL_K: {1, 2, 4, 8}
             * - PREFETCH_DIST: {0, 1, 2, 3}
             * - Register constraint: MR × NR ≤ 48 (AVX512 register file)
             *
             * @return Vector of INT8 variant adapters (typically ~400 variants)
             */
            std::vector<std::unique_ptr<llaminar::v2::kernels::IQuantizedGemmVariant>>
            registerINT8MicroKernelVariants();

        } // namespace gemm
    } // namespace kernels
} // namespace llaminar2
