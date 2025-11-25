/**
 * @file FusedDualGEMM.h
 * @brief Fused dual GEMM for FFN gate/up projections with shared activation caching
 *
 * Fuses gate and up projections in FFN blocks by:
 * 1. Caching the activation quantization state from the first GEMM
 * 2. Reusing the Q8_1 blocks for the second GEMM
 * 3. Both GEMMs output FP32 (via Q8_1GemmKernel)
 *
 * This kernel wraps Q8_1GemmKernel for both projections, leveraging its:
 * - On-the-fly FP32 → Q8_1 quantization
 * - Direct Q8_1 weight consumption (no dequantization)
 * - FP32 output with optional fused bias
 *
 * Performance Benefits:
 * - Potential for shared activation quantization (future optimization)
 * - Better cache locality (input stays hot between GEMMs)
 * - Consistent FP32 residual stream
 *
 * Algorithm:
 * 1. Gate GEMM: FP32[m,k] → Q8_1GemmKernel → FP32[m,n] (+ optional bias)
 * 2. Up GEMM: FP32[m,k] → Q8_1GemmKernel → FP32[m,n] (+ optional bias)
 * 3. Downstream SwiGLU operates on FP32 outputs
 *
 * Fusion Chain (New Architecture):
 *   [FP32 residual] → RMSNorm → FusedDualGEMM → SwiGLU → [FP32 residual]
 *
 * @author David Sanftenberg
 * @date 2025-11-23
 * @updated 2025-11-24 - Reworked for FP32 residual architecture
 */

#pragma once

#include "../CPUKernelBase.h"
#include "../../../tensors/TensorKernels.h"
#include "../../../tensors/Tensors.h"
#include "../gemm_v4/Q8_1GemmKernel.h"
#include <cstdint>
#include <vector>
#include <memory>

namespace llaminar2
{
    /**
     * @brief Fused dual GEMM kernel for FFN gate/up projections
     *
     * Wraps two Q8_1GemmKernel instances for gate and up projections.
     * Both take FP32 input and produce FP32 output.
     *
     * Usage:
     *   FusedDualGEMM kernel(gate_weight_q8_1, up_weight_q8_1);
     *   kernel.execute(input_fp32, gate_output_fp32, up_output_fp32,
     *                  gate_bias, up_bias, m, n, k);
     */
    class FusedDualGEMM : public CPUKernelBase
    {
    public:
        /**
         * @brief Construct fused dual GEMM kernel with Q8_1 weight tensors
         *
         * @param gate_weight Q8_1 quantized gate projection weights [n, k]
         * @param up_weight Q8_1 quantized up projection weights [n, k]
         */
        FusedDualGEMM(const Q8_1Tensor *gate_weight, const Q8_1Tensor *up_weight);
        ~FusedDualGEMM() override = default;

        // =============================================================================
        // CPUKernelBase Interface (Fusion Framework)
        // =============================================================================

        /**
         * @brief Get kernel I/O contract for fusion pattern detection
         */
        KernelContract get_contract() const override
        {
            return KernelContract{
                .accepted_input_formats = {TensorFormat::FP32}, // Accept FP32 activations
                .output_format = TensorFormat::FP32,            // Produce FP32 outputs
                .supports_inplace = false,                      // Need separate output buffers
                .is_fusable = true                              // Can fuse with SwiGLU
            };
        }

        bool supports_fusion() const override
        {
            return true;
        }

        TensorFormat preferred_fusion_format() const override
        {
            return TensorFormat::FP32; // FP32 output for downstream fusion
        }

        // =============================================================================
        // Execution Interface
        // =============================================================================

        /**
         * @brief Execute fused dual GEMM
         *
         * Performs:
         * 1. Gate GEMM: FP32[m,k] × Q8_1_weight[n,k] → FP32[m,n] (+ bias)
         * 2. Up GEMM: FP32[m,k] × Q8_1_weight[n,k] → FP32[m,n] (+ bias)
         *
         * @param input Input activations [m, k] FP32
         * @param gate_output Output gate activations [m, n] FP32
         * @param up_output Output up activations [m, n] FP32
         * @param gate_bias Optional gate bias [n] FP32 (nullptr if none)
         * @param up_bias Optional up bias [n] FP32 (nullptr if none)
         * @param m Batch size (sequence length)
         * @param n Hidden dimension (output features)
         * @param k Input features
         * @param ctx MPI context (optional)
         * @param device_idx Device index (-1 for CPU)
         * @return true on success, false on error
         */
        bool execute(
            const float *input,
            float *gate_output,
            float *up_output,
            const float *gate_bias,
            const float *up_bias,
            int m, int n, int k,
            const MPIContext *ctx = nullptr,
            int device_idx = -1);

        /**
         * @brief Check if kernel supports given device
         * @param device_idx Device index (-1 = CPU)
         * @return true if device is supported
         */
        bool supports_device(int device_idx) const
        {
            return device_idx == -1; // CPU only for now
        }

    private:
        // Q8_1 GEMM kernels for gate and up projections
        std::unique_ptr<gemm_v4::Q8_1GemmKernel> gate_gemm_;
        std::unique_ptr<gemm_v4::Q8_1GemmKernel> up_gemm_;
    };

} // namespace llaminar2
