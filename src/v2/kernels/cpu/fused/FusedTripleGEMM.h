/**
 * @file FusedTripleGEMM.h
 * @brief Fused triple GEMM for attention Q/K/V projections
 *
 * Fuses Q, K, V projections in attention blocks by:
 * 1. Executing 3 Q8_1 GEMMs with shared input (cache locality)
 * 2. All GEMMs output FP32 (via Q8_1GemmKernel)
 *
 * This kernel wraps Q8_1GemmKernel for all three projections, leveraging:
 * - On-the-fly FP32 → Q8_1 quantization
 * - Direct Q8_1 weight consumption (no dequantization)
 * - FP32 output with optional fused bias
 *
 * Performance Benefits:
 * - Potential for shared activation quantization (future optimization)
 * - Better cache locality (input stays hot across 3 GEMMs)
 * - Consistent FP32 residual stream
 *
 * Algorithm:
 * 1. Q GEMM: FP32[m,k] → Q8_1GemmKernel → FP32[m,n_q] (+ optional bias)
 * 2. K GEMM: FP32[m,k] → Q8_1GemmKernel → FP32[m,n_kv] (+ optional bias)
 * 3. V GEMM: FP32[m,k] → Q8_1GemmKernel → FP32[m,n_kv] (+ optional bias)
 *
 * GQA Support: n_q and n_kv can differ for Grouped Query Attention
 *
 * Fusion Chain (New Architecture):
 *   [FP32 residual] → RMSNorm → FusedTripleGEMM → Attention → [FP32 residual]
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
     * @brief Fused triple GEMM kernel for attention Q/K/V projections
     *
     * Wraps three Q8_1GemmKernel instances for Q, K, V projections.
     * All take FP32 input and produce FP32 output.
     *
     * Usage:
     *   FusedTripleGEMM kernel(q_weight_q8_1, k_weight_q8_1, v_weight_q8_1);
     *   kernel.execute(input_fp32, q_out_fp32, k_out_fp32, v_out_fp32,
     *                  q_bias, k_bias, v_bias, m, n_q, n_kv, k);
     */
    class FusedTripleGEMM : public CPUKernelBase
    {
    public:
        /**
         * @brief Construct fused triple GEMM kernel with Q8_1 weight tensors
         *
         * @param q_weight Q8_1 quantized Q projection weights [n_q, k]
         * @param k_weight Q8_1 quantized K projection weights [n_kv, k]
         * @param v_weight Q8_1 quantized V projection weights [n_kv, k]
         */
        FusedTripleGEMM(const Q8_1Tensor *q_weight, const Q8_1Tensor *k_weight, const Q8_1Tensor *v_weight);
        ~FusedTripleGEMM() override = default;

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
                .is_fusable = true                              // Can fuse with attention kernel
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
         * @brief Execute fused triple GEMM
         *
         * Performs:
         * 1. Q GEMM: FP32[m,k] × Q8_1_weight[n_q,k] → FP32[m,n_q] (+ bias)
         * 2. K GEMM: FP32[m,k] × Q8_1_weight[n_kv,k] → FP32[m,n_kv] (+ bias)
         * 3. V GEMM: FP32[m,k] × Q8_1_weight[n_kv,k] → FP32[m,n_kv] (+ bias)
         *
         * @param input Input activations [m, k] FP32
         * @param q_output Output Q activations [m, n_q] FP32
         * @param k_output Output K activations [m, n_kv] FP32
         * @param v_output Output V activations [m, n_kv] FP32
         * @param q_bias Optional Q bias [n_q] FP32 (nullptr if none)
         * @param k_bias Optional K bias [n_kv] FP32 (nullptr if none)
         * @param v_bias Optional V bias [n_kv] FP32 (nullptr if none)
         * @param m Batch size (sequence length)
         * @param n_q Q output dimension (n_heads * head_dim)
         * @param n_kv K/V output dimension (n_kv_heads * head_dim)
         * @param k Input features
         * @param ctx MPI context (optional)
         * @param device_idx Device index (-1 for CPU)
         * @return true on success, false on error
         */
        bool execute(
            const float *input,
            float *q_output,
            float *k_output,
            float *v_output,
            const float *q_bias,
            const float *k_bias,
            const float *v_bias,
            int m, int n_q, int n_kv, int k,
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
        // Q8_1 GEMM kernels for Q, K, V projections
        std::unique_ptr<gemm_v4::Q8_1GemmKernel> q_gemm_;
        std::unique_ptr<gemm_v4::Q8_1GemmKernel> k_gemm_;
        std::unique_ptr<gemm_v4::Q8_1GemmKernel> v_gemm_;
    };

} // namespace llaminar2
