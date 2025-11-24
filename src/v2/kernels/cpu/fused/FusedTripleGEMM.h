/**
 * @file FusedTripleGEMM.h
 * @brief Fused triple GEMM for attention Q/K/V projections with shared input quantization
 *
 * Fuses Q, K, V projections in attention blocks by:
 * 1. Quantizing input activations ONCE (FP32 → INT8 with per-row scales)
 * 2. Performing 3 INT8×INT8 GEMMs with same input (Q_proj, K_proj, V_proj)
 * 3. Outputting INT32 accumulators (defer dequantization to attention kernel)
 *
 * Performance Benefits:
 * - Eliminates 2 redundant quantization passes (Q, K, V share input)
 * - Defers dequantization until after projections (saves 3 dequant passes)
 * - Better cache locality (input stays hot across 3 GEMMs)
 *
 * Expected Speedup: 4-6% in attention block (1 quantization instead of 3)
 *
 * Algorithm:
 * 1. Quantize input once: FP32[m,k] → INT8[m,k] + row_scales[m]
 * 2. Q GEMM: INT8[m,k] × INT8_weight_q[k,n] → INT32_q[m,n]
 * 3. K GEMM: INT8[m,k] × INT8_weight_k[k,n] → INT32_k[m,n]
 * 4. V GEMM: INT8[m,k] × INT8_weight_v[k,n] → INT32_v[m,n]
 * 5. Return INT32 accumulators for downstream attention computation
 *
 * Fusion Chain:
 *   FusedRMSNormQuantize → FusedTripleGEMM → Attention (with INT32 dequant)
 *   (FP32→INT8)         → (INT8→INT32)    → (INT32→FP32 in attention)
 *
 * @author David Sanftenberg
 * @date 2025-11-23
 */

#pragma once

#include "../CPUKernelBase.h"
#include "../../../tensors/TensorKernels.h"
#include "../../../tensors/Tensors.h"
#include <cstdint>
#include <vector>
#include <memory>

namespace llaminar2
{
    /**
     * @brief Fused triple GEMM kernel for attention Q/K/V projections
     *
     * Replaces:
     *   Quant(input) → Q GEMM → Dequant(q_out)
     *   Quant(input) → K GEMM → Dequant(k_out)
     *   Quant(input) → V GEMM → Dequant(v_out)
     *
     * With:
     *   Quant(input) → FusedTripleGEMM(q_weight, k_weight, v_weight) → [q_int32, k_int32, v_int32]
     *
     * Usage:
     *   FusedTripleGEMM kernel(q_weight, k_weight, v_weight);
     *   kernel.execute(input_fp32, q_output_int32, k_output_int32, v_output_int32,
     *                  activation_scales, m, n, k);
     */
    class FusedTripleGEMM : public CPUKernelBase
    {
    public:
        /**
         * @brief Construct fused triple GEMM kernel with weight tensors
         *
         * @param q_weight INT8 quantized Q projection weights [k, n]
         * @param k_weight INT8 quantized K projection weights [k, n]
         * @param v_weight INT8 quantized V projection weights [k, n]
         */
        FusedTripleGEMM(TensorBase *q_weight, TensorBase *k_weight, TensorBase *v_weight);
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
                .output_format = TensorFormat::INT32,           // Produce INT32 accumulators
                .supports_inplace = false,                      // Need separate output buffers
                .is_fusable = true                              // Can fuse with attention kernel
            };
        }

        bool supports_fusion() const override
        {
            return true; // High-priority fusion candidate
        }

        TensorFormat preferred_fusion_format() const override
        {
            return TensorFormat::INT32; // Output format for downstream fusion
        }

        // =============================================================================
        // Execution Interface
        // =============================================================================

        /**
         * @brief Execute fused triple GEMM with shared input quantization
         *
         * Performs:
         * 1. Quantize input: FP32[m,k] → INT8[m,k] + row_scales[m]
         * 2. Q GEMM: INT8[m,k] × q_weight[k,n_q] → q_output[m,n_q] (INT32)
         * 3. K GEMM: INT8[m,k] × k_weight[k,n_kv] → k_output[m,n_kv] (INT32)
         * 4. V GEMM: INT8[m,k] × v_weight[k,n_kv] → v_output[m,n_kv] (INT32)
         *
         * @param input Input activations [m, k] FP32
         * @param q_output Output Q INT32 accumulators [m, n_q]
         * @param k_output Output K INT32 accumulators [m, n_kv]
         * @param v_output Output V INT32 accumulators [m, n_kv]
         * @param activation_scales Per-row quantization scales [m] FP32 (output)
         * @param m Batch size (sequence length)
         * @param n_q Q output dimension (n_heads * head_dim)
         * @param n_kv K/V output dimension (n_kv_heads * head_dim, may differ for GQA)
         * @param k Input features
         * @return true on success, false on error
         *
         * Note: activation_scales are computed during quantization and needed
         *       for downstream dequantization in attention kernel.
         *
         * GQA Support: For Grouped Query Attention, n_q != n_kv (e.g., Qwen2.5: n_q=896, n_kv=128)
         */
        bool execute(
            const float *input,
            int32_t *q_output,
            int32_t *k_output,
            int32_t *v_output,
            float *activation_scales,
            int m, int n_q, int n_kv, int k);

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
        /**
         * @brief Quantize input activations to INT8 with per-row scales
         *
         * Quantization formula:
         *   row_scale[i] = max(|input[i,:]|) / 127.0
         *   quantized[i,j] = round(input[i,j] / row_scale[i])
         *
         * @param input Input FP32 activations [m, k]
         * @param output Output INT8 activations [m, k]
         * @param row_scales Output per-row scales [m]
         * @param m Number of rows
         * @param k Number of columns
         */
        void quantize_per_row(
            const float *input,
            int8_t *output,
            float *row_scales,
            int m, int k);

        /**
         * @brief Execute single INT8×INT8 → INT32 GEMM via OneDNN
         *
         * @param input_int8 Quantized input [m, k] INT8
         * @param weight_tensor Weight tensor (contains packed INT8 weights)
         * @param output_int32 Output accumulator [m, n] INT32
         * @param m Batch size
         * @param n Output features
         * @param k Input features
         * @return true on success
         */
        bool execute_int8_gemm(
            const int8_t *input_int8,
            TensorBase *weight_tensor,
            int32_t *output_int32,
            int m, int n, int k);

        // Weight tensors (owned externally, not by kernel)
        TensorBase *q_weight_;
        TensorBase *k_weight_;
        TensorBase *v_weight_;

        // Temporary INT8 activation buffer (reused across calls)
        std::vector<int8_t> int8_activations_;
    };

} // namespace llaminar2
