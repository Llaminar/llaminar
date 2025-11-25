/**
 * @file FusedDualGEMM.cpp
 * @brief Implementation of fused dual GEMM for FFN gate/up projections
 * @author David Sanftenberg
 * @date 2025-11-23
 * @updated 2025-11-24 - Reworked for FP32 residual architecture
 */

#include "FusedDualGEMM.h"
#include "../../../utils/Logger.h"

namespace llaminar2
{
    FusedDualGEMM::FusedDualGEMM(const Q8_1Tensor *gate_weight, const Q8_1Tensor *up_weight)
    {
        if (!gate_weight || !up_weight)
        {
            throw std::invalid_argument("FusedDualGEMM: Weight tensors cannot be null");
        }

        // Validate that weights have compatible dimensions
        const auto &gate_shape = gate_weight->shape();
        const auto &up_shape = up_weight->shape();
        if (gate_shape.size() != 2 || up_shape.size() != 2)
        {
            throw std::invalid_argument("[FusedDualGEMM] Weight tensors must be 2D");
        }

        // For FFN, gate and up typically have same shape [n, k]
        // But we allow different n (output dim) as long as k (input dim) matches
        if (gate_shape[1] != up_shape[1])
        {
            throw std::invalid_argument("[FusedDualGEMM] Gate and Up weights must have matching input dimension (k)");
        }

        // Create Q8_1 GEMM kernels for each projection
        gate_gemm_ = std::make_unique<gemm_v4::Q8_1GemmKernel>(gate_weight);
        up_gemm_ = std::make_unique<gemm_v4::Q8_1GemmKernel>(up_weight);
    }

    bool FusedDualGEMM::execute(
        const float *input,
        float *gate_output,
        float *up_output,
        const float *gate_bias,
        const float *up_bias,
        int m, int n, int k,
        const MPIContext *ctx,
        int device_idx)
    {
        if (!input || !gate_output || !up_output)
        {
            LOG_ERROR("[FusedDualGEMM] Null pointer in execute()");
            return false;
        }

        if (m <= 0 || n <= 0 || k <= 0)
        {
            LOG_ERROR("[FusedDualGEMM] Invalid dimensions: m=" << m << " n=" << n << " k=" << k);
            return false;
        }

        // =====================================================================
        // Step 1: Execute gate projection GEMM (FP32 → Q8_1 internally → FP32)
        // =====================================================================

        bool success = gate_gemm_->multiply_fused(
            input, gate_output, m, n, k,
            gate_bias,        // Fused bias
            nullptr,          // No mask
            false,            // No softmax
            nullptr, nullptr, // No softmax outputs
            false,            // No accumulation
            1.0f, 0.0f,       // alpha=1, beta=0
            ctx, device_idx);

        if (!success)
        {
            LOG_ERROR("[FusedDualGEMM] Gate GEMM failed");
            return false;
        }

        // =====================================================================
        // Step 2: Execute up projection GEMM (FP32 → Q8_1 internally → FP32)
        // =====================================================================
        // Note: Future optimization - cache the quantized activation from gate GEMM
        // and reuse it for up GEMM to save one quantization pass.

        success = up_gemm_->multiply_fused(
            input, up_output, m, n, k,
            up_bias,          // Fused bias
            nullptr,          // No mask
            false,            // No softmax
            nullptr, nullptr, // No softmax outputs
            false,            // No accumulation
            1.0f, 0.0f,       // alpha=1, beta=0
            ctx, device_idx);

        if (!success)
        {
            LOG_ERROR("[FusedDualGEMM] Up GEMM failed");
            return false;
        }

        return true;
    }

} // namespace llaminar2
