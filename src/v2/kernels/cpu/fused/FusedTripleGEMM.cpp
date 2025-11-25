/**
 * @file FusedTripleGEMM.cpp
 * @brief Implementation of fused triple GEMM for attention Q/K/V projections
 * @author David Sanftenberg
 * @date 2025-11-23
 * @updated 2025-11-24 - Reworked for FP32 residual architecture
 */

#include "FusedTripleGEMM.h"
#include "../../../utils/Logger.h"

namespace llaminar2
{
    FusedTripleGEMM::FusedTripleGEMM(const Q8_1Tensor *q_weight, const Q8_1Tensor *k_weight, const Q8_1Tensor *v_weight)
    {
        if (!q_weight || !k_weight || !v_weight)
        {
            throw std::invalid_argument("FusedTripleGEMM: Weight tensors cannot be null");
        }

        // Validate weight shapes
        const auto &q_shape = q_weight->shape();
        const auto &k_shape = k_weight->shape();
        const auto &v_shape = v_weight->shape();

        if (q_shape.size() != 2 || k_shape.size() != 2 || v_shape.size() != 2)
        {
            throw std::invalid_argument("FusedTripleGEMM: All weight tensors must be 2D");
        }

        // Input dimension (k) must match across all weights
        if (q_shape[1] != k_shape[1] || q_shape[1] != v_shape[1])
        {
            throw std::invalid_argument("FusedTripleGEMM: Q/K/V weights must have matching input dimension (k)");
        }

        // K and V must have same output dimension (n_kv)
        if (k_shape[0] != v_shape[0])
        {
            throw std::invalid_argument("FusedTripleGEMM: K and V weights must have matching output dimension");
        }

        // Create Q8_1 GEMM kernels for each projection
        q_gemm_ = std::make_unique<gemm_v4::Q8_1GemmKernel>(q_weight);
        k_gemm_ = std::make_unique<gemm_v4::Q8_1GemmKernel>(k_weight);
        v_gemm_ = std::make_unique<gemm_v4::Q8_1GemmKernel>(v_weight);
    }

    bool FusedTripleGEMM::execute(
        const float *input,
        float *q_output,
        float *k_output,
        float *v_output,
        const float *q_bias,
        const float *k_bias,
        const float *v_bias,
        int m, int n_q, int n_kv, int k,
        const MPIContext *ctx,
        int device_idx)
    {
        if (!input || !q_output || !k_output || !v_output)
        {
            LOG_ERROR("[FusedTripleGEMM] Null pointer in execute()");
            return false;
        }

        if (m <= 0 || n_q <= 0 || n_kv <= 0 || k <= 0)
        {
            LOG_ERROR("[FusedTripleGEMM] Invalid dimensions: m=" << m << " n_q=" << n_q << " n_kv=" << n_kv << " k=" << k);
            return false;
        }

        // =====================================================================
        // Step 1: Execute Q projection GEMM (FP32 → Q8_1 internally → FP32)
        // =====================================================================

        bool success = q_gemm_->multiply_fused(
            input, q_output, m, n_q, k,
            q_bias,           // Fused bias
            nullptr,          // No mask
            false,            // No softmax
            nullptr, nullptr, // No softmax outputs
            false,            // No accumulation
            1.0f, 0.0f,       // alpha=1, beta=0
            ctx, device_idx);

        if (!success)
        {
            LOG_ERROR("[FusedTripleGEMM] Q GEMM failed");
            return false;
        }

        // =====================================================================
        // Step 2: Execute K projection GEMM (FP32 → Q8_1 internally → FP32)
        // =====================================================================

        success = k_gemm_->multiply_fused(
            input, k_output, m, n_kv, k,
            k_bias,           // Fused bias
            nullptr,          // No mask
            false,            // No softmax
            nullptr, nullptr, // No softmax outputs
            false,            // No accumulation
            1.0f, 0.0f,       // alpha=1, beta=0
            ctx, device_idx);

        if (!success)
        {
            LOG_ERROR("[FusedTripleGEMM] K GEMM failed");
            return false;
        }

        // =====================================================================
        // Step 3: Execute V projection GEMM (FP32 → Q8_1 internally → FP32)
        // =====================================================================
        // Note: Future optimization - cache the quantized activation from Q GEMM
        // and reuse it for K and V GEMMs to save two quantization passes.

        success = v_gemm_->multiply_fused(
            input, v_output, m, n_kv, k,
            v_bias,           // Fused bias
            nullptr,          // No mask
            false,            // No softmax
            nullptr, nullptr, // No softmax outputs
            false,            // No accumulation
            1.0f, 0.0f,       // alpha=1, beta=0
            ctx, device_idx);

        if (!success)
        {
            LOG_ERROR("[FusedTripleGEMM] V GEMM failed");
            return false;
        }

        return true;
    }

} // namespace llaminar2
