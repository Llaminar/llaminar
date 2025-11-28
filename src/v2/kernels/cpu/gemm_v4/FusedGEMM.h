/**
 * @file FusedGEMM.h
 * @brief Generic fused multi-GEMM for shared activation quantization
 *
 * Fuses N GEMM projections by:
 * 1. Quantizing activations once (FP32 → Q8_1)
 * 2. Executing all N GEMMs with the shared quantized buffer
 * 3. All GEMMs output FP32 (via QuantisedGemmKernel)
 *
 * Use Cases:
 * - FFN gate/up projections (2 GEMMs)
 * - Attention Q/K/V projections (3 GEMMs)
 * - Any pattern with shared input across multiple linear projections
 *
 * Performance Benefits:
 * - Saves N-1 activation quantization passes
 * - Better cache locality (input stays hot across GEMMs)
 * - Consistent FP32 residual stream
 *
 * @author David Sanftenberg
 * @date 2025-11-26
 */

#pragma once

#include "../CPUKernelBase.h"
#include "../../../tensors/TensorKernels.h"
#include "../../../tensors/Tensors.h"
#include "../gemm_v4/QuantisedGemmKernel.h"
#include <cstdint>
#include <vector>
#include <memory>
#include <string>
#include <initializer_list>

namespace llaminar2
{
    /**
     * @brief Single projection descriptor for FusedGEMM
     */
    struct GEMMProjection
    {
        float *output;                     ///< Output buffer [m, n]
        const float *bias;                 ///< Optional bias [n] (nullptr if none)
        int n;                             ///< Output dimension
        std::string name;                  ///< Name for error messages (optional)
        const float *gate_input = nullptr; ///< Optional gate input for SwiGLU [m, n]
        bool do_swiglu = false;            ///< Whether to apply SwiGLU fusion
    };

    /**
     * @brief Generic fused multi-GEMM kernel
     *
     * Wraps N QuantisedGemmKernel instances, quantizes activations once,
     * and executes all projections with shared quantized data.
     *
     * Usage (FFN):
     *   FusedGEMM kernel({gate_weight, up_weight});
     *   kernel.execute(input, {{gate_out, gate_bias, n}, {up_out, up_bias, n}}, m, k);
     *
     * Usage (Attention):
     *   FusedGEMM kernel({q_weight, k_weight, v_weight});
     *   kernel.execute(input, {{q_out, q_bias, n_q}, {k_out, k_bias, n_kv}, {v_out, v_bias, n_kv}}, m, k);
     */
    class FusedGEMM : public CPUKernelBase
    {
    public:
        /**
         * @brief Construct fused GEMM kernel with weight tensors
         *
         * All weights must have the same input dimension (k).
         * Output dimensions (n) can differ per projection.
         *
         * @param weights Vector of quantized weight tensors, each [n_i, k]
         * @param projection_names Optional names for each projection (for error messages)
         */
        explicit FusedGEMM(const std::vector<const TensorBase *> &weights,
                           const std::vector<std::string> &projection_names = {});

        /**
         * @brief Convenience constructor for 2 projections (FFN gate/up pattern)
         */
        FusedGEMM(const TensorBase *weight1, const TensorBase *weight2);

        /**
         * @brief Convenience constructor for 3 projections (Attention Q/K/V pattern)
         */
        FusedGEMM(const TensorBase *weight1, const TensorBase *weight2, const TensorBase *weight3);

        ~FusedGEMM() override = default;

        // =============================================================================
        // Execution Interface
        // =============================================================================

        /**
         * @brief Execute fused multi-GEMM
         *
         * Quantizes input once, then executes all projections with shared Q8_1 buffer.
         *
         * @param input Input activations [m, k] FP32
         * @param projections Vector of projection descriptors (output, bias, n)
         * @param m Batch size (sequence length)
         * @param k Input features (must match weight input dimension)
         * @param ctx MPI context (optional)
         * @param device_idx Device index (-1 for CPU)
         * @return true on success, false on error
         */
        bool execute(
            const float *input,
            const std::vector<GEMMProjection> &projections,
            int m, int k,
            const MPIContext *ctx = nullptr,
            int device_idx = -1);

        /**
         * @brief Convenience execute for 2 projections (FFN pattern)
         */
        bool execute(
            const float *input,
            float *output1, float *output2,
            const float *bias1, const float *bias2,
            int m, int n, int k,
            const MPIContext *ctx = nullptr,
            int device_idx = -1);

        /**
         * @brief Convenience execute for 3 projections (Attention pattern)
         */
        bool execute(
            const float *input,
            float *output1, float *output2, float *output3,
            const float *bias1, const float *bias2, const float *bias3,
            int m, int n1, int n2, int n3, int k,
            const MPIContext *ctx = nullptr,
            int device_idx = -1);

        /**
         * @brief Get number of projections
         */
        size_t num_projections() const { return gemm_kernels_.size(); }

        /**
         * @brief Check if kernel supports given device
         */
        bool supports_device(int device_idx) const { return device_idx == -1; }

    private:
        std::vector<std::unique_ptr<gemm_v4::QuantisedGemmKernel>> gemm_kernels_;
        std::vector<std::string> projection_names_;
        int k_dim_; ///< Shared input dimension
    };

    // =============================================================================
    // Type Aliases for Common Patterns
    // =============================================================================

    /// FFN gate/up dual GEMM (backwards compatible alias)
    using FusedDualGEMM_v2 = FusedGEMM;

    /// Attention Q/K/V triple GEMM (backwards compatible alias)
    using FusedTripleGEMM_v2 = FusedGEMM;

} // namespace llaminar2
