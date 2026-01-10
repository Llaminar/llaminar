/**
 * @file FusedGEMM.h
 * @brief Generic fused multi-GEMM for shared activation quantization
 *
 * Fuses N GEMM projections by:
 * 1. Quantizing activations once (FP32 → Q8_1)
 * 2. Executing all N GEMMs with the shared quantized buffer
 * 3. All GEMMs output FP32 (via QuantisedGemmKernel) OR Q8_1 (via fused epilogue)
 *
 * Use Cases:
 * - FFN gate/up projections (2 GEMMs)
 * - Attention Q/K/V projections (3 GEMMs)
 * - Any pattern with shared input across multiple linear projections
 *
 * Performance Benefits:
 * - Saves N-1 activation quantization passes
 * - Better cache locality (input stays hot across GEMMs)
 * - Optional Q8_1 output for memory bandwidth savings
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
     * @brief Single projection descriptor for FusedGEMM (FP32 output)
     */
    struct GEMMProjection
    {
        float *output;                     ///< Output buffer [m, n]
        const TensorBase *bias;            ///< Optional bias tensor [n] (nullptr if none)
        int n;                             ///< Output dimension
        std::string name;                  ///< Name for error messages (optional)
        const float *gate_input = nullptr; ///< Optional gate input for SwiGLU [m, n]
        bool do_swiglu = false;            ///< Whether to apply SwiGLU fusion
    };

    /**
     * @brief Single projection descriptor for FusedGEMM with Q8_1 output
     *
     * Used when the GEMM result should be fused-requantized to Q8_1 format,
     * saving memory bandwidth for subsequent Q8_1-input kernels (e.g., fused attention).
     */
    struct GEMMProjectionQ8_1
    {
        void *output_q8_1;      ///< Output Q8_1 buffer [m, ceil(n/32)] Q8_1Blocks
        const TensorBase *bias; ///< Optional bias tensor [n] to add before requantization (nullptr if none)
        int n;                  ///< Output dimension (rounded up to 32 for Q8_1)
        std::string name;       ///< Name for error messages (optional)
    };

    /**
     * @brief Single projection descriptor for FusedGEMM with Q16_1 output
     *
     * Used when the GEMM result should be fused-requantized to Q16_1 format,
     * providing 256× better precision than Q8_1. This is critical for K projections
     * in HybridQ16 mode where high dynamic range would cause Q8_1 to lose small values.
     */
    struct GEMMProjectionQ16_1
    {
        void *output_q16_1;     ///< Output Q16_1 buffer [m, n/block_size] blocks
        const TensorBase *bias; ///< Optional bias tensor [n] to add before requantization (nullptr if none)
        int n;                  ///< Output dimension (must be divisible by block_size)
        int q16_block_size;     ///< Block size: 32, 64, or 128 (should match head_dim)
        std::string name;       ///< Name for error messages (optional)
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
            const TensorBase *bias1, const TensorBase *bias2,
            int m, int n, int k,
            const MPIContext *ctx = nullptr,
            int device_idx = -1);

        /**
         * @brief Convenience execute for 3 projections (Attention pattern)
         */
        bool execute(
            const float *input,
            float *output1, float *output2, float *output3,
            const TensorBase *bias1, const TensorBase *bias2, const TensorBase *bias3,
            int m, int n1, int n2, int n3, int k,
            const MPIContext *ctx = nullptr,
            int device_idx = -1);

        // =============================================================================
        // Q8_1 Output Interface (for fused Q8_1 activation pipeline)
        // =============================================================================

        /**
         * @brief Execute fused multi-GEMM with Q8_1 output
         *
         * Quantizes input once, then executes all projections with shared Q8_1 buffer,
         * outputting Q8_1 blocks instead of FP32. This is used when subsequent kernels
         * consume Q8_1 activations (e.g., QuantisedAttentionJit_Q8_1_Fused).
         *
         * @param input Input activations [m, k] FP32
         * @param projections Vector of Q8_1 projection descriptors (output_q8_1, n)
         * @param m Batch size (sequence length)
         * @param k Input features (must match weight input dimension)
         * @param ctx MPI context (optional)
         * @param device_idx Device index (-1 for CPU)
         * @return true on success, false on error
         */
        bool execute_to_q8_1(
            const float *input,
            const std::vector<GEMMProjectionQ8_1> &projections,
            int m, int k,
            const MPIContext *ctx = nullptr,
            int device_idx = -1);

        /**
         * @brief Convenience execute for 3 Q8_1 projections (Attention Q/K/V pattern)
         *
         * @param input Input activations [m, k] FP32
         * @param output_q [m, ceil(n_q/32)] Q8_1 blocks for Q projection
         * @param output_k [m, ceil(n_kv/32)] Q8_1 blocks for K projection
         * @param output_v [m, ceil(n_kv/32)] Q8_1 blocks for V projection
         * @param bias_q Optional bias [n_q] for Q projection (nullptr if none)
         * @param bias_k Optional bias [n_kv] for K projection (nullptr if none)
         * @param bias_v Optional bias [n_kv] for V projection (nullptr if none)
         * @param m Batch size
         * @param n_q Q output dimension
         * @param n_kv K/V output dimension
         * @param k Input dimension
         * @param ctx MPI context
         * @param device_idx Device index
         */
        bool execute_to_q8_1(
            const float *input,
            void *output_q, void *output_k, void *output_v,
            const TensorBase *bias_q, const TensorBase *bias_k, const TensorBase *bias_v,
            int m, int n_q, int n_kv, int k,
            const MPIContext *ctx = nullptr,
            int device_idx = -1);

        // =============================================================================
        // Q8_1-to-Q8_1 Interface (for pure Q8_1 activation pipeline)
        // =============================================================================

        /**
         * @brief Execute fused multi-GEMM with Q8_1 input and Q8_1 output
         *
         * Takes pre-quantized Q8_1 activations and outputs Q8_1 blocks.
         * This avoids double quantization when both input and output are Q8_1.
         *
         * @param input_q8_1 Input Q8_1 blocks [m, k/32]
         * @param projections Vector of Q8_1 projection descriptors (output_q8_1, n)
         * @param m Batch size (sequence length)
         * @param k Input features (must match weight input dimension)
         * @param ctx MPI context (optional)
         * @param device_idx Device index (-1 for CPU)
         * @return true on success, false on error
         */
        bool execute_q8_1_to_q8_1(
            const void *input_q8_1,
            const std::vector<GEMMProjectionQ8_1> &projections,
            int m, int k,
            const MPIContext *ctx = nullptr,
            int device_idx = -1);

        /**
         * @brief Convenience execute for 3 Q8_1-to-Q8_1 projections (Attention Q/K/V pattern)
         *
         * @param input_q8_1 Input Q8_1 blocks [m, k/32]
         * @param output_q [m, ceil(n_q/32)] Q8_1 blocks for Q projection
         * @param output_k [m, ceil(n_kv/32)] Q8_1 blocks for K projection
         * @param output_v [m, ceil(n_kv/32)] Q8_1 blocks for V projection
         * @param bias_q Optional bias [n_q] for Q projection (nullptr if none)
         * @param bias_k Optional bias [n_kv] for K projection (nullptr if none)
         * @param bias_v Optional bias [n_kv] for V projection (nullptr if none)
         * @param m Batch size
         * @param n_q Q output dimension
         * @param n_kv K/V output dimension
         * @param k Input dimension
         * @param ctx MPI context
         * @param device_idx Device index
         */
        bool execute_q8_1_to_q8_1(
            const void *input_q8_1,
            void *output_q, void *output_k, void *output_v,
            const TensorBase *bias_q, const TensorBase *bias_k, const TensorBase *bias_v,
            int m, int n_q, int n_kv, int k,
            const MPIContext *ctx = nullptr,
            int device_idx = -1);

        // =============================================================================
        // Mixed-Precision QKV Interface (for HybridQ16 K precision fix)
        // =============================================================================

        /**
         * @brief Execute mixed-precision QKV GEMM: Q=Q8_1, K=Q16_1, V=Q8_1
         *
         * This method addresses the K precision problem in HybridQ16 mode:
         * - K projection has high dynamic range (max_abs ≈ 130)
         * - Q8_1 step = 130/127 ≈ 1.02, which zeros out values < 0.51
         * - Q16_1 step = 130/32767 ≈ 0.004, preserving values down to 0.002
         *
         * By outputting K as Q16_1 directly from GEMM, we avoid the Q8_1 bottleneck.
         *
         * @param input_q8_1 Input Q8_1 blocks [m, k/32] (from attention norm)
         * @param output_q [m, ceil(n_q/32)] Q8_1 blocks for Q projection
         * @param output_k [m, n_kv/block_size] Q16_1 blocks for K projection
         * @param output_v [m, ceil(n_kv/32)] Q8_1 blocks for V projection
         * @param bias_q Optional bias [n_q] for Q projection
         * @param bias_k Optional bias [n_kv] for K projection
         * @param bias_v Optional bias [n_kv] for V projection
         * @param m Batch size (sequence length)
         * @param n_q Q output dimension
         * @param n_kv K/V output dimension
         * @param k Input dimension
         * @param k_block_size Q16_1 block size for K output (64 or 128, matching head_dim)
         * @param ctx MPI context (optional)
         * @param device_idx Device index (-1 for CPU)
         * @return true on success, false on error
         */
        bool execute_q8_1_mixed_qkv(
            const void *input_q8_1,
            void *output_q, // Q8_1 output
            void *output_k, // Q16_1 output (high precision!)
            void *output_v, // Q8_1 output
            const TensorBase *bias_q, const TensorBase *bias_k, const TensorBase *bias_v,
            int m, int n_q, int n_kv, int k,
            int k_block_size = 64, // Q16 block size for K (should match head_dim)
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
        /// Non-owning pointers to kernels obtained via TensorBase::getOrCreateGemm().
        /// Kernel lifetime is tied to the tensor's lifetime, not FusedGEMM's.
        std::vector<gemm_v4::QuantisedGemmKernel *> gemm_kernels_;
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
