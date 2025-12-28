/**
 * @file Q16FusedAttentionKernel.h
 * @brief Q16_1 fused attention kernel implementing ITensorFusedAttentionWo
 *
 * Provides a kernel wrapper that implements the ITensorFusedAttentionWo interface
 * and delegates to either:
 * - Q16FusedAttentionRef: Scalar C++ reference (for testing/debugging)
 * - JitQ16FlashDecodeKernel / JitQ16FA2PrefillKernel: AVX512 JIT (production)
 *
 * The interface enables:
 * - Uniform kernel API across all fused attention implementations
 * - Compile-time snapshot capability enforcement
 * - Clean integration with FusedAttentionWo compute stage
 *
 * @see tensors/TensorKernels.h for ITensorFusedAttentionWo interface
 * @see Q16FusedAttentionRef.h for reference implementation
 * @see JitQ16FusedAttention.h for JIT implementations
 *
 * @author David Sanftenberg
 * @date December 2025
 */
#pragma once

#include "tensors/TensorKernels.h"
#include "tensors/KernelSnapshotInfo.h"
#include "tensors/BlockStructures.h"
#include "ref/Q16FusedAttentionRef.h"
#include <memory>

namespace llaminar2
{
    namespace kernels::q16_1
    {

        /**
         * @brief Q16_1 fused attention kernel implementing ITensorFusedAttentionWo
         *
         * This kernel executes the entire attention block in integer domain:
         * - Q×K^T → INT32 scores
         * - Exp2FixedSoftmax → INT16 attention weights
         * - Weights×V → INT32 context accumulators
         * - Context×Wo (VPDPWSSD) → Q16_1 output
         * - Q16_1 + Q16_1 residual → Q16_1 output
         *
         * Execution paths:
         * - **use_jit=false**: Uses scalar reference implementation (Q16FusedAttentionRef)
         * - **use_jit=true**: Uses AVX512 JIT kernels (JitQ16FlashDecodeKernel, JitQ16FA2PrefillKernel)
         *
         * The kernel automatically dispatches to decode (seq_len_q=1) or prefill (seq_len_q>1)
         * based on the input parameters.
         */
        class Q16FusedAttentionKernel : public ITensorFusedAttentionWo
        {
        public:
            /**
             * @brief Construct Q16 fused attention kernel
             *
             * @param use_jit If true, use JIT kernels when available (default: true)
             */
            explicit Q16FusedAttentionKernel(bool use_jit = true)
                : use_jit_(use_jit)
            {
            }

            ~Q16FusedAttentionKernel() override = default;

            // =================================================================
            // ITensorFusedAttentionWo Interface Implementation
            // =================================================================

            /**
             * @brief Execute fused attention + Wo + residual
             *
             * Converts FusedAttentionWoParams to Q16FusedAttentionWoResidualParams
             * and dispatches to the reference or JIT implementation.
             */
            bool compute(
                const FusedAttentionWoParams &params,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1) override;

            /**
             * @brief Execute decode path (single query token)
             */
            bool compute_decode(
                const FusedAttentionWoParams &params,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1) override;

            /**
             * @brief Execute prefill path (multiple query tokens)
             */
            bool compute_prefill(
                const FusedAttentionWoParams &params,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1) override;

            /**
             * @brief Execute using TensorBase objects
             *
             * Extracts Q16_1 block pointers and VNNI-packed Wo weights from tensors
             * and constructs the appropriate params struct.
             */
            bool compute_tensor(
                const TensorBase *Q,
                const TensorBase *K,
                const TensorBase *V,
                const TensorBase *Wo_tensor,
                const TensorBase *residual_in,
                TensorBase *residual_out,
                int seq_len_q,
                int kv_len,
                int n_heads,
                int n_kv_heads,
                int head_dim,
                bool causal = true,
                int position_offset = 0,
                float *scores_snapshot = nullptr,
                float *context_snapshot = nullptr,
                float *wo_output_snapshot = nullptr,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1) override;

            // =================================================================
            // ITensorKernel Interface
            // =================================================================

            /**
             * @brief Check device support (CPU only for Q16)
             */
            bool supports_device(int device_idx) const override
            {
                return device_idx == -1; // CPU only
            }

            /**
             * @brief Get input precision format (Q16_1)
             */
            ActivationFormat input_format() const override
            {
                return ActivationFormat::Q8_1; // Q16_1 extends Q8_1 semantics
            }

            /**
             * @brief Get output precision format (Q16_1)
             */
            ActivationFormat output_format() const override
            {
                return ActivationFormat::Q8_1; // Q16_1 extends Q8_1 semantics
            }

            /**
             * @brief Check if VNNI-packed Wo is required (yes for Q16)
             */
            bool requires_packed_wo() const override
            {
                return true; // Q16 uses VPDPWSSD with VNNI-packed weights
            }

            // =================================================================
            // IKernelSnapshotCapable Interface
            // =================================================================

            /**
             * @brief Get kernel snapshot metadata
             *
             * Declares all input, output, and intermediate buffers that can be
             * captured for debugging and parity testing.
             */
            KernelSnapshotInfo getKernelSnapshotInfo() const override
            {
                return KernelSnapshotInfo::fusedAttentionWo()
                    // Inputs
                    .withInput("Q", "Query tensor [seq_len_q × n_heads × head_dim]", KernelBufferDtype::Q16_1)
                    .withInput("K", "Key tensor [kv_len × n_kv_heads × head_dim]", KernelBufferDtype::Q16_1)
                    .withInput("V", "Value tensor [kv_len × n_kv_heads × head_dim]", KernelBufferDtype::Q16_1)
                    .withInput("residual_in", "Input residual [seq_len_q × d_model]", KernelBufferDtype::Q16_1)
                    // Weights
                    .withWeight("Wo", "Output projection weights (VNNI-packed)", KernelBufferDtype::INT16)
                    // Outputs
                    .withOutput("residual_out", "Output residual [seq_len_q × d_model]", KernelBufferDtype::Q16_1)
                    // Intermediate (for snapshots)
                    .withOutput("scores", "Attention scores [seq_len_q × n_heads × kv_len]",
                                KernelBufferDtype::FP32, true /* intermediate */)
                    .withOutput("context", "Attention context [seq_len_q × n_heads × head_dim]",
                                KernelBufferDtype::FP32, true /* intermediate */)
                    .withOutput("wo_output", "Wo projection output [seq_len_q × d_model]",
                                KernelBufferDtype::FP32, true /* intermediate */)
                    // Scalars
                    .withScalar("seq_len_q", "Query sequence length", KernelBufferDtype::INT32)
                    .withScalar("kv_len", "KV sequence length", KernelBufferDtype::INT32)
                    .withScalar("n_heads", "Number of query heads", KernelBufferDtype::INT32)
                    .withScalar("n_kv_heads", "Number of KV heads", KernelBufferDtype::INT32)
                    .withScalar("head_dim", "Head dimension", KernelBufferDtype::INT32)
                    .withScalar("scale", "Attention scale (1/sqrt(head_dim))", KernelBufferDtype::FP32)
                    .withScalar("causal", "Apply causal masking", KernelBufferDtype::INT32)
                    .withScalar("position_offset", "Position offset for decode", KernelBufferDtype::INT32);
            }

            // =================================================================
            // Configuration
            // =================================================================

            /**
             * @brief Enable/disable JIT kernels
             */
            void set_use_jit(bool use_jit) { use_jit_ = use_jit; }

            /**
             * @brief Check if JIT kernels are enabled
             */
            bool use_jit() const { return use_jit_; }

        private:
            bool use_jit_;

            /**
             * @brief Convert FusedAttentionWoParams to Q16FusedAttentionWoResidualParams
             */
            Q16FusedAttentionWoResidualParams convert_params(const FusedAttentionWoParams &params) const;

            /**
             * @brief Validate input parameters
             */
            bool validate_params(const FusedAttentionWoParams &params) const;
        };

    } // namespace kernels::q16_1
} // namespace llaminar2
