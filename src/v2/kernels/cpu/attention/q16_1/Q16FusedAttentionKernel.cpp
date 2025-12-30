/**
 * @file Q16FusedAttentionKernel.cpp
 * @brief Implementation of Q16_1 fused attention kernel
 *
 * @author David Sanftenberg
 * @date December 2025
 */

#include "Q16FusedAttentionKernel.h"
#include "utils/Logger.h"

namespace llaminar2
{
    namespace kernels::q16_1
    {

        // =================================================================
        // NOTE: Q16_INTEGER backend is a stub - v2 implementation in progress
        // See: ref/Q16IntegerAttentionRef.h for the new API
        // =================================================================

        // =================================================================
        // Parameter Validation
        // =================================================================

        bool Q16FusedAttentionKernel::validate_params(const FusedAttentionWoParams &params) const
        {
            // Check required pointers
            if (!params.Q || !params.K || !params.V)
            {
                LOG_ERROR("Q16FusedAttentionKernel: Q, K, V tensors are required");
                return false;
            }

            if (!params.Wo_packed)
            {
                LOG_ERROR("Q16FusedAttentionKernel: VNNI-packed Wo weights are required");
                return false;
            }

            if (!params.residual_in || !params.residual_out)
            {
                LOG_ERROR("Q16FusedAttentionKernel: residual_in and residual_out are required");
                return false;
            }

            // Check dimensions
            if (params.seq_len_q < 1)
            {
                LOG_ERROR("Q16FusedAttentionKernel: seq_len_q must be >= 1, got " << params.seq_len_q);
                return false;
            }

            if (params.kv_len < 1)
            {
                LOG_ERROR("Q16FusedAttentionKernel: kv_len must be >= 1, got " << params.kv_len);
                return false;
            }

            if (params.n_heads < 1 || params.n_kv_heads < 1)
            {
                LOG_ERROR("Q16FusedAttentionKernel: n_heads=" << params.n_heads
                                                              << " and n_kv_heads=" << params.n_kv_heads
                                                              << " must be >= 1");
                return false;
            }

            if (params.head_dim <= 0 || params.head_dim > 256)
            {
                LOG_ERROR("Q16FusedAttentionKernel: head_dim must be in (0, 256], got " << params.head_dim);
                return false;
            }

            // head_dim must be multiple of 32 for Q16_1 block alignment
            if (params.head_dim % 32 != 0)
            {
                LOG_ERROR("Q16FusedAttentionKernel: head_dim must be multiple of 32 for Q16_1, got "
                          << params.head_dim);
                return false;
            }

            // GQA validation: n_heads must be divisible by n_kv_heads
            if (params.n_heads % params.n_kv_heads != 0)
            {
                LOG_ERROR("Q16FusedAttentionKernel: n_heads=" << params.n_heads
                                                              << " must be divisible by n_kv_heads="
                                                              << params.n_kv_heads);
                return false;
            }

            return true;
        }

        // =================================================================
        // Main Compute Methods
        // =================================================================

        bool Q16FusedAttentionKernel::compute(
            const FusedAttentionWoParams &params,
            const MPIContext *mpi_ctx,
            int device_idx)
        {
            (void)mpi_ctx;    // Not used in CPU kernel
            (void)device_idx; // Must be -1 (CPU)

            if (device_idx != -1)
            {
                LOG_ERROR("Q16FusedAttentionKernel: only CPU execution supported (device_idx=-1)");
                return false;
            }

            if (!validate_params(params))
            {
                return false;
            }

            // STUB: Q16_INTEGER backend v2 implementation in progress
            // The new API uses Q16IntegerAttentionParams with model-aware block sizes.
            // See: ref/Q16IntegerAttentionRef.h and docs/v2/PROJECT_Q16_INTEGER_ATTENTION_V2.md
            LOG_ERROR("Q16FusedAttentionKernel: Q16_INTEGER backend not yet implemented (v2 in progress)");
            return false;
        }

        bool Q16FusedAttentionKernel::compute_decode(
            const FusedAttentionWoParams &params,
            const MPIContext *mpi_ctx,
            int device_idx)
        {
            if (params.seq_len_q != 1)
            {
                LOG_ERROR("Q16FusedAttentionKernel::compute_decode: seq_len_q must be 1, got "
                          << params.seq_len_q);
                return false;
            }

            // Decode path is handled by compute() based on params.is_decode()
            return compute(params, mpi_ctx, device_idx);
        }

        bool Q16FusedAttentionKernel::compute_prefill(
            const FusedAttentionWoParams &params,
            const MPIContext *mpi_ctx,
            int device_idx)
        {
            if (params.seq_len_q < 1)
            {
                LOG_ERROR("Q16FusedAttentionKernel::compute_prefill: seq_len_q must be >= 1, got "
                          << params.seq_len_q);
                return false;
            }

            // Prefill path is handled by compute() based on params.is_decode()
            return compute(params, mpi_ctx, device_idx);
        }

        // =================================================================
        // Tensor-based Compute
        // =================================================================

        bool Q16FusedAttentionKernel::compute_tensor(
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
            bool causal,
            int position_offset,
            float *scores_snapshot,
            float *context_snapshot,
            float *wo_output_snapshot,
            const MPIContext *mpi_ctx,
            int device_idx)
        {
            // Validate tensor types
            // Note: This is a simplified implementation. In production, we would
            // use dynamic_cast or type introspection to verify Q16_1 tensors.
            if (!Q || !K || !V || !Wo_tensor || !residual_in || !residual_out)
            {
                LOG_ERROR("Q16FusedAttentionKernel::compute_tensor: null tensor(s)");
                return false;
            }

            // Build params struct
            FusedAttentionWoParams params;

            // TODO: Extract Q16_1 blocks from tensor objects
            // For now, this is a placeholder - real implementation needs tensor type checking
            // and proper block extraction

            params.seq_len_q = seq_len_q;
            params.kv_len = kv_len;
            params.n_heads = n_heads;
            params.n_kv_heads = n_kv_heads;
            params.head_dim = head_dim;
            params.d_model = n_heads * head_dim;
            params.causal = causal;
            params.position_offset = position_offset;
            params.scores_snapshot = scores_snapshot;
            params.context_snapshot = context_snapshot;
            params.wo_output_snapshot = wo_output_snapshot;

            // This is a placeholder - real implementation would extract pointers from tensors
            LOG_WARN("Q16FusedAttentionKernel::compute_tensor: tensor extraction not fully implemented");
            (void)Q;
            (void)K;
            (void)V;
            (void)Wo_tensor;
            (void)residual_in;
            (void)residual_out;
            (void)mpi_ctx;
            (void)device_idx;

            return false; // Not fully implemented yet
        }

    } // namespace kernels::q16_1
} // namespace llaminar2
