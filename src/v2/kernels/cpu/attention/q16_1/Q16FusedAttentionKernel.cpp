/**
 * @file Q16FusedAttentionKernel.cpp
 * @brief Implementation of Q16_1 fused attention kernel
 *
 * @author David Sanftenberg
 * @date December 2025
 */

#include "Q16FusedAttentionKernel.h"
#include "ref/Q16IntegerAttentionRef.h"
#include "utils/Logger.h"
#include <vector>

namespace llaminar2
{
    namespace kernels::q16_1
    {

        // =================================================================
        // Q16 Integer Attention Kernel (wired to Q16IntegerAttentionRef)
        // =================================================================

        // Fixed KV cache scale used for Q16_1 quantization throughout the pipeline.
        // This matches the scale used in KVCacheAppendStage, UnifiedKVCache, and RoPE.
        // Block scale d = kv_cache_scale / 32767 is what the kernel needs.
        //
        // IMPORTANT: For this to work correctly, the RoPE stage MUST use fixed-scale
        // quantization (not data-adaptive). See PLAN_FIXED_SCALE_ROPE_Q16.md for details.
        // Currently V uses fixed-scale (from KVCacheAppendStage), but Q and K from RoPE
        // use data-adaptive scales which causes a mismatch.
        constexpr float KV_CACHE_SCALE = 8.0f;
        constexpr float BLOCK_SCALE = KV_CACHE_SCALE / 32767.0f;

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

            // Convert FusedAttentionWoParams to Q16IntegerAttentionParams
            Q16IntegerAttentionParams ref_params;

            // Input tensors (already validated as Q16_1 blocks in FusedAttentionWoStage)
            ref_params.Q = params.Q;
            ref_params.K = params.K;
            ref_params.V = params.V;

            // Block size: auto-detect based on head_dim
            // - 64-element blocks for head_dim=64
            // - 128-element blocks for head_dim=128 (Qwen2, Llama, etc.)
            ref_params.block_size = optimal_q16_block_size(params.head_dim);

            // Head scales: Q16_1 blocks store their scale in the `d` field.
            // RoPE uses data-adaptive quantization, so each block can have a different scale.
            // The reference kernel assumes 1 scale per head (head-aligned blocks).
            // We extract the actual `d` values from the Q and K blocks.
            //
            // NOTE: This is an approximation for K since different positions have different scales.
            // For true integer attention with fixed scales, the entire pipeline should use
            // fixed-scale quantization (like V does for KV cache).
            // Head scales: Use fixed BLOCK_SCALE for all heads.
            // NOTE: This assumes the RoPE stage uses fixed-scale quantization.
            // Currently RoPE uses data-adaptive scaling, which causes a mismatch.
            // See PLAN_FIXED_SCALE_ROPE_Q16.md for the proper fix.
            std::vector<float> q_scales(params.n_heads, BLOCK_SCALE);
            std::vector<float> kv_scales(params.n_kv_heads, BLOCK_SCALE);
            ref_params.q_head_scales = q_scales.data();
            ref_params.kv_head_scales = kv_scales.data();

            // Wo weights: Q16IntegerAttentionRef does not yet implement Wo projection
            // The Wo_packed field in Q16IntegerAttentionParams uses a different type
            // that will be populated when Wo projection is implemented (Phase 7)
            ref_params.Wo_packed = nullptr; // TODO: Wire up when Wo projection is implemented

            // Residual pointers (not yet used by reference kernel - Phase 7)
            ref_params.residual_in = params.residual_in;
            ref_params.residual_out = params.residual_out;

            // Dimensions
            ref_params.seq_len_q = params.seq_len_q;
            ref_params.kv_len = params.kv_len;
            ref_params.num_heads = params.n_heads;
            ref_params.num_kv_heads = params.n_kv_heads;
            ref_params.head_dim = params.head_dim;
            ref_params.d_model = params.d_model;

            LOG_DEBUG("Q16FusedAttentionKernel: Using block_size=" << static_cast<int>(ref_params.block_size)
                                                                   << " for head_dim=" << params.head_dim);

            // Snapshot buffers for debugging
            ref_params.snapshot_scores = params.scores_snapshot;
            ref_params.snapshot_weights = nullptr; // Not directly mapped in FusedAttentionWoParams
            ref_params.snapshot_context = params.context_snapshot;
            ref_params.snapshot_projected = params.wo_output_snapshot;

            // Dispatch to decode or prefill based on seq_len_q
            bool success = q16_integer_attention_reference(ref_params);

            if (!success)
            {
                LOG_ERROR("Q16FusedAttentionKernel: q16_integer_attention_reference failed");
            }

            return success;
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
