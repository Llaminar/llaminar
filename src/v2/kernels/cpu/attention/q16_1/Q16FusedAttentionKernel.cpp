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

        // NOTE: Head scales are now extracted dynamically from Q16 blocks in compute().
        // This supports data-adaptive quantization from RoPE, which assigns different
        // scales to different positions based on the actual value range.

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
            // NOTE: These are Q16BlockPtr types, assignment works directly
            ref_params.Q = params.Q;
            ref_params.K = params.K;
            ref_params.V = params.V;

            // NOTE: block_size is now derived from Q.block_size automatically
            // (no explicit assignment needed)

            // ================================================================
            // HEAD SCALES: Two modes depending on HybridQ16 K precision fix
            // ================================================================
            // Mode 1: Per-position K scales from RoPE (HybridQ16 K precision fix)
            //   - params.k_head_scales is non-null
            //   - K scales vary per position due to dynamic-scale RoPE
            //   - Pass to ref_params.k_position_scales for per-position softmax scaling
            //
            // Mode 2: Uniform K scales from block headers (standard)
            //   - params.k_head_scales is null
            //   - Extract scales from Q16 block headers (first position approximation)
            //   - All positions within a head use the same scale

            const int block_elements = static_cast<int>(params.Q.block_size);
            const int blocks_per_head = (params.head_dim + block_elements - 1) / block_elements;

            std::vector<float> q_scales(params.n_heads);
            std::vector<float> kv_scales(params.n_kv_heads);

            // Lambda to extract d from Q16 block at given index
            auto extract_d = [&params](const Q16BlockPtr &ptr, size_t block_idx) -> float
            {
                switch (ptr.block_size)
                {
                case Q16BlockSize::BLOCK_64:
                    return ptr.as_block_64()[block_idx].d;
                case Q16BlockSize::BLOCK_128:
                    return ptr.as_block_128()[block_idx].d;
                case Q16BlockSize::BLOCK_32:
                    return ptr.as_block_32()[block_idx].d;
                }
                return 1.0f; // Fallback
            };

            // Extract Q head scales from first position of each head (always needed)
            for (int h = 0; h < params.n_heads; ++h)
            {
                // Head h starts at block index: h * seq_len_q * blocks_per_head
                const size_t head_start_block = static_cast<size_t>(h) * params.seq_len_q * blocks_per_head;
                q_scales[h] = extract_d(params.Q, head_start_block);
            }

            // Debug: print Q scales for layer 0
            if (params.layer_idx == 0)
            {
                LOG_DEBUG("Q16FusedAttentionKernel Layer 0: extracted q_scales[0..2]="
                          << q_scales[0] << ", " << q_scales[1] << ", " << q_scales[2]
                          << " (expected ~0.00781 = 256/32767 (kv_cache_scale))");
            }

            const int effective_kv_stride = (params.kv_head_stride > 0) ? params.kv_head_stride : params.kv_len;

            if (params.k_head_scales)
            {
                // DISABLED: Per-position K scale code path was reading the WRONG scale value.
                // K_block.d is the QUANTIZATION scale (~0.008, which is max_abs(K)/32767)
                // NOT the normalization scale (~0.000244, which is 1/(KV_CACHE_SCALE*16))
                // This caused alpha to be 32x too large, making softmax almost one-hot.
                // See Test__OnlineSoftmaxPerPositionKScales.cpp for detailed proof.
                //
                // Until a correct per-position scale source is implemented, use uniform scaling.
                ref_params.read_k_scales_from_blocks = false;
                ref_params.k_position_scales = nullptr;

                // Still need kv_head_scales for V scale (V doesn't have per-position scales)
                // Extract from block headers as fallback
                for (int kv_h = 0; kv_h < params.n_kv_heads; ++kv_h)
                {
                    const size_t head_start_block = static_cast<size_t>(kv_h) * effective_kv_stride * blocks_per_head;
                    kv_scales[kv_h] = extract_d(params.V, head_start_block); // Use V for V scale
                }

                // Debug: Log K scale range from K cache blocks (first layer only)
                if (params.layer_idx == 0)
                {
                    // Read K scales directly from K cache for logging
                    float min_k = std::numeric_limits<float>::max();
                    float max_k = std::numeric_limits<float>::lowest();
                    int zero_count = 0;
                    const int total_positions = params.kv_len;
                    for (int kv_h = 0; kv_h < params.n_kv_heads; ++kv_h)
                    {
                        const size_t head_start_block = static_cast<size_t>(kv_h) * effective_kv_stride * blocks_per_head;
                        for (int pos = 0; pos < total_positions; ++pos)
                        {
                            float k_scale = extract_d(params.K, head_start_block + pos * blocks_per_head);
                            if (k_scale == 0.0f)
                                zero_count++;
                            min_k = std::min(min_k, k_scale);
                            max_k = std::max(max_k, k_scale);
                        }
                    }
                    LOG_DEBUG("Q16FusedAttentionKernel Layer 0: K scales from cache range [" << min_k << " - " << max_k << "]"
                                                                                             << " zeros=" << zero_count << "/" << (total_positions * params.n_kv_heads)
                                                                                             << " ratio=" << (min_k > 0 ? max_k / min_k : 0));
                }

                LOG_DEBUG("Q16FusedAttentionKernel: Per-position K scale path DISABLED (was reading wrong scale)"
                          << " read_k_scales_from_blocks=false"
                          << ", kv_len=" << params.kv_len
                          << ", n_kv_heads=" << params.n_kv_heads << ")");
            }
            else
            {
                // Mode 2: Standard - extract scales from K block headers
                for (int kv_h = 0; kv_h < params.n_kv_heads; ++kv_h)
                {
                    const size_t head_start_block = static_cast<size_t>(kv_h) * effective_kv_stride * blocks_per_head;
                    kv_scales[kv_h] = extract_d(params.K, head_start_block);
                }

                LOG_DEBUG("Q16FusedAttentionKernel: Using uniform K scales from block headers"
                          << " (kv_scales[0]=" << kv_scales[0] << ")");
            }

            ref_params.q_head_scales = q_scales.data();
            ref_params.kv_head_scales = kv_scales.data();

            // Wo weights: Convert from FusedAttentionWoParams to Q16IntegerAttentionParams
            // Both use llaminar2::gemm_v4::QuantisedPackedWeights
            ref_params.Wo_packed = params.Wo_packed;

            // Output buffer for Wo projection: MUST be a separate temporary buffer!
            // The residual add reads from both output (Wo result) and residual_in, then writes to residual_out.
            // If output == residual_in == residual_out (aliasing), the residual add produces garbage.
            // Allocate a temporary Q16_1 buffer for Wo output.
            const int blocks_per_row = (params.d_model + 31) / 32; // Q16_1Block has 32 elements
            const int total_blocks = params.seq_len_q * blocks_per_row;
            std::vector<Q16_1Block> wo_output_buffer(total_blocks);
            ref_params.output = (wo_output_buffer.data());

            // Residual pointers - these can alias (in-place: residual_in == residual_out)
            ref_params.residual_in = params.residual_in;
            ref_params.residual_out = params.residual_out;

            // Dimensions
            ref_params.seq_len_q = params.seq_len_q;
            ref_params.kv_len = params.kv_len;
            ref_params.kv_head_stride = params.kv_head_stride; // Pass through for sparse HEAD_MAJOR cache
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
            ref_params.snapshot_wo_output = params.wo_output_snapshot;
            ref_params.snapshot_residual_out = params.attention_residual_snapshot;

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
