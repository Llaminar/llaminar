/**
 * @file RoPEOp.h
 * @brief Self-validating Rotary Position Embedding operation
 *
 * RoPE applies rotary position embeddings to Q and K tensors.
 * This is applied after Q/K projections, before attention computation.
 *
 * Encapsulates the full RoPE workflow:
 * 1. Validate Q/K tensors and position IDs
 * 2. Apply RoPE via IActivationTensor interface
 * 3. Capture snapshots (Q_ROPE, K_ROPE)
 *
 * Usage:
 * @code
 * RoPEOp rope;
 *
 * TRY_OP(rope(Q, K, position_ids, seq_len, n_heads, n_kv_heads, head_dim,
 *             rope_theta, "layer0", mpi, device));
 * @endcode
 *
 * @author David Sanftenberg
 */

#pragma once

#include "Op.h"
#include <string>

namespace llaminar2
{

    /**
     * @brief Self-validating Rotary Position Embedding operation
     *
     * RoPE applies rotary embeddings based on position IDs:
     * - Q: [seq_len, n_heads * head_dim]
     * - K: [seq_len, n_kv_heads * head_dim]
     *
     * Position IDs of -1 signal padding tokens (RoPE is skipped for those positions).
     *
     * Replaces:
     * @code
     * auto *activation_q = dynamic_cast<IActivationTensor*>(Q.get());
     * VALIDATE_POINTER(activation_q, "Q activation");
     * VALIDATE_OP(activation_q->applyRoPE(K->mutable_data(), position_ids.data(),
     *             seq_len, n_heads, n_kv_heads, head_dim, rope_theta, false, mpi, device),
     *             "RoPE application");
     * CAPTURE_SNAPSHOT_VIEW("layer0_Q_ROPE", Q, seq_len, n_heads * head_dim);
     * CAPTURE_SNAPSHOT_VIEW("layer0_K_ROPE", K, seq_len, n_kv_heads * head_dim);
     * @endcode
     *
     * With:
     * @code
     * TRY_OP(rope(Q, K, position_ids, seq_len, n_heads, n_kv_heads, head_dim,
     *             rope_theta, "layer0", mpi, device));
     * @endcode
     */
    class RoPEOp : public OpBase
    {
    public:
        const char *name() const override { return "RoPEOp"; }

        /**
         * @brief Apply RoPE to Q and K tensors
         *
         * @param Q Query tensor [seq_len, n_heads * head_dim] - modified in-place
         * @param K Key tensor [seq_len, n_kv_heads * head_dim] - modified in-place
         * @param position_ids Position indices [seq_len] (-1 = padding, skip RoPE)
         * @param seq_len Sequence length (number of tokens)
         * @param n_heads Number of query heads
         * @param n_kv_heads Number of key/value heads (for GQA)
         * @param head_dim Dimension per head
         * @param rope_theta RoPE frequency base (10000.0 for LLaMA, 1000000.0 for Qwen2.5)
         * @param snapshot_prefix Prefix for snapshot keys (e.g., "layer0" -> "layer0_Q_ROPE")
         * @param mpi_ctx MPI context (nullptr for single-node)
         * @param device_idx Device index (-1 for CPU)
         *
         * @return true on success, false on validation or execution failure
         */
        bool operator()(
            TensorBase *Q,
            TensorBase *K,
            const int *position_ids,
            int seq_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            float rope_theta = 10000.0f,
            const char *snapshot_prefix = nullptr,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1)
        {
            // 1. Validate inputs
            if (!validateTensor(Q, "Q"))
                return false;
            if (!validateTensor(K, "K"))
                return false;
            if (!validatePointer(position_ids, "position_ids"))
                return false;
            if (seq_len <= 0 || n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0)
            {
                LOG_ERROR(name() << ": invalid dimensions (seq_len=" << seq_len
                                 << ", n_heads=" << n_heads << ", n_kv_heads=" << n_kv_heads
                                 << ", head_dim=" << head_dim << ")");
                return false;
            }

            // 2. Get Q as activation tensor for RoPE application
            auto *activation_q = dynamic_cast<IActivationTensor *>(Q);
            if (!activation_q)
            {
                logError("Q tensor must be IActivationTensor");
                return false;
            }

            // 3. Apply RoPE (modifies both Q and K in-place)
            if (!activation_q->applyRoPE(
                    K->mutable_data(),
                    position_ids,
                    seq_len,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    rope_theta,
                    false, // use_bf16
                    mpi_ctx,
                    device_idx))
            {
                logError("RoPE application failed");
                return false;
            }

            // Note: Snapshot capture is handled by the calling pipeline
            (void)snapshot_prefix;

            return true;
        }
    };

} // namespace llaminar2
