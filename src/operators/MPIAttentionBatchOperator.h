/**
 * @file MPIAttentionBatchOperator.h
 * @brief Batch-aware multi-head attention operator with proper batch independence
 * @author David Sanftenberg
 * @date 2025-10-16
 *
 * This operator properly handles batched attention by maintaining batch dimension
 * throughout the computation and applying causal masking independently per sequence.
 * Unlike MPIAttentionOperator which flattens the batch dimension, this operator
 * ensures sequences in a batch cannot attend to each other's tokens.
 */

#pragma once

#include "../MpiKernelBase.h"
#include "tensors/TensorBase.h"
#include "PipelineStages.h"
#include <memory>
#include <vector>

namespace llaminar
{
    class PipelineBase; // Forward declaration for snapshot access
}

namespace llaminar
{

    /**
     * @brief Batch-aware multi-head attention with RoPE and distributed computation
     *
     * Input shape: [B, T, D] where B=batch_size, T=seq_len, D=d_model
     * Output shape: [B, T, D]
     *
     * Key features:
     * - Maintains batch independence (no cross-sequence attention)
     * - Per-sequence causal masking
     * - Distributed across MPI ranks by head dimension
     * - RoPE positional encoding applied per-sequence
     * - Optional KV cache support (future)
     *
     * Input tensors (10 total):
     *   0: input           [B, T, D]
     *   1: wq              [D, n_heads * head_dim]
     *   2: wk              [D, n_kv_heads * head_dim]
     *   3: wv              [D, n_kv_heads * head_dim]
     *   4: wo              [n_heads * head_dim, D]
     *   5: bq              [n_heads * head_dim] (optional, can be nullptr)
     *   6: bk              [n_kv_heads * head_dim] (optional, can be nullptr)
     *   7: bv              [n_kv_heads * head_dim] (optional, can be nullptr)
     *   8: k_cache         [0] or [B, n_kv_heads, max_seq, head_dim] (future)
     *   9: v_cache         [0] or [B, n_kv_heads, max_seq, head_dim] (future)
     *
     * Output tensors:
     *   0: output          [B, T, D]
     */
    class MPIAttentionBatchOperator : public MPIOperatorBase
    {
    public:
        /**
         * @brief Construct attention operator with architecture parameters
         * @param n_heads Number of query heads
         * @param n_kv_heads Number of key/value heads (for GQA)
         * @param head_dim Dimension per attention head
         * @param rope_freq_base RoPE frequency base (default 10000.0)
         */
        MPIAttentionBatchOperator(
            int n_heads,
            int n_kv_heads,
            int head_dim,
            float rope_freq_base = 10000.0f);

        ~MPIAttentionBatchOperator() override = default;

        // OperatorBase interface
        std::string getOperatorType() const override { return "MPIAttentionBatch"; }
        size_t getExpectedInputCount() const override { return 10; }
        size_t getExpectedOutputCount() const override { return 1; }

        bool execute(
            const std::vector<std::shared_ptr<TensorBase>> &inputs,
            std::vector<std::shared_ptr<TensorBase>> &outputs) override;

        bool validate(
            const std::vector<std::shared_ptr<TensorBase>> &inputs,
            const std::vector<std::shared_ptr<TensorBase>> &outputs) const override;

        /**
         * @brief Set snapshot capture callback for parity testing
         * @param capture_fn Function to call when capturing snapshots
         */
        void setSnapshotCallback(std::function<void(PipelineStage, int, const std::shared_ptr<TensorBase> &)> capture_fn)
        {
            snapshot_callback_ = capture_fn;
        }

        /**
         * @brief Set current layer index for snapshot capture
         * @param layer_idx Current layer being executed
         */
        void setLayerIndex(int layer_idx)
        {
            current_layer_idx_ = layer_idx;
        }

        /**
         * @brief Print performance breakdown (called by batch pipeline)
         */
        void printPerformanceBreakdown() const;

        /**
         * @brief Reset performance counters
         */
        void resetPerformanceCounters();

    private:
        /**
         * @brief Apply RoPE to Q and K tensors
         * @param q Query tensor [B, n_heads, T, head_dim]
         * @param k Key tensor [B, n_kv_heads, T, head_dim]
         * @param seq_len Sequence length
         */
        void applyRoPE(float *q, float *k, int batch_size, int seq_len);

        /**
         * @brief Compute attention scores with per-batch causal masking
         * @param q Query [B, n_heads_local, T, head_dim]
         * @param k Key [B, n_kv_heads_local, T, head_dim]
         * @param scores Output [B, n_heads_local, T, T]
         * @param batch_size Batch size
         * @param seq_len Sequence length
         */
        void computeAttentionScores(
            const float *q,
            const float *k,
            float *scores,
            int batch_size,
            int seq_len);

        /**
         * @brief Apply per-batch causal mask and softmax
         * @param scores [B, n_heads_local, T, T]
         * @param batch_size Batch size
         * @param seq_len Sequence length
         */
        void applyCausalMaskAndSoftmax(
            float *scores,
            int batch_size,
            int seq_len);

        /**
         * @brief Compute attention output: scores @ V
         * @param scores [B, n_heads_local, T, T]
         * @param v Value [B, n_kv_heads_local, T, head_dim]
         * @param output [B, n_heads_local, T, head_dim]
         */
        void computeAttentionOutput(
            const float *scores,
            const float *v,
            float *output,
            int batch_size,
            int seq_len);

        /**
         * @brief Get KV head distribution for current rank
         * @return Pair of (local_kv_heads, kv_head_offset)
         */
        std::pair<int, int> getKVHeadDistribution() const;

        /**
         * @brief Get KV head distribution for specific rank
         * @param rank MPI rank to query
         * @return Pair of (local_kv_heads, kv_head_offset)
         */
        std::pair<int, int> getKVHeadDistribution(int rank) const;

        // Architecture parameters
        int n_heads_;          // Total query heads
        int n_kv_heads_;       // Total KV heads (GQA support)
        int head_dim_;         // Dimension per head
        float rope_freq_base_; // RoPE frequency base

        // MPI distribution
        int n_heads_local_;    // Heads assigned to this rank
        int n_kv_heads_local_; // KV heads assigned to this rank
        int head_offset_;      // Starting head index for this rank

        // Snapshot capture callback (for parity testing)
        std::function<void(PipelineStage, int, const std::shared_ptr<TensorBase> &)> snapshot_callback_;
        int current_layer_idx_ = -1; // Current layer index for snapshot capture

        // Performance profiling accumulators (public for batch pipeline access)
        double total_qkv_proj_ms_ = 0.0;
        double total_rope_ms_ = 0.0;
        double total_gqa_expand_ms_ = 0.0;
        double total_scores_ms_ = 0.0;
        double total_softmax_ms_ = 0.0;
        double total_context_ms_ = 0.0;
        double total_output_prep_ms_ = 0.0;
        double total_output_proj_ms_ = 0.0;
        double total_mpi_reduce_ms_ = 0.0;
    };

} // namespace llaminar
