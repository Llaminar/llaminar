#pragma once

#include "../mpi_kernel_base.h"
#include "../pipeline_stages.h"
#include <string>
#include <vector>
#include <memory>
#include <functional>

namespace llaminar
{

    /**
     * @brief MPI-enabled multi-head attention kernel with head-wise distribution
     *
     * This kernel distributes attention heads across MPI processes for parallel computation.
     * Each process handles a subset of attention heads, with global communication for
     * input distribution and (optionally) output gathering.
     *
     * Distribution Strategy:
     * - HEAD_WISE: Each process handles a subset of attention heads
     * - Input / KV Cache: Replicated across all processes (future optimization may shard)
     * - Q/K/V Projections: Distributed by heads
     * - Attention Computation: Parallel across local heads only
     * - Output Projection: Applied to local attended heads; final tensor may be partial or global
     *
     * @note Output Contract (Always-Partial):
     * The kernel always emits ONLY the local partial contribution corresponding to this rank's owned
     * attention head subset (after local output projection). No implicit MPI gather/reduction occurs
     * inside the kernel. Reconstruction of a fully replicated hidden state (when required) is the
     * responsibility of the caller or downstream pipeline stage (e.g., explicit MPI_Allreduce for
     * additive row-sharded projections or Allgather + reshape for head concatenation). This explicit
     * contract eliminates hidden synchronization, prevents double reductions, and simplifies reasoning
     * about distributed correctness. Any previous environment flags that toggled an internal gather
     * have been removed.
     *
     * Rationale:
     * - Avoids implicit global synchronization hidden inside the kernel.
     * - Enables downstream kernels (norm / MLP) to operate purely on local shards where possible.
     * - Makes communication patterns explicit and testable in higher-level pipeline code.
     *
     * Caller Responsibilities:
     * 1. Determine reconstruction semantics: sum (row/feature split) vs concat (head split).
     * 2. Perform the appropriate collective (e.g. `MPI_Allreduce` for additive partitions).
     * 3. Avoid accidental reuse of the partial tensor as if it were replicated (enable a future
     *    guard flag such as `LLAMINAR_ASSERT_REPLICATED_MISUSE` to catch mistakes).
     *
     * Example (row-sharded Wo producing additive partials):
     * @code
     * // After attention kernel execute with gather disabled
     * std::vector<float> full(d_model * seq_len);
     * MPI_Allreduce(local->data(), full.data(), local->size(), MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
     * @endcode
     *
     * There is no code path that implicitly produces a replicated activation; treating the partial
     * output as fully replicated without reconstruction is a usage error (a future guard flag may
     * detect this misuse).
     *
     * Misuse Guard (optional):
     *  - Set `LLAMINAR_ASSERT_REPLICATED_MISUSE=1` to embed a microscopic per-rank canary value in
     *    the final element of the partial output. A downstream debug pass can inspect divergence in
     *    that terminal element across ranks before reduction to detect unintended replicated use.
     */
    class MPIAttentionKernel : public MPIKernelBase
    {
    public:
        // Output assembly / distribution mode (scaffolding for hybrid head + TP design)
        enum class AttentionOutputMode
        {
            LocalHeads,                ///< Return only local head slice (no gather)
            GatherHeadsPostProjection, ///< Project locally then gather concatenated hidden (future)
            GatherHeadsPreProjection,  ///< Gather head contexts before output projection (future)
            Replicated                 ///< Force fully replicated output (debug / legacy)
        };

        struct AttentionResultMeta
        {
            AttentionOutputMode mode = AttentionOutputMode::LocalHeads;
            bool concatenated = false; ///< true if full hidden dimension assembled
            bool replicated = false;   ///< true if buffer identical on all ranks
            int local_head_offset = 0; ///< starting head index owned by this rank
            int local_head_count = 0;  ///< number of heads owned by this rank
        };
        /**
         * @brief Distribution strategies for attention computation
         */
        enum class DistributionStrategy
        {
            HEAD_WISE,    ///< Distribute by attention heads
            SEQUENCE_WISE ///< Distribute by sequence dimension (future extension)
        };

        /**
         * @brief Constructor for MPI attention kernel
         * @param n_head Total number of attention heads
         * @param n_head_kv Number of key-value heads (for grouped attention)
         * @param head_dim Dimension per attention head
         * @param rope_freq_base Base frequency for rotary embeddings
         * @param strategy Distribution strategy to use
         */
        MPIAttentionKernel(int n_head, int n_head_kv, int head_dim,
                           float rope_freq_base = 10000.0f,
                           DistributionStrategy strategy = DistributionStrategy::HEAD_WISE);

        ~MPIAttentionKernel() = default;

        // KernelBase interface implementation
        bool execute(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                     std::vector<std::shared_ptr<TensorBase>> &outputs) override;

        // Output mode configuration
        void setOutputMode(AttentionOutputMode m) { output_mode_ = m; }
        AttentionOutputMode outputMode() const { return output_mode_; }
        const AttentionResultMeta &last_result_meta() const { return last_meta_; }

        bool validate(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                      const std::vector<std::shared_ptr<TensorBase>> &outputs) const override;

        std::string getKernelType() const override { return "MPIAttention"; }
        size_t getExpectedInputCount() const override { return 7; } // input, wq, wk, wv, wo, k_cache, v_cache
        size_t getExpectedOutputCount() const override { return 1; }

        // Configuration methods
        void setHeadDimensions(int n_head, int n_head_kv, int head_dim);
        void setSequencePosition(int n_past) { n_past_ = n_past; }
        void setLayerIndex(int idx) { layer_index_ = idx; }
        int getSequencePosition() const { return n_past_; }
        // Set externally computed expected total window length (prefill_len + decoded_tokens)
        void setExpectedTotalWindow(size_t expected) { expected_total_window_ = expected; }
        size_t getExpectedTotalWindow() const { return expected_total_window_; }
        void resetDecodeWindowTracking() { last_seen_decode_T_ = 0; }
        void setDistributionStrategy(DistributionStrategy strategy) { strategy_ = strategy; }

        /**
         * @brief Get head distribution for this process
         * @return Pair of (local_heads, head_offset)
         */
        std::pair<int, int> getHeadDistribution() const;

        /**
         * @brief Get head distribution for a specific rank
         * @param rank Target rank
         * @return Pair of (local_heads, head_offset)
         */
        std::pair<int, int> getHeadDistribution(int rank) const;

        /**
         * @brief Get K/V head distribution for this process (for GQA models)
         * @return Pair of (local_kv_heads, kv_head_offset)
         */
        std::pair<int, int> getKVHeadDistribution() const;

        /**
         * @brief Get K/V head distribution for a specific rank (for GQA models)
         * @param rank Target rank
         * @return Pair of (local_kv_heads, kv_head_offset)
         */
        std::pair<int, int> getKVHeadDistribution(int rank) const;

        /**
         * @brief Test harness helper: invoke the private output projection path directly.
         *        Exposed for unit/integration testing of TP simulation logic without
         *        running the full attention pipeline.
         * @note  Not intended for production use; keeps underlying implementation private.
         */
        void testInvokeOutputProjection(const std::shared_ptr<TensorBase> &local_attended_output,
                                        const std::shared_ptr<TensorBase> &local_wo,
                                        std::shared_ptr<TensorBase> &local_final_output,
                                        size_t seq_len, int local_heads, size_t d_model)
        {
            computeLocalOutputProjection(local_attended_output, local_wo, local_final_output,
                                         seq_len, local_heads, d_model);
        }

        /**
         * @brief Callback for capturing intermediate attention snapshots
         * @param stage Pipeline stage identifier
         * @param layer_idx Layer index
         * @param data Pointer to tensor data
         * @param seq_len Sequence length
         * @param feature_dim Feature dimension
         */
        using SnapshotCallback = std::function<void(PipelineStage stage, int layer_idx, const float *data, int seq_len, int feature_dim)>;

        void setSnapshotCallback(SnapshotCallback callback) { snapshot_callback_ = callback; }

    private:
        /**
         * @brief Distribute input data and weights to all processes
         * @param global_input Global input tensor
         * @param global_wq Global query weight matrix
         * @param global_wk Global key weight matrix
         * @param global_wv Global value weight matrix
         * @param global_wo Global output weight matrix
         * @param local_wq Local query weight subset
         * @param local_wk Local key weight subset
         * @param local_wv Local value weight subset
         * @param local_wo Local output weight subset
         * @param seq_len Sequence length
         * @param d_model Model dimension
         */
        void distributeInputs(const std::shared_ptr<TensorBase> &global_input,
                              const std::shared_ptr<TensorBase> &global_wq,
                              const std::shared_ptr<TensorBase> &global_wk,
                              const std::shared_ptr<TensorBase> &global_wv,
                              const std::shared_ptr<TensorBase> &global_wo,
                              std::shared_ptr<TensorBase> &local_wq,
                              std::shared_ptr<TensorBase> &local_wk,
                              std::shared_ptr<TensorBase> &local_wv,
                              std::shared_ptr<TensorBase> &local_wo,
                              size_t seq_len, size_t d_model);

        /**
         * @brief Compute local Q, K, V projections for assigned heads using COSMA
         * @param input Input tensor
         * @param local_wq Local query weight tensor
         * @param local_wk Local key weight tensor
         * @param local_wv Local value weight tensor
         * @param local_q Output local query projection tensor
         * @param local_k Output local key projection tensor
         * @param local_v Output local value projection tensor
         * @param seq_len Sequence length
         * @param d_model Model dimension
         */
        void computeLocalProjections(const std::shared_ptr<TensorBase> &input,
                                     const std::shared_ptr<TensorBase> &local_wq,
                                     const std::shared_ptr<TensorBase> &local_wk,
                                     const std::shared_ptr<TensorBase> &local_wv,
                                     std::shared_ptr<TensorBase> &local_q,
                                     std::shared_ptr<TensorBase> &local_k,
                                     std::shared_ptr<TensorBase> &local_v,
                                     size_t seq_len, size_t d_model);

        /**
         * @brief Compute attention for local heads
         * @param local_q Local query projections
         * @param local_k Local key projections
         * @param local_v Local value projections
         * @param local_output Local attention output
         * @param seq_len Sequence length
         * @param local_heads Number of local heads
         */
        void computeLocalAttention(const float *local_q, const float *local_k, const float *local_v,
                                   float *local_output, size_t seq_len, int local_heads);

        /**
         * @brief Apply RoPE to local Q and K projections
         * @param local_q Local query tensor
         * @param local_k Local key tensor
         * @param seq_len Sequence length
         * @param local_heads Number of local heads
         */
        void applyLocalRoPE(float *local_q, float *local_k, size_t seq_len, int local_heads);

        /**
         * @brief Compute attention scores and softmax for local heads
         * @param local_q Local query projections
         * @param local_k Local key projections
         * @param scores Local attention scores
         * @param seq_len Sequence length
         * @param local_heads Number of local heads
         */
        void computeLocalAttentionScores(const float *local_q, const float *local_k, float *scores,
                                         size_t seq_len, int local_heads);

        /**
         * @brief Apply local attention scores to values
         * @param scores Local attention scores
         * @param local_v Local value projections
         * @param local_attended_output Local attended output
         * @param seq_len Sequence length
         * @param local_heads Number of local heads
         */
        void applyLocalAttention(const float *scores, const float *local_v, float *local_attended_output,
                                 size_t seq_len, int local_heads);

        /**
         * @brief Compute final output projection for local heads using COSMA
         * @param local_attended_output Local attended output tensor
         * @param local_wo Local output weight tensor
         * @param local_final_output Local final output tensor
         * @param seq_len Sequence length
         * @param local_heads Number of local heads
         */
        void computeLocalOutputProjection(const std::shared_ptr<TensorBase> &local_attended_output,
                                          const std::shared_ptr<TensorBase> &local_wo,
                                          std::shared_ptr<TensorBase> &local_final_output,
                                          size_t seq_len, int local_heads, size_t d_model);

        std::shared_ptr<TensorBase> createLocalSimpleTensor(const std::vector<size_t> &shape) const;

        // Configuration parameters
        int n_head_;                                                        ///< Total number of attention heads
        int n_head_kv_;                                                     ///< Number of key-value heads (grouped attention)
        int head_dim_;                                                      ///< Dimension per attention head
        int n_past_;                                                        ///< Number of past tokens for position embedding
        int layer_index_ = -1;                                              ///< Layer index for diagnostics
        int d_model_;                                                       ///< Model dimension
        AttentionOutputMode output_mode_ = AttentionOutputMode::LocalHeads; ///< Attention output mode (DEFAULT: LocalHeads is for future head-sharded TP; current row-partitioned W_o requires GatherHeadsPostProjection!)
        SnapshotCallback snapshot_callback_;                                ///< Optional callback for intermediate snapshots
        size_t expected_total_window_ = 0;                                  ///< Expected causal window (external assertion)
        size_t last_seen_decode_T_ = 0;                                     ///< Per-instance last seen T for monotonic check
        float rope_freq_base_;                                              ///< Base frequency for rotary embeddings
        DistributionStrategy strategy_;                                     ///< Distribution strategy
        AttentionResultMeta last_meta_{};                                   ///< metadata from last execute
    };

} // namespace llaminar