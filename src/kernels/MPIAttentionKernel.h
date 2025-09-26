#pragma once

#include "../mpi_kernel_base.h"
#include <string>
#include <vector>
#include <memory>

namespace llaminar
{

    /**
     * @brief MPI-enabled multi-head attention kernel with head-wise distribution
     *
     * This kernel distributes attention heads across MPI processes for parallel computation.
     * Each process handles a subset of attention heads, with global communication for
     * input distribution and output gathering.
     *
     * Distribution Strategy:
     * - HEAD_WISE: Each process handles a subset of attention heads
     * - Input/KV Cache: Replicated across all processes
     * - Q/K/V Projections: Distributed by heads
     * - Attention Computation: Parallel across heads
     * - Output: Gathered from all processes
     */
    class MPIAttentionKernel : public MPIKernelBase
    {
    public:
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
         * @param strategy Distribution strategy to use
         */
        MPIAttentionKernel(int n_head, int n_head_kv, int head_dim,
                           DistributionStrategy strategy = DistributionStrategy::HEAD_WISE);

        ~MPIAttentionKernel() = default;

        // KernelBase interface implementation
        bool execute(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                     std::vector<std::shared_ptr<TensorBase>> &outputs) override;

        bool validate(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                      const std::vector<std::shared_ptr<TensorBase>> &outputs) const override;

        std::string getKernelType() const override { return "MPIAttention"; }
        size_t getExpectedInputCount() const override { return 7; } // input, wq, wk, wv, wo, k_cache, v_cache
        size_t getExpectedOutputCount() const override { return 1; }

        // Configuration methods
        void setHeadDimensions(int n_head, int n_head_kv, int head_dim);
        void setSequencePosition(int n_past) { n_past_ = n_past; }
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

        /**
         * @brief Gather final outputs from all processes
         * @param local_output Local output from this process
         * @param global_output Global output tensor
         * @param seq_len Sequence length
         * @param d_model Model dimension
         */
        void gatherOutput(const std::shared_ptr<TensorBase> &local_output,
                          std::shared_ptr<TensorBase> &global_output,
                          size_t seq_len, size_t d_model);

        std::shared_ptr<TensorBase> createLocalSimpleTensor(const std::vector<size_t> &shape) const;

        // Configuration parameters
        int n_head_;                    ///< Total number of attention heads
        int n_head_kv_;                 ///< Number of key-value heads (grouped attention)
        int head_dim_;                  ///< Dimension per attention head
        int n_past_;                    ///< Number of past tokens for position embedding
        DistributionStrategy strategy_; ///< Distribution strategy
    };

} // namespace llaminar