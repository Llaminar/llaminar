#pragma once

#include "../MpiKernelBase.h"
#include <string>
#include <vector>
#include <memory>

namespace llaminar
{

    /**
     * @brief MPI-distributed embedding lookup kernel with vocabulary sharding
     *
     * Distribution strategy: VOCAB_WISE sharding
     * - Each rank handles a subset of the vocabulary entries
     * - Token lookups use MPI_Allgather for distributed access
     * - Embedding table is sharded across ranks to reduce memory usage
     *
     * Batch Support:
     * - Accepts 1D [seq_len] or 2D [batch, seq_len] token ID tensors
     * - Outputs [seq_len, embedding_dim] or [batch, seq_len, embedding_dim]
     * - Processes all tokens in flattened order for efficiency
     */
    class MPIEmbeddingOperator : public MPIOperatorBase
    {
    public:
        MPIEmbeddingOperator(size_t vocab_size, size_t embedding_dim);
        ~MPIEmbeddingOperator() = default;

        // OperatorBase interface implementation
        bool execute(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                     std::vector<std::shared_ptr<TensorBase>> &outputs) override;

        std::string getOperatorType() const override { return "MPIEmbedding"; }
        size_t getExpectedInputCount() const override { return 2; }
        size_t getExpectedOutputCount() const override { return 1; }

    private:
        /**
         * @brief Core embedding lookup computation with MPI distribution
         * @param token_ids Token ID array
         * @param local_embedding_table Local portion of embedding table on this rank
         * @param output Output data pointer
         * @param seq_len Sequence length
         * @param embedding_dim Embedding dimension
         */
        void computeLocalEmbedding(const int *token_ids, const float *embedding_table,
                                   float *output, size_t seq_len, size_t embedding_dim,
                                   bool full_table_mode, bool transposed,
                                   size_t table_rows, size_t table_cols);

        /**
         * @brief Gather embeddings from all ranks using MPI_Allgather
         * @param local_output Local embedding output from this rank
         * @param global_output Final combined embedding output
         * @param seq_len Sequence length
         * @param embedding_dim Embedding dimension
         */
        void gatherEmbeddings(const std::shared_ptr<TensorBase> &local_output,
                              std::shared_ptr<TensorBase> &global_output,
                              size_t seq_len, size_t embedding_dim,
                              bool full_table_mode);

        /**
         * @brief Determine which rank owns a particular vocabulary entry
         * @param token_id The token ID to lookup
         * @return The rank that owns this token's embedding
         */
        int getTokenRank(int token_id) const;

        /**
         * @brief Get the local token ID for this rank's vocabulary subset
         * @param global_token_id Global token ID
         * @return Local token ID within this rank's vocabulary partition
         */
        int getLocalTokenId(int global_token_id) const;

        /**
         * @brief Validate tensor shapes and dimensions for MPI embedding
         */
        bool validate(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                      const std::vector<std::shared_ptr<TensorBase>> &outputs) const override;

        size_t vocab_size_;        // Total vocabulary size
        size_t embedding_dim_;     // Embedding dimension
        size_t local_vocab_start_; // Start index of this rank's vocabulary partition
        size_t local_vocab_end_;   // End index of this rank's vocabulary partition
        size_t local_vocab_size_;  // Number of vocabulary entries on this rank
        // Orientation / mode flags (set during validate, used in execute)
        mutable bool full_table_mode_ = false; // True if embedding table contains full vocab on each rank
        mutable bool transposed_ = false;      // True if table shape is [embedding_dim, vocab]
    };

} // namespace llaminar