#pragma once

#include <vector>
#include <memory>
#include "../tensor.h"

namespace llaminar
{

    /**
     * @brief Embedding lookup kernel for transformer input processing
     *
     * Performs embedding table lookup for token IDs:
     * output[seq, :] = embedding_table[token_ids[seq], :]
     *
     * Expected inputs:
     * - token_ids: [seq_len] - token IDs (stored as float but treated as int)
     * - embedding_table: [vocab_size, hidden_size] - embedding lookup table
     *
     * Expected outputs:
     * - output: [seq_len, hidden_size] - embedded token representations
     */
    class EmbeddingKernel
    {
    public:
        EmbeddingKernel();

        /**
         * @brief Execute embedding lookup
         * @param inputs Vector containing token IDs and embedding table
         * @param outputs Vector containing output embeddings
         * @return true if execution succeeded, false otherwise
         */
        bool execute(const std::vector<std::shared_ptr<llaminar::Tensor>> &inputs,
                     std::vector<std::shared_ptr<llaminar::Tensor>> &outputs);

        /**
         * @brief Validate input and output tensor shapes and types
         * @param inputs Input tensors to validate
         * @param outputs Output tensors to validate
         * @return true if tensors are valid, false otherwise
         */
        bool validate(const std::vector<std::shared_ptr<llaminar::Tensor>> &inputs,
                      const std::vector<std::shared_ptr<llaminar::Tensor>> &outputs) const;

    private:
        /**
         * @brief Core embedding lookup computation
         * @param token_ids Token ID array
         * @param embedding_table Embedding table data pointer
         * @param output Output data pointer
         * @param seq_len Sequence length
         * @param vocab_size Vocabulary size
         * @param hidden_size Hidden dimension size
         */
        void lookupEmbeddings(const int *token_ids, const float *embedding_table,
                              float *output, int seq_len, int vocab_size, int hidden_size);
    };

} // namespace llaminar