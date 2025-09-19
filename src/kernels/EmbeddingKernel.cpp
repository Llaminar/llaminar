#include "EmbeddingKernel.h"
#include "graph_compute.h" // For Tensor definition
#include "logger.h"
#include <chrono>

namespace llaminar
{

    EmbeddingKernel::EmbeddingKernel()
    {
        LOG_DEBUG("EmbeddingKernel initialized");
    }

    bool EmbeddingKernel::execute(const std::vector<std::shared_ptr<Tensor>> &inputs,
                                  std::vector<std::shared_ptr<Tensor>> &outputs)
    {
        if (!validate(inputs, outputs))
        {
            return false;
        }

        auto start = std::chrono::high_resolution_clock::now();

        // For embedding, we need token IDs as input (stored as float but used as int)
        auto token_tensor = inputs[0];    // [seq_len] - token IDs
        auto embedding_table = inputs[1]; // [vocab_size, hidden_size]
        auto output = outputs[0];         // [seq_len, hidden_size]

        int seq_len = token_tensor->shape[0];
        int vocab_size = embedding_table->shape[0];
        int hidden_size = embedding_table->shape[1];

        // Convert float token data to int IDs
        std::vector<int> token_ids(seq_len);
        for (int i = 0; i < seq_len; ++i)
        {
            token_ids[i] = static_cast<int>(token_tensor->data[i]);
        }

        lookupEmbeddings(token_ids.data(), embedding_table->data.data(),
                         output->data.data(), seq_len, vocab_size, hidden_size);

        auto end = std::chrono::high_resolution_clock::now();
        double execution_time = std::chrono::duration<double, std::milli>(end - start).count();

        LOG_DEBUG("Embedding executed in " + std::to_string(execution_time) + " ms");
        return true;
    }

    bool EmbeddingKernel::validate(const std::vector<std::shared_ptr<Tensor>> &inputs,
                                   const std::vector<std::shared_ptr<Tensor>> &outputs) const
    {
        if (inputs.size() != 2)
        {
            LOG_ERROR("Embedding requires exactly 2 inputs (token_ids, embedding_table)");
            return false;
        }

        if (outputs.size() != 1)
        {
            LOG_ERROR("Embedding requires exactly 1 output");
            return false;
        }

        auto token_tensor = inputs[0];
        auto embedding_table = inputs[1];
        auto output = outputs[0];

        if (token_tensor->shape.size() != 1)
        {
            LOG_ERROR("Embedding token_ids must be 1D tensor");
            return false;
        }

        if (embedding_table->shape.size() != 2)
        {
            LOG_ERROR("Embedding table must be 2D tensor");
            return false;
        }

        if (output->shape.size() != 2)
        {
            LOG_ERROR("Embedding output must be 2D tensor");
            return false;
        }

        if (token_tensor->shape[0] != output->shape[0] ||
            embedding_table->shape[1] != output->shape[1])
        {
            LOG_ERROR("Embedding dimension mismatch");
            return false;
        }

        return true;
    }

    void EmbeddingKernel::lookupEmbeddings(const int *token_ids, const float *embedding_table,
                                           float *output, int seq_len, int vocab_size, int hidden_size)
    {
        for (int seq = 0; seq < seq_len; ++seq)
        {
            int token_id = token_ids[seq];

            // Bounds check
            if (token_id < 0 || token_id >= vocab_size)
            {
                LOG_WARN("Token ID " + std::to_string(token_id) + " out of bounds [0, " + std::to_string(vocab_size) + ")");
                // Zero out this embedding
                for (int dim = 0; dim < hidden_size; ++dim)
                {
                    output[seq * hidden_size + dim] = 0.0f;
                }
                continue;
            }

            // Copy embedding
            const float *embedding = embedding_table + token_id * hidden_size;
            float *output_embedding = output + seq * hidden_size;
            for (int dim = 0; dim < hidden_size; ++dim)
            {
                output_embedding[dim] = embedding[dim];
            }
        }
    }

} // namespace llaminar