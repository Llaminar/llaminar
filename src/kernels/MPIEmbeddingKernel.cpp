#include "MPIEmbeddingKernel.h"
#include "../logger.h"
#include <algorithm>
#include <cmath>

namespace llaminar
{

    MPIEmbeddingKernel::MPIEmbeddingKernel(size_t vocab_size, size_t embedding_dim)
        : vocab_size_(vocab_size), embedding_dim_(embedding_dim)
    {
        initializeMPI();

        // Calculate vocabulary partition for this rank
        size_t base_partition_size = vocab_size_ / size_;
        size_t remainder = vocab_size_ % size_;

        if (rank_ < static_cast<int>(remainder))
        {
            local_vocab_size_ = base_partition_size + 1;
            local_vocab_start_ = rank_ * local_vocab_size_;
        }
        else
        {
            local_vocab_size_ = base_partition_size;
            local_vocab_start_ = remainder * (base_partition_size + 1) +
                                 (rank_ - remainder) * base_partition_size;
        }

        local_vocab_end_ = local_vocab_start_ + local_vocab_size_;

        LOG_DEBUG("MPIEmbeddingKernel initialized on rank " << rank_ << "/" << size_
                                                            << " with vocab_size=" << vocab_size << ", embedding_dim=" << embedding_dim
                                                            << ", local_vocab_range=[" << local_vocab_start_ << ", " << local_vocab_end_ << ")");
    }

    bool MPIEmbeddingKernel::execute(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                                     std::vector<std::shared_ptr<TensorBase>> &outputs)
    {
        if (!validate(inputs, outputs))
        {
            LOG_ERROR("MPIEmbeddingKernel validation failed");
            return false;
        }

        auto token_ids_tensor = inputs[0];
        auto embedding_table = inputs[1];
        auto output = outputs[0];

        size_t seq_len = token_ids_tensor->shape()[0];

        // Extract token IDs
        std::vector<int> token_ids(seq_len);
        std::copy(token_ids_tensor->data(), token_ids_tensor->data() + seq_len, token_ids.begin());

        // Orientation & mode already determined in validate()
        bool full_table_mode = full_table_mode_;
        bool transposed = transposed_;
        LOG_DEBUG("MPIEmbeddingKernel execute: full_table_mode=" << full_table_mode
                                                                 << " transposed=" << transposed
                                                                 << " shape=[" << embedding_table->shape()[0]
                                                                 << ", " << embedding_table->shape()[1] << "] seq_len=" << seq_len);

        // Create local output buffer for this rank's contribution
        auto local_output = TensorFactory::create_simple({static_cast<int>(seq_len), static_cast<int>(embedding_dim_)});
        std::fill(local_output->data(), local_output->data() + seq_len * embedding_dim_, 0.0f);

        // Local embedding lookup (all tokens if full table, owned tokens if sharded)
        computeLocalEmbedding(token_ids.data(), embedding_table->data(),
                              local_output->data(), seq_len, embedding_dim_, full_table_mode, transposed,
                              embedding_table->shape()[0], embedding_table->shape()[1]);

        // Gather embeddings (Allreduce for sharded, copy for full table mode)
        gatherEmbeddings(local_output, output, seq_len, embedding_dim_, full_table_mode);

        return true;
    }

    bool MPIEmbeddingKernel::validate(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                                      const std::vector<std::shared_ptr<TensorBase>> &outputs) const
    {
        if (inputs.size() != 2 || outputs.size() != 1)
        {
            LOG_ERROR("MPIEmbeddingKernel: Expected 2 inputs and 1 output, got "
                      << inputs.size() << " inputs and " << outputs.size() << " outputs");
            return false;
        }

        auto token_ids = inputs[0];
        auto embedding_table = inputs[1];
        auto output = outputs[0];

        if (!token_ids || !embedding_table || !output)
        {
            LOG_ERROR("MPIEmbeddingKernel: Null tensor provided");
            return false;
        }

        // Check token_ids is 1D
        if (token_ids->shape().size() != 1)
        {
            LOG_ERROR("MPIEmbeddingKernel: Token IDs must be 1D, got "
                      << token_ids->shape().size() << " dimensions");
            return false;
        }

        // Embedding table may be sharded [local_vocab_size, dim] or full [vocab_size, dim]
        if (embedding_table->shape().size() != 2)
        {
            LOG_ERROR("MPIEmbeddingKernel: Embedding table must be 2D, got " << embedding_table->shape().size());
            return false;
        }
        // Accept both standard and transposed orientations:
        // Standard:   [vocab_size, embedding_dim] or [local_vocab_size, embedding_dim]
        // Transposed: [embedding_dim, vocab_size] or [embedding_dim, local_vocab_size]
        size_t r0 = embedding_table->shape()[0];
        size_t r1 = embedding_table->shape()[1];
        bool full_std = (r0 == vocab_size_ && r1 == embedding_dim_);
        bool shard_std = (r0 == local_vocab_size_ && r1 == embedding_dim_);
        bool full_trans = (r0 == embedding_dim_ && r1 == vocab_size_);
        bool shard_trans = (r0 == embedding_dim_ && r1 == local_vocab_size_);
        if (!(full_std || shard_std || full_trans || shard_trans))
        {
            LOG_ERROR("MPIEmbeddingKernel: Embedding table shape mismatch. Expected one of: "
                      << "[" << vocab_size_ << ", " << embedding_dim_ << "] (full), "
                      << "[" << local_vocab_size_ << ", " << embedding_dim_ << "] (shard), "
                      << "[" << embedding_dim_ << ", " << vocab_size_ << "] (full transposed), "
                      << "[" << embedding_dim_ << ", " << local_vocab_size_ << "] (shard transposed); got ["
                      << r0 << ", " << r1 << "]");
            return false;
        }

        // Cache orientation flags for execute()
        full_table_mode_ = (full_std || full_trans);
        transposed_ = (full_trans || shard_trans);
        LOG_DEBUG("MPIEmbeddingKernel validate: full_table_mode_=" << full_table_mode_ << " transposed_=" << transposed_
                                                                   << " table_shape=[" << r0 << ", " << r1 << "]");

        // Output must be [seq_len, embedding_dim]
        size_t seq_len = token_ids->shape()[0];
        if (output->shape().size() != 2 || output->shape()[0] != seq_len || output->shape()[1] != embedding_dim_)
        {
            LOG_ERROR("MPIEmbeddingKernel: Output shape mismatch. Expected [" << seq_len << ", " << embedding_dim_ << "], got ["
                                                                              << output->shape()[0] << ", " << output->shape()[1] << "]");
            return false;
        }

        return true;
    }

    void MPIEmbeddingKernel::computeLocalEmbedding(const int *token_ids, const float *embedding_table,
                                                   float *output, size_t seq_len, size_t embedding_dim,
                                                   bool full_table_mode, bool transposed,
                                                   size_t table_rows, size_t table_cols)
    {
        for (size_t i = 0; i < seq_len; ++i)
        {
            int token_id = token_ids[i];
            if (token_id < 0 || token_id >= static_cast<int>(vocab_size_))
            {
                LOG_WARN("MPIEmbeddingKernel: token id out of range: " << token_id);
                continue;
            }

            if (full_table_mode)
            {
                if (!transposed)
                {
                    const float *embedding = embedding_table + static_cast<size_t>(token_id) * embedding_dim;
                    std::copy(embedding, embedding + embedding_dim, output + i * embedding_dim);
                }
                else
                {
                    // Table layout: [embedding_dim, vocab]; column-major logical access per token
                    // Access element (dim_idx, token_id)
                    for (size_t d = 0; d < embedding_dim; ++d)
                    {
                        output[i * embedding_dim + d] = embedding_table[d * table_cols + token_id];
                    }
                }
            }
            else if (token_id >= static_cast<int>(local_vocab_start_) && token_id < static_cast<int>(local_vocab_end_))
            {
                int local_token_id = token_id - static_cast<int>(local_vocab_start_);
                if (local_token_id >= 0 && local_token_id < static_cast<int>(local_vocab_size_))
                {
                    if (!transposed)
                    {
                        const float *embedding = embedding_table + static_cast<size_t>(local_token_id) * embedding_dim;
                        std::copy(embedding, embedding + embedding_dim, output + i * embedding_dim);
                    }
                    else
                    {
                        for (size_t d = 0; d < embedding_dim; ++d)
                        {
                            output[i * embedding_dim + d] = embedding_table[d * table_cols + local_token_id];
                        }
                    }
                }
            }
        }
    }

    void MPIEmbeddingKernel::gatherEmbeddings(const std::shared_ptr<TensorBase> &local_output,
                                              std::shared_ptr<TensorBase> &global_output,
                                              size_t seq_len, size_t embedding_dim,
                                              bool full_table_mode)
    {
        if (full_table_mode)
        {
            // Every rank already has full embeddings
            std::copy(local_output->data(), local_output->data() + seq_len * embedding_dim, global_output->data());
            return;
        }

        // Reduce (sum) partial embeddings across ranks
        checkMPIError(MPI_Allreduce(local_output->data(), global_output->data(),
                                    static_cast<int>(seq_len * embedding_dim), MPI_FLOAT, MPI_SUM, getComm()),
                      "MPI_Allreduce in gatherEmbeddings");
    }

    int MPIEmbeddingKernel::getTokenRank(int token_id) const
    {
        if (token_id < 0 || token_id >= static_cast<int>(vocab_size_))
        {
            return -1; // Invalid token ID
        }

        size_t base_partition_size = vocab_size_ / size_;
        size_t remainder = vocab_size_ % size_;
        size_t threshold = remainder * (base_partition_size + 1);

        if (token_id < static_cast<int>(threshold))
        {
            return token_id / (base_partition_size + 1);
        }
        else
        {
            return remainder + (token_id - threshold) / base_partition_size;
        }
    }

    int MPIEmbeddingKernel::getLocalTokenId(int global_token_id) const
    {
        if (global_token_id < static_cast<int>(local_vocab_start_) ||
            global_token_id >= static_cast<int>(local_vocab_end_))
        {
            return -1; // Not owned
        }
        return global_token_id - static_cast<int>(local_vocab_start_);
    }

} // namespace llaminar