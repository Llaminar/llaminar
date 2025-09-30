/**
 * @file MPIEmbeddingKernel.cpp
 * @brief Token embedding lookup kernel with optional positional addition; MPI-replicated strategy.
 *
 * @section Contract
 * Inputs:
 *  - inputs[0]: Token ID tensor [seq_len] or [batch, seq_len] of int32/int16 supported types.
 *  - inputs[1]: Embedding weight matrix [vocab_size, embed_dim] (replicated across ranks).
 *  - inputs[2] (optional): Positional embedding matrix [max_pos, embed_dim].
 * Outputs:
 *  - outputs[0]: Activation tensor [seq_len, embed_dim] (or [batch, seq_len, embed_dim]) in row-major contiguous form.
 * Semantics:
 *  - Performs gather rows corresponding to token IDs.
 *  - If positional provided, adds position[row] elementwise (broadcast over batch if present).
 * Distribution:
 *  - All ranks perform identical gathers (replicated); future optimization could shard vocab with AllGather on demand.
 * Error Modes:
 *  - Token ID out of range -> LOG_ERROR + return false.
 *  - Shape/type mismatch across inputs.
 * Performance Notes:
 *  - Small, memory latency bound; parallelized over sequence tokens with OpenMP if beneficial.
 *  - Deterministic given identical token order.
 * Numerical Expectations:
 *  - Exact copy for embedding rows; positional add in float32 accumulation.
 * Threading:
 *  - Read-only access to weights; writes to distinct output rows (no races).
 * Future Extensions:
 *  - Vocabulary sharding + on-demand caching, fused quantized dequant gather.
 *  - Optional stop-gradient semantics for fine-tuning scenarios.
 * @author David Sanftenberg
 */
#include "MPIEmbeddingKernel.h"
#include "../logger.h"
#include "../utils/debug_env.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace
{
    /**
     * @brief Runtime configuration for embedding pipeline diagnostics.
     */
    struct EmbeddingDebugConfig
    {
        bool enabled = false;  //!< Whether extended diagnostics are enabled.
        size_t max_tokens = 2; //!< Maximum number of tokens to preview per log line.
        size_t max_dims = 8;   //!< Maximum number of embedding dimensions to preview per token.
    };

    /**
     * @brief Statistics gathered from an embedding buffer.
     */
    struct EmbeddingStats
    {
        float min = std::numeric_limits<float>::infinity();  //!< Minimum finite value observed.
        float max = -std::numeric_limits<float>::infinity(); //!< Maximum finite value observed.
        size_t finite_count = 0;                             //!< Number of finite values encountered.
        size_t nan_count = 0;                                //!< Count of NaN values.
        size_t inf_count = 0;                                //!< Count of +/- infinity values.
        size_t zero_count = 0;                               //!< Count of exact zeros.
        std::optional<size_t> first_nan_index;               //!< First index where a NaN was encountered.
        std::optional<size_t> first_inf_index;               //!< First index where an infinity was encountered.
    };

    /**
     * @brief Parse the embedding debug configuration from environment variables.
     *
     * Environment variables:
     * - `LLAMINAR_EMBED_TRACE`: enable diagnostics when set to a non-zero value.
     * - `LLAMINAR_EMBED_TRACE_TOKENS`: optional override for maximum tokens previewed.
     * - `LLAMINAR_EMBED_TRACE_DIMS`: optional override for maximum dimensions previewed.
     *
     * @return const EmbeddingDebugConfig& Cached diagnostic configuration.
     */
    const EmbeddingDebugConfig &getEmbeddingDebugConfig()
    {
        static const EmbeddingDebugConfig config = []
        {
            EmbeddingDebugConfig cfg{};
            if (!llaminar::debugEnv().embedding.trace)
            {
                return cfg;
            }

            cfg.enabled = true;

            cfg.max_tokens = static_cast<size_t>(llaminar::debugEnv().embedding.trace_tokens);
            cfg.max_dims = static_cast<size_t>(llaminar::debugEnv().embedding.trace_dims);

            return cfg;
        }();
        return config;
    }

    /**
     * @brief Compute statistics for an embedding buffer.
     *
     * @param data Pointer to embedding values.
     * @param count Number of elements pointed to by @p data.
     * @return EmbeddingStats Summary statistics for the provided data.
     */
    EmbeddingStats computeEmbeddingStats(const float *data, size_t count)
    {
        EmbeddingStats stats;
        if (data == nullptr || count == 0)
        {
            return stats;
        }

        for (size_t idx = 0; idx < count; ++idx)
        {
            float value = data[idx];
            if (std::isnan(value))
            {
                ++stats.nan_count;
                if (!stats.first_nan_index.has_value())
                {
                    stats.first_nan_index = idx;
                }
                continue;
            }

            if (std::isinf(value))
            {
                ++stats.inf_count;
                if (!stats.first_inf_index.has_value())
                {
                    stats.first_inf_index = idx;
                }
                continue;
            }

            if (value == 0.0f)
            {
                ++stats.zero_count;
            }

            stats.min = std::min(stats.min, value);
            stats.max = std::max(stats.max, value);
            ++stats.finite_count;
        }

        if (stats.finite_count == 0)
        {
            stats.min = std::numeric_limits<float>::quiet_NaN();
            stats.max = std::numeric_limits<float>::quiet_NaN();
        }

        return stats;
    }

    /**
     * @brief Format a compact preview of the embedding buffer for logging.
     *
     * @param data Embedding buffer pointer.
     * @param seq_len Sequence length represented by the buffer.
     * @param embedding_dim Embedding dimension.
     * @param max_tokens Maximum number of tokens to preview.
     * @param max_dims Maximum number of dimensions per token to preview.
     * @return std::string Human-readable preview string.
     */
    std::string formatEmbeddingPreview(const float *data, size_t seq_len, size_t embedding_dim,
                                       size_t max_tokens, size_t max_dims)
    {
        if (data == nullptr || seq_len == 0 || embedding_dim == 0)
        {
            return "<empty>";
        }

        const size_t token_limit = std::min(seq_len, max_tokens);
        const size_t dim_limit = std::min(embedding_dim, max_dims);

        std::ostringstream oss;
        oss << std::fixed << std::setprecision(4);

        for (size_t token = 0; token < token_limit; ++token)
        {
            if (token > 0)
            {
                oss << " | ";
            }

            oss << "t" << token << "=";
            oss << "[";
            for (size_t dim = 0; dim < dim_limit; ++dim)
            {
                if (dim > 0)
                {
                    oss << ",";
                }
                oss << data[token * embedding_dim + dim];
            }
            if (dim_limit < embedding_dim)
            {
                oss << ",...";
            }
            oss << "]";
        }

        if (token_limit < seq_len)
        {
            oss << " | ...";
        }

        return oss.str();
    }

    /**
     * @brief Format the token ID preview for diagnostic logging.
     *
     * @param token_ids Sequence of token identifiers.
     * @param max_tokens Maximum number of token IDs to include.
     * @return std::string A compact representation of the token sequence.
     */
    std::string formatTokenPreview(const std::vector<int> &token_ids, size_t max_tokens)
    {
        if (token_ids.empty())
        {
            return "[]";
        }

        const size_t limit = std::min(token_ids.size(), max_tokens);
        std::ostringstream oss;
        oss << "[";
        for (size_t idx = 0; idx < limit; ++idx)
        {
            if (idx > 0)
            {
                oss << ", ";
            }
            oss << token_ids[idx];
        }
        if (limit < token_ids.size())
        {
            oss << ", ...";
        }
        oss << "]";
        return oss.str();
    }

    /**
     * @brief Emit structured diagnostics for an embedding buffer when tracing is enabled.
     *
     * @param rank MPI rank emitting the log.
     * @param label Identifier describing the buffer stage.
     * @param data Pointer to embedding data.
     * @param seq_len Sequence length represented in the buffer.
     * @param embedding_dim Embedding dimension.
     * @param token_ids Token identifiers associated with the sequence.
     */
    void logEmbeddingBufferDiagnostics(int rank, const std::string &label, const float *data,
                                       size_t seq_len, size_t embedding_dim,
                                       const std::vector<int> &token_ids)
    {
        const EmbeddingDebugConfig &config = getEmbeddingDebugConfig();
        if (!config.enabled)
        {
            return;
        }

        const size_t element_count = seq_len * embedding_dim;
        const EmbeddingStats stats = computeEmbeddingStats(data, element_count);

        const std::string preview = formatEmbeddingPreview(
            data, seq_len, embedding_dim, config.max_tokens, config.max_dims);
        const std::string token_preview = formatTokenPreview(token_ids, config.max_tokens);

        std::ostringstream oss;
        oss << "MPIEmbeddingKernel[" << label << "] rank=" << rank
            << " seq_len=" << seq_len
            << " emb_dim=" << embedding_dim
            << " finite=" << stats.finite_count
            << " nan=" << stats.nan_count
            << " inf=" << stats.inf_count
            << " zeros=" << stats.zero_count;

        if (stats.finite_count > 0)
        {
            oss << " min=" << stats.min << " max=" << stats.max;
        }
        else
        {
            oss << " min=NaN max=NaN";
        }

        oss << " tokens=" << token_preview
            << " preview=" << preview;

        LOG_TRACE(oss.str());

        if (stats.first_nan_index.has_value() && embedding_dim > 0)
        {
            const size_t index = *stats.first_nan_index;
            const size_t token = index / embedding_dim;
            const size_t dim = index % embedding_dim;
            LOG_TRACE("MPIEmbeddingKernel[" << label << "] rank=" << rank
                                            << " first_nan at token=" << token
                                            << " dim=" << dim);
        }

        if (stats.first_inf_index.has_value() && embedding_dim > 0)
        {
            const size_t index = *stats.first_inf_index;
            const size_t token = index / embedding_dim;
            const size_t dim = index % embedding_dim;
            LOG_TRACE("MPIEmbeddingKernel[" << label << "] rank=" << rank
                                            << " first_inf at token=" << token
                                            << " dim=" << dim);
        }
    }

    /**
     * @brief Compute the maximum absolute difference between two buffers for owned tokens.
     *
     * @param local Pointer to the local embedding buffer.
     * @param global Pointer to the gathered embedding buffer.
     * @param seq_len Sequence length.
     * @param embedding_dim Embedding dimension.
     * @param token_ids Token identifiers associated with the sequence.
     * @param vocab_start First vocabulary index owned by the current rank.
     * @param vocab_end One-past-the-end vocabulary index owned by the current rank.
     * @return float Maximum absolute difference observed for owned tokens.
     */
    float computeOwnedTokenMaxAbsDiff(const float *local, const float *global,
                                      size_t seq_len, size_t embedding_dim,
                                      const std::vector<int> &token_ids,
                                      size_t vocab_start, size_t vocab_end)
    {
        if (local == nullptr || global == nullptr || embedding_dim == 0)
        {
            return 0.0f;
        }

        float max_abs_diff = 0.0f;
        for (size_t token_idx = 0; token_idx < seq_len; ++token_idx)
        {
            const int token_id = token_ids[token_idx];
            if (token_id < static_cast<int>(vocab_start) || token_id >= static_cast<int>(vocab_end))
            {
                continue;
            }

            for (size_t dim = 0; dim < embedding_dim; ++dim)
            {
                const float local_value = local[token_idx * embedding_dim + dim];
                const float global_value = global[token_idx * embedding_dim + dim];
                const float diff = std::fabs(global_value - local_value);
                if (diff > max_abs_diff || std::isnan(diff))
                {
                    max_abs_diff = diff;
                }
            }
        }
        return max_abs_diff;
    }
} // namespace

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

        const EmbeddingDebugConfig &debug_config = getEmbeddingDebugConfig();
        const bool debug_enabled = debug_config.enabled;
        if (debug_enabled)
        {
            LOG_TRACE("MPIEmbeddingKernel execute rank=" << rank_ << " tokens="
                                                         << formatTokenPreview(token_ids, debug_config.max_tokens));
        }

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

        if (debug_enabled)
        {
            logEmbeddingBufferDiagnostics(rank_, "local_pre_gather", local_output->data(),
                                          seq_len, embedding_dim_, token_ids);
        }

        // Gather embeddings (Allreduce for sharded, copy for full table mode)
        gatherEmbeddings(local_output, output, seq_len, embedding_dim_, full_table_mode);

        if (debug_enabled)
        {
            logEmbeddingBufferDiagnostics(rank_, "global_post_gather", output->data(),
                                          seq_len, embedding_dim_, token_ids);

            if (!full_table_mode)
            {
                const float owned_diff = computeOwnedTokenMaxAbsDiff(
                    local_output->data(), output->data(), seq_len, embedding_dim_,
                    token_ids, local_vocab_start_, local_vocab_end_);
                LOG_TRACE("MPIEmbeddingKernel gather diff rank=" << rank_
                                                                 << " owned_token_max_abs_diff=" << owned_diff);
            }
        }

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
        // Fail-fast mode: if set (non-zero) we abort on first non-finite copied value to surface corruption early.
        static const bool kFailFast = debugEnv().embedding.fail_fast;
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

            // Post-copy finite value validation for this token row.
            float *row_ptr = output + i * embedding_dim;
            bool row_ok = true;
            for (size_t d = 0; d < embedding_dim; ++d)
            {
                float v = row_ptr[d];
                if (!std::isfinite(v))
                {
                    row_ok = false;
                    break;
                }
            }
            if (kFailFast && !row_ok)
            {
                // Provide detailed diagnostic context before aborting
                size_t bad_dim = 0; // find first offending dim for clarity
                for (; bad_dim < embedding_dim; ++bad_dim)
                    if (!std::isfinite(row_ptr[bad_dim]))
                        break;
                LOG_ERROR("MPIEmbeddingKernel: non-finite value detected after embedding copy (token=" << token_id
                                                                                                       << ", seq_index=" << i << ", dim=" << bad_dim << ") value=" << row_ptr[bad_dim]
                                                                                                       << " full_table_mode=" << (full_table_mode ? "1" : "0")
                                                                                                       << " transposed=" << (transposed ? "1" : "0")
                                                                                                       << " table_rows=" << table_rows << " table_cols=" << table_cols
                                                                                                       << " local_vocab_range=[" << local_vocab_start_ << "," << local_vocab_end_ << "]");
                // Print a short preview of first few dims to aid debugging
                std::ostringstream oss;
                oss << "row_preview=[";
                size_t preview = std::min<size_t>(embedding_dim, 16);
                for (size_t d = 0; d < preview; ++d)
                {
                    if (d)
                        oss << ",";
                    oss << row_ptr[d];
                }
                if (preview < embedding_dim)
                    oss << ",...";
                oss << "]";
                LOG_ERROR(oss.str());
                std::abort();
            }
            else if (!row_ok)
            {
                LOG_WARN("MPIEmbeddingKernel: non-finite value detected in embedding row (token=" << token_id
                                                                                                  << ", seq_index=" << i << ") proceeding (LLAMINAR_EMBED_FAIL_FAST unset)");
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