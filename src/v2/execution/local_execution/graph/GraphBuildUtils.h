/**
 * @file GraphBuildUtils.h
 * @brief Shared utilities for declarative graph builders
 * @author David Sanftenberg
 * @date December 2025
 *
 * This header provides common helper functions used across different model
 * graph builders (QwenStandardGraph, Qwen3Graph, DeepseekGraph, etc.).
 *
 * Design Philosophy:
 * - Graph builders should be purely declarative
 * - Common patterns should be shared, not duplicated
 * - Utilities should be stateless functions
 */

#pragma once

#include "../../../tensors/TensorSlice.h"
#include "../../../utils/Logger.h"
#include <vector>

namespace llaminar2
{
    namespace graph_utils
    {

        /**
         * @brief Check if a weight tensor is sharded row-parallel
         *
         * Row-parallel tensors require AllReduce after GEMM to sum partial results.
         *
         * @param weight The tensor to check
         * @return true if tensor is a TensorSlice with row-parallel sharding
         */
        inline bool isRowParallelSharded(const TensorBase *weight)
        {
            if (!weight)
                return false;

            const auto *slice = dynamic_cast<const TensorSlice *>(weight);
            bool result = slice && slice->is_row_parallel();
            LOG_DEBUG("[isRowParallelSharded] weight=" << weight << " is_slice=" << (slice != nullptr)
                                                       << " name=" << (weight->debugName().empty() ? "(unnamed)" : weight->debugName())
                                                       << " shape0=" << (weight->shape().empty() ? 0 : weight->shape()[0])
                                                       << " shape1=" << (weight->shape().size() > 1 ? weight->shape()[1] : 1)
                                                       << " is_row_parallel=" << (slice ? slice->is_row_parallel() : false)
                                                       << " result=" << result);
            return result;
        }

        /**
         * @brief Build position IDs for a batch of sequences
         *
         * Generates position IDs [offset, offset+1, ..., offset+seq_len-1] for each
         * sequence in the batch.
         *
         * @param seq_len Sequence length (tokens per sequence)
         * @param batch_size Number of sequences in batch
         * @param offset Starting position (for decode continuation)
         * @return Vector of position IDs [batch_size * seq_len]
         */
        inline std::vector<int> buildPositionIds(int seq_len, int batch_size, int offset = 0)
        {
            std::vector<int> pos_ids(batch_size * seq_len);

            for (int b = 0; b < batch_size; ++b)
            {
                for (int s = 0; s < seq_len; ++s)
                {
                    pos_ids[b * seq_len + s] = offset + s;
                }
            }

            return pos_ids;
        }

        /**
         * @brief Generate layer-prefixed node name
         *
         * Creates consistent node names like "layer0_attn_norm", "layer5_ffn_residual".
         *
         * @param layer_idx Layer index
         * @param suffix Node suffix (e.g., "attn_norm", "qkv_proj")
         * @return Formatted node name
         */
        inline std::string layerNodeName(int layer_idx, const char *suffix)
        {
            return "layer" + std::to_string(layer_idx) + "_" + suffix;
        }

        /**
         * @brief Check if MPI multi-rank execution is active
         *
         * @param mpi_ctx The MPI context (may be null)
         * @return true if mpi_ctx is valid and world_size > 1
         */
        inline bool hasMultipleRanks(const IMPIContext *mpi_ctx)
        {
            return mpi_ctx && mpi_ctx->world_size() > 1;
        }

        /**
         * @brief Check if a tensor needs AllReduce after GEMM
         *
         * Convenience function combining sharding check and multi-rank check.
         *
         * @param weight Weight tensor to check
         * @param mpi_ctx MPI context for rank count
         * @return true if weight is row-sharded and multiple ranks exist
         */
        inline bool needsAllReduceAfterGemm(const TensorBase *weight, const IMPIContext *mpi_ctx)
        {
            return isRowParallelSharded(weight) && hasMultipleRanks(mpi_ctx);
        }

    } // namespace graph_utils
} // namespace llaminar2
