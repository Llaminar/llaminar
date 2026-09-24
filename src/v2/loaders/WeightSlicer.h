/**
 * @file WeightSlicer.h
 * @brief Stateless weight slicing computations
 *
 * Pure computation component — given model dimensions, sharding config,
 * and rank/world_size, computes slice boundaries for weight tensors.
 * No I/O, no caching, no mutable state.
 *
 * @author David Sanftenberg
 */

#pragma once

#include "IWeightSlicer.h"
#include "WeightManagerConfig.h"
#include "../config/TensorParallelConfig.h"
#include "../execution/local_execution/graph/GraphSchema.h"
#include <memory>
#include <optional>
#include <string>

namespace llaminar2
{

    /**
     * @brief Stateless implementation of weight slicing computations
     *
     * All methods are const and depend only on the ModelDimensions and
     * WeightShardingConfig provided at construction. Thread-safe.
     *
     * Usage:
     * @code
     * ModelDimensions dims{.n_heads=32, .n_kv_heads=8, .head_dim=128};
     * auto config = SchemaFactory::getWeightShardingConfig("qwen2");
     * WeightSlicer slicer(dims, config);
     *
     * auto spec = slicer.computeColumnSlice("blk.0.attn_q.weight", 4096, 0, 2);
     * // spec.start = 0, spec.count = 2048
     * @endcode
     */
    class WeightSlicer : public IWeightSlicer
    {
    public:
        /**
         * @brief Construct a weight slicer
         *
         * @param dimensions Model attention dimensions (heads, head_dim, GDN)
         * @param sharding_config Weight sharding patterns from model schema
         * @param tp_config Optional tensor parallel config for proportional slicing
         */
        WeightSlicer(const ModelDimensions &dimensions,
                     const WeightShardingConfig &sharding_config,
                     std::shared_ptr<TensorParallelConfig> tp_config = nullptr);

        // =========================================================================
        // IWeightSlicer implementation
        // =========================================================================

        SliceSpec computeColumnSlice(
            const std::string &name, size_t total_rows,
            int rank, int world_size) const override;

        SliceSpec computeRowSlice(
            const std::string &name, size_t total_cols,
            int rank, int world_size) const override;

        std::optional<FusedQKVSliceResult> computeFusedQKVSlice(
            const std::string &name, size_t total_rows,
            int rank, int world_size) const override;

        std::optional<FusedQKVSliceResult> computeFusedQKVSliceForAssignment(
            const std::string &name, size_t total_rows,
            const DeviceShardingAssignment &assignment) const override;

        SliceSpec computeSliceForAssignment(
            const std::string &name, size_t total_size,
            const DeviceShardingAssignment &assignment) const override;

        // =========================================================================
        // Weight categorization (public for testing)
        // =========================================================================

        /**
         * @brief Weight category for proportional slicing
         */
        enum class WeightCategory
        {
            ATTENTION_QKV,
            ATTENTION_WO,
            FFN_GATE_UP,
            FFN_DOWN,
            LM_HEAD,
            REPLICATE
        };

        /**
         * @brief Categorize a weight by name for proportional slice dispatch
         * @param name Weight tensor name
         * @return Weight category
         */
        WeightCategory categorizeWeight(const std::string &name) const;

    private:
        // =========================================================================
        // FusedQKV sub-block size computation
        // =========================================================================

        /**
         * @brief Determine FusedQKV sub-block sizes from model dimensions
         *
         * Tries, in order:
         * 1. GQA layout: [Q(n_heads*hd) | K(n_kv*hd) | V(n_kv*hd)]
         * 2. 3 equal blocks if total_rows % 3 == 0
         * 3. GDN layout: [Q(n_k*d) | K(n_k*d) | V(n_v*d)]
         *
         * @param total_rows Total rows in the fused weight
         * @param[out] q_rows Q sub-block row count
         * @param[out] k_rows K sub-block row count
         * @param[out] v_rows V sub-block row count
         * @param[out] modulo_linked_gdn Whether the exact GDN geometry uses
         *        modulo-linked Q/K/V ownership.
         * @return true if sub-block sizes were determined
         */
        bool determineFusedQKVSubBlockSizes(
            size_t total_rows,
            size_t &q_rows, size_t &k_rows, size_t &v_rows,
            bool &modulo_linked_gdn) const;

        /**
         * @brief Compute slice within a single sub-block
         *
         * @param block_rows Total rows in this sub-block
         * @param rank MPI rank
         * @param world_size Total MPI ranks
         * @return SliceSpec within the sub-block
         */
        static SliceSpec computeSubBlockSlice(
            size_t block_rows, int rank, int world_size);

        // =========================================================================
        // Proportional slicing helpers
        // =========================================================================

        /**
         * @brief Compute proportional column slice using TensorParallelConfig
         *
         * Uses the assignment's head/FFN/vocab counts for heterogeneous slicing.
         *
         * @param name Weight name (for category dispatch)
         * @param total_rows Total output dimension
         * @param rank Rank to compute for
         * @return SliceSpec for this rank
         */
        SliceSpec computeProportionalColumnSlice(
            const std::string &name, size_t total_rows, int rank) const;

        /**
         * @brief Compute proportional row slice using TensorParallelConfig
         *
         * @param name Weight name (for category dispatch)
         * @param total_cols Total input dimension
         * @param rank Rank to compute for
         * @return SliceSpec for this rank
         */
        SliceSpec computeProportionalRowSlice(
            const std::string &name, size_t total_cols, int rank) const;

        // Immutable configuration (set once at construction)
        ModelDimensions dimensions_;
        WeightShardingConfig sharding_config_;
        std::shared_ptr<TensorParallelConfig> tp_config_;
    };

} // namespace llaminar2
