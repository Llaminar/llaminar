/**
 * @file IWeightSlicer.h
 * @brief Interface for stateless weight slicing computations
 *
 * Encapsulates the math for determining how to slice weight tensors
 * across ranks/devices for tensor parallelism. Pure computation —
 * no I/O, no caching, no state mutation.
 *
 * This enables:
 * 1. Unit testing slicing math with zero I/O dependencies
 * 2. Mocking slicing for higher-level tests
 * 3. Separation of "what to slice" from "how to load/cache"
 *
 * @author David Sanftenberg
 */

#pragma once

#include "WeightManagerConfig.h"
#include "../config/TensorParallelConfig.h"
#include "../execution/local_execution/graph/GraphSchema.h"
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace llaminar2
{

    /**
     * @brief Specification for a contiguous slice within a dimension
     */
    struct SliceSpec
    {
        size_t start = 0; ///< Start index (0-based)
        size_t count = 0; ///< Number of elements in the slice

        size_t end() const { return start + count; }
        bool empty() const { return count == 0; }
        constexpr bool operator==(const SliceSpec &) const noexcept = default;
    };

    /**
     * @brief Slice specifications for fused QKV sub-blocks
     *
     * When a weight is [Q | K | V] concatenated vertically,
     * each sub-block may need independent slicing (GQA, GDN asymmetric).
     */
    struct FusedQKVSliceResult
    {
        SliceSpec q;                    ///< Q sub-block slice.
        SliceSpec k;                    ///< K sub-block slice.
        std::vector<SliceSpec> v;       ///< V slices in participant-local packed order.
        size_t q_total = 0;             ///< Total Q rows in original weight.
        size_t k_total = 0;             ///< Total K rows in original weight.
        size_t v_total = 0;             ///< Total V rows in original weight.
        bool modulo_linked_gdn = false; ///< Whether V spans use modulo-linked GDN ownership.
    };

    /**
     * @brief Interface for stateless weight slicing computations
     *
     * Pure computation: given weight name, total size, rank, and world_size,
     * compute the slice boundaries. No side effects, no I/O.
     */
    class IWeightSlicer
    {
    public:
        virtual ~IWeightSlicer() = default;

        /**
         * @brief Compute column-parallel slice (split output dimension / rows)
         *
         * For weight [N, K], splits N across ranks. Each rank gets [N/ws, K].
         *
         * @param name Weight tensor name (determines if head-based, FFN-based, or vocab-based)
         * @param total_rows Total output dimension (rows)
         * @param rank MPI rank
         * @param world_size Total MPI ranks
         * @return SliceSpec with start row and row count for this rank
         */
        virtual SliceSpec computeColumnSlice(
            const std::string &name, size_t total_rows,
            int rank, int world_size) const = 0;

        /**
         * @brief Compute row-parallel / input-parallel slice (split input dimension / columns)
         *
         * For weight [N, K], splits K across ranks. Each rank gets [N, K/ws].
         *
         * @param name Weight tensor name
         * @param total_cols Total input dimension (columns)
         * @param rank MPI rank
         * @param world_size Total MPI ranks
         * @return SliceSpec with start column and column count for this rank
         */
        virtual SliceSpec computeRowSlice(
            const std::string &name, size_t total_cols,
            int rank, int world_size) const = 0;

        /**
         * @brief Compute fused QKV sub-block aware slicing
         *
         * When a weight is tagged FusedQKVHeads, it has [Q | K | V] vertically
         * concatenated. Each sub-block must be sliced independently to preserve
         * the Q/K/V ordering after sharding.
         *
         * Returns std::nullopt if the weight is not a fused QKV weight or
         * if simple equal row splitting should be used instead.
         *
         * @param name Weight tensor name
         * @param total_rows Total rows in the fused weight
         * @param rank MPI rank
         * @param world_size Total MPI ranks
         * @return FusedQKVSliceResult if sub-block slicing applies, nullopt otherwise
         */
        virtual std::optional<FusedQKVSliceResult> computeFusedQKVSlice(
            const std::string &name, size_t total_rows,
            int rank, int world_size) const = 0;

        /**
         * @brief Compute slice for a specific device assignment (LOCAL TP)
         *
         * Uses DeviceShardingAssignment for proportional (heterogeneous) slicing.
         * The dimension type is determined from the WeightShardingConfig.
         *
         * @param name Weight tensor name
         * @param total_size Total size of the dimension being sliced
         * @param assignment Device's sharding assignment from TensorParallelConfig
         * @return SliceSpec for this device, or empty spec on error
         */
        virtual SliceSpec computeSliceForAssignment(
            const std::string &name, size_t total_size,
            const DeviceShardingAssignment &assignment) const = 0;

        /**
         * @brief Compute fused QKV sub-block aware slicing for a device assignment
         *
         * Like computeFusedQKVSlice but uses DeviceShardingAssignment instead of rank/ws.
         *
         * @param name Weight tensor name
         * @param total_rows Total rows in the fused weight
         * @param assignment Device's sharding assignment
         * @return FusedQKVSliceResult if sub-block slicing applies, nullopt otherwise
         */
        virtual std::optional<FusedQKVSliceResult> computeFusedQKVSliceForAssignment(
            const std::string &name, size_t total_rows,
            const DeviceShardingAssignment &assignment) const = 0;
    };

} // namespace llaminar2
