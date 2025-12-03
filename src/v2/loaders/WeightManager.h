/**
 * @file WeightManager.h
 * @brief Weight distribution and caching for MPI-aware pipelines
 *
 * Manages model weight tensors with support for different distribution strategies:
 * - REPLICATED: Full copy per rank (default, simple)
 * - SHARDED: Partition across ranks (memory efficient, requires Allreduce)
 * - INTERLEAVED: NUMA-aware global allocation (shared memory optimization)
 *
 * For SHARDED strategy, weights are partitioned as follows:
 * - Column-parallel weights (QKV, Gate/Up): Split output dimension
 * - Row-parallel weights (Wo, Down): Split input dimension
 * - Non-parallel weights (norms, embeddings): Replicated
 *
 * @author David Sanftenberg
 */

#pragma once

#include "ModelLoader.h"
#include "WeightPlacementMap.h"
#include "../utils/MPIContext.h"
#include "../tensors/Tensors.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <set>

namespace llaminar2
{

    /**
     * @brief Weight distribution strategy
     */
    enum class WeightDistributionStrategy
    {
        REPLICATED, ///< Full copy per rank (2x memory on 2-socket, no communication)
        SHARDED,    ///< Partition across ranks (1x memory, Allreduce after matmul)
        INTERLEAVED ///< NUMA-aware global (shared memory, remote access penalty)
    };

    /**
     * @brief Sharding mode for a weight tensor
     */
    enum class ShardingMode
    {
        REPLICATE,       ///< Not sharded, full copy on each rank (norms, biases, embeddings)
        COLUMN_PARALLEL, ///< Split output dimension (rows of weight) - for Gate/Up, QKV
        ROW_PARALLEL,    ///< Split output dimension (rows of weight) + allreduce - for Wo
        INPUT_PARALLEL   ///< Split input dimension (columns of weight) + allreduce - for Down
    };

    /**
     * @brief Weight manager with distribution strategy and caching
     *
     * Sits between ModelContext and pipelines:
     * - Loads weights from ModelLoader
     * - Applies distribution strategy (replicated/sharded/interleaved)
     * - Caches loaded tensors for reuse
     * - Coordinates across MPI ranks
     *
     * Usage:
     *   auto mgr = std::make_shared<WeightManager>(loader, mpi_ctx);
     *   auto wq = mgr->getWeight("blk.0.attn_q.weight", device_idx);
     *   auto wk = mgr->getWeight("blk.0.attn_k.weight", device_idx);
     */
    class WeightManager
    {
    public:
        /**
         * @brief Construct weight manager
         *
         * @param loader ModelLoader for GGUF tensor loading
         * @param mpi_ctx MPI context for rank coordination (nullptr = single rank)
         * @param placement_map Fine-grained weight→device mapping (nullptr = default to device 0)
         * @param strategy Distribution strategy (default: REPLICATED)
         * @param weight_precision How weights are loaded (NATIVE, CONVERT_TO_FP32, etc.)
         */
        WeightManager(ModelLoader &loader,
                      std::shared_ptr<MPIContext> mpi_ctx = nullptr,
                      std::shared_ptr<WeightPlacementMap> placement_map = nullptr,
                      WeightDistributionStrategy strategy = WeightDistributionStrategy::REPLICATED,
                      WeightPrecision weight_precision = WeightPrecision::CONVERT_TO_FP32);

        /**
         * @brief Get weight tensor by name
         *
         * Loads from GGUF if not cached, applies distribution strategy.
         * Device placement is determined by placement_map if provided.
         *
         * @param name GGUF tensor name (e.g., "token_embd.weight", "blk.0.attn_q.weight")
         * @param device_idx Device override (if -1, use placement_map or default)
         * @param layer_idx Optional layer index for placement map lookup
         * @return Shared pointer to tensor, or nullptr on error
         */
        std::shared_ptr<TensorBase> getWeight(const std::string &name, int device_idx = -1, int layer_idx = -1);

        /**
         * @brief Get current distribution strategy
         */
        WeightDistributionStrategy strategy() const { return strategy_; }

        /**
         * @brief Get number of cached weights
         */
        size_t cacheSize() const { return cache_.size(); }

        /**
         * @brief Clear weight cache (frees memory)
         */
        void clearCache() { cache_.clear(); }

        /**
         * @brief Check if a weight is sharded (only valid for SHARDED strategy)
         * @param name Weight tensor name
         * @return true if weight is column or row parallel sharded
         */
        bool isWeightSharded(const std::string &name) const;

        /**
         * @brief Get sharding mode for a weight
         * @param name Weight tensor name
         * @return ShardingMode indicating how weight is partitioned
         */
        ShardingMode getShardingMode(const std::string &name) const;

        /**
         * @brief Check if a weight is used for GEMM operations
         *
         * GEMM weights (2D matrices for matmul) should have their raw data
         * released after GEMM packing to save memory. Non-GEMM weights like
         * norms, biases, and embeddings need their raw data retained.
         *
         * @param name Weight tensor name
         * @return true if weight is a GEMM matrix (should release raw data after packing)
         */
        static bool isGemmWeight(const std::string &name);

        // =========================================================================
        // Static utility methods (public for testing and reuse)
        // =========================================================================

        /**
         * @brief Determine sharding mode from weight name (static for testability)
         *
         * Pattern matching (Phase 1 - row-parallel only):
         * - Row-parallel: attn_output, ffn_down
         * - Replicate: everything else (QKV, Gate/Up, norms, biases, embeddings)
         *
         * @param name Weight tensor name
         * @return ShardingMode indicating partition strategy
         */
        static ShardingMode determineShardingMode(const std::string &name);

        /**
         * @brief Slice tensor columns for column-parallel sharding
         *
         * For weight matrix [out_dim, in_dim], extracts [out_local, in_dim]
         * where out_local = out_dim / world_size for this rank.
         *
         * @param full_tensor Full weight tensor
         * @param rank MPI rank
         * @param world_size Total MPI ranks
         * @return Sliced tensor containing only this rank's columns
         */
        static std::shared_ptr<TensorBase> sliceColumns(
            const std::shared_ptr<TensorBase> &full_tensor,
            int rank, int world_size);

        /**
         * @brief Slice tensor rows for row-parallel sharding
         *
         * For weight matrix [out_dim, in_dim], extracts [out_dim, in_local]
         * where in_local = in_dim / world_size for this rank.
         *
         * @param full_tensor Full weight tensor
         * @param rank MPI rank
         * @param world_size Total MPI ranks
         * @return Sliced tensor containing only this rank's rows
         */
        static std::shared_ptr<TensorBase> sliceRows(
            const std::shared_ptr<TensorBase> &full_tensor,
            int rank, int world_size);

    private:
        /**
         * @brief Load weight with replicated strategy
         *
         * Each rank loads full tensor independently.
         */
        std::shared_ptr<TensorBase> getReplicatedWeight(const std::string &name, int device_idx);

        /**
         * @brief Load weight with sharded strategy
         *
         * Tensor partitioned across ranks based on weight type:
         * - Column-parallel: QKV, Gate/Up projections (split output dim)
         * - Row-parallel: Wo, Down projections (split input dim)
         * - Replicated: Norms, biases, embeddings (full copy)
         */
        std::shared_ptr<TensorBase> getShardedWeight(const std::string &name, int device_idx);

        /**
         * @brief Load weight with interleaved strategy (not yet implemented)
         *
         * NUMA-aware allocation with page interleaving.
         */
        std::shared_ptr<TensorBase> getInterleavedWeight(const std::string &name, int device_idx);

        ModelLoader &loader_;                                                ///< GGUF loader
        std::shared_ptr<MPIContext> mpi_ctx_;                                ///< MPI context (nullptr = single rank)
        std::shared_ptr<WeightPlacementMap> placement_map_;                  ///< Fine-grained placement decisions
        WeightDistributionStrategy strategy_;                                ///< Distribution strategy
        WeightPrecision weight_precision_;                                   ///< How weights are loaded (NATIVE, CONVERT_TO_FP32, etc.)
        std::unordered_map<std::string, std::shared_ptr<TensorBase>> cache_; ///< Weight cache
        std::unordered_map<std::string, ShardingMode> sharding_mode_cache_;  ///< Cached sharding modes
    };

} // namespace llaminar2
