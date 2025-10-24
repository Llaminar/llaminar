/**
 * @file WeightManager.h
 * @brief Weight distribution and caching for MPI-aware pipelines
 *
 * Manages model weight tensors with support for different distribution strategies:
 * - REPLICATED: Full copy per rank (default, simple)
 * - SHARDED: Partition across ranks (memory efficient, requires Allreduce)
 * - INTERLEAVED: NUMA-aware global allocation (shared memory optimization)
 *
 * Phase 1 (current): Replicated strategy only with caching
 * Phase 2 (future): Sharded strategy for large models
 * Phase 3 (future): Interleaved strategy for NUMA systems
 *
 * @author David Sanftenberg
 */

#pragma once

#include "ModelLoader.h"
#include "../utils/MPIContext.h"
#include "../tensors/Tensors.h"
#include <memory>
#include <string>
#include <unordered_map>

namespace llaminar2
{

    /**
     * @brief Weight distribution strategy
     */
    enum class WeightDistributionStrategy
    {
        REPLICATED,  ///< Full copy per rank (2x memory on 2-socket, no communication)
        SHARDED,     ///< Partition across ranks (1x memory, Allreduce after matmul)
        INTERLEAVED  ///< NUMA-aware global (shared memory, remote access penalty)
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
         * @param strategy Distribution strategy (default: REPLICATED)
         */
        WeightManager(ModelLoader &loader,
                      std::shared_ptr<MPIContext> mpi_ctx = nullptr,
                      WeightDistributionStrategy strategy = WeightDistributionStrategy::REPLICATED);

        /**
         * @brief Get weight tensor by name
         *
         * Loads from GGUF if not cached, applies distribution strategy.
         *
         * @param name GGUF tensor name (e.g., "token_embd.weight", "blk.0.attn_q.weight")
         * @param device_idx Device to place tensor on (-1 = CPU, ≥0 = GPU)
         * @return Shared pointer to tensor, or nullptr on error
         */
        std::shared_ptr<TensorBase> getWeight(const std::string &name, int device_idx = -1);

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

    private:
        /**
         * @brief Load weight with replicated strategy
         *
         * Each rank loads full tensor independently.
         */
        std::shared_ptr<TensorBase> getReplicatedWeight(const std::string &name, int device_idx);

        /**
         * @brief Load weight with sharded strategy (Phase 2 - not yet implemented)
         *
         * Tensor partitioned across ranks (column-wise for linear layers).
         * Requires MPI coordination.
         */
        std::shared_ptr<TensorBase> getShardedWeight(const std::string &name, int device_idx);

        /**
         * @brief Load weight with interleaved strategy (Phase 3 - not yet implemented)
         *
         * NUMA-aware allocation with page interleaving.
         */
        std::shared_ptr<TensorBase> getInterleavedWeight(const std::string &name, int device_idx);

        ModelLoader &loader_;                                               ///< GGUF loader
        std::shared_ptr<MPIContext> mpi_ctx_;                               ///< MPI context (nullptr = single rank)
        WeightDistributionStrategy strategy_;                               ///< Distribution strategy
        std::unordered_map<std::string, std::shared_ptr<TensorBase>> cache_; ///< Weight cache
    };

} // namespace llaminar2
