/**
 * @file WeightManager.h
 * @brief Weight distribution and caching for MPI-aware pipelines
 *
 * Manages model weight tensors with support for different distribution strategies:
 * - REPLICATED: Full copy per rank (default, simple)
 * - SHARDED: Partition across ranks (memory efficient, requires Allreduce)
 * - INTERLEAVED: NUMA-aware global allocation (shared memory optimization)
 *
 * For SHARDED strategy, weights are partitioned based on model-specific
 * sharding configuration defined in the model's schema (e.g., Qwen2Schema).
 * This keeps WeightManager generic and model-agnostic.
 *
 * @see GraphSchema.h for WeightShardingConfig structure
 * @see models/qwen/Qwen2Schema.h for Qwen2-specific sharding patterns
 *
 * @author David Sanftenberg
 */

#pragma once

#include "ModelLoader.h"
#include "WeightPlacementMap.h"
#include "../backends/DeviceId.h"
#include "../config/TensorParallelConfig.h"
#include "../execution/GraphSchema.h"
#include "../interfaces/IWeightManager.h"
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
    class WeightManager : public IWeightManager
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
         * @param device Device for tensor placement (default: CPU)
         * @param layer_idx Optional layer index for placement map lookup
         * @return Shared pointer to tensor, or nullptr on error
         */
        std::shared_ptr<TensorBase> getWeight(const std::string &name, DeviceId device = DeviceId::cpu(), int layer_idx = -1) override;

        /**
         * @brief Get current distribution strategy
         */
        WeightDistributionStrategy strategy() const override { return strategy_; }

        /**
         * @brief Get number of cached weights
         */
        size_t cacheSize() const override { return cache_.size(); }

        /**
         * @brief Clear weight cache (frees memory)
         */
        void clearCache() override { cache_.clear(); }

        /**
         * @brief Set model-specific weight sharding configuration
         *
         * This should be called after construction with the config from
         * the model's schema factory (e.g., Qwen2SchemaFactory).
         *
         * @param config Weight sharding configuration from model schema
         */
        void setWeightShardingConfig(const WeightShardingConfig &config)
        {
            sharding_config_ = config;
            has_sharding_config_ = true;
            sharding_mode_cache_.clear(); // Invalidate cache
        }

        /**
         * @brief Set tensor parallelism configuration for proportional slicing
         *
         * When set, weight slicing uses the assignment from TensorParallelConfig
         * instead of the default 1/world_size calculation. This enables
         * heterogeneous tensor parallelism where devices get proportional work.
         *
         * @param config Tensor parallelism configuration with device assignments
         */
        void setTensorParallelConfig(std::shared_ptr<TensorParallelConfig> config)
        {
            tp_config_ = std::move(config);
            cache_.clear(); // Invalidate cache since slices may change
        }

        /**
         * @brief Get tensor parallelism configuration
         * @return Pointer to config, or nullptr if not set
         */
        const TensorParallelConfig *tensorParallelConfig() const { return tp_config_.get(); }

        /**
         * @brief Check if a weight is sharded (only valid for SHARDED strategy)
         * @param name Weight tensor name
         * @return true if weight is column or row parallel sharded
         */
        bool isWeightSharded(const std::string &name) const override;

        /**
         * @brief Get sharding mode for a weight
         * @param name Weight tensor name
         * @return ShardingMode indicating how weight is partitioned
         */
        ShardingMode getShardingMode(const std::string &name) const override;

        /**
         * @brief Check if a weight is used for GEMM operations
         *
         * Uses the model's sharding config if set, otherwise uses default patterns.
         *
         * @param name Weight tensor name
         * @return true if weight is a GEMM matrix (should release raw data after packing)
         */
        bool isGemmWeight(const std::string &name) const override;

        // =========================================================================
        // Static utility methods (public for tensor slicing)
        // =========================================================================

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

        /**
         * @brief Get the placement map (Phase 6: Multi-GPU support)
         *
         * @return Shared pointer to placement map (may be nullptr)
         */
        std::shared_ptr<WeightPlacementMap> placementMap() const { return placement_map_; }

        /**
         * @brief Get or create decode weight shard for a weight tensor
         *
         * For Option A (Selective Duplication) in CPU decode participation:
         * - Returns a SLICED copy of the weight for decode phase
         * - The shard is the "tail" portion based on fraction
         * - Cached separately from the full prefill weight
         *
         * Slicing behavior by ShardingMode:
         * - COLUMN_PARALLEL (Q, K, V, Gate, Up): slice tail rows (output dimension)
         * - ROW_PARALLEL (Wo): slice tail columns (input dimension)
         * - INPUT_PARALLEL (Down): slice tail columns (input dimension)
         * - REPLICATE (norms): return full copy (no slicing needed)
         *
         * @param name Weight tensor name (e.g., "blk.0.attn_q.weight")
         * @param decode_device Device for the decode shard (typically CPU)
         * @param fraction Fraction of weight for this shard (e.g., 0.20 = tail 20%)
         * @param layer_idx Layer index for sharding mode lookup
         * @return Shared pointer to sliced weight tensor, or nullptr on error
         */
        std::shared_ptr<TensorBase> getDecodeWeight(
            const std::string &name,
            DeviceId decode_device,
            float fraction,
            int layer_idx = -1) override;

        /**
         * @brief Get number of cached decode weight shards
         */
        size_t decodeCacheSize() const override { return decode_cache_.size(); }

        /**
         * @brief Clear decode weight cache (frees decode shard memory)
         */
        void clearDecodeCache() override { decode_cache_.clear(); }

        // =========================================================================
        // Static utility methods for decode shard slicing
        // =========================================================================

        /**
         * @brief Slice tail rows from a tensor (for column-parallel decode shards)
         *
         * For weight matrix [out_dim, in_dim], extracts [out_local, in_dim]
         * where out_local = out_dim * fraction, starting from out_dim - out_local.
         *
         * @param full_tensor Full weight tensor
         * @param fraction Fraction of rows to extract (from tail)
         * @return Sliced tensor containing tail rows
         */
        static std::shared_ptr<TensorBase> sliceTailRows(
            const std::shared_ptr<TensorBase> &full_tensor,
            float fraction);

        /**
         * @brief Slice tail columns from a tensor (for row/input-parallel decode shards)
         *
         * For weight matrix [out_dim, in_dim], extracts [out_dim, in_local]
         * where in_local = in_dim * fraction, starting from in_dim - in_local.
         *
         * @param full_tensor Full weight tensor
         * @param fraction Fraction of columns to extract (from tail)
         * @return Sliced tensor containing tail columns
         */
        static std::shared_ptr<TensorBase> sliceTailColumns(
            const std::shared_ptr<TensorBase> &full_tensor,
            float fraction);

    private:
        // Allow WeightPreloader to access cache_ directly for iteration
        friend class WeightPreloader;
        /**
         * @brief Load weight with replicated strategy
         *
         * Each rank loads full tensor independently.
         */
        std::shared_ptr<TensorBase> getReplicatedWeight(const std::string &name, DeviceId device);

        /**
         * @brief Load weight with sharded strategy
         *
         * Tensor partitioned across ranks based on weight type:
         * - Column-parallel: QKV, Gate/Up projections (split output dim)
         * - Row-parallel: Wo, Down projections (split input dim)
         * - Replicated: Norms, biases, embeddings (full copy)
         */
        std::shared_ptr<TensorBase> getShardedWeight(const std::string &name, DeviceId device);

        /**
         * @brief Load weight with interleaved strategy (not yet implemented)
         *
         * NUMA-aware allocation with page interleaving.
         */
        std::shared_ptr<TensorBase> getInterleavedWeight(const std::string &name, DeviceId device);

        /**
         * @brief Determine sharding mode using config or legacy patterns
         *
         * Uses sharding_config_ if set, otherwise falls back to legacy hardcoded patterns.
         */
        ShardingMode determineShardingMode(const std::string &name) const;

        /**
         * @brief Convert WeightShardingMode (from schema) to ShardingMode (internal)
         */
        static ShardingMode toShardingMode(WeightShardingMode mode);

        ModelLoader &loader_;                                                       ///< GGUF loader
        std::shared_ptr<MPIContext> mpi_ctx_;                                       ///< MPI context (nullptr = single rank)
        std::shared_ptr<WeightPlacementMap> placement_map_;                         ///< Fine-grained placement decisions
        std::shared_ptr<TensorParallelConfig> tp_config_;                           ///< Tensor parallelism configuration (optional)
        WeightDistributionStrategy strategy_;                                       ///< Distribution strategy
        WeightPrecision weight_precision_;                                          ///< How weights are loaded (NATIVE, CONVERT_TO_FP32, etc.)
        std::unordered_map<std::string, std::shared_ptr<TensorBase>> cache_;        ///< Weight cache
        mutable std::unordered_map<std::string, ShardingMode> sharding_mode_cache_; ///< Cached sharding modes
        WeightShardingConfig sharding_config_;                                      ///< Model-specific sharding patterns
        bool has_sharding_config_ = false;                                          ///< True if config was set explicitly

        // Decode weight shard cache (separate from prefill cache)
        std::unordered_map<std::string, std::shared_ptr<TensorBase>> decode_cache_; ///< Decode shard cache

        // =========================================================================
        // Proportional slicing helpers (used when tp_config_ is set)
        // =========================================================================

        /**
         * @brief Calculate proportional column slice for this rank
         *
         * Uses TensorParallelConfig to determine slice bounds instead of 1/world_size.
         * Returns {start_row, num_rows} for column-parallel weights.
         *
         * @param name Weight tensor name (to determine weight type)
         * @param total_rows Total number of rows in the weight
         * @return {start_row, num_rows} pair for this rank's slice
         */
        std::pair<size_t, size_t> calculateProportionalColumnSlice(
            const std::string &name, size_t total_rows) const;

        /**
         * @brief Calculate proportional row slice for this rank
         *
         * Uses TensorParallelConfig to determine slice bounds instead of 1/world_size.
         * Returns {start_col, num_cols} for row-parallel/input-parallel weights.
         *
         * @param name Weight tensor name (to determine weight type)
         * @param total_cols Total number of columns in the weight
         * @return {start_col, num_cols} pair for this rank's slice
         */
        std::pair<size_t, size_t> calculateProportionalRowSlice(
            const std::string &name, size_t total_cols) const;

        /**
         * @brief Determine weight category from name
         *
         * Categories:
         * - ATTENTION_QKV: Q, K, V projections (column-parallel by heads)
         * - ATTENTION_WO: Output projection (row-parallel, matches Q output)
         * - FFN_GATE_UP: Gate and Up projections (column-parallel by d_ff)
         * - FFN_DOWN: Down projection (input-parallel, matches gate/up output)
         * - LM_HEAD: Language model head (column-parallel by vocab)
         * - REPLICATE: Everything else (norms, biases, embeddings)
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

        WeightCategory categorizeWeight(const std::string &name) const;
    };

} // namespace llaminar2
