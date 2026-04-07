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

#include "IModelLoader.h"
#include "WeightPlacementMap.h"
#include "../backends/DeviceId.h"
#include "../config/TensorParallelConfig.h"
#include "../execution/local_execution/graph/GraphSchema.h"
#include "IWeightManager.h"
#include "../utils/MPIContext.h"
#include "../tensors/Tensors.h"
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <set>

namespace llaminar2
{

    /**
     * @brief Weight distribution strategy
     */
    enum class WeightDistributionStrategy
    {
        REPLICATED,       ///< Full copy per rank (2x memory on 2-socket, no communication)
        SHARDED,          ///< Partition across ranks (1x memory, Allreduce after matmul)
        INTERLEAVED,      ///< NUMA-aware global (shared memory, remote access penalty)
        LAYER_PARTITIONED ///< Load only weights for assigned layers (for Pipeline Parallelism)
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
     *   auto wq = mgr->getWeightForDevice("blk.0.attn_q.weight", device_idx);
     *   auto wk = mgr->getWeightForDevice("blk.0.attn_k.weight", device_idx);
     */
    class WeightManager : public IWeightManager
    {
    public:
        /**
         * @brief Construct weight manager
         *
         * @param loader Model loader interface for tensor loading (GGUF, mock, etc.)
         * @param mpi_ctx MPI context for rank coordination (nullptr = single rank)
         * @param placement_map Fine-grained weight→device mapping (nullptr = default to device 0)
         * @param strategy Distribution strategy (default: REPLICATED)
         * @param weight_precision How weights are loaded (NATIVE, CONVERT_TO_FP32, etc.)
         */
        WeightManager(IModelLoader &loader,
                      std::shared_ptr<MPIContext> mpi_ctx = nullptr,
                      std::shared_ptr<WeightPlacementMap> placement_map = nullptr,
                      WeightDistributionStrategy strategy = WeightDistributionStrategy::REPLICATED,
                      WeightPrecision weight_precision = WeightPrecision::CONVERT_TO_FP32);

        /**
         * @brief Get weight tensor for a specific device (device-isolated instance)
         *
         * Loads from GGUF if not cached, applies distribution strategy.
         * For multi-device scenarios (LOCAL TP), each device needs its own tensor
         * instance to track coherence state independently. This method:
         * - Returns the original tensor for the first device that requests it
         * - Creates and caches a clone for subsequent devices
         * - Clones are uploaded to GPU independently, avoiding race conditions
         *
         * @param name GGUF tensor name (e.g., "token_embd.weight", "blk.0.attn_q.weight")
         * @param device Target device for this tensor instance (default: CPU)
         * @param layer_idx Optional layer index for placement map lookup
         * @return Device-specific tensor instance, or nullptr on error
         */
        std::shared_ptr<TensorBase> getWeightForDevice(const std::string &name, DeviceId device = DeviceId::cpu(), int layer_idx = -1) override;

        /**
         * @brief Pre-load and upload weights for multiple devices
         *
         * For LOCAL tensor parallelism scenarios, this method:
         * 1. Creates device-specific clones of all cached weights
         * 2. Uploads each clone to its target device
         * 3. Caches the device-specific tensors for getWeightForDevice()
         *
         * This MUST be called BEFORE creating device runners.
         *
         * @param devices List of devices that will use the weights
         * @return true if all weights were pre-loaded successfully
         */
        bool preloadForDevices(const std::vector<DeviceId> &devices) override;

        // =========================================================================
        // Weight Packing and Preloading (folded from WeightPreloader)
        // =========================================================================

        /**
         * @brief Pack all GEMM weights for a target device
         *
         * Creates GEMM kernels for all GEMM weights and calls prepareWeights()
         * on each kernel. For GPU kernels, this uploads weights to device memory.
         * For CPU kernels, this is typically a no-op (packing is lazy).
         *
         * @param target_device Target device for weight packing
         * @param progress_cb Optional callback for progress reporting
         * @param release_raw_data If true, release raw tensor data after packing (CPU only)
         * @return true if all GEMM weights were packed successfully
         */
        bool packGemmWeights(
            DeviceId target_device,
            PreloadProgressCallback progress_cb = nullptr,
            bool release_raw_data = false) override;

        /**
         * @brief Upload all non-GEMM weights to GPU
         *
         * Non-GEMM weights (norms, embeddings, biases) don't need GEMM packing
         * but still need to be uploaded to GPU for kernel access.
         * This eliminates lazy upload overhead during inference.
         *
         * @param target_device Target GPU device
         * @return true if all non-GEMM weights were uploaded successfully
         */
        bool uploadNonGemmWeights(DeviceId target_device) override;

        /**
         * @brief Release ALL host-side weight data after all GPU uploads are complete
         * @return Number of tensors whose host data was released
         */
        size_t releaseAllHostWeightData() override;

        /**
         * @brief Get statistics about preloaded weights
         *
         * @return Pair of (num_cpu_packed, num_gpu_packed)
         */
        std::pair<size_t, size_t> preloadStats() const override { return {num_cpu_packed_, num_gpu_packed_}; }

        /**
         * @brief Get current distribution strategy
         */
        WeightDistributionStrategy strategy() const override { return strategy_; }

        /**
         * @brief Get number of cached weights
         */
        size_t cacheSize() const override;

        /**
         * @brief Clear weight cache (frees memory)
         */
        void clearCache() override;

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
         * @brief Set model head dimensions for FusedQKV sub-block computation
         *
         * Required for correct FusedQKVHeads sharding under GQA (n_kv_heads < n_heads).
         * Without this, FusedQKV weights fall back to simple equal row splitting.
         *
         * @param n_heads Number of query attention heads
         * @param n_kv_heads Number of key/value attention heads (GQA)
         * @param head_dim Dimension per attention head
         */
        void setModelDimensions(int n_heads, int n_kv_heads, int head_dim)
        {
            model_n_heads_ = n_heads;
            model_n_kv_heads_ = n_kv_heads;
            model_head_dim_ = head_dim;
            has_model_dimensions_ = true;
        }

        /**
         * @brief Set GDN (Gated Delta Net) dimensions for FusedQKV sub-block slicing
         *
         * GDN layers have asymmetric QKV: [Q(n_k*d) | K(n_k*d) | V(n_v*d)]
         * where n_k_heads != n_v_heads. The standard FA-based slicing doesn't
         * handle this layout. This provides the dimensions needed for correct
         * sub-block aware TP sharding.
         *
         * @param n_k_heads Number of key heads (ssm.group_count)
         * @param n_v_heads Number of value heads (ssm.time_step_rank)
         * @param d_state State dimension per head (ssm.state_size, used as d_k = d_v)
         */
        void setGDNDimensions(int n_k_heads, int n_v_heads, int d_state)
        {
            gdn_n_k_heads_ = n_k_heads;
            gdn_n_v_heads_ = n_v_heads;
            gdn_d_state_ = d_state;
            has_gdn_dimensions_ = true;
        }

        /**
         * @brief Set layer range for LAYER_PARTITIONED strategy
         *
         * For Pipeline Parallelism, restricts which layer weights are loaded.
         * Weights outside this range will return nullptr from getWeightForDevice().
         *
         * Layer range is [first, last) - first is inclusive, last is exclusive.
         *
         * Special weights:
         * - has_embedding: if true, token embedding weight is loaded
         * - has_lm_head: if true, output norm and LM head weights are loaded
         *
         * @param first_layer First layer index (inclusive)
         * @param last_layer Last layer index (exclusive)
         * @param has_embedding True if this stage should load embedding
         * @param has_lm_head True if this stage should load output norm and LM head
         */
        void setLayerRange(int first_layer, int last_layer, bool has_embedding, bool has_lm_head)
        {
            layer_first_ = first_layer;
            layer_last_ = last_layer;
            has_embedding_ = has_embedding;
            has_lm_head_ = has_lm_head;
            has_layer_range_ = true;
            LOG_INFO("[WeightManager] Layer range set: layers [" << first_layer << ", " << last_layer
                                                                 << "), embedding=" << (has_embedding ? "yes" : "no")
                                                                 << ", lm_head=" << (has_lm_head ? "yes" : "no"));
        }

        /**
         * @brief Set layer range without changing global weight flags
         *
         * Use setHasEmbedding() and setHasLmHead() separately if needed.
         *
         * @param first_layer First layer index (inclusive)
         * @param last_layer Last layer index (exclusive, -1 = all remaining)
         */
        void setLayerRange(int first_layer, int last_layer)
        {
            layer_first_ = first_layer;
            layer_last_ = last_layer;
            has_layer_range_ = true;
            LOG_DEBUG("[WeightManager] Layer range set: layers [" << first_layer << ", " << last_layer << ")");
        }

        /**
         * @brief Check if a layer range is configured
         */
        bool hasLayerRange() const { return has_layer_range_; }

        /**
         * @brief Get configured layer range
         * @return Pair of (first_layer, last_layer_exclusive)
         */
        std::pair<int, int> layerRange() const { return {layer_first_, layer_last_}; }

        /**
         * @brief Check if this stage has embedding
         */
        bool hasEmbedding() const override { return has_embedding_; }

        /**
         * @brief Check if this stage has LM head
         */
        bool hasLMHead() const override { return has_lm_head_; }

        /**
         * @brief Set whether this weight manager should provide embedding weights
         * @param has_embedding True if embedding should be loaded
         */
        void setHasEmbedding(bool has_embedding);

        /**
         * @brief Set whether this weight manager should provide lm_head weights
         * @param has_lm_head True if output_norm and lm_head should be loaded
         */
        void setHasLmHead(bool has_lm_head);

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
        size_t decodeCacheSize() const override;

        /**
         * @brief Clear decode weight cache (frees decode shard memory)
         */
        void clearDecodeCache() override;

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

        /**
         * @brief Pack a single weight tensor for a target device
         *
         * Creates a GEMM kernel and calls prepareWeights() on it.
         *
         * @param tensor Weight tensor to pack
         * @param target_device Target device
         * @param release_raw_data If true, release raw tensor data after packing
         * @return true on success
         */
        bool packWeight(TensorBase *tensor, DeviceId target_device, bool release_raw_data);

        IModelLoader &loader_;                                                      ///< Model loader (GGUF, mock, etc.)
        std::shared_ptr<MPIContext> mpi_ctx_;                                       ///< MPI context (nullptr = single rank)
        std::shared_ptr<WeightPlacementMap> placement_map_;                         ///< Fine-grained placement decisions
        std::shared_ptr<TensorParallelConfig> tp_config_;                           ///< Tensor parallelism configuration (optional)
        WeightDistributionStrategy strategy_;                                       ///< Distribution strategy
        WeightPrecision weight_precision_;                                          ///< How weights are loaded (NATIVE, CONVERT_TO_FP32, etc.)
        std::unordered_map<std::string, std::shared_ptr<TensorBase>> cache_;        ///< Weight cache
        mutable std::mutex cache_mutex_;                                            ///< Protects cache_ and decode_cache_ access
        mutable std::unordered_map<std::string, ShardingMode> sharding_mode_cache_; ///< Cached sharding modes
        WeightShardingConfig sharding_config_;                                      ///< Model-specific sharding patterns
        bool has_sharding_config_ = false;                                          ///< True if config was set explicitly

        // =========================================================================
        // Model head dimensions for FusedQKV sub-block computation
        // =========================================================================
        int model_n_heads_ = 0;           ///< Number of query attention heads
        int model_n_kv_heads_ = 0;        ///< Number of KV attention heads (GQA)
        int model_head_dim_ = 0;          ///< Dimension per attention head
        bool has_model_dimensions_ = false; ///< True if setModelDimensions() was called

        // GDN (Gated Delta Net) head dimensions for FusedQKV sharding
        int gdn_n_k_heads_ = 0;           ///< Number of GDN key heads (ssm.group_count)
        int gdn_n_v_heads_ = 0;           ///< Number of GDN value heads (ssm.time_step_rank)
        int gdn_d_state_ = 0;             ///< GDN state dimension per head (ssm.state_size)
        bool has_gdn_dimensions_ = false;  ///< True if setGDNDimensions() was called

        // Decode weight shard cache (separate from prefill cache)
        std::unordered_map<std::string, std::shared_ptr<TensorBase>> decode_cache_; ///< Decode shard cache

        // =========================================================================
        // Layer range for Pipeline Parallelism (LAYER_PARTITIONED strategy)
        // =========================================================================

        int layer_first_ = 0;          ///< First layer index (inclusive)
        int layer_last_ = 0;           ///< Last layer index (exclusive)
        bool has_embedding_ = true;    ///< True if this stage loads embedding
        bool has_lm_head_ = true;      ///< True if this stage loads LM head
        bool has_layer_range_ = false; ///< True if setLayerRange() was called

        /**
         * @brief Check if a weight should be loaded based on layer range
         *
         * When LAYER_PARTITIONED strategy is active, this filters weights:
         * - "token_embd.weight" only if has_embedding_
         * - "output_norm.weight", "output.weight" only if has_lm_head_
         * - "blk.N.*" only if N is in [layer_first_, layer_last_)
         *
         * @param name Weight tensor name
         * @return true if weight should be loaded, false if filtered out
         */
        bool isWeightInLayerRange(const std::string &name) const;

        // =========================================================================
        // Per-device tensor cache for multi-device scenarios (LOCAL TP)
        // =========================================================================

        /// Key: "device_type:ordinal:weight_name" e.g. "cuda:0:token_embd.weight"
        /// Value: Device-specific tensor clone, uploaded to that device
        std::unordered_map<std::string, std::shared_ptr<TensorBase>> per_device_cache_;

        /// First device that requested weights - original tensors are used for this device
        std::optional<DeviceId> first_device_;

        /// Helper to create a clone of a tensor for a different device
        std::shared_ptr<TensorBase> cloneTensorForDevice(
            const std::string &name,
            const std::shared_ptr<TensorBase> &original,
            DeviceId target_device);

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

    public:
        // =========================================================================
        // Device-aware weight slicing for LOCAL TP (Phase 1)
        // =========================================================================

        /**
         * @brief Load weight with device-specific sharding from TensorParallelConfig
         *
         * Uses the DeviceShardingAssignment to determine which slice of the weight
         * this device should receive. Supports proportional slicing for heterogeneous
         * GPU configurations.
         *
         * @param name Weight tensor name
         * @param device Target device
         * @param assignment Device's sharding assignment from TensorParallelConfig
         * @param layer_idx Layer index for logging
         * @return Shared pointer to sliced weight tensor, or nullptr on error
         */
        std::shared_ptr<TensorBase> getShardedWeightForAssignment(
            const std::string &name,
            DeviceId device,
            const DeviceShardingAssignment &assignment,
            int layer_idx);

        // Weight category detection helpers for LOCAL TP slicing
        // DEPRECATED: Use WeightShardingConfig::getDimensionType() instead.
        // These methods hardcode Qwen2 weight name patterns.
        [[deprecated("Use WeightShardingConfig::getDimensionType() instead")]]
        static bool isQKVWeight(const std::string &name);
        [[deprecated("Use WeightShardingConfig::getDimensionType() instead")]]
        static bool isQKVBias(const std::string &name);
        [[deprecated("Use WeightShardingConfig::getDimensionType() instead")]]
        static bool isFFNGateUpWeight(const std::string &name);
        [[deprecated("Use WeightShardingConfig::getDimensionType() instead")]]
        static bool isFFNDownWeight(const std::string &name);
        [[deprecated("Use WeightShardingConfig::getDimensionType() instead")]]
        static bool isLMHeadWeight(const std::string &name);
        [[deprecated("Use WeightShardingConfig::getDimensionType() instead")]]
        static bool isWoWeight(const std::string &name);
        [[deprecated("Use WeightShardingConfig::getDimensionType() instead")]]
        static bool isEmbeddingWeight(const std::string &name);
        [[deprecated("Use WeightShardingConfig::getDimensionType() instead")]]
        static bool isOutputNormWeight(const std::string &name);

        /**
         * @brief Slice a specific row range from tensor
         *
         * Creates a new tensor containing only the specified rows.
         * Supports FP32 tensors (quantized tensors should use GGUF row slice loading).
         *
         * @param tensor Source tensor to slice
         * @param row_start First row index (0-based)
         * @param row_count Number of rows to extract
         * @return New tensor with the specified row range, or nullptr on error
         */
        static std::shared_ptr<TensorBase> sliceRowRange(
            const std::shared_ptr<TensorBase> &tensor,
            size_t row_start,
            size_t row_count);

    private:
        // ========================================================================
        // Sharding helper methods (used by getShardedWeightForAssignment)
        // ========================================================================

        /**
         * @brief Compute slice boundaries based on dimension type from WeightShardingConfig
         *
         * Uses the sharding config to determine which dimension (Heads, KVHeads, FFNHidden, Vocab)
         * should be used for slicing, then computes the appropriate start/count values.
         *
         * @param name Weight tensor name
         * @param total_size Total size of the dimension to slice
         * @param assignment Device's sharding assignment
         * @param out_start Output: start index for slicing
         * @param out_count Output: count of elements to include
         * @return true if boundaries computed successfully, false on error
         */
        bool computeSliceBoundaries(
            const std::string &name,
            size_t total_size,
            const DeviceShardingAssignment &assignment,
            size_t &out_start,
            size_t &out_count) const;

        /**
         * @brief Load a column-parallel 1D bias tensor (e.g., Q/K/V biases)
         *
         * Slices a 1D bias tensor based on head assignment.
         *
         * @param name Tensor name
         * @param device Target device
         * @param assignment Device's sharding assignment
         * @param dimensions Tensor dimensions
         * @return Sliced bias tensor, or nullptr on error
         */
        std::shared_ptr<TensorBase> loadColumnParallel1DBias(
            const std::string &name,
            DeviceId device,
            const DeviceShardingAssignment &assignment,
            const std::vector<size_t> &dimensions);

        /**
         * @brief Load a column-parallel 2D weight tensor (Q/K/V, Gate/Up, LM Head)
         *
         * Slices rows based on the weight type (heads, d_ff, or vocab).
         *
         * @param name Tensor name
         * @param device Target device
         * @param assignment Device's sharding assignment
         * @param dimensions Tensor dimensions
         * @return Sliced weight tensor wrapped in TensorSlice, or nullptr on error
         */
        std::shared_ptr<TensorBase> loadColumnParallel2DWeight(
            const std::string &name,
            DeviceId device,
            const DeviceShardingAssignment &assignment,
            const std::vector<size_t> &dimensions);

        /**
         * @brief Load a fused QKV weight with per-sub-block head slicing
         *
         * Handles weights stored as [Q_all | K_all | V_all] concatenated rows.
         * Splits each sub-block independently by heads, then reassembles as
         * [Q_local | K_local | V_local] so downstream code sees the correct
         * per-head Q/K/V ordering.
         *
         * @param name Tensor name (e.g., "blk.0.attn_qkv.weight")
         * @param device Target device
         * @param assignment Device's sharding assignment
         * @param dimensions Tensor dimensions [total_rows, cols]
         * @return FP32 tensor with correctly ordered local Q/K/V rows
         */
        std::shared_ptr<TensorBase> loadFusedQKVColumnParallel(
            const std::string &name,
            DeviceId device,
            const DeviceShardingAssignment &assignment,
            const std::vector<size_t> &dimensions);

        /**
         * @brief Load a row-parallel weight tensor (unused - Wo uses INPUT_PARALLEL)
         *
         * Slices rows based on head assignment for weights where output is reduced.
         *
         * @param name Tensor name
         * @param device Target device
         * @param assignment Device's sharding assignment
         * @param dimensions Tensor dimensions
         * @return Sliced weight tensor wrapped in TensorSlice, or nullptr on error
         */
        std::shared_ptr<TensorBase> loadRowParallelWeight(
            const std::string &name,
            DeviceId device,
            const DeviceShardingAssignment &assignment,
            const std::vector<size_t> &dimensions);

        /**
         * @brief Load an input-parallel weight tensor (Wo, FFN Down)
         *
         * Slices columns based on weight type:
         * - Wo: slices by head assignment (matches Q/K/V output)
         * - FFN Down: slices by d_ff assignment (matches Gate/Up output)
         *
         * @param name Tensor name
         * @param device Target device
         * @param assignment Device's sharding assignment
         * @param dimensions Tensor dimensions
         * @return Sliced weight tensor wrapped in TensorSlice, or nullptr on error
         */
        std::shared_ptr<TensorBase> loadInputParallelWeight(
            const std::string &name,
            DeviceId device,
            const DeviceShardingAssignment &assignment,
            const std::vector<size_t> &dimensions);

    public:
        /**
         * @brief Slice a specific column range from tensor
         *
         * Creates a new tensor containing only the specified columns.
         * Supports FP32 tensors (quantized tensors should use GGUF column slice loading).
         *
         * This is used for INPUT_PARALLEL weight slicing where the input dimension
         * (columns) is split across devices. Different from row slicing because
         * columns are non-contiguous in row-major memory layout.
         *
         * @param tensor Source tensor to slice
         * @param col_start First column index (0-based)
         * @param col_count Number of columns to extract
         * @return New tensor with the specified column range, or nullptr on error
         */
        static std::shared_ptr<TensorBase> sliceColumnRange(
            const std::shared_ptr<TensorBase> &tensor,
            size_t col_start,
            size_t col_count);

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

        // =========================================================================
        // Preload statistics (folded from WeightPreloader)
        // =========================================================================
        size_t num_cpu_packed_ = 0;
        size_t num_gpu_packed_ = 0;

    private:
        // =========================================================================
        // Phase 2: Per-weight/device readiness tickets and TP-safe reclaim eligibility
        // =========================================================================

        enum class WeightPrepState
        {
            UNKNOWN = 0,
            LOADED_HOST,
            PACKED_HOST,
            UPLOADED_DEVICE,
            READY,
            FAILED
        };

        struct WeightPrepTicket
        {
            WeightPrepState state = WeightPrepState::UNKNOWN;
            bool is_gemm = false;
            std::string detail;
        };

        mutable std::mutex prep_ticket_mutex_;
        std::unordered_map<std::string, std::unordered_map<std::string, WeightPrepTicket>> prep_tickets_;
        std::unordered_map<std::string, std::unordered_set<std::string>> expected_devices_by_weight_;
        std::unordered_set<std::string> reclaim_ready_weights_;
        std::unordered_set<std::string> reclaim_applied_weights_;

        void registerExpectedDeviceForWeight(const std::string &name, DeviceId device);
        void markPrepState(const std::string &name,
                           DeviceId device,
                           WeightPrepState state,
                           bool is_gemm,
                           const std::string &detail = "");
        void evaluateReclaimEligibility(const std::string &name, bool is_gemm);
        bool tryReleaseReclaimHostRawData(const std::string &name);
        static const char *weightPrepStateName(WeightPrepState state);

    public:
        // Friend class for WeightPreloader (deprecated, will be removed)
        // WeightPreloader needs access to cache_ and private members until fully removed
        friend class WeightPreloader;
    };

} // namespace llaminar2
