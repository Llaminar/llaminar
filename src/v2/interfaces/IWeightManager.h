/**
 * @file IWeightManager.h
 * @brief Interface for managing model weights with distribution strategies
 *
 * Abstracts weight management to enable:
 * 1. Testing weight sharding without MPI
 * 2. Mock weights for stage unit tests
 * 3. Testing different distribution strategies
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "../backends/DeviceId.h"
#include <memory>
#include <string>
#include <cstddef>

namespace llaminar2 {

// Forward declaration for sharding config
struct WeightShardingConfig;

// Forward declarations
class TensorBase;
enum class ShardingMode;
enum class WeightDistributionStrategy;

/**
 * @brief Interface for managing model weights with distribution strategies
 *
 * Abstracts weight management to enable:
 * 1. Testing weight sharding without MPI
 * 2. Mock weights for stage unit tests
 * 3. Testing different distribution strategies
 * 4. Testing allreduce stages with simulated sharded weights
 *
 * Implementations:
 * - WeightManager: Real implementation with GGUF loader integration
 * - MockWeightManager: Test implementation with in-memory weights
 *
 * Usage:
 * @code
 * // Production code works with interface
 * void loadProjection(IWeightManager& weights) {
 *     auto wq = weights.getWeight("blk.0.attn_q.weight");
 *     if (weights.isWeightSharded("blk.0.attn_q.weight")) {
 *         // Need allreduce after matmul
 *     }
 * }
 *
 * // Test code uses mock
 * auto mock = MockWeightManagerBuilder()
 *     .addWeight("blk.0.attn_q.weight", create_fp32_tensor({896, 896}))
 *     .setSharded("blk.0.attn_q.weight", ShardingMode::COLUMN_PARALLEL)
 *     .build();
 * loadProjection(*mock);
 * @endcode
 */
class IWeightManager {
public:
    virtual ~IWeightManager() = default;

    // =========================================================================
    // Weight Access
    // =========================================================================

    /**
     * @brief Get weight tensor by name
     *
     * Loads from storage if not cached, applies distribution strategy.
     *
     * @param name GGUF tensor name (e.g., "token_embd.weight", "blk.0.attn_q.weight")
     * @param device Device for tensor placement (default: CPU)
     * @param layer_idx Optional layer index for placement map lookup
     * @return Shared pointer to tensor, or nullptr on error
     */
    virtual std::shared_ptr<TensorBase> getWeight(
        const std::string& name,
        DeviceId device = DeviceId::cpu(),
        int layer_idx = -1) = 0;

    // =========================================================================
    // Decode Weight Access (Option A: Selective Duplication)
    // =========================================================================

    /**
     * @brief Get or create decode weight shard for a weight tensor
     *
     * For Option A (Selective Duplication) in CPU decode participation:
     * - Returns a SLICED copy of the weight for decode phase
     * - The shard is the "tail" portion based on fraction
     * - Cached separately from the full prefill weight
     *
     * @param name Weight tensor name (e.g., "blk.0.attn_q.weight")
     * @param decode_device Device for the decode shard (typically CPU)
     * @param fraction Fraction of weight for this shard (e.g., 0.20 = tail 20%)
     * @param layer_idx Layer index for sharding mode lookup
     * @return Shared pointer to sliced weight tensor, or nullptr on error
     */
    virtual std::shared_ptr<TensorBase> getDecodeWeight(
        const std::string& name,
        DeviceId decode_device,
        float fraction,
        int layer_idx = -1) = 0;

    // =========================================================================
    // Sharding Info
    // =========================================================================

    /**
     * @brief Check if a weight is sharded (only valid for SHARDED strategy)
     * @param name Weight tensor name
     * @return true if weight is column, row, or input parallel sharded
     */
    virtual bool isWeightSharded(const std::string& name) const = 0;

    /**
     * @brief Get sharding mode for a weight
     * @param name Weight tensor name
     * @return ShardingMode indicating how weight is partitioned
     */
    virtual ShardingMode getShardingMode(const std::string& name) const = 0;

    /**
     * @brief Check if a weight is used for GEMM operations
     *
     * Non-GEMM weights (norms, biases, embeddings) need raw data retained.
     * GEMM weights can have raw data released after packing.
     *
     * @param name Weight tensor name
     * @return true if weight is a GEMM matrix
     */
    virtual bool isGemmWeight(const std::string& name) const = 0;

    // =========================================================================
    // Strategy
    // =========================================================================

    /**
     * @brief Get current distribution strategy
     * @return Current weight distribution strategy
     */
    virtual WeightDistributionStrategy strategy() const = 0;

    // =========================================================================
    // Cache Management
    // =========================================================================

    /**
     * @brief Get number of cached weights
     * @return Number of weights in the primary cache
     */
    virtual size_t cacheSize() const = 0;

    /**
     * @brief Clear weight cache (frees memory)
     */
    virtual void clearCache() = 0;

    /**
     * @brief Get number of cached decode weight shards
     * @return Number of weights in the decode cache
     */
    virtual size_t decodeCacheSize() const = 0;

    /**
     * @brief Clear decode weight cache (frees decode shard memory)
     */
    virtual void clearDecodeCache() = 0;

    // =========================================================================
    // Configuration
    // =========================================================================

    /**
     * @brief Set model-specific weight sharding configuration
     *
     * This should be called after construction with the config from
     * the model's schema factory (e.g., Qwen2SchemaFactory).
     *
     * @param config Weight sharding configuration from model schema
     */
    virtual void setWeightShardingConfig(const WeightShardingConfig& config) = 0;
};

} // namespace llaminar2
