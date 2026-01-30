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
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <cstddef>

namespace llaminar2
{

    /**
     * @brief Pre-packing progress callback for weight preloading
     *
     * Called during preloading with:
     * @param current Number of weights packed so far
     * @param total Total number of weights to pack
     * @param name Current weight name being packed
     */
    using PreloadProgressCallback = std::function<void(size_t current, size_t total, const std::string &name)>;

    // Forward declaration for sharding config
    struct WeightShardingConfig;

    // Forward declarations
    class TensorBase;
    class TensorParallelConfig;
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
    class IWeightManager
    {
    public:
        virtual ~IWeightManager() = default;

        // =========================================================================
        // Weight Access
        // =========================================================================

        /**
         * @brief Get weight tensor by name (shared instance)
         *
         * Loads from storage if not cached, applies distribution strategy.
         * Returns the SAME tensor instance for repeated calls with the same name.
         *
         * WARNING: For multi-device scenarios where each device needs independent
         * coherence tracking, use getWeightForDevice() instead.
         *
         * @param name GGUF tensor name (e.g., "token_embd.weight", "blk.0.attn_q.weight")
         * @param device Device for tensor placement (default: CPU)
         * @param layer_idx Optional layer index for placement map lookup
         * @return Shared pointer to tensor, or nullptr on error
         */
        virtual std::shared_ptr<TensorBase> getWeight(
            const std::string &name,
            DeviceId device = DeviceId::cpu(),
            int layer_idx = -1) = 0;

        /**
         * @brief Get weight tensor for a specific device (device-isolated instance)
         *
         * For multi-device scenarios (LOCAL TP), each device needs its own tensor
         * instance to track coherence state independently. This method:
         * - Returns the original tensor for the first device that requests it
         * - Creates and caches a clone for subsequent devices
         * - Clones are uploaded to GPU independently, avoiding race conditions
         *
         * This solves the problem where multiple devices running forward() in parallel
         * would race to call ensureOnDevice() on a shared tensor.
         *
         * @param name GGUF tensor name
         * @param device Target device for this tensor instance
         * @param layer_idx Optional layer index for placement map lookup
         * @return Device-specific tensor instance, or nullptr on error
         */
        virtual std::shared_ptr<TensorBase> getWeightForDevice(
            const std::string &name,
            DeviceId device,
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
            const std::string &name,
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
        virtual bool isWeightSharded(const std::string &name) const = 0;

        /**
         * @brief Get sharding mode for a weight
         * @param name Weight tensor name
         * @return ShardingMode indicating how weight is partitioned
         */
        virtual ShardingMode getShardingMode(const std::string &name) const = 0;

        /**
         * @brief Check if a weight is used for GEMM operations
         *
         * Non-GEMM weights (norms, biases, embeddings) need raw data retained.
         * GEMM weights can have raw data released after packing.
         *
         * @param name Weight tensor name
         * @return true if weight is a GEMM matrix
         */
        virtual bool isGemmWeight(const std::string &name) const = 0;

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
        virtual void setWeightShardingConfig(const WeightShardingConfig &config) = 0;

        /**
         * @brief Set tensor parallel configuration for proportional sharding
         *
         * Used by LOCAL TP to configure device-aware weight slicing.
         * When set, getShardedWeightForAssignment() uses this config to
         * determine slice bounds for each device.
         *
         * @param config Tensor parallel config from ILocalTPContext
         */
        virtual void setTensorParallelConfig(std::shared_ptr<TensorParallelConfig> config) = 0;

        // =========================================================================
        // Multi-Device Pre-loading
        // =========================================================================

        /**
         * @brief Pre-load and upload weights for multiple devices
         *
         * For LOCAL tensor parallelism scenarios, this method:
         * 1. Creates device-specific clones of all cached weights
         * 2. Uploads each clone to its target device
         * 3. Caches the device-specific tensors for getWeightForDevice()
         *
         * This MUST be called BEFORE creating device runners to avoid race conditions
         * where multiple devices try to upload the same tensor concurrently.
         *
         * @param devices List of devices that will use the weights
         * @return true if all weights were pre-loaded successfully
         */
        virtual bool preloadForDevices(const std::vector<DeviceId> &devices) = 0;

        // =========================================================================
        // Weight Packing and Preloading
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
        virtual bool packGemmWeights(
            DeviceId target_device,
            PreloadProgressCallback progress_cb = nullptr,
            bool release_raw_data = false) = 0;

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
        virtual bool uploadNonGemmWeights(DeviceId target_device) = 0;

        /**
         * @brief Get statistics about preloaded weights
         *
         * @return Pair of (num_cpu_packed, num_gpu_packed)
         */
        virtual std::pair<size_t, size_t> preloadStats() const = 0;
    };

} // namespace llaminar2
