/**
 * @file IWeightManager.h
 * @brief Interface for managing model weights with distribution strategies
 *
 * Abstracts weight management to enable:
 * 1. Testing weight sharding without MPI
 * 2. Mock weights for stage unit tests
 * 3. Testing different distribution strategies
 *
 * ## Interface Design
 *
 * Core interface (pure virtual) — the 9 methods that MUST be implemented:
 * - Weight access: getWeightForDevice, getDecodeWeight
 * - Sharding info: isWeightSharded, getShardingMode, isGemmWeight
 * - Strategy: strategy
 * - Configuration: setWeightShardingConfig, setTensorParallelConfig, setWeightPreprocessor
 *
 * Default no-ops — methods with default implementations for backward compatibility:
 * - setModelDimensions, setGDNDimensions (model dimension hints)
 * - hasLMHead, hasEmbedding (PP stage info)
 *
 * Lifecycle methods (packGemmWeights, uploadNonGemmWeights, preloadForDevices,
 * releaseAllHostWeightData, preloadStats) have default no-op implementations.
 * Concrete WeightManager overrides these.
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "WeightTypes.h"
#include "WeightManagerConfig.h"
#include "../backends/DeviceId.h"
#include <memory>
#include <string>
#include <vector>
#include <cstddef>
#include <functional>

namespace llaminar2
{

    // Forward declaration for sharding config
    struct WeightShardingConfig;

    // Forward declarations
    class TensorBase;
    class TensorParallelConfig;

    /**
     * @brief Interface for managing model weights with distribution strategies
     *
     * Implementations:
     * - WeightManager: Real implementation with GGUF loader integration
     * - MockWeightManager: Test implementation with in-memory weights
     *
     * Usage:
     * @code
     * void loadProjection(IWeightManager& weights) {
     *     auto wq = weights.getWeightForDevice("blk.0.attn_q.weight");
     *     if (weights.isWeightSharded("blk.0.attn_q.weight")) {
     *         // Need allreduce after matmul
     *     }
     * }
     * @endcode
     */
    class IWeightManager
    {
    public:
        virtual ~IWeightManager() = default;

        // =========================================================================
        // Core: Weight Access (pure virtual)
        // =========================================================================

        /**
         * @brief Get weight tensor for a specific device (device-isolated instance)
         *
         * Loads from storage if not cached, applies distribution strategy.
         * For multi-device scenarios (LOCAL TP), returns device-specific clones
         * to enable independent coherence tracking.
         *
         * @param name GGUF tensor name (e.g., "token_embd.weight", "blk.0.attn_q.weight")
         * @param device Target device for this tensor instance (default: CPU)
         * @param layer_idx Optional layer index for placement map lookup
         * @return Device-specific tensor instance, or nullptr on error
         */
        virtual std::shared_ptr<TensorBase> getWeightForDevice(
            const std::string &name,
            DeviceId device = DeviceId::cpu(),
            int layer_idx = -1) = 0;

        /**
         * @brief Get or create decode weight shard for CPU decode participation
         *
         * Returns a sliced copy of the weight (tail portion based on fraction).
         * Cached separately from the full prefill weight.
         *
         * @param name Weight tensor name
         * @param decode_device Device for the decode shard (typically CPU)
         * @param fraction Fraction of weight for this shard (e.g., 0.20 = tail 20%)
         * @param layer_idx Layer index for sharding mode lookup
         * @return Sliced weight tensor, or nullptr on error
         */
        virtual std::shared_ptr<TensorBase> getDecodeWeight(
            const std::string &name,
            DeviceId decode_device,
            float fraction,
            int layer_idx = -1) = 0;

        // =========================================================================
        // Core: Sharding Info (pure virtual)
        // =========================================================================

        /**
         * @brief Check if a weight is sharded
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
         * @param name Weight tensor name
         * @return true if weight is a GEMM matrix
         */
        virtual bool isGemmWeight(const std::string &name) const = 0;

        // =========================================================================
        // Core: Strategy (pure virtual)
        // =========================================================================

        /**
         * @brief Get current distribution strategy
         */
        virtual WeightDistributionStrategy strategy() const = 0;

        // =========================================================================
        // Core: Configuration (pure virtual)
        // =========================================================================

        /**
         * @brief Set model-specific weight sharding configuration
         * @param config Weight sharding configuration from model schema
         */
        virtual void setWeightShardingConfig(const WeightShardingConfig &config) = 0;

        /**
         * @brief Set tensor parallel configuration for proportional sharding
         * @param config Tensor parallel config from ILocalTPContext
         */
        virtual void setTensorParallelConfig(std::shared_ptr<TensorParallelConfig> config) = 0;

        /**
         * @brief Register a preprocessor applied to each GEMM weight before packing
         * @param preprocessor Callable that transforms (name, tensor) → tensor
         */
        virtual void setWeightPreprocessor(WeightPreprocessor preprocessor) = 0;

        /**
         * @brief Configure weight manager with a unified config struct
         *
         * Replaces multi-step setter chain (setWeightShardingConfig + setModelDimensions
         * + setGDNDimensions + setTensorParallelConfig) with a single call.
         *
         * Default implementation calls individual setters for backward compatibility.
         * WeightManager overrides for more efficient initialization.
         *
         * @param config Configuration struct with all sharding/dimension settings
         */
        virtual void configure(const WeightManagerConfig &config)
        {
            if (config.hasShardingConfig())
                setWeightShardingConfig(config.sharding);
            if (config.hasTensorParallelConfig())
                setTensorParallelConfig(config.tp_config);
            if (config.hasModelDimensions())
                setModelDimensions(config.dimensions.n_heads, config.dimensions.n_kv_heads, config.dimensions.head_dim);
            if (config.hasGDNDimensions())
                setGDNDimensions(config.dimensions.gdn_n_k_heads, config.dimensions.gdn_n_v_heads, config.dimensions.gdn_d_state);
            if (config.preprocessor)
                setWeightPreprocessor(config.preprocessor);
        }

        // =========================================================================
        // Optional: Model Dimensions (default no-ops)
        // =========================================================================

        /**
         * @brief Set model head dimensions for FusedQKV sub-block slicing
         */
        virtual void setModelDimensions(int n_heads, int n_kv_heads, int head_dim)
        {
            (void)n_heads;
            (void)n_kv_heads;
            (void)head_dim;
        }

        /**
         * @brief Set GDN dimensions for asymmetric FusedQKV sub-block slicing
         */
        virtual void setGDNDimensions(int n_k_heads, int n_v_heads, int d_state)
        {
            (void)n_k_heads;
            (void)n_v_heads;
            (void)d_state;
        }

        // =========================================================================
        // Optional: PP Stage Info (default implementations)
        // =========================================================================

        /**
         * @brief Check if this weight manager owns the LM head
         */
        virtual bool hasLMHead() const { return true; }

        /**
         * @brief Check if this weight manager owns the token embedding
         */
        virtual bool hasEmbedding() const { return true; }

        // =========================================================================
        // Lifecycle (default no-ops — concrete WeightManager overrides these)
        // =========================================================================

        // =========================================================================
        // Weight Preparation (repack + upload, NO host release)
        // =========================================================================

        /**
         * @brief Prepare weights for a single device (pack + upload, no release)
         *
         * Runs GEMM packing, non-GEMM upload, and embedding preparation.
         * Does NOT release host copies — call releaseAllHostWeightData() separately
         * after ALL devices are prepared.
         *
         * @param device Target device
         * @return true on success
         */
        virtual bool prepareWeightsForDevice(DeviceId /*device*/) { return true; }

        /**
         * @brief Prepare weights for a single device, filtered to a layer range
         *
         * Same as prepareWeightsForDevice(device) but only processes weights
         * belonging to the specified layer range. Used by Pipeline Parallelism.
         *
         * @param device Target device
         * @param first_layer First layer index (inclusive)
         * @param last_layer Last layer index (exclusive)
         * @param has_embedding Whether this stage owns the embedding table
         * @param has_lm_head Whether this stage owns the LM head
         * @return true on success
         */
        virtual bool prepareWeightsForDevice(
            DeviceId /*device*/,
            int /*first_layer*/, int /*last_layer*/,
            bool /*has_embedding*/, bool /*has_lm_head*/) { return true; }

        // =========================================================================
        // Weight Lifecycle (convenience entry points)
        // =========================================================================

        /**
         * @brief Complete weight lifecycle for a single device
         *
         * Convenience: prepareWeightsForDevice() + releaseAllHostWeightData().
         * Use when there is only ONE device and no PP sharing.
         *
         * @param device Target device
         * @return true on success
         */
        virtual bool finalizeForDevice(DeviceId /*device*/) { return true; }

        /**
         * @brief Complete weight lifecycle for multiple LOCAL TP devices
         *
         * Runs the full multi-device sequence:
         * 1. Clone and upload weights to all devices (preloadForDevices)
         * 2. Pack GEMM weights per device
         * 3. Release all host weight data (cache_ + per_device_cache_)
         *
         * Call ONCE during RankOrchestrator init.
         *
         * @param devices List of devices that will use the weights
         * @param release_host_data If true, release host weight copies after
         *                          packing and upload. Set false for nested
         *                          TP-in-PP setups where a later PP stage on
         *                          a different device still needs host copies.
         * @return true on success
         */
        virtual bool finalizeForDevices(const std::vector<DeviceId> & /*devices*/,
                                         bool /*release_host_data*/ = true) { return true; }

        // =========================================================================
        // Internal Lifecycle (called by prepareWeightsForDevice/finalizeForDevice)
        // =========================================================================
        // These methods are public for testability but should NOT be called
        // directly in production code. Use prepareWeightsForDevice() or
        // finalizeForDevice() instead.

        virtual bool packGemmWeights(
            DeviceId /*target_device*/,
            PreloadProgressCallback /*progress_cb*/ = nullptr,
            bool /*release_raw_data*/ = false,
            std::function<bool(const std::string &)> /*layer_filter*/ = nullptr) { return true; }

        virtual bool uploadNonGemmWeights(
            DeviceId /*target_device*/,
            std::function<bool(const std::string &)> /*layer_filter*/ = nullptr) { return true; }

        virtual bool preloadForDevices(const std::vector<DeviceId> & /*devices*/) { return true; }

        virtual size_t releaseAllHostWeightData() { return 0; }

        /**
         * @brief Release host data for tensors that were retained as host-resident
         *
         * Called after the first forward pass completes, when GPU kernels have
         * uploaded their own device copies (e.g., embedding repack+upload to workspace).
         * At that point, the host data is no longer needed.
         *
         * @return Number of tensors whose host data was released
         */
        virtual size_t releaseHostResidentWeightData() { return 0; }

        /**
         * @brief Query how many prepared embedding entries exist
         *
         * Used by releaseAllHostWeightData() to decide whether host-resident
         * embedding tensors can be freed (because a GPU-side repack exists).
         * Override in tests to control release decisions without depending
         * on KernelFactory static state.
         */
        virtual size_t getPreparedEmbeddingCount() const { return 0; }

        virtual std::pair<size_t, size_t> preloadStats() const { return {0, 0}; }
    };

} // namespace llaminar2
