/**
 * @file MockWeightManager.h
 * @brief Mock weight manager for unit testing without GGUF files
 *
 * This mock enables:
 * - Testing weight loading logic without actual model files
 * - Testing weight sharding without MPI
 * - Testing stage unit tests with controlled weight data
 * - Testing allreduce stages with simulated sharded weights
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "loaders/IWeightManager.h"
#include "loaders/WeightManager.h"                       // For ShardingMode and WeightDistributionStrategy enums
#include "execution/local_execution/graph/GraphSchema.h" // For WeightShardingConfig
#include "config/TensorParallelConfig.h"                 // For TensorParallelConfig (LOCAL TP support)
#include "tensors/Tensors.h"
#include "tensors/FP16Utils.h" // For fp32_to_fp16
#include "backends/DeviceId.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <random>
#include <stdexcept>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <optional>

namespace llaminar2::test
{

    // Forward declaration
    class MockWeightManagerBuilder;

    /**
     * @brief Mock weight manager for unit testing
     *
     * Provides in-memory weights without file I/O for testing:
     * - Configurable weight distribution strategy
     * - Pre-built tensors with random or specified data
     * - Configurable sharding behavior per weight
     * - Support for FP32, Q4_0, Q8_0, IQ4_NL tensors
     * - Standalone mode (no ModelLoader required)
     *
     * Usage:
     * @code
     * // Simple usage with builder
     * auto mock = MockWeightManagerBuilder()
     *     .setStrategy(WeightDistributionStrategy::SHARDED)
     *     .addFP32RandomWeight("blk.0.attn_q.weight", {896, 896})
     *     .setSharded("blk.0.attn_q.weight", ShardingMode::COLUMN_PARALLEL)
     *     .build();
     *
     * auto weight = mock->getWeight("blk.0.attn_q.weight");
     * bool is_sharded = mock->isWeightSharded("blk.0.attn_q.weight");
     * @endcode
     */
    class MockWeightManager : public IWeightManager
    {
    public:
        // =========================================================================
        // Presets for Common Configurations
        // =========================================================================

        /**
         * @brief Create mock with replicated strategy and minimal weights
         *
         * Suitable for testing stages that don't care about sharding.
         */
        static std::shared_ptr<MockWeightManager> createReplicated();

        /**
         * @brief Create mock with sharded strategy and standard Qwen2 weight layout
         *
         * Sets up sharding modes matching Qwen2 tensor parallelism:
         * - Column-parallel: QKV, Gate/Up projections
         * - Row-parallel: Wo projection
         * - Input-parallel: Down projection
         * - Replicated: Norms, embeddings
         */
        static std::shared_ptr<MockWeightManager> createShardedQwen2();

        // =========================================================================
        // Construction
        // =========================================================================

        MockWeightManager();
        ~MockWeightManager() override = default;

        // =========================================================================
        // IWeightManager Implementation
        // =========================================================================

        std::shared_ptr<TensorBase> getWeight(
            const std::string &name,
            DeviceId device = DeviceId::cpu(),
            int layer_idx = -1) override;

        std::shared_ptr<TensorBase> getWeightForDevice(
            const std::string &name,
            DeviceId device,
            int layer_idx = -1) override;

        bool preloadForDevices(const std::vector<DeviceId> &devices) override;

        std::shared_ptr<TensorBase> getDecodeWeight(
            const std::string &name,
            DeviceId decode_device,
            float fraction,
            int layer_idx = -1) override;

        bool isWeightSharded(const std::string &name) const override;
        ShardingMode getShardingMode(const std::string &name) const override;
        bool isGemmWeight(const std::string &name) const override;

        WeightDistributionStrategy strategy() const override { return strategy_; }

        size_t cacheSize() const override { return weights_.size(); }
        void clearCache() override { weights_.clear(); }
        size_t decodeCacheSize() const override { return decode_weights_.size(); }
        void clearDecodeCache() override { decode_weights_.clear(); }

        /**
         * @brief Set model-specific weight sharding configuration
         * @param config Weight sharding configuration from model schema
         */
        void setWeightShardingConfig(const WeightShardingConfig &config) override
        {
            sharding_config_ = config;
        }

        /**
         * @brief Set tensor parallel configuration for LOCAL TP weight slicing
         * @param config Tensor parallel config from ILocalTPContext
         */
        void setTensorParallelConfig(std::shared_ptr<TensorParallelConfig> config) override
        {
            tp_config_ = std::move(config);
        }

        // =========================================================================
        // Weight Packing and Preloading (stub implementations for mock)
        // =========================================================================

        /**
         * @brief Pack all GEMM weights for a target device (mock: no-op)
         */
        bool packGemmWeights(
            DeviceId /* target_device */,
            PreloadProgressCallback /* progress_cb */ = nullptr,
            bool /* release_raw_data */ = false) override
        {
            return true; // Mock always succeeds
        }

        /**
         * @brief Upload all non-GEMM weights to GPU (mock: no-op)
         */
        bool uploadNonGemmWeights(DeviceId /* target_device */) override
        {
            return true; // Mock always succeeds
        }

        /**
         * @brief Get statistics about preloaded weights (mock: always 0)
         */
        std::pair<size_t, size_t> preloadStats() const override
        {
            return {0, 0}; // Mock tracks no preload statistics
        }

        // =========================================================================
        // Test Configuration API (for use by Builder and tests)
        // =========================================================================

        /**
         * @brief Set distribution strategy
         */
        void setStrategy(WeightDistributionStrategy strategy) { strategy_ = strategy; }

        /**
         * @brief Add a pre-built weight tensor
         * @param name Weight name (e.g., "blk.0.attn_q.weight")
         * @param tensor Tensor to add
         */
        void addWeight(const std::string &name, std::shared_ptr<TensorBase> tensor);

        /**
         * @brief Add FP32 weight with random data
         * @param name Weight name
         * @param shape Tensor dimensions [rows, cols]
         * @param min Minimum random value
         * @param max Maximum random value
         * @param seed Random seed for reproducibility
         */
        void addFP32RandomWeight(
            const std::string &name,
            const std::vector<size_t> &shape,
            float min = -1.0f, float max = 1.0f,
            uint32_t seed = 42);

        /**
         * @brief Add FP32 weight filled with zeros
         */
        void addFP32ZerosWeight(const std::string &name, const std::vector<size_t> &shape);

        /**
         * @brief Add FP32 weight filled with ones
         */
        void addFP32OnesWeight(const std::string &name, const std::vector<size_t> &shape);

        /**
         * @brief Add Q4_0 weight with random quantized data
         */
        void addQ4_0RandomWeight(
            const std::string &name,
            const std::vector<size_t> &shape,
            float min = -1.0f, float max = 1.0f,
            uint32_t seed = 42);

        /**
         * @brief Add Q8_0 weight with random quantized data
         */
        void addQ8_0RandomWeight(
            const std::string &name,
            const std::vector<size_t> &shape,
            float min = -1.0f, float max = 1.0f,
            uint32_t seed = 42);

        /**
         * @brief Set sharding mode for a weight
         * @param name Weight name
         * @param mode Sharding mode
         */
        void setShardingMode(const std::string &name, ShardingMode mode);

        /**
         * @brief Mark a weight as GEMM (default) or non-GEMM
         * @param name Weight name
         * @param is_gemm true if this is a GEMM weight
         */
        void setGemmWeight(const std::string &name, bool is_gemm);

        /**
         * @brief Add decode weight shard directly (for testing)
         * @param name Weight name
         * @param tensor Decode weight tensor
         */
        void addDecodeWeight(const std::string &name, std::shared_ptr<TensorBase> tensor);

        // =========================================================================
        // Test Inspection API
        // =========================================================================

        /**
         * @brief Check if a weight exists
         */
        bool hasWeight(const std::string &name) const;

        /**
         * @brief Get all weight names
         */
        std::vector<std::string> weightNames() const;

        /**
         * @brief Get number of times getWeight was called
         */
        size_t getWeightCallCount() const { return get_weight_calls_; }

        /**
         * @brief Get names of weights that were requested but not found
         */
        const std::vector<std::string> &missingWeightRequests() const { return missing_requests_; }

        /**
         * @brief Reset call counts and missing requests
         */
        void resetCounters();

    private:
        // Distribution strategy
        WeightDistributionStrategy strategy_ = WeightDistributionStrategy::REPLICATED;

        // Weight storage
        std::unordered_map<std::string, std::shared_ptr<TensorBase>> weights_;
        std::unordered_map<std::string, std::shared_ptr<TensorBase>> decode_weights_;

        // Sharding configuration
        std::unordered_map<std::string, ShardingMode> sharding_modes_;
        std::unordered_set<std::string> non_gemm_weights_;
        std::optional<WeightShardingConfig> sharding_config_;
        std::shared_ptr<TensorParallelConfig> tp_config_;

        // Call tracking
        mutable size_t get_weight_calls_ = 0;
        mutable std::vector<std::string> missing_requests_;
    };

    /**
     * @brief Fluent builder for MockWeightManager
     *
     * Example:
     * @code
     * auto mock = MockWeightManagerBuilder()
     *     .setStrategy(WeightDistributionStrategy::SHARDED)
     *     .addFP32RandomWeight("blk.0.attn_q.weight", {896, 896})
     *     .setSharded("blk.0.attn_q.weight", ShardingMode::COLUMN_PARALLEL)
     *     .addFP32RandomWeight("blk.0.attn_norm.weight", {896})
     *     .setNonGemm("blk.0.attn_norm.weight")
     *     .build();
     * @endcode
     */
    class MockWeightManagerBuilder
    {
    public:
        MockWeightManagerBuilder();

        // Strategy config
        MockWeightManagerBuilder &setStrategy(WeightDistributionStrategy strategy);

        // Weight addition
        MockWeightManagerBuilder &addWeight(const std::string &name, std::shared_ptr<TensorBase> tensor);
        MockWeightManagerBuilder &addFP32RandomWeight(
            const std::string &name,
            const std::vector<size_t> &shape,
            float min = -1.0f, float max = 1.0f,
            uint32_t seed = 42);
        MockWeightManagerBuilder &addFP32ZerosWeight(const std::string &name, const std::vector<size_t> &shape);
        MockWeightManagerBuilder &addFP32OnesWeight(const std::string &name, const std::vector<size_t> &shape);
        MockWeightManagerBuilder &addQ4_0RandomWeight(
            const std::string &name,
            const std::vector<size_t> &shape,
            float min = -1.0f, float max = 1.0f,
            uint32_t seed = 42);
        MockWeightManagerBuilder &addQ8_0RandomWeight(
            const std::string &name,
            const std::vector<size_t> &shape,
            float min = -1.0f, float max = 1.0f,
            uint32_t seed = 42);

        // Sharding configuration
        MockWeightManagerBuilder &setSharded(const std::string &name, ShardingMode mode);
        MockWeightManagerBuilder &setReplicated(const std::string &name);
        MockWeightManagerBuilder &setColumnParallel(const std::string &name);
        MockWeightManagerBuilder &setRowParallel(const std::string &name);
        MockWeightManagerBuilder &setInputParallel(const std::string &name);

        // GEMM weight configuration
        MockWeightManagerBuilder &setGemm(const std::string &name);
        MockWeightManagerBuilder &setNonGemm(const std::string &name);

        // Decode weights
        MockWeightManagerBuilder &addDecodeWeight(const std::string &name, std::shared_ptr<TensorBase> tensor);

        // Preset layers (adds weights with appropriate sharding)
        MockWeightManagerBuilder &addAttentionLayer(int layer_idx, size_t hidden_dim, size_t head_dim, size_t num_heads, size_t num_kv_heads);
        MockWeightManagerBuilder &addFFNLayer(int layer_idx, size_t hidden_dim, size_t ffn_dim);
        MockWeightManagerBuilder &addNormWeights(int layer_idx, size_t hidden_dim);
        MockWeightManagerBuilder &addEmbedding(size_t vocab_size, size_t hidden_dim);
        MockWeightManagerBuilder &addLMHead(size_t vocab_size, size_t hidden_dim);

        // Build
        std::shared_ptr<MockWeightManager> build();

    private:
        std::shared_ptr<MockWeightManager> mock_;
    };

    // =============================================================================
    // IMPLEMENTATION
    // =============================================================================

    inline MockWeightManager::MockWeightManager() = default;

    inline std::shared_ptr<TensorBase> MockWeightManager::getWeight(
        const std::string &name,
        DeviceId device,
        int layer_idx)
    {
        ++get_weight_calls_;
        (void)device;    // Unused in mock
        (void)layer_idx; // Unused in mock

        auto it = weights_.find(name);
        if (it == weights_.end())
        {
            missing_requests_.push_back(name);
            return nullptr;
        }
        return it->second;
    }

    inline std::shared_ptr<TensorBase> MockWeightManager::getWeightForDevice(
        const std::string &name,
        DeviceId device,
        int layer_idx)
    {
        // For mock, just delegate to getWeight - no per-device cloning needed
        return getWeight(name, device, layer_idx);
    }

    inline bool MockWeightManager::preloadForDevices(const std::vector<DeviceId> &devices)
    {
        // For mock, nothing to preload - all weights are already in memory
        (void)devices;
        return true;
    }

    inline std::shared_ptr<TensorBase> MockWeightManager::getDecodeWeight(
        const std::string &name,
        DeviceId decode_device,
        float fraction,
        int layer_idx)
    {
        (void)decode_device;
        (void)fraction;
        (void)layer_idx;

        // First check decode cache
        auto it = decode_weights_.find(name);
        if (it != decode_weights_.end())
        {
            return it->second;
        }

        // Fall back to main weight (for simple tests)
        return getWeight(name);
    }

    inline bool MockWeightManager::isWeightSharded(const std::string &name) const
    {
        if (strategy_ != WeightDistributionStrategy::SHARDED)
        {
            return false;
        }

        auto mode = getShardingMode(name);
        return mode != ShardingMode::REPLICATE;
    }

    inline ShardingMode MockWeightManager::getShardingMode(const std::string &name) const
    {
        auto it = sharding_modes_.find(name);
        if (it != sharding_modes_.end())
        {
            return it->second;
        }
        // Default: REPLICATE
        return ShardingMode::REPLICATE;
    }

    inline bool MockWeightManager::isGemmWeight(const std::string &name) const
    {
        // If explicitly marked as non-GEMM, return false
        return non_gemm_weights_.find(name) == non_gemm_weights_.end();
    }

    inline void MockWeightManager::addWeight(const std::string &name, std::shared_ptr<TensorBase> tensor)
    {
        weights_[name] = tensor;
    }

    inline void MockWeightManager::addFP32RandomWeight(
        const std::string &name,
        const std::vector<size_t> &shape,
        float min, float max,
        uint32_t seed)
    {
        size_t rows = shape.size() >= 1 ? shape[0] : 1;
        size_t cols = shape.size() >= 2 ? shape[1] : 1;

        auto tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{rows, cols});
        float *data = tensor->mutable_data();

        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(min, max);
        for (size_t i = 0; i < rows * cols; ++i)
        {
            data[i] = dist(rng);
        }

        weights_[name] = tensor;
    }

    inline void MockWeightManager::addFP32ZerosWeight(const std::string &name, const std::vector<size_t> &shape)
    {
        size_t rows = shape.size() >= 1 ? shape[0] : 1;
        size_t cols = shape.size() >= 2 ? shape[1] : 1;

        auto tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{rows, cols});
        float *data = tensor->mutable_data();
        std::fill(data, data + rows * cols, 0.0f);

        weights_[name] = tensor;
    }

    inline void MockWeightManager::addFP32OnesWeight(const std::string &name, const std::vector<size_t> &shape)
    {
        size_t rows = shape.size() >= 1 ? shape[0] : 1;
        size_t cols = shape.size() >= 2 ? shape[1] : 1;

        auto tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{rows, cols});
        float *data = tensor->mutable_data();
        std::fill(data, data + rows * cols, 1.0f);

        weights_[name] = tensor;
    }

    inline void MockWeightManager::addQ4_0RandomWeight(
        const std::string &name,
        const std::vector<size_t> &shape,
        float min, float max,
        uint32_t seed)
    {
        constexpr size_t BLOCK_SIZE = 32;

        size_t rows = shape.size() >= 1 ? shape[0] : 1;
        size_t cols = shape.size() >= 2 ? shape[1] : 1;
        size_t blocks_per_row = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
        size_t total_blocks = rows * blocks_per_row;

        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(min, max);

        // Q4_0: 32 elements per block, 18 bytes per block (2 bytes scale + 16 bytes quants)
        std::vector<uint8_t> raw_data(total_blocks * 18);

        for (size_t b = 0; b < total_blocks; ++b)
        {
            float block_vals[BLOCK_SIZE];
            float abs_max = 0.0f;
            for (size_t i = 0; i < BLOCK_SIZE; ++i)
            {
                block_vals[i] = dist(rng);
                abs_max = std::max(abs_max, std::abs(block_vals[i]));
            }

            float scale = abs_max / 7.0f;
            uint16_t scale_fp16 = fp32_to_fp16(scale);

            uint8_t *block_ptr = raw_data.data() + b * 18;
            std::memcpy(block_ptr, &scale_fp16, 2);

            float inv_scale = scale > 0 ? 1.0f / scale : 0.0f;
            for (size_t i = 0; i < BLOCK_SIZE / 2; ++i)
            {
                int q0 = std::clamp(static_cast<int>(std::round(block_vals[2 * i] * inv_scale)) + 8, 0, 15);
                int q1 = std::clamp(static_cast<int>(std::round(block_vals[2 * i + 1] * inv_scale)) + 8, 0, 15);
                block_ptr[2 + i] = static_cast<uint8_t>((q1 << 4) | q0);
            }
        }

        auto tensor = std::make_shared<Q4_0Tensor>(std::vector<size_t>{rows, cols}, raw_data);
        weights_[name] = tensor;
    }

    inline void MockWeightManager::addQ8_0RandomWeight(
        const std::string &name,
        const std::vector<size_t> &shape,
        float min, float max,
        uint32_t seed)
    {
        constexpr size_t BLOCK_SIZE = 32;

        size_t rows = shape.size() >= 1 ? shape[0] : 1;
        size_t cols = shape.size() >= 2 ? shape[1] : 1;
        size_t blocks_per_row = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
        size_t total_blocks = rows * blocks_per_row;

        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(min, max);

        // Q8_0: 32 elements per block, 34 bytes per block (2 bytes scale + 32 bytes quants)
        std::vector<uint8_t> raw_data(total_blocks * 34);

        for (size_t b = 0; b < total_blocks; ++b)
        {
            float block_vals[BLOCK_SIZE];
            float abs_max = 0.0f;
            for (size_t i = 0; i < BLOCK_SIZE; ++i)
            {
                block_vals[i] = dist(rng);
                abs_max = std::max(abs_max, std::abs(block_vals[i]));
            }

            float scale = abs_max / 127.0f;
            uint16_t scale_fp16 = fp32_to_fp16(scale);

            uint8_t *block_ptr = raw_data.data() + b * 34;
            std::memcpy(block_ptr, &scale_fp16, 2);

            float inv_scale = scale > 0 ? 1.0f / scale : 0.0f;
            for (size_t i = 0; i < BLOCK_SIZE; ++i)
            {
                int q = std::clamp(static_cast<int>(std::round(block_vals[i] * inv_scale)), -128, 127);
                block_ptr[2 + i] = static_cast<uint8_t>(static_cast<int8_t>(q));
            }
        }

        auto tensor = std::make_shared<Q8_0Tensor>(std::vector<size_t>{rows, cols}, raw_data);
        weights_[name] = tensor;
    }

    inline void MockWeightManager::setShardingMode(const std::string &name, ShardingMode mode)
    {
        sharding_modes_[name] = mode;
    }

    inline void MockWeightManager::setGemmWeight(const std::string &name, bool is_gemm)
    {
        if (is_gemm)
        {
            non_gemm_weights_.erase(name);
        }
        else
        {
            non_gemm_weights_.insert(name);
        }
    }

    inline void MockWeightManager::addDecodeWeight(const std::string &name, std::shared_ptr<TensorBase> tensor)
    {
        decode_weights_[name] = tensor;
    }

    inline bool MockWeightManager::hasWeight(const std::string &name) const
    {
        return weights_.find(name) != weights_.end();
    }

    inline std::vector<std::string> MockWeightManager::weightNames() const
    {
        std::vector<std::string> names;
        names.reserve(weights_.size());
        for (const auto &[name, _] : weights_)
        {
            names.push_back(name);
        }
        return names;
    }

    inline void MockWeightManager::resetCounters()
    {
        get_weight_calls_ = 0;
        missing_requests_.clear();
    }

    inline std::shared_ptr<MockWeightManager> MockWeightManager::createReplicated()
    {
        return MockWeightManagerBuilder()
            .setStrategy(WeightDistributionStrategy::REPLICATED)
            .build();
    }

    inline std::shared_ptr<MockWeightManager> MockWeightManager::createShardedQwen2()
    {
        return MockWeightManagerBuilder()
            .setStrategy(WeightDistributionStrategy::SHARDED)
            .build();
    }

    // =============================================================================
    // Builder Implementation
    // =============================================================================

    inline MockWeightManagerBuilder::MockWeightManagerBuilder()
        : mock_(std::make_shared<MockWeightManager>()) {}

    inline MockWeightManagerBuilder &MockWeightManagerBuilder::setStrategy(WeightDistributionStrategy strategy)
    {
        mock_->setStrategy(strategy);
        return *this;
    }

    inline MockWeightManagerBuilder &MockWeightManagerBuilder::addWeight(
        const std::string &name, std::shared_ptr<TensorBase> tensor)
    {
        mock_->addWeight(name, tensor);
        return *this;
    }

    inline MockWeightManagerBuilder &MockWeightManagerBuilder::addFP32RandomWeight(
        const std::string &name,
        const std::vector<size_t> &shape,
        float min, float max,
        uint32_t seed)
    {
        mock_->addFP32RandomWeight(name, shape, min, max, seed);
        return *this;
    }

    inline MockWeightManagerBuilder &MockWeightManagerBuilder::addFP32ZerosWeight(
        const std::string &name, const std::vector<size_t> &shape)
    {
        mock_->addFP32ZerosWeight(name, shape);
        return *this;
    }

    inline MockWeightManagerBuilder &MockWeightManagerBuilder::addFP32OnesWeight(
        const std::string &name, const std::vector<size_t> &shape)
    {
        mock_->addFP32OnesWeight(name, shape);
        return *this;
    }

    inline MockWeightManagerBuilder &MockWeightManagerBuilder::addQ4_0RandomWeight(
        const std::string &name,
        const std::vector<size_t> &shape,
        float min, float max,
        uint32_t seed)
    {
        mock_->addQ4_0RandomWeight(name, shape, min, max, seed);
        return *this;
    }

    inline MockWeightManagerBuilder &MockWeightManagerBuilder::addQ8_0RandomWeight(
        const std::string &name,
        const std::vector<size_t> &shape,
        float min, float max,
        uint32_t seed)
    {
        mock_->addQ8_0RandomWeight(name, shape, min, max, seed);
        return *this;
    }

    inline MockWeightManagerBuilder &MockWeightManagerBuilder::setSharded(
        const std::string &name, ShardingMode mode)
    {
        mock_->setShardingMode(name, mode);
        return *this;
    }

    inline MockWeightManagerBuilder &MockWeightManagerBuilder::setReplicated(const std::string &name)
    {
        return setSharded(name, ShardingMode::REPLICATE);
    }

    inline MockWeightManagerBuilder &MockWeightManagerBuilder::setColumnParallel(const std::string &name)
    {
        return setSharded(name, ShardingMode::COLUMN_PARALLEL);
    }

    inline MockWeightManagerBuilder &MockWeightManagerBuilder::setRowParallel(const std::string &name)
    {
        return setSharded(name, ShardingMode::ROW_PARALLEL);
    }

    inline MockWeightManagerBuilder &MockWeightManagerBuilder::setInputParallel(const std::string &name)
    {
        return setSharded(name, ShardingMode::INPUT_PARALLEL);
    }

    inline MockWeightManagerBuilder &MockWeightManagerBuilder::setGemm(const std::string &name)
    {
        mock_->setGemmWeight(name, true);
        return *this;
    }

    inline MockWeightManagerBuilder &MockWeightManagerBuilder::setNonGemm(const std::string &name)
    {
        mock_->setGemmWeight(name, false);
        return *this;
    }

    inline MockWeightManagerBuilder &MockWeightManagerBuilder::addDecodeWeight(
        const std::string &name, std::shared_ptr<TensorBase> tensor)
    {
        mock_->addDecodeWeight(name, tensor);
        return *this;
    }

    inline MockWeightManagerBuilder &MockWeightManagerBuilder::addAttentionLayer(
        int layer_idx, size_t hidden_dim, size_t head_dim, size_t num_heads, size_t num_kv_heads)
    {
        std::string prefix = "blk." + std::to_string(layer_idx) + ".";
        size_t q_dim = num_heads * head_dim;
        size_t kv_dim = num_kv_heads * head_dim;

        // QKV projections (column-parallel for tensor parallelism)
        addFP32RandomWeight(prefix + "attn_q.weight", {q_dim, hidden_dim});
        setColumnParallel(prefix + "attn_q.weight");

        addFP32RandomWeight(prefix + "attn_k.weight", {kv_dim, hidden_dim});
        setColumnParallel(prefix + "attn_k.weight");

        addFP32RandomWeight(prefix + "attn_v.weight", {kv_dim, hidden_dim});
        setColumnParallel(prefix + "attn_v.weight");

        // Output projection (row-parallel, needs allreduce)
        addFP32RandomWeight(prefix + "attn_output.weight", {hidden_dim, q_dim});
        setRowParallel(prefix + "attn_output.weight");

        return *this;
    }

    inline MockWeightManagerBuilder &MockWeightManagerBuilder::addFFNLayer(
        int layer_idx, size_t hidden_dim, size_t ffn_dim)
    {
        std::string prefix = "blk." + std::to_string(layer_idx) + ".";

        // Gate/Up projections (column-parallel)
        addFP32RandomWeight(prefix + "ffn_gate.weight", {ffn_dim, hidden_dim});
        setColumnParallel(prefix + "ffn_gate.weight");

        addFP32RandomWeight(prefix + "ffn_up.weight", {ffn_dim, hidden_dim});
        setColumnParallel(prefix + "ffn_up.weight");

        // Down projection (input-parallel, needs allreduce)
        addFP32RandomWeight(prefix + "ffn_down.weight", {hidden_dim, ffn_dim});
        setInputParallel(prefix + "ffn_down.weight");

        return *this;
    }

    inline MockWeightManagerBuilder &MockWeightManagerBuilder::addNormWeights(
        int layer_idx, size_t hidden_dim)
    {
        std::string prefix = "blk." + std::to_string(layer_idx) + ".";

        // Norms are replicated (1D tensors)
        addFP32OnesWeight(prefix + "attn_norm.weight", {hidden_dim});
        setReplicated(prefix + "attn_norm.weight");
        setNonGemm(prefix + "attn_norm.weight");

        addFP32OnesWeight(prefix + "ffn_norm.weight", {hidden_dim});
        setReplicated(prefix + "ffn_norm.weight");
        setNonGemm(prefix + "ffn_norm.weight");

        return *this;
    }

    inline MockWeightManagerBuilder &MockWeightManagerBuilder::addEmbedding(
        size_t vocab_size, size_t hidden_dim)
    {
        addFP32RandomWeight("token_embd.weight", {vocab_size, hidden_dim});
        setReplicated("token_embd.weight");
        setNonGemm("token_embd.weight");
        return *this;
    }

    inline MockWeightManagerBuilder &MockWeightManagerBuilder::addLMHead(
        size_t vocab_size, size_t hidden_dim)
    {
        addFP32RandomWeight("output.weight", {vocab_size, hidden_dim});
        setColumnParallel("output.weight"); // LM head is column-parallel
        return *this;
    }

    inline std::shared_ptr<MockWeightManager> MockWeightManagerBuilder::build()
    {
        return mock_;
    }

} // namespace llaminar2::test
