/**
 * @file MockWeightPlacementMap.h
 * @brief Mock weight placement map for unit testing without real device placement
 *
 * This mock enables:
 * - Testing weight loading logic without real device placement
 * - Simulating different placement scenarios (CPU-only, GPU, hybrid)
 * - Call tracking for behavior verification
 * - Failure injection for robustness testing
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "loaders/IWeightPlacementMap.h"
#include <vector>
#include <unordered_map>
#include <functional>
#include <mutex>
#include <atomic>
#include <stdexcept>

namespace llaminar2::test
{

    /**
     * @brief Mock weight placement map for unit testing
     *
     * Provides configurable behavior for testing weight placement logic:
     * - Configurable default device
     * - Tensor-to-device mapping
     * - Layer-to-device mapping
     * - Call tracking for verification
     * - Presets for common scenarios
     *
     * Usage:
     * @code
     * // Simple CPU-only mock
     * auto mock = MockWeightPlacementMap::Builder()
     *     .withDefaultDevice(DeviceId::cpu())
     *     .build();
     *
     * // GPU with specific layers on CPU
     * auto mock = MockWeightPlacementMap::Builder()
     *     .withDefaultDevice(DeviceId::cuda(0))
     *     .withLayerDevice(0, DeviceId::cpu())  // First layer on CPU
     *     .withLayerDevice(23, DeviceId::cpu()) // Last layer on CPU
     *     .build();
     *
     * // Use preset for common scenario
     * auto mock = MockWeightPlacementMap::cpuOnlyPreset();
     *
     * // Verify behavior
     * EXPECT_EQ(mock->getDeviceForWeight_call_count(), 5);
     * @endcode
     */
    class MockWeightPlacementMap : public IWeightPlacementMap
    {
    public:
        // =========================================================================
        // Configuration
        // =========================================================================

        /**
         * @brief Configuration options for the mock
         */
        struct Config
        {
            DeviceId default_device = DeviceId::cpu(); ///< Default device for unmapped weights
            bool track_calls = true;                   ///< Record call counts for verification
            std::string applied_strategy_name;         ///< Strategy name to report
            bool has_plan = false;                     ///< Whether a plan has been applied

            // Failure injection
            bool lookup_should_fail = false; ///< If true, getDeviceForWeight throws
            std::string failure_message = "MockWeightPlacementMap: injected failure";
        };

        /**
         * @brief Builder for fluent mock configuration
         */
        class Builder
        {
        public:
            Builder() = default;

            /// Set the default device
            Builder &withDefaultDevice(DeviceId device)
            {
                config_.default_device = device;
                return *this;
            }

            /// Enable/disable call tracking
            Builder &withCallTracking(bool enabled)
            {
                config_.track_calls = enabled;
                return *this;
            }

            /// Set the applied strategy name
            Builder &withStrategyName(const std::string &name)
            {
                config_.applied_strategy_name = name;
                config_.has_plan = true;
                return *this;
            }

            /// Mark that a plan has been applied
            Builder &withPlanApplied(bool applied)
            {
                config_.has_plan = applied;
                return *this;
            }

            /// Add a tensor → device mapping
            Builder &withTensorDevice(const std::string &tensor_name, DeviceId device)
            {
                tensor_mappings_[tensor_name] = device;
                return *this;
            }

            /// Add a layer → device mapping
            Builder &withLayerDevice(int layer_idx, DeviceId device)
            {
                layer_mappings_[layer_idx] = device;
                return *this;
            }

            /// Add a layer range → device mapping
            Builder &withLayerRange(int start_layer, int end_layer, DeviceId device)
            {
                for (int i = start_layer; i <= end_layer; ++i)
                {
                    layer_mappings_[i] = device;
                }
                return *this;
            }

            /// Add a pattern → device mapping
            Builder &withPatternDevice(const std::string &pattern, DeviceId device)
            {
                pattern_mappings_[pattern] = device;
                return *this;
            }

            /// Add attention block → device mapping
            Builder &withAttentionDevice(int layer_idx, DeviceId device)
            {
                attention_mappings_[layer_idx] = device;
                return *this;
            }

            /// Add FFN block → device mapping
            Builder &withFFNDevice(int layer_idx, DeviceId device)
            {
                ffn_mappings_[layer_idx] = device;
                return *this;
            }

            /// Add shared expert → device mapping
            Builder &withSharedExpertDevice(int expert_idx, DeviceId device)
            {
                shared_expert_mappings_[expert_idx] = device;
                return *this;
            }

            /// Add local expert → device mapping
            Builder &withLocalExpertDevice(int layer_idx, int expert_idx, DeviceId device)
            {
                local_expert_mappings_[std::to_string(layer_idx) + ":" + std::to_string(expert_idx)] = device;
                return *this;
            }

            /// Enable failure injection
            Builder &withFailure(bool should_fail, const std::string &message = "")
            {
                config_.lookup_should_fail = should_fail;
                if (!message.empty())
                {
                    config_.failure_message = message;
                }
                return *this;
            }

            /// Build the mock
            std::shared_ptr<MockWeightPlacementMap> build()
            {
                auto mock = std::make_shared<MockWeightPlacementMap>(config_);

                // Apply pre-configured mappings
                for (const auto &[name, device] : tensor_mappings_)
                {
                    mock->setTensorDevice(name, device);
                }
                for (const auto &[layer, device] : layer_mappings_)
                {
                    mock->setLayerDevice(layer, device);
                }
                for (const auto &[pattern, device] : pattern_mappings_)
                {
                    mock->setPatternDevice(pattern, device);
                }
                for (const auto &[layer, device] : attention_mappings_)
                {
                    mock->setAttentionDevice(layer, device);
                }
                for (const auto &[layer, device] : ffn_mappings_)
                {
                    mock->setFFNDevice(layer, device);
                }
                for (const auto &[expert, device] : shared_expert_mappings_)
                {
                    mock->setSharedExpertDevice(expert, device);
                }
                for (const auto &[key, device] : local_expert_mappings_)
                {
                    // Parse "layer:expert" key
                    auto pos = key.find(':');
                    int layer = std::stoi(key.substr(0, pos));
                    int expert = std::stoi(key.substr(pos + 1));
                    mock->setLocalExpertDevice(layer, expert, device);
                }

                // Reset call counts after setup
                mock->reset_call_counts();

                return mock;
            }

        private:
            Config config_;
            std::unordered_map<std::string, DeviceId> tensor_mappings_;
            std::unordered_map<int, DeviceId> layer_mappings_;
            std::unordered_map<std::string, DeviceId> pattern_mappings_;
            std::unordered_map<int, DeviceId> attention_mappings_;
            std::unordered_map<int, DeviceId> ffn_mappings_;
            std::unordered_map<int, DeviceId> shared_expert_mappings_;
            std::unordered_map<std::string, DeviceId> local_expert_mappings_;
        };

        // =========================================================================
        // Presets for Common Scenarios
        // =========================================================================

        /**
         * @brief Create a CPU-only placement map
         *
         * All weights on CPU, no GPU placement.
         */
        static std::shared_ptr<MockWeightPlacementMap> cpuOnlyPreset()
        {
            return Builder()
                .withDefaultDevice(DeviceId::cpu())
                .withStrategyName("cpu_only")
                .build();
        }

        /**
         * @brief Create a single-GPU placement map
         *
         * All weights on GPU 0.
         */
        static std::shared_ptr<MockWeightPlacementMap> singleGpuPreset()
        {
            return Builder()
                .withDefaultDevice(DeviceId::cuda(0))
                .withStrategyName("single_gpu")
                .build();
        }

        /**
         * @brief Create a multi-GPU placement map (even layer split)
         *
         * Even layers on GPU 0, odd layers on GPU 1.
         *
         * @param num_layers Total number of layers
         */
        static std::shared_ptr<MockWeightPlacementMap> multiGpuPreset(int num_layers)
        {
            auto builder = Builder()
                               .withDefaultDevice(DeviceId::cuda(0))
                               .withStrategyName("multi_gpu_even_split");

            for (int i = 0; i < num_layers; ++i)
            {
                builder.withLayerDevice(i, DeviceId::cuda(i % 2));
            }

            return builder.build();
        }

        /**
         * @brief Create a CPU offload placement map
         *
         * First N layers on GPU, rest on CPU.
         *
         * @param num_gpu_layers Number of layers to keep on GPU
         * @param total_layers Total number of layers
         */
        static std::shared_ptr<MockWeightPlacementMap> cpuOffloadPreset(int num_gpu_layers, int total_layers)
        {
            auto builder = Builder()
                               .withDefaultDevice(DeviceId::cpu())
                               .withStrategyName("cpu_offload");

            // First N layers on GPU
            for (int i = 0; i < num_gpu_layers && i < total_layers; ++i)
            {
                builder.withLayerDevice(i, DeviceId::cuda(0));
            }

            return builder.build();
        }

        /**
         * @brief Create a replicated placement map
         *
         * All weights replicated (returned as default device, typically for tensor parallelism).
         */
        static std::shared_ptr<MockWeightPlacementMap> replicatedPreset()
        {
            return Builder()
                .withDefaultDevice(DeviceId::cpu())
                .withStrategyName("replicated")
                .build();
        }

        /**
         * @brief Create a column parallel placement map
         *
         * Simulates column-parallel sharding where attention Q/K/V and FFN gate/up
         * are column-sharded.
         */
        static std::shared_ptr<MockWeightPlacementMap> columnParallelPreset()
        {
            return Builder()
                .withDefaultDevice(DeviceId::cuda(0))
                .withStrategyName("column_parallel")
                .withPatternDevice("attn_q", DeviceId::cuda(0))
                .withPatternDevice("attn_k", DeviceId::cuda(0))
                .withPatternDevice("attn_v", DeviceId::cuda(0))
                .withPatternDevice("ffn_gate", DeviceId::cuda(0))
                .withPatternDevice("ffn_up", DeviceId::cuda(0))
                .build();
        }

        /**
         * @brief Create a row parallel placement map
         *
         * Simulates row-parallel sharding where attention output and FFN down
         * are row-sharded.
         */
        static std::shared_ptr<MockWeightPlacementMap> rowParallelPreset()
        {
            return Builder()
                .withDefaultDevice(DeviceId::cuda(0))
                .withStrategyName("row_parallel")
                .withPatternDevice("attn_output", DeviceId::cuda(0))
                .withPatternDevice("ffn_down", DeviceId::cuda(0))
                .build();
        }

        // =========================================================================
        // Constructors
        // =========================================================================

        /**
         * @brief Default constructor (CPU-only mock)
         */
        MockWeightPlacementMap()
            : config_{DeviceId::cpu(), true, "", false, false, ""} {}

        /**
         * @brief Construct mock with configuration
         * @param config Configuration options
         */
        explicit MockWeightPlacementMap(const Config &config)
            : config_(config) {}

        /**
         * @brief Convenience constructor with default device
         * @param default_device Default device for unmapped weights
         */
        explicit MockWeightPlacementMap(DeviceId default_device)
            : config_{default_device, true, "", false, false, ""} {}

        // =========================================================================
        // IWeightPlacementMap Implementation - Core Lookup
        // =========================================================================

        DeviceId getDeviceForWeight(const std::string &tensor_name, int layer_idx = -1) const override
        {
            if (config_.track_calls)
            {
                get_device_calls_.fetch_add(1, std::memory_order_relaxed);
            }
            if (config_.lookup_should_fail)
            {
                throw std::runtime_error(config_.failure_message);
            }

            // Priority 1: Exact tensor name match
            auto tensor_it = tensor_to_device_.find(tensor_name);
            if (tensor_it != tensor_to_device_.end())
            {
                return tensor_it->second;
            }

            // Priority 2: Layer-based lookup
            if (layer_idx >= 0)
            {
                auto layer_it = layer_to_device_.find(layer_idx);
                if (layer_it != layer_to_device_.end())
                {
                    return layer_it->second;
                }
            }

            // Try to extract layer from tensor name if not provided
            if (layer_idx < 0)
            {
                int extracted = extractLayerIndex(tensor_name);
                if (extracted >= 0)
                {
                    auto layer_it = layer_to_device_.find(extracted);
                    if (layer_it != layer_to_device_.end())
                    {
                        return layer_it->second;
                    }
                }
            }

            // Priority 3: Pattern-based match (simple substring matching)
            for (const auto &[pattern, device] : pattern_to_device_)
            {
                if (tensor_name.find(pattern) != std::string::npos)
                {
                    return device;
                }
            }

            // Priority 4: Default device
            return config_.default_device;
        }

        WeightDeviceInfo getDeviceInfoForWeight(const std::string &tensor_name, int layer_idx = -1) const override
        {
            if (config_.track_calls)
            {
                get_device_info_calls_.fetch_add(1, std::memory_order_relaxed);
            }
            if (config_.lookup_should_fail)
            {
                throw std::runtime_error(config_.failure_message);
            }

            DeviceId device = getDeviceForWeight(tensor_name, layer_idx);
            return WeightDeviceInfo(device);
        }

        // =========================================================================
        // IWeightPlacementMap Implementation - Configuration
        // =========================================================================

        void setTensorDevice(const std::string &tensor_name, DeviceId device) override
        {
            if (config_.track_calls)
            {
                set_tensor_calls_.fetch_add(1, std::memory_order_relaxed);
            }
            tensor_to_device_[tensor_name] = device;
        }

        void setLayerDevice(int layer_idx, DeviceId device) override
        {
            if (config_.track_calls)
            {
                set_layer_calls_.fetch_add(1, std::memory_order_relaxed);
            }
            layer_to_device_[layer_idx] = device;
        }

        void setLayerRange(int start_layer, int end_layer, DeviceId device) override
        {
            if (config_.track_calls)
            {
                set_layer_range_calls_.fetch_add(1, std::memory_order_relaxed);
            }
            for (int i = start_layer; i <= end_layer; ++i)
            {
                layer_to_device_[i] = device;
            }
        }

        void setPatternDevice(const std::string &pattern, DeviceId device) override
        {
            if (config_.track_calls)
            {
                set_pattern_calls_.fetch_add(1, std::memory_order_relaxed);
            }
            pattern_to_device_[pattern] = device;
        }

        // =========================================================================
        // IWeightPlacementMap Implementation - Query
        // =========================================================================

        DeviceId defaultDevice() const override
        {
            return config_.default_device;
        }

        size_t tensorMappingCount() const override
        {
            return tensor_to_device_.size();
        }

        size_t layerMappingCount() const override
        {
            return layer_to_device_.size();
        }

        // =========================================================================
        // IWeightPlacementMap Implementation - Block-Level
        // =========================================================================

        void setAttentionDevice(int layer_idx, DeviceId device) override
        {
            if (config_.track_calls)
            {
                set_attention_calls_.fetch_add(1, std::memory_order_relaxed);
            }
            attention_to_device_[layer_idx] = device;
        }

        DeviceId getAttentionDevice(int layer_idx) const override
        {
            if (config_.track_calls)
            {
                get_attention_calls_.fetch_add(1, std::memory_order_relaxed);
            }
            auto it = attention_to_device_.find(layer_idx);
            if (it != attention_to_device_.end())
            {
                return it->second;
            }
            // Fallback to layer mapping or default
            auto layer_it = layer_to_device_.find(layer_idx);
            if (layer_it != layer_to_device_.end())
            {
                return layer_it->second;
            }
            return config_.default_device;
        }

        void setFFNDevice(int layer_idx, DeviceId device) override
        {
            if (config_.track_calls)
            {
                set_ffn_calls_.fetch_add(1, std::memory_order_relaxed);
            }
            ffn_to_device_[layer_idx] = device;
        }

        DeviceId getFFNDevice(int layer_idx) const override
        {
            if (config_.track_calls)
            {
                get_ffn_calls_.fetch_add(1, std::memory_order_relaxed);
            }
            auto it = ffn_to_device_.find(layer_idx);
            if (it != ffn_to_device_.end())
            {
                return it->second;
            }
            // Fallback to layer mapping or default
            auto layer_it = layer_to_device_.find(layer_idx);
            if (layer_it != layer_to_device_.end())
            {
                return layer_it->second;
            }
            return config_.default_device;
        }

        // =========================================================================
        // IWeightPlacementMap Implementation - MoE
        // =========================================================================

        void setSharedExpertDevice(int expert_idx, DeviceId device) override
        {
            if (config_.track_calls)
            {
                set_shared_expert_calls_.fetch_add(1, std::memory_order_relaxed);
            }
            shared_expert_to_device_[expert_idx] = device;
        }

        DeviceId getSharedExpertDevice(int expert_idx) const override
        {
            if (config_.track_calls)
            {
                get_shared_expert_calls_.fetch_add(1, std::memory_order_relaxed);
            }
            auto it = shared_expert_to_device_.find(expert_idx);
            if (it != shared_expert_to_device_.end())
            {
                return it->second;
            }
            return config_.default_device;
        }

        void setLocalExpertDevice(int layer_idx, int expert_idx, DeviceId device) override
        {
            if (config_.track_calls)
            {
                set_local_expert_calls_.fetch_add(1, std::memory_order_relaxed);
            }
            std::string key = std::to_string(layer_idx) + ":" + std::to_string(expert_idx);
            local_expert_to_device_[key] = device;
        }

        DeviceId getLocalExpertDevice(int layer_idx, int expert_idx) const override
        {
            if (config_.track_calls)
            {
                get_local_expert_calls_.fetch_add(1, std::memory_order_relaxed);
            }
            std::string key = std::to_string(layer_idx) + ":" + std::to_string(expert_idx);
            auto it = local_expert_to_device_.find(key);
            if (it != local_expert_to_device_.end())
            {
                return it->second;
            }
            return config_.default_device;
        }

        // =========================================================================
        // IWeightPlacementMap Implementation - PlacementPlan
        // =========================================================================

        void applyPlan(const PlacementPlan & /*plan*/) override
        {
            if (config_.track_calls)
            {
                apply_plan_calls_.fetch_add(1, std::memory_order_relaxed);
            }
            config_.has_plan = true;
            // Mock doesn't actually parse the plan - use Builder to configure
        }

        bool hasPlan() const override
        {
            return config_.has_plan;
        }

        const std::string &appliedStrategyName() const override
        {
            return config_.applied_strategy_name;
        }

        void clear() override
        {
            if (config_.track_calls)
            {
                clear_calls_.fetch_add(1, std::memory_order_relaxed);
            }
            tensor_to_device_.clear();
            layer_to_device_.clear();
            pattern_to_device_.clear();
            attention_to_device_.clear();
            ffn_to_device_.clear();
            shared_expert_to_device_.clear();
            local_expert_to_device_.clear();
            config_.has_plan = false;
            config_.applied_strategy_name.clear();
        }

        // =========================================================================
        // Test Utilities - Call Counting
        // =========================================================================

        /**
         * @brief Get the number of getDeviceForWeight() calls
         */
        size_t getDeviceForWeight_call_count() const
        {
            return get_device_calls_.load(std::memory_order_relaxed);
        }

        /**
         * @brief Get the number of getDeviceInfoForWeight() calls
         */
        size_t getDeviceInfoForWeight_call_count() const
        {
            return get_device_info_calls_.load(std::memory_order_relaxed);
        }

        /**
         * @brief Get the number of setTensorDevice() calls
         */
        size_t setTensorDevice_call_count() const
        {
            return set_tensor_calls_.load(std::memory_order_relaxed);
        }

        /**
         * @brief Get the number of setLayerDevice() calls
         */
        size_t setLayerDevice_call_count() const
        {
            return set_layer_calls_.load(std::memory_order_relaxed);
        }

        /**
         * @brief Get the number of setLayerRange() calls
         */
        size_t setLayerRange_call_count() const
        {
            return set_layer_range_calls_.load(std::memory_order_relaxed);
        }

        /**
         * @brief Get the number of setPatternDevice() calls
         */
        size_t setPatternDevice_call_count() const
        {
            return set_pattern_calls_.load(std::memory_order_relaxed);
        }

        /**
         * @brief Get the number of setAttentionDevice() calls
         */
        size_t setAttentionDevice_call_count() const
        {
            return set_attention_calls_.load(std::memory_order_relaxed);
        }

        /**
         * @brief Get the number of getAttentionDevice() calls
         */
        size_t getAttentionDevice_call_count() const
        {
            return get_attention_calls_.load(std::memory_order_relaxed);
        }

        /**
         * @brief Get the number of setFFNDevice() calls
         */
        size_t setFFNDevice_call_count() const
        {
            return set_ffn_calls_.load(std::memory_order_relaxed);
        }

        /**
         * @brief Get the number of getFFNDevice() calls
         */
        size_t getFFNDevice_call_count() const
        {
            return get_ffn_calls_.load(std::memory_order_relaxed);
        }

        /**
         * @brief Get the number of setSharedExpertDevice() calls
         */
        size_t setSharedExpertDevice_call_count() const
        {
            return set_shared_expert_calls_.load(std::memory_order_relaxed);
        }

        /**
         * @brief Get the number of getSharedExpertDevice() calls
         */
        size_t getSharedExpertDevice_call_count() const
        {
            return get_shared_expert_calls_.load(std::memory_order_relaxed);
        }

        /**
         * @brief Get the number of setLocalExpertDevice() calls
         */
        size_t setLocalExpertDevice_call_count() const
        {
            return set_local_expert_calls_.load(std::memory_order_relaxed);
        }

        /**
         * @brief Get the number of getLocalExpertDevice() calls
         */
        size_t getLocalExpertDevice_call_count() const
        {
            return get_local_expert_calls_.load(std::memory_order_relaxed);
        }

        /**
         * @brief Get the number of applyPlan() calls
         */
        size_t applyPlan_call_count() const
        {
            return apply_plan_calls_.load(std::memory_order_relaxed);
        }

        /**
         * @brief Get the number of clear() calls
         */
        size_t clear_call_count() const
        {
            return clear_calls_.load(std::memory_order_relaxed);
        }

        /**
         * @brief Get total number of lookup calls (getDevice + getDeviceInfo)
         */
        size_t total_lookup_calls() const
        {
            return getDeviceForWeight_call_count() + getDeviceInfoForWeight_call_count();
        }

        /**
         * @brief Get total number of setter calls
         */
        size_t total_setter_calls() const
        {
            return setTensorDevice_call_count() + setLayerDevice_call_count() +
                   setLayerRange_call_count() + setPatternDevice_call_count() +
                   setAttentionDevice_call_count() + setFFNDevice_call_count() +
                   setSharedExpertDevice_call_count() + setLocalExpertDevice_call_count();
        }

        /**
         * @brief Reset all call counters
         */
        void reset_call_counts()
        {
            get_device_calls_.store(0, std::memory_order_relaxed);
            get_device_info_calls_.store(0, std::memory_order_relaxed);
            set_tensor_calls_.store(0, std::memory_order_relaxed);
            set_layer_calls_.store(0, std::memory_order_relaxed);
            set_layer_range_calls_.store(0, std::memory_order_relaxed);
            set_pattern_calls_.store(0, std::memory_order_relaxed);
            set_attention_calls_.store(0, std::memory_order_relaxed);
            get_attention_calls_.store(0, std::memory_order_relaxed);
            set_ffn_calls_.store(0, std::memory_order_relaxed);
            get_ffn_calls_.store(0, std::memory_order_relaxed);
            set_shared_expert_calls_.store(0, std::memory_order_relaxed);
            get_shared_expert_calls_.store(0, std::memory_order_relaxed);
            set_local_expert_calls_.store(0, std::memory_order_relaxed);
            get_local_expert_calls_.store(0, std::memory_order_relaxed);
            apply_plan_calls_.store(0, std::memory_order_relaxed);
            clear_calls_.store(0, std::memory_order_relaxed);
        }

        // =========================================================================
        // Test Utilities - Configuration Modification
        // =========================================================================

        /**
         * @brief Set default device at runtime
         */
        void set_default_device(DeviceId device)
        {
            config_.default_device = device;
        }

        /**
         * @brief Enable/disable lookup failure injection
         */
        void set_lookup_fails(bool fails)
        {
            config_.lookup_should_fail = fails;
        }

        /**
         * @brief Set failure message
         */
        void set_failure_message(const std::string &message)
        {
            config_.failure_message = message;
        }

        /**
         * @brief Get current configuration (read-only)
         */
        const Config &config() const { return config_; }

    private:
        /**
         * @brief Try to extract layer index from tensor name
         */
        int extractLayerIndex(const std::string &tensor_name) const
        {
            // Match patterns like "blk.5.attn_q" or "layer5_attn"
            size_t pos = tensor_name.find("blk.");
            if (pos != std::string::npos)
            {
                size_t start = pos + 4;
                size_t end = tensor_name.find('.', start);
                if (end != std::string::npos)
                {
                    try
                    {
                        return std::stoi(tensor_name.substr(start, end - start));
                    }
                    catch (...)
                    {
                        return -1;
                    }
                }
            }
            return -1;
        }

        Config config_;

        // Mapping storage
        std::unordered_map<std::string, DeviceId> tensor_to_device_;
        std::unordered_map<int, DeviceId> layer_to_device_;
        std::unordered_map<std::string, DeviceId> pattern_to_device_;
        std::unordered_map<int, DeviceId> attention_to_device_;
        std::unordered_map<int, DeviceId> ffn_to_device_;
        std::unordered_map<int, DeviceId> shared_expert_to_device_;
        std::unordered_map<std::string, DeviceId> local_expert_to_device_;

        // Atomic call counters (thread-safe)
        mutable std::atomic<size_t> get_device_calls_{0};
        mutable std::atomic<size_t> get_device_info_calls_{0};
        mutable std::atomic<size_t> set_tensor_calls_{0};
        mutable std::atomic<size_t> set_layer_calls_{0};
        mutable std::atomic<size_t> set_layer_range_calls_{0};
        mutable std::atomic<size_t> set_pattern_calls_{0};
        mutable std::atomic<size_t> set_attention_calls_{0};
        mutable std::atomic<size_t> get_attention_calls_{0};
        mutable std::atomic<size_t> set_ffn_calls_{0};
        mutable std::atomic<size_t> get_ffn_calls_{0};
        mutable std::atomic<size_t> set_shared_expert_calls_{0};
        mutable std::atomic<size_t> get_shared_expert_calls_{0};
        mutable std::atomic<size_t> set_local_expert_calls_{0};
        mutable std::atomic<size_t> get_local_expert_calls_{0};
        mutable std::atomic<size_t> apply_plan_calls_{0};
        mutable std::atomic<size_t> clear_calls_{0};
    };

} // namespace llaminar2::test
