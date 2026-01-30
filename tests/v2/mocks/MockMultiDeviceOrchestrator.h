/**
 * @file MockMultiDeviceOrchestrator.h
 * @brief Mock implementation of IMultiDeviceOrchestrator for unit testing
 *
 * This mock enables:
 * - Testing code that depends on IMultiDeviceOrchestrator without real devices
 * - Configuring mock per-device runners
 * - Failure injection for robustness testing
 * - Call tracking for behavior verification
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "execution/local_execution/orchestrators/IMultiDeviceOrchestrator.h"
#include "collective/ILocalTPContext.h"
#include <vector>
#include <memory>
#include <functional>
#include <atomic>
#include <stdexcept>
#include <string>
#include <sstream>
#include <cstring>

namespace llaminar2::test
{

    /**
     * @brief Mock inference runner for use as device runners within MockMultiDeviceOrchestrator
     *
     * Lightweight mock that implements IInferenceRunner for testing device runner access.
     * Can be configured with mock logits and failure modes.
     */
    class MockDeviceRunner : public IInferenceRunner
    {
    public:
        /**
         * @brief Configuration for mock device runner
         */
        struct Config
        {
            int device_idx = 0;                      ///< Device index in orchestrator
            int vocab_size = 32000;                  ///< Vocabulary size
            int position = 0;                        ///< Current cache position
            std::vector<float> mock_logits;          ///< Mock logits to return
            bool forward_should_fail = false;        ///< If true, forward() returns false
            std::string architecture = "mock_qwen2"; ///< Architecture name

            Config() = default;
        };

        MockDeviceRunner() : MockDeviceRunner(Config{}) {}

        explicit MockDeviceRunner(const Config &config)
            : config_(config)
        {
            if (config_.mock_logits.empty())
            {
                config_.mock_logits.resize(config_.vocab_size, 0.0f);
            }
        }

        // =====================================================================
        // IInferenceRunner Implementation
        // =====================================================================

        bool forward(const int *tokens, int seq_len) override
        {
            forward_calls_.fetch_add(1, std::memory_order_relaxed);
            last_tokens_.assign(tokens, tokens + seq_len);
            last_seq_len_ = seq_len;

            if (config_.forward_should_fail)
            {
                return false;
            }

            config_.position += seq_len;
            return true;
        }

        const float *logits() const override
        {
            return config_.mock_logits.data();
        }

        int vocab_size() const override
        {
            return config_.vocab_size;
        }

        void clear_cache() override
        {
            clear_cache_calls_.fetch_add(1, std::memory_order_relaxed);
            config_.position = 0;
        }

        int get_position() const override
        {
            return config_.position;
        }

        ExecutionPath executionPath() const override
        {
            return ExecutionPath::GRAPH;
        }

        const char *architecture() const override
        {
            return config_.architecture.c_str();
        }

        // =====================================================================
        // Test Utilities - Call Tracking
        // =====================================================================

        size_t forward_call_count() const
        {
            return forward_calls_.load(std::memory_order_relaxed);
        }

        size_t clear_cache_call_count() const
        {
            return clear_cache_calls_.load(std::memory_order_relaxed);
        }

        const std::vector<int> &last_tokens() const { return last_tokens_; }
        int last_seq_len() const { return last_seq_len_; }

        void reset_call_counts()
        {
            forward_calls_.store(0, std::memory_order_relaxed);
            clear_cache_calls_.store(0, std::memory_order_relaxed);
        }

        // =====================================================================
        // Test Utilities - Configuration Modification
        // =====================================================================

        void set_mock_logits(const std::vector<float> &logits) { config_.mock_logits = logits; }
        void set_forward_fails(bool fails) { config_.forward_should_fail = fails; }
        void set_position(int pos) { config_.position = pos; }

        const Config &config() const { return config_; }

    private:
        Config config_;
        mutable std::atomic<size_t> forward_calls_{0};
        mutable std::atomic<size_t> clear_cache_calls_{0};
        std::vector<int> last_tokens_;
        int last_seq_len_ = 0;
    };

    /**
     * @brief Mock LOCAL TP context for testing multi-device coordination
     *
     * Lightweight mock that implements ILocalTPContext with no-op collective operations.
     */
    class MockLocalTPContext : public ILocalTPContext
    {
    public:
        struct Config
        {
            std::vector<GlobalDeviceAddress> devices;
            std::vector<float> weights;
            CollectiveBackendType backend = CollectiveBackendType::HOST;
            bool allreduce_should_fail = false;
            bool allgather_should_fail = false;

            Config() = default;
        };

        MockLocalTPContext() : MockLocalTPContext(Config{}) {}

        explicit MockLocalTPContext(const Config &config)
            : config_(config)
        {
            // Default to single CPU device if none specified
            if (config_.devices.empty())
            {
                config_.devices.push_back(GlobalDeviceAddress::cpu());
            }
            // Default to equal weights
            if (config_.weights.empty())
            {
                config_.weights.resize(config_.devices.size(), 1.0f / config_.devices.size());
            }
        }

        // =====================================================================
        // ILocalTPContext Implementation - Configuration
        // =====================================================================

        const std::vector<GlobalDeviceAddress> &devices() const override
        {
            return config_.devices;
        }

        const std::vector<float> &weights() const override
        {
            return config_.weights;
        }

        CollectiveBackendType backend() const override
        {
            return config_.backend;
        }

        int degree() const override
        {
            return static_cast<int>(config_.devices.size());
        }

        // =====================================================================
        // ILocalTPContext Implementation - Collective Operations
        // =====================================================================

        bool allreduce(TensorBase * /*tensor*/) override
        {
            allreduce_calls_.fetch_add(1, std::memory_order_relaxed);
            return !config_.allreduce_should_fail;
        }

        bool allreduce(TensorBase *tensor, const std::string & /*stage_name*/) override
        {
            return allreduce(tensor);
        }

        bool allreduce(const TensorBase * /*input*/, TensorBase * /*output*/) override
        {
            allreduce_calls_.fetch_add(1, std::memory_order_relaxed);
            return !config_.allreduce_should_fail;
        }

        bool allgather(const TensorBase * /*local_shard*/, TensorBase * /*global_tensor*/) override
        {
            allgather_calls_.fetch_add(1, std::memory_order_relaxed);
            return !config_.allgather_should_fail;
        }

        bool gatherFromDevices(
            const std::vector<const TensorBase *> &shards,
            TensorBase *output) override
        {
            gather_from_devices_calls_.fetch_add(1, std::memory_order_relaxed);

            // Simple mock implementation: copy data from shards to output
            if (shards.empty() || !output)
            {
                return false;
            }

            float *dst = output->mutable_data();
            size_t offset = 0;
            for (const auto *shard : shards)
            {
                if (shard)
                {
                    const float *src = shard->data();
                    size_t count = shard->numel();
                    std::memcpy(dst + offset, src, count * sizeof(float));
                    offset += count;
                }
            }
            return !config_.allgather_should_fail;
        }

        bool reduceScatter(const TensorBase * /*input*/, TensorBase * /*output_shard*/) override
        {
            reduce_scatter_calls_.fetch_add(1, std::memory_order_relaxed);
            return true;
        }

        // =====================================================================
        // ILocalTPContext Implementation - Synchronization
        // =====================================================================

        void synchronize() override
        {
            synchronize_calls_.fetch_add(1, std::memory_order_relaxed);
        }

        // =====================================================================
        // ILocalTPContext Implementation - Device Management
        // =====================================================================

        int indexForDevice(const GlobalDeviceAddress &device) const override
        {
            for (size_t i = 0; i < config_.devices.size(); ++i)
            {
                if (config_.devices[i] == device)
                {
                    return static_cast<int>(i);
                }
            }
            return -1;
        }

        const GlobalDeviceAddress &deviceAt(int index) const override
        {
            if (index < 0 || index >= static_cast<int>(config_.devices.size()))
            {
                throw std::out_of_range("MockLocalTPContext::deviceAt: index out of range");
            }
            return config_.devices[index];
        }

        float weightForDevice(const GlobalDeviceAddress &device) const override
        {
            int idx = indexForDevice(device);
            return (idx >= 0) ? config_.weights[idx] : 0.0f;
        }

        // =====================================================================
        // ILocalTPContext Implementation - Sharding Utilities
        // =====================================================================

        int headsForDevice(const GlobalDeviceAddress &device, int total_heads) const override
        {
            float w = weightForDevice(device);
            return static_cast<int>(w * total_heads + 0.5f);
        }

        std::pair<int, int> rowRangeForDevice(
            const GlobalDeviceAddress &device, int total_rows) const override
        {
            int idx = indexForDevice(device);
            if (idx < 0)
                return {0, 0};

            float cumulative = 0.0f;
            for (int i = 0; i < idx; ++i)
            {
                cumulative += config_.weights[i];
            }
            int start = static_cast<int>(cumulative * total_rows);
            int end = static_cast<int>((cumulative + config_.weights[idx]) * total_rows);
            return {start, end};
        }

        std::pair<int, int> colRangeForDevice(
            const GlobalDeviceAddress &device, int total_cols) const override
        {
            return rowRangeForDevice(device, total_cols);
        }

        // =====================================================================
        // ILocalTPContext Implementation - BAR Registry (no-ops for tests)
        // =====================================================================

        void registerBARBackedOutput(
            const std::string & /*stage_name*/,
            const GlobalDeviceAddress & /*device*/,
            TensorBase * /*tensor*/) override
        {
            // No-op for unit tests
        }

        bool hasBARBackedOutputs(const std::string & /*stage_name*/) const override { return false; }
        void clearBARBackedOutputs() override {}
        std::shared_ptr<DirectP2PEngine> getDirectP2PEngine() const override { return nullptr; }
        bool reserveTempBufferBytes(size_t /*bytes*/) override { return true; }

        // =====================================================================
        // Test Utilities - Call Tracking
        // =====================================================================

        size_t allreduce_call_count() const
        {
            return allreduce_calls_.load(std::memory_order_relaxed);
        }

        size_t allgather_call_count() const
        {
            return allgather_calls_.load(std::memory_order_relaxed);
        }

        size_t gather_from_devices_call_count() const
        {
            return gather_from_devices_calls_.load(std::memory_order_relaxed);
        }

        size_t reduce_scatter_call_count() const
        {
            return reduce_scatter_calls_.load(std::memory_order_relaxed);
        }

        size_t synchronize_call_count() const
        {
            return synchronize_calls_.load(std::memory_order_relaxed);
        }

        void reset_call_counts()
        {
            allreduce_calls_.store(0, std::memory_order_relaxed);
            allgather_calls_.store(0, std::memory_order_relaxed);
            gather_from_devices_calls_.store(0, std::memory_order_relaxed);
            reduce_scatter_calls_.store(0, std::memory_order_relaxed);
            synchronize_calls_.store(0, std::memory_order_relaxed);
        }

        // =====================================================================
        // Test Utilities - Configuration Modification
        // =====================================================================

        void set_allreduce_fails(bool fails) { config_.allreduce_should_fail = fails; }
        void set_allgather_fails(bool fails) { config_.allgather_should_fail = fails; }

        const Config &config() const { return config_; }

    private:
        Config config_;
        mutable std::atomic<size_t> allreduce_calls_{0};
        mutable std::atomic<size_t> allgather_calls_{0};
        mutable std::atomic<size_t> gather_from_devices_calls_{0};
        mutable std::atomic<size_t> reduce_scatter_calls_{0};
        mutable std::atomic<size_t> synchronize_calls_{0};
    };

    /**
     * @brief Mock multi-device orchestrator for unit testing
     *
     * Provides configurable behavior for testing multi-device logic:
     * - Configurable number of mock device runners
     * - Mock logits for each device
     * - Call tracking for verification
     * - Optional failure injection
     *
     * Usage:
     * @code
     * // Simple construction with Builder
     * auto mock = MockMultiDeviceOrchestrator::Builder()
     *     .withDeviceCount(2)
     *     .withVocabSize(32000)
     *     .withMockLogits({1.0f, 2.0f, 3.0f})  // Same for all devices
     *     .build();
     *
     * // Use as IMultiDeviceOrchestrator
     * mock->forward(tokens, seq_len);
     *
     * // Verify behavior
     * EXPECT_EQ(mock->forward_call_count(), 1);
     * EXPECT_EQ(mock->device_count(), 2);
     *
     * // Access device runners
     * auto* runner0 = mock->deviceRunner(0);
     * EXPECT_EQ(runner0->forward_call_count(), 1);
     * @endcode
     */
    class MockMultiDeviceOrchestrator : public IMultiDeviceOrchestrator
    {
    public:
        // =====================================================================
        // Configuration
        // =====================================================================

        /**
         * @brief Configuration options for the mock
         */
        struct Config
        {
            int device_count = 1;                    ///< Number of mock devices
            int vocab_size = 32000;                  ///< Vocabulary size
            std::string architecture = "mock_qwen2"; ///< Architecture name
            std::vector<float> mock_logits;          ///< Mock logits (shared or per-device)
            bool track_calls = true;                 ///< Record call counts for verification

            // Failure injection
            bool forward_should_fail = false;   ///< If true, forward() returns false
            bool allreduce_should_fail = false; ///< If true, LOCAL TP allreduce fails

            // LOCAL TP configuration
            CollectiveBackendType tp_backend = CollectiveBackendType::HOST;
            std::vector<float> tp_weights; ///< TP weights (empty = equal distribution)

            Config() = default;
        };

        /**
         * @brief Builder pattern for MockMultiDeviceOrchestrator configuration
         */
        class Builder
        {
        public:
            Builder() = default;

            /// Set the number of mock devices
            Builder &withDeviceCount(int count)
            {
                config_.device_count = count;
                return *this;
            }

            /// Set the vocabulary size
            Builder &withVocabSize(int vocab_size)
            {
                config_.vocab_size = vocab_size;
                return *this;
            }

            /// Set the architecture name
            Builder &withArchitecture(const std::string &arch)
            {
                config_.architecture = arch;
                return *this;
            }

            /// Set mock logits (applied to all devices)
            Builder &withMockLogits(std::vector<float> logits)
            {
                config_.mock_logits = std::move(logits);
                return *this;
            }

            /// Enable/disable call tracking
            Builder &withCallTracking(bool enabled)
            {
                config_.track_calls = enabled;
                return *this;
            }

            /// Enable forward failure injection
            Builder &withForwardFails(bool fails)
            {
                config_.forward_should_fail = fails;
                return *this;
            }

            /// Enable allreduce failure injection
            Builder &withAllreduceFails(bool fails)
            {
                config_.allreduce_should_fail = fails;
                return *this;
            }

            /// Set LOCAL TP backend type
            Builder &withTPBackend(CollectiveBackendType backend)
            {
                config_.tp_backend = backend;
                return *this;
            }

            /// Set LOCAL TP weights (for proportional distribution)
            Builder &withTPWeights(std::vector<float> weights)
            {
                config_.tp_weights = std::move(weights);
                return *this;
            }

            /// Build the mock orchestrator
            std::unique_ptr<MockMultiDeviceOrchestrator> build()
            {
                return std::make_unique<MockMultiDeviceOrchestrator>(config_);
            }

            /// Build as shared_ptr
            std::shared_ptr<MockMultiDeviceOrchestrator> buildShared()
            {
                return std::make_shared<MockMultiDeviceOrchestrator>(config_);
            }

        private:
            Config config_;
        };

        // =====================================================================
        // Presets
        // =====================================================================

        /**
         * @brief Create a single-device mock (no TP)
         */
        static std::unique_ptr<MockMultiDeviceOrchestrator> singleDevice()
        {
            return Builder()
                .withDeviceCount(1)
                .build();
        }

        /**
         * @brief Create a multi-device mock with equal weights
         * @param device_count Number of devices
         */
        static std::unique_ptr<MockMultiDeviceOrchestrator> multiDevice(int device_count)
        {
            return Builder()
                .withDeviceCount(device_count)
                .build();
        }

        /**
         * @brief Create a mock that simulates forward failure
         */
        static std::unique_ptr<MockMultiDeviceOrchestrator> failingForward()
        {
            return Builder()
                .withDeviceCount(2)
                .withForwardFails(true)
                .build();
        }

        /**
         * @brief Create a mock that simulates collective failure
         */
        static std::unique_ptr<MockMultiDeviceOrchestrator> failingCollectives()
        {
            return Builder()
                .withDeviceCount(2)
                .withAllreduceFails(true)
                .build();
        }

        // =====================================================================
        // Constructors
        // =====================================================================

        /**
         * @brief Default constructor (single device)
         */
        MockMultiDeviceOrchestrator()
            : MockMultiDeviceOrchestrator(Config{}) {}

        /**
         * @brief Construct with configuration
         * @param config Configuration options
         */
        explicit MockMultiDeviceOrchestrator(const Config &config)
            : config_(config)
        {
            // Validate configuration
            if (config_.device_count < 1)
            {
                throw std::invalid_argument(
                    "MockMultiDeviceOrchestrator: device_count must be >= 1. Got " +
                    std::to_string(config_.device_count));
            }

            // Create device runners
            for (int i = 0; i < config_.device_count; ++i)
            {
                MockDeviceRunner::Config runner_config;
                runner_config.device_idx = i;
                runner_config.vocab_size = config_.vocab_size;
                runner_config.architecture = config_.architecture;
                runner_config.forward_should_fail = config_.forward_should_fail;
                if (!config_.mock_logits.empty())
                {
                    runner_config.mock_logits = config_.mock_logits;
                }
                device_runners_.push_back(std::make_unique<MockDeviceRunner>(runner_config));
            }

            // Create LOCAL TP context if multi-device
            if (config_.device_count > 1)
            {
                MockLocalTPContext::Config tp_config;
                for (int i = 0; i < config_.device_count; ++i)
                {
                    tp_config.devices.push_back(GlobalDeviceAddress::cpu());
                }
                tp_config.weights = config_.tp_weights;
                tp_config.backend = config_.tp_backend;
                tp_config.allreduce_should_fail = config_.allreduce_should_fail;
                tp_context_ = std::make_unique<MockLocalTPContext>(tp_config);
            }

            // Initialize aggregated logits
            if (!config_.mock_logits.empty())
            {
                aggregated_logits_ = config_.mock_logits;
            }
            else
            {
                aggregated_logits_.resize(config_.vocab_size, 0.0f);
            }
        }

        // =====================================================================
        // IInferenceRunner Implementation
        // =====================================================================

        bool forward(const int *tokens, int seq_len) override
        {
            if (config_.track_calls)
            {
                forward_calls_.fetch_add(1, std::memory_order_relaxed);
            }

            if (config_.forward_should_fail)
            {
                return false;
            }

            // Forward to all device runners
            bool all_success = true;
            for (auto &runner : device_runners_)
            {
                if (!runner->forward(tokens, seq_len))
                {
                    all_success = false;
                }
            }

            position_ += seq_len;
            return all_success;
        }

        const float *logits() const override
        {
            return aggregated_logits_.data();
        }

        int vocab_size() const override
        {
            return config_.vocab_size;
        }

        void clear_cache() override
        {
            if (config_.track_calls)
            {
                clear_cache_calls_.fetch_add(1, std::memory_order_relaxed);
            }

            for (auto &runner : device_runners_)
            {
                runner->clear_cache();
            }
            position_ = 0;
        }

        int get_position() const override
        {
            return position_;
        }

        ExecutionPath executionPath() const override
        {
            return ExecutionPath::GRAPH;
        }

        const char *architecture() const override
        {
            return config_.architecture.c_str();
        }

        // =====================================================================
        // IMultiDeviceOrchestrator Implementation
        // =====================================================================

        int device_count() const override
        {
            return config_.device_count;
        }

        IInferenceRunner *deviceRunner(int device_idx) override
        {
            if (device_idx < 0 || device_idx >= static_cast<int>(device_runners_.size()))
            {
                throw std::out_of_range(
                    "MockMultiDeviceOrchestrator::deviceRunner: device_idx out of range. Got " +
                    std::to_string(device_idx) + ", device_count=" + std::to_string(device_count()));
            }
            return device_runners_[device_idx].get();
        }

        const IInferenceRunner *deviceRunner(int device_idx) const override
        {
            if (device_idx < 0 || device_idx >= static_cast<int>(device_runners_.size()))
            {
                throw std::out_of_range(
                    "MockMultiDeviceOrchestrator::deviceRunner: device_idx out of range. Got " +
                    std::to_string(device_idx) + ", device_count=" + std::to_string(device_count()));
            }
            return device_runners_[device_idx].get();
        }

        ILocalTPContext *localTPContext() override
        {
            return tp_context_.get();
        }

        const ILocalTPContext *localTPContext() const override
        {
            return tp_context_.get();
        }

        bool allDevicesReady() const override
        {
            return all_devices_ready_;
        }

        void synchronizeDevices() override
        {
            if (config_.track_calls)
            {
                synchronize_calls_.fetch_add(1, std::memory_order_relaxed);
            }
            if (tp_context_)
            {
                tp_context_->synchronize();
            }
        }

        // =====================================================================
        // Test Utilities - Call Tracking
        // =====================================================================

        /**
         * @brief Get the number of forward() calls
         */
        size_t forward_call_count() const
        {
            return forward_calls_.load(std::memory_order_relaxed);
        }

        /**
         * @brief Get the number of clear_cache() calls
         */
        size_t clear_cache_call_count() const
        {
            return clear_cache_calls_.load(std::memory_order_relaxed);
        }

        /**
         * @brief Get the number of synchronizeDevices() calls
         */
        size_t synchronize_call_count() const
        {
            return synchronize_calls_.load(std::memory_order_relaxed);
        }

        /**
         * @brief Reset all call counters (including device runners)
         */
        void reset_call_counts()
        {
            forward_calls_.store(0, std::memory_order_relaxed);
            clear_cache_calls_.store(0, std::memory_order_relaxed);
            synchronize_calls_.store(0, std::memory_order_relaxed);

            for (auto &runner : device_runners_)
            {
                runner->reset_call_counts();
            }

            if (tp_context_)
            {
                tp_context_->reset_call_counts();
            }
        }

        // =====================================================================
        // Test Utilities - Configuration Modification
        // =====================================================================

        /**
         * @brief Set mock logits (updates all devices)
         */
        void set_mock_logits(const std::vector<float> &logits)
        {
            aggregated_logits_ = logits;
            for (auto &runner : device_runners_)
            {
                runner->set_mock_logits(logits);
            }
        }

        /**
         * @brief Enable/disable forward failure injection
         */
        void set_forward_fails(bool fails)
        {
            config_.forward_should_fail = fails;
            for (auto &runner : device_runners_)
            {
                runner->set_forward_fails(fails);
            }
        }

        /**
         * @brief Set all devices ready state
         */
        void set_all_devices_ready(bool ready) { all_devices_ready_ = ready; }

        /**
         * @brief Get typed mock device runner for detailed test configuration
         * @param device_idx Device index
         * @return Pointer to MockDeviceRunner
         */
        MockDeviceRunner *mockDeviceRunner(int device_idx)
        {
            if (device_idx < 0 || device_idx >= static_cast<int>(device_runners_.size()))
            {
                return nullptr;
            }
            return device_runners_[device_idx].get();
        }

        /**
         * @brief Get typed mock LOCAL TP context for detailed test configuration
         * @return Pointer to MockLocalTPContext (may be nullptr for single device)
         */
        MockLocalTPContext *mockLocalTPContext()
        {
            return tp_context_.get();
        }

        /**
         * @brief Get current configuration (read-only)
         */
        const Config &config() const { return config_; }

        /**
         * @brief Get description string for debugging
         */
        std::string description() const
        {
            std::ostringstream oss;
            oss << "MockMultiDeviceOrchestrator{devices=" << config_.device_count
                << ", vocab_size=" << config_.vocab_size
                << ", forward_calls=" << forward_call_count()
                << ", position=" << position_ << "}";
            return oss.str();
        }

        // =====================================================================
        // Test Utilities - Inject Custom Runners
        // =====================================================================

        /**
         * @brief Inject a custom mock device runner
         *
         * Allows injecting pre-configured MockDeviceRunner instances for
         * fine-grained test control.
         *
         * @param device_idx Device index to replace
         * @param runner Mock device runner to inject (takes ownership)
         */
        void injectDeviceRunner(int device_idx, std::unique_ptr<MockDeviceRunner> runner)
        {
            if (device_idx >= 0 && device_idx < static_cast<int>(device_runners_.size()))
            {
                device_runners_[device_idx] = std::move(runner);
            }
        }

        /**
         * @brief Inject a custom mock LOCAL TP context
         *
         * @param context Mock LOCAL TP context to inject (takes ownership)
         */
        void injectLocalTPContext(std::unique_ptr<MockLocalTPContext> context)
        {
            tp_context_ = std::move(context);
        }

    private:
        Config config_;
        std::vector<std::unique_ptr<MockDeviceRunner>> device_runners_;
        std::unique_ptr<MockLocalTPContext> tp_context_;
        std::vector<float> aggregated_logits_;
        int position_ = 0;
        bool all_devices_ready_ = true;

        // Atomic call counters (thread-safe)
        mutable std::atomic<size_t> forward_calls_{0};
        mutable std::atomic<size_t> clear_cache_calls_{0};
        mutable std::atomic<size_t> synchronize_calls_{0};
    };

} // namespace llaminar2::test
