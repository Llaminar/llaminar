/**
 * @file MockLocalPPContext.h
 * @brief Mock implementation of ILocalPPContext for unit testing
 *
 * This mock enables:
 * - Testing code that depends on ILocalPPContext without real devices
 * - Configuring mock stage devices and layer boundaries
 * - Failure injection for robustness testing
 * - Comprehensive call tracking for behavior verification
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "collective/ILocalPPContext.h"
#include <atomic>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace llaminar2::test
{

    /**
     * @brief Record of a single transfer() or transferAsync() call
     *
     * Captures all parameters passed to the transfer method for verification.
     */
    struct TransferCall
    {
        TensorBase *activations = nullptr; ///< Tensor that was transferred
        int stage_from = -1;               ///< Source stage index
        int stage_to = -1;                 ///< Destination stage index
        bool was_async = false;            ///< true if called via transferAsync()
        void *stream = nullptr;            ///< Stream handle (for async calls only)

        TransferCall() = default;

        TransferCall(TensorBase *act, int from, int to, bool async = false, void *strm = nullptr)
            : activations(act), stage_from(from), stage_to(to), was_async(async), stream(strm)
        {
        }

        /**
         * @brief Check if this call matches expected parameters
         */
        bool matches(int expected_from, int expected_to) const
        {
            return stage_from == expected_from && stage_to == expected_to;
        }

        bool matches(TensorBase *expected_tensor, int expected_from, int expected_to) const
        {
            return activations == expected_tensor &&
                   stage_from == expected_from &&
                   stage_to == expected_to;
        }
    };

    /**
     * @brief Mock LOCAL PP context for testing pipeline parallelism
     *
     * Lightweight mock that implements ILocalPPContext with no-op transfer operations.
     * Tracks all method calls for verification and allows configurable return values.
     *
     * Usage:
     * @code
     * // Create mock with 3 stages
     * MockLocalPPContext::Config config;
     * config.stage_devices = {GlobalDeviceAddress::cuda(0),
     *                         GlobalDeviceAddress::cuda(1),
     *                         GlobalDeviceAddress::rocm(0)};
     * config.layer_boundaries = {0, 8, 16, 24};
     *
     * MockLocalPPContext mock(config);
     *
     * // Use the mock
     * mock.transfer(tensor, 0, 1);
     *
     * // Verify
     * EXPECT_EQ(mock.transferCallCount(), 1);
     * EXPECT_EQ(mock.transferCalls()[0].stage_from, 0);
     * EXPECT_EQ(mock.transferCalls()[0].stage_to, 1);
     * @endcode
     *
     * Thread-safe: All public methods are thread-safe.
     */
    class MockLocalPPContext : public ILocalPPContext
    {
    public:
        /**
         * @brief Configuration for MockLocalPPContext
         */
        struct Config
        {
            /// Device assignment for each PP stage
            std::vector<GlobalDeviceAddress> stage_devices;

            /// Layer boundaries (size = stage_devices.size() + 1)
            std::vector<int> layer_boundaries;

            /// Default backend for transfers
            CollectiveBackendType default_backend = CollectiveBackendType::HOST;

            /// If true, transfer() returns false (failure injection)
            bool transfer_should_fail = false;

            /// If true, transferAsync() returns false (failure injection)
            bool transfer_async_should_fail = false;

            Config() = default;
        };

        // =====================================================================
        // Constructors
        // =====================================================================

        /**
         * @brief Default constructor - creates single-stage CPU PP context
         */
        MockLocalPPContext()
            : MockLocalPPContext(Config{})
        {
        }

        /**
         * @brief Construct mock with explicit configuration
         * @param config Configuration specifying devices and layer boundaries
         */
        explicit MockLocalPPContext(const Config &config)
            : config_(config)
        {
            // Default to single CPU device if none specified
            if (config_.stage_devices.empty())
            {
                config_.stage_devices.push_back(GlobalDeviceAddress::cpu());
            }

            // Default to all layers on stage 0 if boundaries not specified
            if (config_.layer_boundaries.empty())
            {
                // Assume 24 layers as default for single-stage
                config_.layer_boundaries = {0, 24};
            }

            // Validate configuration
            if (config_.layer_boundaries.size() != config_.stage_devices.size() + 1)
            {
                throw std::invalid_argument(
                    "MockLocalPPContext: layer_boundaries.size() must equal stage_devices.size() + 1");
            }
        }

        // =====================================================================
        // ILocalPPContext Implementation - Configuration
        // =====================================================================

        int numStages() const override
        {
            return static_cast<int>(config_.stage_devices.size());
        }

        const GlobalDeviceAddress &deviceForStage(int stage) const override
        {
            if (stage < 0 || stage >= numStages())
            {
                throw std::out_of_range("MockLocalPPContext::deviceForStage: stage out of range");
            }
            return config_.stage_devices[stage];
        }

        CollectiveBackendType backendForTransfer(int /*stage_from*/, int /*stage_to*/) const override
        {
            return config_.default_backend;
        }

        std::pair<int, int> layerRangeForStage(int stage) const override
        {
            if (stage < 0 || stage >= numStages())
            {
                return {-1, -1};
            }
            return {config_.layer_boundaries[stage], config_.layer_boundaries[stage + 1]};
        }

        int stageForLayer(int layer) const override
        {
            for (int s = 0; s < numStages(); ++s)
            {
                if (layer >= config_.layer_boundaries[s] &&
                    layer < config_.layer_boundaries[s + 1])
                {
                    return s;
                }
            }
            return -1;
        }

        const std::vector<GlobalDeviceAddress> &stageDevices() const override
        {
            return config_.stage_devices;
        }

        const std::vector<int> &layerBoundaries() const override
        {
            return config_.layer_boundaries;
        }

        // =====================================================================
        // ILocalPPContext Implementation - Transfer Operations
        // =====================================================================

        bool transfer(TensorBase *activations, int stage_from, int stage_to,
                      size_t active_bytes = 0) override
        {
            (void)active_bytes;
            // Track the call
            {
                std::lock_guard<std::mutex> lock(calls_mutex_);
                transfer_calls_.emplace_back(activations, stage_from, stage_to, /*async=*/false);
            }
            transfer_call_count_.fetch_add(1, std::memory_order_relaxed);

            return !config_.transfer_should_fail;
        }

        bool transferAsync(TensorBase *activations, int stage_from, int stage_to, void *stream) override
        {
            // Track the call
            {
                std::lock_guard<std::mutex> lock(calls_mutex_);
                transfer_calls_.emplace_back(activations, stage_from, stage_to, /*async=*/true, stream);
            }
            transfer_async_call_count_.fetch_add(1, std::memory_order_relaxed);

            return !config_.transfer_async_should_fail;
        }

        // =====================================================================
        // ILocalPPContext Implementation - Synchronization
        // =====================================================================

        void synchronize() override
        {
            synchronize_call_count_.fetch_add(1, std::memory_order_relaxed);
        }

        void synchronizeStream(void * /*stream*/) override
        {
            synchronize_stream_call_count_.fetch_add(1, std::memory_order_relaxed);
        }

        // =====================================================================
        // ILocalPPContext Implementation - Utility
        // =====================================================================

        bool sameDevice(int stage_a, int stage_b) const override
        {
            if (stage_a < 0 || stage_a >= numStages() ||
                stage_b < 0 || stage_b >= numStages())
            {
                return false;
            }
            return config_.stage_devices[stage_a] == config_.stage_devices[stage_b];
        }

        int totalLayers() const override
        {
            if (config_.layer_boundaries.empty())
            {
                return 0;
            }
            return config_.layer_boundaries.back();
        }

        bool reserveStagingBufferBytes(size_t /*bytes*/) override
        {
            reserve_staging_call_count_.fetch_add(1, std::memory_order_relaxed);
            return true; // Always succeeds in mock
        }

        // =====================================================================
        // Test Utilities - Call Counting
        // =====================================================================

        /**
         * @brief Get number of transfer() calls (sync only)
         */
        int transferCallCount() const
        {
            return static_cast<int>(transfer_call_count_.load(std::memory_order_relaxed));
        }

        /**
         * @brief Get number of transferAsync() calls
         */
        int transferAsyncCallCount() const
        {
            return static_cast<int>(transfer_async_call_count_.load(std::memory_order_relaxed));
        }

        /**
         * @brief Get total number of transfer calls (sync + async)
         */
        int totalTransferCallCount() const
        {
            return transferCallCount() + transferAsyncCallCount();
        }

        /**
         * @brief Get number of synchronize() calls
         */
        int synchronizeCallCount() const
        {
            return static_cast<int>(synchronize_call_count_.load(std::memory_order_relaxed));
        }

        /**
         * @brief Get number of synchronizeStream() calls
         */
        int synchronizeStreamCallCount() const
        {
            return static_cast<int>(synchronize_stream_call_count_.load(std::memory_order_relaxed));
        }

        /**
         * @brief Get number of reserveStagingBufferBytes() calls
         */
        int reserveStagingCallCount() const
        {
            return static_cast<int>(reserve_staging_call_count_.load(std::memory_order_relaxed));
        }

        // =====================================================================
        // Test Utilities - Call Recording
        // =====================================================================

        /**
         * @brief Get all recorded transfer calls (sync and async)
         *
         * Returns a copy to avoid lock contention issues.
         */
        std::vector<TransferCall> transferCalls() const
        {
            std::lock_guard<std::mutex> lock(calls_mutex_);
            return transfer_calls_;
        }

        /**
         * @brief Get the last transfer call
         * @throws std::out_of_range if no transfers were made
         */
        TransferCall lastTransferCall() const
        {
            std::lock_guard<std::mutex> lock(calls_mutex_);
            if (transfer_calls_.empty())
            {
                throw std::out_of_range("MockLocalPPContext::lastTransferCall: no transfers recorded");
            }
            return transfer_calls_.back();
        }

        /**
         * @brief Check if a specific transfer was made
         */
        bool hasTransfer(int stage_from, int stage_to) const
        {
            std::lock_guard<std::mutex> lock(calls_mutex_);
            for (const auto &call : transfer_calls_)
            {
                if (call.matches(stage_from, stage_to))
                {
                    return true;
                }
            }
            return false;
        }

        /**
         * @brief Count transfers matching criteria
         */
        int countTransfers(int stage_from, int stage_to) const
        {
            std::lock_guard<std::mutex> lock(calls_mutex_);
            int count = 0;
            for (const auto &call : transfer_calls_)
            {
                if (call.matches(stage_from, stage_to))
                {
                    ++count;
                }
            }
            return count;
        }

        // =====================================================================
        // Test Utilities - Reset
        // =====================================================================

        /**
         * @brief Reset all call counters and recorded calls
         */
        void resetCallCounts()
        {
            transfer_call_count_.store(0, std::memory_order_relaxed);
            transfer_async_call_count_.store(0, std::memory_order_relaxed);
            synchronize_call_count_.store(0, std::memory_order_relaxed);
            synchronize_stream_call_count_.store(0, std::memory_order_relaxed);
            reserve_staging_call_count_.store(0, std::memory_order_relaxed);

            std::lock_guard<std::mutex> lock(calls_mutex_);
            transfer_calls_.clear();
        }

        // =====================================================================
        // Test Utilities - Configuration Modification
        // =====================================================================

        /**
         * @brief Configure return value for transfer()
         * @param result If false, transfer() will return false (simulate failure)
         */
        void setTransferResult(bool result)
        {
            config_.transfer_should_fail = !result;
        }

        /**
         * @brief Configure return value for transferAsync()
         * @param result If false, transferAsync() will return false (simulate failure)
         */
        void setTransferAsyncResult(bool result)
        {
            config_.transfer_async_should_fail = !result;
        }

        /**
         * @brief Set stage devices
         *
         * Note: This will NOT update layer_boundaries automatically.
         * Call setLayerBoundaries() if needed to maintain consistency.
         *
         * @param devices New device list
         */
        void setStageDevices(std::vector<GlobalDeviceAddress> devices)
        {
            config_.stage_devices = std::move(devices);
        }

        /**
         * @brief Set layer boundaries
         *
         * Note: Size must be stage_devices.size() + 1 for valid configuration.
         *
         * @param boundaries New layer boundary list
         */
        void setLayerBoundaries(std::vector<int> boundaries)
        {
            config_.layer_boundaries = std::move(boundaries);
        }

        /**
         * @brief Set default collective backend
         */
        void setDefaultBackend(CollectiveBackendType backend)
        {
            config_.default_backend = backend;
        }

        /**
         * @brief Get current configuration (const)
         */
        const Config &config() const
        {
            return config_;
        }

        /**
         * @brief Get mutable configuration reference
         *
         * Allows direct modification for complex test scenarios.
         */
        Config &mutableConfig()
        {
            return config_;
        }

    private:
        Config config_;

        // Atomic counters for thread-safe counting
        mutable std::atomic<size_t> transfer_call_count_{0};
        mutable std::atomic<size_t> transfer_async_call_count_{0};
        mutable std::atomic<size_t> synchronize_call_count_{0};
        mutable std::atomic<size_t> synchronize_stream_call_count_{0};
        mutable std::atomic<size_t> reserve_staging_call_count_{0};

        // Mutex-protected call recording
        mutable std::mutex calls_mutex_;
        std::vector<TransferCall> transfer_calls_;
    };

    /**
     * @brief Builder for MockLocalPPContext with fluent API
     *
     * Usage:
     * @code
     * auto mock = MockLocalPPContextBuilder()
     *     .withStages(3)
     *     .withDevices({cuda(0), cuda(1), rocm(0)})
     *     .withEqualLayerSplit(24)  // 24 layers split evenly
     *     .withBackend(CollectiveBackendType::HOST)
     *     .build();
     * @endcode
     */
    class MockLocalPPContextBuilder
    {
    public:
        MockLocalPPContextBuilder() = default;

        /**
         * @brief Set stage devices explicitly
         */
        MockLocalPPContextBuilder &withDevices(std::vector<GlobalDeviceAddress> devices)
        {
            config_.stage_devices = std::move(devices);
            return *this;
        }

        /**
         * @brief Set layer boundaries explicitly
         */
        MockLocalPPContextBuilder &withLayerBoundaries(std::vector<int> boundaries)
        {
            config_.layer_boundaries = std::move(boundaries);
            return *this;
        }

        /**
         * @brief Create N CPU devices
         */
        MockLocalPPContextBuilder &withCpuStages(int count)
        {
            config_.stage_devices.clear();
            for (int i = 0; i < count; ++i)
            {
                config_.stage_devices.push_back(GlobalDeviceAddress::cpu());
            }
            return *this;
        }

        /**
         * @brief Create N CUDA devices (cuda:0, cuda:1, ...)
         */
        MockLocalPPContextBuilder &withCudaStages(int count)
        {
            config_.stage_devices.clear();
            for (int i = 0; i < count; ++i)
            {
                config_.stage_devices.push_back(GlobalDeviceAddress::cuda(i));
            }
            return *this;
        }

        /**
         * @brief Split total_layers evenly across configured stages
         *
         * Requires that devices have been set first via withDevices(),
         * withCpuStages(), or withCudaStages().
         */
        MockLocalPPContextBuilder &withEqualLayerSplit(int total_layers)
        {
            int num_stages = static_cast<int>(config_.stage_devices.size());
            if (num_stages == 0)
            {
                num_stages = 1;
            }

            config_.layer_boundaries.clear();
            int layers_per_stage = total_layers / num_stages;
            int remainder = total_layers % num_stages;

            int current = 0;
            for (int s = 0; s < num_stages; ++s)
            {
                config_.layer_boundaries.push_back(current);
                current += layers_per_stage + (s < remainder ? 1 : 0);
            }
            config_.layer_boundaries.push_back(total_layers);

            return *this;
        }

        /**
         * @brief Set default collective backend
         */
        MockLocalPPContextBuilder &withBackend(CollectiveBackendType backend)
        {
            config_.default_backend = backend;
            return *this;
        }

        /**
         * @brief Configure transfer() to fail
         */
        MockLocalPPContextBuilder &withTransferFailure(bool should_fail = true)
        {
            config_.transfer_should_fail = should_fail;
            return *this;
        }

        /**
         * @brief Configure transferAsync() to fail
         */
        MockLocalPPContextBuilder &withTransferAsyncFailure(bool should_fail = true)
        {
            config_.transfer_async_should_fail = should_fail;
            return *this;
        }

        /**
         * @brief Build the mock context
         */
        std::unique_ptr<MockLocalPPContext> build()
        {
            return std::make_unique<MockLocalPPContext>(config_);
        }

        /**
         * @brief Build and return shared_ptr (useful for shared ownership)
         */
        std::shared_ptr<MockLocalPPContext> buildShared()
        {
            return std::make_shared<MockLocalPPContext>(config_);
        }

    private:
        MockLocalPPContext::Config config_;
    };

} // namespace llaminar2::test
