/**
 * @file MockLocalTPContext.h
 * @brief Mock implementation of ILocalTPContext for unit testing
 *
 * This mock enables:
 * - Testing code that depends on ILocalTPContext without real devices
 * - Configuring mock device lists and backends
 * - Call tracking for allreduce operations
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#pragma once

#include "collective/ILocalTPContext.h"
#include "backends/GlobalDeviceAddress.h"
#include "config/OrchestrationConfig.h"  // CollectiveBackendType
#include <atomic>
#include <cmath>
#include <mutex>
#include <vector>

namespace llaminar2::test
{

    /**
     * @brief Record of a single allreduce() call
     */
    struct AllreduceCall
    {
        TensorBase *tensor = nullptr;
        std::string stage_name;
        size_t count = 0;

        AllreduceCall() = default;
        AllreduceCall(TensorBase *t, const std::string& name = "", size_t c = 0)
            : tensor(t), stage_name(name), count(c) {}
    };

    /**
     * @brief Mock implementation of ILocalTPContext for unit testing
     *
     * Provides configurable device lists, backend types, and call tracking
     * without requiring actual GPU hardware.
     *
     * Usage:
     * ```cpp
     * auto mock = std::make_shared<MockLocalTPContext>();
     * mock->setDevices({GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)});
     * mock->setBackend(CollectiveBackendType::RCCL);
     *
     * // Use mock in code under test...
     *
     * EXPECT_EQ(mock->allreduceCallCount(), 1);
     * ```
     */
    class MockLocalTPContext : public ILocalTPContext
    {
    public:
        MockLocalTPContext() = default;

        // =====================================================================
        // Configuration
        // =====================================================================

        void setDevices(std::vector<GlobalDeviceAddress> devices)
        {
            std::lock_guard<std::mutex> lock(mutex_);
            devices_ = std::move(devices);
        }

        void setBackend(CollectiveBackendType backend)
        {
            backend_ = backend;
        }

        void setWeights(std::vector<float> weights)
        {
            std::lock_guard<std::mutex> lock(mutex_);
            weights_ = std::move(weights);
        }

        void setAllreduceShouldFail(bool fail)
        {
            allreduce_should_fail_ = fail;
        }

        void setBroadcastShouldFail(bool fail)
        {
            broadcast_should_fail_ = fail;
        }

        // =====================================================================
        // ILocalTPContext Interface
        // =====================================================================

        int degree() const override
        {
            std::lock_guard<std::mutex> lock(mutex_);
            return static_cast<int>(devices_.size());
        }

        int myIndex() const override { return 0; }

        const std::vector<GlobalDeviceAddress>& devices() const override
        {
            std::lock_guard<std::mutex> lock(mutex_);
            return devices_;
        }

        CollectiveBackendType backend() const override
        {
            return backend_;
        }

        const std::vector<float>& weights() const override
        {
            std::lock_guard<std::mutex> lock(mutex_);
            return weights_;
        }

        // Allreduce overloads
        bool allreduce(TensorBase* tensor) override
        {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                allreduce_calls_.emplace_back(tensor);
            }
            ++allreduce_call_count_;
            return !allreduce_should_fail_;
        }

        bool allreduce(TensorBase* tensor, const std::string& stage_name, size_t count = 0) override
        {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                allreduce_calls_.emplace_back(tensor, stage_name, count);
            }
            ++allreduce_call_count_;
            return !allreduce_should_fail_;
        }

        bool allreduce(const TensorBase* input, TensorBase* output) override
        {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                allreduce_calls_.emplace_back(const_cast<TensorBase*>(input));
            }
            ++allreduce_call_count_;
            return !allreduce_should_fail_;
        }

        bool allgather(const TensorBase* local_shard, TensorBase* global_tensor) override
        {
            ++allgather_call_count_;
            return true;
        }

        bool gatherFromDevices(
            const std::vector<const TensorBase*>& shards,
            TensorBase* output) override
        {
            ++gather_call_count_;
            return true;
        }

        bool reduceScatter(const TensorBase* input, TensorBase* output_shard) override
        {
            ++reduce_scatter_call_count_;
            return true;
        }

        bool broadcast(TensorBase* tensor, int source_device_index = 0) override
        {
            (void)tensor;
            (void)source_device_index;
            ++broadcast_call_count_;
            return !broadcast_should_fail_;
        }

        void synchronize() override
        {
            ++synchronize_call_count_;
        }

        int indexForDevice(const GlobalDeviceAddress& device) const override
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (size_t i = 0; i < devices_.size(); ++i)
            {
                if (devices_[i] == device)
                    return static_cast<int>(i);
            }
            return -1;
        }

        const GlobalDeviceAddress& deviceAt(int index) const override
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (index < 0 || index >= static_cast<int>(devices_.size()))
            {
                static GlobalDeviceAddress cpu_addr = GlobalDeviceAddress::cpu();
                return cpu_addr;
            }
            return devices_[index];
        }

        float weightForDevice(const GlobalDeviceAddress& device) const override
        {
            std::lock_guard<std::mutex> lock(mutex_);
            int idx = -1;
            for (size_t i = 0; i < devices_.size(); ++i)
            {
                if (devices_[i] == device) {
                    idx = static_cast<int>(i);
                    break;
                }
            }
            if (idx < 0) return 0.0f;
            if (weights_.empty()) {
                // Equal distribution
                return 1.0f / static_cast<float>(devices_.size());
            }
            return (idx < static_cast<int>(weights_.size())) ? weights_[idx] : 0.0f;
        }

        int headsForDevice(const GlobalDeviceAddress& device, int total_heads) const override
        {
            // Simple proportional distribution
            float weight = weightForDevice(device);
            return static_cast<int>(std::round(weight * total_heads));
        }

        std::pair<int, int> rowRangeForDevice(
            const GlobalDeviceAddress& device, int total_rows) const override
        {
            std::lock_guard<std::mutex> lock(mutex_);
            int idx = -1;
            for (size_t i = 0; i < devices_.size(); ++i)
            {
                if (devices_[i] == device) {
                    idx = static_cast<int>(i);
                    break;
                }
            }
            if (idx < 0 || devices_.empty()) return {0, 0};
            
            int rows_per_device = total_rows / static_cast<int>(devices_.size());
            int remainder = total_rows % static_cast<int>(devices_.size());
            int start = idx * rows_per_device + std::min(idx, remainder);
            int end = start + rows_per_device + (idx < remainder ? 1 : 0);
            return {start, end};
        }

        std::pair<int, int> colRangeForDevice(
            const GlobalDeviceAddress& device, int total_cols) const override
        {
            // Same logic as rowRangeForDevice for symmetry
            return rowRangeForDevice(device, total_cols);
        }

        void registerBARBackedOutput(
            const std::string& stage_name,
            const GlobalDeviceAddress& device,
            TensorBase* tensor) override
        {
            // Mock: No-op
            (void)stage_name;
            (void)device;
            (void)tensor;
        }

        bool hasBARBackedOutputs(const std::string& stage_name) const override
        {
            (void)stage_name;
            return false;  // Mock: Always false
        }

        void clearBARBackedOutputs() override
        {
            // Mock: No-op
        }

        std::shared_ptr<DirectP2PEngine> getDirectP2PEngine() const override
        {
            return nullptr;  // Mock: No P2P engine
        }

        bool reserveTempBufferBytes(size_t bytes) override
        {
            (void)bytes;
            return true;  // Mock: Always succeed
        }

        // =====================================================================
        // Call Tracking
        // =====================================================================

        int allreduceCallCount() const { return allreduce_call_count_.load(); }
        int allgatherCallCount() const { return allgather_call_count_.load(); }
        int gatherCallCount() const { return gather_call_count_.load(); }
        int reduceScatterCallCount() const { return reduce_scatter_call_count_.load(); }
        int broadcastCallCount() const { return broadcast_call_count_.load(); }
        int synchronizeCallCount() const { return synchronize_call_count_.load(); }

        std::vector<AllreduceCall> getAllreduceCalls() const
        {
            std::lock_guard<std::mutex> lock(mutex_);
            return allreduce_calls_;
        }

        void resetCallTracking()
        {
            std::lock_guard<std::mutex> lock(mutex_);
            allreduce_calls_.clear();
            allreduce_call_count_ = 0;
            allgather_call_count_ = 0;
            gather_call_count_ = 0;
            reduce_scatter_call_count_ = 0;
            broadcast_call_count_ = 0;
            synchronize_call_count_ = 0;
        }

    private:
        mutable std::mutex mutex_;
        std::vector<GlobalDeviceAddress> devices_;
        std::vector<float> weights_;
        CollectiveBackendType backend_ = CollectiveBackendType::AUTO;

        std::vector<AllreduceCall> allreduce_calls_;
        std::atomic<int> allreduce_call_count_{0};
        std::atomic<int> allgather_call_count_{0};
        std::atomic<int> gather_call_count_{0};
        std::atomic<int> reduce_scatter_call_count_{0};
        std::atomic<int> broadcast_call_count_{0};
        std::atomic<int> synchronize_call_count_{0};
        bool allreduce_should_fail_ = false;
        bool broadcast_should_fail_ = false;
    };

} // namespace llaminar2::test
