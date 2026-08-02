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
#include "config/OrchestrationConfig.h" // CollectiveBackendType
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
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
        void *stream = nullptr;
        std::string precision;

        AllreduceCall() = default;
        AllreduceCall(TensorBase *t, const std::string &name = "", size_t c = 0)
            : tensor(t), stage_name(name), count(c) {}
        AllreduceCall(TensorBase *t, const std::string &name, size_t c, void *s, std::string p)
            : tensor(t), stage_name(name), count(c), stream(s), precision(std::move(p)) {}
    };

    /**
     * @brief Record of one graph-stream sideband collective request.
     */
    struct SidebandCall
    {
        int device_index = -1;
        void *stream = nullptr;
        std::string stage_name;
        size_t sideband_count = 0;
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

        void setSidebandShouldFail(bool fail)
        {
            sideband_should_fail_ = fail;
        }

        void setRawAllgatherGraphCaptureSupported(bool supported)
        {
            raw_allgather_graph_capture_supported_ = supported;
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

        const std::vector<GlobalDeviceAddress> &devices() const override
        {
            std::lock_guard<std::mutex> lock(mutex_);
            return devices_;
        }

        CollectiveBackendType backend() const override
        {
            return backend_;
        }

        const std::vector<float> &weights() const override
        {
            std::lock_guard<std::mutex> lock(mutex_);
            return weights_;
        }

        // Allreduce overloads
        bool allreduce(TensorBase *tensor) override
        {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                allreduce_calls_.emplace_back(tensor);
            }
            ++allreduce_call_count_;
            return !allreduce_should_fail_;
        }

        bool allreduce(TensorBase *tensor, const std::string &stage_name, size_t count = 0) override
        {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                allreduce_calls_.emplace_back(tensor, stage_name, count);
            }
            ++allreduce_call_count_;
            return !allreduce_should_fail_;
        }

        bool allreduceOnStream(TensorBase *tensor, const std::string &stage_name,
                               size_t count, void *stream,
                               const std::string &precision = "") override
        {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                allreduce_calls_.emplace_back(tensor, stage_name, count, stream, precision);
            }
            ++allreduce_call_count_;
            return !allreduce_should_fail_;
        }

        bool allreduce(const TensorBase *input, TensorBase *output) override
        {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                allreduce_calls_.emplace_back(const_cast<TensorBase *>(input));
            }
            ++allreduce_call_count_;
            return !allreduce_should_fail_;
        }

        bool allgather(const TensorBase *local_shard, TensorBase *global_tensor) override
        {
            ++allgather_call_count_;
            return true;
        }

        bool supportsRawAllgatherOnStreamGraphCapture() const override
        {
            return raw_allgather_graph_capture_supported_;
        }

        bool gatherFromDevices(
            const std::vector<const TensorBase *> &shards,
            TensorBase *output) override
        {
            ++gather_call_count_;
            return true;
        }

        bool reduceScatter(const TensorBase *input, TensorBase *output_shard) override
        {
            ++reduce_scatter_call_count_;
            return true;
        }

        bool broadcast(TensorBase *tensor, int source_device_index = 0) override
        {
            (void)tensor;
            (void)source_device_index;
            ++broadcast_call_count_;
            return !broadcast_should_fail_;
        }

        bool collectiveSidebandOnStream(
            const std::vector<LocalTPCollectiveSidebandBuffer> &sidebands,
            int device_index,
            void *producer_stream,
            const std::string &anchor_stage_name) override
        {
            if (!producer_stream)
            {
                throw std::invalid_argument(
                    "MockLocalTPContext::collectiveSidebandOnStream requires a non-null stream");
            }

            {
                std::lock_guard<std::mutex> lock(mutex_);
                sideband_calls_.push_back(
                    SidebandCall{device_index,
                                 producer_stream,
                                 anchor_stage_name,
                                 sidebands.size()});
                if (device_index < 0 ||
                    device_index >= static_cast<int>(devices_.size()))
                {
                    ++sideband_call_count_;
                    return false;
                }
            }

            for (const auto &sideband : sidebands)
            {
                if (!sideband.recv_buffer || sideband.element_count == 0)
                {
                    ++sideband_call_count_;
                    return false;
                }
            }

            ++sideband_call_count_;
            if (sideband_should_fail_)
                return false;

            const int participant_count = degree();
            if (participant_count <= 1 || sidebands.empty())
                return true;

            /*
             * RankOrchestrator dispatches one collective call per participant.
             * Preserve that rendezvous here instead of merely returning true:
             * device-slot publication tests need the peer mailbox bytes to
             * change before its readiness event is recorded. The mock performs
             * the transfer only after every participant has arrived, matching
             * NCCL/RCCL's collective ordering contract closely enough for unit
             * tests without pretending host staging occurred in production.
             */
            std::unique_lock<std::mutex> lock(sideband_collective_mutex_);
            const int generation = sideband_generation_;
            if (sideband_generation_requests_.empty())
            {
                sideband_generation_requests_.resize(
                    static_cast<size_t>(participant_count));
            }
            auto &participant_requests =
                sideband_generation_requests_[static_cast<size_t>(device_index)];
            if (!participant_requests.empty())
                return false;
            participant_requests = sidebands;
            ++sideband_arrivals_;

            if (sideband_arrivals_ == participant_count)
            {
                sideband_generation_result_ =
                    completeSidebandGenerationLocked();
                sideband_arrivals_ = 0;
                sideband_generation_requests_.clear();
                ++sideband_generation_;
                lock.unlock();
                sideband_collective_cv_.notify_all();
                return sideband_generation_result_;
            }

            const bool completed = sideband_collective_cv_.wait_for(
                lock,
                std::chrono::seconds(2),
                [this, generation]()
                {
                    return sideband_generation_ != generation;
                });
            return completed && sideband_generation_result_;
        }

        bool collectiveSidebandSpanOnStream(
            std::span<const LocalTPCollectiveSidebandBuffer> sidebands,
            int device_index,
            void *producer_stream,
            const std::string &anchor_stage_name) override
        {
            return collectiveSidebandOnStream(
                std::vector<LocalTPCollectiveSidebandBuffer>(
                    sidebands.begin(),
                    sidebands.end()),
                device_index,
                producer_stream,
                anchor_stage_name);
        }

        bool supportsCollectiveSidebandOnStreamGraphCapture() const override
        {
            return raw_allgather_graph_capture_supported_;
        }

        /**
         * @brief Execute one participant-major mock sideband collective.
         *
         * Production NCCL/RCCL code submits every participant and its exact
         * producer stream through one grouped host call.  The mock mirrors that
         * rank-level API directly: it validates the complete matrix, records
         * one logical collective call, and copies broadcast payloads only
         * through the shared descriptor interpreter.  No worker rendezvous is
         * needed because every participant is already present in this call.
         *
         * @param participant_sidebands Sidebands indexed by LocalTP participant.
         * @param producer_streams Exact producer stream for each participant.
         * @param publication_name Human-readable collective identity.
         * @return true when the complete matrix is valid and was applied.
         */
        bool collectiveSidebandsMultiOnStreams(
            const std::vector<std::vector<LocalTPCollectiveSidebandBuffer>>
                &participant_sidebands,
            const std::vector<void *> &producer_streams,
            const std::string &publication_name) override
        {
            int participant_count = 0;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                participant_count = static_cast<int>(devices_.size());
                if (participant_sidebands.size() !=
                        static_cast<size_t>(participant_count) ||
                    producer_streams.size() !=
                        static_cast<size_t>(participant_count))
                {
                    ++sideband_call_count_;
                    return false;
                }

                for (int participant = 0;
                     participant < participant_count;
                     ++participant)
                {
                    void *stream =
                        producer_streams[static_cast<size_t>(participant)];
                    if (!stream)
                    {
                        throw std::invalid_argument(
                            "MockLocalTPContext::collectiveSidebandsMultiOnStreams "
                            "requires a non-null stream for every participant");
                    }
                    sideband_calls_.push_back(
                        SidebandCall{
                            participant,
                            stream,
                            publication_name,
                            participant_sidebands[
                                static_cast<size_t>(participant)]
                                .size()});
                }
            }

            ++sideband_call_count_;
            if (sideband_should_fail_)
                return false;

            std::lock_guard<std::mutex> collective_lock(
                sideband_collective_mutex_);
            if (sideband_arrivals_ != 0 ||
                !sideband_generation_requests_.empty())
            {
                return false;
            }

            sideband_generation_requests_ = participant_sidebands;
            const bool result = completeSidebandGenerationLocked();
            sideband_generation_requests_.clear();
            sideband_generation_result_ = result;
            ++sideband_generation_;
            return result;
        }

        void synchronize() override
        {
            ++synchronize_call_count_;
        }

        int indexForDevice(const GlobalDeviceAddress &device) const override
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (size_t i = 0; i < devices_.size(); ++i)
            {
                if (devices_[i] == device)
                    return static_cast<int>(i);
            }
            return -1;
        }

        const GlobalDeviceAddress &deviceAt(int index) const override
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (index < 0 || index >= static_cast<int>(devices_.size()))
            {
                static GlobalDeviceAddress cpu_addr = GlobalDeviceAddress::cpu();
                return cpu_addr;
            }
            return devices_[index];
        }

        float weightForDevice(const GlobalDeviceAddress &device) const override
        {
            std::lock_guard<std::mutex> lock(mutex_);
            int idx = -1;
            for (size_t i = 0; i < devices_.size(); ++i)
            {
                if (devices_[i] == device)
                {
                    idx = static_cast<int>(i);
                    break;
                }
            }
            if (idx < 0)
                return 0.0f;
            if (weights_.empty())
            {
                // Equal distribution
                return 1.0f / static_cast<float>(devices_.size());
            }
            return (idx < static_cast<int>(weights_.size())) ? weights_[idx] : 0.0f;
        }

        int headsForDevice(const GlobalDeviceAddress &device, int total_heads) const override
        {
            // Simple proportional distribution
            float weight = weightForDevice(device);
            return static_cast<int>(std::round(weight * total_heads));
        }

        std::pair<int, int> rowRangeForDevice(
            const GlobalDeviceAddress &device, int total_rows) const override
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (total_rows <= 0 || devices_.empty())
                return {0, 0};

            int idx = -1;
            for (size_t i = 0; i < devices_.size(); ++i)
            {
                if (devices_[i] == device)
                {
                    idx = static_cast<int>(i);
                    break;
                }
            }
            if (idx < 0)
                return {0, 0};

            auto cumulative = computeCumulativeCounts(total_rows);
            return {cumulative[idx], cumulative[idx + 1]};
        }

        std::pair<int, int> colRangeForDevice(
            const GlobalDeviceAddress &device, int total_cols) const override
        {
            // Same logic as rowRangeForDevice for symmetry
            return rowRangeForDevice(device, total_cols);
        }

        void registerBARBackedOutput(
            const std::string &stage_name,
            const GlobalDeviceAddress &device,
            TensorBase *tensor) override
        {
            // Mock: No-op
            (void)stage_name;
            (void)device;
            (void)tensor;
        }

        bool hasBARBackedOutputs(const std::string &stage_name) const override
        {
            (void)stage_name;
            return false; // Mock: Always false
        }

        void clearBARBackedOutputs() override
        {
            // Mock: No-op
        }

        bool reserveCollectiveResources(
            size_t bytes,
            size_t /*fp16_scratch_elements*/) override
        {
            (void)bytes;
            return true; // Mock: Always succeed
        }

        // =====================================================================
        // Call Tracking
        // =====================================================================

        int allreduceCallCount() const { return allreduce_call_count_.load(); }
        int allgatherCallCount() const { return allgather_call_count_.load(); }
        int gatherCallCount() const { return gather_call_count_.load(); }
        int reduceScatterCallCount() const { return reduce_scatter_call_count_.load(); }
        int broadcastCallCount() const { return broadcast_call_count_.load(); }
        int sidebandCallCount() const { return sideband_call_count_.load(); }
        int synchronizeCallCount() const { return synchronize_call_count_.load(); }

        void requestAbort() override {}
        bool isAbortRequested() const override { return false; }

        std::vector<AllreduceCall> getAllreduceCalls() const
        {
            std::lock_guard<std::mutex> lock(mutex_);
            return allreduce_calls_;
        }

        std::vector<SidebandCall> getSidebandCalls() const
        {
            std::lock_guard<std::mutex> lock(mutex_);
            return sideband_calls_;
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
            sideband_call_count_ = 0;
            synchronize_call_count_ = 0;
            sideband_calls_.clear();
        }

    private:
        static size_t collectiveDataTypeBytes(CollectiveDataType dtype)
        {
            switch (dtype)
            {
            case CollectiveDataType::FLOAT32:
            case CollectiveDataType::INT32:
                return sizeof(std::uint32_t);
            case CollectiveDataType::FLOAT16:
            case CollectiveDataType::BFLOAT16:
                return sizeof(std::uint16_t);
            case CollectiveDataType::INT8:
                return sizeof(std::uint8_t);
            }
            return 0;
        }

        /** Complete one mock sideband generation while its mutex is held. */
        bool completeSidebandGenerationLocked()
        {
            const int participant_count = static_cast<int>(devices_.size());
            if (participant_count <= 0 ||
                static_cast<int>(sideband_generation_requests_.size()) !=
                    participant_count)
            {
                return false;
            }

            const size_t sideband_count =
                sideband_generation_requests_.front().size();
            for (const auto &requests : sideband_generation_requests_)
            {
                if (requests.size() != sideband_count)
                    return false;
            }

            for (size_t sideband_index = 0;
                 sideband_index < sideband_count;
                 ++sideband_index)
            {
                const auto &reference =
                    sideband_generation_requests_.front()[sideband_index];
                if (reference.root_device_index < 0 ||
                    reference.root_device_index >= participant_count ||
                    reference.element_count == 0)
                {
                    return false;
                }

                for (int participant = 0;
                     participant < participant_count;
                     ++participant)
                {
                    const auto &request =
                        sideband_generation_requests_[
                            static_cast<size_t>(participant)][sideband_index];
                    if (request.kind != reference.kind ||
                        request.element_count != reference.element_count ||
                        request.dtype != reference.dtype ||
                        request.root_device_index !=
                            reference.root_device_index)
                    {
                        return false;
                    }
                }

                if (reference.kind !=
                    LocalTPCollectiveSidebandKind::Broadcast)
                {
                    continue;
                }

                const auto &root =
                    sideband_generation_requests_[static_cast<size_t>(
                        reference.root_device_index)][sideband_index];
                const void *source =
                    root.send_buffer ? root.send_buffer : root.recv_buffer;
                const size_t bytes =
                    reference.element_count *
                    collectiveDataTypeBytes(reference.dtype);
                if (!source || bytes == 0)
                    return false;

                for (int participant = 0;
                     participant < participant_count;
                     ++participant)
                {
                    auto &request =
                        sideband_generation_requests_[
                            static_cast<size_t>(participant)][sideband_index];
                    if (!request.recv_buffer)
                        return false;
                    std::memcpy(request.recv_buffer, source, bytes);
                }
            }
            return true;
        }

        mutable std::mutex mutex_;
        std::mutex sideband_collective_mutex_;
        std::condition_variable sideband_collective_cv_;
        int sideband_generation_ = 0;
        int sideband_arrivals_ = 0;
        bool sideband_generation_result_ = false;
        std::vector<std::vector<LocalTPCollectiveSidebandBuffer>>
            sideband_generation_requests_;
        std::vector<GlobalDeviceAddress> devices_;
        std::vector<float> weights_;
        CollectiveBackendType backend_ = CollectiveBackendType::AUTO;

        /**
         * @brief Proportional distribution matching LocalTPContext::computeCumulativeCounts
         *
         * When weights_ is empty, distributes equally. Otherwise distributes
         * proportionally with proper rounding to ensure exact total.
         */
        std::vector<int> computeCumulativeCounts(int total) const
        {
            // Build normalized weights
            std::vector<float> norm_weights;
            if (weights_.empty())
            {
                float eq = 1.0f / static_cast<float>(devices_.size());
                norm_weights.assign(devices_.size(), eq);
            }
            else
            {
                float sum = 0.0f;
                for (float w : weights_)
                    sum += w;
                norm_weights.resize(weights_.size());
                for (size_t i = 0; i < weights_.size(); ++i)
                    norm_weights[i] = weights_[i] / sum;
            }

            std::vector<int> cumulative(norm_weights.size() + 1);
            cumulative[0] = 0;

            int remaining = total;
            float remaining_weight = 1.0f;

            for (size_t i = 0; i < norm_weights.size(); ++i)
            {
                if (i == norm_weights.size() - 1)
                {
                    cumulative[i + 1] = total;
                }
                else
                {
                    float proportion = norm_weights[i] / remaining_weight;
                    int count = static_cast<int>(std::round(proportion * remaining));
                    if (count == 0 && remaining > 0)
                        count = 1;
                    count = std::min(count, remaining);

                    cumulative[i + 1] = cumulative[i] + count;
                    remaining -= count;
                    remaining_weight -= norm_weights[i];
                }
            }

            return cumulative;
        }

        std::vector<AllreduceCall> allreduce_calls_;
        std::vector<SidebandCall> sideband_calls_;
        std::atomic<int> allreduce_call_count_{0};
        std::atomic<int> allgather_call_count_{0};
        std::atomic<int> gather_call_count_{0};
        std::atomic<int> reduce_scatter_call_count_{0};
        std::atomic<int> broadcast_call_count_{0};
        std::atomic<int> sideband_call_count_{0};
        std::atomic<int> synchronize_call_count_{0};
        bool allreduce_should_fail_ = false;
        bool broadcast_should_fail_ = false;
        bool sideband_should_fail_ = false;
        bool raw_allgather_graph_capture_supported_ = false;
    };

} // namespace llaminar2::test
