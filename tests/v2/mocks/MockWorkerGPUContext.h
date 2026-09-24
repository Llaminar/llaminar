/**
 * @file MockWorkerGPUContext.h
 * @brief Hardware-free worker GPU context for unit tests.
 *
 * GPU-shaped orchestration tests need stable non-null stream and event handles,
 * plus a graph-capture lifecycle, but unit tests must not initialize CUDA or
 * ROCm. These test doubles implement the worker interfaces entirely in host
 * memory. Handles are opaque addresses only; no backend API ever observes or
 * dereferences them.
 */

#pragma once

#include "backends/GPUDeviceContextPool.h"
#include "backends/IGPUGraphCapture.h"
#include "backends/IWorkerGPUContext.h"

#include <atomic>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace llaminar2::testing
{
    /**
     * @brief In-memory implementation of the GPU graph lifecycle.
     *
     * The object models capture, instantiation, update, and replay state closely
     * enough for executor lifecycle tests. It deliberately does not emulate
     * kernels: the surrounding mock stages remain responsible for recording
     * their own execution observations.
     */
    class MockGPUGraphCapture final : public IGPUGraphCapture
    {
    public:
        explicit MockGPUGraphCapture(void *stream) : stream_(stream) {}

        /** @brief Preserve graph-only fragment construction in controller unit tests. */
        std::unique_ptr<IGPUGraphCapture> createOrderedTimelineFragment() override
        {
            return std::make_unique<MockGPUGraphCapture>(stream_);
        }

        bool beginCapture() override
        {
            if (capturing_)
                return false;
            capturing_ = true;
            captured_ = false;
            return true;
        }

        bool endCapture() override
        {
            if (!capturing_)
                return false;
            capturing_ = false;
            captured_ = true;
            node_count_ = 1;
            return true;
        }

        bool instantiate() override
        {
            if (!captured_)
                return false;
            executable_ = true;
            return true;
        }

        bool launch() override
        {
            ++launch_count_;
            return executable_;
        }

        [[nodiscard]] void *executionStream() const noexcept override { return stream_; }

        GraphUpdateResult tryUpdate() override
        {
            return captured_ && executable_
                       ? GraphUpdateResult::Success
                       : GraphUpdateResult::NeedsReinstantiate;
        }

        [[nodiscard]] bool supportsExecutableUpdate() const noexcept override { return true; }
        bool hasExecutable() const override { return executable_; }
        [[nodiscard]] std::size_t residentMemoryBytes() const noexcept override
        {
            return 0u;
        }
        size_t nodeCount() const override { return node_count_; }

        void reset() override
        {
            capturing_ = false;
            captured_ = false;
            executable_ = false;
            node_count_ = 0;
        }

        const char *backendName() const override { return "MockGPU"; }
        int launchCount() const { return launch_count_; }

    private:
        void *stream_ = nullptr;
        bool capturing_ = false;
        bool captured_ = false;
        bool executable_ = false;
        size_t node_count_ = 0;
        int launch_count_ = 0;
    };

    /**
     * @brief Synchronous, hardware-free implementation of IWorkerGPUContext.
     *
     * Work submissions execute immediately on the calling thread. Every stream
     * and event is represented by a stable host allocation whose address serves
     * only as an opaque non-null identity. This preserves the production rule
     * that GPU operations require explicit streams while guaranteeing that a
     * unit test cannot accidentally create a physical device context.
     */
    class MockWorkerGPUContext final : public IWorkerGPUContext
    {
    public:
        explicit MockWorkerGPUContext(int device_ordinal = 0)
            : device_ordinal_(device_ordinal)
        {
        }

        int deviceOrdinal() const override { return device_ordinal_; }
        std::string deviceName() const override { return "HardwareFreeMockGPU"; }
        bool isInitialized() const override { return true; }

        void submitAndWait(std::function<void()> work) override { work(); }

        std::future<void> submitAsync(std::function<void()> work) override
        {
            work();
            std::promise<void> completed;
            completed.set_value();
            return completed.get_future();
        }

        void *defaultStream() override { return &default_stream_token_; }

        void *createStream() override
        {
            streams_.push_back(std::make_unique<int>(++next_stream_token_));
            return streams_.back().get();
        }

        void destroyStream(void *) override {}

        void *getOrCreateAuxiliaryStream(
            const std::string &name,
            bool *created = nullptr) override
        {
            auto [it, inserted] = auxiliary_streams_.try_emplace(name);
            if (inserted)
                it->second = std::make_unique<int>(++next_stream_token_);
            if (created)
                *created = inserted;
            return it->second.get();
        }

        void resetAuxiliaryStreams() override { auxiliary_streams_.clear(); }

        void *createEvent() override
        {
            events_.push_back(std::make_unique<int>(++next_event_token_));
            return events_.back().get();
        }

        void destroyEvent(void *) override {}
        void recordEvent(void *, void *) override {}
        void waitEvent(void *, void *) override {}

        bool queryEventChecked(void *event, bool &ready) override
        {
            ready = event != nullptr;
            return ready;
        }

        void synchronizeEvent(void *) override {}
        float eventElapsedTime(void *, void *) override { return 0.0f; }
        void *blasHandle() override { return nullptr; }
        void *blasLtHandle() override { return nullptr; }

        void setCollectiveComm(void *comm) override { collective_comm_ = comm; }
        void *collectiveComm() const override { return collective_comm_; }

        void synchronize() override {}
        bool synchronizeChecked() override { return true; }
        void synchronizeStream(void *) override {}
        bool synchronizeStreamChecked(void *stream) override { return stream != nullptr; }
        GPUStreamExecutionState queryStreamExecutionState(
            void *stream,
            std::string_view boundary) override
        {
            if (!stream || boundary.empty())
                throw std::invalid_argument("mock stream query requires an exact stream and boundary");
            return GPUStreamExecutionState::Complete;
        }
        bool insertStreamDependency(void *, void *) override { return true; }

        std::unique_ptr<IGPUGraphCapture> createGraphCapture() override
        {
            ++graph_capture_create_count_;
            return std::make_unique<MockGPUGraphCapture>(defaultStream());
        }

        std::unique_ptr<IGPUGraphCapture> createGraphCapture(void *stream) override
        {
            if (!stream)
                return nullptr;
            ++graph_capture_create_count_;
            return std::make_unique<MockGPUGraphCapture>(stream);
        }

        void setGraphCaptureActive(bool active) override
        {
            graph_capture_active_.store(active, std::memory_order_release);
        }

        bool isDeviceGraphCaptureActive() const override
        {
            return graph_capture_active_.load(std::memory_order_acquire);
        }

        int graphCaptureCreateCount() const { return graph_capture_create_count_; }

    private:
        int device_ordinal_ = 0;
        int default_stream_token_ = 1;
        int next_stream_token_ = 1;
        int next_event_token_ = 0;
        int graph_capture_create_count_ = 0;
        void *collective_comm_ = nullptr;
        std::vector<std::unique_ptr<int>> streams_;
        std::vector<std::unique_ptr<int>> events_;
        std::unordered_map<std::string, std::unique_ptr<int>> auxiliary_streams_;
        std::atomic<bool> graph_capture_active_{false};
    };

    /**
     * @brief Return a process-lifetime mock worker for cached-executor tests.
     *
     * Forward graph caches may outlive an individual mock host object. A shared
     * process-lifetime worker mirrors the real process-owned GPU context pool and
     * keeps cached stream/event cleanup valid after the host fixture is gone.
     */
    inline MockWorkerGPUContext &sharedMockWorkerGPUContext()
    {
        static MockWorkerGPUContext context{0};
        return context;
    }

    /**
     * @brief Install hardware-free CUDA and ROCm worker-context factories.
     *
     * Tensor transfer tests often inject a MockBackend so allocation, copies,
     * and events remain in host memory. The production transfer API also asks
     * GPUDeviceContextPool for a persistent default stream when a caller uses
     * the convenience overload without an explicit stream. Injecting only the
     * backend therefore leaves half of the GPU execution interface real and
     * can accidentally initialize CUDA or ROCm from a unit test.
     *
     * This helper installs the matching worker-context half of that interface.
     * Each returned stream and event is a stable opaque host address; no CUDA
     * or HIP API observes it. Registration is process-idempotent because unit
     * test executables may contain several fixtures that share the same pool.
     *
     * @note Call this only from hardware-free unit-test setup. Integration
     *       tests intentionally retain the real backend factories.
     */
    inline void installHardwareFreeGPUContextFactories()
    {
        static std::once_flag installed;
        std::call_once(
            installed,
            []
            {
                constexpr int kMockDeviceCount = 64;
                auto &pool = GPUDeviceContextPool::instance();
                pool.registerNvidiaFactory(
                    [](int ordinal)
                    {
                        return std::make_unique<MockWorkerGPUContext>(ordinal);
                    },
                    kMockDeviceCount);
                pool.registerAMDFactory(
                    [](int ordinal)
                    {
                        return std::make_unique<MockWorkerGPUContext>(ordinal);
                    },
                    kMockDeviceCount);
            });
    }
} // namespace llaminar2::testing
